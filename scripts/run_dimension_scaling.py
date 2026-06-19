"""Run dimension scaling experiments for CNCV.

This script trains and evaluates CNCV across multiple dimensions to study
how variance reduction scales with problem dimensionality.

Usage:
    python scripts/run_dimension_scaling.py --phase train --dimensions 2 4 8 16
    python scripts/run_dimension_scaling.py --phase test --dimensions 2 4 8 16
    python scripts/run_dimension_scaling.py --phase plot --output_dir figs/
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'text.usetex': False,
})

# Color scheme for publication
COLORS = {
    'vanilla': '#7f7f7f',
    'cv': '#d62728',
    'training': '#1f77b4',
    'validation': '#ff7f0e',
    'reference': '#8c564b',
    'dim_cmap': plt.cm.viridis,
}

# Figure sizes for two-column format
HALF_COL = 3.25
FULL_COL = 6.75


@dataclass
class DimensionResult:
    """Results for a single dimension experiment."""
    dimension: int
    qofi: str
    vrf_mean: float  # Average VRF across components
    vrf_std: float   # Std of VRF across components
    vrf_per_component: List[float]
    correlation_mean: float
    correlation_per_component: List[float]
    variance_vanilla: List[float]
    variance_cv: List[float]
    mse_at_n100: List[float]  # MSE at n=100 samples
    training_loss_final: float
    validation_loss_final: float


def get_config_for_dimension(dim: int, qofi: str = "mean") -> dict:
    """Generate config for a given dimension."""
    # Scale network capacity with dimension
    if dim <= 4:
        n_hidden = 64
        n_layers = 3
        n_ensemble = 6
    elif dim <= 8:
        n_hidden = 128
        n_layers = 4
        n_ensemble = 8
    else:
        n_hidden = 256
        n_layers = 5
        n_ensemble = 10

    return {
        "experiment_name": f"gaussian_dim{dim}_{qofi}",
        "input_dim": dim,
        "max_epochs": 50,
        "lr": 0.001,
        "lr_final": 0.0001,
        "sigma": 0.3,
        "n_ensemble_members": n_ensemble,
        "batchsize": min(2048, 256 * dim),
        "n_hidden": n_hidden,
        "n_layers": n_layers,
        "num_train": 65536,
        "num_val": 2048,
        "num_samples": 10000,
        "save_freq": 50,
        "gpu_id": 0,
        "seed": 42,
        "phase": "train",
        "testing_epoch": -1,
        "qofi": qofi,
        "ddpm": 0
    }


def run_single_experiment(dim: int, qofi: str, phase: str) -> Optional[DimensionResult]:
    """Run experiment for a single dimension and collect results."""
    config = get_config_for_dimension(dim, qofi)
    config["phase"] = phase

    # Write config to temp file
    config_path = f"/tmp/gaussian_dim{dim}_{qofi}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Run experiment with the current interpreter
    python_path = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "gaussian_ensemble_cv.py")

    print(f"\n{'='*60}")
    print(f"Running {phase} for dimension {dim}, qofi={qofi}")
    print(f"{'='*60}")

    # Collect results from existing experiments

    return None


def collect_results_from_logs(dimensions: List[int], qofi: str) -> List[DimensionResult]:
    """Collect results from existing experiment logs."""
    results = []
    logs_dir = "logs"

    for dim in dimensions:
        # Find matching experiment directory
        pattern = f"gaussian_ensemble_cv_input_dim-{dim}_"
        matching_dirs = [d for d in os.listdir(logs_dir)
                        if d.startswith(pattern) and f"qofi-{qofi}" in d]

        if not matching_dirs:
            print(f"Warning: No experiment found for dim={dim}, qofi={qofi}")
            continue

        # Use most recent (by modification time)
        exp_dir = max(matching_dirs,
                     key=lambda d: os.path.getmtime(os.path.join(logs_dir, d)))
        exp_path = os.path.join(logs_dir, exp_dir)

        # Load checkpoint to get training history
        checkpoint_files = [f for f in os.listdir(exp_path) if f.startswith("checkpoint_")]
        if not checkpoint_files:
            # Try checkpoints subdirectory
            ckpt_subdir = os.path.join(logs_dir.replace("logs", "checkpoints"), exp_dir)
            if os.path.exists(ckpt_subdir):
                checkpoint_files = [f for f in os.listdir(ckpt_subdir) if f.startswith("checkpoint_")]
                exp_path = ckpt_subdir

        print(f"Found experiment for dim={dim}: {exp_dir}")

    return results


def run_test_and_collect(dim: int, qofi: str, device: str = "cuda:0", layer_type: str = "flat",
                         depth: int = None, n_ensemble: int = None,
                         max_epochs: int = None) -> Optional[DimensionResult]:
    """Run test phase and collect metrics for a dimension."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    from cncv import (
        create_random_split_ensemble,
        create_hierarchical_ensemble,
        create_shared_hierarchical_ensemble,
        compute_score_posterior_gaussian,
        compute_exact_posterior,
        load_ensemble_checkpoint,
    )

    # Setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Find checkpoint - use data/checkpoints directory
    ckpt_dir = "data/checkpoints"

    # Find matching experiment
    pattern = f"gaussian_ensemble_cv_input_dim-{dim}_"
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    all_dirs = os.listdir(ckpt_dir)

    # Filter by layer type
    # Note: projorg may truncate long directory names with a hash suffix,
    # so shared_hierarchical dirs won't contain "layer_type-shared_hierarchical".
    # For truncated dirs, we verify layer_type from the checkpoint itself.
    import re
    base_matching = [d for d in all_dirs if d.startswith(pattern) and f"qofi-{qofi}" in d]

    if layer_type == "flat":
        matching = [d for d in base_matching if "ddpm-0" in d and "layer_type-" not in d]
    elif layer_type == "shared_hierarchical":
        # Match either explicit name or hash-truncated projorg directories
        matching = [d for d in base_matching
                    if f"layer_type-{layer_type}" in d
                    or (re.search(r'\.[0-9a-f]{40}$', d) and "layer_type-hierarchical" not in d)]
    else:
        matching = [d for d in base_matching if f"layer_type-{layer_type}" in d]

    # Filter by depth if specified
    if depth is not None:
        matching = [d for d in matching if f"depth-{depth}" in d]

    # Filter by n_ensemble_members if specified
    if n_ensemble is not None:
        matching = [d for d in matching if f"n_ensemble_members-{n_ensemble}" in d]

    # Filter by max_epochs if specified
    if max_epochs is not None:
        matching = [d for d in matching if f"max_epochs-{max_epochs}" in d]

    if not matching:
        print(f"No experiment found for dim={dim}, qofi={qofi}, layer_type={layer_type}")
        return None

    exp_name = sorted(matching)[-1]  # Most recent by name
    print(f"Loading: {exp_name}")

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, exp_name)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint directory not found: {ckpt_path}")
        return None

    ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith("checkpoint_") and f.endswith(".pth")]
    if not ckpt_files:
        print(f"No checkpoint files found in {ckpt_path}")
        return None

    # Get latest checkpoint
    latest_ckpt = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    ckpt_file = os.path.join(ckpt_path, latest_ckpt)

    print(f"Loading checkpoint: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    # Extract problem parameters
    mu_prior = checkpoint["mu_prior"].to(device)
    Sigma_prior = checkpoint["Sigma_prior"].to(device)
    sigma = checkpoint["sigma"]
    n_dim = checkpoint.get("input_dim", dim)
    split_config = checkpoint.get("split_config", None)

    # Recreate ensemble with same configuration
    args = checkpoint["args"]
    ckpt_layer_type = checkpoint.get("layer_type", "flat")
    if ckpt_layer_type == "shared_hierarchical":
        ckpt_depth = getattr(args, "depth", None) or None
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim,
            args.n_hidden,
            args.n_ensemble_members,
            n_layers=args.n_layers,
            depth=ckpt_depth,
            n_cv=n_dim,
            seed=12,
        ).to(device)
    elif ckpt_layer_type == "hierarchical":
        ckpt_depth = getattr(args, "depth", None) or None  # 0 means auto
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim,
            args.n_hidden,
            args.n_ensemble_members,
            n_layers=args.n_layers,
            depth=ckpt_depth,
            n_cv=n_dim,
            seed=12,
        ).to(device)
    else:
        ensemble = create_random_split_ensemble(
            n_dim, n_dim,
            args.n_hidden,
            args.n_ensemble_members,
            args.n_layers,
            n_cv=n_dim,
            seed=12,
        ).to(device)

    # Load state dict
    ensemble.load_state_dict(checkpoint["ensemble_state_dict"])
    ensemble.eval()

    # Generate test data - average over multiple observations for stability
    L_prior = torch.linalg.cholesky(Sigma_prior)
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    I = torch.eye(n_dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1/sigma**2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)

    n_test_obs = 100
    num_samples = 10000
    all_vrf_per_component = []
    all_correlations = []
    all_var_vanilla = []
    all_var_cv = []
    all_mse_at_n100 = []

    torch.manual_seed(42)
    for obs_idx in range(n_test_obs):
        X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        Y_obs = X_true + sigma * torch.randn(n_dim, device=device)

        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1/sigma**2) * Y_obs)

        noise = torch.randn(num_samples, n_dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)

        with torch.no_grad():
            score = compute_score_posterior_gaussian(
                X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma
            )

            if qofi == "mean":
                h_x = X_samples
                true_qoi = mu_post
            else:
                h_x = (X_samples - mu_post) ** 2
                true_qoi = torch.diag(Sigma_post)

            g_combined, all_g = ensemble(X_samples, Y_samples, score)

        # Compute per-component metrics for this observation
        obs_correlations = []
        obs_var_vanilla = []
        obs_var_cv = []
        obs_vrf = []
        for i in range(n_dim):
            corr = torch.corrcoef(torch.stack([h_x[:, i], g_combined[i, :]]))[0, 1].item()
            obs_correlations.append(corr)
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            var_controlled = controlled.var().item()
            obs_var_vanilla.append(var_h)
            obs_var_cv.append(var_controlled)
            obs_vrf.append(var_controlled / max(var_h, 1e-12))

        all_vrf_per_component.append(obs_vrf)
        all_correlations.append(obs_correlations)
        all_var_vanilla.append(obs_var_vanilla)
        all_var_cv.append(obs_var_cv)

        # MSE analysis at n=100
        obs_mse = []
        for i in range(n_dim):
            cv_estimates = []
            for _ in range(200):
                idx = torch.randperm(num_samples)[:100]
                controlled = h_x[idx, i] - g_combined[i, idx]
                cv_estimates.append(controlled.mean().item())
            mse = np.mean([(est - true_qoi[i].item())**2 for est in cv_estimates])
            obs_mse.append(mse)
        all_mse_at_n100.append(obs_mse)

    # Average over observations
    avg_vrf = np.mean(all_vrf_per_component, axis=0).tolist()
    avg_corr = np.mean(all_correlations, axis=0).tolist()
    avg_var_vanilla = np.mean(all_var_vanilla, axis=0).tolist()
    avg_var_cv = np.mean(all_var_cv, axis=0).tolist()
    avg_mse = np.mean(all_mse_at_n100, axis=0).tolist()

    result = DimensionResult(
        dimension=n_dim,
        qofi=qofi,
        vrf_mean=np.mean(avg_vrf),
        vrf_std=np.std(avg_vrf),
        vrf_per_component=avg_vrf,
        correlation_mean=np.mean(avg_corr),
        correlation_per_component=avg_corr,
        variance_vanilla=avg_var_vanilla,
        variance_cv=avg_var_cv,
        mse_at_n100=avg_mse,
        training_loss_final=checkpoint["train_obj"][-1] if checkpoint["train_obj"] else 0,
        validation_loss_final=checkpoint["val_obj"][-1] if checkpoint["val_obj"] else 0,
    )

    print(f"  VRF: {result.vrf_mean:.3f} ± {result.vrf_std:.3f} (avg over {n_test_obs} obs)")
    print(f"  Correlation: {result.correlation_mean:.3f}")
    print(f"  Variance reduction: {(1-result.vrf_mean)*100:.1f}%")

    return result


def plot_dimension_scaling(results: List[DimensionResult], output_dir: str, qofi: str):
    """Create consolidated publication figure for dimension scaling."""
    os.makedirs(output_dir, exist_ok=True)

    dims = [r.dimension for r in results]
    vrfs = [r.vrf_mean for r in results]
    vrf_stds = [r.vrf_std for r in results]
    corrs = [r.correlation_mean for r in results]
    reductions = [(1 - v) * 100 for v in vrfs]

    # Figure 1: VRF and Correlation vs Dimension (2-panel, half-column each)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_COL, 2.2))

    # Panel A: Variance Reduction Factor vs Dimension
    ax1.errorbar(dims, vrfs, yerr=vrf_stds, fmt='o-', color=COLORS['cv'],
                 capsize=3, capthick=1, markersize=6, linewidth=1.5)
    ax1.axhline(1.0, color=COLORS['reference'], linestyle='--', alpha=0.6, linewidth=1)
    ax1.fill_between(dims, 0, vrfs, alpha=0.2, color=COLORS['cv'])
    ax1.set_xlabel('Dimension $d$')
    ax1.set_ylabel('VRF (lower is better)')
    ax1.set_ylim([0, 0.5])
    ax1.set_xticks(dims)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    # Add percentage annotations
    for d, r in zip(dims, reductions):
        ax1.annotate(f'{r:.0f}%', (d, vrfs[dims.index(d)] + 0.03),
                    ha='center', va='bottom', fontsize=7,
                    color=COLORS['cv'])

    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=10,
            fontweight='bold', va='top')

    # Panel B: Correlation vs Dimension
    ax2.plot(dims, corrs, 'o-', color=COLORS['training'], markersize=6, linewidth=1.5)
    ax2.axhline(1.0, color=COLORS['reference'], linestyle='--', alpha=0.6, linewidth=1)
    ax2.fill_between(dims, 0, corrs, alpha=0.2, color=COLORS['training'])
    ax2.set_xlabel('Dimension $d$')
    ax2.set_ylabel(r'Correlation $\rho(h, g)$')
    ax2.set_ylim([0.8, 1.0])
    ax2.set_xticks(dims)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=10,
            fontweight='bold', va='top')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'dimension_scaling_{qofi}.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()

    # Figure 2: Per-component VRF heatmap (if we have per-component data)
    if all(len(r.vrf_per_component) > 0 for r in results):
        max_dim = max(len(r.vrf_per_component) for r in results)
        vrf_matrix = np.full((len(results), max_dim), np.nan)

        for i, r in enumerate(results):
            vrf_matrix[i, :len(r.vrf_per_component)] = r.vrf_per_component

        fig, ax = plt.subplots(figsize=(HALF_COL, 2.0))

        im = ax.imshow(vrf_matrix.T, aspect='auto', cmap='RdYlGn_r',
                      vmin=0, vmax=0.5, interpolation='nearest')

        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel('Dimension $d$')
        ax.set_ylabel('Component')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('VRF')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'vrf_heatmap_{qofi}.pdf')
        plt.savefig(save_path)
        plt.savefig(save_path.replace('.pdf', '.png'))
        print(f"Saved: {save_path}")
        plt.close()


def plot_combined_mean_var(mean_results: List[DimensionResult],
                           var_results: List[DimensionResult],
                           output_dir: str):
    """Create single figure comparing mean and variance estimation."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_COL, 2.2))

    # Match dimensions
    mean_dims = {r.dimension: r for r in mean_results}
    var_dims = {r.dimension: r for r in var_results}
    common_dims = sorted(set(mean_dims.keys()) & set(var_dims.keys()))

    if not common_dims:
        print("No common dimensions between mean and var results")
        return

    # Panel A: VRF comparison
    mean_vrfs = [mean_dims[d].vrf_mean for d in common_dims]
    var_vrfs = [var_dims[d].vrf_mean for d in common_dims]

    x = np.arange(len(common_dims))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mean_vrfs, width,
                    label='Mean', color=COLORS['training'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, var_vrfs, width,
                    label='Variance', color=COLORS['cv'], alpha=0.8)

    ax1.set_xlabel('Dimension $d$')
    ax1.set_ylabel('VRF (lower is better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in common_dims])
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax1.set_ylim([0, 0.5])

    ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=10,
            fontweight='bold', va='top')

    # Panel B: Correlation comparison
    mean_corrs = [mean_dims[d].correlation_mean for d in common_dims]
    var_corrs = [var_dims[d].correlation_mean for d in common_dims]

    ax2.plot(common_dims, mean_corrs, 'o-', color=COLORS['training'],
             label='Mean', markersize=6, linewidth=1.5)
    ax2.plot(common_dims, var_corrs, 's--', color=COLORS['cv'],
             label='Variance', markersize=6, linewidth=1.5)

    ax2.set_xlabel('Dimension $d$')
    ax2.set_ylabel(r'Correlation $\rho(h, g)$')
    ax2.set_xticks(common_dims)
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    min_corr = min(min(mean_corrs), min(var_corrs))
    ax2.set_ylim([min_corr - 0.05, 1.01])

    ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=10,
            fontweight='bold', va='top')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'dimension_scaling_combined.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Saved: {save_path}")
    plt.close()


def generate_latex_table(results: List[DimensionResult], qofi: str) -> str:
    """Generate LaTeX table summarizing results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Variance reduction results for " + qofi + r" estimation across dimensions.}",
        r"\label{tab:" + qofi + r"_results}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Dimension & VRF & Reduction (\%) & Correlation \\",
        r"\midrule",
    ]

    for r in sorted(results, key=lambda x: x.dimension):
        reduction = (1 - r.vrf_mean) * 100
        lines.append(
            f"${r.dimension}$ & ${r.vrf_mean:.3f} \\pm {r.vrf_std:.3f}$ & "
            f"${reduction:.1f}$ & ${r.correlation_mean:.3f}$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run dimension scaling experiments')
    parser.add_argument('--phase', type=str, default='test',
                       choices=['train', 'test', 'plot'],
                       help='Phase: train, test, or plot')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[2, 4, 8, 16],
                       help='Dimensions to test')
    parser.add_argument('--qofi', type=str, nargs='+', default=['mean', 'var'],
                       help='Quantities of interest')
    parser.add_argument('--output_dir', type=str,
                       default='figs/gaussian',
                       help='Output directory for figures')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--layer_type', type=str, default='flat',
                       choices=['flat', 'hierarchical', 'shared_hierarchical'],
                       help='Layer type to evaluate')
    parser.add_argument('--depth', type=int, default=None,
                       help='Filter checkpoints by depth (e.g. 2)')
    parser.add_argument('--n_ensemble', type=int, default=16,
                       help='Filter checkpoints by n_ensemble_members')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Filter checkpoints by max_epochs')
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    all_results = {}

    for qofi in args.qofi:
        print(f"\n{'#'*60}")
        print(f"Processing qofi={qofi}, layer_type={args.layer_type}")
        print(f"{'#'*60}")

        results = []
        for dim in args.dimensions:
            if args.phase == 'test':
                result = run_test_and_collect(dim, qofi, device, layer_type=args.layer_type,
                                                     depth=args.depth, n_ensemble=args.n_ensemble,
                                                     max_epochs=args.max_epochs)
                if result:
                    results.append(result)
            elif args.phase == 'train':
                run_single_experiment(dim, qofi, 'train')

        if results:
            all_results[qofi] = results

            # Generate plots
            plot_dimension_scaling(results, args.output_dir, qofi)

            # Print LaTeX table
            print(f"\n=== LaTeX Table for {qofi} ===")
            print(generate_latex_table(results, qofi))

    # Combined plot if both mean and var results exist
    if 'mean' in all_results and 'var' in all_results:
        plot_combined_mean_var(all_results['mean'], all_results['var'], args.output_dir)

    # Save results to JSON
    results_file = os.path.join(args.output_dir, 'dimension_scaling_results.json')
    save_results = {qofi: [asdict(r) for r in results]
                   for qofi, results in all_results.items()}
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved results to: {results_file}")


if __name__ == "__main__":
    main()
