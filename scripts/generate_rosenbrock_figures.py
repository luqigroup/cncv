#!/usr/bin/env python
"""Generate publication-quality figures for Rosenbrock example.

Loads the trained hierarchical checkpoint and evaluates on multiple
test observations to demonstrate amortization.
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv import (
    create_hierarchical_ensemble,
    create_shared_hierarchical_ensemble,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
    load_ensemble_checkpoint,
)
from cncv.utils import CustomLRScheduler, pSGLD

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
})

FULL_COL = 6.75


def compute_rosenbrock_posterior_grid(X1, X2, y1, y2, mu=0.0, a=0.5, b=1.0, sigma=0.3):
    """Compute Rosenbrock posterior density on grid."""
    log_prior = -a * (X1 - mu)**2 - b * (X2 - X1**2)**2
    log_like = -0.5 * ((X1 - y1)**2 + (X2 - y2)**2) / sigma**2
    log_post = log_prior + log_like
    return np.exp(log_post - log_post.max())


def sample_posterior_psgld(Y_obs, rosenbrock_dist, sigma, num_samples, device):
    """Sample posterior via pSGLD."""
    n_dim = Y_obs.shape[0]
    lr_initial = 0.01
    lr_final = 0.001
    max_itr = 10 * num_samples
    thinning = 2

    x = rosenbrock_dist.sample(1, device=device).squeeze(0)
    x = x.detach().requires_grad_(True)

    optimizer = pSGLD([x], lr=lr_initial)
    lr_scheduler = CustomLRScheduler(optimizer, lr_initial, lr_final, max_itr)

    all_samples = []
    for itr in range(max_itr):
        neg_log_lik = 0.5 * ((Y_obs - x) ** 2).sum() / (sigma ** 2)
        neg_log_prior = -rosenbrock_dist.logpdf(x.unsqueeze(0)).squeeze()
        energy = neg_log_lik + neg_log_prior

        optimizer.zero_grad()
        energy.backward()
        lr_scheduler.step()
        optimizer.step()

        all_samples.append(x.detach().clone())

    all_samples = torch.stack(all_samples)
    burnin = max_itr // 2
    return all_samples[burnin::thinning]


def evaluate_checkpoint(checkpoint_path, test_observations, device="cuda:0"):
    """Load checkpoint and evaluate VRFs on test observations."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]

    # Reconstruct distribution
    rosenbrock_dist = RosenbrockDistribution(
        mu=checkpoint["mu"], a=checkpoint["a"],
        b=checkpoint["b"], n1=checkpoint["n1"], n2=checkpoint["n2"],
    )
    n_dim = rosenbrock_dist.ndim
    sigma = checkpoint["sigma"]

    # Reconstruct ensemble
    depth = getattr(args, "depth", None) or None
    layer_type = checkpoint.get("layer_type", getattr(args, "layer_type", "hierarchical"))

    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim, args.n_hidden, args.n_ensemble_members,
            n_layers=args.n_layers, depth=depth, n_cv=n_dim, seed=1,
        ).to(device)
    else:
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim, args.n_hidden, args.n_ensemble_members,
            n_layers=args.n_layers, depth=depth, n_cv=n_dim, seed=1,
        ).to(device)

    ensemble.load_state_dict(checkpoint["ensemble_state_dict"])
    ensemble.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  layer_type={layer_type}, n_dim={n_dim}, n_hidden={args.n_hidden}, n_layers={args.n_layers}")
    print(f"  n_ensemble={args.n_ensemble_members}, depth={depth}")

    results = []
    num_samples = 5000

    for obs_idx, y_tuple in enumerate(test_observations):
        Y_obs = torch.tensor(y_tuple, dtype=torch.float32, device=device)

        # Sample posterior
        print(f"  Sampling posterior for y={y_tuple}...")
        torch.manual_seed(42 + obs_idx)
        X_samples = sample_posterior_psgld(Y_obs, rosenbrock_dist, sigma, num_samples, device)
        Y_samples = Y_obs.repeat(X_samples.shape[0], 1)

        # Compute CV
        with torch.no_grad():
            score = compute_score_posterior_rosenbrock(
                X_samples, Y_samples, rosenbrock_dist, sigma
            )
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        # h(x) = x for mean estimation
        h_x = X_samples

        # Per-component VRF
        vrf_per_comp = []
        for k in range(n_dim):
            var_h = h_x[:, k].var().item()
            var_controlled = (h_x[:, k] - g_combined[k, :]).var().item()
            vrf_per_comp.append(var_controlled / max(var_h, 1e-12))

        avg_vrf = np.mean(vrf_per_comp)
        reduction = (1 - avg_vrf) * 100
        print(f"  y={y_tuple}: VRF={vrf_per_comp}, avg={avg_vrf:.3f} ({reduction:.1f}%)")
        results.append({'y': y_tuple, 'vrf': vrf_per_comp})

    return results, rosenbrock_dist, sigma


def create_rosenbrock_figure(output_dir, eval_results):
    """Create figure showing amortization across different observations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ['#E94F37', '#2E86AB', '#4ECDC4']
    labels = [r'$\mathbf{y}^{(1)}$', r'$\mathbf{y}^{(2)}$', r'$\mathbf{y}^{(3)}$']

    test_cases = []
    for i, r in enumerate(eval_results):
        test_cases.append({
            'y': r['y'], 'label': labels[i], 'color': colors[i], 'vrf': r['vrf'],
        })

    fig = plt.figure(figsize=(FULL_COL, 2.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.45)

    for idx, case in enumerate(test_cases):
        ax = fig.add_subplot(gs[idx])
        y1, y2 = case['y']

        if y1 < 0:
            x1 = np.linspace(-2.0, 1.5, 120)
            x2 = np.linspace(-0.3, 2.5, 120)
            xlim, ylim = (-1.8, 1.3), (-0.2, 2.3)
        elif y1 > 1:
            x1 = np.linspace(-0.5, 2.2, 120)
            x2 = np.linspace(-0.3, 4.0, 120)
            xlim, ylim = (-0.3, 2.0), (-0.2, 3.5)
        else:
            x1 = np.linspace(-1.5, 1.8, 120)
            x2 = np.linspace(-0.3, 3.0, 120)
            xlim, ylim = (-1.3, 1.6), (-0.2, 2.8)

        X1, X2 = np.meshgrid(x1, x2)
        post = compute_rosenbrock_posterior_grid(X1, X2, y1, y2)

        ax.contourf(X1, X2, post, levels=12, cmap='Blues', alpha=0.5)
        ax.contour(X1, X2, post, levels=5, colors='#2E86AB', alpha=0.6, linewidths=0.5)

        # Rejection samples for visualization
        np.random.seed(int(abs(y1 * 1000 + y2 * 100)) % 2**31)
        sx1, sx2 = [], []
        while len(sx1) < 150:
            x1_p = np.random.uniform(xlim[0], xlim[1], 3000)
            x2_p = np.random.uniform(ylim[0], ylim[1], 3000)
            lp = -0.5*(x1_p - 0)**2 - 1.0*(x2_p - x1_p**2)**2
            ll = -0.5*((x1_p - y1)**2 + (x2_p - y2)**2) / 0.3**2
            lpost = lp + ll
            accept = np.log(np.random.rand(3000)) < (lpost - lpost.max() + 3.0)
            sx1.extend(x1_p[accept][:150 - len(sx1)])
            sx2.extend(x2_p[accept][:150 - len(sx2)])

        ax.scatter(sx1, sx2, s=2, alpha=0.4, c=case['color'], zorder=5)
        ax.scatter([y1], [y2], s=70, marker='*', c=case['color'],
                   edgecolors='black', linewidths=0.6, zorder=10)

        avg_vrf = np.mean(case['vrf'])
        ax.text(0.97, 0.97, f'VRF={avg_vrf:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor=case['color'], alpha=0.9, linewidth=0.8))

        ax.set_xlabel(r'$x_1$')
        if idx == 0:
            ax.set_ylabel(r'$x_2$')
        ax.set_title(f'({chr(97+idx)}) {case["label"]}', pad=2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', pad=1)

    # Panel (d): Summary bar chart
    ax4 = fig.add_subplot(gs[3])
    n_dim = len(test_cases[0]['vrf'])
    x_pos = np.arange(3)
    width = 0.38

    vrf_x1 = [c['vrf'][0] for c in test_cases]
    vrf_x2 = [c['vrf'][1] for c in test_cases]

    ax4.bar(x_pos - width/2, vrf_x1, width, label=r'$x_1$',
            color=[c['color'] for c in test_cases], alpha=0.7)
    ax4.bar(x_pos + width/2, vrf_x2, width, label=r'$x_2$',
            color=[c['color'] for c in test_cases], alpha=0.95,
            hatch='///', edgecolor='white', linewidth=0.5)

    ax4.set_ylabel('VRF')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([r'$\mathbf{y}^{(1)}$', r'$\mathbf{y}^{(2)}$', r'$\mathbf{y}^{(3)}$'])
    ax4.set_ylim(0, 0.5)
    ax4.axhline(y=0.2, color='gray', linestyle='--', alpha=0.4, linewidth=0.6)
    ax4.legend(loc='upper right', framealpha=0.9, handlelength=1.2)
    ax4.set_title('(d) Per-component', pad=2)
    ax4.tick_params(axis='both', which='major', pad=1)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18)

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'rosenbrock_results.{fmt}')
    print(f"Saved rosenbrock_results.pdf/png to {output_dir}")
    plt.close(fig)

    avg_reduction = np.mean([(1 - np.mean(c['vrf'])) * 100 for c in test_cases])
    return {'avg_reduction': avg_reduction, 'test_cases': test_cases}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pth file')
    parser.add_argument('--output_dir', type=str,
                        default='figs/rosenbrock')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    test_observations = [(-0.8, 0.9), (1.3, 2.0), (0.2, 1.5)]

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    eval_results, _, _ = evaluate_checkpoint(
        args.checkpoint, test_observations, device=device
    )

    results = create_rosenbrock_figure(args.output_dir, eval_results)

    print(f"\n=== Rosenbrock Results ===")
    print(f"Average variance reduction: {results['avg_reduction']:.1f}%")
    for case in results['test_cases']:
        red = (1 - np.mean(case['vrf'])) * 100
        print(f"  {case['label']}: y={case['y']}, VRF={case['vrf']}, {red:.0f}% reduction")
