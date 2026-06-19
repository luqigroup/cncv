"""Generate Figure 3: sample efficiency plots for d=4 mean estimation.

Loads the d=4 mean shared_hierarchical checkpoint, runs bootstrap trials
at varying sample sizes, and produces a 1x2 panel of VRF and MSE vs N
for all d components.

Usage:
    python scripts/generate_sample_efficiency.py
    python scripts/generate_sample_efficiency.py --output_dir figs/gaussian
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cncv import (
    create_shared_hierarchical_ensemble,
    compute_score_posterior_gaussian,
    compute_exact_posterior,
)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
})

COMP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def find_checkpoint(dim, qofi, layer_type="shared_hierarchical", n_ensemble=16):
    """Find the latest matching checkpoint directory (paper config: L=16)."""
    import re

    ckpt_base = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints")
    ckpt_base = os.path.normpath(ckpt_base)
    pattern = f"gaussian_ensemble_cv_input_dim-{dim}_"

    all_dirs = os.listdir(ckpt_base)
    base_matching = [d for d in all_dirs if d.startswith(pattern) and f"qofi-{qofi}" in d]

    if layer_type == "shared_hierarchical":
        matching = [
            d for d in base_matching
            if f"layer_type-{layer_type}" in d
            or (re.search(r"\.[0-9a-f]{40}$", d) and "layer_type-hierarchical" not in d)
        ]
    else:
        matching = [d for d in base_matching if f"layer_type-{layer_type}" in d]

    # Restrict to the requested ensemble size. Without this, sorted()[-1] can
    # select a run whose state dict is incompatible with the current model.
    if n_ensemble is not None:
        matching = [d for d in matching if f"n_ensemble_members-{n_ensemble}" in d]

    # Keep only runs that actually contain checkpoint files. A stale or aborted
    # run can leave an empty directory whose name still matches the pattern;
    # selecting it would crash max() on an empty checkpoint list below.
    def _has_checkpoints(name):
        run_dir = os.path.join(ckpt_base, name)
        return os.path.isdir(run_dir) and any(
            f.startswith("checkpoint_") and f.endswith(".pth")
            for f in os.listdir(run_dir)
        )

    matching = [d for d in matching if _has_checkpoints(d)]

    if not matching:
        raise FileNotFoundError(
            f"No checkpoint found for dim={dim}, qofi={qofi}, layer_type={layer_type}"
        )

    exp_name = sorted(matching)[-1]
    ckpt_path = os.path.join(ckpt_base, exp_name)
    ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith("checkpoint_") and f.endswith(".pth")]
    latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckpt_path, latest)


def main():
    parser = argparse.ArgumentParser(description="Generate sample efficiency figure")
    parser.add_argument("--dim", type=int, default=4, help="Dimension")
    parser.add_argument("--qofi", type=str, default="mean", help="Quantity of interest")
    parser.add_argument("--n_ensemble", type=int, default=16,
                        help="Ensemble size L to select (paper uses 16)")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Total posterior samples to draw")
    parser.add_argument("--n_trials", type=int, default=1000,
                        help="Bootstrap trials per sample size")
    parser.add_argument("--output_dir", type=str,
                        default="figs/gaussian")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Direct checkpoint path; overrides recipe discovery")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt_file = args.checkpoint or find_checkpoint(args.dim, args.qofi, n_ensemble=args.n_ensemble)
    print(f"Loading: {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    n_dim = args.dim
    train_args = ckpt["args"]
    mu_prior = ckpt["mu_prior"].to(device)
    Sigma_prior = ckpt["Sigma_prior"].to(device)
    sigma = ckpt["sigma"]
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)

    # Recreate ensemble
    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_dim, train_args.n_hidden, train_args.n_ensemble_members,
        n_layers=train_args.n_layers,
        depth=getattr(train_args, "depth", None) or None,
        n_cv=n_dim, seed=12,
    ).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

    # Generate test data
    torch.manual_seed(42)
    L_prior = torch.linalg.cholesky(Sigma_prior)
    X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
    Y_obs = X_true + sigma * torch.randn(n_dim, device=device)
    mu_post, Sigma_post = compute_exact_posterior(Y_obs, mu_prior, Sigma_prior, sigma)
    L_post = torch.linalg.cholesky(Sigma_post)

    # Draw large pool of samples
    noise = torch.randn(args.num_samples, n_dim, device=device)
    X_samples = mu_post + (L_post @ noise.T).T
    Y_samples = Y_obs.repeat(args.num_samples, 1)

    with torch.no_grad():
        score = compute_score_posterior_gaussian(
            X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma
        )
        h_x = X_samples  # [N, d]
        g_combined, _ = ensemble(X_samples, Y_samples, score)  # [d, N]

    h_np = h_x.cpu().numpy()        # [N, d]
    g_np = g_combined.cpu().numpy()  # [d, N]
    exact_mean = mu_post.cpu().numpy()  # [d]

    # Bootstrap trials
    rng = np.random.default_rng(123)
    # vrf_results[i][j] = list of VRF values for component i at sample size j
    vrf_results = [[[] for _ in range(len(SAMPLE_SIZES))] for _ in range(n_dim)]
    mse_raw = [[[] for _ in range(len(SAMPLE_SIZES))] for _ in range(n_dim)]
    mse_cv = [[[] for _ in range(len(SAMPLE_SIZES))] for _ in range(n_dim)]

    print("Running bootstrap trials...")
    for si, N in enumerate(SAMPLE_SIZES):
        for _ in range(args.n_trials):
            idx = rng.choice(args.num_samples, size=N, replace=False)
            for comp in range(n_dim):
                h_sub = h_np[idx, comp]
                g_sub = g_np[comp, idx]
                resid = h_sub - g_sub

                var_h = np.var(h_sub, ddof=1)
                var_r = np.var(resid, ddof=1)
                vrf = var_r / var_h if var_h > 0 else 1.0
                vrf_results[comp][si].append(vrf)

                est_raw = np.mean(h_sub)
                est_cv = np.mean(resid)
                mse_raw[comp][si].append((est_raw - exact_mean[comp]) ** 2)
                mse_cv[comp][si].append((est_cv - exact_mean[comp]) ** 2)
        print(f"  N={N:5d} done")

    # Compute medians
    vrf_med = np.array([[np.median(vrf_results[c][s]) for s in range(len(SAMPLE_SIZES))]
                        for c in range(n_dim)])
    mse_raw_med = np.array([[np.median(mse_raw[c][s]) for s in range(len(SAMPLE_SIZES))]
                            for c in range(n_dim)])
    mse_cv_med = np.array([[np.median(mse_cv[c][s]) for s in range(len(SAMPLE_SIZES))]
                           for c in range(n_dim)])

    # Mean across components
    vrf_mean = vrf_med.mean(axis=0)
    mse_raw_mean = mse_raw_med.mean(axis=0)
    mse_cv_mean = mse_cv_med.mean(axis=0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.5))

    # Panel (a): VRF vs N
    for c in range(n_dim):
        ax1.plot(SAMPLE_SIZES, vrf_med[c], "-o", color=COMP_COLORS[c],
                 markersize=4, linewidth=1.2, alpha=0.7, label=f"$j={c+1}$")
    ax1.plot(SAMPLE_SIZES, vrf_mean, "k-s", markersize=5, linewidth=1.8,
             label="Mean", zorder=5)
    ax1.set_xscale("log")
    ax1.set_xlabel("Sample size $N$")
    ax1.set_ylabel("VRF")
    ax1.set_title("(a) Variance reduction factor")
    ax1.legend(ncol=5, fontsize=8, loc="upper center",
               bbox_to_anchor=(0.5, -0.26), frameon=False, columnspacing=1.0,
               handletextpad=0.4)
    ax1.grid(alpha=0.2, linewidth=0.5)

    # Panel (b): MSE vs N
    for c in range(n_dim):
        ax2.plot(SAMPLE_SIZES, mse_raw_med[c], "--", color=COMP_COLORS[c],
                 markersize=3, linewidth=1.0, alpha=0.5)
        ax2.plot(SAMPLE_SIZES, mse_cv_med[c], "-o", color=COMP_COLORS[c],
                 markersize=4, linewidth=1.2, alpha=0.7, label=f"CV $j={c+1}$")
    # Proxy artist labelling all dashed coloured lines as Raw MC
    ax2.plot([], [], "--", color="gray", alpha=0.6, label="Raw MC (per $j$)")
    ax2.plot(SAMPLE_SIZES, mse_raw_mean, "k--^", markersize=4, linewidth=1.5,
             alpha=0.6, label="Raw (mean)")
    ax2.plot(SAMPLE_SIZES, mse_cv_mean, "k-s", markersize=5, linewidth=1.8,
             label="CV (mean)", zorder=5)
    # 1/N reference line
    n_arr = np.array(SAMPLE_SIZES, dtype=float)
    c_ref = mse_raw_mean[0] * n_arr[0]
    ax2.plot(n_arr, c_ref / n_arr, "k:", linewidth=1.0, alpha=0.5,
             label="$1/N$ reference")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample size $N$")
    ax2.set_ylabel("MSE")
    ax2.set_title("(b) Mean squared error")
    ax2.legend(ncol=4, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, -0.26), frameon=False, columnspacing=1.0,
               handletextpad=0.4)
    ax2.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout(w_pad=0.5)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "sample_efficiency.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
