"""Generate Figure 2: scatter correlation plots for d=4 mean estimation.

Loads the d=4 mean shared_hierarchical checkpoint, draws posterior samples,
and produces a 1x4 panel of h_i vs g_i scatter plots.

Usage:
    python scripts/generate_scatter_correlation.py
    python scripts/generate_scatter_correlation.py --output_dir figs/gaussian
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
    parser = argparse.ArgumentParser(description="Generate scatter correlation figure")
    parser.add_argument("--dim", type=int, default=4, help="Dimension")
    parser.add_argument("--qofi", type=str, default="mean", help="Quantity of interest")
    parser.add_argument("--n_ensemble", type=int, default=16,
                        help="Ensemble size L to select (paper uses 16)")
    parser.add_argument("--num_samples", type=int, default=10000)
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

    noise = torch.randn(args.num_samples, n_dim, device=device)
    X_samples = mu_post + (L_post @ noise.T).T
    Y_samples = Y_obs.repeat(args.num_samples, 1)

    with torch.no_grad():
        score = compute_score_posterior_gaussian(
            X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma
        )
        h_x = X_samples
        g_combined, _ = ensemble(X_samples, Y_samples, score)

    # Plot 1xd layout
    fig, axes = plt.subplots(1, n_dim, figsize=(6.75, 2.2))
    if n_dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        h_i = h_x[:, i].cpu().numpy()
        g_i = g_combined[i, :].cpu().numpy()
        corr = np.corrcoef(h_i, g_i)[0, 1]

        ax.scatter(h_i, g_i, alpha=0.1, s=1.5, c=COMP_COLORS[i % len(COMP_COLORS)],
                   rasterized=True)
        lims = [min(h_i.min(), g_i.min()), max(h_i.max(), g_i.max())]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=0.8)

        ax.set_xlabel(r"$h_%d$" % (i + 1))
        if i == 0:
            ax.set_ylabel(r"$g_i$")
        else:
            ax.set_yticklabels([])
        ax.set_title(r"$\rho=%.3f$" % corr)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout(w_pad=0.3)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "scatter_correlation.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
