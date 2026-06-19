"""Appendix figures: CNF quality diagnostics for Darcy flow.

Generates three publication figures:
  Figure: KL scatter plots (3x2) — prior + posterior for 3 observations
  Figure: CNF score vs physics score comparison
  Figure: Training and validation loss curves

Usage:
    python scripts/plot_cnf_diagnostics.py --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cncv.models.conditional_nf import ConditionalNF
from cncv.utils.darcy import DarcyKLPrior, DarcyForwardModel, compute_score_darcy
from cncv.utils.download import ensure_dataset


# ── Style ───────────────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })


def load_cnf(ckpt_path, device):
    """Load CNF model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cnf = ConditionalNF(
        n_in=ckpt["d_z"], n_cond=ckpt["d_y"],
        n_hidden=ckpt["cnf_hidden"],
        n_flow_layers=ckpt["cnf_flow_layers"],
        depth=ckpt.get("cnf_depth", None),
        n_mlp_layers=ckpt["cnf_mlp_layers"],
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)
    print(f"CNF: epoch {ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params")
    return cnf, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnf_checkpoint", type=str,
                        default="data/darcy/checkpoints_cnf_kl_v2_beskos/cnf_best.pth")
    parser.add_argument("--cncv_checkpoint", type=str,
                        default="data/darcy/checkpoints_cncv_kl_v2_beskos/cncv_best.pth")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--n_cnf_samples", type=int, default=1000,
                        help="CNF posterior samples per observation")
    parser.add_argument("--n_score_obs", type=int, default=100,
                        help="Number of test observations for score comparison")
    parser.add_argument("--output_dir", type=str, default="figs/darcy_kl_v2")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CNF
    cnf, cnf_ckpt = load_cnf(args.cnf_checkpoint, device)
    y_mean = cnf_ckpt["y_mean"].to(device)
    y_std = cnf_ckpt["y_std"].to(device)
    d_z = cnf_ckpt["d_z"]
    d_y = cnf_ckpt["d_y"]

    # Load data
    print(f"\nLoading: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    z_all = torch.tensor(data["u_kl"], dtype=torch.float32)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    sigma_y = float(data["sigma_y"])
    grid_size = int(data["grid_size"])

    # Prior samples for scatter
    z_prior = z_all[:1000]

    # Test observations (from end of dataset)
    test_start = z_all.shape[0] - 100
    obs_indices = [0, 2, 4]
    obs_labels = ["Obs 1", "Obs 2", "Obs 3"]

    setup_style()

    # ================================================================
    # FIGURE: KL scatter — 3x2 panel
    # ================================================================
    print(f"\n{'='*60}")
    print("KL scatter plots (3 observations)")
    print(f"{'='*60}")

    fig, axes = plt.subplots(3, 2, figsize=(3.4, 4.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.14, right=0.96,
                        top=0.95, bottom=0.06)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for row, (oidx, olabel) in enumerate(zip(obs_indices, obs_labels)):
        global_idx = test_start + oidx
        if global_idx >= z_all.shape[0]:
            print(f"  Warning: index {global_idx} out of range, skipping")
            continue

        z_true = z_all[global_idx]
        y_obs = y_all[global_idx]
        y_obs_n = ((y_obs - y_mean.cpu()) / y_std.cpu()).to(device)

        with torch.no_grad():
            z_post = cnf.sample(y_obs_n, args.n_cnf_samples).cpu()
        z_post_mean = z_post.mean(0)

        # Left panel: prior + posterior + z_true
        ax_l = axes[row, 0]
        ax_l.scatter(z_prior[:, 0].numpy(), z_prior[:, 1].numpy(),
                     s=1, alpha=0.15, c="#4C72B0", rasterized=True,
                     label="Prior")
        ax_l.scatter(z_post[:, 0].numpy(), z_post[:, 1].numpy(),
                     s=2, alpha=0.3, c="#DD8452", rasterized=True,
                     label="CNF post.")
        ax_l.scatter(z_true[0].item(), z_true[1].item(),
                     s=25, marker="*", c="red", zorder=5,
                     label=r"$\mathbf{z}^\dagger$")
        ax_l.set_title(f"{panel_labels[2*row]} {olabel}", fontsize=8, pad=3)
        if row == 2:
            ax_l.set_xlabel(r"$z_1$")
        ax_l.set_ylabel(r"$z_2$")
        if row == 0:
            ax_l.legend(fontsize=5.5, loc="upper right", handletextpad=0.3,
                        markerscale=1.5)

        # Right panel: zoomed posterior
        ax_r = axes[row, 1]
        ax_r.scatter(z_post[:, 0].numpy(), z_post[:, 1].numpy(),
                     s=2, alpha=0.3, c="#DD8452", rasterized=True)
        ax_r.scatter(z_true[0].item(), z_true[1].item(),
                     s=25, marker="*", c="red", zorder=5,
                     label=r"$\mathbf{z}^\dagger$")
        ax_r.scatter(z_post_mean[0].item(), z_post_mean[1].item(),
                     s=18, marker="D", c="#55A868", zorder=5,
                     label=r"CNF mean")

        z0_lo = z_post[:, 0].quantile(0.01).item()
        z0_hi = z_post[:, 0].quantile(0.99).item()
        z1_lo = z_post[:, 1].quantile(0.01).item()
        z1_hi = z_post[:, 1].quantile(0.99).item()
        margin0 = 0.15 * (z0_hi - z0_lo)
        margin1 = 0.15 * (z1_hi - z1_lo)
        ax_r.set_xlim(z0_lo - margin0, z0_hi + margin0)
        ax_r.set_ylim(z1_lo - margin1, z1_hi + margin1)

        ax_r.set_title(f"{panel_labels[2*row+1]} {olabel} (zoom)",
                       fontsize=8, pad=3)
        if row == 2:
            ax_r.set_xlabel(r"$z_1$")
        if row == 0:
            ax_r.legend(fontsize=5.5, loc="upper right", handletextpad=0.3,
                        markerscale=1.5)

    for suffix in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir,
                                 f"darcy_kl_scatter.{suffix}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_dir}/darcy_kl_scatter.pdf")

    # ================================================================
    # FIGURE: Per-component score correlation (CNF vs physics)
    # ================================================================
    print(f"\n{'='*60}")
    print("Per-component score correlation (CNF vs physics)")
    print(f"{'='*60}")

    prior = DarcyKLPrior(K=10, grid_size=grid_size,
                            alpha=0.0, s=1.1, sigma=1.0)
    print("Initializing PDE solver...")
    fwd = DarcyForwardModel(prior, n_sensors=33, time_steps=5000)

    n_score_obs = min(args.n_score_obs, z_all.shape[0] - test_start)
    all_physics_scores = []
    all_cnf_scores = []

    for i_obs in tqdm(range(n_score_obs), desc="Score comparison"):
        global_idx = test_start + i_obs
        if global_idx >= z_all.shape[0]:
            break

        y_obs_raw = y_all[global_idx].numpy()
        y_obs_n = ((y_all[global_idx] - y_mean.cpu()) / y_std.cpu()).to(device)

        z_true = z_all[global_idx]
        z_np = z_true.numpy()

        phys_score = compute_score_darcy(
            z_np, y_obs_raw, fwd, prior, sigma_y)
        all_physics_scores.append(phys_score)

        z_t = z_true.unsqueeze(0).to(device)
        cnf_score = cnf.score(
            z_t, y_obs_n.unsqueeze(0)
        ).cpu().numpy().ravel()
        all_cnf_scores.append(cnf_score)

    all_physics_scores = np.array(all_physics_scores)
    all_cnf_scores = np.array(all_cnf_scores)

    per_comp_r = np.array([
        np.corrcoef(all_physics_scores[:, k],
                     all_cnf_scores[:, k])[0, 1]
        for k in range(d_z)])

    print(f"  Per-comp r: mean={np.nanmean(per_comp_r):.3f}, "
          f"median={np.nanmedian(per_comp_r):.3f}, "
          f"min={np.nanmin(per_comp_r):.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.4))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.25)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, d_z))
    ax.bar(range(d_z), per_comp_r, color=colors,
           width=1.0, edgecolor="none", alpha=0.85)
    ax.axhline(np.nanmean(per_comp_r), color="C3", ls="--", lw=0.8,
               label=f"Mean $r = {np.nanmean(per_comp_r):.2f}$")
    ax.set_xlabel("KL component index $j$")
    ax.set_ylabel(r"Correlation $r_j$")
    ax.set_xlim(-1, d_z)
    ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=6, loc="lower right")

    for suffix in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir,
                                 f"darcy_score_comparison.{suffix}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_dir}/darcy_score_comparison.pdf")

    # ================================================================
    # FIGURE: Training and validation loss curves
    # ================================================================
    print(f"\n{'='*60}")
    print("Loss curves")
    print(f"{'='*60}")

    cncv_ckpt = torch.load(args.cncv_checkpoint, map_location="cpu",
                           weights_only=False)

    cnf_train = np.array(cnf_ckpt["train_losses"])
    cnf_val = np.array(cnf_ckpt["val_losses"])
    cncv_train = np.array(cncv_ckpt["train_losses"])
    cncv_val = np.array(cncv_ckpt["val_losses"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.4, 1.6))
    fig.subplots_adjust(wspace=0.50, left=0.12, right=0.97,
                        top=0.88, bottom=0.22)

    def smooth(x, k=50):
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="valid")

    # (a) CNF loss
    ax_a.plot(cnf_train, alpha=0.15, lw=0.3, c="#4C72B0")
    cnf_smooth = smooth(cnf_train)
    ax_a.plot(np.arange(49, 49 + len(cnf_smooth)), cnf_smooth,
              lw=1.0, c="#4C72B0", label="Train")

    batches_per_epoch = len(cnf_train) / len(cnf_val)
    val_iters = np.arange(len(cnf_val)) * batches_per_epoch
    ax_a.plot(val_iters, cnf_val, lw=1.0, c="#DD8452", label="Val")
    ax_a.set_xlabel("Batch iteration")
    ax_a.set_ylabel("NLL")
    ax_a.set_title("(a) CNF", fontsize=8, pad=3)
    ax_a.legend(fontsize=6, loc="upper right")
    ax_a.tick_params(labelsize=7)

    # (b) CNCV loss
    ax_b.plot(cncv_train, alpha=0.15, lw=0.3, c="#4C72B0")
    cncv_smooth = smooth(cncv_train)
    ax_b.plot(np.arange(49, 49 + len(cncv_smooth)), cncv_smooth,
              lw=1.0, c="#4C72B0", label="Train")

    batches_per_epoch_cv = len(cncv_train) / len(cncv_val)
    val_iters_cv = np.arange(len(cncv_val)) * batches_per_epoch_cv
    ax_b.plot(val_iters_cv, cncv_val, lw=1.0, c="#DD8452", label="Val")
    ax_b.set_xlabel("Batch iteration")
    ax_b.set_ylabel("MSE")
    ax_b.set_title("(b) CNCV", fontsize=8, pad=3)
    ax_b.legend(fontsize=6, loc="upper right")
    ax_b.tick_params(labelsize=7)

    # (c) Score norms (if available)
    cnf_score_norms = cnf_ckpt.get("score_norms", [])
    cncv_score_norms = cncv_ckpt.get("score_norms", [])

    for suffix in ("pdf", "png"):
        fig.savefig(os.path.join(args.output_dir,
                                 f"darcy_loss_curves.{suffix}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_dir}/darcy_loss_curves.pdf")

    # Score norms figure (if available)
    if cnf_score_norms or cncv_score_norms:
        n_panels = sum([bool(cnf_score_norms), bool(cncv_score_norms)])
        fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels / 2, 1.4))
        if n_panels == 1:
            axes = [axes]
        idx = 0
        if cnf_score_norms:
            axes[idx].plot(cnf_score_norms, lw=1.0)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Mean score norm")
            axes[idx].set_title("CNF score norms", fontsize=8)
            axes[idx].tick_params(labelsize=7)
            idx += 1
        if cncv_score_norms:
            axes[idx].plot(cncv_score_norms, lw=1.0)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel("Mean score norm")
            axes[idx].set_title("CNCV score norms", fontsize=8)
            axes[idx].tick_params(labelsize=7)

        plt.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(os.path.join(args.output_dir,
                                     f"darcy_score_norms.{suffix}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {args.output_dir}/darcy_score_norms.pdf")

    # ================================================================
    print(f"\n{'='*60}")
    print("All figures saved to:", args.output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
