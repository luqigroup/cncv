"""Quantify CNF posterior-mean bias on Darcy flow.

Measures the bias of the CNF posterior mean against ground truth, rather than
against the CNF sample mean.

For each held-out test observation y_obs with known truth z_true:
  1. Sample N posterior samples from CNF: z_samples ~ q_phi(z|y_obs)
  2. CNF posterior mean: hat_mu = E[z_samples]
  3. True KL coefficients: z_true (held out)
  4. Bias = hat_mu - z_true

Reports:
  - Per-component RMSE of CNF mean vs truth
  - Aggregate KL-coefficient bias norm
  - Pixel-wise field bias map (after KL reconstruction)

Output: figs/darcy/cnf_bias_quantification.pdf

Usage:
  python scripts/quantify_cnf_bias_darcy.py --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv.models.conditional_nf import ConditionalNF
from cncv.utils.darcy import DarcyKLPrior
from cncv.utils.download import ensure_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnf_checkpoint", type=str,
                        default="data/darcy/checkpoints_cnf_kl_v2b/cnf_best.pth")
    parser.add_argument("--data_path", type=str,
                        default="data/darcy/darcy_K10_N131072_grid40.npz")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_train", type=int, default=120000)
    parser.add_argument("--num_val", type=int, default=11072)
    parser.add_argument("--output_dir", type=str, default="figs/darcy")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading CNF: {args.cnf_checkpoint}")
    ckpt = torch.load(args.cnf_checkpoint, map_location=device, weights_only=False)
    d_z = ckpt["d_z"]
    d_y = ckpt["d_y"]
    y_mean = ckpt["y_mean"].cpu()
    y_std = ckpt["y_std"].cpu()
    cnf = ConditionalNF(
        n_in=d_z, n_cond=d_y,
        n_hidden=ckpt["cnf_hidden"],
        n_flow_layers=ckpt["cnf_flow_layers"],
        depth=ckpt.get("cnf_depth", None),
        n_mlp_layers=ckpt["cnf_mlp_layers"],
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)
    print(f"  CNF: epoch={ckpt['epoch']}, val_nll={ckpt['val_losses'][-1]:.3f}")

    # Test data: held-out portion after train+val
    print(f"\nLoading data: {args.data_path}")
    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    z_all = data["u_kl"]
    y_all = data["y_obs"]
    eigenvalues = data["eigenvalues"]

    # Held-out: prefer post-train+val region; fall back to val region
    test_start = args.num_train + args.num_val
    n_available = z_all.shape[0] - test_start
    if n_available < args.n_test_obs:
        # Use last n_test_obs val samples (CNF trained on training set only)
        test_start = z_all.shape[0] - args.n_test_obs
        print(f"  No post-val held-out; using last {args.n_test_obs} val samples")
    z_test = z_all[test_start:test_start + args.n_test_obs]
    y_test = y_all[test_start:test_start + args.n_test_obs]
    y_test_n = (y_test - y_mean.numpy()) / y_std.numpy()

    print(f"  Test: {args.n_test_obs} obs, indices {test_start}..{test_start+args.n_test_obs}")

    # KL prior for field reconstruction
    prior = DarcyKLPrior(K=int(np.sqrt(d_z)), grid_size=40)

    # Compute CNF posterior means + biases
    biases_kl = np.zeros((args.n_test_obs, d_z), dtype=np.float32)
    cnf_means_kl = np.zeros((args.n_test_obs, d_z), dtype=np.float32)
    truths_kl = np.zeros((args.n_test_obs, d_z), dtype=np.float32)
    field_biases = np.zeros((args.n_test_obs, 40, 40), dtype=np.float32)

    for i in tqdm(range(args.n_test_obs), desc="Bias eval"):
        y_obs = torch.tensor(y_test_n[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            z_samples = cnf.sample(y_obs.unsqueeze(0), args.num_samples)
        cnf_mean_kl = z_samples.mean(dim=0).cpu().numpy()
        truth_kl = z_test[i]

        cnf_means_kl[i] = cnf_mean_kl
        truths_kl[i] = truth_kl
        biases_kl[i] = cnf_mean_kl - truth_kl

        # Field-space bias
        cnf_mean_field = prior.reconstruct(cnf_mean_kl[None])[0]
        truth_field = prior.reconstruct(truth_kl[None])[0]
        field_biases[i] = cnf_mean_field - truth_field

    # Aggregate metrics
    rmse_kl_per_comp = np.sqrt((biases_kl ** 2).mean(axis=0))  # (d_z,)
    rmse_kl_overall = np.sqrt((biases_kl ** 2).mean())
    rmse_field_per_pixel = np.sqrt((field_biases ** 2).mean(axis=0))  # (40, 40)
    rmse_field_overall = np.sqrt((field_biases ** 2).mean())

    print(f"\n=== CNF posterior-mean bias on Darcy ===")
    print(f"  KL space:     RMSE = {rmse_kl_overall:.4f}")
    print(f"  Field space:  RMSE = {rmse_field_overall:.4f}")
    print(f"  Per-component KL RMSE: min={rmse_kl_per_comp.min():.4f}, "
          f"max={rmse_kl_per_comp.max():.4f}, mean={rmse_kl_per_comp.mean():.4f}")

    # Reference scales
    z_train_std = z_all[:args.num_train].std()
    field_std = prior.reconstruct(z_all[:1000])[0:1000].std() if z_all.shape[0] >= 1000 else 1.0
    print(f"  Reference: z train std = {z_train_std:.3f}, field std = {field_std:.3f}")
    print(f"  Relative bias (KL): {rmse_kl_overall / z_train_std * 100:.1f}% of prior std")

    # Save
    np.savez(os.path.join(args.output_dir, "cnf_bias_quantification.npz"),
             biases_kl=biases_kl,
             cnf_means_kl=cnf_means_kl,
             truths_kl=truths_kl,
             field_biases=field_biases,
             rmse_kl_per_comp=rmse_kl_per_comp,
             rmse_field_per_pixel=rmse_field_per_pixel,
             eigenvalues=eigenvalues)

    # ── Figure ──
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))

    # Panel A: per-component KL RMSE sorted by eigenvalue
    ax = axes[0]
    sort_idx = np.argsort(eigenvalues)[::-1]
    rmse_sorted = rmse_kl_per_comp[sort_idx]
    eig_sorted = eigenvalues[sort_idx]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, d_z))
    ax.bar(range(d_z), rmse_sorted, color=colors, width=1.0,
           edgecolor="none", alpha=0.85)
    ax.set_xlabel("KL component (sorted by eigenvalue)")
    ax.set_ylabel("CNF posterior-mean RMSE")
    ax.set_title(f"(a) Per-component bias\nMean = {rmse_kl_overall:.3f}")
    ax2 = ax.twinx()
    ax2.plot(range(d_z), eig_sorted, color="C1", lw=1.5, alpha=0.7)
    ax2.set_ylabel("Prior eigenvalue", color="C1")
    ax2.tick_params(axis="y", colors="C1")

    # Panel B: pixelwise field RMSE
    ax = axes[1]
    im = ax.imshow(rmse_field_per_pixel, cmap="magma", origin="lower")
    ax.set_title(f"(b) Pixelwise field bias\nField RMSE = {rmse_field_overall:.3f}")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel C: scatter CNF mean vs truth on first KL component
    ax = axes[2]
    for j in range(min(3, d_z)):
        ax.scatter(truths_kl[:, j], cnf_means_kl[:, j],
                   alpha=0.7, s=20, label=f"$j={j+1}$")
    lim = max(np.abs(truths_kl[:, :3]).max(), np.abs(cnf_means_kl[:, :3]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("True KL coefficient $z_j^\\dagger$")
    ax.set_ylabel("CNF posterior mean $\\hat{\\mu}_j$")
    ax.set_title("(c) CNF mean vs truth (top-3 modes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(args.output_dir, "cnf_bias_quantification.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
