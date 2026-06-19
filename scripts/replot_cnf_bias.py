"""Re-render the Darcy CNF posterior-mean bias figure with paper-grade fonts.

Reads the cached figs/darcy/cnf_bias_quantification.npz produced by
quantify_cnf_bias_darcy.py and re-renders cnf_bias_quantification.{pdf,png}
with larger fonts and thicker lines so the three panels stay legible when the
figure is placed full-width in the paper. Pure replot from cached arrays --
no model, CNF, or Devito dependency.

Usage:
    python scripts/replot_cnf_bias.py \
        --npz figs/darcy/cnf_bias_quantification.npz \
        --out figs/darcy/cnf_bias_quantification.pdf
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Replot the CNF-bias figure")
    parser.add_argument("--npz", type=str,
                        default="figs/darcy/cnf_bias_quantification.npz")
    parser.add_argument("--out", type=str,
                        default="figs/darcy/cnf_bias_quantification.pdf")
    args = parser.parse_args()

    d = np.load(args.npz)
    rmse_kl_per_comp = d["rmse_kl_per_comp"]
    rmse_field_per_pixel = d["rmse_field_per_pixel"]
    biases_kl = d["biases_kl"]
    cnf_means_kl = d["cnf_means_kl"]
    truths_kl = d["truths_kl"]
    eigenvalues = d["eigenvalues"]
    d_z = int(rmse_kl_per_comp.shape[0])
    rmse_kl_overall = float(np.sqrt((biases_kl ** 2).mean()))
    rmse_field_overall = float(np.sqrt((rmse_field_per_pixel ** 2).mean()))

    # Larger base fonts than the original export (font.size 9 -> 12) so the
    # three panels remain readable once scaled to full text width.
    plt.rcParams.update({
        "font.size": 12, "font.family": "serif",
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "axes.linewidth": 0.8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # Panel (a): per-component KL RMSE sorted by prior eigenvalue.
    ax = axes[0]
    sort_idx = np.argsort(eigenvalues)[::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, d_z))
    ax.bar(range(d_z), rmse_kl_per_comp[sort_idx], color=colors, width=1.0,
           edgecolor="none", alpha=0.85)
    ax.set_xlabel("KL component (sorted by eigenvalue)")
    ax.set_ylabel("CNF posterior-mean RMSE")
    ax.set_title(f"(a) Per-component bias\nMean = {rmse_kl_overall:.3f}")
    ax2 = ax.twinx()
    ax2.plot(range(d_z), eigenvalues[sort_idx], color="C1", lw=2.0, alpha=0.8)
    ax2.set_ylabel("Prior eigenvalue", color="C1")
    ax2.tick_params(axis="y", colors="C1")

    # Panel (b): pixelwise field-space RMSE.
    ax = axes[1]
    im = ax.imshow(rmse_field_per_pixel, cmap="magma", origin="lower")
    ax.set_title(f"(b) Pixelwise field bias\nField RMSE = {rmse_field_overall:.3f}")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (c): CNF posterior mean vs truth on the leading KL components.
    ax = axes[2]
    for j in range(min(3, d_z)):
        ax.scatter(truths_kl[:, j], cnf_means_kl[:, j], alpha=0.7, s=32,
                   label=f"$j={j+1}$")
    lim = max(np.abs(truths_kl[:, :3]).max(),
              np.abs(cnf_means_kl[:, :3]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, lw=1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"True KL coefficient $z_j^\dagger$")
    ax.set_ylabel(r"CNF posterior mean $\hat{\mu}_j$")
    ax.set_title("(c) CNF mean vs truth (top-3 modes)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out)
    plt.savefig(args.out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
