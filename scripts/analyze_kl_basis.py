"""Analyze KL basis expansion for Darcy flow prior.

Follows Beskos et al. 2017 (Section 4.1-4.2):
  - Covariance operator: sigma^2 * (alpha*I - Delta)^{-s}
  - Eigenfunctions: phi_i(x) = 2*cos(pi*(i1+0.5)*x1)*cos(pi*(i2+0.5)*x2)
  - Eigenvalues:  lambda_i^2 = sigma^2 * (alpha + pi^2*((i1+0.5)^2+(i2+0.5)^2))^{-s}
  - Parameters: alpha=0, s=1.1, sigma=1

Produces:
  1. Eigenvalue spectrum (sorted by magnitude)
  2. Cumulative energy fraction vs number of modes
  3. Reconstruction error vs K for sample fields
  4. Gallery of truncated fields at different K values

Usage:
    python scripts/analyze_kl_basis.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cncv.utils.darcy import DarcyKLPrior


def compute_eigenvalues_sorted(K_max, alpha=0.0, s=1.1, sigma=1.0):
    """Compute all eigenvalues for i1,i2 in 0..K_max-1, sorted descending."""
    i1 = np.arange(K_max)
    i2 = np.arange(K_max)
    i1g, i2g = np.meshgrid(i1, i2, indexing="ij")
    i1_flat = i1g.ravel()
    i2_flat = i2g.ravel()

    arg = alpha + np.pi**2 * ((i1_flat + 0.5)**2 + (i2_flat + 0.5)**2)
    eigenvalues = sigma * arg ** (-s / 2)

    # Sort descending
    order = np.argsort(-eigenvalues)
    return eigenvalues[order], i1_flat[order], i2_flat[order]


def build_basis_matrix(K_max, grid_size, alpha=0.0, s=1.1, sigma=1.0):
    """Build full basis: eigenvalues (sorted) and eigenfunctions on grid.

    Returns:
        eigenvalues: (K_max^2,) sorted descending
        phi: (K_max^2, grid_size, grid_size) basis functions (sorted)
        order: index mapping from sorted to (i1,i2) ordering
    """
    i1 = np.arange(K_max)
    i2 = np.arange(K_max)
    i1g, i2g = np.meshgrid(i1, i2, indexing="ij")
    i1_flat = i1g.ravel()
    i2_flat = i2g.ravel()
    d = K_max ** 2

    # Eigenvalues
    arg = alpha + np.pi**2 * ((i1_flat + 0.5)**2 + (i2_flat + 0.5)**2)
    eigenvalues = sigma * arg ** (-s / 2)

    # Sort descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    i1_sorted = i1_flat[order]
    i2_sorted = i2_flat[order]

    # Grid points
    x = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size
    x1g, x2g = np.meshgrid(x, x, indexing="ij")

    # Eigenfunctions
    phi = np.zeros((d, grid_size, grid_size))
    for k in range(d):
        phi[k] = 2.0 * (
            np.cos(np.pi * (i1_sorted[k] + 0.5) * x1g)
            * np.cos(np.pi * (i2_sorted[k] + 0.5) * x2g)
        )

    return eigenvalues, phi, order


def main():
    grid_size = 40
    K_max = 20  # Up to 400 modes
    seed = 42
    n_samples = 10  # For reconstruction error analysis

    out_dir = "figs/darcy_kl"
    os.makedirs(out_dir, exist_ok=True)

    print("=== KL Basis Analysis for Darcy Flow Prior ===")
    print(f"  Beskos parameters: alpha=0, s=1.1, sigma=1")
    print(f"  Grid: {grid_size}x{grid_size}, K_max={K_max} ({K_max**2} modes)")
    print()

    # --- Eigenvalue spectrum ---
    print("Computing eigenvalue spectrum...")
    eigenvalues, phi, order = build_basis_matrix(K_max, grid_size)
    d_total = len(eigenvalues)
    print(f"  {d_total} modes, lambda range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")

    # Cumulative energy = sum(lambda_i^2) for first k modes / total
    lambda_sq = eigenvalues ** 2
    cum_energy = np.cumsum(lambda_sq) / np.sum(lambda_sq)

    # --- Generate "truth" samples from full expansion ---
    print(f"\nGenerating {n_samples} truth samples (K_max={K_max})...")
    rng = np.random.default_rng(seed)
    z_full = rng.standard_normal((n_samples, d_total)).astype(np.float32)

    # Full reconstruction: u = sum z_k * lambda_k * phi_k
    weighted = z_full * eigenvalues[None, :]  # (n_samples, d_total)
    u_full = np.einsum("ni,ijk->njk", weighted, phi)  # (n_samples, 40, 40)
    print(f"  Field range: [{u_full.min():.3f}, {u_full.max():.3f}]")

    # --- Reconstruction error vs number of modes ---
    print("\nComputing reconstruction error vs K...")
    K_values = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400]
    # Also add some intermediate values
    K_values = sorted(set(K_values))

    rel_errors = []
    abs_errors = []
    for K in K_values:
        weighted_trunc = z_full[:, :K] * eigenvalues[None, :K]
        u_trunc = np.einsum("ni,ijk->njk", weighted_trunc, phi[:K])
        residual = u_full - u_trunc

        # Relative L2 error (per sample, then average)
        rel_err = np.sqrt(np.mean(residual**2, axis=(1, 2))) / np.sqrt(np.mean(u_full**2, axis=(1, 2)))
        abs_err = np.sqrt(np.mean(residual**2, axis=(1, 2)))
        rel_errors.append(rel_err.mean())
        abs_errors.append(abs_err.mean())

    rel_errors = np.array(rel_errors)
    abs_errors = np.array(abs_errors)

    # Print table
    print(f"\n{'K':>5s}  {'Modes':>5s}  {'RelErr':>10s}  {'AbsErr':>10s}  {'CumEnergy':>10s}")
    print("-" * 50)
    for i, K in enumerate(K_values):
        ce = cum_energy[K - 1] if K <= d_total else 1.0
        print(f"{int(np.sqrt(K)) if int(np.sqrt(K))**2 == K else '':>5}  {K:>5d}  {rel_errors[i]:>10.6f}  {abs_errors[i]:>10.6f}  {ce:>10.8f}")

    # --- Figure 1: Eigenvalue spectrum ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(np.arange(1, d_total + 1), eigenvalues, "b-", lw=1)
    ax.set_xlabel("Mode index (sorted)")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title("Eigenvalue spectrum")
    ax.grid(True, alpha=0.3)
    # Mark K^2 boundaries
    for K in [5, 10, 15]:
        K2 = K ** 2
        if K2 <= d_total:
            ax.axvline(K2, color="r", ls="--", alpha=0.5, lw=0.8)
            ax.text(K2 + 2, eigenvalues[0] * 0.5, f"K={K}", fontsize=8, color="r")

    ax = axes[1]
    ax.plot(np.arange(1, d_total + 1), cum_energy, "b-", lw=1.5)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title(r"$\sum_{i=1}^{k} \lambda_i^2 \;/\; \sum_{i=1}^{\infty} \lambda_i^2$")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.001)
    for K in [5, 10, 15]:
        K2 = K ** 2
        if K2 <= d_total:
            ax.axvline(K2, color="r", ls="--", alpha=0.5, lw=0.8)
            ce = cum_energy[K2 - 1]
            ax.text(K2 + 2, ce - 0.005, f"K={K}: {ce:.4f}", fontsize=8, color="r")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out_dir}/eigenvalue_spectrum.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved eigenvalue_spectrum")

    # --- Figure 2: Reconstruction error vs modes ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(K_values, rel_errors, "bo-", ms=4, lw=1.5)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Reconstruction error vs truncation")
    ax.grid(True, alpha=0.3)
    for K in [25, 100]:
        idx = K_values.index(K) if K in K_values else None
        if idx is not None:
            ax.annotate(f"K²={K}\n({rel_errors[idx]:.4f})",
                       (K, rel_errors[idx]),
                       textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax = axes[1]
    ax.semilogy(K_values, abs_errors, "ro-", ms=4, lw=1.5)
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Absolute L2 error (RMSE)")
    ax.set_title("Absolute reconstruction error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out_dir}/reconstruction_error.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved reconstruction_error")

    # --- Figure 3: Gallery of truncated fields ---
    # Show 1 sample at different truncation levels
    sample_idx = 0
    K_show = [4, 9, 25, 64, 100, 196, 400]
    K_show = [k for k in K_show if k <= d_total]

    n_cols = len(K_show) + 1  # +1 for "truth"
    fig, axes = plt.subplots(2, n_cols, figsize=(2.5 * n_cols, 5))

    # Determine common colorbar range
    vmin, vmax = u_full[sample_idx].min(), u_full[sample_idx].max()

    # Truth
    im = axes[0, 0].imshow(u_full[sample_idx].T, origin="lower", extent=[0, 1, 0, 1],
                            vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0, 0].set_title(f"Truth\n({d_total} modes)", fontsize=9)
    axes[1, 0].axis("off")  # No residual for truth

    for j, K in enumerate(K_show):
        weighted_trunc = z_full[sample_idx:sample_idx+1, :K] * eigenvalues[None, :K]
        u_trunc = np.einsum("ni,ijk->njk", weighted_trunc, phi[:K])[0]
        residual = u_full[sample_idx] - u_trunc

        axes[0, j + 1].imshow(u_trunc.T, origin="lower", extent=[0, 1, 0, 1],
                               vmin=vmin, vmax=vmax, cmap="RdBu_r")
        sqrt_K = int(np.sqrt(K))
        label = f"K={sqrt_K}" if sqrt_K**2 == K else f"{K} modes"
        axes[0, j + 1].set_title(f"{label}\n({K} modes)", fontsize=9)

        # Residual
        res_max = max(abs(residual.min()), abs(residual.max()))
        if res_max < 1e-10:
            res_max = 1.0
        axes[1, j + 1].imshow(residual.T, origin="lower", extent=[0, 1, 0, 1],
                               vmin=-res_max, vmax=res_max, cmap="RdBu_r")
        rmse = np.sqrt(np.mean(residual**2))
        axes[1, j + 1].set_title(f"Residual\nRMSE={rmse:.4f}", fontsize=8)

    # Labels
    axes[0, 0].set_ylabel("Truncated field")
    axes[1, 0].set_ylabel("Residual")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("KL expansion truncation: Beskos prior (α=0, s=1.1, σ=1)", fontsize=11, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out_dir}/truncation_gallery.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved truncation_gallery")

    # --- Figure 4: Multiple samples at K=25 (Beskos default) and K=100 ---
    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    n_show = 5

    for row, (K, label) in enumerate([(400, f"Truth ({d_total} modes)"),
                                       (25, "K=5 (25 modes)"),
                                       (100, "K=10 (100 modes)")]):
        for col in range(n_show):
            weighted_trunc = z_full[col:col+1, :K] * eigenvalues[None, :K]
            u_trunc = np.einsum("ni,ijk->njk", weighted_trunc, phi[:K])[0]
            axes[row, col].imshow(u_trunc.T, origin="lower", extent=[0, 1, 0, 1],
                                   cmap="RdBu_r")
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.suptitle("KL expansion: sample gallery", fontsize=12)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"{out_dir}/sample_gallery.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sample_gallery")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary: recommended truncation levels")
    print("=" * 60)
    for K2 in [25, 49, 64, 100, 144, 196]:
        if K2 in K_values:
            idx = K_values.index(K2)
            ce = cum_energy[K2 - 1]
            print(f"  {K2:>4d} modes: RelErr={rel_errors[idx]:.6f}, CumEnergy={ce:.8f}")

    print(f"\nBeskos et al. use |I0|=100 modes (K=10) for MCMC")
    print(f"  with splitting at D0=25 modes (K=5)")
    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
