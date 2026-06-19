"""Per-observation baseline comparison for CNCV rebuttal.

Compares against:
  - Polynomial-1 Stein CV (linear, Mira et al. style)
  - Polynomial-2 Stein CV (quadratic)
  - Control Functionals (Oates et al., 2017) — kernel-based, with train-test split

Reports VRF AND wall-clock cost per test observation, then computes break-even
N_obs at which CNCV's amortized cost wins.

Output:
  figs/baselines/per_obs_baseline_comparison.pdf

Usage:
  python scripts/run_per_obs_baseline_comparison.py --gpu 0 --dim 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cncv import compute_score_posterior_gaussian
from cncv.models import create_shared_hierarchical_ensemble

# Reuse polynomial / CF implementations
from run_baselines import (
    polynomial_cv,
    control_functionals_cv,
)


# ---------------------------------------------------------------------------
# CNCV checkpoint loading (matches paper's shared_hierarchical)
# ---------------------------------------------------------------------------

def find_paper_cncv_checkpoint(dim, qofi="mean"):
    """Find the paper's shared_hierarchical CNCV checkpoint."""
    import re
    base = "data/checkpoints"
    pattern = f"gaussian_ensemble_cv_input_dim-{dim}_"
    all_dirs = [d for d in os.listdir(base) if d.startswith(pattern)
                and f"qofi-{qofi}" in d]
    # Paper's checkpoint is in a hashed dir (projorg shortens long names)
    # Prefer dirs with sha1 hash AND L=16 (paper's setting)
    matches = [d for d in all_dirs if re.search(r"\.[0-9a-f]{40}$", d)
               and "n_ensemble_members-16" in d
               and "max_epochs-50" in d
               and "lr_final-0.0001" in d
               and "n_hidden-64" in d]
    if not matches:
        # Fallback: any shared_hierarchical
        matches = [d for d in all_dirs if "n_ensemble_members-16" in d]
    if not matches:
        raise FileNotFoundError(f"No paper CNCV ckpt for dim={dim}")
    chosen = sorted(matches)[-1]
    ckpt_dir = os.path.join(base, chosen)
    ckpt_files = [f for f in os.listdir(ckpt_dir)
                  if f.startswith("checkpoint_") and f.endswith(".pth")]
    latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckpt_dir, latest)


def load_paper_cncv(dim, device, qofi="mean"):
    """Load paper's CNCV (shared_hierarchical, L=16, depth=2, h=64, n_layers=3)."""
    ckpt_path = find_paper_cncv_checkpoint(dim, qofi)
    print(f"  Paper CNCV ckpt: {os.path.basename(os.path.dirname(ckpt_path))}/{os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_ckpt = ckpt["args"]

    ensemble = create_shared_hierarchical_ensemble(
        dim, dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
        n_layers=args_ckpt.n_layers,
        depth=getattr(args_ckpt, "depth", None) or None,
        n_cv=dim, seed=12,
    ).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

    n_params = sum(p.numel() for p in ensemble.parameters())
    print(f"  Paper CNCV params: {n_params:,}")
    return ensemble, args_ckpt, n_params


# ---------------------------------------------------------------------------
# Problem setup (matches gaussian_ensemble_cv.py for paper Table 1)
# ---------------------------------------------------------------------------

def gaussian_setup(dim, sigma, device):
    mu_prior = torch.zeros(dim, dtype=torch.float32, device=device)
    torch.manual_seed(42)
    diag_values = torch.linspace(1.0, 2.0, dim)
    D = torch.diag(diag_values).to(device=device, dtype=torch.float32)
    if dim > 1:
        A = torch.randn(dim, dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        Sigma_prior = Q @ D @ Q.T
    else:
        Sigma_prior = D
    L_prior = torch.linalg.cholesky(Sigma_prior)
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    I = torch.eye(dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1.0 / sigma ** 2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)
    return mu_prior, Sigma_prior, Sigma_prior_inv, L_prior, Sigma_post, L_post


def vrf_per_component(h_x, g_values):
    """Per-component VRF averaged across components."""
    var_h = h_x.var(dim=0)
    var_hg = (h_x - g_values).var(dim=0)
    return (var_hg / var_h.clamp(min=1e-12)).mean().item()


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--qofi", type=str, default="mean")
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Posterior samples per obs (capped at 2000 for CF O(N^2))")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str,
                        default="figs/baselines")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup problem
    mu_prior, Sigma_prior, Sigma_prior_inv, L_prior, Sigma_post, L_post = \
        gaussian_setup(args.dim, args.sigma, device)

    # Load paper CNCV
    print("\nLoading paper CNCV (shared_hierarchical)")
    cncv, _, cncv_params = load_paper_cncv(args.dim, device, args.qofi)

    # Test observations
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = ["cncv", "poly1", "poly2", "cf"]
    vrfs = {m: [] for m in methods}
    timings = {m: [] for m in methods}  # per-obs wall time (excluding sample/score gen)

    for obs_idx in tqdm(range(args.n_test_obs), desc="Test obs"):
        X_true = mu_prior + L_prior @ torch.randn(args.dim, device=device)
        Y_obs = X_true + args.sigma * torch.randn(args.dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / args.sigma ** 2) * Y_obs)

        # Sample posterior (analytical)
        noise = torch.randn(args.num_samples, args.dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(args.num_samples, 1)
        score = compute_score_posterior_gaussian(
            X_samples, Y_samples, mu_prior, Sigma_prior_inv, args.sigma
        )

        h_x = X_samples  # qofi=mean

        # Time each method's CV computation (per-obs)
        # CNCV: just forward pass (model already trained)
        if cncv.parameters:
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            with torch.no_grad():
                g_combined, _ = cncv(X_samples, Y_samples, score)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            g_cncv = g_combined.T
            timings["cncv"].append(t1 - t0)
            vrfs["cncv"].append(vrf_per_component(h_x, g_cncv))

        # Polynomial-1 (per-obs LSQ)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        g_poly1 = polynomial_cv(X_samples, score, h_x, degree=1)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        timings["poly1"].append(t1 - t0)
        vrfs["poly1"].append(vrf_per_component(h_x, g_poly1))

        # Polynomial-2 (per-obs LSQ)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        g_poly2 = polynomial_cv(X_samples, score, h_x, degree=2)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        timings["poly2"].append(t1 - t0)
        vrfs["poly2"].append(vrf_per_component(h_x, g_poly2))

        # Control Functionals (kernel, O(N^2) — train-test split)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        h_cf, g_cf = control_functionals_cv(X_samples, score, h_x,
                                             reg_lambda=1e-4, max_n=2000)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        timings["cf"].append(t1 - t0)
        vrfs["cf"].append(vrf_per_component(h_cf, g_cf))

    # Aggregate
    print("\n=== Per-method results (Gaussian d={}) ===".format(args.dim))
    print(f"{'Method':<10} {'VRF mean':>12} {'VRF std':>10} {'time/obs (ms)':>15}")
    for m in methods:
        vmean = np.mean(vrfs[m])
        vstd = np.std(vrfs[m])
        tmean = np.mean(timings[m]) * 1000  # ms
        print(f"{m:<10} {vmean:>12.4f} {vstd:>10.4f} {tmean:>15.3f}")

    # CNCV training cost (one-time fixed)
    # Paper Table 3: 50 epochs, 65536 train samples, RTX 2000 Ada — ~5 min for stylized
    cncv_train_cost = 5 * 60.0  # seconds

    # Break-even analysis: total cost = train_cost + N_obs * inference_cost
    # CNCV: 5 min one-shot + per-obs forward (negligible)
    # Per-obs methods: 0 train + N_obs * per-obs cost
    cncv_per_obs = np.mean(timings["cncv"])
    poly1_per_obs = np.mean(timings["poly1"])
    poly2_per_obs = np.mean(timings["poly2"])
    cf_per_obs = np.mean(timings["cf"])

    # Break-even: cncv_train + n*cncv_per_obs = n*per_obs_cost
    # n = cncv_train / (per_obs_cost - cncv_per_obs)
    breakeven = {}
    for m in ["poly1", "poly2", "cf"]:
        per_obs = np.mean(timings[m])
        if per_obs > cncv_per_obs:
            breakeven[m] = cncv_train_cost / (per_obs - cncv_per_obs)
        else:
            breakeven[m] = float("inf")

    print(f"\n=== Wall-clock break-even ===")
    print(f"CNCV training: {cncv_train_cost} s (one-shot, paper §E.4)")
    print(f"CNCV per-obs inference: {cncv_per_obs*1000:.2f} ms")
    for m in ["poly1", "poly2", "cf"]:
        print(f"{m} per-obs refit: {np.mean(timings[m])*1000:.2f} ms")
        print(f"  break-even N_obs: {breakeven[m]:.0f}")

    # Save results
    save = {
        "vrfs": {m: vrfs[m] for m in methods},
        "timings": {m: timings[m] for m in methods},
        "vrf_mean": {m: float(np.mean(vrfs[m])) for m in methods},
        "vrf_std": {m: float(np.std(vrfs[m])) for m in methods},
        "time_mean_ms": {m: float(np.mean(timings[m]) * 1000) for m in methods},
        "cncv_train_cost_s": cncv_train_cost,
        "breakeven_n_obs": breakeven,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, f"per_obs_results_d{args.dim}.json"), "w") as f:
        json.dump(save, f, indent=2)

    # ── Figure ──────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))

    # Panel A: VRF comparison
    ax = axes[0]
    method_labels = {"cncv": "CNCV (ours)", "poly1": "Poly-1", "poly2": "Poly-2", "cf": "CF"}
    method_colors = {"cncv": "#4C72B0", "poly1": "#DD8452", "poly2": "#55A868", "cf": "#C44E52"}
    means = [np.mean(vrfs[m]) for m in methods]
    stds = [np.std(vrfs[m]) for m in methods]
    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=4,
                  color=[method_colors[m] for m in methods], alpha=0.85)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.6, label="VRF=1 (no reduction)")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_labels[m] for m in methods], rotation=10)
    ax.set_ylabel("VRF (lower is better)")
    ax.set_title(f"(a) VRF, Gaussian d={args.dim}")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2, axis="y", which="both")

    # Panel B: per-obs time
    ax = axes[1]
    times_ms = [np.mean(timings[m]) * 1000 for m in methods]
    ax.bar(range(len(methods)), times_ms,
           color=[method_colors[m] for m in methods], alpha=0.85)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_labels[m] for m in methods], rotation=10)
    ax.set_ylabel("Per-obs time (ms)")
    ax.set_yscale("log")
    ax.set_title("(b) Per-observation cost")
    ax.grid(alpha=0.2, axis="y", which="both")

    # Panel C: per-obs cost × K (excluding pre-training)
    ax = axes[2]
    n_obs_axis = np.logspace(0, 4, 100)
    for m in methods:
        per_obs = np.mean(timings[m])
        ax.plot(n_obs_axis, n_obs_axis * per_obs, color=method_colors[m], lw=1.8,
                label=f"{method_labels[m]}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of test observations $K$")
    ax.set_ylabel("Total per-obs cost (s)")
    ax.set_title("(c) Per-obs cost × $K$ (no pre-training)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2, which="both")

    # Panel D: total cost INCLUDING pre-training, with break-even points
    ax = axes[3]
    cncv_total = cncv_train_cost + n_obs_axis * cncv_per_obs
    ax.plot(n_obs_axis, cncv_total, color=method_colors["cncv"], lw=2,
            label=f"CNCV (offline + amortized)")
    for m in ["poly1", "poly2", "cf"]:
        per_obs = np.mean(timings[m])
        total = n_obs_axis * per_obs
        ax.plot(n_obs_axis, total, color=method_colors[m], lw=1.5, ls="--",
                label=f"{method_labels[m]} (per-obs refit)")
        if breakeven[m] < float("inf"):
            ax.axvline(breakeven[m], color=method_colors[m], ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of test observations $K$")
    ax.set_ylabel("Total wall-clock cost (s)")
    ax.set_title("(d) Total cost incl. pre-training")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2, which="both")

    plt.tight_layout()
    out_pdf = os.path.join(args.output_dir, f"per_obs_baseline_comparison_d{args.dim}.pdf")
    plt.savefig(out_pdf)
    plt.savefig(out_pdf.replace(".pdf", ".png"))
    plt.close()
    print(f"\nSaved: {out_pdf}")


if __name__ == "__main__":
    main()
