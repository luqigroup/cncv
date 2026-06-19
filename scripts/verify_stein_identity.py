"""Empirical verification of Stein's identity for CNCV.

For each test observation y, we draw N posterior samples, compute the
dimension-averaged control variate mean:
    bar_g(y) = (1/N) sum_i (1/d) sum_j g_j(x_i, y),
which should be approximately zero by Stein's identity. We repeat for
many observations and plot a histogram of bar_g values.

Produces a 2x2 panel:
  (a) Gaussian d=4    (analytical score, exact posterior)
  (b) Rosenbrock d=2  (CNF score, CNF posterior)
  (c) Nonlinear d=4   (CNF score, CNF posterior)
  (d) Darcy d=100     (CNF score, CNF posterior)

Usage:
    python scripts/verify_stein_identity.py --gpu 0
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv.utils.download import ensure_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from cncv.utils.gaussian import compute_score_posterior_gaussian
from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import (
    create_shared_ensemble, SharedEnsembleCVWrapper,
)
from cncv.models.ensemble import create_shared_hierarchical_ensemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cnf_from_dir(cnf_dir, device):
    """Load a CNF from a checkpoint directory."""
    files = [f for f in os.listdir(cnf_dir) if f.endswith(".pth")]
    latest = max(files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    path = os.path.join(cnf_dir, latest)
    print(f"  CNF: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Config may be in top-level keys or in ckpt["args"]
    args = ckpt.get("args", None)
    def _get(key):
        if key in ckpt:
            return ckpt[key]
        return getattr(args, key)
    n_in = _get("input_dim")
    cnf = ConditionalNF(
        n_in=n_in,
        n_cond=n_in,
        n_hidden=_get("n_hidden"),
        n_flow_layers=_get("n_flow_layers"),
        depth=_get("depth"),
        n_mlp_layers=_get("n_mlp_layers"),
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)
    print(f"  CNF: epoch {ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params")
    return cnf


def load_ensemble_from_ckpt(ckpt_path, device):
    """Load shared_hierarchical ensemble from a CNCV checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    n_dim = ckpt.get("input_dim", getattr(args, "input_dim", None))
    n_cond = n_dim  # for stylized problems, n_cond = n_dim
    depth = getattr(args, "depth", None) or None
    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_cond,
        args.n_hidden,
        args.n_ensemble_members,
        n_layers=args.n_layers,
        depth=depth,
        n_cv=n_dim,
        seed=12,
    ).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    print(f"  Ensemble: L={args.n_ensemble_members}, "
          f"hidden={args.n_hidden}, depth={depth}")
    return ensemble, ckpt


# ---------------------------------------------------------------------------
# Gaussian experiment
# ---------------------------------------------------------------------------

def run_gaussian(device, n_obs=250, n_samples=5000, seed=42):
    """Verify Stein's identity on Gaussian d=4."""
    ckpt_dir = (
        "data/checkpoints/"
        "gaussian_ensemble_cv_input_dim-4_max_epochs-50_lr-0.001_"
        "lr_final-0.0001_sigma-0.3_n_ensemble_members-16_batchsize-2048_"
        "n_hidden-64_n_layers-3_num_train-65536_num_val-2048_"
        "num_samples-10000_save_freq-50_qofi-mean_dd"
        ".8e12cf8ef38e4051cf24e794ebe27b21a5d97a38"
    )
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_49.pth")
    print(f"\n=== Gaussian d=4 ===")

    ensemble, ckpt = load_ensemble_from_ckpt(ckpt_path, device)
    mu_prior = ckpt["mu_prior"].to(device)
    Sigma_prior = ckpt["Sigma_prior"].to(device)
    sigma = ckpt["sigma"]
    n_dim = ckpt.get("input_dim", 4)

    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    I = torch.eye(n_dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1 / sigma ** 2) * I)
    L_prior = torch.linalg.cholesky(Sigma_prior)
    L_post = torch.linalg.cholesky(Sigma_post)

    torch.manual_seed(seed)
    g_means = []

    for i in range(n_obs):
        x_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        y_obs = x_true + sigma * torch.randn(n_dim, device=device)
        mu_post = Sigma_post @ (
            Sigma_prior_inv @ mu_prior + (1 / sigma ** 2) * y_obs
        )
        X = mu_post + (L_post @ torch.randn(n_samples, n_dim,
                                             device=device).T).T
        Y = y_obs.unsqueeze(0).expand(n_samples, -1)

        with torch.no_grad():
            score = compute_score_posterior_gaussian(
                X, Y, mu_prior, Sigma_prior_inv, sigma
            )
            g_combined, _ = ensemble(X, Y, score)
        g_means.append(g_combined.mean().item())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_obs}")

    g_means = np.array(g_means)
    print(f"  mean={g_means.mean():.6f}, std={g_means.std():.6f}")
    return g_means


# ---------------------------------------------------------------------------
# CNF-score experiments (Rosenbrock, Nonlinear)
# ---------------------------------------------------------------------------

def run_cnf_score_experiment(name, cncv_ckpt_dir, cncv_file,
                             cnf_dir, n_dim, device,
                             n_obs=250, n_samples=5000, seed=42):
    """Verify Stein's identity using CNF score + CNF samples."""
    print(f"\n=== {name} d={n_dim} ===")

    cncv_path = os.path.join(cncv_ckpt_dir, cncv_file)
    ensemble, ckpt = load_ensemble_from_ckpt(cncv_path, device)
    cnf = load_cnf_from_dir(cnf_dir, device)

    # Generate test observations from prior predictive
    torch.manual_seed(seed)
    g_means = []

    for i in range(n_obs):
        z_base = torch.randn(1, n_dim, device=device)
        # Sample random observations in conditioning space (n_cond = n_dim),
        # then draw CNF posterior samples conditioned on each y.
        y_obs = torch.randn(n_dim, device=device)
        y_rep = y_obs.unsqueeze(0).expand(n_samples, -1)

        with torch.no_grad():
            z_samples = cnf.sample(y_obs.unsqueeze(0), n_samples)
        s = cnf.score(z_samples, y_rep)
        with torch.no_grad():
            g_combined, _ = ensemble(z_samples, y_rep, s)
        g_means.append(g_combined.mean().item())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_obs}")

    g_means = np.array(g_means)
    print(f"  mean={g_means.mean():.6f}, std={g_means.std():.6f}")
    return g_means


# ---------------------------------------------------------------------------
# Darcy experiment
# ---------------------------------------------------------------------------

def run_darcy(device, n_obs=250, n_samples=5000, seed=42):
    """Verify Stein's identity on Darcy d=100."""
    cncv_ckpt_path = "data/darcy/checkpoints_cncv_kl_v2b/cncv_best.pth"
    data_path = "data/darcy/darcy_K10_N131072_grid40.npz"

    print(f"\n=== Darcy d=100 ===")
    print(f"  CNCV: {cncv_ckpt_path}")
    cncv_ckpt = torch.load(cncv_ckpt_path, map_location=device,
                           weights_only=False)

    d_z = cncv_ckpt["d_z"]
    d_y = cncv_ckpt["d_y"]
    y_mean = cncv_ckpt["y_mean"].to(device)
    y_std = cncv_ckpt["y_std"].to(device)

    ensemble = create_shared_ensemble(
        n_in=d_z, n_cond=d_y,
        n_hidden=cncv_ckpt["cncv_hidden"],
        n_ensemble=cncv_ckpt["n_ensemble"],
        depth=cncv_ckpt["cncv_depth"],
        n_mlp_layers=cncv_ckpt["cncv_mlp_layers"],
        seed=12,
    )
    ensemble = SharedEnsembleCVWrapper(ensemble).to(device)
    ensemble.load_state_dict(cncv_ckpt["ensemble_state_dict"])
    ensemble.eval()
    print(f"  Ensemble: L={cncv_ckpt['n_ensemble']}, "
          f"hidden={cncv_ckpt['cncv_hidden']}")

    cnf_ckpt_path = cncv_ckpt["cnf_checkpoint"]
    print(f"  CNF: {cnf_ckpt_path}")
    cnf_ckpt = torch.load(cnf_ckpt_path, map_location=device,
                          weights_only=False)
    cnf = ConditionalNF(
        n_in=cnf_ckpt["d_z"], n_cond=cnf_ckpt["d_y"],
        n_hidden=cnf_ckpt["cnf_hidden"],
        n_flow_layers=cnf_ckpt["cnf_flow_layers"],
        depth=cnf_ckpt.get("cnf_depth", None),
        n_mlp_layers=cnf_ckpt["cnf_mlp_layers"],
    ).to(device)
    cnf.load_state_dict(cnf_ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    print(f"  Loading data: {data_path}")
    ensure_dataset(data_path)
    data = np.load(data_path)
    y_all = torch.tensor(data["y_obs"], dtype=torch.float32)
    y_test = y_all[-n_obs:]

    torch.manual_seed(seed)
    g_means = []
    chunk = 2000

    for i in range(n_obs):
        y_obs = y_test[i].to(device)
        y_obs_n = (y_obs - y_mean) / y_std

        g_sum = 0.0
        done = 0
        while done < n_samples:
            bs = min(chunk, n_samples - done)
            with torch.no_grad():
                z = cnf.sample(y_obs_n.unsqueeze(0), bs)
            yr = y_obs_n.unsqueeze(0).expand(bs, -1)
            s = cnf.score(z, yr)
            with torch.no_grad():
                g_combined, _ = ensemble(z, yr, s)
            g_sum += g_combined.sum().item()
            done += bs

        g_bar = g_sum / (n_samples * d_z)
        g_means.append(g_bar)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_obs}")

    g_means = np.array(g_means)
    print(f"  mean={g_means.mean():.6f}, std={g_means.std():.6f}")
    return g_means


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histograms(results, output_dir):
    """2x2 panel of histograms with KDE overlays."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.8))
    fig.subplots_adjust(wspace=0.35, hspace=0.55, left=0.10, right=0.97,
                        top=0.95, bottom=0.08)

    titles = ["Gaussian", "Rosenbrock", "Nonlinear", "Darcy flow"]

    for ax, (name, data), title in zip(axes.ravel(), results, titles):
        ax.hist(data, bins=20, density=True, color="steelblue",
                edgecolor="white", linewidth=0.5, alpha=0.5)
        # KDE overlay
        kde = gaussian_kde(data)
        rng = data.max() - data.min()
        xs = np.linspace(data.min() - 0.2 * rng,
                         data.max() + 0.2 * rng, 300)
        ax.plot(xs, kde(xs), color="steelblue", linewidth=1.5)
        ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
        ax.axvline(data.mean(), color="crimson", linewidth=1.0,
                   linestyle="-", label=f"mean = {data.mean():.4f}")
        ax.set_xlabel(
            r"$\frac{1}{Nd}\sum_{i,j} g_j(\mathbf{x}_i, \mathbf{y})$"
        )
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    for suffix in ("pdf", "png"):
        out = os.path.join(output_dir,
                           f"stein_identity_verification.{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {os.path.join(output_dir, 'stein_identity_verification.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_obs", type=int, default=250)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str,
                        default="figs/stein_verification")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # 1. Gaussian
    g = run_gaussian(device, args.n_obs, args.n_samples, args.seed)
    results.append(("Gaussian", g))

    # 2. Rosenbrock (CNF score)
    g = run_cnf_score_experiment(
        "Rosenbrock",
        "data/checkpoints/rosenbrock_cnf_cv_input_dim-2_max_epochs-200_"
        "lr-0.001_lr_final-0.0001_mu-0.0_a-0.5_b-1.0_sigma-0.3_"
        "n_ensemble_members-16_batchsize-4096_n_hidden-64_n_layers-3_"
        "num_train-65536_num_val-2048_num_samples-10000_save_fre"
        ".1b96f5bd140924ef1a0b8bb09481278bdedcde61",
        "checkpoint_199.pth",
        "data/checkpoints/cnf_rosenbrock_input_dim-2_n_hidden-128_"
        "n_mlp_layers-3_n_flow_layers-12_depth-2_max_epochs-1000_"
        "lr-0.001_lr_final-1e-05_batchsize-4096_num_train-262144_"
        "num_val-8192_num_samples-50000_mu-0.0_a-0.5_b-1.0_sigma-0.3_"
        "save_freq-999",
        n_dim=2, device=device,
        n_obs=args.n_obs, n_samples=args.n_samples, seed=args.seed,
    )
    results.append(("Rosenbrock", g))

    # 3. Nonlinear (CNF score)
    g = run_cnf_score_experiment(
        "Nonlinear",
        "data/checkpoints/nonlinear_cnf_cv_input_dim-4_max_epochs-200_"
        "lr-0.001_lr_final-0.0001_sigma-0.3_cond_number-2.0_"
        "n_ensemble_members-16_batchsize-8192_n_hidden-64_n_layers-3_"
        "num_train-65536_num_val-2048_num_samples-10000_save_freq-19"
        ".0c17b4e937ffc64796337e472e8e089915d67a3b",
        "checkpoint_199.pth",
        "data/checkpoints/cnf_nonlinear_input_dim-4_n_hidden-64_"
        "n_mlp_layers-3_n_flow_layers-8_depth-2_max_epochs-1000_"
        "lr-0.001_lr_final-1e-05_batchsize-8192_num_train-262144_"
        "num_val-8192_num_samples-50000_sigma-0.3_cond_number-2.0_"
        "save_freq-999",
        n_dim=4, device=device,
        n_obs=args.n_obs, n_samples=args.n_samples, seed=args.seed,
    )
    results.append(("Nonlinear", g))

    # 4. Darcy
    g = run_darcy(device, args.n_obs, args.n_samples, args.seed)
    results.append(("Darcy", g))

    plot_histograms(results, args.output_dir)


if __name__ == "__main__":
    main()
