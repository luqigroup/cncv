"""Generate combined sample efficiency figure for all three CNF experiments.

Produces a single 1x2 figure:
  (a) VRF vs sample size N -- mean across dimensions, one curve per problem
  (b) MSE vs sample size N -- raw and CV estimators, 1/N reference line

Usage:
    python scripts/generate_cnf_efficiency_combined.py --gpu 0
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

from scripts._cnf_helpers import load_cnf_and_ensemble, setup_problem
from projorg import checkpointsdir, setup_environment
from cncv.models.conditional_nf import ConditionalNF
from cncv.models import create_shared_hierarchical_ensemble, create_hierarchical_ensemble
from cncv.utils.rosenbrock import RosenbrockDistribution
from cncv.utils.nonlinear import setup_nonlinear_problem

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.8,
})

SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
PROBLEM_COLORS = {"Gaussian": "#1f77b4", "Rosenbrock": "#ff7f0e", "Nonlinear": "#2ca02c"}
PROBLEM_MARKERS = {"Gaussian": "o", "Rosenbrock": "s", "Nonlinear": "D"}


def load_gaussian_data(device, num_samples):
    """Load Gaussian CNF + ensemble, return h, g, exact_mean."""
    cnf, ensemble, mu_prior, Sigma_prior, sigma = load_cnf_and_ensemble(
        dim=4, device=str(device),
    )
    Y_obs, mu_post, Sigma_post = setup_problem(mu_prior, Sigma_prior, sigma, 4, device)

    with torch.no_grad():
        X_samples = cnf.sample(Y_obs, num_samples)
        Y_samples = Y_obs.repeat(num_samples, 1)
        score = cnf.score(X_samples, Y_samples)
        h_x = X_samples
        g_combined, _ = ensemble(X_samples, Y_samples, score)

    # Use CNF sample mean as ground truth (consistent with Rosenbrock/Nonlinear)
    exact_mean = X_samples.mean(dim=0).cpu().numpy()

    return h_x.cpu().numpy(), g_combined.cpu().numpy(), exact_mean, 4


def load_rosenbrock_data(device, num_samples):
    """Load Rosenbrock CNF + ensemble, return h, g, exact_mean."""
    cnf_args = setup_environment(
        "cnf_rosenbrock.json",
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed", "testing_epoch"],
    )
    cnf_epoch = cnf_args.max_epochs - 1 if cnf_args.testing_epoch == -1 else cnf_args.testing_epoch
    cnf_ckpt_path = os.path.join(
        checkpointsdir(cnf_args.experiment), f"checkpoint_{cnf_epoch}.pth",
    )
    cnf = ConditionalNF(
        n_in=cnf_args.input_dim, n_cond=cnf_args.input_dim,
        n_hidden=cnf_args.n_hidden, n_flow_layers=cnf_args.n_flow_layers,
        depth=cnf_args.depth, n_mlp_layers=cnf_args.n_mlp_layers,
    ).to(device)
    cnf_ckpt = torch.load(cnf_ckpt_path, map_location=device, weights_only=False)
    cnf.load_state_dict(cnf_ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    rosenbrock_dist = RosenbrockDistribution(
        mu=cnf_ckpt["mu"], a=cnf_ckpt["a"],
        b=cnf_ckpt["b"], n1=cnf_ckpt["n1"], n2=cnf_ckpt["n2"],
    )
    sigma = cnf_ckpt["sigma"]
    n_dim = rosenbrock_dist.ndim

    cv_args = setup_environment(
        "rosenbrock_cnf_cv.json",
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed", "testing_epoch"],
    )
    cv_epoch = cv_args.max_epochs - 1 if cv_args.testing_epoch == -1 else cv_args.testing_epoch
    cv_ckpt_path = os.path.join(
        checkpointsdir(cv_args.experiment), f"checkpoint_{cv_epoch}.pth",
    )
    cv_ckpt = torch.load(cv_ckpt_path, map_location=device, weights_only=False)
    train_args = cv_ckpt["args"]
    layer_type = cv_ckpt.get("layer_type", getattr(train_args, "layer_type", "shared_hierarchical"))
    depth = getattr(train_args, "depth", None) or None

    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim, train_args.n_hidden, train_args.n_ensemble_members,
            n_layers=train_args.n_layers, depth=depth, n_cv=n_dim, seed=1,
        ).to(device)
    else:
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim, train_args.n_hidden, train_args.n_ensemble_members,
            n_layers=train_args.n_layers, depth=depth, n_cv=n_dim, seed=1,
        ).to(device)
    ensemble.load_state_dict(cv_ckpt["ensemble_state_dict"])
    ensemble.eval()

    torch.manual_seed(42)
    X_true = rosenbrock_dist.sample(1, device=device).squeeze(0)
    Y_obs = X_true + sigma * torch.randn(n_dim, device=device)

    with torch.no_grad():
        X_samples = cnf.sample(Y_obs, num_samples)
        Y_samples = Y_obs.repeat(num_samples, 1)
        score = cnf.score(X_samples, Y_samples)
        h_x = X_samples
        g_combined, _ = ensemble(X_samples, Y_samples, score)

    # No closed-form mean for Rosenbrock -- use large-sample mean
    exact_mean = X_samples.mean(dim=0).cpu().numpy()

    return h_x.cpu().numpy(), g_combined.cpu().numpy(), exact_mean, n_dim


def load_nonlinear_data(device, num_samples):
    """Load Nonlinear CNF + ensemble, return h, g, exact_mean."""
    n_dim = 4
    sigma = 0.3
    cond_number = 2.0

    cnf_args = setup_environment(
        "cnf_nonlinear.json",
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed", "testing_epoch"],
    )
    cnf_epoch = cnf_args.max_epochs - 1 if cnf_args.testing_epoch == -1 else cnf_args.testing_epoch
    cnf_ckpt_path = os.path.join(
        checkpointsdir(cnf_args.experiment), f"checkpoint_{cnf_epoch}.pth",
    )
    cnf = ConditionalNF(
        n_in=n_dim, n_cond=n_dim,
        n_hidden=cnf_args.n_hidden, n_flow_layers=cnf_args.n_flow_layers,
        depth=cnf_args.depth, n_mlp_layers=cnf_args.n_mlp_layers,
    ).to(device)
    cnf_ckpt = torch.load(cnf_ckpt_path, map_location=device, weights_only=False)
    cnf.load_state_dict(cnf_ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)

    problem = setup_nonlinear_problem(n_dim, sigma=sigma, cond_number=cond_number, device=device)
    forward_model = problem["forward_model"]
    mu_prior = problem["mu_prior"]
    L_prior = problem["L_prior"]

    if "forward_model_A" in cnf_ckpt:
        forward_model.A = cnf_ckpt["forward_model_A"].to(device)

    cv_args = setup_environment(
        "nonlinear_cnf_cv.json",
        ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed", "testing_epoch"],
    )
    cv_epoch = cv_args.max_epochs - 1 if cv_args.testing_epoch == -1 else cv_args.testing_epoch
    cv_ckpt_path = os.path.join(
        checkpointsdir(cv_args.experiment), f"checkpoint_{cv_epoch}.pth",
    )
    cv_ckpt = torch.load(cv_ckpt_path, map_location=device, weights_only=False)
    train_args = cv_ckpt["args"]
    layer_type = cv_ckpt.get("layer_type", getattr(train_args, "layer_type", "shared_hierarchical"))
    depth = getattr(train_args, "depth", None) or None

    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim, train_args.n_hidden, train_args.n_ensemble_members,
            n_layers=train_args.n_layers, depth=depth, n_cv=n_dim, seed=12,
        ).to(device)
    else:
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim, train_args.n_hidden, train_args.n_ensemble_members,
            n_layers=train_args.n_layers, depth=depth, n_cv=n_dim, seed=12,
        ).to(device)
    ensemble.load_state_dict(cv_ckpt["ensemble_state_dict"])
    ensemble.eval()

    torch.manual_seed(42)
    noise = torch.randn(n_dim, device=device)
    X_true = mu_prior + L_prior @ noise
    Y_obs = forward_model.forward(X_true.unsqueeze(0)).squeeze(0) + sigma * torch.randn(n_dim, device=device)

    with torch.no_grad():
        X_samples = cnf.sample(Y_obs, num_samples)
        Y_samples = Y_obs.repeat(num_samples, 1)
        score = cnf.score(X_samples, Y_samples)
        h_x = X_samples
        g_combined, _ = ensemble(X_samples, Y_samples, score)

    # No closed-form mean -- use large-sample mean
    exact_mean = X_samples.mean(dim=0).cpu().numpy()

    return h_x.cpu().numpy(), g_combined.cpu().numpy(), exact_mean, n_dim


def bootstrap_analysis(h_np, g_np, exact_mean, n_dim, n_trials, rng):
    """Run bootstrap trials, return mean-across-dimensions VRF, MSE_raw, MSE_cv."""
    N_total = h_np.shape[0]

    vrf_all = np.zeros((n_dim, len(SAMPLE_SIZES), n_trials))
    mse_raw_all = np.zeros((n_dim, len(SAMPLE_SIZES), n_trials))
    mse_cv_all = np.zeros((n_dim, len(SAMPLE_SIZES), n_trials))

    for si, N in enumerate(SAMPLE_SIZES):
        for t in range(n_trials):
            idx = rng.choice(N_total, size=N, replace=False)
            for c in range(n_dim):
                h_sub = h_np[idx, c]
                g_sub = g_np[c, idx]
                resid = h_sub - g_sub

                var_h = np.var(h_sub, ddof=1)
                var_r = np.var(resid, ddof=1)
                vrf_all[c, si, t] = var_r / var_h if var_h > 0 else 1.0

                est_raw = np.mean(h_sub)
                est_cv = np.mean(resid)
                mse_raw_all[c, si, t] = (est_raw - exact_mean[c]) ** 2
                mse_cv_all[c, si, t] = (est_cv - exact_mean[c]) ** 2

    # Median across trials, then mean across dimensions
    vrf_med = np.median(vrf_all, axis=2).mean(axis=0)       # [n_sizes]
    mse_raw_med = np.median(mse_raw_all, axis=2).mean(axis=0)
    mse_cv_med = np.median(mse_cv_all, axis=2).mean(axis=0)

    return vrf_med, mse_raw_med, mse_cv_med


def main():
    parser = argparse.ArgumentParser(description="Generate combined CNF sample efficiency figure")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--n_trials", type=int, default=1000)
    parser.add_argument("--output_dir", type=str,
                        default="figs/experiments")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load all three datasets
    problems = {}

    print("=" * 60)
    print("Loading Gaussian (d=4)...")
    print("=" * 60)
    h, g, exact, d = load_gaussian_data(device, args.num_samples)
    problems["Gaussian"] = (h, g, exact, d)

    print("\n" + "=" * 60)
    print("Loading Rosenbrock (d=2)...")
    print("=" * 60)
    h, g, exact, d = load_rosenbrock_data(device, args.num_samples)
    problems["Rosenbrock"] = (h, g, exact, d)

    print("\n" + "=" * 60)
    print("Loading Nonlinear (d=4)...")
    print("=" * 60)
    h, g, exact, d = load_nonlinear_data(device, args.num_samples)
    problems["Nonlinear"] = (h, g, exact, d)

    # Bootstrap analysis for each problem
    rng = np.random.default_rng(123)
    results = {}
    for name, (h_np, g_np, exact_mean, n_dim) in problems.items():
        print(f"\nBootstrap analysis: {name} (d={n_dim})...")
        vrf, mse_raw, mse_cv = bootstrap_analysis(
            h_np, g_np, exact_mean, n_dim, args.n_trials, rng,
        )
        results[name] = (vrf, mse_raw, mse_cv)
        print(f"  VRF (mean across dims): {vrf}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))
    N_arr = np.array(SAMPLE_SIZES)

    # Panel (a): VRF vs N
    for name in ["Gaussian", "Rosenbrock", "Nonlinear"]:
        vrf, _, _ = results[name]
        ax1.plot(N_arr, vrf, "-" + PROBLEM_MARKERS[name],
                 color=PROBLEM_COLORS[name], markersize=4.5, linewidth=1.5,
                 label=name)

    ax1.set_xscale("log")
    ax1.set_xlabel("Sample size $N$")
    ax1.set_ylabel("VRF")
    ax1.set_title("(a) Variance reduction factor")
    ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2, linewidth=0.5)

    # Panel (b): MSE vs N
    # 1/N reference line
    ref_scale = results["Gaussian"][1][0] * N_arr[0]  # anchor to Gaussian raw at N=10
    ax2.plot(N_arr, ref_scale / N_arr, "k:", linewidth=1.0, alpha=0.5, label="$1/N$")

    for name in ["Gaussian", "Rosenbrock", "Nonlinear"]:
        _, mse_raw, mse_cv = results[name]
        ax2.plot(N_arr, mse_raw, "--", marker=PROBLEM_MARKERS[name],
                 color=PROBLEM_COLORS[name], markersize=3.5, linewidth=1.0,
                 alpha=0.5)
        ax2.plot(N_arr, mse_cv, "-", marker=PROBLEM_MARKERS[name],
                 color=PROBLEM_COLORS[name], markersize=4.5, linewidth=1.5,
                 label=name)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Sample size $N$")
    ax2.set_ylabel("MSE")
    ax2.set_title("(b) Mean squared error")
    # Custom legend: solid=CV, dashed=Raw
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, label="CV"),
               Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="Raw")]
    for name in ["Gaussian", "Rosenbrock", "Nonlinear"]:
        handles.append(Line2D([0], [0], color=PROBLEM_COLORS[name],
                              marker=PROBLEM_MARKERS[name], linestyle="-",
                              markersize=4.5, linewidth=1.5, label=name))
    handles.append(Line2D([0], [0], color="k", linestyle=":", linewidth=1.0, label="$1/N$"))
    ax2.legend(handles=handles, fontsize=8, ncol=2)
    ax2.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout(w_pad=0.6)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "cnf_sample_efficiency_combined.pdf")
    plt.savefig(out)
    plt.savefig(out.replace(".pdf", ".png"))
    print(f"\nSaved: {out}")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean VRF across dimensions and sample sizes)")
    print("=" * 60)
    for name in ["Gaussian", "Rosenbrock", "Nonlinear"]:
        vrf, mse_raw, mse_cv = results[name]
        print(f"  {name:12s}: VRF={vrf.mean():.4f}, "
              f"MSE reduction at N=1000: {mse_cv[6]/mse_raw[6]:.3f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
