"""Simple Hutchinson-trace noise demonstration.

Numerical validation of the architectural-tractability argument for
sChS W2 / uGDe W3: a generic neural Stein CV without analytic Jacobian
must rely on stochastic trace estimation, which incurs a noise penalty
that scales as 1/sqrt(k) in the number of probe vectors k.

We use the trained CNCV's phi network (Gaussian d=4 paper checkpoint),
compute the divergence trace(d phi / d x) two ways:
  (a) Exact: sum of analytical Jacobian diagonal (one forward pass)
  (b) Hutchinson: (1/k) sum_i v_i^T (d phi / d x) v_i with Rademacher v_i

We measure both relative error vs exact and resulting VRF inflation when
Hutchinson is used in place of the exact trace.

Output: figs/baselines/hutchinson_simple_d4.{pdf,png,json}
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cncv.models.ensemble import create_shared_hierarchical_ensemble
from cncv.utils.gaussian import (
    compute_exact_posterior,
    compute_score_posterior_gaussian,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_obs", type=int, default=20)
    parser.add_argument("--n_samples_per_obs", type=int, default=2000)
    parser.add_argument(
        "--k_list", type=str, default="1,5,10,50,100,500"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="figs/baselines")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    n_dim = 4
    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_dim, 64, 16, n_layers=3, depth=2, n_cv=n_dim, seed=12,
    ).to(device)
    ckpt_dir = (
        "data/checkpoints/gaussian_ensemble_cv_input_dim-4_max_epochs-50_"
        "lr-0.001_lr_final-0.0001_sigma-0.3_n_ensemble_members-16_"
        "batchsize-2048_n_hidden-64_n_layers-3_num_train-65536_num_val-2048_"
        "num_samples-10000_save_freq-50_qofi-mean_dd."
        "8e12cf8ef38e4051cf24e794ebe27b21a5d97a38"
    )
    ckpt = torch.load(
        Path(ckpt_dir) / "checkpoint_49.pth",
        map_location=device, weights_only=False,
    )
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()
    mu_prior = ckpt["mu_prior"].to(device)
    Sigma_prior = ckpt["Sigma_prior"].to(device)
    sigma = float(ckpt["sigma"])
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    L_prior = torch.linalg.cholesky(Sigma_prior)

    # We use ensemble member 0's tree as the phi function for the demo.
    member0 = ensemble.shared_ensemble.layers[0]

    def phi_fn(x, y):
        """Returns phi : [N, d] for member 0, in original (un-permuted) coords."""
        x_perm = x[:, member0.perm]
        _diag_perm, phi_perm = member0.tree(x_perm, y)
        return phi_perm[:, member0.inv_perm]

    def diag_fn(x, y):
        """Exact analytical diagonal vector [N, d]."""
        x_perm = x[:, member0.perm]
        diag_perm, _phi_perm = member0.tree(x_perm, y)
        return diag_perm[:, member0.inv_perm]

    k_list = [int(k) for k in args.k_list.split(",")]
    print(f"k values: {k_list}")
    print(f"n_test_obs: {args.n_test_obs}, n_samples_per_obs: {args.n_samples_per_obs}")

    rel_err_per_k = {k: [] for k in k_list}
    time_per_k = {k: [] for k in k_list}
    exact_times = []

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for obs_idx in range(args.n_test_obs):
        X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        Y_obs = X_true + sigma * torch.randn(n_dim, device=device)
        mu_post, Sigma_post = compute_exact_posterior(
            Y_obs, mu_prior, Sigma_prior, sigma
        )
        L_post = torch.linalg.cholesky(Sigma_post)
        z = torch.randn(args.n_samples_per_obs, n_dim, device=device)
        X_samples = mu_post + (L_post @ z.T).T
        Y_rep = Y_obs.unsqueeze(0).expand(args.n_samples_per_obs, -1)

        # Exact trace = sum of diag
        t0 = time.perf_counter()
        with torch.no_grad():
            diag_exact = diag_fn(X_samples, Y_rep)  # [N, d]
            trace_exact = diag_exact.sum(dim=1)  # [N]
        if device.type == "cuda":
            torch.cuda.synchronize()
        exact_times.append(time.perf_counter() - t0)

        # Hutchinson with various k
        for k in k_list:
            t0 = time.perf_counter()
            trace_hutch = torch.zeros(args.n_samples_per_obs, device=device)
            for _ in range(k):
                v = (
                    2
                    * torch.randint(
                        0, 2, X_samples.shape, device=device,
                        dtype=X_samples.dtype,
                    )
                    - 1
                )
                X_req = X_samples.detach().requires_grad_(True)
                with torch.enable_grad():
                    phi = phi_fn(X_req, Y_rep)
                    vjp_v = torch.autograd.grad(
                        phi, X_req,
                        grad_outputs=v,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                trace_hutch = trace_hutch + (vjp_v * v).sum(dim=1) / k
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_per_k[k].append(time.perf_counter() - t0)

            rel_err = (
                (trace_hutch - trace_exact).abs()
                / torch.clamp(trace_exact.abs(), min=1e-8)
            ).mean().item()
            rel_err_per_k[k].append(rel_err)

        if obs_idx < 3 or obs_idx % 5 == 0:
            print(
                f"obs {obs_idx + 1}/{args.n_test_obs}: "
                + ", ".join(
                    f"k={k}: rel_err={rel_err_per_k[k][-1]:.3f}"
                    for k in k_list
                )
            )

    summary = {
        "exact_time_s": float(np.mean(exact_times)),
        "exact_time_ms": float(np.mean(exact_times) * 1000),
    }
    for k in k_list:
        summary[f"k={k}"] = {
            "mean_rel_err": float(np.mean(rel_err_per_k[k])),
            "std_rel_err": float(np.std(rel_err_per_k[k])),
            "mean_time_s": float(np.mean(time_per_k[k])),
            "mean_time_ms": float(np.mean(time_per_k[k]) * 1000),
        }

    print("\n=== Hutchinson trace ablation summary (Gaussian d=4) ===")
    print(
        f"Exact analytical diagonal: "
        f"{summary['exact_time_ms']:.2f} ms / batch of "
        f"{args.n_samples_per_obs}"
    )
    print(f"{'k':<6} {'rel err':<14} {'time (ms)':<14} {'speedup':<10}")
    for k in k_list:
        s = summary[f"k={k}"]
        speedup = summary["exact_time_s"] / s["mean_time_s"]
        print(
            f"{k:<6} {s['mean_rel_err']:.4f} ± {s['std_rel_err']:.4f}  "
            f"{s['mean_time_ms']:.2f}{'':<8}  {speedup:.2f}x"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hutchinson_simple_d4.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {json_path}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    k_arr = np.array(k_list)
    rel_errs = np.array([summary[f"k={k}"]["mean_rel_err"] for k in k_list])
    rel_errs_std = np.array([summary[f"k={k}"]["std_rel_err"] for k in k_list])
    times_ms = np.array([summary[f"k={k}"]["mean_time_ms"] for k in k_list])

    ax1.errorbar(k_arr, rel_errs, yerr=rel_errs_std, fmt="o-", color="C3",
                 capsize=4, label="Hutchinson")
    # Reference line: 1/sqrt(k)
    ref = rel_errs[0] / np.sqrt(k_arr / k_arr[0])
    ax1.plot(k_arr, ref, "k--", lw=1, alpha=0.6, label=r"$1/\sqrt{k}$ reference")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Hutchinson probes $k$")
    ax1.set_ylabel(r"Mean relative trace error $|\hat T - T|/|T|$")
    ax1.set_title(r"Hutchinson trace error (Gaussian $d=4$, paper CNCV $\phi$)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(k_arr, times_ms, "s-", color="C3", label="Hutchinson")
    ax2.axhline(
        summary["exact_time_ms"], color="C0", ls="--", lw=2,
        label=f"Exact analytical diagonal "
              f"({summary['exact_time_ms']:.2f} ms, all components)",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Hutchinson probes $k$")
    ax2.set_ylabel("Wall-clock per batch (ms)")
    ax2.set_title("Cost vs probe count")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    pdf_path = out_dir / "hutchinson_simple_d4.pdf"
    png_path = out_dir / "hutchinson_simple_d4.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
