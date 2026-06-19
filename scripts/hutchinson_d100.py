"""Hutchinson trace ablation at d=100 — Darcy and Helmholtz.

Strongest version of the architectural-advantage demonstration: at d=100,
Hutchinson estimator variance scales with ||J||_F^2, so the noise penalty
worsens compared to the d=4 baseline. We hold the trained CNCV phi fixed
(Darcy or Helmholtz checkpoint) and compute trace(d phi / d x) two ways:

  (a) Exact analytical diagonal (one forward pass) — what CNCV uses.
  (b) Hutchinson with k Rademacher probes — what a generic neural Stein CV
      would require.

Output: figs/baselines/hutchinson_{darcy,helmholtz}_d100.{pdf,png,json}
"""

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cncv.models.conditional_nf import ConditionalNF
from cncv.models.shared_cv import (
    SharedEnsembleCV,
    SharedHierarchicalCV,
)
from cncv.models.ensemble import SharedEnsembleCVWrapper
from cncv.utils.download import ensure_dataset


def create_shared_ensemble(n_in, n_cond, n_hidden, n_ensemble, depth,
                           n_mlp_layers, seed=12):
    if seed is not None:
        torch.manual_seed(seed)
    layers = [
        SharedHierarchicalCV(
            n_in=n_in, n_cond=n_cond, n_hidden=n_hidden,
            depth=depth, n_mlp_layers=n_mlp_layers,
        ) for _ in range(n_ensemble)
    ]
    return SharedEnsembleCV(layers)


def load_models(cncv_ckpt_path, device):
    ckpt = torch.load(cncv_ckpt_path, map_location=device, weights_only=False)
    d_z, d_y = ckpt["d_z"], ckpt["d_y"]
    y_mean = ckpt["y_mean"].cpu()
    y_std = ckpt["y_std"].cpu()
    inner = create_shared_ensemble(
        n_in=d_z, n_cond=d_y,
        n_hidden=ckpt["cncv_hidden"],
        n_ensemble=ckpt["n_ensemble"],
        depth=ckpt["cncv_depth"],
        n_mlp_layers=ckpt["cncv_mlp_layers"],
        seed=12,
    )
    ensemble = SharedEnsembleCVWrapper(inner).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

    cnf_ckpt = torch.load(ckpt["cnf_checkpoint"], map_location=device,
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

    return cnf, ensemble, y_mean, y_std, d_z, d_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cncv_checkpoint", type=str, required=True,
        help="path to CNCV checkpoint",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="path to .npz with z and y_obs (for sampling test obs)",
    )
    parser.add_argument("--problem_name", type=str, required=True,
                        help="darcy or helmholtz; used in output names")
    parser.add_argument("--n_test_obs", type=int, default=10)
    parser.add_argument("--n_samples_per_obs", type=int, default=500)
    parser.add_argument("--k_list", type=str, default="1,5,10,50,100,500")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="figs/baselines")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.problem_name} CNCV from {args.cncv_checkpoint}")

    cnf, ensemble, y_mean, y_std, d_z, d_y = load_models(
        args.cncv_checkpoint, device
    )
    print(f"d_z={d_z}, d_y={d_y}")

    # Use member 0 of the ensemble as the phi function for the demo
    member0 = ensemble.shared_ensemble.layers[0]

    def phi_fn(x, y):
        x_perm = x[:, member0.perm]
        _diag, phi_perm = member0.tree(x_perm, y)
        return phi_perm[:, member0.inv_perm]

    def diag_fn(x, y):
        x_perm = x[:, member0.perm]
        diag_perm, _phi = member0.tree(x_perm, y)
        return diag_perm[:, member0.inv_perm]

    # Sample test observations from data
    ensure_dataset(args.data_path)
    data = np.load(args.data_path)
    y_all = data["y_obs"]
    n_total = y_all.shape[0]
    test_idx = np.random.RandomState(args.seed).choice(
        n_total, size=args.n_test_obs, replace=False
    )
    print(f"Loaded {n_total} samples; using {args.n_test_obs} for test")

    k_list = [int(k) for k in args.k_list.split(",")]
    print(f"k values: {k_list}")
    print(f"n_samples_per_obs: {args.n_samples_per_obs}")

    rel_err_per_k = {k: [] for k in k_list}
    time_per_k = {k: [] for k in k_list}
    exact_times = []

    torch.manual_seed(args.seed)

    for obs_idx in range(args.n_test_obs):
        idx = test_idx[obs_idx]
        y_obs_raw = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
        y_obs = (y_obs_raw - y_mean.to(device)) / y_std.to(device)

        with torch.no_grad():
            X_samples = cnf.sample(y_obs.unsqueeze(0), args.n_samples_per_obs)
        Y_rep = y_obs.unsqueeze(0).expand(args.n_samples_per_obs, -1)

        # Exact trace = sum(diag)
        t0 = time.perf_counter()
        with torch.no_grad():
            diag_exact = diag_fn(X_samples, Y_rep)
            trace_exact = diag_exact.sum(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        exact_times.append(time.perf_counter() - t0)

        # Hutchinson sweep
        for k in k_list:
            t0 = time.perf_counter()
            trace_h = torch.zeros(args.n_samples_per_obs, device=device)
            for _ in range(k):
                v = (
                    2 * torch.randint(
                        0, 2, X_samples.shape, device=device,
                        dtype=X_samples.dtype,
                    ) - 1
                )
                X_req = X_samples.detach().requires_grad_(True)
                with torch.enable_grad():
                    phi = phi_fn(X_req, Y_rep)
                    vjp_v = torch.autograd.grad(
                        phi, X_req, grad_outputs=v,
                        retain_graph=False, create_graph=False,
                    )[0]
                trace_h = trace_h + (vjp_v * v).sum(dim=1) / k
            if device.type == "cuda":
                torch.cuda.synchronize()
            time_per_k[k].append(time.perf_counter() - t0)
            rel_err = (
                (trace_h - trace_exact).abs()
                / torch.clamp(trace_exact.abs(), min=1e-8)
            ).mean().item()
            rel_err_per_k[k].append(rel_err)

        line = f"obs {obs_idx + 1}/{args.n_test_obs}: "
        line += ", ".join(
            f"k={k}: rel_err={rel_err_per_k[k][-1]:.3f}" for k in k_list
        )
        print(line)

    summary = {
        "problem": args.problem_name,
        "d_z": d_z,
        "exact_time_ms": float(np.mean(exact_times) * 1000),
    }
    for k in k_list:
        summary[f"k={k}"] = {
            "mean_rel_err": float(np.mean(rel_err_per_k[k])),
            "std_rel_err": float(np.std(rel_err_per_k[k])),
            "mean_time_ms": float(np.mean(time_per_k[k]) * 1000),
        }

    print(f"\n=== {args.problem_name.upper()} Hutchinson trace ablation (d={d_z}) ===")
    print(f"Exact analytical diagonal: {summary['exact_time_ms']:.2f} ms / batch")
    print(f"{'k':<6} {'rel err':<22} {'time (ms)':<14}")
    for k in k_list:
        s = summary[f"k={k}"]
        print(f"{k:<6} {s['mean_rel_err']:.4f} ± {s['std_rel_err']:.4f}     {s['mean_time_ms']:.2f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"hutchinson_{args.problem_name}_d{d_z}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {json_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    k_arr = np.array(k_list)
    rel_errs = np.array([summary[f"k={k}"]["mean_rel_err"] for k in k_list])
    rel_errs_std = np.array([summary[f"k={k}"]["std_rel_err"] for k in k_list])
    times_ms = np.array([summary[f"k={k}"]["mean_time_ms"] for k in k_list])

    ax1.errorbar(k_arr, rel_errs, yerr=rel_errs_std, fmt="o-",
                 color="C3", capsize=4, label="Hutchinson")
    ref = rel_errs[0] / np.sqrt(k_arr / k_arr[0])
    ax1.plot(k_arr, ref, "k--", lw=1, alpha=0.6, label=r"$1/\sqrt{k}$ reference")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Hutchinson probes $k$")
    ax1.set_ylabel(r"Mean relative trace error $|\hat T - T|/|T|$")
    ax1.set_title(f"{args.problem_name.capitalize()} $d={d_z}$: Hutchinson trace error")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(k_arr, times_ms, "s-", color="C3", label="Hutchinson")
    ax2.axhline(
        summary["exact_time_ms"], color="C0", ls="--", lw=2,
        label=f"Exact analytical diagonal "
              f"({summary['exact_time_ms']:.2f} ms)",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Hutchinson probes $k$")
    ax2.set_ylabel("Wall-clock per batch (ms)")
    ax2.set_title("Cost vs probe count")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    pdf_path = out_dir / f"hutchinson_{args.problem_name}_d{d_z}.pdf"
    png_path = out_dir / f"hutchinson_{args.problem_name}_d{d_z}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
