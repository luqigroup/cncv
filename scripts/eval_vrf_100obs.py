"""Honest 100-observation VRF evaluation of a Gaussian CNCV checkpoint.

This is the evaluation backbone for the per-d hyperparameter study. It mirrors
the exact protocol of ``run_dimension_scaling.py::run_test_and_collect`` (the
function that produced the paper's Fig. 1 / Table 1 numbers): for a trained
shared-hierarchical ensemble it draws ``n_test_obs`` independent observations
y = x_true + sigma*noise, samples the analytical Gaussian posterior p(x|y), and
reports the per-component variance reduction factor VRF = Var(h - g)/Var(h),
averaged over observations.

Unlike ``run_dimension_scaling``, it loads a CHECKPOINT BY PATH (not by config
discovery), which avoids the recipe-ambiguity when several runs share a
dimension, and it reports BOTH summary statistics that matter for the per-d
study:
  - the paper-comparable mean over components (mean over obs)  -- vs Fig. 1
  - the median over components (mean over obs)
  - the per-observation distribution (mean over components per obs), whose
    median is the "median VRF" the rebuttal reported.

The point: a single-observation eval can report a per-component median far below
the honest 100-observation mean. This harness makes the two protocols explicit
so the per-d HP-tuned numbers cannot be conflated.

Usage:
    python scripts/eval_vrf_100obs.py --gpu 0 --qofi mean \
        --checkpoint data/checkpoints/<run-dir>/checkpoint_99.pth \
        --label tuned_d16 --out figs/per_d_hp/eval_tuned_d16.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cncv import (
    create_random_split_ensemble,
    create_hierarchical_ensemble,
    create_shared_hierarchical_ensemble,
    compute_score_posterior_gaussian,
)


def _build_ensemble(ckpt, n_dim, device):
    """Reconstruct the ensemble from a checkpoint's stored args + layer_type.

    Mirrors run_dimension_scaling.run_test_and_collect so the two evaluators
    agree on the architecture for any recipe.
    """
    args = ckpt["args"]
    layer_type = ckpt.get("layer_type", "shared_hierarchical")
    depth = getattr(args, "depth", None) or None  # 0/None -> auto
    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim, args.n_hidden, args.n_ensemble_members,
            n_layers=args.n_layers, depth=depth, n_cv=n_dim, seed=12,
        )
    elif layer_type == "hierarchical":
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim, args.n_hidden, args.n_ensemble_members,
            n_layers=args.n_layers, depth=depth, n_cv=n_dim, seed=12,
        )
    else:
        ensemble = create_random_split_ensemble(
            n_dim, n_dim, args.n_hidden, args.n_ensemble_members,
            args.n_layers, n_cv=n_dim, seed=12,
        )
    return ensemble.to(device)


def evaluate_checkpoint_100obs(ckpt_path, qofi="mean", device="cuda:0",
                               n_test_obs=100, num_samples=10000, seed=42):
    """Evaluate per-component VRF of a Gaussian CNCV checkpoint over many obs.

    Returns a dict with per-component (obs-averaged) VRF and correlation, plus
    summary statistics under both the mean-over-components (paper) and
    median-based (rebuttal) conventions.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mu_prior = ckpt["mu_prior"].to(device)
    Sigma_prior = ckpt["Sigma_prior"].to(device)
    sigma = ckpt["sigma"]
    n_dim = ckpt.get("input_dim", mu_prior.shape[0])

    ensemble = _build_ensemble(ckpt, n_dim, device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

    # Analytical Gaussian posterior moments (identity observation operator).
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    L_prior = torch.linalg.cholesky(Sigma_prior)
    I = torch.eye(n_dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1.0 / sigma ** 2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)

    per_obs_component_vrf = []   # [n_test_obs, n_dim]
    per_obs_component_corr = []

    torch.manual_seed(seed)
    for _ in range(n_test_obs):
        X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        Y_obs = X_true + sigma * torch.randn(n_dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior
                                + (1.0 / sigma ** 2) * Y_obs)

        noise = torch.randn(num_samples, n_dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)

        with torch.no_grad():
            score = compute_score_posterior_gaussian(
                X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_samples if qofi == "mean" else (X_samples - mu_post) ** 2
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        obs_vrf, obs_corr = [], []
        for i in range(n_dim):
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            obs_vrf.append(controlled.var().item() / max(var_h, 1e-12))
            corr = torch.corrcoef(
                torch.stack([h_x[:, i], g_combined[i, :]]))[0, 1].item()
            obs_corr.append(corr)
        per_obs_component_vrf.append(obs_vrf)
        per_obs_component_corr.append(obs_corr)

    per_obs_component_vrf = np.array(per_obs_component_vrf)   # [obs, dim]
    per_obs_component_corr = np.array(per_obs_component_corr)

    # Paper convention: average over obs, then summarize over components.
    comp_vrf = per_obs_component_vrf.mean(axis=0)             # [dim]
    comp_corr = per_obs_component_corr.mean(axis=0)
    # Per-obs scalar (mean over components) -> its median is a robust per-obs stat.
    per_obs_mean_vrf = per_obs_component_vrf.mean(axis=1)     # [obs]

    return {
        "checkpoint": ckpt_path,
        "qofi": qofi,
        "n_dim": int(n_dim),
        "n_test_obs": int(n_test_obs),
        "num_samples": int(num_samples),
        "seed": int(seed),
        "recipe": {
            "n_hidden": getattr(ckpt["args"], "n_hidden", None),
            "n_layers": getattr(ckpt["args"], "n_layers", None),
            "n_ensemble_members": getattr(ckpt["args"], "n_ensemble_members", None),
            "depth": getattr(ckpt["args"], "depth", None),
            "layer_type": ckpt.get("layer_type", "shared_hierarchical"),
            "epoch": ckpt.get("epoch", None),
        },
        # Paper-comparable headline (mean over obs and components):
        "vrf_mean": float(comp_vrf.mean()),
        "vrf_std_over_components": float(comp_vrf.std()),
        # Robust summaries:
        "vrf_median_over_components": float(np.median(comp_vrf)),
        "vrf_median_over_obs": float(np.median(per_obs_mean_vrf)),
        "correlation_mean": float(comp_corr.mean()),
        "vrf_per_component": comp_vrf.tolist(),
        "per_obs_mean_vrf": per_obs_mean_vrf.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Honest 100-obs VRF eval of a Gaussian CNCV checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a checkpoint_*.pth file")
    parser.add_argument("--qofi", type=str, default="mean", choices=["mean", "var"])
    parser.add_argument("--n_test_obs", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--label", type=str, default="",
                        help="Optional label recorded in the JSON")
    parser.add_argument("--out", type=str, default="",
                        help="Optional JSON output path")
    args = parser.parse_args()

    res = evaluate_checkpoint_100obs(
        args.checkpoint, qofi=args.qofi, device=f"cuda:{args.gpu}",
        n_test_obs=args.n_test_obs, num_samples=args.num_samples, seed=args.seed)
    res["label"] = args.label

    print(f"\n=== {args.label or 'checkpoint'} (d={res['n_dim']}, "
          f"{res['recipe']}) ===")
    print(f"  VRF mean over {args.n_test_obs} obs x components : "
          f"{res['vrf_mean']:.4f} +/- {res['vrf_std_over_components']:.4f}")
    print(f"  VRF median over components (obs-avg)            : "
          f"{res['vrf_median_over_components']:.4f}")
    print(f"  VRF median over obs (component-avg)             : "
          f"{res['vrf_median_over_obs']:.4f}")
    print(f"  Correlation (mean)                              : "
          f"{res['correlation_mean']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
