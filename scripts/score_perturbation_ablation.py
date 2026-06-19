"""Score-perturbation ablation for the Gaussian d=4 CNCV checkpoint.

Measures how CNF score-approximation error propagates to the variance
estimate: we perturb the analytical posterior score with
Gaussian noise of varying std and measure the resulting VRF on a fixed
trained CNCV ensemble.

Output: figs/baselines/score_perturbation_d4.{pdf,png,json}
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cncv.models.ensemble import create_shared_hierarchical_ensemble
from cncv.utils.gaussian import (
    compute_exact_posterior,
    compute_score_posterior_gaussian,
)


def load_ensemble(ckpt_dir: str, device: torch.device):
    """Load the Gaussian d=4 ensemble; architecture is read from the checkpoint.

    ``ckpt_dir`` may be a run directory (the latest checkpoint is used) or a
    direct checkpoint path, so the per-dimension-tuned recipe loads correctly.
    """
    n_dim = 4
    p = Path(ckpt_dir)
    if p.is_dir():
        cks = sorted(p.glob("checkpoint_*.pth"),
                     key=lambda f: int(f.stem.split("_")[1]))
        ckpt_path = cks[-1]
    else:
        ckpt_path = p
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args")
    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_dim,
        getattr(a, "n_hidden", 64),
        getattr(a, "n_ensemble_members", 16),
        n_layers=getattr(a, "n_layers", 3),
        depth=getattr(a, "depth", 2),
        n_cv=n_dim,
        seed=12,
    ).to(device)
    state_dict = ckpt.get(
        "ensemble_state_dict", ckpt.get("model_state_dict", ckpt)
    )
    ensemble.load_state_dict(state_dict)
    ensemble.eval()
    return ensemble


def evaluate_vrf(
    ensemble,
    n_test_obs: int,
    n_samples_per_obs: int,
    sigma_y: float,
    sigma_score_noise: float,
    seed: int,
    device: torch.device,
):
    """Compute mean-VRF averaged over n_test_obs Gaussian d=4 observations,
    perturbing the analytical score with Gaussian noise of std sigma_score_noise.

    Returns:
        dict with vrf_per_component (list of length n_dim), vrf_mean (scalar),
        score_rmse (scalar, mean ||noise||/||score||), correlation_per_component (list).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_dim = 4
    mu_prior = torch.zeros(n_dim, device=device)
    Sigma_prior = torch.eye(n_dim, device=device)
    Sigma_prior_inv = torch.eye(n_dim, device=device)
    L_prior = torch.eye(n_dim, device=device)

    vrf_per_obs_per_comp = []  # [n_test_obs, n_dim]
    score_rmse_per_obs = []

    for obs_idx in range(n_test_obs):
        # Sample (x_true, y_obs) from prior predictive
        X_true = mu_prior + L_prior @ torch.randn(n_dim, device=device)
        Y_obs = X_true + sigma_y * torch.randn(n_dim, device=device)

        # Exact posterior
        mu_post, Sigma_post = compute_exact_posterior(
            Y_obs, mu_prior, Sigma_prior, sigma_y
        )

        # Sample N from exact posterior
        L_post = torch.linalg.cholesky(Sigma_post)
        noise = torch.randn(n_samples_per_obs, n_dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(n_samples_per_obs, 1)

        # Analytical score
        with torch.no_grad():
            score_clean = compute_score_posterior_gaussian(
                X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma_y
            )

            # Perturb score with Gaussian noise of std sigma_score_noise
            if sigma_score_noise > 0.0:
                score_noise = sigma_score_noise * torch.randn_like(score_clean)
                score = score_clean + score_noise
                rmse = score_noise.norm(dim=1).mean().item()
                score_norm = score_clean.norm(dim=1).mean().item()
                relative_rmse = rmse / max(score_norm, 1e-8)
            else:
                score = score_clean
                relative_rmse = 0.0

            score_rmse_per_obs.append(relative_rmse)

            g_combined, all_g = ensemble(X_samples, Y_samples, score)
            # g_combined shape: [n_dim, n_samples]

        # h(x) for mean estimation = x
        h_x = X_samples  # [n_samples, n_dim]

        # Per-component VRF: Var(h_k - g_k) / Var(h_k)
        vrf_components = []
        for k in range(n_dim):
            h_k = h_x[:, k]
            g_k = g_combined[k, :]
            var_h = h_k.var().item()
            var_h_minus_g = (h_k - g_k).var().item()
            vrf_components.append(var_h_minus_g / max(var_h, 1e-12))
        vrf_per_obs_per_comp.append(vrf_components)

    vrf_per_obs_per_comp = np.array(vrf_per_obs_per_comp)  # [n_obs, n_dim]
    # Mean over observations -> per-component VRF
    vrf_per_component = vrf_per_obs_per_comp.mean(axis=0)
    # Mean over components -> scalar
    vrf_mean = float(vrf_per_component.mean())
    return {
        "vrf_per_component": vrf_per_component.tolist(),
        "vrf_mean": vrf_mean,
        "score_relative_rmse": float(np.mean(score_rmse_per_obs)),
        "n_test_obs": n_test_obs,
        "sigma_score_noise": sigma_score_noise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="data/checkpoints/gaussian_ensemble_cv_input_dim-4_max_epochs-50_lr-0.001_lr_final-0.0001_sigma-0.3_n_ensemble_members-16_batchsize-2048_n_hidden-64_n_layers-3_num_train-65536_num_val-2048_num_samples-10000_save_freq-50_qofi-mean_dd.8e12cf8ef38e4051cf24e794ebe27b21a5d97a38",
    )
    parser.add_argument("--n_test_obs", type=int, default=100)
    parser.add_argument("--n_samples_per_obs", type=int, default=10000)
    parser.add_argument("--sigma_y", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="figs/baselines")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print("Loading ensemble...")
    ensemble = load_ensemble(args.ckpt_dir, device)
    print(f"  loaded from {args.ckpt_dir}")

    sigma_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = []
    for sigma_score_noise in sigma_levels:
        print(f"\n=== sigma_score_noise = {sigma_score_noise} ===")
        out = evaluate_vrf(
            ensemble=ensemble,
            n_test_obs=args.n_test_obs,
            n_samples_per_obs=args.n_samples_per_obs,
            sigma_y=args.sigma_y,
            sigma_score_noise=sigma_score_noise,
            seed=args.seed,
            device=device,
        )
        print(
            f"  VRF mean: {out['vrf_mean']:.4f}, "
            f"score relative RMSE: {out['score_relative_rmse']:.4f}"
        )
        print(f"  per-component VRF: {[f'{v:.3f}' for v in out['vrf_per_component']]}")
        results.append(out)

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "score_perturbation_d4.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "ckpt_dir": args.ckpt_dir,
                "n_test_obs": args.n_test_obs,
                "n_samples_per_obs": args.n_samples_per_obs,
                "sigma_y": args.sigma_y,
                "seed": args.seed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {json_path}")

    # Plot
    sigmas = [r["sigma_score_noise"] for r in results]
    rmses = [r["score_relative_rmse"] for r in results]
    vrfs = [r["vrf_mean"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 7))
    ax1.plot(sigmas, vrfs, "o-", color="C0", lw=2, ms=8)
    ax1.axhline(1.0, color="gray", ls="--", lw=1, label="VRF = 1 (no improvement)")
    ax1.axhline(vrfs[0], color="C2", ls=":", lw=1, label=f"clean-score VRF = {vrfs[0]:.3f}")
    ax1.set_xlabel(r"Score-perturbation noise std $\sigma_{\mathrm{noise}}$")
    ax1.set_ylabel("Mean VRF (averaged over components, 100 test obs)")
    ax1.set_title(r"VRF vs. score noise (Gaussian $d=4$, mean estimation)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(rmses, vrfs, "o-", color="C1", lw=2, ms=8)
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="VRF = 1 (no improvement)")
    ax2.axhline(vrfs[0], color="C2", ls=":", lw=1, label=f"clean-score VRF = {vrfs[0]:.3f}")
    ax2.set_xlabel(r"Score relative RMSE  $\mathbb{E}\|\epsilon\|/\|s\|$")
    ax2.set_ylabel("Mean VRF")
    ax2.set_title("VRF vs. relative score-RMSE")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    pdf_path = out_dir / "score_perturbation_d4.pdf"
    png_path = out_dir / "score_perturbation_d4.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
