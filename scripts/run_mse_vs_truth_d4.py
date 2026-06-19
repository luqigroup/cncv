"""MSE vs analytical posterior mean — Gaussian d=4 panel.

Loads the paper's Gaussian d=4 mean-estimation CNCV checkpoint
(hash 8e12cf8e) and produces an MSE-vs-N curve where the *truth* is the
analytical posterior mean (closed-form for the Gaussian linear model),
not the sample mean of the posterior sampler.

The CNCV checkpoint at hash 8e12cf8e was trained with **analytical**
scores (ddpm=0). To isolate CV bias from sampler bias we therefore draw
posterior samples from the analytical Gaussian posterior (which we know
exactly) and apply the trained ensemble CV with analytical scores. Any
bias floor in the resulting MSE curve reflects only the residual
violation of Stein's identity by the trained CV.

Output:
    figs/baselines/gaussian_d4_mse_vs_truth.{pdf,png,json}
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from projorg import checkpointsdir, setup_environment

from cncv import (
    compute_score_posterior_gaussian,
    create_shared_hierarchical_ensemble,
    load_ensemble_checkpoint,
)


CONFIG_FILE = "gaussian_ensemble_cv.json"
FIG_DIR = "figs/baselines"
N_TEST_OBS = 20
N_LIST = [10, 30, 100, 300, 1000, 3000, 10000]
N_SEEDS = 200  # bootstrap seeds for MC error bars
N_POOL = 1_000_000  # large posterior-sample pool per observation


def build_problem(n_dim: int, sigma: float, device):
    """Reconstruct the deterministic prior covariance used at training time."""
    diag_values = torch.linspace(1.0, 2.0, n_dim)
    D = torch.diag(diag_values).to(device=device, dtype=torch.float32)
    torch.manual_seed(42)  # SAME seed as in gaussian_ensemble_cv.py
    A = torch.randn(n_dim, n_dim, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    Sigma_prior = Q @ D @ Q.T
    return Sigma_prior


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=[
            "experiment_name",
            "gpu_id",
            "phase",
            "seed",
            "testing_epoch",
        ],
    )

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")
    print(f"Experiment dir: {args.experiment}")

    n_dim = int(args.input_dim)
    sigma = float(args.sigma)
    print(f"n_dim={n_dim}, sigma={sigma}")

    # --- Build prior / posterior parameters ---
    mu_prior = torch.zeros(n_dim, dtype=torch.float32, device=device)
    Sigma_prior = build_problem(n_dim, sigma, device)
    L_prior = torch.linalg.cholesky(Sigma_prior)
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    I = torch.eye(n_dim, device=device, dtype=torch.float32)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1.0 / sigma**2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)

    # --- Build ensemble and load checkpoint ---
    layer_type = getattr(args, "layer_type", "shared_hierarchical")
    depth = getattr(args, "depth", None) or None
    assert layer_type == "shared_hierarchical", layer_type
    ensemble = create_shared_hierarchical_ensemble(
        n_dim,
        n_dim,
        args.n_hidden,
        args.n_ensemble_members,
        n_layers=args.n_layers,
        depth=depth,
        n_cv=n_dim,
        seed=12,
    ).to(device)

    epoch = args.max_epochs - 1
    # The paper's Gaussian d=4 mean checkpoint lives at hash 8e12cf8e,
    # which is locked-in regardless of any local working-tree changes.
    # We override the experiment-name hash so re-running this script under
    # later code does not silently miss the checkpoint.
    paper_ckpt_dir = (
        "gaussian_ensemble_cv_input_dim-4_max_epochs-50_lr-0.001_lr_final-0.0001"
        "_sigma-0.3_n_ensemble_members-16_batchsize-2048_n_hidden-64_n_layers-3"
        "_num_train-65536_num_val-2048_num_samples-10000_save_freq-50_qofi-mean"
        "_dd.8e12cf8ef38e4051cf24e794ebe27b21a5d97a38"
    )
    ckpt_path = os.path.join(
        "data/checkpoints",
        paper_ckpt_dir,
        f"checkpoint_{epoch}.pth",
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_ensemble_checkpoint(
        ensemble, ckpt_path, device, validate_splits=False
    )
    ensemble.eval()
    for p in ensemble.parameters():
        p.requires_grad_(False)

    # Sanity: verify problem parameters match
    assert torch.allclose(ckpt["mu_prior"].to(device), mu_prior, atol=1e-6)
    assert torch.allclose(
        ckpt["Sigma_prior"].to(device), Sigma_prior, atol=1e-5
    ), "Sigma_prior mismatch with checkpoint!"
    assert abs(ckpt["sigma"] - sigma) < 1e-6
    print("Checkpoint sanity check passed.")

    # --- Generate held-out test observations ---
    # Use a different seed than training (training used seed 12).
    test_seed = 2026
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)

    X_true_test = mu_prior + (
        L_prior @ torch.randn(N_TEST_OBS, n_dim, device=device).T
    ).T
    Y_test = X_true_test + sigma * torch.randn_like(X_true_test)

    # Analytical posterior mean for each test observation.
    # Sigma_prior_inv @ mu_prior is zero since mu_prior=0.
    mu_post_test = (Sigma_post @ (Y_test.T / sigma**2)).T  # [N_TEST_OBS, n_dim]
    print(f"Generated {N_TEST_OBS} test observations.")

    # --- For each test obs: draw a large pool of analytical posterior samples,
    # evaluate h(x) and g(x) once, then bootstrap-subsample for MC curves. ---
    n_grid = len(N_LIST)
    # Per-(test_obs, n_idx, seed): squared-error component-wise summed -> store norm sq
    se_raw = np.zeros((N_TEST_OBS, n_grid, N_SEEDS), dtype=np.float64)
    se_cncv = np.zeros((N_TEST_OBS, n_grid, N_SEEDS), dtype=np.float64)
    # Asymptotic CNCV mean estimate (using full pool) per test obs
    asymp_mean_cncv = np.zeros((N_TEST_OBS, n_dim), dtype=np.float64)
    asymp_mean_raw = np.zeros((N_TEST_OBS, n_dim), dtype=np.float64)

    print(f"Running over {N_TEST_OBS} obs, pool={N_POOL}, n_seeds={N_SEEDS} ...")
    rng_master = np.random.default_rng(test_seed + 1)
    for i in range(N_TEST_OBS):
        Y_obs = Y_test[i]  # [n_dim]
        mu_post = mu_post_test[i]  # [n_dim]

        # Large pool of exact analytical posterior samples.
        torch.manual_seed(test_seed * 100 + i)
        eps = torch.randn(N_POOL, n_dim, device=device)
        X_pool = mu_post[None, :] + (L_post @ eps.T).T  # [N_POOL, n_dim]
        Y_pool = Y_obs[None, :].expand(N_POOL, -1)

        # Evaluate ensemble CV in chunks to fit on GPU.
        chunk = 50_000
        g_list = []
        with torch.no_grad():
            for s in range(0, N_POOL, chunk):
                Xc = X_pool[s : s + chunk]
                Yc = Y_pool[s : s + chunk]
                score = compute_score_posterior_gaussian(
                    Xc, Yc, mu_prior, Sigma_prior_inv, sigma
                )
                g_combined, _ = ensemble(Xc, Yc, score)
                # g_combined is [n_dim, batch]; we want [batch, n_dim]
                g_list.append(g_combined.T.detach().cpu().numpy())
        g_pool = np.concatenate(g_list, axis=0)  # [N_POOL, n_dim]
        h_pool = X_pool.detach().cpu().numpy()  # [N_POOL, n_dim], h(x)=x
        mu_post_np = mu_post.detach().cpu().numpy()

        # Asymptotic estimates from full pool
        asymp_mean_raw[i] = h_pool.mean(axis=0)
        asymp_mean_cncv[i] = (h_pool - g_pool).mean(axis=0)

        # Bootstrap subsamples per N value
        for j, n in enumerate(N_LIST):
            for s in range(N_SEEDS):
                idx = rng_master.integers(0, N_POOL, size=n)
                hat_raw = h_pool[idx].mean(axis=0)
                hat_cncv = (h_pool[idx] - g_pool[idx]).mean(axis=0)
                se_raw[i, j, s] = np.sum((hat_raw - mu_post_np) ** 2)
                se_cncv[i, j, s] = np.sum((hat_cncv - mu_post_np) ** 2)

        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"  obs {i+1}/{N_TEST_OBS}: "
                f"raw asym L2 err = {np.linalg.norm(asymp_mean_raw[i] - mu_post_np):.4e}, "
                f"cncv asym L2 err = {np.linalg.norm(asymp_mean_cncv[i] - mu_post_np):.4e}"
            )

    # --- Aggregate: MSE = mean over obs and seeds of squared L2 error ---
    mse_raw = se_raw.mean(axis=(0, 2))  # [n_grid]
    mse_cncv = se_cncv.mean(axis=(0, 2))
    # Standard error across (obs * seeds) for confidence band
    sem_raw = se_raw.reshape(-1, n_grid).std(axis=0) / np.sqrt(N_TEST_OBS * N_SEEDS)
    sem_cncv = se_cncv.reshape(-1, n_grid).std(axis=0) / np.sqrt(
        N_TEST_OBS * N_SEEDS
    )

    # Asymptotic bias floor: squared L2 error of the asymptotic CNCV mean
    # vs analytical posterior mean, averaged over test obs.
    bias_floor_per_obs = np.sum(
        (asymp_mean_cncv - mu_post_test.detach().cpu().numpy()) ** 2, axis=1
    )
    bias_floor_mean = float(bias_floor_per_obs.mean())
    bias_floor_max = float(bias_floor_per_obs.max())
    print(
        f"\nAsymptotic CNCV bias floor (avg over {N_TEST_OBS} obs): "
        f"{bias_floor_mean:.4e}  (max: {bias_floor_max:.4e})"
    )

    # Same for raw MC (should be ~ noise from huge pool, basically zero)
    asymp_floor_raw = float(
        np.mean(
            np.sum(
                (asymp_mean_raw - mu_post_test.detach().cpu().numpy()) ** 2, axis=1
            )
        )
    )
    print(f"Asymptotic raw-MC residual at N={N_POOL}: {asymp_floor_raw:.4e}")

    # --- Variance reduction factor at each N (CNCV/raw ratio) ---
    vrf_per_n = mse_cncv / mse_raw
    for n, m_r, m_c, v in zip(N_LIST, mse_raw, mse_cncv, vrf_per_n):
        print(
            f"  N={n:>5d}: MSE_raw={m_r:.4e}  MSE_cncv={m_c:.4e}  "
            f"ratio={v:.4f}"
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ns = np.array(N_LIST, dtype=float)

    ax.errorbar(
        ns,
        mse_raw,
        yerr=sem_raw,
        marker="o",
        markersize=6,
        capsize=3,
        linewidth=1.6,
        color="#888888",
        label="Raw MC",
    )
    ax.errorbar(
        ns,
        mse_cncv,
        yerr=sem_cncv,
        marker="s",
        markersize=6,
        capsize=3,
        linewidth=1.6,
        color="#1f77b4",
        label="CNCV-corrected",
    )

    # 1/N reference: anchor at the raw curve's first point
    ref_const = mse_raw[0] * ns[0]
    ax.plot(
        ns,
        ref_const / ns,
        ":",
        color="#888888",
        alpha=0.7,
        linewidth=1.2,
        label=r"$\propto 1/N$ (raw MC slope)",
    )

    # 1/N reference for CNCV (parallel-slope expectation)
    ref_const_cncv = mse_cncv[0] * ns[0]
    ax.plot(
        ns,
        ref_const_cncv / ns,
        ":",
        color="#1f77b4",
        alpha=0.7,
        linewidth=1.2,
        label=r"$\propto 1/N$ (CNCV ideal)",
    )

    # Asymptotic CNCV bias floor
    ax.axhline(
        bias_floor_mean,
        linestyle="--",
        color="#d62728",
        linewidth=1.4,
        label=f"CNCV bias floor = {bias_floor_mean:.2e}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of posterior samples $N$")
    ax.set_ylabel(
        r"$\mathrm{MSE}\left(\hat{\mu}(N), \mu_{\mathrm{post}}^{\mathrm{exact}}\right)$"
    )
    ax.set_title("Gaussian $d{=}4$: MSE vs. analytical posterior mean")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    pdf_path = os.path.join(FIG_DIR, "gaussian_d4_mse_vs_truth.pdf")
    png_path = os.path.join(FIG_DIR, "gaussian_d4_mse_vs_truth.png")
    json_path = os.path.join(FIG_DIR, "gaussian_d4_mse_vs_truth.json")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Save numerical results
    results = {
        "n_dim": n_dim,
        "sigma": sigma,
        "n_test_obs": N_TEST_OBS,
        "n_pool": N_POOL,
        "n_seeds": N_SEEDS,
        "n_list": list(N_LIST),
        "mse_raw": mse_raw.tolist(),
        "mse_cncv": mse_cncv.tolist(),
        "sem_raw": sem_raw.tolist(),
        "sem_cncv": sem_cncv.tolist(),
        "vrf_per_n": vrf_per_n.tolist(),
        "bias_floor_mean": bias_floor_mean,
        "bias_floor_max": bias_floor_max,
        "asymp_floor_raw": asymp_floor_raw,
        "checkpoint": ckpt_path,
    }
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
