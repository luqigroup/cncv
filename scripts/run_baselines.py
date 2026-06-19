"""Baseline comparisons for CNCV paper.

Compares CNCV against two classical methods on Gaussian and Rosenbrock:
1. Control Functionals (Oates et al., 2017) - kernel-based Stein CV
2. Polynomial Stein CV (degree 1 and 2)

Usage:
    python scripts/run_baselines.py --experiment all --gpu 0
    python scripts/run_baselines.py --experiment gaussian --dims 2 4 8 16 --gpu 0
    python scripts/run_baselines.py --experiment rosenbrock --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv import (
    create_random_split_ensemble,
    compute_score_posterior_gaussian,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from cncv.utils import CustomLRScheduler, pSGLD


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------

def median_heuristic(X):
    """Median heuristic for RBF kernel bandwidth."""
    with torch.no_grad():
        dists = torch.cdist(X, X)
        # Use upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        median_dist = dists[mask].median().item()
    return max(median_dist, 1e-6)


def stein_kernel_matrix(X1, X2, score1, score2, sigma_kernel):
    """Compute Stein kernel matrix using RBF base kernel.

    K_0(x,x') = k(x,x') [s(x)·s(x') + (d/σ² - ||x-x'||²/σ⁴)]
              + (x'-x)·s(x)/σ² * k(x,x')
              + (x-x')·s(x')/σ² * k(x,x')

    Uses the full Stein kernel from Oates et al. (2017).
    Supports cross-kernel: X1 (test) vs X2 (train).

    Args:
        X1: [N1, d] first set of samples (rows)
        X2: [N2, d] second set of samples (columns)
        score1: [N1, d] score at X1
        score2: [N2, d] score at X2
        sigma_kernel: RBF bandwidth

    Returns:
        K_stein: [N1, N2] Stein kernel matrix
    """
    d = X1.shape[1]
    sigma2 = sigma_kernel ** 2

    # Pairwise differences: diff[i,j] = X1[i] - X2[j], shape [N1, N2, d]
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)

    # RBF kernel: k(x,x') = exp(-||x-x'||^2 / (2*sigma^2))
    sq_dists = (diff ** 2).sum(dim=-1)  # [N1, N2]
    K_rbf = torch.exp(-sq_dists / (2 * sigma2))  # [N1, N2]

    # Term 1: k(x,x') * s(x)·s(x')
    score_dot = score1 @ score2.T  # [N1, N2]
    term1 = K_rbf * score_dot

    # Term 2: k(x,x') * sum_l (-diff_l / sigma^2) * s_l(x')
    diff_dot_score_j = torch.einsum('ijd,jd->ij', diff, score2)  # [N1, N2]
    term2 = K_rbf * (-diff_dot_score_j / sigma2)

    # Term 3: k(x,x') * sum_l (diff_l / sigma^2) * s_l(x)
    diff_dot_score_i = torch.einsum('ijd,id->ij', diff, score1)  # [N1, N2]
    term3 = K_rbf * (diff_dot_score_i / sigma2)

    # Term 4: k(x,x') * [d/sigma^2 - ||x-x'||^2/sigma^4]
    term4 = K_rbf * (d / sigma2 - sq_dists / sigma2 ** 2)

    K_stein = term1 + term2 + term3 + term4
    return K_stein


def control_functionals_cv(X, score, h_x, reg_lambda=1e-4, max_n=2000):
    """Control Functionals baseline (Oates et al., 2017).

    Uses train/test split to avoid overfitting: fit alpha on train half,
    evaluate g on test half using cross-kernel.

    Args:
        X: [N, d] samples
        score: [N, d] score at each sample
        h_x: [N, n_cv] function values
        reg_lambda: regularization
        max_n: cap total sample size (O(N^2) memory)

    Returns:
        h_x_out: [M, n_cv] test-half function values
        g_values: [M, n_cv] estimated CV values on test half
    """
    N = X.shape[0]
    if N > max_n:
        idx = torch.randperm(N)[:max_n]
        X = X[idx]
        score = score[idx]
        h_x = h_x[idx]
        N = max_n

    # Train/test split to avoid overfitting
    perm = torch.randperm(N)
    n_train = N // 2
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, score_train, h_train = X[train_idx], score[train_idx], h_x[train_idx]
    X_test, score_test, h_test = X[test_idx], score[test_idx], h_x[test_idx]

    sigma_kernel = median_heuristic(X_train)

    # Fit on train data
    K_train = stein_kernel_matrix(X_train, X_train, score_train, score_train, sigma_kernel)
    K_trace = K_train.diag().mean().item()
    effective_lambda = reg_lambda * max(K_trace, 1.0)
    K_reg = K_train + effective_lambda * torch.eye(n_train, device=X.device)

    try:
        alpha = torch.linalg.solve(K_reg, h_train)
    except torch.linalg.LinAlgError:
        alpha = torch.linalg.lstsq(K_reg, h_train).solution

    # Evaluate on test data using cross-kernel
    K_cross = stein_kernel_matrix(X_test, X_train, score_test, score_train, sigma_kernel)
    g_values = K_cross @ alpha

    # Clamp extreme values for numerical stability
    g_max = 10 * h_test.abs().max().item()
    g_values = g_values.clamp(-g_max, g_max)

    return h_test, g_values


def polynomial_basis(X, score, degree=2):
    """Build polynomial Stein CV basis functions and evaluate g.

    For degree 1: phi(x) = [I; score_terms] -> g = tr(A) + (Ax+b)·score
    For degree 2: adds quadratic cross terms

    Args:
        X: [N, d]
        score: [N, d]
        degree: 1 or 2

    Returns:
        G_basis: [N, n_basis] matrix of Stein CV basis function evaluations
    """
    N, d = X.shape
    basis_list = []

    # Degree 1: Linear CV
    # For each dimension k, phi_k(x) = x_k
    # g_k = 1 + x_k * score_k  (trace of identity + phi·score for that component)
    # More general: phi(x) = A @ x + b
    # g = tr(A) + (A @ x + b) · score
    # Use individual components as basis:
    # Basis function j: phi_j(x) = e_j (unit vector in direction j)
    # g_j = score_j (derivative of phi_j w.r.t. x_j = 0, but phi_j · score = score_j... )

    # Actually, for linear Stein CV:
    # phi_k(x) = e_k => div(phi_k) = 0, phi_k · score = score_k
    # So g_k = score_k
    for k in range(d):
        basis_list.append(score[:, k:k+1])  # [N, 1]

    # phi_k(x) = x_k * e_k => div = 1, phi · score = x_k * score_k
    # g = 1 + x_k * score_k
    for k in range(d):
        basis_list.append((1.0 + X[:, k:k+1] * score[:, k:k+1]))

    # Constant: phi(x) = c => div = 0, phi·score = c·score_k
    # Already covered above

    if degree >= 2:
        # Quadratic terms: phi(x) = x_j * e_k
        # div_x(phi) = delta_{jk}
        # phi · score = x_j * score_k
        # g = delta_{jk} + x_j * score_k
        for j in range(d):
            for k in range(d):
                if j != k:  # j==k already in linear
                    g_jk = X[:, j:j+1] * score[:, k:k+1]
                    basis_list.append(g_jk)

        # phi(x) = x_j * x_k * e_l
        # div_x(phi) = delta_{jl} * x_k + delta_{kl} * x_j
        # phi · score = x_j * x_k * score_l
        # g = delta_{jl}*x_k + delta_{kl}*x_j + x_j*x_k*score_l
        for j in range(d):
            for k in range(j, d):
                for l in range(d):
                    div_term = 0.0
                    if j == l:
                        div_term = div_term + X[:, k:k+1]
                    if k == l:
                        div_term = div_term + X[:, j:j+1]
                    g_term = div_term + X[:, j:j+1] * X[:, k:k+1] * score[:, l:l+1]
                    basis_list.append(g_term)

    G_basis = torch.cat(basis_list, dim=1)  # [N, n_basis]
    return G_basis


def polynomial_cv(X, score, h_x, degree=2):
    """Polynomial Stein CV via least-squares regression.

    Args:
        X: [N, d]
        score: [N, d]
        h_x: [N, n_cv]
        degree: 1 or 2

    Returns:
        g_values: [N, n_cv] CV values
    """
    G_basis = polynomial_basis(X, score, degree=degree)  # [N, n_basis]
    n_basis = G_basis.shape[1]

    # Least squares: minimize ||h_x - G_basis @ beta||^2
    # beta = (G^T G + lambda I)^{-1} G^T h_x
    reg = 1e-6 * torch.eye(n_basis, device=X.device)
    GtG = G_basis.T @ G_basis + reg
    Gth = G_basis.T @ h_x  # [n_basis, n_cv]

    beta = torch.linalg.solve(GtG, Gth)  # [n_basis, n_cv]
    g_values = G_basis @ beta  # [N, n_cv]

    return g_values


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_vrf(h_x, g_values, true_qoi, num_trials=1000, sample_sizes=None):
    """Evaluate VRF and MSE over multiple trials.

    Args:
        h_x: [N, n_cv] function values
        g_values: [N, n_cv] CV values
        true_qoi: [n_cv] true QoI values
        num_trials: number of MC trials
        sample_sizes: list of sample sizes to evaluate

    Returns:
        dict with 'vrf_mean', 'vrf_std', 'mse_vanilla', 'mse_cv' arrays
    """
    if sample_sizes is None:
        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]

    N = h_x.shape[0]
    n_cv = h_x.shape[1]

    vanilla_mse = np.zeros((len(sample_sizes), n_cv))
    cv_mse = np.zeros((len(sample_sizes), n_cv))

    for idx, n in enumerate(sample_sizes):
        if n > N:
            n = N
        for trial in range(num_trials):
            indices = torch.randperm(N)[:n]
            for i in range(n_cv):
                vanilla_est = h_x[indices, i].mean().item()
                cv_est = (h_x[indices, i] - g_values[indices, i]).mean().item()
                vanilla_mse[idx, i] += (vanilla_est - true_qoi[i].item()) ** 2
                cv_mse[idx, i] += (cv_est - true_qoi[i].item()) ** 2

    vanilla_mse /= num_trials
    cv_mse /= num_trials

    # Average VRF across sample sizes and components
    vrf_per_comp = cv_mse.mean(axis=0) / vanilla_mse.mean(axis=0)
    vrf_mean = np.mean(vrf_per_comp)
    vrf_std = np.std(vrf_per_comp)

    return {
        'vrf_mean': vrf_mean,
        'vrf_std': vrf_std,
        'vrf_per_component': vrf_per_comp,
        'mse_vanilla': vanilla_mse,
        'mse_cv': cv_mse,
        'sample_sizes': np.array(sample_sizes),
    }


# ---------------------------------------------------------------------------
# Gaussian experiment
# ---------------------------------------------------------------------------

def run_gaussian_baselines(dim, device, num_trials=1000):
    """Run all baselines on Gaussian posterior for given dimension."""
    print(f"\n{'='*60}")
    print(f"Gaussian baselines, dim={dim}")
    print(f"{'='*60}")

    # Recreate the same problem setup as train_all_dimensions.py
    sigma = 0.3
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
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1 / sigma ** 2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)

    # Load CNCV checkpoint
    ckpt_dir = f"data/checkpoints/gaussian_dim{dim}_mean_scaling"
    cncv_ensemble = None
    if os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if ckpt_files:
            latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
            ckpt = torch.load(os.path.join(ckpt_dir, latest), map_location=device, weights_only=False)
            args_ckpt = ckpt["args"]
            cncv_ensemble = create_random_split_ensemble(
                dim, dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
                args_ckpt.n_layers, n_cv=dim, seed=12
            ).to(device)
            cncv_ensemble.load_state_dict(ckpt["ensemble_state_dict"])
            cncv_ensemble.eval()
            print(f"Loaded CNCV checkpoint: {latest}")

    # Evaluate over multiple test observations
    n_test_obs = 10
    num_samples = 10000
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]

    results = {method: [] for method in ['cncv', 'cf', 'poly1', 'poly2']}

    torch.manual_seed(123)
    for obs_idx in tqdm(range(n_test_obs), desc=f"Gaussian d={dim} test obs"):
        X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
        Y_obs = X_true + sigma * torch.randn(dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / sigma ** 2) * Y_obs)

        # Sample from exact posterior
        noise = torch.randn(num_samples, dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)
        score = compute_score_posterior_gaussian(X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma)

        h_x = X_samples  # qofi = mean
        true_qoi = mu_post

        # CNCV
        if cncv_ensemble is not None:
            with torch.no_grad():
                g_combined, _ = cncv_ensemble(X_samples, Y_samples, score)
            # g_combined is [n_cv, N], need [N, n_cv]
            g_cncv = g_combined.T
            res = evaluate_vrf(h_x, g_cncv, true_qoi, num_trials, sample_sizes)
            results['cncv'].append(res)

        # Control Functionals (cap at 2000)
        h_x_cf, g_cf = control_functionals_cv(X_samples, score, h_x, reg_lambda=1e-4, max_n=2000)
        res = evaluate_vrf(h_x_cf, g_cf, true_qoi, num_trials, sample_sizes)
        results['cf'].append(res)

        # Polynomial degree 1
        g_poly1 = polynomial_cv(X_samples, score, h_x, degree=1)
        res = evaluate_vrf(h_x, g_poly1, true_qoi, num_trials, sample_sizes)
        results['poly1'].append(res)

        # Polynomial degree 2
        g_poly2 = polynomial_cv(X_samples, score, h_x, degree=2)
        res = evaluate_vrf(h_x, g_poly2, true_qoi, num_trials, sample_sizes)
        results['poly2'].append(res)

    # Aggregate results
    aggregated = {}
    for method, obs_results in results.items():
        if not obs_results:
            continue
        vrf_means = [r['vrf_mean'] for r in obs_results]
        mse_vanilla = np.mean([r['mse_vanilla'] for r in obs_results], axis=0)
        mse_cv = np.mean([r['mse_cv'] for r in obs_results], axis=0)
        aggregated[method] = {
            'vrf_mean': np.mean(vrf_means),
            'vrf_std': np.std(vrf_means),
            'mse_vanilla': mse_vanilla,
            'mse_cv': mse_cv,
            'sample_sizes': sample_sizes,
        }
        print(f"  {method}: VRF = {np.mean(vrf_means):.4f} +/- {np.std(vrf_means):.4f}")

    return aggregated


# ---------------------------------------------------------------------------
# Rosenbrock experiment
# ---------------------------------------------------------------------------

def sample_rosenbrock_posterior(rosenbrock_dist, Y_obs, sigma, num_samples, device):
    """Sample from Rosenbrock posterior via pSGLD."""
    n_dim = rosenbrock_dist.ndim
    lr_initial = 0.1
    lr_final = 0.01
    max_itr = 2 * num_samples

    x = rosenbrock_dist.sample(1, device=device).squeeze()
    x.requires_grad_(True)

    optimizer = pSGLD([x], lr=lr_initial)
    lr_scheduler = CustomLRScheduler(optimizer, lr_initial, lr_final, max_itr)

    all_samples = []
    for itr in range(max_itr):
        neg_log_lik = 0.5 * ((Y_obs - x) ** 2).sum() / (sigma ** 2)
        neg_log_prior = -rosenbrock_dist.logpdf(x.unsqueeze(0)).squeeze()
        energy = neg_log_lik + neg_log_prior

        optimizer.zero_grad()
        energy.backward()
        lr_scheduler.step()
        optimizer.step()

        all_samples.append(x.detach().clone())

    all_samples = torch.stack(all_samples)
    burnin = max_itr // 2
    return all_samples[burnin:]


def run_rosenbrock_baselines(device, num_trials=1000):
    """Run all baselines on Rosenbrock posterior."""
    print(f"\n{'='*60}")
    print(f"Rosenbrock baselines")
    print(f"{'='*60}")

    rosenbrock_dist = RosenbrockDistribution(mu=0.0, a=0.5, b=1.0, n1=2, n2=1)
    n_dim = rosenbrock_dist.ndim
    sigma = 0.3

    # Load CNCV checkpoint
    ckpt_dir = (
        "data/checkpoints/"
        "rosenbrock_ensemble_cv_input_dim-2_max_epochs-50_lr-0.001_lr_final-0.0001"
        "_mu-0.0_a-0.5_b-1.0_sigma-0.3_batchsize-256_n_hidden-32_n_layers-3"
        "_n_ensemble_members-6_num_train-65536_num_val-2048_save_freq-50_qofi-mean_ddpm-0"
    )
    cncv_ensemble = None
    if os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if ckpt_files:
            latest = max(ckpt_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
            ckpt = torch.load(os.path.join(ckpt_dir, latest), map_location=device, weights_only=False)
            args_ckpt = ckpt["args"]
            cncv_ensemble = create_random_split_ensemble(
                n_dim, n_dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
                args_ckpt.n_layers, n_cv=n_dim, seed=1  # seed=1 for Rosenbrock
            ).to(device)
            cncv_ensemble.load_state_dict(ckpt["ensemble_state_dict"])
            cncv_ensemble.eval()
            print(f"Loaded CNCV checkpoint: {latest}")

    n_test_obs = 10
    num_samples = 10000
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000]

    results = {method: [] for method in ['cncv', 'cf', 'poly1', 'poly2']}

    torch.manual_seed(123)
    for obs_idx in tqdm(range(n_test_obs), desc="Rosenbrock test obs"):
        X_true = rosenbrock_dist.sample(1, device=device).squeeze()
        Y_obs = X_true + sigma * torch.randn(n_dim, device=device)

        # Sample posterior via pSGLD
        X_samples = sample_rosenbrock_posterior(rosenbrock_dist, Y_obs, sigma, num_samples, device)
        Y_samples = Y_obs.repeat(X_samples.shape[0], 1)
        score = compute_score_posterior_rosenbrock(X_samples, Y_samples, rosenbrock_dist, sigma)

        h_x = X_samples
        true_qoi = X_samples.mean(dim=0)  # Use empirical posterior mean

        # CNCV
        if cncv_ensemble is not None:
            with torch.no_grad():
                g_combined, _ = cncv_ensemble(X_samples, Y_samples, score)
            g_cncv = g_combined.T
            res = evaluate_vrf(h_x, g_cncv, true_qoi, num_trials, sample_sizes)
            results['cncv'].append(res)

        # Control Functionals
        h_x_cf, g_cf = control_functionals_cv(X_samples, score, h_x, reg_lambda=1e-4, max_n=2000)
        res = evaluate_vrf(h_x_cf, g_cf, true_qoi, num_trials, sample_sizes)
        results['cf'].append(res)

        # Polynomial degree 1
        g_poly1 = polynomial_cv(X_samples, score, h_x, degree=1)
        res = evaluate_vrf(h_x, g_poly1, true_qoi, num_trials, sample_sizes)
        results['poly1'].append(res)

        # Polynomial degree 2
        g_poly2 = polynomial_cv(X_samples, score, h_x, degree=2)
        res = evaluate_vrf(h_x, g_poly2, true_qoi, num_trials, sample_sizes)
        results['poly2'].append(res)

    # Aggregate
    aggregated = {}
    for method, obs_results in results.items():
        if not obs_results:
            continue
        vrf_means = [r['vrf_mean'] for r in obs_results]
        mse_vanilla = np.mean([r['mse_vanilla'] for r in obs_results], axis=0)
        mse_cv = np.mean([r['mse_cv'] for r in obs_results], axis=0)
        aggregated[method] = {
            'vrf_mean': np.mean(vrf_means),
            'vrf_std': np.std(vrf_means),
            'mse_vanilla': mse_vanilla,
            'mse_cv': mse_cv,
            'sample_sizes': sample_sizes,
        }
        print(f"  {method}: VRF = {np.mean(vrf_means):.4f} +/- {np.std(vrf_means):.4f}")

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "gaussian", "rosenbrock"])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = "data/baselines"
    os.makedirs(save_dir, exist_ok=True)

    if args.experiment in ["all", "gaussian"]:
        for dim in args.dims:
            results = run_gaussian_baselines(dim, device, args.num_trials)
            save_path = os.path.join(save_dir, f"gaussian_dim{dim}_baselines.npz")
            np.savez(save_path, **{
                f"{method}_{key}": val
                for method, method_results in results.items()
                for key, val in method_results.items()
                if isinstance(val, (np.ndarray, float, int, list))
            })
            print(f"Saved: {save_path}")

    if args.experiment in ["all", "rosenbrock"]:
        results = run_rosenbrock_baselines(device, args.num_trials)
        save_path = os.path.join(save_dir, "rosenbrock_baselines.npz")
        np.savez(save_path, **{
            f"{method}_{key}": val
            for method, method_results in results.items()
            for key, val in method_results.items()
            if isinstance(val, (np.ndarray, float, int, list))
        })
        print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
