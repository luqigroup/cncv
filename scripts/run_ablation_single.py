"""Single-job ablation: train one L config and save VRF.

This script trains a hierarchical ensemble with L members, evaluates VRF
over multiple test observations, and saves results to an .npz file.

Designed to be launched in parallel by launch_ablation_sweep.sh.

Usage:
    python scripts/run_ablation_single.py --experiment gaussian --L 16 --gpu 0
    python scripts/run_ablation_single.py --experiment rosenbrock --L 8 --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv import (
    create_shared_hierarchical_ensemble,
    compute_score_posterior_gaussian,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from cncv.utils import CustomLRScheduler, pSGLD


def train_and_evaluate_gaussian(dim, n_ensemble, max_epochs, device):
    """Train CNCV with given L on Gaussian and evaluate VRF."""
    n_hidden = 64
    n_layers = 3
    batchsize = 2048
    num_train = 65536
    lr = 0.001
    lr_final = 0.0001
    sigma = 0.3

    # Problem setup (matching gaussian_ensemble_cv.py)
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

    # Create ensemble
    ensemble = create_shared_hierarchical_ensemble(
        dim, dim, n_hidden, n_ensemble, n_layers=n_layers,
        n_cv=dim, seed=12,
    ).to(device)

    num_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"  L={n_ensemble}: {num_params:,} parameters")

    # Generate training data (qofi=mean)
    torch.manual_seed(0)
    noise = torch.randn(num_train, dim, dtype=torch.float32, device=device)
    X_train = mu_prior + (L_prior @ noise.T).T
    Y_train = X_train + sigma * torch.randn_like(X_train)

    # Train
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    num_batches_per_epoch = num_train // batchsize
    max_step = num_batches_per_epoch * max_epochs
    lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True)

    for epoch in tqdm(range(max_epochs), desc=f"L={n_ensemble}", leave=False):
        ensemble.train()
        for X_batch, Y_batch in train_loader:
            score = compute_score_posterior_gaussian(
                X_batch, Y_batch, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_batch
            g_combined, _ = ensemble(X_batch, Y_batch, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

    # Evaluate over multiple test observations
    ensemble.eval()
    I = torch.eye(dim, device=device)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1 / sigma ** 2) * I)
    L_post = torch.linalg.cholesky(Sigma_post)

    n_test_obs = 10
    num_samples = 10000
    all_vrfs = []

    torch.manual_seed(42)
    for obs_idx in range(n_test_obs):
        X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
        Y_obs = X_true + sigma * torch.randn(dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / sigma ** 2) * Y_obs)

        noise = torch.randn(num_samples, dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)

        with torch.no_grad():
            score = compute_score_posterior_gaussian(
                X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_samples
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        vrfs = []
        for i in range(dim):
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            var_controlled = controlled.var().item()
            vrfs.append(var_controlled / max(var_h, 1e-12))
        all_vrfs.append(np.mean(vrfs))

    return np.mean(all_vrfs), np.std(all_vrfs), all_vrfs, num_params


def train_and_evaluate_rosenbrock(n_ensemble, max_epochs, device):
    """Train CNCV with given L on Rosenbrock and evaluate VRF."""
    n_hidden = 64
    n_layers = 3
    batchsize = 2048
    num_train = 65536
    lr = 0.001
    lr_final = 0.0001
    sigma = 0.3

    rosenbrock_dist = RosenbrockDistribution(mu=0.0, a=0.5, b=1.0, n1=2, n2=1)
    n_dim = rosenbrock_dist.ndim

    # Create ensemble
    ensemble = create_shared_hierarchical_ensemble(
        n_dim, n_dim, n_hidden, n_ensemble, n_layers=n_layers,
        n_cv=n_dim, seed=1,
    ).to(device)

    num_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"  L={n_ensemble}: {num_params:,} parameters")

    # Generate training data (qofi=mean)
    torch.manual_seed(0)
    X_train = rosenbrock_dist.sample(num_train, device=device)
    Y_train = X_train + sigma * torch.randn_like(X_train)

    # Train
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    num_batches_per_epoch = num_train // batchsize
    max_step = num_batches_per_epoch * max_epochs
    lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True)

    for epoch in tqdm(range(max_epochs), desc=f"L={n_ensemble}", leave=False):
        ensemble.train()
        for X_batch, Y_batch in train_loader:
            score = compute_score_posterior_rosenbrock(
                X_batch, Y_batch, rosenbrock_dist, sigma)
            h_x = X_batch
            g_combined, _ = ensemble(X_batch, Y_batch, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

    # Evaluate via pSGLD posterior samples
    ensemble.eval()
    n_test_obs = 10
    num_samples = 10000
    all_vrfs = []

    torch.manual_seed(42)
    for obs_idx in tqdm(range(n_test_obs), desc="Eval", leave=False):
        X_true = rosenbrock_dist.sample(1, device=device).squeeze()
        Y_obs = X_true + sigma * torch.randn(n_dim, device=device)

        # pSGLD sampling
        x = rosenbrock_dist.sample(1, device=device).squeeze()
        x.requires_grad_(True)
        mcmc_opt = pSGLD([x], lr=0.1)
        mcmc_lr = CustomLRScheduler(mcmc_opt, 0.1, 0.01, 2 * num_samples)

        samples = []
        for itr in range(2 * num_samples):
            neg_ll = 0.5 * ((Y_obs - x) ** 2).sum() / (sigma ** 2)
            neg_lp = -rosenbrock_dist.logpdf(x.unsqueeze(0)).squeeze()
            energy = neg_ll + neg_lp
            mcmc_opt.zero_grad()
            energy.backward()
            mcmc_lr.step()
            mcmc_opt.step()
            samples.append(x.detach().clone())

        X_samples = torch.stack(samples)[num_samples:]  # discard burnin
        Y_samples = Y_obs.repeat(X_samples.shape[0], 1)
        score = compute_score_posterior_rosenbrock(
            X_samples, Y_samples, rosenbrock_dist, sigma)

        with torch.no_grad():
            h_x = X_samples
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        vrfs = []
        for i in range(n_dim):
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            var_controlled = controlled.var().item()
            vrfs.append(var_controlled / max(var_h, 1e-12))
        all_vrfs.append(np.mean(vrfs))

    return np.mean(all_vrfs), np.std(all_vrfs), all_vrfs, num_params


def main():
    parser = argparse.ArgumentParser(description="Single-job ablation over L")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["gaussian", "rosenbrock"])
    parser.add_argument("--L", type=int, required=True, help="Number of ensemble members")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--dim", type=int, default=4, help="Dimension (Gaussian only)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = "data/ablations"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Experiment: {args.experiment}, L={args.L}")

    if args.experiment == "gaussian":
        mean, std, vrfs, num_params = train_and_evaluate_gaussian(
            args.dim, args.L, args.epochs, device)
    else:
        mean, std, vrfs, num_params = train_and_evaluate_rosenbrock(
            args.L, args.epochs, device)

    print(f"  VRF = {mean:.4f} +/- {std:.4f}  ({num_params:,} params)")

    save_path = os.path.join(
        save_dir,
        f"heatmap_{args.experiment}_L{args.L}.npz")
    np.savez(save_path,
             vrf_mean=mean, vrf_std=std, all_vrfs=np.array(vrfs),
             L=args.L, num_params=num_params)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
