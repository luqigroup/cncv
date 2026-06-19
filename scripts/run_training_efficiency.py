"""Training efficiency experiments for CNCV.

Part A: VRF vs training epoch (learning curve).
Part B: VRF vs training set size.

Usage:
    python scripts/run_training_efficiency.py --experiment gaussian --dim 4 --gpu 0
    python scripts/run_training_efficiency.py --experiment rosenbrock --gpu 0
    python scripts/run_training_efficiency.py --experiment all --gpu 0
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
    create_hierarchical_ensemble,
    create_shared_hierarchical_ensemble,
    compute_score_posterior_gaussian,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from cncv.utils import CustomLRScheduler, pSGLD


TRAIN_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

def evaluate_vrf_gaussian(ensemble, dim, mu_prior, Sigma_prior_inv, Sigma_post,
                          L_post, sigma, device, n_test_obs=10, num_samples=10000):
    """Evaluate VRF for Gaussian posterior mean estimation."""
    ensemble.eval()
    all_vrfs = []

    for obs_idx in range(n_test_obs):
        L_prior = torch.linalg.cholesky(
            torch.linalg.inv(Sigma_prior_inv)
        )
        X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
        Y_obs = X_true + sigma * torch.randn(dim, device=device)
        mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / sigma ** 2) * Y_obs)

        noise = torch.randn(num_samples, dim, device=device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)

        with torch.no_grad():
            score = compute_score_posterior_gaussian(X_samples, Y_samples, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_samples
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        vrfs = []
        for i in range(dim):
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            var_controlled = controlled.var().item()
            vrfs.append(var_controlled / max(var_h, 1e-12))
        all_vrfs.append(np.mean(vrfs))

    return np.mean(all_vrfs), np.std(all_vrfs)


def evaluate_vrf_rosenbrock(ensemble, rosenbrock_dist, sigma, device,
                             n_test_obs=5, num_samples=5000):
    """Evaluate VRF for Rosenbrock posterior mean estimation."""
    ensemble.eval()
    n_dim = rosenbrock_dist.ndim
    all_vrfs = []

    for obs_idx in range(n_test_obs):
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

        X_samples = torch.stack(samples)[num_samples:]
        Y_samples = Y_obs.repeat(X_samples.shape[0], 1)
        score = compute_score_posterior_rosenbrock(X_samples, Y_samples, rosenbrock_dist, sigma)

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

    return np.mean(all_vrfs), np.std(all_vrfs)


# ---------------------------------------------------------------------------
# Part A: VRF vs epoch (learning curve)
# ---------------------------------------------------------------------------

def run_epoch_curve_gaussian(dim, max_epochs, eval_freq, device, layer_type="flat"):
    """Train Gaussian CNCV and evaluate VRF every eval_freq epochs."""
    print(f"\n{'='*60}")
    print(f"Gaussian epoch curve, dim={dim}, layer_type={layer_type}")
    print(f"{'='*60}")

    # Hyperparameters (match paper config)
    n_hidden = 64
    n_layers = 3
    n_ensemble = 4
    batchsize = 2048
    num_train = 65536
    lr = 0.001
    lr_final = 0.0001
    sigma = 0.3

    # Problem setup
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

    # Create ensemble
    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            dim, dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=dim, seed=12
        ).to(device)
    elif layer_type == "hierarchical":
        ensemble = create_hierarchical_ensemble(
            dim, dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=dim, seed=12
        ).to(device)
    else:
        ensemble = create_random_split_ensemble(
            dim, dim, n_hidden, n_ensemble, n_layers, n_cv=dim, seed=12
        ).to(device)

    # Generate training data
    torch.manual_seed(0)
    noise = torch.randn(num_train, dim, dtype=torch.float32, device=device)
    X_train = mu_prior + (L_prior @ noise.T).T
    Y_train = X_train + sigma * torch.randn_like(X_train)

    # Optimizer
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    num_batches_per_epoch = num_train // batchsize
    max_step = num_batches_per_epoch * max_epochs
    lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    # Track VRF at each evaluation point
    eval_epochs = []
    vrf_means = []
    vrf_stds = []

    # Evaluate at epoch 0 (before training)
    torch.manual_seed(42)
    vrf_m, vrf_s = evaluate_vrf_gaussian(
        ensemble, dim, mu_prior, Sigma_prior_inv, Sigma_post, L_post, sigma, device
    )
    eval_epochs.append(0)
    vrf_means.append(vrf_m)
    vrf_stds.append(vrf_s)
    print(f"  Epoch 0: VRF = {vrf_m:.4f}")

    for epoch in tqdm(range(max_epochs), desc="Training"):
        ensemble.train()
        for X_batch, Y_batch in train_loader:
            score = compute_score_posterior_gaussian(X_batch, Y_batch, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_batch
            g_combined, _ = ensemble(X_batch, Y_batch, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Evaluate every eval_freq epochs
        if (epoch + 1) % eval_freq == 0 or epoch == max_epochs - 1:
            torch.manual_seed(42)
            vrf_m, vrf_s = evaluate_vrf_gaussian(
                ensemble, dim, mu_prior, Sigma_prior_inv, Sigma_post, L_post, sigma, device
            )
            eval_epochs.append(epoch + 1)
            vrf_means.append(vrf_m)
            vrf_stds.append(vrf_s)
            print(f"  Epoch {epoch+1}: VRF = {vrf_m:.4f}")

    return np.array(eval_epochs), np.array(vrf_means), np.array(vrf_stds)


def run_epoch_curve_rosenbrock(max_epochs, eval_freq, device, layer_type="flat"):
    """Train Rosenbrock CNCV and evaluate VRF every eval_freq epochs."""
    print(f"\n{'='*60}")
    print(f"Rosenbrock epoch curve, layer_type={layer_type}")
    print(f"{'='*60}")

    n_hidden = 64
    n_layers = 3
    n_ensemble = 4
    batchsize = 2048
    num_train = 65536
    lr = 0.001
    lr_final = 0.0001
    sigma = 0.3

    rosenbrock_dist = RosenbrockDistribution(mu=0.0, a=0.5, b=1.0, n1=2, n2=1)
    n_dim = rosenbrock_dist.ndim

    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            n_dim, n_dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=n_dim, seed=12
        ).to(device)
    elif layer_type == "hierarchical":
        ensemble = create_hierarchical_ensemble(
            n_dim, n_dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=n_dim, seed=12
        ).to(device)
    else:
        ensemble = create_random_split_ensemble(
            n_dim, n_dim, n_hidden, n_ensemble, n_layers, n_cv=n_dim, seed=12
        ).to(device)

    torch.manual_seed(0)
    X_train = rosenbrock_dist.sample(num_train, device=device)
    Y_train = X_train + sigma * torch.randn_like(X_train)

    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    num_batches_per_epoch = num_train // batchsize
    max_step = num_batches_per_epoch * max_epochs
    lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    eval_epochs = []
    vrf_means = []
    vrf_stds = []

    # Evaluate at epoch 0
    torch.manual_seed(42)
    vrf_m, vrf_s = evaluate_vrf_rosenbrock(ensemble, rosenbrock_dist, sigma, device)
    eval_epochs.append(0)
    vrf_means.append(vrf_m)
    vrf_stds.append(vrf_s)
    print(f"  Epoch 0: VRF = {vrf_m:.4f}")

    for epoch in tqdm(range(max_epochs), desc="Training"):
        ensemble.train()
        for X_batch, Y_batch in train_loader:
            score = compute_score_posterior_rosenbrock(X_batch, Y_batch, rosenbrock_dist, sigma)
            h_x = X_batch
            g_combined, _ = ensemble(X_batch, Y_batch, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

        if (epoch + 1) % eval_freq == 0 or epoch == max_epochs - 1:
            torch.manual_seed(42)
            vrf_m, vrf_s = evaluate_vrf_rosenbrock(ensemble, rosenbrock_dist, sigma, device)
            eval_epochs.append(epoch + 1)
            vrf_means.append(vrf_m)
            vrf_stds.append(vrf_s)
            print(f"  Epoch {epoch+1}: VRF = {vrf_m:.4f}")

    return np.array(eval_epochs), np.array(vrf_means), np.array(vrf_stds)


# ---------------------------------------------------------------------------
# Part B: VRF vs training set size
# ---------------------------------------------------------------------------

def run_size_curve_gaussian(dim, max_epochs, device, layer_type="flat"):
    """Train Gaussian CNCV with different training set sizes."""
    print(f"\n{'='*60}")
    print(f"Gaussian size curve, dim={dim}, layer_type={layer_type}")
    print(f"{'='*60}")

    n_hidden = 64
    n_layers = 3
    n_ensemble = 4
    batchsize = 2048
    lr = 0.001
    lr_final = 0.0001
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

    # Generate the largest training set once (use subsets)
    max_num_train = max(TRAIN_SIZES)
    torch.manual_seed(0)
    noise_full = torch.randn(max_num_train, dim, dtype=torch.float32, device=device)
    X_train_full = mu_prior + (L_prior @ noise_full.T).T
    Y_train_full = X_train_full + sigma * torch.randn_like(X_train_full)

    vrf_means = []
    vrf_stds = []

    for num_train in TRAIN_SIZES:
        print(f"\n  N_train = {num_train}")

        X_train = X_train_full[:num_train]
        Y_train = Y_train_full[:num_train]

        # Fresh ensemble for each size
        if layer_type == "shared_hierarchical":
            ensemble = create_shared_hierarchical_ensemble(
                dim, dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=dim, seed=12
            ).to(device)
        elif layer_type == "hierarchical":
            ensemble = create_hierarchical_ensemble(
                dim, dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=dim, seed=12
            ).to(device)
        else:
            ensemble = create_random_split_ensemble(
                dim, dim, n_hidden, n_ensemble, n_layers, n_cv=dim, seed=12
            ).to(device)

        bs = min(batchsize, num_train)
        optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
        num_batches_per_epoch = max(1, num_train // bs)
        max_step = num_batches_per_epoch * max_epochs
        lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

        for epoch in tqdm(range(max_epochs), desc=f"N={num_train}", leave=False):
            ensemble.train()
            for X_batch, Y_batch in train_loader:
                score = compute_score_posterior_gaussian(X_batch, Y_batch, mu_prior, Sigma_prior_inv, sigma)
                h_x = X_batch
                g_combined, _ = ensemble(X_batch, Y_batch, score)
                residual = h_x.T - g_combined
                loss = (residual ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

        # Evaluate
        torch.manual_seed(42)
        vrf_m, vrf_s = evaluate_vrf_gaussian(
            ensemble, dim, mu_prior, Sigma_prior_inv, Sigma_post, L_post, sigma, device
        )
        vrf_means.append(vrf_m)
        vrf_stds.append(vrf_s)
        print(f"    VRF = {vrf_m:.4f} +/- {vrf_s:.4f}")

    return np.array(TRAIN_SIZES), np.array(vrf_means), np.array(vrf_stds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training efficiency experiments")
    parser.add_argument("--experiment", type=str, default="gaussian",
                        choices=["gaussian", "rosenbrock", "all"])
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="Evaluate VRF every N epochs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer_type", type=str, default="flat",
                        choices=["flat", "hierarchical", "shared_hierarchical"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = "data/training_efficiency"
    os.makedirs(save_dir, exist_ok=True)

    suffix = f"_{args.layer_type}" if args.layer_type != "flat" else ""

    if args.experiment in ["all", "gaussian"]:
        # Part A: Epoch curve
        epochs, vrf_m, vrf_s = run_epoch_curve_gaussian(
            args.dim, args.epochs, args.eval_freq, device, layer_type=args.layer_type
        )
        save_path = os.path.join(save_dir, f"gaussian_dim{args.dim}_epoch_curve{suffix}.npz")
        np.savez(save_path, epochs=epochs, vrf_means=vrf_m, vrf_stds=vrf_s, dim=args.dim)
        print(f"Saved: {save_path}")

        # Part B: Size curve
        sizes, vrf_m, vrf_s = run_size_curve_gaussian(
            args.dim, args.epochs, device, layer_type=args.layer_type
        )
        save_path = os.path.join(save_dir, f"gaussian_dim{args.dim}_size_curve{suffix}.npz")
        np.savez(save_path, train_sizes=sizes, vrf_means=vrf_m, vrf_stds=vrf_s, dim=args.dim)
        print(f"Saved: {save_path}")

    if args.experiment in ["all", "rosenbrock"]:
        # Part A: Epoch curve only (size curve very expensive for Rosenbrock MCMC eval)
        rosenbrock_epochs = min(args.epochs, 50)
        epochs, vrf_m, vrf_s = run_epoch_curve_rosenbrock(
            rosenbrock_epochs, args.eval_freq, device, layer_type=args.layer_type
        )
        save_path = os.path.join(save_dir, f"rosenbrock_epoch_curve{suffix}.npz")
        np.savez(save_path, epochs=epochs, vrf_means=vrf_m, vrf_stds=vrf_s)
        print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
