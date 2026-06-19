"""Nonlinear forward model experiment + MCMC diagnostics.

Part A: F(x) = Ax + sin(x), Gaussian prior, pSGLD posterior sampling.
Part B: MCMC diagnostics (R-hat, ESS, trace plots) for Rosenbrock + nonlinear.

Usage:
    python scripts/run_nonlinear_forward.py --dim 4 --phase train --gpu 0
    python scripts/run_nonlinear_forward.py --dim 4 --phase eval --gpu 0
    python scripts/run_nonlinear_forward.py --dim 4 --phase mcmc_diag --gpu 0
    python scripts/run_nonlinear_forward.py --dim 4 --phase all --gpu 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cncv import create_random_split_ensemble, create_hierarchical_ensemble, create_shared_hierarchical_ensemble, get_split_config
from cncv.utils import CustomLRScheduler, pSGLD
from cncv.utils.nonlinear import (
    NonlinearForwardModel,
    compute_score_nonlinear,
    setup_nonlinear_problem,
)
from cncv import RosenbrockDistribution, compute_score_posterior_rosenbrock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_map_estimate(forward_model, Y_obs, mu_prior, Sigma_prior_inv, sigma, device,
                      n_steps=500, lr=0.01):
    """Find MAP estimate via gradient descent on posterior energy."""
    dim = mu_prior.shape[0]
    x = mu_prior.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(n_steps):
        F_x = forward_model.forward(x.unsqueeze(0)).squeeze(0)
        neg_log_lik = 0.5 * ((Y_obs - F_x) ** 2).sum() / (sigma ** 2)
        neg_log_prior = 0.5 * ((x - mu_prior) @ Sigma_prior_inv @ (x - mu_prior))
        energy = neg_log_lik + neg_log_prior
        opt.zero_grad()
        energy.backward()
        opt.step()
    return x.detach()


def sample_posterior_nonlinear(forward_model, Y_obs, mu_prior, Sigma_prior, sigma,
                               num_samples, device, rosenbrock_dist=None):
    """Sample posterior via pSGLD for nonlinear model."""
    dim = mu_prior.shape[0]
    lr_initial = 0.005
    lr_final = 0.0005
    max_itr = 20 * num_samples  # Many iterations for nonlinear model

    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)

    # Initialize from MAP estimate for better convergence
    x = find_map_estimate(forward_model, Y_obs, mu_prior, Sigma_prior_inv, sigma, device)
    x = x.detach().requires_grad_(True)

    optimizer = pSGLD([x], lr=lr_initial)
    lr_scheduler = CustomLRScheduler(optimizer, lr_initial, lr_final, max_itr)

    all_samples = []
    for itr in range(max_itr):
        F_x = forward_model.forward(x.unsqueeze(0)).squeeze(0)
        neg_log_lik = 0.5 * ((Y_obs - F_x) ** 2).sum() / (sigma ** 2)
        neg_log_prior = 0.5 * ((x - mu_prior) @ Sigma_prior_inv @ (x - mu_prior))
        energy = neg_log_lik + neg_log_prior

        optimizer.zero_grad()
        energy.backward()
        lr_scheduler.step()
        optimizer.step()

        all_samples.append(x.detach().clone())

    all_samples = torch.stack(all_samples)
    # Discard first 50% as burnin
    burnin = max_itr // 2
    # Thin by factor of 2 to reduce autocorrelation
    return all_samples[burnin::2]


def sample_posterior_nonlinear_chain(forward_model, Y_obs, mu_prior, Sigma_prior, sigma,
                                     num_samples, device, chain_seed=0):
    """Single MCMC chain with different initialization seed."""
    torch.manual_seed(chain_seed)
    return sample_posterior_nonlinear(forward_model, Y_obs, mu_prior, Sigma_prior,
                                      sigma, num_samples, device)


# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------

def compute_rhat(chains):
    """Gelman-Rubin R-hat per parameter.

    Args:
        chains: list of [n_samples, dim] arrays (one per chain)

    Returns:
        rhat: [dim] array
    """
    m = len(chains)
    n = min(c.shape[0] for c in chains)
    d = chains[0].shape[1]

    # Truncate to same length
    chains_arr = np.stack([c[:n].cpu().numpy() if torch.is_tensor(c) else c[:n]
                           for c in chains])  # [m, n, d]

    # Chain means
    chain_means = chains_arr.mean(axis=1)  # [m, d]
    overall_mean = chain_means.mean(axis=0)  # [d]

    # Between-chain variance
    B = n * np.var(chain_means, axis=0, ddof=1)  # [d]

    # Within-chain variance
    W = np.mean([np.var(chains_arr[j], axis=0, ddof=1) for j in range(m)], axis=0)  # [d]

    # R-hat
    var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-12))

    return rhat


def compute_ess(chain):
    """Effective sample size via autocorrelation.

    Args:
        chain: [n_samples, dim] tensor or array

    Returns:
        ess: [dim] array
    """
    if torch.is_tensor(chain):
        chain = chain.cpu().numpy()

    n, d = chain.shape
    ess = np.zeros(d)

    for j in range(d):
        x = chain[:, j]
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-12:
            ess[j] = 1.0
            continue

        # Compute autocorrelation using FFT
        f = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(f * np.conj(f)).real[:n]
        acf = acf / acf[0]

        # Sum pairs of autocorrelations (Geyer's initial monotone sequence)
        tau = 1.0
        for k in range(1, n // 2):
            rho_pair = acf[2 * k - 1] + acf[2 * k]
            if rho_pair < 0:
                break
            tau += 2 * rho_pair

        ess[j] = n / tau

    return ess


def plot_mcmc_diagnostics(chains, rhat, ess, save_dir, prefix=""):
    """Generate MCMC diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)
    m = len(chains)
    d = chains[0].shape[1]

    # --- Trace plots ---
    fig, axes = plt.subplots(d, 1, figsize=(6.75, 1.5 * d), sharex=True)
    if d == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, m))

    for j in range(d):
        ax = axes[j]
        for c_idx, chain in enumerate(chains):
            vals = chain[:, j].cpu().numpy() if torch.is_tensor(chain) else chain[:, j]
            ax.plot(vals, alpha=0.6, linewidth=0.5, color=colors[c_idx],
                    label=f"Chain {c_idx+1}")
        ax.set_ylabel(f"$x_{{{j+1}}}$")
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        if j == 0:
            ax.legend(fontsize=6, ncol=m, loc='upper right')
    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"{prefix}trace_plots.{ext}"), dpi=300)
    plt.close()

    # --- R-hat bar plot ---
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    x_pos = np.arange(d)
    bars = ax.bar(x_pos, rhat, color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.axhline(1.1, color='red', linestyle='--', linewidth=1, label='$\\hat{R}=1.1$')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$x_{{{j+1}}}$" for j in range(d)])
    ax.set_ylabel("$\\hat{R}$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"{prefix}rhat.{ext}"), dpi=300)
    plt.close()

    # --- ESS bar plot ---
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    bars = ax.bar(x_pos, ess, color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax.axhline(100, color='red', linestyle='--', linewidth=1, label='ESS=100')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$x_{{{j+1}}}$" for j in range(d)])
    ax.set_ylabel("ESS")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(save_dir, f"{prefix}ess.{ext}"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_nonlinear(dim, max_epochs, device, layer_type="flat"):
    """Train CNCV on nonlinear forward model.

    Args:
        dim: Problem dimension
        max_epochs: Number of training epochs
        device: Computation device
        layer_type: 'flat' for random split ensemble, 'hierarchical' for HINT-style
    """
    print(f"\n{'='*60}")
    print(f"Training CNCV on nonlinear forward model, dim={dim}, layer_type={layer_type}")
    print(f"{'='*60}")

    # Hyperparameters (match Gaussian/Rosenbrock experiments)
    n_hidden = 64
    n_layers = 3
    n_ensemble = 16
    batchsize = 2048
    num_train = 65536
    num_val = 2048
    lr = 0.001
    lr_final = 0.0001
    sigma = 0.3

    # Problem setup
    problem = setup_nonlinear_problem(dim, sigma=sigma, cond_number=2.0, device=device)
    mu_prior = problem["mu_prior"]
    Sigma_prior = problem["Sigma_prior"]
    L_prior = problem["L_prior"]
    Sigma_prior_inv = problem["Sigma_prior_inv"]
    forward_model = problem["forward_model"]

    # Create ensemble
    if layer_type == "shared_hierarchical":
        ensemble = create_shared_hierarchical_ensemble(
            dim, dim, n_hidden, n_ensemble, n_layers=n_layers, n_cv=dim, seed=12
        ).to(device)
    elif layer_type == "hierarchical":
        # Use reduced hidden size + depth=1 to match flat param count
        hier_hidden = 64
        ensemble = create_hierarchical_ensemble(
            dim, dim, hier_hidden, n_ensemble, n_layers=n_layers, depth=1, n_cv=dim, seed=12
        ).to(device)
    else:
        ensemble = create_random_split_ensemble(
            dim, dim, n_hidden, n_ensemble, n_layers, n_cv=dim, seed=12
        ).to(device)
    print(f"Parameters: {sum(p.numel() for p in ensemble.parameters() if p.requires_grad):,}")

    # Generate training data
    # Sample x from prior, compute y = F(x) + noise
    torch.manual_seed(0)
    noise_x = torch.randn(num_train, dim, dtype=torch.float32, device=device)
    X_train = mu_prior + (L_prior @ noise_x.T).T
    Y_train = forward_model.forward(X_train) + sigma * torch.randn(num_train, dim, device=device)

    noise_xv = torch.randn(num_val, dim, dtype=torch.float32, device=device)
    X_val = mu_prior + (L_prior @ noise_xv.T).T
    Y_val = forward_model.forward(X_val) + sigma * torch.randn(num_val, dim, device=device)

    # Train
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=lr)
    num_batches_per_epoch = num_train // batchsize
    max_step = num_batches_per_epoch * max_epochs
    lr_scheduler = CustomLRScheduler(optimizer, lr, lr_final, max_step)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    train_obj, val_obj = [], []

    for epoch in tqdm(range(max_epochs), desc="Training"):
        # Validation
        ensemble.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, Y_val), batch_size=batchsize
            ):
                score = compute_score_nonlinear(X_b, Y_b, forward_model, mu_prior, Sigma_prior_inv, sigma)
                h_x = X_b
                g_combined, _ = ensemble(X_b, Y_b, score)
                residual = h_x.T - g_combined
                val_loss += (residual ** 2).mean().item()
        val_loss /= max(1, num_val // batchsize)
        val_obj.append(val_loss)

        # Training
        ensemble.train()
        for X_b, Y_b in train_loader:
            score = compute_score_nonlinear(X_b, Y_b, forward_model, mu_prior, Sigma_prior_inv, sigma)
            h_x = X_b
            g_combined, _ = ensemble(X_b, Y_b, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            train_obj.append(loss.item())

    # Save checkpoint
    save_dir = "data/nonlinear"
    os.makedirs(save_dir, exist_ok=True)

    # Get split config only for flat layers (hierarchical/shared don't use splits)
    split_config = get_split_config(ensemble) if layer_type == "flat" else None

    checkpoint = {
        "ensemble_state_dict": ensemble.state_dict(),
        "epoch": max_epochs - 1,
        "train_obj": train_obj,
        "val_obj": val_obj,
        "mu_prior": mu_prior.cpu(),
        "Sigma_prior": Sigma_prior.cpu(),
        "sigma": sigma,
        "input_dim": dim,
        "split_config": split_config,
        "forward_model_A": forward_model.A.cpu(),
        "layer_type": layer_type,
        "args": argparse.Namespace(
            n_hidden=n_hidden,
            n_layers=n_layers, n_ensemble_members=n_ensemble,
            depth=1 if layer_type == "hierarchical" else None,
        ),
    }
    suffix = f"_{layer_type}" if layer_type != "flat" else ""
    save_path = os.path.join(save_dir, f"nonlinear_dim{dim}{suffix}_checkpoint.pth")
    torch.save(checkpoint, save_path)
    print(f"Saved: {save_path}")

    return ensemble, forward_model, mu_prior, Sigma_prior, Sigma_prior_inv, L_prior


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_nonlinear(dim, device, layer_type="flat", use_ddpm=False, ddpm_checkpoint=None):
    """Evaluate CNCV on nonlinear forward model.

    Args:
        dim: Problem dimension
        device: Computation device
        layer_type: 'flat' or 'hierarchical'
        use_ddpm: If True, use DDPM for posterior sampling instead of pSGLD
        ddpm_checkpoint: Path to trained DDPM checkpoint (required if use_ddpm=True)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating nonlinear forward model, dim={dim}, layer_type={layer_type}")
    print(f"Posterior sampling: {'DDPM' if use_ddpm else 'pSGLD'}")
    print(f"{'='*60}")

    save_dir = "data/nonlinear"
    suffix = f"_{layer_type}" if layer_type != "flat" else ""
    ckpt_path = os.path.join(save_dir, f"nonlinear_dim{dim}{suffix}_checkpoint.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mu_prior = ckpt["mu_prior"].to(device)
    Sigma_prior = ckpt["Sigma_prior"].to(device)
    sigma = ckpt["sigma"]
    args_ckpt = ckpt["args"]
    ckpt_layer_type = ckpt.get("layer_type", "flat")

    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    L_prior = torch.linalg.cholesky(Sigma_prior)

    forward_model = NonlinearForwardModel(dim, cond_number=2.0, seed=42, device=device)
    forward_model.A = ckpt["forward_model_A"].to(device)

    # Reconstruct ensemble
    if ckpt_layer_type == "shared_hierarchical":
        ckpt_depth = getattr(args_ckpt, "depth", None) or None
        ensemble = create_shared_hierarchical_ensemble(
            dim, dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
            n_layers=args_ckpt.n_layers, depth=ckpt_depth, n_cv=dim, seed=12,
        ).to(device)
    elif ckpt_layer_type == "hierarchical":
        ckpt_depth = getattr(args_ckpt, "depth", None) or None
        ensemble = create_hierarchical_ensemble(
            dim, dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
            n_layers=args_ckpt.n_layers, depth=ckpt_depth, n_cv=dim, seed=12
        ).to(device)
    else:
        ensemble = create_random_split_ensemble(
            dim, dim, args_ckpt.n_hidden, args_ckpt.n_ensemble_members,
            args_ckpt.n_layers, n_cv=dim, seed=12
        ).to(device)
    ensemble.load_state_dict(ckpt["ensemble_state_dict"])
    ensemble.eval()

    # Load DDPM if requested
    ddpm_model = None
    ddpm_noise_scheduler = None
    if use_ddpm:
        if ddpm_checkpoint is None:
            raise ValueError("ddpm_checkpoint path required when use_ddpm=True")
        from cncv.models.ddpm_mlp import ConditionalScoreMLPSimple
        from cncv.diffusion import NoiseScheduler

        ddpm_ckpt = torch.load(ddpm_checkpoint, map_location=device, weights_only=False)
        ddpm_model = ConditionalScoreMLPSimple(
            x_dim=dim, y_dim=dim,
            hidden_dim=512, n_layers=5, time_emb_dim=64,
            nt=ddpm_ckpt.get("nt", 1000),
        ).to(device)
        ddpm_model.load_state_dict(ddpm_ckpt["model_state_dict"])
        ddpm_model.eval()

        ddpm_noise_scheduler = NoiseScheduler(
            nt=ddpm_ckpt.get("nt", 1000),
            beta_start=0.0001, beta_end=0.02,
            beta_schedule=ddpm_ckpt.get("beta_schedule", "linear"),
            device=device,
        )
        print(f"Loaded DDPM from {ddpm_checkpoint}")

    # Evaluate over test observations
    n_test_obs = 10
    num_samples = 10000
    all_vrfs = []

    torch.manual_seed(99)
    for obs_idx in tqdm(range(n_test_obs), desc="Test obs"):
        X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
        Y_obs = forward_model.forward(X_true.unsqueeze(0)).squeeze(0) + sigma * torch.randn(dim, device=device)

        # Sample posterior
        if use_ddpm:
            # DDPM sampling
            x = torch.randn(num_samples, dim, device=device)
            Y_batch = Y_obs.unsqueeze(0).repeat(num_samples, 1)
            with torch.no_grad():
                for t in reversed(range(ddpm_noise_scheduler.nt)):
                    t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                    noise_pred = ddpm_model(x, Y_batch, t_batch)
                    x = ddpm_noise_scheduler.step(noise_pred, t, x)
            X_samples = x
        else:
            # pSGLD sampling
            X_samples = sample_posterior_nonlinear(
                forward_model, Y_obs, mu_prior, Sigma_prior, sigma, num_samples, device
            )

        Y_samples = Y_obs.repeat(X_samples.shape[0], 1)
        # Always use physics-based score (exact, regardless of sampler)
        score = compute_score_nonlinear(X_samples, Y_samples, forward_model, mu_prior, Sigma_prior_inv, sigma)

        with torch.no_grad():
            h_x = X_samples
            g_combined, _ = ensemble(X_samples, Y_samples, score)

        vrfs = []
        for i in range(dim):
            var_h = h_x[:, i].var().item()
            controlled = h_x[:, i] - g_combined[i, :]
            var_controlled = controlled.var().item()
            vrfs.append(var_controlled / max(var_h, 1e-12))
        all_vrfs.append(vrfs)

        print(f"  obs {obs_idx}: VRF = {np.mean(vrfs):.4f}")

    all_vrfs = np.array(all_vrfs)
    vrf_mean = all_vrfs.mean()
    vrf_std = all_vrfs.std()
    print(f"\nOverall VRF: {vrf_mean:.4f} +/- {vrf_std:.4f}")

    sampler_str = "ddpm" if use_ddpm else "mcmc"
    np.savez(os.path.join(save_dir, f"nonlinear_dim{dim}{suffix}_{sampler_str}_results.npz"),
             all_vrfs=all_vrfs,
             vrf_mean=vrf_mean,
             vrf_std=vrf_std,
             dim=dim,
             layer_type=layer_type,
             use_ddpm=use_ddpm)
    print(f"Saved results.")


# ---------------------------------------------------------------------------
# MCMC diagnostics
# ---------------------------------------------------------------------------

def run_mcmc_diagnostics(dim, device):
    """Run MCMC diagnostics for both nonlinear and Rosenbrock."""
    save_dir = "data/nonlinear"
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    num_samples = 10000
    n_chains = 4

    # --- Nonlinear model diagnostics ---
    print(f"\n{'='*60}")
    print(f"MCMC diagnostics: Nonlinear forward model, dim={dim}")
    print(f"{'='*60}")

    mu_prior = torch.zeros(dim, dtype=torch.float32, device=device)
    torch.manual_seed(42)
    diag_values = torch.linspace(1.0, 2.0, dim)
    D_diag = torch.diag(diag_values).to(device=device, dtype=torch.float32)
    if dim > 1:
        A = torch.randn(dim, dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        Sigma_prior = Q @ D_diag @ Q.T
    else:
        Sigma_prior = D_diag

    forward_model = NonlinearForwardModel(dim, cond_number=2.0, seed=42, device=device)

    # Generate observation
    L_prior = torch.linalg.cholesky(Sigma_prior)
    torch.manual_seed(99)
    X_true = mu_prior + L_prior @ torch.randn(dim, device=device)
    Y_obs = forward_model.forward(X_true.unsqueeze(0)).squeeze(0) + 0.3 * torch.randn(dim, device=device)

    # Run parallel chains
    nonlinear_chains = []
    for c in tqdm(range(n_chains), desc="Nonlinear chains"):
        chain = sample_posterior_nonlinear_chain(
            forward_model, Y_obs, mu_prior, Sigma_prior, 0.3, num_samples, device,
            chain_seed=c * 1000
        )
        nonlinear_chains.append(chain)

    nl_rhat = compute_rhat(nonlinear_chains)
    # Use the longest chain for ESS
    nl_ess = compute_ess(nonlinear_chains[0])

    print(f"Nonlinear R-hat: {nl_rhat}")
    print(f"Nonlinear ESS:   {nl_ess}")
    print(f"R-hat < 1.1: {np.all(nl_rhat < 1.1)}")
    print(f"ESS > 100:   {np.all(nl_ess > 100)}")

    plot_mcmc_diagnostics(nonlinear_chains, nl_rhat, nl_ess, plots_dir, prefix="nonlinear_")

    # --- Rosenbrock diagnostics ---
    print(f"\n{'='*60}")
    print(f"MCMC diagnostics: Rosenbrock")
    print(f"{'='*60}")

    rosenbrock_dist = RosenbrockDistribution(mu=0.0, a=0.5, b=1.0, n1=2, n2=1)
    n_dim_r = rosenbrock_dist.ndim
    sigma_r = 0.3

    torch.manual_seed(99)
    X_true_r = rosenbrock_dist.sample(1, device=device).squeeze()
    Y_obs_r = X_true_r + sigma_r * torch.randn(n_dim_r, device=device)

    rosenbrock_chains = []
    for c in tqdm(range(n_chains), desc="Rosenbrock chains"):
        torch.manual_seed(c * 1000)
        x = rosenbrock_dist.sample(1, device=device).squeeze()
        x.requires_grad_(True)
        mcmc_opt = pSGLD([x], lr=0.1)
        mcmc_lr = CustomLRScheduler(mcmc_opt, 0.1, 0.01, 2 * num_samples)

        samples = []
        for itr in range(2 * num_samples):
            neg_ll = 0.5 * ((Y_obs_r - x) ** 2).sum() / (sigma_r ** 2)
            neg_lp = -rosenbrock_dist.logpdf(x.unsqueeze(0)).squeeze()
            energy = neg_ll + neg_lp
            mcmc_opt.zero_grad()
            energy.backward()
            mcmc_lr.step()
            mcmc_opt.step()
            samples.append(x.detach().clone())

        chain = torch.stack(samples)[num_samples:]
        rosenbrock_chains.append(chain)

    r_rhat = compute_rhat(rosenbrock_chains)
    r_ess = compute_ess(rosenbrock_chains[0])

    print(f"Rosenbrock R-hat: {r_rhat}")
    print(f"Rosenbrock ESS:   {r_ess}")
    print(f"R-hat < 1.1: {np.all(r_rhat < 1.1)}")
    print(f"ESS > 100:   {np.all(r_ess > 100)}")

    plot_mcmc_diagnostics(rosenbrock_chains, r_rhat, r_ess, plots_dir, prefix="rosenbrock_")

    # Save diagnostics
    np.savez(os.path.join(save_dir, "mcmc_diagnostics.npz"),
             nonlinear_rhat=nl_rhat, nonlinear_ess=nl_ess,
             rosenbrock_rhat=r_rhat, rosenbrock_ess=r_ess,
             dim=dim)
    print(f"Saved MCMC diagnostics.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nonlinear forward model experiment")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["train", "eval", "mcmc_diag", "all"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer_type", type=str, default="shared_hierarchical",
                        choices=["flat", "hierarchical", "shared_hierarchical"])
    parser.add_argument("--use_ddpm", action="store_true",
                        help="Use DDPM for posterior sampling instead of pSGLD")
    parser.add_argument("--ddpm_checkpoint", type=str, default=None,
                        help="Path to trained DDPM checkpoint")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.phase in ["all", "train"]:
        train_nonlinear(args.dim, args.epochs, device, layer_type=args.layer_type)

    if args.phase in ["all", "eval"]:
        evaluate_nonlinear(args.dim, device, layer_type=args.layer_type,
                          use_ddpm=args.use_ddpm, ddpm_checkpoint=args.ddpm_checkpoint)

    if args.phase in ["all", "mcmc_diag"]:
        run_mcmc_diagnostics(args.dim, device)

    print("\nDone!")


if __name__ == "__main__":
    main()
