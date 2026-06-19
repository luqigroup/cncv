"""Variance estimation (UQ) experiments for Rosenbrock and nonlinear problems.

Trains CNCV to reduce variance of h(x) = (x - mu_est)^2, i.e., pointwise
posterior variance estimation. Uses pretrained CNFs for:
  - Score computation (learned score)
  - Quick posterior mean estimates (mu_est)

Usage:
    python scripts/run_var_estimation.py --problem rosenbrock --gpu 0
    python scripts/run_var_estimation.py --problem nonlinear --gpu 0
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cncv.models.conditional_nf import ConditionalNF
from cncv.models import create_shared_hierarchical_ensemble
from cncv.utils import CustomLRScheduler


# ============================================================================
# Problem setup helpers
# ============================================================================

def setup_rosenbrock(device, seed=1):
    """Setup Rosenbrock problem and return CNF + problem params."""
    from cncv.utils.rosenbrock import RosenbrockDistribution

    dist = RosenbrockDistribution(mu=0.0, a=0.5, b=1.0, n1=2, n2=1)
    sigma = 0.3
    d = 2

    # Find CNF checkpoint
    import glob
    pattern = "data/checkpoints/cnf_rosenbrock_*save_freq-999/"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No CNF checkpoint found at {pattern}")
    ckpt_dir = dirs[-1]
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth")))
    cnf_path = ckpt_files[-1]
    print(f"CNF checkpoint: {cnf_path}")

    ckpt = torch.load(cnf_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    cnf = ConditionalNF(
        n_in=d, n_cond=d,
        n_hidden=args.n_hidden,
        n_flow_layers=args.n_flow_layers,
        depth=args.depth,
        n_mlp_layers=args.n_mlp_layers,
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)
    print(f"CNF loaded (epoch {ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params)")

    def sample_prior(n):
        return dist.sample(n, device=device)

    return cnf, sample_prior, sigma, d, "rosenbrock"


def setup_nonlinear(device, seed=12):
    """Setup nonlinear forward model F(x) = Ax + sin(x)."""
    import glob

    # Find CNF checkpoint
    pattern = "data/checkpoints/cnf_nonlinear_*save_freq-999/"
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"No CNF checkpoint found at {pattern}")
    ckpt_dir = dirs[-1]
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pth")))
    cnf_path = ckpt_files[-1]
    print(f"CNF checkpoint: {cnf_path}")

    ckpt = torch.load(cnf_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    d = args.input_dim
    cnf = ConditionalNF(
        n_in=d, n_cond=d,
        n_hidden=args.n_hidden,
        n_flow_layers=args.n_flow_layers,
        depth=args.depth,
        n_mlp_layers=args.n_mlp_layers,
    ).to(device)
    cnf.load_state_dict(ckpt["model_state_dict"])
    cnf.eval()
    for p in cnf.parameters():
        p.requires_grad_(False)
    print(f"CNF loaded (epoch {ckpt['epoch']}, "
          f"{sum(p.numel() for p in cnf.parameters()):,} params)")

    # Recover problem parameters from the nonlinear training checkpoint
    # Prior
    torch.manual_seed(seed)
    mu_prior = ckpt.get("mu_prior", torch.zeros(d))
    Sigma_prior = ckpt.get("Sigma_prior", None)
    sigma = float(ckpt.get("sigma", 0.3))

    if Sigma_prior is None:
        # Reconstruct from gaussian setup
        from cncv.utils.gaussian import create_prior_covariance
        Sigma_prior = create_prior_covariance(d, seed=seed)

    mu_prior = mu_prior.to(device)
    Sigma_prior = Sigma_prior.to(device)
    L_prior = torch.linalg.cholesky(Sigma_prior)

    def sample_prior(n):
        noise = torch.randn(n, d, device=device)
        return mu_prior + (L_prior @ noise.T).T

    return cnf, sample_prior, sigma, d, "nonlinear"


# ============================================================================
# Estimate posterior means using CNF
# ============================================================================

@torch.no_grad()
def estimate_posterior_means(cnf, Y, n_mc=256, batch_size=512):
    """Estimate E[x|y] for each y using CNF samples.

    Args:
        cnf: Pretrained ConditionalNF
        Y: Observations [N, d_y]
        n_mc: Number of CNF samples per observation
        batch_size: Process this many observations at a time

    Returns:
        mu_est: [N, d_x]
    """
    N = Y.shape[0]
    d = cnf.n_in
    mu_est = torch.zeros(N, d, device=Y.device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Y_batch = Y[start:end]  # [B, d_y]
        B = Y_batch.shape[0]

        # Sample n_mc per observation
        samples = torch.zeros(B, n_mc, d, device=Y.device)
        for k in range(B):
            samples[k] = cnf.sample(Y_batch[k], n_mc)

        mu_est[start:end] = samples.mean(dim=1)

    return mu_est


# ============================================================================
# Training
# ============================================================================

def train_var_cncv(cnf, sample_prior, sigma, d, problem_name, args):
    """Train CNCV for variance estimation h(x) = (x - mu(y))^2."""
    device = next(cnf.parameters()).device
    t0 = time.time()

    # Generate training data from joint p(x,y)
    print("\n=== Generating Training Data ===")
    torch.manual_seed(args.seed)
    X_train = sample_prior(args.num_train)
    Y_train = X_train + sigma * torch.randn_like(X_train)
    X_val = sample_prior(args.num_val)
    Y_val = X_val + sigma * torch.randn_like(X_val)

    # Estimate posterior means using CNF
    print(f"Estimating posterior means for {args.num_train} training obs "
          f"({args.n_mc_mean} CNF samples each)...")
    mu_train = estimate_posterior_means(cnf, Y_train, n_mc=args.n_mc_mean)
    mu_val = estimate_posterior_means(cnf, Y_val, n_mc=args.n_mc_mean)
    print(f"  mu_train range: [{mu_train.min():.3f}, {mu_train.max():.3f}]")
    print(f"  Posterior mean estimation took {time.time() - t0:.1f}s")

    # Create ensemble
    print("\n=== Creating CNCV Ensemble ===")
    ensemble = create_shared_hierarchical_ensemble(
        d, d, args.n_hidden, args.n_ensemble,
        n_layers=args.n_layers, depth=args.depth,
        n_cv=d, seed=args.seed,
    ).to(device)

    n_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"Ensemble: L={args.n_ensemble}, depth={args.depth}, "
          f"hidden={args.n_hidden}, mlp_layers={args.n_layers}, "
          f"params={n_params:,}")

    optimizer = torch.optim.Adam(ensemble.parameters(), lr=args.lr)
    num_batches = args.num_train // args.batchsize
    max_step = num_batches * args.max_epochs
    lr_sched = CustomLRScheduler(optimizer, args.lr, args.lr_final, max_step)

    # DataLoaders (include mu for each sample)
    train_ds = torch.utils.data.TensorDataset(X_train, Y_train, mu_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batchsize, shuffle=True)
    val_ds = torch.utils.data.TensorDataset(X_val, Y_val, mu_val)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batchsize, shuffle=False)

    # Training loop
    print(f"\n=== Training: {args.max_epochs} epochs ===")
    train_losses, val_losses = [], []
    ckpt_dir = f"results/{problem_name}_var_cnf"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_freq = max(1, args.max_epochs // 5)

    for epoch in tqdm(range(args.max_epochs), desc="CNCV-var", unit="ep"):
        # Validation
        ensemble.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                h_x = (xb - mb) ** 2  # [B, d]
                score = cnf.score(xb, yb)
                g_combined, _ = ensemble(xb, yb, score)
                residual = h_x.T - g_combined  # [d, B]
                val_loss += (residual ** 2).mean().item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Training
        ensemble.train()
        epoch_loss = 0.0
        for xb, yb, mb in train_loader:
            h_x = (xb - mb) ** 2
            score = cnf.score(xb, yb)
            g_combined, _ = ensemble(xb, yb, score)
            residual = h_x.T - g_combined
            loss = (residual ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 10.0)
            optimizer.step()
            lr_sched.step()
            epoch_loss += loss.item()
            train_losses.append(loss.item())

        epoch_loss /= num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"  Epoch {epoch}: train={epoch_loss:.6f}, "
                       f"val={val_loss:.6f}")

        if epoch % save_freq == 0 or epoch == args.max_epochs - 1:
            torch.save({
                "ensemble_state_dict": ensemble.state_dict(),
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "problem": problem_name,
                "d": d, "sigma": sigma,
                "n_params": n_params,
                "training_time_s": time.time() - t0,
            }, os.path.join(ckpt_dir, f"checkpoint_{epoch}.pth"))

    total_time = time.time() - t0
    print(f"\nTraining done in {total_time:.1f}s")
    print(f"Final: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")

    return ensemble, train_losses, val_losses, ckpt_dir


# ============================================================================
# Testing
# ============================================================================

@torch.no_grad()
def test_var_cncv(cnf, ensemble, sample_prior, sigma, d, problem_name,
                  ckpt_dir, args):
    """Evaluate variance reduction for posterior variance estimation."""
    device = next(cnf.parameters()).device
    ensemble.eval()

    print("\n=== Testing Variance Estimation ===")
    n_test_obs = args.n_test_obs
    n_samples = args.num_samples
    n_mc_mean = 1024  # more samples for test mean estimates

    # Generate test observations
    torch.manual_seed(args.seed + 1000)
    X_true_all = sample_prior(n_test_obs)
    Y_obs_all = X_true_all + sigma * torch.randn_like(X_true_all)

    all_vrfs = []
    all_corrs = []

    for i in range(n_test_obs):
        Y_obs = Y_obs_all[i]

        # Sample posterior from CNF
        X_samples = cnf.sample(Y_obs, n_samples)
        Y_rep = Y_obs.unsqueeze(0).expand(n_samples, -1)

        # Estimate posterior mean
        mu_est = X_samples.mean(dim=0)

        # h(x) = (x - mu)^2
        h_x = (X_samples - mu_est) ** 2  # [N, d]

        # Control variate
        score = cnf.score(X_samples, Y_rep)
        g_combined, _ = ensemble(X_samples, Y_rep, score)

        # VRF per component
        vrfs = []
        corrs = []
        for k in range(d):
            h_k = h_x[:, k]
            g_k = g_combined[k, :]
            var_h = h_k.var().item()
            var_hg = (h_k - g_k).var().item()
            vrf = var_hg / max(var_h, 1e-12)
            corr = torch.corrcoef(torch.stack([h_k, g_k]))[0, 1].item()
            vrfs.append(vrf)
            corrs.append(corr)

        all_vrfs.append(vrfs)
        all_corrs.append(corrs)

        if i < 3:
            print(f"  Obs {i}: VRF = {np.mean(vrfs):.3f} "
                  f"(per-comp: {[f'{v:.3f}' for v in vrfs]}), "
                  f"corr = {[f'{c:.3f}' for c in corrs]}")

    all_vrfs = np.array(all_vrfs)  # [n_test, d]
    all_corrs = np.array(all_corrs)

    mean_vrf = all_vrfs.mean()
    std_vrf = all_vrfs.std()
    reduction = (1 - mean_vrf) * 100

    print(f"\n=== Results ({problem_name}, d={d}, variance estimation) ===")
    print(f"VRF: {mean_vrf:.3f} ± {std_vrf:.3f} "
          f"({reduction:.1f}% variance reduction)")
    print(f"Per-component mean VRF: {all_vrfs.mean(axis=0)}")
    print(f"Mean correlation: {all_corrs.mean():.3f}")

    # Save results
    np.savez(os.path.join(ckpt_dir, "var_results.npz"),
             all_vrfs=all_vrfs, all_corrs=all_corrs,
             mean_vrf=mean_vrf, std_vrf=std_vrf)

    # --- Figure: per-component and per-observation VRF ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Per-component VRF
    comp_means = all_vrfs.mean(axis=0)
    comp_stds = all_vrfs.std(axis=0)
    axes[0].bar(range(d), comp_means, yerr=comp_stds, capsize=5,
                color="#4C72B0", alpha=0.8)
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
                    label="VRF = 1")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("VRF")
    axes[0].set_title(f"Per-component VRF ({problem_name})")
    axes[0].set_xticks(range(d))
    upper_a = max(1.5, float((comp_means + comp_stds).max()) * 1.05)
    axes[0].set_ylim(0, upper_a)
    axes[0].legend()

    # Per-observation VRF
    obs_means = all_vrfs.mean(axis=1)
    obs_stds = all_vrfs.std(axis=1)
    axes[1].bar(range(n_test_obs), obs_means, yerr=obs_stds, capsize=4,
                color="#DD8452", alpha=0.8)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
                    label="VRF = 1")
    axes[1].axhline(y=mean_vrf, color="red", linestyle="-", alpha=0.7,
                    label=f"Mean={mean_vrf:.3f}")
    axes[1].set_xlabel("Test observation")
    axes[1].set_ylabel("VRF (mean across components)")
    axes[1].set_title(f"Per-observation VRF ({problem_name})")
    upper_b = max(1.5, float((obs_means + obs_stds).max()) * 1.05)
    axes[1].set_ylim(0, upper_b)
    axes[1].legend()

    plt.suptitle(f"Variance Estimation: VRF = {mean_vrf:.3f} "
                 f"({reduction:.1f}% reduction)", fontweight="bold")
    plt.tight_layout()

    fig_path = os.path.join(ckpt_dir, f"{problem_name}_var_results.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    return all_vrfs, all_corrs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variance estimation with CNCV (posterior UQ)")
    parser.add_argument("--problem", type=str, required=True,
                        choices=["rosenbrock", "nonlinear"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    # Architecture
    parser.add_argument("--n_ensemble", type=int, default=16)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)

    # Training
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_final", type=float, default=1e-4)
    parser.add_argument("--num_train", type=int, default=65536)
    parser.add_argument("--num_val", type=int, default=2048)
    parser.add_argument("--n_mc_mean", type=int, default=256,
                        help="CNF samples for posterior mean estimation")

    # Testing
    parser.add_argument("--n_test_obs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Setup problem
    if args.problem == "rosenbrock":
        cnf, sample_prior, sigma, d, name = setup_rosenbrock(device, args.seed)
    elif args.problem == "nonlinear":
        cnf, sample_prior, sigma, d, name = setup_nonlinear(device, args.seed)

    # Train
    ensemble, train_losses, val_losses, ckpt_dir = train_var_cncv(
        cnf, sample_prior, sigma, d, name, args)

    # Test
    test_var_cncv(cnf, ensemble, sample_prior, sigma, d, name, ckpt_dir, args)


if __name__ == "__main__":
    main()
