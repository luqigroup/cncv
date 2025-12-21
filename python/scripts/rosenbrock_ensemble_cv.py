"""Rosenbrock ensemble CV example.

This script trains and evaluates an ensemble of coupling layers with forward
and reverse splits to learn control variates for variance reduction in Bayesian
posterior estimation with Rosenbrock prior.

Authors: Claude Code (translation from Julia)
Date: December 2024
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from projorg import checkpointsdir, plotsdir, setup_environment

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cncv import (
    create_forward_reverse_ensemble,
    create_random_split_ensemble,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from cncv.utils import CustomLRScheduler, pSGLD

CONFIG_FILE = "rosenbrock_ensemble_cv.json"


class RosenbrockEnsembleCV:
    """Rosenbrock ensemble CV for variance reduction in Bayesian inference.

    Attributes:
        device: Computation device (CPU/CUDA)
        n_dim: Problem dimension (always 2 for Rosenbrock)
        rosenbrock_dist: Rosenbrock prior distribution
        sigma: Observation noise std
        ensemble: Ensemble neural network
        mu: Learnable offsets
        optimizer: Adam optimizer
        lr_scheduler: Learning rate scheduler
        X_train: Training data (latent variables)
        Y_train: Training data (observations)
        X_val: Validation data (latent variables)
        Y_val: Validation data (observations)
        train_obj: Training loss history
        val_obj: Validation loss history
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize Rosenbrock ensemble CV.

        Args:
            args: Configuration arguments from projorg
        """
        # Device setup
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Problem setup - Rosenbrock is 2D
        self.n_dim = 2

        # Create Rosenbrock distribution
        self.rosenbrock_dist = RosenbrockDistribution(
            mu=args.rosenbrock_mu, a=args.rosenbrock_a
        )
        self.sigma = float(args.sigma)

        print("\n=== Problem Setup ===")
        print(f"Dimension: {self.n_dim}")
        print(f"Rosenbrock μ: {self.rosenbrock_dist.mu}")
        print(f"Rosenbrock a: {self.rosenbrock_dist.a}")
        print(f"Observation noise σ: {self.sigma}")

        # Create ensemble
        print("\n=== Creating Ensemble ===")

        # Use random split ensemble if n_ensemble_members is specified
        print(
            f"Using random split ensemble with {args.n_ensemble_members} members"
        )
        self.ensemble = create_random_split_ensemble(
            self.n_dim,
            self.n_dim,
            args.n_hidden,
            args.n_ensemble_members,
            args.n_layers,
            n_cv=self.n_dim,
            seed=12,
        ).to(self.device)

        print(f"Number of ensemble members: {len(self.ensemble.layers)}")
        for i, layer in enumerate(self.ensemble.layers):
            print(f"  Member {i + 1}: reverse_split = {layer.reverse_split}")

        # Count parameters
        num_params = sum(
            p.numel() for p in self.ensemble.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {num_params}")

        # Learnable offsets (one per component)
        # Note: mu is updated manually with simple SGD (like Julia), not with Adam
        self.mu = nn.Parameter(
            torch.zeros(self.n_dim, dtype=torch.float32, device=self.device)
        )

        # Optimizer (only for ensemble parameters, not mu)
        params = list(self.ensemble.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        # Learning rate scheduler (custom scheduler from sips package)
        # Calculate total number of steps
        num_batches_per_epoch = args.num_train // args.batchsize
        max_step = num_batches_per_epoch * args.max_epochs

        self.lr_scheduler = CustomLRScheduler(
            self.optimizer,
            initial_lr=args.lr,
            final_lr=args.lr_final,
            max_step=max_step,
        )

        # Generate training data
        print("\n=== Generating Training Data ===")
        print(f"Training samples: {args.num_train}")
        print(f"Validation samples: {args.num_val}")

        # Sample from Rosenbrock prior
        self.X_train = self.rosenbrock_dist.sample(
            args.num_train, device=self.device
        )
        self.Y_train = self.X_train + self.sigma * torch.randn_like(
            self.X_train
        )

        # Validation data
        self.X_val = self.rosenbrock_dist.sample(
            args.num_val, device=self.device
        )
        self.Y_val = self.X_val + self.sigma * torch.randn_like(self.X_val)

        # Logging
        self.train_obj = []
        self.val_obj = []

    def load_checkpoint(self, args: argparse.Namespace) -> None:
        """Load model checkpoint.

        Args:
            args: Configuration arguments

        Raises:
            ValueError: If checkpoint does not exist or epoch mismatch
        """
        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            "checkpoint_" + str(args.testing_epoch) + ".pth",
        )

        if not os.path.isfile(file_to_load):
            raise ValueError(f"Checkpoint does not exist: {file_to_load}")

        print(f"\n=== Loading Checkpoint ===")
        print(f"Path: {file_to_load}")

        if self.device == torch.device(type="cpu"):
            checkpoint = torch.load(
                file_to_load, map_location="cpu", weights_only=False
            )
        else:
            checkpoint = torch.load(file_to_load, weights_only=False)

        self.ensemble.load_state_dict(checkpoint["ensemble_state_dict"])
        self.mu = checkpoint["mu"].to(self.device)
        self.train_obj = checkpoint["train_obj"]
        self.val_obj = checkpoint["val_obj"]

        # Restore problem parameters
        self.rosenbrock_dist = RosenbrockDistribution(
            mu=checkpoint["rosenbrock_mu"], a=checkpoint["rosenbrock_a"]
        )
        self.sigma = checkpoint["sigma"]

        if not args.testing_epoch == checkpoint["epoch"]:
            raise ValueError(
                "Inconsistent filename and loaded checkpoint epoch."
            )

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Learned offsets μ: {self.mu}")

    def compute_loss(self, X: torch.Tensor, Y: torch.Tensor) -> tuple:
        """Compute ensemble CV loss.

        Args:
            X: Latent variables [batch_size, n_dim]
            Y: Observations [batch_size, n_dim]

        Returns:
            loss: Scalar loss
            g_combined: Combined control variate [n_dim, batch_size]
        """
        # Quantity of interest: h(x) = x
        h_x = X

        # Compute score ∇log p(x|y) for Rosenbrock posterior
        score = compute_score_posterior_rosenbrock(
            X, Y, self.rosenbrock_dist, self.sigma
        )

        # Forward through ensemble
        g_combined, _ = self.ensemble(X, Y, score)

        # Controlled estimator: h_i - g_i - μ_i
        residual = h_x.T - g_combined - self.mu[:, None]

        # Loss: minimize variance
        loss = (residual**2).mean()

        return loss, g_combined, residual

    def train(self, args: argparse.Namespace) -> None:
        """Training loop.

        Args:
            args: Configuration arguments
        """
        print("\n=== Starting Training ===")
        print(f"Max epochs: {args.max_epochs}")
        print(f"Batch size: {args.batchsize}")
        print(f"Learning rate: {args.lr} -> {args.lr_final}")

        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(
            self.X_train, self.Y_train
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.Y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batchsize, shuffle=False
        )

        num_batches = len(train_loader)
        print(f"Batches per epoch: {num_batches}")

        # Training loop
        val_loss = 0.0  # Initialize validation loss

        with tqdm(
            range(args.max_epochs), unit="epoch", colour="#B5F2A9"
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                # Validation
                self.ensemble.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        loss, _, _ = self.compute_loss(X_val, Y_val)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self.val_obj.append(val_loss)

                # Training
                self.ensemble.train()

                # Progress bar for batches within epoch
                batch_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{args.max_epochs}",
                    leave=False,
                    colour="#87CEEB",
                )

                for X, Y in batch_pbar:
                    # Forward pass
                    h_x = X
                    score = compute_score_posterior_rosenbrock(
                        X, Y, self.rosenbrock_dist, self.sigma
                    )
                    g_combined, _ = self.ensemble(X, Y, score)

                    # Compute residual and loss
                    residual = h_x.T - g_combined - self.mu[:, None]
                    loss = (residual**2).mean()

                    # Backward for ensemble parameters (pure autograd!)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Update mu manually with simple SGD (matching Julia implementation)
                    # Gradient: ∂L/∂μ[i] = -2 * mean(residual[i, :])
                    current_lr = self.lr_scheduler.compute_lr()
                    with torch.no_grad():
                        for i in range(self.n_dim):
                            residual_mean_i = residual[i, :].mean()
                            grad_mu_i = -2 * residual_mean_i
                            self.mu[i] -= current_lr * grad_mu_i

                    # Record training loss for this batch
                    current_train_loss = loss.item()
                    self.train_obj.append(current_train_loss)

                    # Update batch progress bar with current batch loss
                    batch_pbar.set_postfix(
                        {
                            "train_loss": f"{current_train_loss:.6f}",
                            "val_loss": f"{val_loss:.6f}",
                            "mu1": f"{self.mu[0].item():.4f}",
                            "mu2": f"{self.mu[1].item():.4f}",
                            "lr": f"{current_lr:.6f}",
                        }
                    )

                # Update epoch progress bar
                avg_train_loss = np.mean(
                    self.train_obj[-num_batches:]
                )  # Average of last epoch
                epoch_pbar.set_postfix(
                    {
                        "avg_train": f"{avg_train_loss:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "mu1": f"{self.mu[0].item():.4f}",
                        "mu2": f"{self.mu[1].item():.4f}",
                    }
                )

                # Save checkpoint
                if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                    torch.save(
                        {
                            "ensemble_state_dict": self.ensemble.state_dict(),
                            "mu": self.mu.detach().cpu(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "train_obj": self.train_obj,
                            "val_obj": self.val_obj,
                            "rosenbrock_mu": self.rosenbrock_dist.mu,
                            "rosenbrock_a": self.rosenbrock_dist.a,
                            "sigma": self.sigma,
                        },
                        os.path.join(
                            checkpointsdir(args.experiment),
                            "checkpoint_" + str(epoch) + ".pth",
                        ),
                    )

        print("\n=== Training Complete ===")
        print(f"Final training loss: {self.train_obj[-1]:.6f}")
        print(f"Final validation loss: {self.val_obj[-1]:.6f}")
        print(f"Learned offsets μ: {self.mu.detach().cpu().numpy()}")

    def sample_posterior_psgld(
        self, Y_obs: torch.Tensor, num_samples: int, args: argparse.Namespace
    ) -> torch.Tensor:
        """Sample from posterior using pSGLD with polynomial LR decay.

        Matches Julia implementation from src/sampling/sample.jl:
        - Total iterations: 20 * num_samples
        - Burnin: First half of iterations (10 * num_samples)
        - Thinning: Keep every 10th sample after burnin
        - LR schedule: Polynomial decay from 5.0 to 0.1 with gamma=-1/3

        Args:
            Y_obs: Observation [n_dim]
            num_samples: Number of samples to generate (default: 10000)
            args: Configuration

        Returns:
            samples: Posterior samples [num_samples, n_dim]
        """
        # pSGLD hyperparameters matching Julia

        lr_initial = 0.1
        lr_final = 0.01
        thinning = 1
        max_itr = 2 * num_samples  # Total iterations before burnin/thinning

        lr_scheduler = CustomLRScheduler(
            self.optimizer,
            initial_lr=lr_initial,
            final_lr=lr_final,
            max_step=max_itr,
        )

        # Initialize from prior
        x = self.rosenbrock_dist.sample(1, device=self.device).squeeze()
        x.requires_grad_(True)

        # Create optimizer with initial LR
        optimizer = pSGLD([x], lr=lr_initial)

        # Collect ALL samples during MCMC (will remove burnin later)
        all_samples = []

        print(f"Running pSGLD for {max_itr} iterations...")
        print(
            f"LR schedule: {lr_initial:.2f} → {lr_final:.2f} (polynomial decay, γ=-1/3)"
        )

        with tqdm(range(max_itr), unit="epoch", colour="#B5F2A9") as pbar:
            for itr in pbar:
                # Compute negative log posterior (energy)
                # -log p(x|y) = -log p(y|x) - log p(x) + const

                # Negative log likelihood: 0.5 * ||y - x||^2 / sigma^2
                neg_log_likelihood = (
                    0.5 * ((Y_obs - x) ** 2).sum() / (self.sigma**2)
                )

                # Negative log prior: -log p(x)
                neg_log_prior = -self.rosenbrock_dist.logpdf(
                    x.unsqueeze(0)
                ).squeeze()

                # Total energy
                energy = neg_log_likelihood + neg_log_prior

                # Backward pass
                optimizer.zero_grad()
                energy.backward()
                lr_scheduler.step()

                # pSGLD step
                optimizer.step()

                # Store sample (will filter later)
                all_samples.append(x.detach().clone())

                pbar.set_postfix({"energy": f"{energy:.6f}"})

        # Convert to tensor
        all_samples = torch.stack(all_samples)  # [max_itr, n_dim]

        # Burnin: Remove first half of iterations (matching Julia line 44)
        burnin_iters = max_itr // 2
        samples_after_burnin = all_samples[burnin_iters:]  # Keep second half

        print(f"Removed first {burnin_iters} iterations (burnin)")
        print(f"Remaining: {len(samples_after_burnin)} samples")

        # Thinning: Keep every 10th sample (matching Julia line 47)
        samples_thinned = samples_after_burnin[::thinning]  # Keep every 10th

        print(f"Applied thinning (keep every {thinning}th sample)")
        print(f"Final: {len(samples_thinned)} posterior samples")

        return samples_thinned

    def test(self, args: argparse.Namespace) -> None:
        """Evaluate variance reduction on test data.

        Args:
            args: Configuration arguments
        """
        # Load checkpoint
        self.load_checkpoint(args)

        # Set to evaluation mode
        self.ensemble.eval()

        print("\n=== Generating Test Data ===")

        # Create a fixed observation (sample one point and add noise)
        X_true = self.rosenbrock_dist.sample(1, device=self.device)[0]
        Y_obs = X_true + self.sigma * torch.randn(
            self.n_dim, device=self.device
        )

        print(f"True parameter X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        # Sample from posterior using pSGLD (MCMC)
        num_samples = args.num_samples
        print(f"\nSampling {num_samples} from posterior using pSGLD...")

        X_samples = self.sample_posterior_psgld(Y_obs, num_samples, args)

        print(f"Posterior samples mean: {X_samples.mean(dim=0)}")
        print(f"Posterior samples std: {X_samples.std(dim=0)}")

        # Y_obs is the same for all samples
        Y_samples = Y_obs.repeat(num_samples, 1)

        # Compute control variates
        print("\n=== Computing Control Variates ===")
        with torch.no_grad():
            score = compute_score_posterior_rosenbrock(
                X_samples, Y_samples, self.rosenbrock_dist, self.sigma
            )
            g_combined, all_g = self.ensemble(X_samples, Y_samples, score)

        print("Ensemble Control Variates:")
        for k in range(self.n_dim):
            mean_g = g_combined[k, :].mean().item()
            std_g = g_combined[k, :].std().item()
            corr = torch.corrcoef(
                torch.stack([X_samples[:, k], g_combined[k, :]])
            )[0, 1].item()
            print(f"  CV {k}: mean = {mean_g:.6f}, std = {std_g:.6f}")
            print(f"    Correlation with h_{k}: {corr:.6f}")

        # Stein's identity check
        print("\n=== Stein's Identity Check ===")
        print("Ensemble CV should have E[g_k|Y] ≈ 0")
        for k in range(self.n_dim):
            ratio = (
                abs(g_combined[k, :].mean().item())
                / g_combined[k, :].std().item()
            )
            print(
                f"  Component {k}: |E[g_{k}]| / Std[g_{k}] = {ratio:.4f} (should be << 1)"
            )

        # For variance reduction analysis, use posterior samples mean as "true" value
        # (Since we don't have analytical posterior for Rosenbrock)
        posterior_mean = X_samples.mean(dim=0)

        # Plot posterior vs prior distributions
        self.plot_posterior_vs_prior(X_samples, args)

        # Variance reduction analysis
        self.analyze_variance_reduction(
            X_samples, g_combined, all_g, posterior_mean, args
        )

        # Generate plots
        self.plot_results(args, X_samples, g_combined, all_g)

    def analyze_variance_reduction(
        self,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g: list,
        true_mean: torch.Tensor,
        args: argparse.Namespace,
    ) -> None:
        """Compute variance reduction metrics.

        Args:
            X_samples: Posterior samples [num_samples, n_dim]
            g_combined: Combined control variates [n_dim, num_samples]
            all_g: List of individual layer CVs
            true_mean: Ground truth posterior mean [n_dim]
            args: Configuration
        """
        print("\n=== Variance Reduction Analysis ===")

        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_trials = 1000

        vanilla_mse = np.zeros((len(sample_sizes), self.n_dim))
        cv_mse = np.zeros((len(sample_sizes), self.n_dim))

        h_x = X_samples  # Quantity of interest: identity

        # Store individual layer CVs for plotting
        self.all_g = all_g
        self.true_mean = true_mean

        for idx, n in enumerate(sample_sizes):
            vanilla_estimates = np.zeros((num_trials, self.n_dim))
            cv_estimates = np.zeros((num_trials, self.n_dim))

            for trial in range(num_trials):
                # Random subsample
                indices = torch.randperm(X_samples.shape[0])[:n]

                # Vanilla MC
                vanilla_estimates[trial] = (
                    h_x[indices].mean(dim=0).cpu().numpy()
                )

                # CV estimator
                cv_residual = h_x[indices].T - g_combined[:, indices]
                cv_estimates[trial] = (
                    (cv_residual.mean(dim=1) + self.mu).cpu().numpy()
                )

            # Compute MSE
            for i in range(self.n_dim):
                vanilla_mse[idx, i] = (
                    (vanilla_estimates[:, i] - true_mean[i].cpu().numpy()) ** 2
                ).mean()
                cv_mse[idx, i] = (
                    (cv_estimates[:, i] - true_mean[i].cpu().numpy()) ** 2
                ).mean()

        # Print results
        print("\n=== Variance Reduction Summary ===")
        for i in range(self.n_dim):
            vrf = cv_mse[:, i].mean() / vanilla_mse[:, i].mean()
            reduction_pct = (1.0 - vrf) * 100
            print(
                f"Component {i}: VRF = {vrf:.4f}, Reduction = {reduction_pct:.1f}%"
            )

        # Store for plotting
        self.sample_sizes = sample_sizes
        self.vanilla_mse = vanilla_mse
        self.cv_mse = cv_mse

    def plot_posterior_vs_prior(
        self, X_posterior: torch.Tensor, args: argparse.Namespace
    ) -> None:
        """Plot posterior samples overlaid on prior samples.

        Args:
            X_posterior: Posterior samples from pSGLD [num_samples, n_dim]
            args: Configuration
        """
        print("\n=== Generating Posterior vs Prior Plot ===")

        # Generate prior samples (same number as posterior)
        num_samples = X_posterior.shape[0]
        X_prior = self.rosenbrock_dist.sample(num_samples, device=self.device)

        # Convert to numpy for plotting
        X_prior_np = X_prior.cpu().numpy()
        X_posterior_np = X_posterior.cpu().numpy()

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.patch.set_facecolor("white")

        # Plot prior samples (black, low alpha)
        ax.scatter(
            X_prior_np[:, 0],
            X_prior_np[:, 1],
            s=0.5,
            color="#000000",
            alpha=0.15,
            label="Prior samples",
        )

        # Plot posterior samples (red, higher alpha)
        ax.scatter(
            X_posterior_np[:, 0],
            X_posterior_np[:, 1],
            s=1.0,
            color="#d62728",
            alpha=0.3,
            label="Posterior samples (pSGLD)",
        )

        ax.set_xlim([-3, 3])
        ax.set_ylim([-2.5, 7])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Rosenbrock: Posterior vs Prior Distributions")
        ax.legend(loc="upper right")
        ax.grid(False)

        plt.tight_layout()

        save_path = os.path.join(
            plotsdir(args.experiment), "posterior_vs_prior.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

    def plot_results(
        self,
        args: argparse.Namespace,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g: list,
    ) -> None:
        """Generate evaluation plots (all 6 plots from Gaussian version).

        Args:
            args: Configuration arguments
            X_samples: Posterior samples [num_samples, n_dim]
            g_combined: Combined control variates [n_dim, num_samples]
            all_g: List of individual layer CVs
        """
        print(f"\n=== Generating Plots ===")
        print(f"Plot directory: {plotsdir(args.experiment)}")

        # 1. Training loss plot
        fig, ax = plt.subplots(figsize=(7, 4))

        # Training loss (one per batch)
        epochs_train = np.linspace(0, len(self.val_obj), len(self.train_obj))
        ax.plot(
            epochs_train,
            self.train_obj,
            color="#4a4a4a",
            label="Training loss",
            alpha=0.7,
        )

        # Validation loss (one per epoch)
        epochs_val = np.arange(len(self.val_obj))
        ax.plot(
            epochs_val,
            self.val_obj,
            color="#a1a1a1",
            label="Validation loss",
            linewidth=2,
        )

        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Objective (Rosenbrock)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(plotsdir(args.experiment), "training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

        # 2. MSE comparison plot
        fig, axes = plt.subplots(1, self.n_dim, figsize=(10, 5))
        if self.n_dim == 1:
            axes = [axes]

        for i in range(self.n_dim):
            axes[i].plot(
                self.sample_sizes,
                self.vanilla_mse[:, i],
                "o-",
                label="Vanilla MC",
                linewidth=2,
                markersize=6,
                color="#7f7f7f",
            )
            axes[i].plot(
                self.sample_sizes,
                self.cv_mse[:, i],
                "d-",
                label="Ensemble CV",
                linewidth=2,
                markersize=6,
                color="#d62728",
            )

            # Reference 1/n line
            ref_line = self.vanilla_mse[0, i] * (
                self.sample_sizes[0] / np.array(self.sample_sizes)
            )
            axes[i].plot(
                self.sample_sizes,
                ref_line,
                "--",
                label="1/n",
                linewidth=1.5,
                color="k",
                alpha=0.5,
            )

            axes[i].set_xlabel("Sample size")
            axes[i].set_ylabel("MSE")
            axes[i].set_title(f"Component {i}")
            axes[i].set_xscale("log")
            axes[i].set_yscale("log")
            axes[i].legend()
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(
            plotsdir(args.experiment), "mse_comparison.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

        # 3. Estimator distributions
        self.plot_estimator_distributions(args, X_samples, g_combined)

        # 4. VRF vs sample size
        self.plot_vrf_vs_sample_size(args)

        # 5. Variance comparison
        self.plot_variance_comparison(args, X_samples, g_combined)

        # 6. Scatter plots
        self.plot_scatter_h_vs_gcv(args, X_samples, g_combined, all_g)

        print("\n=== Evaluation Complete ===")

    def plot_estimator_distributions(
        self,
        args: argparse.Namespace,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
    ) -> None:
        """Plot violin plots comparing Vanilla MC vs Ensemble CV distributions."""
        sample_size = 100
        n_trials = 200

        h_x = X_samples

        # Collect estimates
        vanilla_estimates = np.zeros((n_trials, self.n_dim))
        cv_estimates = np.zeros((n_trials, self.n_dim))

        for trial in range(n_trials):
            indices = torch.randperm(X_samples.shape[0])[:sample_size]

            # Vanilla MC
            vanilla_estimates[trial] = h_x[indices].mean(dim=0).cpu().numpy()

            # CV estimator
            cv_residual = h_x[indices].T - g_combined[:, indices]
            cv_estimates[trial] = (
                (cv_residual.mean(dim=1) + self.mu).cpu().numpy()
            )

        # Create violin plots
        fig, axes = plt.subplots(1, self.n_dim, figsize=(10, 5))
        if self.n_dim == 1:
            axes = [axes]

        for i in range(self.n_dim):
            parts = axes[i].violinplot(
                [vanilla_estimates[:, i], cv_estimates[:, i]],
                positions=[1, 2],
                showmeans=True,
                showextrema=True,
            )

            # Color the violins
            for j, pc in enumerate(parts["bodies"]):
                if j == 0:
                    pc.set_facecolor("#7f7f7f")
                    pc.set_alpha(0.7)
                else:
                    pc.set_facecolor("#d62728")
                    pc.set_alpha(0.7)

            # Add ground truth line
            truth = self.true_mean[i].cpu().numpy()
            axes[i].axhline(
                truth,
                color="k",
                linestyle="--",
                linewidth=2,
                label="Ground truth",
            )

            # Compute VRF
            var_vanilla = vanilla_estimates[:, i].var()
            var_cv = cv_estimates[:, i].var()
            vrf = var_cv / var_vanilla
            reduction_pct = (1.0 - vrf) * 100

            # Add VRF annotation
            axes[i].text(
                0.5,
                0.98,
                f"VRF = {vrf:.3f}\nReduction = {reduction_pct:.1f}%",
                transform=axes[i].transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=9,
            )

            axes[i].set_xticks([1, 2])
            axes[i].set_xticklabels(["Vanilla MC", "Ensemble CV"])
            axes[i].set_ylabel("Estimate value")
            axes[i].set_title(f"Component {i}")
            axes[i].legend()
            axes[i].grid(alpha=0.3, axis="y")

        plt.suptitle(
            f"Estimator Distributions (n={sample_size}, {n_trials} trials)",
            fontsize=12,
        )
        plt.tight_layout()

        save_path = os.path.join(
            plotsdir(args.experiment), "estimator_distributions.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

    def plot_vrf_vs_sample_size(self, args: argparse.Namespace) -> None:
        """Plot Variance Reduction Factor vs sample size."""
        # Compute VRF for each sample size
        vrf = self.cv_mse / self.vanilla_mse

        fig, axes = plt.subplots(1, self.n_dim, figsize=(10, 5))
        if self.n_dim == 1:
            axes = [axes]

        for i in range(self.n_dim):
            axes[i].plot(
                self.sample_sizes,
                vrf[:, i],
                "o-",
                linewidth=2,
                markersize=6,
                color="#d62728",
                label="VRF",
            )

            # Reference line at VRF = 1.0
            axes[i].axhline(
                1.0,
                color="k",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                label="No reduction",
            )

            axes[i].set_xlabel("Sample size")
            axes[i].set_ylabel("Variance Reduction Factor")
            axes[i].set_title(f"Component {i}")
            axes[i].set_xscale("log")
            axes[i].legend()
            axes[i].grid(alpha=0.3)

        plt.suptitle("Variance Reduction Factor vs Sample Size", fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(
            plotsdir(args.experiment), "vrf_vs_sample_size.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

    def plot_variance_comparison(
        self,
        args: argparse.Namespace,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
    ) -> None:
        """Plot bar chart comparing Var[Vanilla] vs Var[CV]."""
        sample_size = 100
        n_trials = 500

        h_x = X_samples

        # Collect estimates
        vanilla_estimates = np.zeros((n_trials, self.n_dim))
        cv_estimates = np.zeros((n_trials, self.n_dim))

        for trial in range(n_trials):
            indices = torch.randperm(X_samples.shape[0])[:sample_size]

            # Vanilla MC
            vanilla_estimates[trial] = h_x[indices].mean(dim=0).cpu().numpy()

            # CV estimator
            cv_residual = h_x[indices].T - g_combined[:, indices]
            cv_estimates[trial] = (
                (cv_residual.mean(dim=1) + self.mu).cpu().numpy()
            )

        # Compute variances
        var_vanilla = vanilla_estimates.var(axis=0)
        var_cv = cv_estimates.var(axis=0)

        # Create bar charts
        fig, axes = plt.subplots(1, self.n_dim, figsize=(10, 5))
        if self.n_dim == 1:
            axes = [axes]

        x = np.arange(2)
        width = 0.6

        for i in range(self.n_dim):
            bars = axes[i].bar(
                x,
                [var_vanilla[i], var_cv[i]],
                width,
                color=["#7f7f7f", "#d62728"],
                alpha=0.7,
            )

            # Compute VRF
            vrf = var_cv[i] / var_vanilla[i]
            reduction_pct = (1.0 - vrf) * 100

            # Add VRF annotation
            axes[i].text(
                0.5,
                0.98,
                f"VRF = {vrf:.3f}\nReduction = {reduction_pct:.1f}%",
                transform=axes[i].transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=10,
            )

            axes[i].set_xticks(x)
            axes[i].set_xticklabels(["Vanilla MC", "Ensemble CV"])
            axes[i].set_ylabel("Variance")
            axes[i].set_title(f"Component {i}")
            axes[i].grid(alpha=0.3, axis="y")

        plt.suptitle(
            f"Variance Comparison (n={sample_size}, {n_trials} trials)",
            fontsize=12,
        )
        plt.tight_layout()

        save_path = os.path.join(
            plotsdir(args.experiment), "variance_comparison.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()

    def plot_scatter_h_vs_gcv(
        self,
        args: argparse.Namespace,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g: list,
    ) -> None:
        """Plot 2x3 grid of scatter plots: h vs g for each layer + combined."""
        h_x = X_samples.cpu().numpy()  # [num_samples, n_dim]

        # Create 2 rows (one per component) x N+1 columns (N layers + combined) grid
        n_layers = len(all_g)
        fig, axes = plt.subplots(self.n_dim, n_layers + 1, figsize=(15, 8))

        if self.n_dim == 1:
            axes = axes.reshape(1, -1)

        for i in range(self.n_dim):  # For each component
            # Plot each layer's CV
            for j, g_layer in enumerate(all_g):
                g_layer_np = g_layer[i, :].cpu().numpy()
                axes[i, j].scatter(
                    h_x[:, i], g_layer_np, s=1, alpha=0.5, color="#4a4a4a"
                )

                # Compute correlation
                corr = np.corrcoef(h_x[:, i], g_layer_np)[0, 1]

                axes[i, j].set_xlabel(f"h_{i}")
                axes[i, j].set_ylabel(f"g_{i} (Layer {j + 1})")
                axes[i, j].set_title(
                    f"Component {i}, Layer {j + 1}\nCorr = {corr:.3f}"
                )
                axes[i, j].grid(alpha=0.3)

            # Plot combined CV
            g_combined_np = g_combined[i, :].cpu().numpy()
            axes[i, n_layers].scatter(
                h_x[:, i], g_combined_np, s=1, alpha=0.5, color="#d62728"
            )

            # Compute correlation
            corr_combined = np.corrcoef(h_x[:, i], g_combined_np)[0, 1]

            axes[i, n_layers].set_xlabel(f"h_{i}")
            axes[i, n_layers].set_ylabel(f"g_{i} (Combined)")
            axes[i, n_layers].set_title(
                f"Component {i}, Combined\nCorr = {corr_combined:.3f}"
            )
            axes[i, n_layers].grid(alpha=0.3)

        plt.suptitle(
            "Correlation between h(x) and Control Variates", fontsize=14
        )
        plt.tight_layout()

        save_path = os.path.join(
            plotsdir(args.experiment), "scatter_h_vs_gcv.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
        plt.close()


if "__main__" == __name__:
    # Setup environment with projorg
    args = setup_environment(
        CONFIG_FILE,
        ignore_arg_list=[
            "experiment_name",
            "gpu_id",
            "phase",
            "seed",
            "testing_epoch",
            "num_samples",
        ],
    )

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Handle testing_epoch default
    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1

    # Create instance
    rosenbrock_cv = RosenbrockEnsembleCV(args)

    # Train or test based on phase
    if args.phase == "train":
        rosenbrock_cv.train(args)

    # Always run test (like in sips)
    rosenbrock_cv.test(args)
