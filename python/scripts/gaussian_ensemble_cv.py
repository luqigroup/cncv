"""Gaussian ensemble CV example.

This script trains and evaluates an ensemble of coupling layers with forward
and reverse splits to learn control variates for variance reduction in Bayesian
posterior estimation.

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
    compute_score_posterior_gaussian,
    compute_exact_posterior,
)
from cncv.utils import CustomLRScheduler

CONFIG_FILE = "gaussian_ensemble_cv.json"


class GaussianEnsembleCV:
    """Gaussian ensemble CV for variance reduction in Bayesian inference.

    Attributes:
        device: Computation device (CPU/CUDA)
        n_dim: Problem dimension
        mu_prior: Prior mean
        Sigma_prior: Prior covariance
        L_prior: Cholesky decomposition of prior covariance
        Sigma_prior_inv: Inverse prior covariance
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
        """Initialize Gaussian ensemble CV.

        Args:
            args: Configuration arguments from projorg
        """
        # Device setup
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Problem setup
        self.n_dim = 2
        self.mu_prior = torch.zeros(
            self.n_dim, dtype=torch.float32, device=self.device
        )
        self.Sigma_prior = torch.tensor(
            [[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32, device=self.device
        )
        self.L_prior = torch.linalg.cholesky(self.Sigma_prior)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
        self.sigma = float(args.sigma)

        print("\n=== Problem Setup ===")
        print(f"Dimension: {self.n_dim}")
        print(f"Prior mean: {self.mu_prior}")
        print(f"Prior covariance:\n{self.Sigma_prior}")
        print(f"Observation noise σ: {self.sigma}")

        # Create ensemble
        print("\n=== Creating Ensemble ===")
        self.ensemble = create_forward_reverse_ensemble(
            self.n_dim,
            self.n_dim,
            args.n_hidden,
            args.n_layers,
            n_cv=self.n_dim,
        ).to(self.device)

        print(
            f"Layer 1 (forward split): reverse_split = {self.ensemble.layers[0].reverse_split}"
        )
        print(
            f"Layer 2 (reverse split): reverse_split = {self.ensemble.layers[1].reverse_split}"
        )
        print(f"Number of layers in ensemble: {len(self.ensemble.layers)}")

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

        noise = torch.randn(
            args.num_train, self.n_dim, dtype=torch.float32, device=self.device
        )
        self.X_train = self.mu_prior + (self.L_prior @ noise.T).T
        self.Y_train = self.X_train + self.sigma * torch.randn_like(
            self.X_train
        )

        # Validation data
        noise_val = torch.randn(
            args.num_val, self.n_dim, dtype=torch.float32, device=self.device
        )
        self.X_val = self.mu_prior + (self.L_prior @ noise_val.T).T
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
        self.mu_prior = checkpoint["mu_prior"].to(self.device)
        self.Sigma_prior = checkpoint["Sigma_prior"].to(self.device)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
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

        # Compute score ∇log p(x|y)
        score = compute_score_posterior_gaussian(
            X, Y, self.mu_prior, self.Sigma_prior_inv, self.sigma
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
                    score = compute_score_posterior_gaussian(
                        X, Y, self.mu_prior, self.Sigma_prior_inv, self.sigma
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
                            "mu_prior": self.mu_prior.cpu(),
                            "Sigma_prior": self.Sigma_prior.cpu(),
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

        # Generate test observation
        X_true = self.mu_prior + self.L_prior @ torch.randn(
            self.n_dim, device=self.device
        )
        Y_obs = X_true + self.sigma * torch.randn(
            self.n_dim, device=self.device
        )

        print(f"True parameter X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        # Compute exact posterior
        mu_post, Sigma_post = compute_exact_posterior(
            Y_obs, self.mu_prior, self.Sigma_prior, self.sigma
        )

        print("\n=== Exact Posterior (Analytical) ===")
        print(f"Posterior mean: {mu_post}")
        print(f"Posterior covariance:\n{Sigma_post}")

        # Sample from exact posterior
        num_samples = args.num_samples
        print(f"\nSampling {num_samples} from exact posterior...")

        L_post = torch.linalg.cholesky(Sigma_post)
        noise = torch.randn(num_samples, self.n_dim, device=self.device)
        X_samples = mu_post + (L_post @ noise.T).T
        Y_samples = Y_obs.repeat(num_samples, 1)

        # Compute control variates
        print("\n=== Computing Control Variates ===")
        with torch.no_grad():
            score = compute_score_posterior_gaussian(
                X_samples,
                Y_samples,
                self.mu_prior,
                self.Sigma_prior_inv,
                self.sigma,
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

        # Variance reduction analysis
        self.analyze_variance_reduction(X_samples, g_combined, mu_post, args)

        # Generate plots
        self.plot_results(args)

    def analyze_variance_reduction(
        self,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        true_mean: torch.Tensor,
        args: argparse.Namespace,
    ) -> None:
        """Compute variance reduction metrics.

        Args:
            X_samples: Posterior samples [num_samples, n_dim]
            g_combined: Control variates [n_dim, num_samples]
            true_mean: Ground truth posterior mean [n_dim]
            args: Configuration
        """
        print("\n=== Variance Reduction Analysis ===")

        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_trials = 1000

        vanilla_mse = np.zeros((len(sample_sizes), self.n_dim))
        cv_mse = np.zeros((len(sample_sizes), self.n_dim))

        h_x = X_samples  # Quantity of interest: identity

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

    def plot_results(self, args: argparse.Namespace) -> None:
        """Generate evaluation plots.

        Args:
            args: Configuration arguments
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
        ax.set_title("Training Objective")
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

        print("\n=== Evaluation Complete ===")


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
    gaussian_cv = GaussianEnsembleCV(args)

    # Train or test based on phase
    if args.phase == "train":
        gaussian_cv.train(args)

    # Always run test (like in sips)
    gaussian_cv.test(args)
