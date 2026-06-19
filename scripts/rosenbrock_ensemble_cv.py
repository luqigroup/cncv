"""Rosenbrock ensemble CV example.

This script trains and evaluates an ensemble of coupling layers with forward
and reverse splits to learn control variates for variance reduction in Bayesian
posterior estimation with Rosenbrock prior.
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from projorg import checkpointsdir, plotsdir, setup_environment, upload_to_cloud

from cncv import (
    create_random_split_ensemble,
    create_hierarchical_ensemble,
    create_shared_hierarchical_ensemble,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
    load_ensemble_checkpoint,
    get_split_config,
    plot_results,
    plot_posterior_vs_prior,
    compute_h_x,
)
from cncv.models.ddpm_mlp import ConditionalScoreMLPSimple
from cncv.diffusion import NoiseScheduler, extract_score_from_ddpm, load_ddpm_checkpoint
from cncv.utils import CustomLRScheduler, pSGLD

CONFIG_FILE = "rosenbrock_ensemble_cv.json"
DDPM_CONFIG_FILE = "ddpm_rosenbrock.json"


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

        # Validate and store quantity of interest
        if args.qofi not in ["mean", "var"]:
            raise ValueError(f"qofi must be 'mean' or 'var', got '{args.qofi}'")
        self.qofi = args.qofi
        print(f"Quantity of interest (qofi): {self.qofi}")

        # Problem setup - Rosenbrock with configurable dimensions
        # Dimension specified via input_dim (uses chain structure by default)
        # Can optionally override n1, n2 for custom block structure
        input_dim = int(args.input_dim)

        # Check if custom block structure is specified
        if hasattr(args, "n1") and hasattr(args, "n2"):
            n1 = int(args.n1)
            n2 = int(args.n2)
            # Validate dimension formula
            computed_dim = (n1 - 1) * n2 + 1
            if computed_dim != input_dim:
                raise ValueError(
                    f"Dimension mismatch! input_dim={input_dim} but (n1-1)*n2+1 = "
                    f"({n1}-1)*{n2}+1 = {computed_dim}. "
                    f"Please ensure n1,n2 satisfy: (n1-1)*n2+1 = input_dim"
                )
        else:
            # Default chain structure: n1=2, n2=input_dim-1
            n1 = 2
            n2 = input_dim - 1

        # Create Rosenbrock distribution
        self.rosenbrock_dist = RosenbrockDistribution(
            mu=args.mu, a=args.a, b=args.b, n1=n1, n2=n2
        )
        self.n_dim = self.rosenbrock_dist.ndim
        self.sigma = float(args.sigma)

        print("\n=== Problem Setup ===")
        print(f"Dimension: {self.n_dim} (n1={n1}, n2={n2})")
        print(f"Rosenbrock μ: {self.rosenbrock_dist.mu}")
        print(f"Rosenbrock a: {self.rosenbrock_dist.a}")
        print(f"Rosenbrock b: {self.rosenbrock_dist.b}")
        print(f"Observation noise σ: {self.sigma}")

        # Create ensemble
        print("\n=== Creating Ensemble ===")

        self.layer_type = getattr(args, "layer_type", "flat")

        if self.layer_type == "shared_hierarchical":
            print(f"Using shared hierarchical ensemble with {args.n_ensemble_members} members")
            depth = getattr(args, "depth", None)
            self.ensemble = create_shared_hierarchical_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                n_layers=args.n_layers,
                depth=depth,
                n_cv=self.n_dim,
                seed=1,
            ).to(self.device)
            self.split_config = None
        elif self.layer_type == "hierarchical":
            print(f"Using hierarchical ensemble with {args.n_ensemble_members} members")
            depth = getattr(args, "depth", None)
            self.ensemble = create_hierarchical_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                n_layers=args.n_layers,
                depth=depth,
                n_cv=self.n_dim,
                seed=1,
            ).to(self.device)
            self.split_config = None
        else:
            print(f"Using random split ensemble with {args.n_ensemble_members} members")
            self.ensemble = create_random_split_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                args.n_layers,
                n_cv=self.n_dim,
                seed=1,
            ).to(self.device)
            self.split_config = get_split_config(self.ensemble)

        # Count parameters
        num_params = sum(
            p.numel() for p in self.ensemble.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {num_params}")

        # Optimizer for ensemble parameters
        params = list(self.ensemble.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        # Learning rate scheduler
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

        if self.qofi == "mean":
            # For mean mode: sample from joint distribution p(x,y)
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

        elif self.qofi == "var":
            # For variance mode: sample from posterior p(x|y) using MCMC
            print("Running MCMC preprocessing to estimate posterior means...")

            # Generate observations Y first
            X_true_train = self.rosenbrock_dist.sample(
                args.num_train, device=self.device
            )
            self.Y_train = X_true_train + self.sigma * torch.randn_like(
                X_true_train
            )

            # For each Y, run MCMC to estimate posterior and sample X
            self.X_train = torch.zeros(
                args.num_train, self.n_dim, device=self.device
            )
            self.mu_post_train = torch.zeros(
                args.num_train, self.n_dim, device=self.device
            )

            # MCMC hyperparameters for preprocessing (lighter than test)
            mcmc_samples = 1000  # Fewer samples for preprocessing
            mcmc_batch_size = 256  # Process observations in parallel batches

            print(
                f"Running batched MCMC for {args.num_train} training samples..."
            )
            print(
                f"Batch size: {mcmc_batch_size}, MCMC samples per obs: {mcmc_samples}"
            )

            num_batches = (
                args.num_train + mcmc_batch_size - 1
            ) // mcmc_batch_size
            for batch_idx in tqdm(
                range(num_batches), desc="Training MCMC", colour="#87CEEB"
            ):
                start_idx = batch_idx * mcmc_batch_size
                end_idx = min(start_idx + mcmc_batch_size, args.num_train)

                # Get batch of Y observations
                Y_batch = self.Y_train[start_idx:end_idx]

                # Run batched MCMC
                posterior_samples_batch, mu_post_batch = (
                    self.sample_posterior_psgld_light_batch(
                        Y_batch, mcmc_samples
                    )
                )  # [batch_size, num_samples, n_dim], [batch_size, n_dim]

                # Store posterior means
                self.mu_post_train[start_idx:end_idx] = mu_post_batch

                # Sample one X from posterior for each observation
                batch_size_actual = end_idx - start_idx
                for i in range(batch_size_actual):
                    idx = torch.randint(mcmc_samples, (1,)).item()
                    self.X_train[start_idx + i] = posterior_samples_batch[
                        i, idx
                    ]

            # Validation data
            X_true_val = self.rosenbrock_dist.sample(
                args.num_val, device=self.device
            )
            self.Y_val = X_true_val + self.sigma * torch.randn_like(X_true_val)

            self.X_val = torch.zeros(
                args.num_val, self.n_dim, device=self.device
            )
            self.mu_post_val = torch.zeros(
                args.num_val, self.n_dim, device=self.device
            )

            print(
                f"Running batched MCMC for {args.num_val} validation samples..."
            )
            num_batches_val = (
                args.num_val + mcmc_batch_size - 1
            ) // mcmc_batch_size
            for batch_idx in tqdm(
                range(num_batches_val), desc="Validation MCMC", colour="#FFB6C1"
            ):
                start_idx = batch_idx * mcmc_batch_size
                end_idx = min(start_idx + mcmc_batch_size, args.num_val)

                # Get batch of Y observations
                Y_batch = self.Y_val[start_idx:end_idx]

                # Run batched MCMC
                posterior_samples_batch, mu_post_batch = (
                    self.sample_posterior_psgld_light_batch(
                        Y_batch, mcmc_samples
                    )
                )

                # Store posterior means
                self.mu_post_val[start_idx:end_idx] = mu_post_batch

                # Sample one X from posterior for each observation
                batch_size_actual = end_idx - start_idx
                for i in range(batch_size_actual):
                    idx = torch.randint(mcmc_samples, (1,)).item()
                    self.X_val[start_idx + i] = posterior_samples_batch[i, idx]

            print("MCMC preprocessing complete!")
            print(
                f"Training mu_post stats: mean={self.mu_post_train.mean():.4f}, "
                f"std={self.mu_post_train.std():.4f}, "
                f"min={self.mu_post_train.min():.4f}, max={self.mu_post_train.max():.4f}"
            )
            print(
                f"Training X stats: mean={self.X_train.mean():.4f}, "
                f"std={self.X_train.std():.4f}, "
                f"min={self.X_train.min():.4f}, max={self.X_train.max():.4f}"
            )
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

        # Score mode setup
        self.use_ddpm = args.ddpm
        print(
            f"\n=== Score Mode: {'DDPM (learned)' if self.use_ddpm else 'Analytical'} ==="
        )

        if self.use_ddpm:
            # Load DDPM config using setup_environment
            print(f"Loading DDPM config from: {DDPM_CONFIG_FILE}")
            ddpm_args = setup_environment(
                DDPM_CONFIG_FILE,
                ignore_arg_list=[
                    "experiment_name",
                    "gpu_id",
                    "phase",
                    "seed",
                    "testing_epoch",
                ],
            )

            # Get DDPM epoch from args, or use last epoch from ddpm config
            if ddpm_args.testing_epoch == -1:
                ddpm_epoch = ddpm_args.max_epochs - 1

            ddpm_checkpoint_path = os.path.join(
                checkpointsdir(ddpm_args.experiment),
                "checkpoint_" + str(ddpm_epoch) + ".pth",
            )
            print(f"Loading checkpoint from epoch: {ddpm_epoch}")
            print(f"Checkpoint path: {ddpm_checkpoint_path}")

            # Create DDPM model using parameters from ddpm_args
            self.ddpm_model = ConditionalScoreMLPSimple(
                x_dim=self.n_dim,
                y_dim=self.n_dim,
                hidden_dim=ddpm_args.hidden_dim,
                n_layers=ddpm_args.n_layers,
                time_emb_dim=64,
                nt=ddpm_args.nt,
            ).to(self.device)

            # Load DDPM checkpoint
            self.ddpm_model, ddpm_checkpoint = load_ddpm_checkpoint(
                ddpm_checkpoint_path, self.ddpm_model, self.device
            )

            # Create noise scheduler using parameters from ddpm_args
            self.ddpm_noise_scheduler = NoiseScheduler(
                nt=ddpm_args.nt,
                beta_start=ddpm_args.beta_start,
                beta_end=ddpm_args.beta_end,
                beta_schedule=ddpm_args.beta_schedule,
                device=self.device,
            )

            print(
                f"✓ DDPM loaded successfully (epoch {ddpm_checkpoint['epoch']})"
            )
        else:
            self.ddpm_model = None
            self.ddpm_noise_scheduler = None
            print("Using analytical score computation")

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

        # Skip split validation for hierarchical/shared layers
        validate = self.layer_type not in ("hierarchical", "shared_hierarchical")
        checkpoint = load_ensemble_checkpoint(
            self.ensemble, file_to_load, self.device, validate_splits=validate
        )

        self.train_obj = checkpoint["train_obj"]
        self.val_obj = checkpoint["val_obj"]

        # Restore problem parameters
        self.rosenbrock_dist = RosenbrockDistribution(
            mu=checkpoint["mu"],
            a=checkpoint["a"],
            b=checkpoint["b"],
            n1=checkpoint["n1"],
            n2=checkpoint["n2"],
        )
        self.n_dim = self.rosenbrock_dist.ndim
        self.sigma = checkpoint["sigma"]

        # Validate dimension
        checkpoint_dim = checkpoint["input_dim"]
        if checkpoint_dim != self.n_dim:
            raise ValueError(
                f"Dimension mismatch! Checkpoint was trained with input_dim={checkpoint_dim}, "
                f"but loaded distribution has input_dim={self.n_dim}. "
                f"Please use the same dimension for training and testing."
            )

        if not args.testing_epoch == checkpoint["epoch"]:
            raise ValueError(
                "Inconsistent filename and loaded checkpoint epoch."
            )

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def compute_loss(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mu_post_batch: torch.Tensor = None,
    ) -> tuple:
        """Compute ensemble CV loss.

        Args:
            X: Latent variables [batch_size, n_dim]
            Y: Observations [batch_size, n_dim]
            mu_post_batch: Posterior means for this batch [batch_size, n_dim], for var mode

        Returns:
            loss: Scalar loss
            g_combined: Combined control variate [n_dim, batch_size]
        """
        # Quantity of interest: h(x)
        if self.qofi == "mean":
            # h(x) = x
            h_x = X
        elif self.qofi == "var":
            # h(x) = (x - mu_post)^2 for pointwise variance
            if mu_post_batch is None:
                raise ValueError(
                    "mu_post_batch must be provided when qofi='var'"
                )
            h_x = (X - mu_post_batch) ** 2
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

        # Compute score ∇log p(x|y) for Rosenbrock posterior
        if self.use_ddpm:
            score = extract_score_from_ddpm(
                self.ddpm_model,
                self.ddpm_noise_scheduler,
                X,
                Y,
                self.device,
                timestep=1,
            )
        else:
            score = compute_score_posterior_rosenbrock(
                X, Y, self.rosenbrock_dist, self.sigma
            )

        # Forward through ensemble
        g_combined, _ = self.ensemble(X, Y, score)

        # Controlled estimator: h_i - g_i
        residual = h_x.T - g_combined

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
        if self.qofi == "var":
            # Include mu_post in dataset for variance mode
            train_dataset = torch.utils.data.TensorDataset(
                self.X_train, self.Y_train, self.mu_post_train
            )
            val_dataset = torch.utils.data.TensorDataset(
                self.X_val, self.Y_val, self.mu_post_val
            )
        else:
            # Standard dataset for mean mode
            train_dataset = torch.utils.data.TensorDataset(
                self.X_train, self.Y_train
            )
            val_dataset = torch.utils.data.TensorDataset(self.X_val, self.Y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True
        )
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
                    if self.qofi == "var":
                        for X_val, Y_val, mu_post_val in val_loader:
                            loss, _, _ = self.compute_loss(
                                X_val, Y_val, mu_post_val
                            )
                            val_loss += loss.item()
                    else:
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

                if self.qofi == "var":
                    for X, Y, mu_post_batch in batch_pbar:
                        # Forward pass - use compute_loss to respect qofi setting
                        loss, g_combined, residual = self.compute_loss(
                            X, Y, mu_post_batch
                        )

                        # Backward for ensemble parameters
                        self.optimizer.zero_grad()
                        loss.backward()
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.ensemble.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        # Record training loss for this batch
                        current_lr = self.lr_scheduler.compute_lr()
                        current_train_loss = loss.item()
                        self.train_obj.append(current_train_loss)

                        # Update batch progress bar with current batch loss
                        batch_pbar.set_postfix(
                            {
                                "train_loss": f"{current_train_loss:.6f}",
                                "val_loss": f"{val_loss:.6f}",
                                "lr": f"{current_lr:.6f}",
                            }
                        )
                else:
                    for X, Y in batch_pbar:
                        # Forward pass - use compute_loss to respect qofi setting
                        loss, g_combined, residual = self.compute_loss(X, Y)

                        # Backward for ensemble parameters
                        self.optimizer.zero_grad()
                        loss.backward()
                        # Clip gradients to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.ensemble.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        # Record training loss for this batch
                        current_lr = self.lr_scheduler.compute_lr()
                        current_train_loss = loss.item()
                        self.train_obj.append(current_train_loss)

                        # Update batch progress bar with current batch loss
                        batch_pbar.set_postfix(
                            {
                                "train_loss": f"{current_train_loss:.6f}",
                                "val_loss": f"{val_loss:.6f}",
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
                    }
                )

                # Save checkpoint
                if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                    torch.save(
                        {
                            "ensemble_state_dict": self.ensemble.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "train_obj": self.train_obj,
                            "val_obj": self.val_obj,
                            "mu": self.rosenbrock_dist.mu,
                            "a": self.rosenbrock_dist.a,
                            "b": self.rosenbrock_dist.b,
                            "n1": self.rosenbrock_dist.n1,
                            "n2": self.rosenbrock_dist.n2,
                            "input_dim": self.n_dim,
                            "sigma": self.sigma,
                            "split_config": self.split_config,
                            "layer_type": self.layer_type,
                        },
                        os.path.join(
                            checkpointsdir(args.experiment),
                            "checkpoint_" + str(epoch) + ".pth",
                        ),
                    )

        print("\n=== Training Complete ===")
        print(f"Final training loss: {self.train_obj[-1]:.6f}")
        print(f"Final validation loss: {self.val_obj[-1]:.6f}")

    def sample_posterior_psgld_light(
        self, Y_obs: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Lightweight MCMC for preprocessing (faster, fewer samples).

        Args:
            Y_obs: Observation [n_dim]
            num_samples: Number of samples to generate

        Returns:
            samples: Posterior samples [num_samples, n_dim]
        """
        # Lighter hyperparameters for preprocessing
        lr_initial = 0.1
        lr_final = 0.01
        thinning = 1
        max_itr = 2 * num_samples

        # Initialize from prior
        x = self.rosenbrock_dist.sample(1, device=self.device).squeeze()
        x.requires_grad_(True)

        # Create optimizer with initial LR
        optimizer = pSGLD([x], lr=lr_initial)

        # Create LR scheduler
        lr_scheduler = CustomLRScheduler(
            optimizer,
            initial_lr=lr_initial,
            final_lr=lr_final,
            max_step=max_itr,
        )

        # Collect ALL samples during MCMC
        all_samples = []

        for itr in range(max_itr):
            # Negative log likelihood
            neg_log_likelihood = (
                0.5 * ((Y_obs - x) ** 2).sum() / (self.sigma**2)
            )

            # Negative log prior
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

            # Store sample
            all_samples.append(x.detach().clone())

        # Convert to tensor
        all_samples = torch.stack(all_samples)

        # Burnin: Remove first half
        burnin_iters = max_itr // 2
        samples_after_burnin = all_samples[burnin_iters:]

        # Thinning
        samples_thinned = samples_after_burnin[::thinning]

        return samples_thinned

    def sample_posterior_psgld_light_batch(
        self, Y_obs_batch: torch.Tensor, num_samples: int
    ) -> tuple:
        """Lightweight MCMC for preprocessing in parallel batches.

        Args:
            Y_obs_batch: Batch of observations [batch_size, n_dim]
            num_samples: Number of samples to generate per observation

        Returns:
            posterior_samples: Samples from posterior [batch_size, num_samples, n_dim]
            mu_post_batch: Estimated posterior means [batch_size, n_dim]
        """
        batch_size = Y_obs_batch.shape[0]

        # Lighter hyperparameters for preprocessing
        lr_initial = 0.1
        lr_final = 0.01
        thinning = 1
        max_itr = 2 * num_samples

        # Initialize from prior for all batch elements
        x = self.rosenbrock_dist.sample(
            batch_size, device=self.device
        )  # [batch_size, n_dim]
        x.requires_grad_(True)

        # Create optimizer with initial LR
        optimizer = pSGLD([x], lr=lr_initial)

        # Create LR scheduler
        lr_scheduler = CustomLRScheduler(
            optimizer,
            initial_lr=lr_initial,
            final_lr=lr_final,
            max_step=max_itr,
        )

        # Collect ALL samples during MCMC
        all_samples = []

        for itr in range(max_itr):
            # Negative log likelihood for each sample in batch
            neg_log_likelihood = (
                0.5 * ((Y_obs_batch - x) ** 2).sum(dim=1) / (self.sigma**2)
            )  # [batch_size]

            # Negative log prior for each sample
            neg_log_prior = -self.rosenbrock_dist.logpdf(x)  # [batch_size]

            # Total energy
            energy = (neg_log_likelihood + neg_log_prior).sum()  # Scalar

            # Backward pass
            optimizer.zero_grad()
            energy.backward()
            lr_scheduler.step()

            # pSGLD step
            optimizer.step()

            # Store sample
            all_samples.append(x.detach().clone())

        # Convert to tensor [max_itr, batch_size, n_dim]
        all_samples = torch.stack(all_samples)

        # Burnin: Remove first half
        burnin_iters = max_itr // 2
        samples_after_burnin = all_samples[
            burnin_iters:
        ]  # [num_itr_after_burnin, batch_size, n_dim]

        # Thinning
        samples_thinned = samples_after_burnin[
            ::thinning
        ]  # [num_samples, batch_size, n_dim]

        # Transpose to [batch_size, num_samples, n_dim]
        samples_thinned = samples_thinned.transpose(0, 1)

        # Estimate posterior means for each observation
        mu_post_batch = samples_thinned.mean(dim=1)  # [batch_size, n_dim]

        return samples_thinned, mu_post_batch

    def sample_posterior_psgld(
        self, Y_obs: torch.Tensor, num_samples: int, args: argparse.Namespace
    ) -> torch.Tensor:
        """Sample from posterior using pSGLD with polynomial LR decay.

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

        # Burnin: Remove first half of iterations
        burnin_iters = max_itr // 2
        samples_after_burnin = all_samples[burnin_iters:]  # Keep second half

        print(f"Removed first {burnin_iters} iterations (burnin)")
        print(f"Remaining: {len(samples_after_burnin)} samples")

        # Thinning: Keep every 10th sample
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
            if self.use_ddpm:
                score = extract_score_from_ddpm(
                    self.ddpm_model,
                    self.ddpm_noise_scheduler,
                    X_samples,
                    Y_samples,
                    self.device,
                    timestep=1,
                )
            else:
                score = compute_score_posterior_rosenbrock(
                    X_samples, Y_samples, self.rosenbrock_dist, self.sigma
                )
            g_combined, all_g = self.ensemble(X_samples, Y_samples, score)

        print("Ensemble Control Variates:")
        # Compute h(x) for correlation calculation
        posterior_mean = X_samples.mean(dim=0)
        if self.qofi == "mean":
            h_x = X_samples
        elif self.qofi == "var":
            h_x = (X_samples - posterior_mean) ** 2
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

        for k in range(self.n_dim):
            mean_g = g_combined[k, :].mean().item()
            std_g = g_combined[k, :].std().item()
            corr = torch.corrcoef(torch.stack([h_x[:, k], g_combined[k, :]]))[
                0, 1
            ].item()
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

        # Determine true value for variance reduction analysis
        if self.qofi == "mean":
            # Use posterior samples mean as "true" value
            true_qoi = posterior_mean
        elif self.qofi == "var":
            # Use empirical posterior variance as "true" value
            true_qoi = X_samples.var(dim=0)
            print(f"\nQuantity of Interest: Posterior variance (empirical)")
            print(f"True value: {true_qoi}")
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

        # Plot posterior vs prior distributions
        plot_posterior_vs_prior(
            X_posterior=X_samples,
            rosenbrock_dist=self.rosenbrock_dist,
            save_path=os.path.join(plotsdir(args.experiment), "posterior_vs_prior.png"),
        )

        # Variance reduction analysis
        self.analyze_variance_reduction(
            X_samples, g_combined, all_g, true_qoi, args, mu_post=posterior_mean
        )

        # Generate plots
        h_x = compute_h_x(X_samples, self.qofi, posterior_mean)
        plot_results(
            plots_dir=plotsdir(args.experiment),
            train_obj=self.train_obj,
            val_obj=self.val_obj,
            sample_sizes=self.sample_sizes,
            vanilla_mse=self.vanilla_mse,
            cv_mse=self.cv_mse,
            h_x=h_x,
            g_combined=g_combined,
            all_g=all_g,
            true_mean=self.true_mean,
            title_suffix=" (Rosenbrock)",
        )

    def analyze_variance_reduction(
        self,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g: list,
        true_mean: torch.Tensor,
        args: argparse.Namespace,
        mu_post: torch.Tensor = None,
    ) -> None:
        """Compute variance reduction metrics.

        Args:
            X_samples: Posterior samples [num_samples, n_dim]
            g_combined: Combined control variates [n_dim, num_samples]
            all_g: List of individual layer CVs
            true_mean: Ground truth posterior mean/variance [n_dim]
            args: Configuration
            mu_post: Posterior mean for variance mode [n_dim], optional
        """
        print("\n=== Variance Reduction Analysis ===")

        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_trials = 1000

        vanilla_mse = np.zeros((len(sample_sizes), self.n_dim))
        cv_mse = np.zeros((len(sample_sizes), self.n_dim))

        # Quantity of interest: h(x)
        if self.qofi == "mean":
            h_x = X_samples  # h(x) = x
        elif self.qofi == "var":
            # h(x) = (x - mu_post)^2 for pointwise variance
            if mu_post is None:
                raise ValueError("mu_post must be provided when qofi='var'")
            h_x = (X_samples - mu_post) ** 2
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

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
                cv_estimates[trial] = cv_residual.mean(dim=1).cpu().numpy()

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

    # Always run test
    rosenbrock_cv.test(args)

    upload_to_cloud(args)
