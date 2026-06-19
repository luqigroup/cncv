"""Gaussian ensemble CV example.

This script trains and evaluates an ensemble of coupling layers with forward
and reverse splits to learn control variates for variance reduction in Bayesian
posterior estimation.
"""

import argparse
import os
import torch
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
    compute_score_posterior_gaussian,
    compute_exact_posterior,
    load_ensemble_checkpoint,
    get_split_config,
    plot_results,
    compute_h_x,
)
from cncv.models.ddpm_mlp import ConditionalScoreMLPSimple
from cncv.diffusion import (
    NoiseScheduler,
    extract_score_from_ddpm,
    load_ddpm_checkpoint,
    sample_from_ddpm,
    diffuse_samples,
)
from cncv.utils import CustomLRScheduler

CONFIG_FILE = "gaussian_ensemble_cv.json"
DDPM_CONFIG_FILE = "ddpm_gaussian.json"
DIFFUSION_TIMESTEP = 50  # Timestep for diffused CV


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

        # Validate and store quantity of interest
        if args.qofi not in ["mean", "var"]:
            raise ValueError(f"qofi must be 'mean' or 'var', got '{args.qofi}'")
        self.qofi = args.qofi
        print(f"Quantity of interest (qofi): {self.qofi}")

        # Problem setup
        self.n_dim = int(args.input_dim)
        self.mu_prior = torch.zeros(
            self.n_dim, dtype=torch.float32, device=self.device
        )

        # Generate random positive definite covariance matrix
        # Create a diagonal matrix with positive values
        diag_values = torch.linspace(1.0, 2.0, self.n_dim)
        D = torch.diag(diag_values).to(device=self.device, dtype=torch.float32)

        # Add some off-diagonal structure for correlation
        if self.n_dim > 1:
            # Create a random orthogonal matrix for rotation
            torch.manual_seed(42)  # For reproducibility
            A = torch.randn(
                self.n_dim, self.n_dim, device=self.device, dtype=torch.float32
            )
            Q, _ = torch.linalg.qr(A)
            # Construct Sigma = Q @ D @ Q^T (positive definite)
            self.Sigma_prior = Q @ D @ Q.T
        else:
            self.Sigma_prior = D

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

        self.layer_type = getattr(args, "layer_type", "flat")

        if self.layer_type == "shared_hierarchical":
            print(f"Using shared hierarchical ensemble with {args.n_ensemble_members} members")
            depth = getattr(args, "depth", None) or None
            self.ensemble = create_shared_hierarchical_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                n_layers=args.n_layers,
                depth=depth,
                n_cv=self.n_dim,
                seed=12,
            ).to(self.device)
            self.split_config = None
        elif self.layer_type == "hierarchical":
            print(f"Using hierarchical ensemble with {args.n_ensemble_members} members")
            depth = getattr(args, "depth", None) or None  # 0 means auto
            self.ensemble = create_hierarchical_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                n_layers=args.n_layers,
                depth=depth,
                n_cv=self.n_dim,
                seed=12,
            ).to(self.device)
            self.split_config = None  # Hierarchical doesn't use split configs
        else:
            print(f"Using random split ensemble with {args.n_ensemble_members} members")
            self.ensemble = create_random_split_ensemble(
                self.n_dim,
                self.n_dim,
                args.n_hidden,
                args.n_ensemble_members,
                args.n_layers,
                n_cv=self.n_dim,
                seed=12,
            ).to(self.device)
            self.split_config = get_split_config(self.ensemble)

        print(f"\nNumber of ensemble members: {len(self.ensemble.layers)}")

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
            noise = torch.randn(
                args.num_train,
                self.n_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self.X_train = self.mu_prior + (self.L_prior @ noise.T).T
            self.Y_train = self.X_train + self.sigma * torch.randn_like(
                self.X_train
            )

            # Validation data
            noise_val = torch.randn(
                args.num_val,
                self.n_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self.X_val = self.mu_prior + (self.L_prior @ noise_val.T).T
            self.Y_val = self.X_val + self.sigma * torch.randn_like(self.X_val)

        elif self.qofi == "var":
            # For variance mode: sample from posterior p(x|y)
            # First generate observations Y
            X_true = (
                self.mu_prior
                + (
                    self.L_prior
                    @ torch.randn(
                        args.num_train,
                        self.n_dim,
                        dtype=torch.float32,
                        device=self.device,
                    ).T
                ).T
            )
            self.Y_train = X_true + self.sigma * torch.randn_like(X_true)

            # Compute posterior parameters (same for all Y, independent of observations)
            I = torch.eye(self.n_dim, device=self.device, dtype=torch.float32)
            Sigma_post = torch.linalg.inv(
                self.Sigma_prior_inv + (1 / self.sigma**2) * I
            )
            L_post = torch.linalg.cholesky(Sigma_post)

            # Sample X from posterior p(x|y) for each y
            mu_post_train = (
                Sigma_post
                @ (
                    self.Sigma_prior_inv @ self.mu_prior[:, None]
                    + (1 / self.sigma**2) * self.Y_train.T
                )
            ).T  # Shape: [num_train, n_dim]

            # Sample from N(mu_post, Sigma_post)
            self.X_train = (
                mu_post_train
                + (
                    L_post
                    @ torch.randn(
                        args.num_train,
                        self.n_dim,
                        dtype=torch.float32,
                        device=self.device,
                    ).T
                ).T
            )

            # Validation data
            X_true_val = (
                self.mu_prior
                + (
                    self.L_prior
                    @ torch.randn(
                        args.num_val,
                        self.n_dim,
                        dtype=torch.float32,
                        device=self.device,
                    ).T
                ).T
            )
            self.Y_val = X_true_val + self.sigma * torch.randn_like(X_true_val)

            mu_post_val = (
                Sigma_post
                @ (
                    self.Sigma_prior_inv @ self.mu_prior[:, None]
                    + (1 / self.sigma**2) * self.Y_val.T
                )
            ).T

            self.X_val = (
                mu_post_val
                + (
                    L_post
                    @ torch.randn(
                        args.num_val,
                        self.n_dim,
                        dtype=torch.float32,
                        device=self.device,
                    ).T
                ).T
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

        # Validate dimensions match
        checkpoint_mu_prior = checkpoint["mu_prior"]
        checkpoint_dim = checkpoint.get(
            "input_dim", checkpoint_mu_prior.shape[0]
        )  # Use saved dim or infer
        if checkpoint_dim != self.n_dim:
            raise ValueError(
                f"Dimension mismatch! Checkpoint was trained with input_dim={checkpoint_dim}, "
                f"but current config has input_dim={self.n_dim}. "
                f"Please use the same input_dim for training and testing."
            )
        self.train_obj = checkpoint["train_obj"]
        self.val_obj = checkpoint["val_obj"]

        # Restore problem parameters
        self.mu_prior = checkpoint_mu_prior.to(self.device)
        self.Sigma_prior = checkpoint["Sigma_prior"].to(self.device)
        self.L_prior = torch.linalg.cholesky(self.Sigma_prior)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
        self.sigma = checkpoint["sigma"]

        if not args.testing_epoch == checkpoint["epoch"]:
            raise ValueError(
                "Inconsistent filename and loaded checkpoint epoch."
            )

        print(f"Validated: Dimension = {self.n_dim}")

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def compute_loss(self, X: torch.Tensor, Y: torch.Tensor) -> tuple:
        """Compute ensemble CV loss.

        Args:
            X: Latent variables [batch_size, n_dim]
            Y: Observations [batch_size, n_dim]

        Returns:
            loss: Scalar loss
            g_combined: Combined control variate [n_dim, batch_size]
        """
        if self.use_ddpm:
            # Work at diffused level t=DIFFUSION_TIMESTEP for DDPM
            # Diffuse samples: x_t = √ᾱ * x + √(1-ᾱ) * ε
            X_t, _ = diffuse_samples(
                X, self.ddpm_noise_scheduler, DIFFUSION_TIMESTEP, self.device
            )

            # Quantity of interest at diffused level: h(x_t)
            if self.qofi == "mean":
                # h(x_t) = x_t (will rescale by 1/√ᾱ at test time)
                h_x = X_t
            elif self.qofi == "var":
                # For variance, compute at diffused level
                # h(x_t) = (x_t - μ_t)^2 where μ_t = √ᾱ * μ_post
                Sigma_prior_inv = self.Sigma_prior_inv
                I = torch.eye(
                    self.n_dim, device=self.device, dtype=torch.float32
                )
                Sigma_post = torch.linalg.inv(
                    Sigma_prior_inv + (1 / self.sigma**2) * I
                )
                mu_post_batch = (
                    Sigma_post
                    @ (
                        Sigma_prior_inv @ self.mu_prior[:, None]
                        + (1 / self.sigma**2) * Y.T
                    )
                ).T
                sqrt_alpha_bar = self.ddpm_noise_scheduler.sqrt_alphas_cumprod[
                    DIFFUSION_TIMESTEP
                ]
                mu_t_batch = sqrt_alpha_bar * mu_post_batch
                h_x = (X_t - mu_t_batch) ** 2
            else:
                raise ValueError(f"Unknown qofi: {self.qofi}")

            # Compute score at diffused level: ∇log p_t(x_t|y)
            score = extract_score_from_ddpm(
                self.ddpm_model,
                self.ddpm_noise_scheduler,
                X_t,
                Y,
                self.device,
                timestep=DIFFUSION_TIMESTEP,
            )

            # Forward through ensemble (at diffused level)
            g_combined, _ = self.ensemble(X_t, Y, score)
        else:
            # Analytical mode: work at clean level
            if self.qofi == "mean":
                h_x = X
            elif self.qofi == "var":
                Sigma_prior_inv = self.Sigma_prior_inv
                I = torch.eye(
                    self.n_dim, device=self.device, dtype=torch.float32
                )
                Sigma_post = torch.linalg.inv(
                    Sigma_prior_inv + (1 / self.sigma**2) * I
                )
                mu_post_batch = (
                    Sigma_post
                    @ (
                        Sigma_prior_inv @ self.mu_prior[:, None]
                        + (1 / self.sigma**2) * Y.T
                    )
                ).T
                h_x = (X - mu_post_batch) ** 2
            else:
                raise ValueError(f"Unknown qofi: {self.qofi}")

            score = compute_score_posterior_gaussian(
                X, Y, self.mu_prior, self.Sigma_prior_inv, self.sigma
            )
            g_combined, _ = self.ensemble(X, Y, score)

        # Controlled estimator: h - g
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
                    # Forward pass - use compute_loss to respect qofi setting
                    loss, g_combined, residual = self.compute_loss(X, Y)

                    # Backward for ensemble parameters (pure autograd!)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability at high d / deep trees
                    grad_clip = float(getattr(args, "grad_clip", 5.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.ensemble.parameters(), grad_clip)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Record training loss for this batch
                    current_lr = self.lr_scheduler.compute_lr()
                    current_train_loss = loss.item()
                    self.train_obj.append(current_train_loss)

                    # Update batch progress bar with current batch loss
                    postfix_dict = {
                        "train_loss": f"{current_train_loss:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "lr": f"{current_lr:.6f}",
                    }

                    batch_pbar.set_postfix(postfix_dict)

                # Update epoch progress bar
                avg_train_loss = np.mean(
                    self.train_obj[-num_batches:]
                )  # Average of last epoch

                postfix_dict = {
                    "avg_train": f"{avg_train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                }

                epoch_pbar.set_postfix(postfix_dict)

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
                            "mu_prior": self.mu_prior.cpu(),
                            "Sigma_prior": self.Sigma_prior.cpu(),
                            "sigma": self.sigma,
                            "input_dim": self.n_dim,
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

        if self.qofi == "mean":
            print(f"\nQuantity of Interest: Posterior mean")
            print(f"True value: {mu_post}")
        elif self.qofi == "var":
            print(f"\nQuantity of Interest: Posterior variance (diagonal)")
            print(f"True value: {torch.diag(Sigma_post)}")

        # Sample from posterior
        num_samples = args.num_samples
        if self.use_ddpm:
            # Sample from learned DDPM posterior
            print(f"\nSampling {num_samples} from DDPM posterior...")
            X_samples_clean = sample_from_ddpm(
                self.ddpm_model,
                self.ddpm_noise_scheduler,
                Y_obs,
                num_samples,
                self.device,
                show_progress=True,
            )

            # Diffuse samples to t=DIFFUSION_TIMESTEP level
            print(f"\nDiffusing samples to t={DIFFUSION_TIMESTEP}...")
            X_samples, diffusion_noise = diffuse_samples(
                X_samples_clean,
                self.ddpm_noise_scheduler,
                DIFFUSION_TIMESTEP,
                self.device,
            )

            # Get diffusion parameters for rescaling
            sqrt_alpha_bar = self.ddpm_noise_scheduler.sqrt_alphas_cumprod[
                DIFFUSION_TIMESTEP
            ]
            alpha_bar = self.ddpm_noise_scheduler.alphas_cumprod[
                DIFFUSION_TIMESTEP
            ]

            # Compute analytical diffused distribution parameters for comparison
            mu_t_analytical = sqrt_alpha_bar * mu_post
            I = torch.eye(self.n_dim, device=self.device)
            Sigma_t_analytical = alpha_bar * Sigma_post + (1 - alpha_bar) * I

            # Plot posterior diagnostic at t=DIFFUSION_TIMESTEP
            self.plot_diffused_posterior_diagnostic(
                X_samples, mu_t_analytical, Sigma_t_analytical, args
            )
        else:
            # Sample from exact analytical posterior
            print(f"\nSampling {num_samples} from exact posterior...")
            L_post = torch.linalg.cholesky(Sigma_post)
            noise = torch.randn(num_samples, self.n_dim, device=self.device)
            X_samples = mu_post + (L_post @ noise.T).T
            sqrt_alpha_bar = None  # Not used for analytical mode

        Y_samples = Y_obs.repeat(num_samples, 1)

        # Compute control variates
        print("\n=== Computing Control Variates ===")
        with torch.no_grad():
            if self.use_ddpm:
                # Score at diffused level
                score = extract_score_from_ddpm(
                    self.ddpm_model,
                    self.ddpm_noise_scheduler,
                    X_samples,
                    Y_samples,
                    self.device,
                    timestep=DIFFUSION_TIMESTEP,
                )
                # h(x_t) at diffused level
                if self.qofi == "mean":
                    h_x = X_samples  # h(x_t) = x_t
                elif self.qofi == "var":
                    mu_t = sqrt_alpha_bar * mu_post
                    h_x = (X_samples - mu_t) ** 2
                else:
                    raise ValueError(f"Unknown qofi: {self.qofi}")
            else:
                score = compute_score_posterior_gaussian(
                    X_samples,
                    Y_samples,
                    self.mu_prior,
                    self.Sigma_prior_inv,
                    self.sigma,
                )
                if self.qofi == "mean":
                    h_x = X_samples
                elif self.qofi == "var":
                    h_x = (X_samples - mu_post) ** 2
                else:
                    raise ValueError(f"Unknown qofi: {self.qofi}")

            g_combined, all_g = self.ensemble(X_samples, Y_samples, score)

        print("Ensemble Control Variates:")
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

        # Variance reduction analysis
        if self.qofi == "mean":
            if self.use_ddpm:
                # True QoI at diffused level: E[x_t] = √ᾱ * μ_post
                true_qoi = sqrt_alpha_bar * mu_post
            else:
                true_qoi = mu_post
        elif self.qofi == "var":
            if self.use_ddpm:
                # True variance at diffused level: Var[x_t] = ᾱ * Σ_post + (1-ᾱ) * I
                true_qoi = torch.diag(
                    alpha_bar * Sigma_post + (1 - alpha_bar) * I
                )
            else:
                true_qoi = torch.diag(Sigma_post)
        else:
            raise ValueError(f"Unknown qofi: {self.qofi}")

        self.analyze_variance_reduction(
            X_samples,
            g_combined,
            all_g,
            true_qoi,
            args,
            mu_post=mu_post,
            sqrt_alpha_bar=sqrt_alpha_bar if self.use_ddpm else None,
        )

        # Generate plots
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
            title_suffix="",
        )

    def plot_diffused_posterior_diagnostic(
        self,
        X_t_samples: torch.Tensor,
        mu_t: torch.Tensor,
        Sigma_t: torch.Tensor,
        args: argparse.Namespace,
    ) -> None:
        """Plot diagnostic comparing DDPM diffused samples vs analytical diffused distribution.

        Args:
            X_t_samples: Diffused samples from DDPM [num_samples, n_dim]
            mu_t: Analytical mean of diffused distribution [n_dim]
            Sigma_t: Analytical covariance of diffused distribution [n_dim, n_dim]
            args: Configuration arguments
        """
        plots_dir = plotsdir(args.experiment)

        X_np = X_t_samples.cpu().numpy()
        mu_t_np = mu_t.cpu().numpy()
        std_t_np = torch.sqrt(torch.diag(Sigma_t)).cpu().numpy()

        print(
            f"\n=== Diffused Posterior Diagnostic (t={DIFFUSION_TIMESTEP}) ==="
        )
        print(f"DDPM diffused samples mean: {X_t_samples.mean(dim=0)}")
        print(f"DDPM diffused samples std:  {X_t_samples.std(dim=0)}")
        print(f"Analytical diffused mean:   {mu_t}")
        print(f"Analytical diffused std:    {torch.sqrt(torch.diag(Sigma_t))}")

        # Histogram comparison per dimension
        fig, axes = plt.subplots(
            1, min(self.n_dim, 4), figsize=(4 * min(self.n_dim, 4), 4)
        )
        if self.n_dim == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= self.n_dim:
                break

            # DDPM diffused samples histogram
            ax.hist(
                X_np[:, i],
                bins=50,
                density=True,
                alpha=0.7,
                label="DDPM diffused",
                color="#1f77b4",
            )

            # Analytical diffused distribution (Gaussian)
            x_range = np.linspace(X_np[:, i].min(), X_np[:, i].max(), 100)
            gaussian_pdf = (
                1
                / (std_t_np[i] * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x_range - mu_t_np[i]) / std_t_np[i]) ** 2)
            )
            ax.plot(
                x_range,
                gaussian_pdf,
                "r-",
                linewidth=2,
                label="Analytical diffused",
            )
            ax.axvline(mu_t_np[i], color="r", linestyle="--", alpha=0.5)

            ax.set_xlabel(f"$x_{{t,{i + 1}}}$")
            ax.set_ylabel("Density")
            ax.set_title(f"Dim {i} (t={DIFFUSION_TIMESTEP})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plots_dir, f"diffused_posterior_t{DIFFUSION_TIMESTEP}.png"
            ),
            dpi=300,
        )
        plt.close()

        # Scatter plot: DDPM samples vs analytical samples (first 2 dims)
        if self.n_dim >= 2:
            fig, ax = plt.subplots(figsize=(6, 6))

            # DDPM diffused samples
            ax.scatter(
                X_np[:, 0],
                X_np[:, 1],
                alpha=0.3,
                s=5,
                c="#1f77b4",
                label="DDPM diffused",
            )

            # Analytical: sample from diffused distribution for comparison
            L_t = torch.linalg.cholesky(Sigma_t)
            X_t_analytical = (
                mu_t
                + (
                    L_t
                    @ torch.randn(len(X_np), self.n_dim, device=mu_t.device).T
                ).T
            )
            X_t_analytical_np = X_t_analytical.cpu().numpy()
            ax.scatter(
                X_t_analytical_np[:, 0],
                X_t_analytical_np[:, 1],
                alpha=0.3,
                s=5,
                c="#ff7f0e",
                label="Analytical diffused",
            )

            ax.set_xlabel(f"$x_{{t,1}}$")
            ax.set_ylabel(f"$x_{{t,2}}$")
            ax.set_title(f"Diffused Posterior Samples (t={DIFFUSION_TIMESTEP})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    plots_dir,
                    f"diffused_posterior_scatter_t{DIFFUSION_TIMESTEP}.png",
                ),
                dpi=300,
            )
            plt.close()

        print(f"Diffused posterior diagnostic plots saved to {plots_dir}")

    def analyze_variance_reduction(
        self,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g: list,
        true_mean: torch.Tensor,
        args: argparse.Namespace,
        mu_post: torch.Tensor = None,
        sqrt_alpha_bar: float = None,
    ) -> None:
        """Compute variance reduction metrics.

        Args:
            X_samples: Posterior samples [num_samples, n_dim] (at diffused level if DDPM)
            g_combined: Combined control variates [n_dim, num_samples]
            all_g: List of individual layer CVs [n_ensemble_members, n_dim, num_samples]
            true_mean: Ground truth QoI [n_dim] (at diffused level if DDPM)
            args: Configuration
            mu_post: Posterior mean for variance mode [n_dim], optional
            sqrt_alpha_bar: Diffusion scaling factor for DDPM mode, optional
        """
        print("\n=== Variance Reduction Analysis ===")

        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_trials = 1000

        vanilla_mse = np.zeros((len(sample_sizes), self.n_dim))
        cv_mse = np.zeros((len(sample_sizes), self.n_dim))

        # Quantity of interest: h(x) or h(x_t) depending on mode
        if self.use_ddpm and sqrt_alpha_bar is not None:
            # Working at diffused level
            if self.qofi == "mean":
                h_x = X_samples  # h(x_t) = x_t
            elif self.qofi == "var":
                mu_t = sqrt_alpha_bar * mu_post
                h_x = (X_samples - mu_t) ** 2
            else:
                raise ValueError(f"Unknown qofi: {self.qofi}")
        else:
            # Working at clean level
            if self.qofi == "mean":
                h_x = X_samples
            elif self.qofi == "var":
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
        if self.use_ddpm and sqrt_alpha_bar is not None:
            print(f"(Results at diffused level t={DIFFUSION_TIMESTEP})")
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

    # Always run test
    gaussian_cv.test(args)

    upload_to_cloud(args)
