"""Rosenbrock ensemble CV using CNF-learned score and samples.

Trains an ensemble of CNCV coupling layers using the score from a pre-trained
Conditional Normalizing Flow. At test time, uses CNF samples (no MCMC needed).

This is the Rosenbrock counterpart to gaussian_cnf_cv.py: CNCV with a learned
score on a non-Gaussian posterior.
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

from cncv.models.conditional_nf import ConditionalNF
from cncv.models import (
    create_shared_hierarchical_ensemble,
    create_hierarchical_ensemble,
    load_ensemble_checkpoint,
)
from cncv.utils.rosenbrock import RosenbrockDistribution, compute_score_posterior_rosenbrock
from cncv.utils import CustomLRScheduler
from cncv.plotting import plot_results

CONFIG_FILE = "rosenbrock_cnf_cv.json"
CNF_CONFIG_FILE = "cnf_rosenbrock.json"


class RosenbrockCNFCV:
    """Ensemble CV trained with CNF score, evaluated with CNF samples."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Device
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Problem setup - Rosenbrock prior
        self.n_dim = int(args.input_dim)
        n1 = 2
        n2 = self.n_dim - 1

        self.rosenbrock_dist = RosenbrockDistribution(
            mu=args.mu, a=args.a, b=args.b, n1=n1, n2=n2
        )
        self.sigma = float(args.sigma)

        print("\n=== Problem Setup ===")
        print(f"Dimension: {self.n_dim} (n1={n1}, n2={n2})")
        print(f"Rosenbrock mu: {self.rosenbrock_dist.mu}")
        print(f"Rosenbrock a: {self.rosenbrock_dist.a}")
        print(f"Rosenbrock b: {self.rosenbrock_dist.b}")
        print(f"Observation noise sigma: {self.sigma}")

        # Restore seed for data generation
        torch.manual_seed(args.seed)

        # --- Load pre-trained CNF ---
        print("\n=== Loading Pre-trained CNF ===")
        cnf_config_file = getattr(args, "cnf_config", CNF_CONFIG_FILE)
        cnf_args = setup_environment(
            cnf_config_file,
            ignore_arg_list=["experiment_name", "gpu_id", "phase", "seed", "testing_epoch"],
        )
        if cnf_args.testing_epoch == -1:
            cnf_epoch = cnf_args.max_epochs - 1
        else:
            cnf_epoch = cnf_args.testing_epoch

        cnf_checkpoint_path = os.path.join(
            checkpointsdir(cnf_args.experiment),
            f"checkpoint_{cnf_epoch}.pth",
        )
        print(f"CNF checkpoint: {cnf_checkpoint_path}")

        self.cnf = ConditionalNF(
            n_in=self.n_dim,
            n_cond=self.n_dim,
            n_hidden=cnf_args.n_hidden,
            n_flow_layers=cnf_args.n_flow_layers,
            depth=cnf_args.depth,
            n_mlp_layers=cnf_args.n_mlp_layers,
        ).to(self.device)

        cnf_ckpt = torch.load(cnf_checkpoint_path, map_location=self.device, weights_only=False)
        self.cnf.load_state_dict(cnf_ckpt["model_state_dict"])
        self.cnf.eval()
        for p in self.cnf.parameters():
            p.requires_grad_(False)

        cnf_params = sum(p.numel() for p in self.cnf.parameters())
        print(f"CNF loaded (epoch {cnf_ckpt['epoch']}, {cnf_params:,} params, frozen)")

        # --- Create ensemble CV ---
        print("\n=== Creating Ensemble ===")
        self.layer_type = getattr(args, "layer_type", "shared_hierarchical")
        depth = getattr(args, "depth", None) or None

        if self.layer_type == "shared_hierarchical":
            print(f"Using shared hierarchical ensemble with {args.n_ensemble_members} members")
            self.ensemble = create_shared_hierarchical_ensemble(
                self.n_dim, self.n_dim, args.n_hidden,
                args.n_ensemble_members, n_layers=args.n_layers,
                depth=depth, n_cv=self.n_dim, seed=1,
            ).to(self.device)
        elif self.layer_type == "hierarchical":
            print(f"Using hierarchical ensemble with {args.n_ensemble_members} members")
            self.ensemble = create_hierarchical_ensemble(
                self.n_dim, self.n_dim, args.n_hidden,
                args.n_ensemble_members, n_layers=args.n_layers,
                depth=depth, n_cv=self.n_dim, seed=1,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        num_params = sum(p.numel() for p in self.ensemble.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params}")

        # Optimizer
        params = list(self.ensemble.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.lr)

        # Learning rate scheduler
        num_batches_per_epoch = args.num_train // args.batchsize
        max_step = num_batches_per_epoch * args.max_epochs
        self.lr_scheduler = CustomLRScheduler(
            self.optimizer, initial_lr=args.lr, final_lr=args.lr_final, max_step=max_step,
        )

        # --- Generate training data from joint p(x,y) ---
        print("\n=== Generating Training Data ===")
        print(f"Training samples: {args.num_train}")
        print(f"Validation samples: {args.num_val}")

        self.X_train = self.rosenbrock_dist.sample(
            args.num_train, device=self.device
        )
        self.Y_train = self.X_train + self.sigma * torch.randn_like(self.X_train)

        self.X_val = self.rosenbrock_dist.sample(
            args.num_val, device=self.device
        )
        self.Y_val = self.X_val + self.sigma * torch.randn_like(self.X_val)

        # Score mode
        print("\n=== Score Mode: CNF (learned) ===")
        print("Using CNF score computation")

        self.train_obj = []
        self.val_obj = []

    def load_checkpoint(self, args: argparse.Namespace) -> None:
        """Load model checkpoint."""
        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            f"checkpoint_{args.testing_epoch}.pth",
        )
        checkpoint = torch.load(file_to_load, map_location=self.device, weights_only=False)
        self.ensemble.load_state_dict(checkpoint["ensemble_state_dict"])

        # Validate dimensions match
        checkpoint_dim = checkpoint.get("input_dim", self.n_dim)
        if checkpoint_dim != self.n_dim:
            raise ValueError(
                f"Dimension mismatch! Checkpoint was trained with input_dim={checkpoint_dim}, "
                f"but current config has input_dim={self.n_dim}."
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
        self.sigma = checkpoint["sigma"]

        if not args.testing_epoch == checkpoint["epoch"]:
            raise ValueError("Inconsistent filename and loaded checkpoint epoch.")

        print(f"Validated: Dimension = {self.n_dim}")
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def compute_loss(self, X, Y):
        """Compute ensemble CV loss using CNF score.

        Args:
            X: Latent variables [batch_size, n_dim]
            Y: Observations [batch_size, n_dim]

        Returns:
            loss: Scalar loss
            g_combined: Combined control variate [n_dim, batch_size]
            residual: h - g [n_dim, batch_size]
        """
        h_x = X  # QoI = posterior mean

        # CNF score (no grad through CNF -- it's frozen)
        score = self.cnf.score(X, Y)

        g_combined, _ = self.ensemble(X, Y, score)

        # Controlled estimator: h - g
        residual = h_x.T - g_combined

        # Loss: minimize variance
        loss = (residual ** 2).mean()
        return loss, g_combined, residual

    def train(self, args):
        """Training loop."""
        print("\n=== Starting Training ===")
        print(f"Max epochs: {args.max_epochs}")
        print(f"Batch size: {args.batchsize}")
        print(f"Learning rate: {args.lr} -> {args.lr_final}")

        # DataLoaders
        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True,
        )

        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.Y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batchsize, shuffle=False,
        )

        num_batches = len(train_loader)
        print(f"Batches per epoch: {num_batches}")

        # Training loop
        val_loss = 0.0

        with tqdm(range(args.max_epochs), unit="epoch", colour="#B5F2A9") as epoch_pbar:
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

                batch_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{args.max_epochs}",
                    leave=False, colour="#87CEEB",
                )

                for X, Y in batch_pbar:
                    loss, g_combined, residual = self.compute_loss(X, Y)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    current_lr = self.lr_scheduler.compute_lr()
                    current_train_loss = loss.item()
                    self.train_obj.append(current_train_loss)

                    batch_pbar.set_postfix({
                        "train_loss": f"{current_train_loss:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "lr": f"{current_lr:.6f}",
                    })

                # Update epoch progress bar
                avg_train_loss = np.mean(self.train_obj[-num_batches:])
                epoch_pbar.set_postfix({
                    "avg_train": f"{avg_train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                })

                # Save checkpoint
                if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                    torch.save({
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
                        "sigma": self.sigma,
                        "input_dim": self.n_dim,
                        "layer_type": self.layer_type,
                    }, os.path.join(
                        checkpointsdir(args.experiment),
                        f"checkpoint_{epoch}.pth",
                    ))

        print("\n=== Training Complete ===")
        print(f"Final training loss: {self.train_obj[-1]:.6f}")
        print(f"Final validation loss: {self.val_obj[-1]:.6f}")

    def test(self, args):
        """Evaluate variance reduction on test data using CNF samples."""
        # Load checkpoint
        self.load_checkpoint(args)

        # Set to evaluation mode
        self.ensemble.eval()

        print("\n=== Generating Test Data ===")

        # Generate test observation
        torch.manual_seed(args.seed + 1000)
        X_true = self.rosenbrock_dist.sample(1, device=self.device).squeeze(0)
        Y_obs = X_true + self.sigma * torch.randn(
            self.n_dim, device=self.device
        )

        print(f"True parameter X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        # Sample from CNF posterior
        num_samples = args.num_samples
        print(f"\nSampling {num_samples} from CNF posterior...")
        with torch.no_grad():
            X_samples = self.cnf.sample(Y_obs, num_samples)

        # Report CNF sample quality
        sample_mean = X_samples.mean(dim=0)
        sample_std = X_samples.std(dim=0)
        print(f"CNF sample mean: {sample_mean}")
        print(f"CNF sample std:  {sample_std}")

        Y_samples = Y_obs.repeat(num_samples, 1)

        # Compute control variates
        print("\n=== Computing Control Variates ===")
        with torch.no_grad():
            score = self.cnf.score(X_samples, Y_samples)
            g_combined, all_g = self.ensemble(X_samples, Y_samples, score)

        h_x = X_samples

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
        print("Ensemble CV should have E[g_k|Y] ~ 0")
        for k in range(self.n_dim):
            ratio = (
                abs(g_combined[k, :].mean().item())
                / g_combined[k, :].std().item()
            )
            print(
                f"  Component {k}: |E[g_{k}]| / Std[g_{k}] = {ratio:.4f} (should be << 1)"
            )

        # Variance reduction analysis
        # Ground truth: use large sample mean (no closed form for Rosenbrock posterior)
        true_qoi = sample_mean
        print(f"\nGround truth QoI (large-sample mean): {true_qoi}")

        self.analyze_variance_reduction(
            X_samples, g_combined, all_g, true_qoi, args,
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
            title_suffix=" (Rosenbrock, CNF score)",
        )

    def analyze_variance_reduction(
        self,
        X_samples: torch.Tensor,
        g_combined: torch.Tensor,
        all_g,
        true_mean: torch.Tensor,
        args: argparse.Namespace,
    ) -> None:
        """Compute variance reduction metrics."""
        print("\n=== Variance Reduction Analysis ===")

        sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        num_trials = 1000

        vanilla_mse = np.zeros((len(sample_sizes), self.n_dim))
        cv_mse = np.zeros((len(sample_sizes), self.n_dim))

        h_x = X_samples

        # Store for plotting
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
    runner = RosenbrockCNFCV(args)

    # Train or test based on phase
    if args.phase == "train":
        runner.train(args)

    # Always run test
    runner.test(args)

    upload_to_cloud(args)
