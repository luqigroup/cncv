"""Train a Conditional Normalizing Flow on the Rosenbrock inverse problem.

Learns p(x|y) via maximum likelihood using HINT-style hierarchical coupling
layers. After training, evaluates:
  - Sample quality vs pSGLD posterior samples (banana shape)
  - Score accuracy vs analytical score
  - Log-likelihood on held-out posterior samples
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
from cncv.utils.rosenbrock import RosenbrockDistribution, compute_score_posterior_rosenbrock
from cncv.utils import CustomLRScheduler, pSGLD

CONFIG_FILE = "cnf_rosenbrock.json"


class RosenbrockCNF:
    """Conditional normalizing flow for Rosenbrock posterior learning."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Device setup
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Problem setup - Rosenbrock prior
        self.n_dim = int(args.input_dim)

        # Block structure: n1=2, n2=1 for 2D Rosenbrock
        n1 = 2
        n2 = self.n_dim - 1  # For dim=2: n2=1

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

        # Set seed for data generation
        torch.manual_seed(args.seed)

        # Create CNF model
        self.cnf = ConditionalNF(
            n_in=self.n_dim,
            n_cond=self.n_dim,
            n_hidden=args.n_hidden,
            n_flow_layers=args.n_flow_layers,
            depth=args.depth,
            n_mlp_layers=args.n_mlp_layers,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.cnf.parameters() if p.requires_grad)
        print(f"\n=== CNF Model ===")
        print(f"Flow layers: {args.n_flow_layers}")
        print(f"Tree depth: {args.depth}")
        print(f"Hidden dim: {args.n_hidden}")
        print(f"Total trainable parameters: {num_params}")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.cnf.parameters(), lr=args.lr)

        # LR scheduler
        num_batches_per_epoch = args.num_train // args.batchsize
        max_step = num_batches_per_epoch * args.max_epochs
        self.lr_scheduler = CustomLRScheduler(
            self.optimizer,
            initial_lr=args.lr,
            final_lr=args.lr_final,
            max_step=max_step,
        )

        # Generate training data: joint p(x, y) samples
        print(f"\n=== Generating Training Data ===")
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

        # Logging
        self.train_obj = []
        self.val_obj = []

    def train(self, args: argparse.Namespace) -> None:
        """Training loop: minimize NLL = -E[log p(x|y)]."""
        print("\n=== Starting Training ===")
        print(f"Max epochs: {args.max_epochs}")
        print(f"Batch size: {args.batchsize}")
        print(f"Learning rate: {args.lr} -> {args.lr_final}")

        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.Y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batchsize, shuffle=False
        )

        num_batches = len(train_loader)
        print(f"Batches per epoch: {num_batches}")

        val_loss = 0.0

        with tqdm(
            range(args.max_epochs), unit="epoch", colour="#B5F2A9"
        ) as epoch_pbar:
            for epoch in epoch_pbar:
                # Validation
                self.cnf.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        nll = -self.cnf.log_prob(X_val, Y_val).mean()
                        val_loss += nll.item()
                val_loss /= len(val_loader)
                self.val_obj.append(val_loss)

                # Training
                self.cnf.train()

                batch_pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch + 1}/{args.max_epochs}",
                    leave=False,
                    colour="#87CEEB",
                )

                for X, Y in batch_pbar:
                    loss = -self.cnf.log_prob(X, Y).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.cnf.parameters(), 10.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    current_lr = self.lr_scheduler.compute_lr()
                    current_train_loss = loss.item()
                    self.train_obj.append(current_train_loss)

                    batch_pbar.set_postfix({
                        "nll": f"{current_train_loss:.4f}",
                        "val_nll": f"{val_loss:.4f}",
                        "lr": f"{current_lr:.6f}",
                    })

                avg_train_loss = np.mean(self.train_obj[-num_batches:])
                epoch_pbar.set_postfix({
                    "avg_nll": f"{avg_train_loss:.4f}",
                    "val_nll": f"{val_loss:.4f}",
                })

                # Save checkpoint
                if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                    torch.save(
                        {
                            "model_state_dict": self.cnf.state_dict(),
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
                        },
                        os.path.join(
                            checkpointsdir(args.experiment),
                            f"checkpoint_{epoch}.pth",
                        ),
                    )

        print("\n=== Training Complete ===")
        print(f"Final training NLL: {self.train_obj[-1]:.4f}")
        print(f"Final validation NLL: {self.val_obj[-1]:.4f}")

    def test(self, args: argparse.Namespace) -> None:
        """Evaluate CNF against pSGLD posterior samples."""
        # Load checkpoint
        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            f"checkpoint_{args.testing_epoch}.pth",
        )
        checkpoint = torch.load(file_to_load, map_location=self.device, weights_only=False)
        self.cnf.load_state_dict(checkpoint["model_state_dict"])
        self.train_obj = checkpoint["train_obj"]
        self.val_obj = checkpoint["val_obj"]

        # Restore problem parameters from checkpoint
        self.rosenbrock_dist = RosenbrockDistribution(
            mu=checkpoint["mu"],
            a=checkpoint["a"],
            b=checkpoint["b"],
            n1=checkpoint["n1"],
            n2=checkpoint["n2"],
        )
        self.sigma = checkpoint["sigma"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        self.cnf.eval()

        # Generate test observation
        torch.manual_seed(args.seed + 1000)
        X_true = self.rosenbrock_dist.sample(1, device=self.device).squeeze(0)
        Y_obs = X_true + self.sigma * torch.randn(
            self.n_dim, device=self.device
        )

        print(f"\nTrue parameter X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        num_samples = args.num_samples

        # 1. CNF sample quality
        print(f"\n=== CNF Sample Quality ({num_samples} samples) ===")
        with torch.no_grad():
            x_cnf = self.cnf.sample(Y_obs, num_samples)

        cnf_mean = x_cnf.mean(dim=0)
        cnf_cov = torch.cov(x_cnf.T)

        print(f"CNF sample mean: {cnf_mean}")
        print(f"CNF sample std:  {x_cnf.std(dim=0)}")

        # 2. Compare with pSGLD samples
        print(f"\n=== pSGLD Reference ({num_samples} samples) ===")
        x_mcmc = self._sample_posterior_psgld(Y_obs, num_samples)

        mcmc_mean = x_mcmc.mean(dim=0)
        mcmc_cov = torch.cov(x_mcmc.T)

        print(f"pSGLD sample mean: {mcmc_mean}")
        print(f"pSGLD sample std:  {x_mcmc.std(dim=0)}")

        mean_error = (cnf_mean - mcmc_mean).abs().max().item()
        cov_error = (cnf_cov - mcmc_cov).abs().max().item()
        print(f"\nMax mean error (CNF vs pSGLD): {mean_error:.6f}")
        print(f"Max cov error (CNF vs pSGLD):  {cov_error:.6f}")

        # 3. Score accuracy
        print(f"\n=== Score Accuracy ===")
        x_test = x_cnf[:100]
        Y_test = Y_obs.repeat(100, 1)

        score_cnf = self.cnf.score(x_test, Y_test)
        score_exact = compute_score_posterior_rosenbrock(
            x_test, Y_test, self.rosenbrock_dist, self.sigma
        )

        score_mse = (score_cnf - score_exact).pow(2).mean().item()
        score_rel_error = (
            (score_cnf - score_exact).pow(2).sum(dim=1).sqrt()
            / score_exact.pow(2).sum(dim=1).sqrt()
        ).mean().item()

        print(f"Score MSE: {score_mse:.6f}")
        print(f"Relative score error: {score_rel_error:.6f}")

        # 4. Plots
        plots_dir = plotsdir(args.experiment)

        # Training loss curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.train_obj, alpha=0.3, label="Train (per batch)")
        val_x = np.arange(len(self.val_obj)) * (len(self.train_obj) // max(len(self.val_obj), 1))
        ax.plot(val_x, self.val_obj, "r-", linewidth=2, label="Validation")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NLL")
        ax.set_title("CNF Training Loss (Rosenbrock)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=300)
        plt.close()

        # 2D posterior comparison: CNF vs pSGLD
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # CNF samples
        x_np = x_cnf.cpu().numpy()
        axes[0].scatter(x_np[:, 0], x_np[:, 1], alpha=0.2, s=3, c="#1f77b4")
        axes[0].set_title("CNF Samples")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].grid(True, alpha=0.3)

        # pSGLD samples
        x_mcmc_np = x_mcmc.cpu().numpy()
        axes[1].scatter(x_mcmc_np[:, 0], x_mcmc_np[:, 1], alpha=0.2, s=3, c="#ff7f0e")
        axes[1].set_title("pSGLD Samples")
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "posterior_comparison.png"), dpi=300)
        plt.close()

        # Marginal histograms
        fig, axes = plt.subplots(1, self.n_dim, figsize=(4 * self.n_dim, 4))
        if self.n_dim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.hist(x_cnf[:, i].cpu().numpy(), bins=50, density=True,
                    alpha=0.7, label="CNF", color="#1f77b4")
            ax.hist(x_mcmc[:, i].cpu().numpy(), bins=50, density=True,
                    alpha=0.5, label="pSGLD", color="#ff7f0e")
            ax.set_title(f"Dim {i}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "marginal_histograms.png"), dpi=300)
        plt.close()

        print(f"\nPlots saved to {plots_dir}")

        # Summary
        print(f"\n{'='*50}")
        print(f"SUMMARY")
        print(f"{'='*50}")
        print(f"Max mean error (vs pSGLD): {mean_error:.6f}")
        print(f"Max cov error (vs pSGLD):  {cov_error:.6f}")
        print(f"Score MSE:                 {score_mse:.6f}")
        print(f"Score relative err:        {score_rel_error:.6f}")
        print(f"{'='*50}")

    def _sample_posterior_psgld(self, Y_obs, num_samples):
        """Sample posterior via pSGLD for evaluation."""
        lr_initial = 0.01
        lr_final = 0.001
        max_itr = 10 * num_samples
        thinning = 2

        x = self.rosenbrock_dist.sample(1, device=self.device).squeeze(0)
        x = x.detach().requires_grad_(True)

        optimizer = pSGLD([x], lr=lr_initial)
        lr_scheduler = CustomLRScheduler(optimizer, lr_initial, lr_final, max_itr)

        all_samples = []
        for itr in range(max_itr):
            neg_log_lik = 0.5 * ((Y_obs - x) ** 2).sum() / (self.sigma ** 2)
            neg_log_prior = -self.rosenbrock_dist.logpdf(x.unsqueeze(0)).squeeze()
            energy = neg_log_lik + neg_log_prior

            optimizer.zero_grad()
            energy.backward()
            lr_scheduler.step()
            optimizer.step()

            all_samples.append(x.detach().clone())

        all_samples = torch.stack(all_samples)
        burnin = max_itr // 2
        return all_samples[burnin::thinning]


if "__main__" == __name__:
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
    cnf = RosenbrockCNF(args)

    # Train or test based on phase
    if args.phase == "train":
        cnf.train(args)

    cnf.test(args)

    upload_to_cloud(args)
