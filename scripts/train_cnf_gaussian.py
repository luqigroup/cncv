"""Train a Conditional Normalizing Flow on the Gaussian inverse problem.

Learns p(x|y) via maximum likelihood using HINT-style hierarchical coupling
layers. After training, evaluates:
  - Sample quality vs analytical posterior (mean, covariance)
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
from cncv.utils.gaussian import compute_score_posterior_gaussian, compute_exact_posterior
from cncv.utils import CustomLRScheduler

CONFIG_FILE = "cnf_gaussian.json"


class GaussianCNF:
    """Conditional normalizing flow for Gaussian posterior learning."""

    def __init__(self, args: argparse.Namespace) -> None:
        # Device setup
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Problem setup
        self.n_dim = int(args.input_dim)
        self.mu_prior = torch.zeros(
            self.n_dim, dtype=torch.float32, device=self.device
        )

        # Generate random positive definite covariance matrix (same as gaussian_ensemble_cv)
        diag_values = torch.linspace(1.0, 2.0, self.n_dim)
        D = torch.diag(diag_values).to(device=self.device, dtype=torch.float32)

        if self.n_dim > 1:
            torch.manual_seed(42)  # Fixed seed for reproducible Sigma
            A = torch.randn(
                self.n_dim, self.n_dim, device=self.device, dtype=torch.float32
            )
            Q, _ = torch.linalg.qr(A)
            self.Sigma_prior = Q @ D @ Q.T
        else:
            self.Sigma_prior = D

        self.L_prior = torch.linalg.cholesky(self.Sigma_prior)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
        self.sigma = float(args.sigma)

        print("\n=== Problem Setup ===")
        print(f"Dimension: {self.n_dim}")
        print(f"Prior covariance:\n{self.Sigma_prior}")
        print(f"Observation noise sigma: {self.sigma}")

        # Restore seed for data generation
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

        noise = torch.randn(
            args.num_train, self.n_dim, dtype=torch.float32, device=self.device
        )
        self.X_train = self.mu_prior + (self.L_prior @ noise.T).T
        self.Y_train = self.X_train + self.sigma * torch.randn_like(self.X_train)

        noise_val = torch.randn(
            args.num_val, self.n_dim, dtype=torch.float32, device=self.device
        )
        self.X_val = self.mu_prior + (self.L_prior @ noise_val.T).T
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
                            "mu_prior": self.mu_prior.cpu(),
                            "Sigma_prior": self.Sigma_prior.cpu(),
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
        """Evaluate CNF against analytical Gaussian posterior."""
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
        self.mu_prior = checkpoint["mu_prior"].to(self.device)
        self.Sigma_prior = checkpoint["Sigma_prior"].to(self.device)
        self.L_prior = torch.linalg.cholesky(self.Sigma_prior)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
        self.sigma = checkpoint["sigma"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        self.cnf.eval()

        # Generate test observation
        torch.manual_seed(args.seed + 1000)
        X_true = self.mu_prior + self.L_prior @ torch.randn(
            self.n_dim, device=self.device
        )
        Y_obs = X_true + self.sigma * torch.randn(
            self.n_dim, device=self.device
        )

        print(f"\nTrue parameter X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        # Analytical posterior
        mu_post, Sigma_post = compute_exact_posterior(
            Y_obs, self.mu_prior, self.Sigma_prior, self.sigma
        )
        print(f"\n=== Analytical Posterior ===")
        print(f"Posterior mean: {mu_post}")
        print(f"Posterior cov diagonal: {torch.diag(Sigma_post)}")

        num_samples = args.num_samples

        # 1. Sample quality
        print(f"\n=== CNF Sample Quality ({num_samples} samples) ===")
        with torch.no_grad():
            x_samples = self.cnf.sample(Y_obs, num_samples)

        sample_mean = x_samples.mean(dim=0)
        sample_cov = torch.cov(x_samples.T)

        mean_error = (sample_mean - mu_post).abs().max().item()
        cov_error = (sample_cov - Sigma_post).abs().max().item()

        print(f"Sample mean:     {sample_mean}")
        print(f"Analytical mean: {mu_post}")
        print(f"Max mean error:  {mean_error:.6f}")
        print(f"Sample cov diag:     {torch.diag(sample_cov)}")
        print(f"Analytical cov diag: {torch.diag(Sigma_post)}")
        print(f"Max cov error:   {cov_error:.6f}")

        # 2. Log-likelihood on analytical posterior samples
        print(f"\n=== CNF Log-Likelihood on Posterior Samples ===")
        L_post = torch.linalg.cholesky(Sigma_post)
        noise = torch.randn(num_samples, self.n_dim, device=self.device)
        x_post_samples = mu_post + (L_post @ noise.T).T
        Y_repeated = Y_obs.repeat(num_samples, 1)

        with torch.no_grad():
            log_p_cnf = self.cnf.log_prob(x_post_samples, Y_repeated)

        # Analytical log-likelihood for comparison
        log_p_exact = self._gaussian_log_prob(x_post_samples, mu_post, Sigma_post)

        print(f"CNF mean log-prob:       {log_p_cnf.mean().item():.4f}")
        print(f"Analytical mean log-prob: {log_p_exact.mean().item():.4f}")
        print(f"Difference:              {(log_p_cnf.mean() - log_p_exact.mean()).item():.4f}")

        # 3. Score accuracy
        print(f"\n=== Score Accuracy ===")
        x_test = x_post_samples[:100]
        Y_test = Y_obs.repeat(100, 1)

        score_cnf = self.cnf.score(x_test, Y_test)
        score_exact = compute_score_posterior_gaussian(
            x_test, Y_test, self.mu_prior, self.Sigma_prior_inv, self.sigma
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
        ax.set_title("CNF Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=300)
        plt.close()

        # Posterior comparison (first 2 dims)
        if self.n_dim >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # CNF samples
            x_np = x_samples.cpu().numpy()
            axes[0].scatter(x_np[:, 0], x_np[:, 1], alpha=0.2, s=3, c="#1f77b4")
            axes[0].set_title("CNF Samples")
            axes[0].set_xlabel("$x_1$")
            axes[0].set_ylabel("$x_2$")
            axes[0].set_aspect("equal")
            axes[0].grid(True, alpha=0.3)

            # Analytical posterior samples
            x_exact_np = x_post_samples.cpu().numpy()
            axes[1].scatter(x_exact_np[:, 0], x_exact_np[:, 1], alpha=0.2, s=3, c="#ff7f0e")
            axes[1].set_title("Analytical Posterior Samples")
            axes[1].set_xlabel("$x_1$")
            axes[1].set_ylabel("$x_2$")
            axes[1].set_aspect("equal")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "posterior_comparison.png"), dpi=300)
            plt.close()

        # Marginal histograms
        fig, axes = plt.subplots(1, min(self.n_dim, 4), figsize=(4 * min(self.n_dim, 4), 4))
        if self.n_dim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            if i >= self.n_dim:
                break
            ax.hist(x_samples[:, i].cpu().numpy(), bins=50, density=True,
                    alpha=0.7, label="CNF", color="#1f77b4")
            # Analytical marginal
            mu_i = mu_post[i].item()
            std_i = Sigma_post[i, i].sqrt().item()
            x_range = np.linspace(mu_i - 4 * std_i, mu_i + 4 * std_i, 100)
            pdf = np.exp(-0.5 * ((x_range - mu_i) / std_i) ** 2) / (std_i * np.sqrt(2 * np.pi))
            ax.plot(x_range, pdf, "r-", linewidth=2, label="Analytical")
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
        print(f"Max mean error:      {mean_error:.6f}")
        print(f"Max cov error:       {cov_error:.6f}")
        print(f"Score MSE:           {score_mse:.6f}")
        print(f"Score relative err:  {score_rel_error:.6f}")
        print(f"CNF mean log-prob:   {log_p_cnf.mean().item():.4f}")
        print(f"Exact mean log-prob: {log_p_exact.mean().item():.4f}")
        print(f"{'='*50}")

    def _gaussian_log_prob(self, x, mu, Sigma):
        """Compute log N(x; mu, Sigma) for batch of x."""
        d = mu.shape[0]
        L = torch.linalg.cholesky(Sigma)
        diff = x - mu
        # Solve L @ z = diff^T
        z = torch.linalg.solve_triangular(L, diff.T, upper=False)
        log_det = 2.0 * torch.log(torch.diag(L)).sum()
        return -0.5 * (d * np.log(2 * np.pi) + log_det + z.pow(2).sum(dim=0))


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
    cnf = GaussianCNF(args)

    # Train or test based on phase
    if args.phase == "train":
        cnf.train(args)

    cnf.test(args)

    upload_to_cloud(args)
