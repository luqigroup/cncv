"""DDPM training for nonlinear forward model posterior.

Trains a conditional DDPM to learn the posterior distribution p(x|y)
for the nonlinear forward model F(x) = Ax + sin(x) with Gaussian prior.

This provides amortized posterior samples at evaluation time, used in place
of pSGLD MCMC for the nonlinear model.

Usage:
    python scripts/ddpm_nonlinear.py --phase train --gpu 0
    python scripts/ddpm_nonlinear.py --phase test --testing_epoch 999 --gpu 0
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

from cncv.models.ddpm_mlp import ConditionalScoreMLPSimple
from cncv.diffusion import NoiseScheduler
from cncv.utils import CustomLRScheduler
from cncv.utils.nonlinear import NonlinearForwardModel, setup_nonlinear_problem

CONFIG_FILE = "ddpm_nonlinear.json"


class NonlinearDDPM:
    """DDPM for learning posterior p(x|y) with nonlinear forward model.

    The forward model is F(x) = Ax + sin(x), with Gaussian prior x ~ N(mu, Sigma)
    and Gaussian likelihood y|x ~ N(F(x), sigma^2 I).

    Attributes:
        device: Computation device
        n_dim: Problem dimension
        forward_model: NonlinearForwardModel instance
        mu_prior, Sigma_prior, L_prior: Prior parameters
        sigma: Observation noise std
        model: DDPM model
        noise_scheduler: Diffusion noise scheduler
    """

    def __init__(self, args):
        if torch.cuda.is_available() and args.gpu_id > -1:
            self.device = torch.device("cuda:" + str(args.gpu_id))
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.n_dim = int(args.input_dim)
        self.sigma = float(args.sigma)
        cond_number = float(getattr(args, "cond_number", 2.0))

        # Setup problem (prior, forward model)
        problem = setup_nonlinear_problem(
            self.n_dim, sigma=self.sigma, cond_number=cond_number, device=self.device
        )
        self.mu_prior = problem["mu_prior"]
        self.Sigma_prior = problem["Sigma_prior"]
        self.L_prior = problem["L_prior"]
        self.Sigma_prior_inv = problem["Sigma_prior_inv"]
        self.forward_model = problem["forward_model"]

        print("\n=== Nonlinear Forward Model Setup ===")
        print(f"Dimension: {self.n_dim}")
        print(f"F(x) = Ax + sin(x), cond(A) = {cond_number}")
        print(f"Prior: N(0, Sigma_prior)")
        print(f"Observation noise sigma: {self.sigma}")

        # Create DDPM model
        # y_dim = n_dim because y = F(x) + noise has same dimension
        self.model = ConditionalScoreMLPSimple(
            x_dim=self.n_dim,
            y_dim=self.n_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            time_emb_dim=64,
            nt=args.nt,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"DDPM parameters: {num_params:,}")

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            nt=args.nt,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            device=self.device,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = CustomLRScheduler(
            self.optimizer,
            initial_lr=args.lr,
            final_lr=args.lr_final,
            max_step=args.max_epochs,
        )

        # Generate training data: x ~ prior, y = F(x) + noise
        print(f"\n=== Generating Training Data ===")
        print(f"Training samples: {args.num_train}")

        torch.manual_seed(0)
        noise_x = torch.randn(args.num_train, self.n_dim, device=self.device)
        self.X_train = self.mu_prior + (self.L_prior @ noise_x.T).T
        self.Y_train = (
            self.forward_model.forward(self.X_train)
            + self.sigma * torch.randn(args.num_train, self.n_dim, device=self.device)
        )

        noise_xv = torch.randn(args.num_val, self.n_dim, device=self.device)
        self.X_val = self.mu_prior + (self.L_prior @ noise_xv.T).T
        self.Y_val = (
            self.forward_model.forward(self.X_val)
            + self.sigma * torch.randn(args.num_val, self.n_dim, device=self.device)
        )

        self.train_obj = []
        self.val_obj = []

    def load_checkpoint(self, args):
        file_to_load = os.path.join(
            checkpointsdir(args.experiment),
            "checkpoint_" + str(args.testing_epoch) + ".pth",
        )
        if not os.path.isfile(file_to_load):
            raise FileNotFoundError(f"Checkpoint not found: {file_to_load}")

        checkpoint = torch.load(
            file_to_load,
            map_location=self.device if self.device != torch.device("cpu") else "cpu",
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_obj = checkpoint["train_obj"]
        self.val_obj = checkpoint["val_obj"]

        # Restore forward model A matrix
        self.forward_model.A = checkpoint["forward_model_A"].to(self.device)
        self.mu_prior = checkpoint["mu_prior"].to(self.device)
        self.Sigma_prior = checkpoint["Sigma_prior"].to(self.device)
        self.L_prior = torch.linalg.cholesky(self.Sigma_prior)
        self.Sigma_prior_inv = torch.linalg.inv(self.Sigma_prior)
        self.sigma = checkpoint["sigma"]

        if not args.testing_epoch == checkpoint["epoch"]:
            raise ValueError("Inconsistent filename and loaded checkpoint epoch.")

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def compute_loss(self, X, Y):
        batch_size = X.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.nt, (batch_size,), device=self.device).long()
        noise = torch.randn_like(X)
        X_noisy = self.noise_scheduler.add_noise(X, noise, timesteps)
        noise_pred = self.model(X_noisy, Y, timesteps)
        loss = ((noise_pred - noise) ** 2).mean()
        return loss

    def train(self, args):
        print("\n=== Starting Training ===")
        print(f"Max epochs: {args.max_epochs}, Batch size: {args.batchsize}")

        train_dataset = torch.utils.data.TensorDataset(self.X_train, self.Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True
        )
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.Y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batchsize, shuffle=False
        )

        with tqdm(range(args.max_epochs), unit="epoch", colour="#B5F2A9") as epoch_pbar:
            for epoch in epoch_pbar:
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val, Y_val in val_loader:
                        loss = self.compute_loss(X_val, Y_val)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self.val_obj.append(val_loss)

                # Training
                self.model.train()
                self.lr_scheduler.step()

                epoch_train_losses = []
                for X, Y in train_loader:
                    loss = self.compute_loss(X, Y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.train_obj.append(loss.item())
                    epoch_train_losses.append(loss.item())

                avg_train_loss = np.mean(epoch_train_losses)
                epoch_pbar.set_postfix(
                    {"train": f"{avg_train_loss:.6f}", "val": f"{val_loss:.6f}"}
                )

                # Save checkpoint
                if epoch % args.save_freq == 0 or epoch == args.max_epochs - 1:
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                            "train_obj": self.train_obj,
                            "val_obj": self.val_obj,
                            "mu_prior": self.mu_prior.cpu(),
                            "Sigma_prior": self.Sigma_prior.cpu(),
                            "sigma": self.sigma,
                            "input_dim": self.n_dim,
                            "forward_model_A": self.forward_model.A.cpu(),
                            "nt": args.nt,
                            "beta_schedule": args.beta_schedule,
                        },
                        os.path.join(
                            checkpointsdir(args.experiment),
                            "checkpoint_" + str(epoch) + ".pth",
                        ),
                    )

        print(f"\nTraining complete. Final train: {self.train_obj[-1]:.6f}, val: {self.val_obj[-1]:.6f}")

    def sample_from_posterior(self, Y_obs, num_samples):
        """Sample from posterior p(x|y) using reverse diffusion."""
        self.model.eval()
        x = torch.randn(num_samples, self.n_dim, device=self.device)
        Y_batch = Y_obs.unsqueeze(0).repeat(num_samples, 1)

        with torch.no_grad():
            for t in tqdm(
                reversed(range(self.noise_scheduler.nt)),
                desc="Sampling",
                total=self.noise_scheduler.nt,
                colour="#FFB6C1",
            ):
                t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
                noise_pred = self.model(x, Y_batch, t_batch)
                x = self.noise_scheduler.step(noise_pred, t, x)

        return x

    def test(self, args):
        self.load_checkpoint(args)
        self.model.eval()

        print("\n=== Generating Test Data ===")
        torch.manual_seed(99)
        X_true = self.mu_prior + self.L_prior @ torch.randn(self.n_dim, device=self.device)
        Y_obs = (
            self.forward_model.forward(X_true.unsqueeze(0)).squeeze(0)
            + self.sigma * torch.randn(self.n_dim, device=self.device)
        )

        print(f"True X: {X_true}")
        print(f"Observation Y: {Y_obs}")

        # Sample from DDPM posterior
        num_samples = args.num_samples
        print(f"\nSampling {num_samples} from DDPM posterior...")
        X_samples = self.sample_from_posterior(Y_obs, num_samples)

        print(f"DDPM samples mean: {X_samples.mean(dim=0)}")
        print(f"DDPM samples std:  {X_samples.std(dim=0)}")

        # Quality check: forward model residuals
        F_samples = self.forward_model.forward(X_samples)
        residuals = (F_samples - Y_obs.unsqueeze(0)).norm(dim=1)
        print(f"Forward residual ||F(x)-y||: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

        # Plot results
        self.plot_results(args, X_samples, X_true, Y_obs)

    def plot_results(self, args, X_samples, X_true, Y_obs):
        plots_dir = plotsdir(args.experiment)

        # Training curves
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.train_obj, label="Training", color="#1f77b4", alpha=0.7)
        ax.plot(
            np.arange(len(self.val_obj)) * (len(self.train_obj) // max(len(self.val_obj), 1)),
            self.val_obj,
            label="Validation",
            color="#ff7f0e",
            linewidth=2,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("DDPM Nonlinear Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=300)
        plt.close()

        # Posterior samples histograms
        X_np = X_samples.cpu().numpy()
        X_true_np = X_true.cpu().numpy()

        fig, axes = plt.subplots(1, min(self.n_dim, 4), figsize=(4 * min(self.n_dim, 4), 4))
        if self.n_dim == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i >= self.n_dim:
                break
            ax.hist(X_np[:, i], bins=50, density=True, alpha=0.7, label="DDPM samples")
            ax.axvline(X_true_np[i], color="r", linestyle="--", linewidth=2, label=f"True x={X_true_np[i]:.2f}")
            ax.set_xlabel(f"$x_{{{i+1}}}$")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "posterior_samples.png"), dpi=300)
        plt.close()

        print(f"Plots saved to {plots_dir}")


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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.testing_epoch == -1:
        args.testing_epoch = args.max_epochs - 1

    ddpm = NonlinearDDPM(args)

    if args.phase == "train":
        ddpm.train(args)

    ddpm.test(args)

    upload_to_cloud(args)
