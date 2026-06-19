"""DDPM noise scheduler for forward and reverse diffusion processes.

Adapted from https://github.com/tanelp/tiny-diffusion.
Implements the noise scheduling for Denoising Diffusion Probabilistic Models (DDPM).
"""

import torch
from torch.nn import functional as F


class NoiseScheduler:
    """DDPM noise scheduler for forward and reverse diffusion.

    Implements the forward diffusion process q(x_t|x_0) and reverse sampling
    process p_θ(x_{t-1}|x_t) for Denoising Diffusion Probabilistic Models.

    The forward process gradually adds Gaussian noise according to a schedule:
        q(x_t|x_0) = N(√(ᾱ_t)x_0, (1-ᾱ_t)I)

    where ᾱ_t = ∏_{s=1}^t α_s and α_t = 1 - β_t.

    Attributes:
        nt: Number of diffusion timesteps
        betas: Noise schedule β_t ∈ (0,1) for each timestep
        alphas: 1 - β_t
        alphas_cumprod: Cumulative product ᾱ_t = ∏_{s=1}^t α_s
        sqrt_alphas_cumprod: √ᾱ_t (for forward diffusion)
        sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t) (for forward diffusion)
    """

    def __init__(
        self,
        nt: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cpu",
    ) -> None:
        """Initialize noise scheduler.

        Args:
            nt: Number of diffusion timesteps (default: 1000)
            beta_start: Starting value of β schedule (default: 0.0001)
            beta_end: Ending value of β schedule (default: 0.02)
            beta_schedule: Type of schedule ('linear' or 'quadratic')
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.nt = nt

        # Create beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, nt, dtype=torch.float32, device=device
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    nt,
                    dtype=torch.float32,
                    device=device,
                )
                ** 2
            )
        else:
            raise ValueError(
                f"Unknown beta_schedule: {beta_schedule}. "
                f"Choose 'linear' or 'quadratic'."
            )

        # Compute α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # Compute cumulative product ᾱ_t = ∏_{s=1}^t α_s
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # ᾱ_{t-1} (padded with 1.0 at the beginning for t=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for forward diffusion (add_noise)
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # Ensure these are on the correct device
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )

        # Precompute values for x_0 reconstruction from x_t
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1
        )

        # Precompute coefficients for posterior q(x_{t-1}|x_t, x_0)
        # μ̃_t(x_t, x_0) = coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def add_noise(
        self, x_start: torch.Tensor, x_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean samples according to forward diffusion process.

        Implements q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I) as:
            x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            x_start: Clean samples x_0, shape [batch_size, ...]
            x_noise: Noise samples ε ~ N(0,I), same shape as x_start
            timesteps: Timestep indices, shape [batch_size]

        Returns:
            Noisy samples x_t, same shape as x_start
        """
        # Get coefficients for this timestep
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape to match input dimensions: [batch_size, 1, 1, ...]
        s1 = s1.reshape(-1, *[1 for _ in range(x_start.dim() - 1)])
        s2 = s2.reshape(-1, *[1 for _ in range(x_start.dim() - 1)])

        return s1 * x_start + s2 * x_noise

    def reconstruct_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x_0 from x_t and predicted noise.

        Uses the relationship:
            x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
               = 1/√ᾱ_t * x_t - √(1/ᾱ_t - 1) * ε

        Args:
            x_t: Noisy sample at timestep t
            t: Timestep index (scalar or [batch_size])
            noise: Predicted noise ε_θ(x_t, t)

        Returns:
            Reconstructed x_0
        """
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]

        # Reshape coefficients for broadcasting with arbitrary dims [B, ...]
        s1 = s1.reshape(-1, *[1 for _ in range(x_t.dim() - 1)])
        s2 = s2.reshape(-1, *[1 for _ in range(x_t.dim() - 1)])

        return s1 * x_t - s2 * noise

    def q_posterior(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean of posterior q(x_{t-1}|x_t, x_0).

        The posterior mean is:
            μ̃_t = β_t√ᾱ_{t-1}/(1-ᾱ_t) * x_0 + (1-ᾱ_{t-1})√α_t/(1-ᾱ_t) * x_t

        Args:
            x_0: Clean sample (reconstructed)
            x_t: Noisy sample at timestep t
            t: Timestep index

        Returns:
            Posterior mean μ̃_t
        """
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]

        # Reshape for broadcasting with arbitrary dims [B, ...]
        s1 = s1.reshape(-1, *[1 for _ in range(x_0.dim() - 1)])
        s2 = s2.reshape(-1, *[1 for _ in range(x_0.dim() - 1)])

        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t: int) -> float:
        """Get posterior variance for timestep t.

        The posterior variance is:
            σ̃_t² = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)

        For t=0, returns 0 (no noise added).

        Args:
            t: Timestep index (integer)

        Returns:
            Variance σ̃_t² (scalar)
        """
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)  # Avoid numerical issues
        return variance

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor
    ) -> torch.Tensor:
        """Perform one reverse diffusion step: p_θ(x_{t-1}|x_t).

        Given x_t and predicted noise ε_θ(x_t, t):
        1. Reconstruct x_0 from x_t and ε_θ
        2. Compute posterior mean μ̃_t(x_t, x_0)
        3. Add variance: x_{t-1} = μ̃_t + σ̃_t * z, z ~ N(0,I)

        Args:
            model_output: Predicted noise ε_θ(x_t, t)
            timestep: Current timestep t (integer or tensor)
            sample: Current noisy sample x_t

        Returns:
            Previous sample x_{t-1}
        """
        # Convert timestep to integer if it's a tensor
        if isinstance(timestep, torch.Tensor):
            t = timestep.item() if timestep.numel() == 1 else timestep[0].item()
        else:
            t = timestep

        # Reconstruct x_0
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)

        # Compute posterior mean
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        # Add variance (unless t=0)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def __len__(self) -> int:
        """Return number of timesteps."""
        return self.nt
