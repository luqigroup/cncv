"""Utility functions for DDPM score extraction and sampling.

This module provides utilities for extracting score functions from trained DDPM
models and using them in the neural control variate framework.
"""

import torch
from typing import Tuple


def extract_score_from_ddpm(
    model: torch.nn.Module,
    noise_scheduler,
    x_t: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    timestep: int = 50,
) -> torch.Tensor:
    """Extract score function ∇log p_t(x_t|y) from trained DDPM.

    Uses the relationship between noise prediction and score:
        ∇log p_t(x_t|y) = -ε_θ(x_t, y, t) / √(1-ᾱ_t)

    For control variates, use timestep=50 where the model is well-calibrated.
    The input x_t should already be at the diffused level (noisy samples).

    Args:
        model: Trained DDPM model (ConditionalScoreMLPSimple)
        noise_scheduler: NoiseScheduler instance
        x_t: Diffused samples at timestep t [batch_size, x_dim]
        y: Conditioning observations [batch_size, y_dim]
        device: Computation device
        timestep: Timestep to evaluate at (default: 50)

    Returns:
        score: ∇log p_t(x_t|y), shape [batch_size, x_dim]
    """
    model.eval()

    # Ensure timestep >= 1 for proper noise scaling
    timestep = max(1, timestep)

    batch_size = x_t.shape[0]

    # Create timestep tensor
    t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        # Predict noise (x_t is already at the diffused level)
        noise_pred = model(x_t, y, t)

        # Convert noise prediction to score
        # score = -ε_θ(x_t, y, t) / √(1-ᾱ_t)
        sqrt_one_minus_alpha_cumprod = noise_scheduler.sqrt_one_minus_alphas_cumprod[
            timestep
        ]
        score = -noise_pred / sqrt_one_minus_alpha_cumprod

    return score


def diffuse_samples(
    x: torch.Tensor,
    noise_scheduler,
    timestep: int,
    device: torch.device,
) -> tuple:
    """Add noise to samples to get diffused samples at timestep t.

    Args:
        x: Clean samples [batch_size, x_dim]
        noise_scheduler: NoiseScheduler instance
        timestep: Diffusion timestep
        device: Computation device

    Returns:
        x_t: Diffused samples [batch_size, x_dim]
        noise: The noise that was added [batch_size, x_dim]
    """
    batch_size = x.shape[0]
    t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
    noise = torch.randn_like(x)
    x_t = noise_scheduler.add_noise(x, noise, t)
    return x_t, noise


def load_ddpm_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """Load DDPM checkpoint and return model and metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        model: DDPM model instance (will be populated with loaded weights)
        device: Device to load checkpoint onto

    Returns:
        model: Loaded model in eval mode
        checkpoint: Checkpoint dictionary with metadata

    Raises:
        FileNotFoundError: If checkpoint does not exist
    """
    import os

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"DDPM checkpoint not found: {checkpoint_path}")

    if device == torch.device("cpu"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def sample_from_ddpm(
    model: torch.nn.Module,
    noise_scheduler,
    y_obs: torch.Tensor,
    num_samples: int,
    device: torch.device,
    show_progress: bool = False,
) -> torch.Tensor:
    """Sample from posterior p(x|y) using trained DDPM.

    Performs reverse diffusion starting from pure noise.

    Args:
        model: Trained DDPM model
        noise_scheduler: NoiseScheduler instance
        y_obs: Observation [y_dim] or [batch_size, y_dim]
        num_samples: Number of samples to generate
        device: Computation device
        show_progress: Whether to show progress bar (requires tqdm)

    Returns:
        samples: Posterior samples [num_samples, x_dim]
    """
    model.eval()

    # Handle y_obs shape
    if y_obs.ndim == 1:
        y_obs = y_obs.unsqueeze(0)

    x_dim = model.x_dim

    # Start from pure noise
    x = torch.randn(num_samples, x_dim, device=device)

    # Repeat observation for all samples
    Y_batch = y_obs.repeat(num_samples, 1)

    # Reverse diffusion
    timesteps = range(noise_scheduler.nt)
    if show_progress:
        from tqdm import tqdm

        timesteps = tqdm(
            reversed(timesteps),
            desc="Sampling",
            total=noise_scheduler.nt,
        )
    else:
        timesteps = reversed(timesteps)

    with torch.no_grad():
        for t in timesteps:
            t_batch = torch.full(
                (num_samples,), t, device=device, dtype=torch.long
            )
            noise_pred = model(x, Y_batch, t_batch)
            x = noise_scheduler.step(noise_pred, t, x)

    return x
