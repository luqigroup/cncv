"""Minimal MLP-based conditional DDPM for learning posterior scores.

Implements a simple fully-connected network with sinusoidal time embeddings
for learning the score function of conditional distributions p(x|y).
"""

import math
import torch
import torch.nn as nn
from typing import Optional


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Standard positional encoding used in transformers and diffusion models.

    Args:
        timesteps: Timestep indices, shape [batch_size]
        dim: Embedding dimension (should be even)
        max_period: Maximum period for sinusoidal encoding

    Returns:
        Embeddings of shape [batch_size, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # If dim is odd, pad with zeros
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class ConditionalScoreMLPSimple(nn.Module):
    """Minimal MLP for learning conditional score function in DDPM.

    A simple fully-connected network that takes noisy samples x_t, conditioning
    observations y, and timestep t as input, and predicts the noise ε that was
    added to create x_t.

    The learned noise prediction can be used to extract the score function:
        ∇log p(x_t|y) = -ε_θ(x_t, y, t) / √(1-ᾱ_t)

    Architecture:
        1. Sinusoidal time embedding for t
        2. Concatenate [x_t, y, time_emb]
        3. MLP with configurable layers and hidden dimensions
        4. Output: predicted noise ε

    Args:
        x_dim: Dimension of samples x
        y_dim: Dimension of conditioning y (usually same as x_dim)
        hidden_dim: Hidden layer width (default: 128)
        n_layers: Number of hidden layers (default: 3)
        time_emb_dim: Dimension of time embeddings (default: 64)
        activation: Activation function (default: SiLU)
        nt: Number of diffusion timesteps (for normalization, default: 1000)

    Attributes:
        x_dim: Input dimension
        y_dim: Conditioning dimension
        hidden_dim: Hidden layer size
        n_layers: Number of layers
        time_emb_dim: Time embedding dimension
        nt: Number of timesteps
        mlp: The main MLP network
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 3,
        time_emb_dim: int = 64,
        activation: Optional[nn.Module] = None,
        nt: int = 1000,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.time_emb_dim = time_emb_dim
        self.nt = nt

        # Default to SiLU activation (common in diffusion models)
        if activation is None:
            activation = nn.SiLU()

        # Input dimension: x + y + time_embedding
        input_dim = x_dim + y_dim + time_emb_dim

        # Build MLP
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        # Output layer (predicts noise, same dimension as x)
        layers.append(nn.Linear(hidden_dim, x_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: predict noise ε from (x_t, y, t).

        Args:
            x: Noisy samples x_t, shape [batch_size, x_dim]
            y: Conditioning observations, shape [batch_size, y_dim]
            t: Timesteps, shape [batch_size] (integer timestep indices)

        Returns:
            Predicted noise ε, shape [batch_size, x_dim]
        """
        # Ensure timesteps are long integers
        if t.dtype != torch.long:
            t = t.long()

        # Create time embeddings
        t_emb = timestep_embedding(t, self.time_emb_dim)

        # Concatenate inputs: [x, y, time_emb]
        z = torch.cat([x, y, t_emb], dim=-1)

        # Forward through MLP
        noise_pred = self.mlp(z)

        return noise_pred

    def extra_repr(self) -> str:
        """Extra representation string for print/summary."""
        return (
            f"x_dim={self.x_dim}, y_dim={self.y_dim}, "
            f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, "
            f"time_emb_dim={self.time_emb_dim}, nt={self.nt}"
        )
