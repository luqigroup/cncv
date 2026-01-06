"""Rosenbrock distribution for Bayesian inference problems.

Based on https://arxiv.org/abs/1903.09556

The 2D Rosenbrock distribution is defined as:
    x1 ~ N(μ, 1/(2a))
    x2 | x1 ~ N(x1^2, 1/2)

The joint density is:
    p(x) ∝ sqrt(a) * exp(-a(x1 - μ)^2 - (x2 - x1^2)^2)
"""

import torch
import numpy as np
from typing import Tuple


class RosenbrockDistribution:
    """2-dimensional Rosenbrock distribution.

    Parameters:
        mu: Mean parameter (scalar)
        a: Scaling parameter (scalar)

    Attributes:
        mu: Mean parameter
        a: Scaling parameter
        ndim: Dimension (always 2)
    """

    def __init__(self, mu: float, a: float):
        """Initialize Rosenbrock distribution.

        Args:
            mu: Mean parameter
            a: Scaling parameter (a > 0)
        """
        self.mu = float(mu)
        self.a = float(a)
        self.ndim = 2

    def sample(
        self, n_samples: int, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Sample from the 2D Rosenbrock distribution.

        Args:
            n_samples: Number of samples to generate
            device: Device to place samples on ('cpu' or 'cuda')
            dtype: Data type for samples

        Returns:
            samples: Tensor of shape [n_samples, 2]
        """
        # x1 ~ N(μ, 1/(2a))
        x1 = (
            torch.randn(n_samples, device=device, dtype=dtype) / np.sqrt(2 * self.a)
            + self.mu
        )

        # x2 ~ N(x1^2, 1/2)
        x2 = torch.randn(n_samples, device=device, dtype=dtype) / np.sqrt(2) + x1**2

        return torch.stack([x1, x2], dim=1)  # [n_samples, 2]

    def logpdf(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log probability density.

        Args:
            X: Input tensor of shape [n_samples, 2]

        Returns:
            log_pdf: Log probability density, shape [n_samples]
        """
        assert X.shape[1] == 2, "X must have 2 columns (dimensions)"

        x1, x2 = X[:, 0], X[:, 1]

        # Log pdf (unnormalized): -a(x1 - μ)^2 - (x2 - x1^2)^2
        log_pdf = -self.a * (x1 - self.mu) ** 2 - (x2 - x1**2) ** 2

        # Normalization constant: sqrt(a) / π
        log_norm = np.log(np.sqrt(self.a) / np.pi)

        return log_pdf + log_norm

    def gradlogpdf(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gradient of log probability density.

        Args:
            X: Input tensor of shape [n_samples, 2]

        Returns:
            grad: Gradient tensor of shape [n_samples, 2]
                grad[:, 0] = ∂log p/∂x1
                grad[:, 1] = ∂log p/∂x2
        """
        assert X.shape[1] == 2, "X must have 2 columns (dimensions)"

        x1, x2 = X[:, 0], X[:, 1]

        # ∂log p/∂x1 = -2a(x1 - μ) + 4x1(x2 - x1^2)
        grad_x1 = -2 * self.a * (x1 - self.mu) + 4 * x1 * (x2 - x1**2)

        # ∂log p/∂x2 = -2(x2 - x1^2)
        grad_x2 = -2 * (x2 - x1**2)

        return torch.stack([grad_x1, grad_x2], dim=1)  # [n_samples, 2]


def compute_score_posterior_rosenbrock(
    X: torch.Tensor,
    Y: torch.Tensor,
    rosenbrock_dist: RosenbrockDistribution,
    sigma_obs: float = 0.3,
) -> torch.Tensor:
    """Compute score function for Rosenbrock posterior with Gaussian likelihood.

    The score function is the gradient of the log posterior:
        ∇log p(x|y) = ∇log p(y|x) + ∇log p(x)

    For Gaussian likelihood p(y|x) = N(x, σ²I) and Rosenbrock prior p(x):
        ∇log p(y|x) = -(x - y) / σ²
        ∇log p(x) = gradlogpdf(rosenbrock_dist, x)

    Args:
        X: Latent variables [batch_size, 2]
        Y: Observations [batch_size, 2] or [2] (broadcasted)
        rosenbrock_dist: RosenbrockDistribution instance
        sigma_obs: Observation noise standard deviation

    Returns:
        score: ∇log p(x|y), shape [batch_size, 2]
    """
    # Ensure Y has correct shape
    if Y.ndim == 1:
        Y = Y.unsqueeze(0)  # [1, 2]

    # Gaussian likelihood: y = x + noise, noise ~ N(0, σ²I)
    # ∇log p(y|x) = -(x - y) / σ²
    grad_log_likelihood = -(X - Y) / (sigma_obs**2)

    # Rosenbrock prior: ∇log p(x)
    grad_log_prior = rosenbrock_dist.gradlogpdf(X)

    return grad_log_likelihood + grad_log_prior
