"""Rosenbrock distribution for Bayesian inference problems.

Based on https://arxiv.org/abs/1903.09556

The Hybrid Rosenbrock distribution is defined as:
    π(x) ∝ exp{-a(x₁ - μ)² - Σⱼ₌₁ⁿ² Σᵢ₌₂ⁿ¹ bⱼ,ᵢ(xⱼ,ᵢ - x²ⱼ,ᵢ₋₁)²}

For the simple 2D case with n1=2, n2=1:
    x1 ~ N(μ, 1/(2a))
    x2 | x1 ~ N(x1^2, 1/(2b))

The joint density is:
    p(x) = sqrt(a) * sqrt(b) / π * exp(-a(x1 - μ)^2 - b(x2 - x1^2)^2)

The dimension formula is: n = (n1 - 1) * n2 + 1
"""

import torch
import numpy as np
from typing import Tuple, Union


class RosenbrockDistribution:
    """Hybrid Rosenbrock distribution.

    Parameters:
        mu: Mean parameter (scalar)
        a: Scaling parameter for x1 (scalar, a > 0)
        b: Scaling parameter(s) for Rosenbrock terms
            - If scalar: same b for all terms
            - If array: b[j, i] for each block j and position i
        n1: Block size (number of variables per block)
        n2: Number of blocks

    Attributes:
        mu: Mean parameter
        a: Scaling parameter for x1
        b: Scaling parameters for Rosenbrock terms (shape [n2, n1-1])
        n1: Block size
        n2: Number of blocks
        ndim: Total dimension n = (n1 - 1) * n2 + 1
    """

    def __init__(
        self,
        mu: float,
        a: float,
        b: Union[float, np.ndarray] = 1.0,
        n1: int = 2,
        n2: int = 1,
    ):
        """Initialize Rosenbrock distribution.

        Args:
            mu: Mean parameter
            a: Scaling parameter for x1 (a > 0)
            b: Scaling parameter(s) for Rosenbrock terms
                - If scalar: same b for all terms
                - If array: shape [n2, n1-1] with b[j, i-2] for term (j, i)
            n1: Block size (default 2 for 2D case)
            n2: Number of blocks (default 1 for 2D case)
        """
        self.mu = float(mu)
        self.a = float(a)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.ndim = (n1 - 1) * n2 + 1

        # Set up b parameters
        if isinstance(b, (int, float)):
            # Same b for all terms
            self.b = np.full((n2, n1 - 1), float(b))
        else:
            b = np.asarray(b)
            if b.shape != (n2, n1 - 1):
                raise ValueError(
                    f"b must have shape ({n2}, {n1 - 1}), got {b.shape}"
                )
            self.b = b

        # Compute normalization constant: sqrt(a) * prod(sqrt(b[j,i])) / pi^(n/2)
        log_norm = 0.5 * np.log(self.a)  # sqrt(a)
        log_norm += 0.5 * np.sum(np.log(self.b))  # prod(sqrt(b[j,i]))
        log_norm -= (self.ndim / 2.0) * np.log(np.pi)  # / pi^(n/2)
        self.log_norm_const = log_norm

    def sample(
        self, n_samples: int, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Sample from the Hybrid Rosenbrock distribution.

        Uses the conditional normal structure:
        x1 ~ N(μ, 1/(2a))
        xj,i | xj,i-1 ~ N(xj,i-1^2, 1/(2*b[j,i]))

        Args:
            n_samples: Number of samples to generate
            device: Device to place samples on ('cpu' or 'cuda')
            dtype: Data type for samples

        Returns:
            samples: Tensor of shape [n_samples, ndim]
        """
        # x1 ~ N(μ, 1/(2a))
        x1 = (
            torch.randn(n_samples, device=device, dtype=dtype) / np.sqrt(2 * self.a)
            + self.mu
        )

        samples = [x1]

        # Sample each block
        for j in range(self.n2):
            # Start of block: condition on x1
            prev = x1

            # Sample within block
            for i in range(self.n1 - 1):
                b_ji = self.b[j, i]
                # x[j,i+1] | x[j,i] ~ N(x[j,i]^2, 1/(2*b[j,i]))
                curr = (
                    torch.randn(n_samples, device=device, dtype=dtype)
                    / np.sqrt(2 * b_ji)
                    + prev**2
                )
                samples.append(curr)
                prev = curr

        return torch.stack(samples, dim=1)  # [n_samples, ndim]

    def logpdf(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log probability density.

        Args:
            X: Input tensor of shape [n_samples, ndim]

        Returns:
            log_pdf: Log probability density, shape [n_samples]
        """
        assert X.shape[1] == self.ndim, f"X must have {self.ndim} columns (dimensions)"

        x1 = X[:, 0]

        # First term: -a(x1 - μ)^2
        log_pdf = -self.a * (x1 - self.mu) ** 2

        # Rosenbrock terms for each block
        idx = 1  # Current index in X
        for j in range(self.n2):
            # Start of block: condition on x1
            prev = x1

            # Terms within block
            for i in range(self.n1 - 1):
                curr = X[:, idx]
                b_ji = self.b[j, i]
                # Add term: -b[j,i] * (x[j,i+1] - x[j,i]^2)^2
                log_pdf = log_pdf - b_ji * (curr - prev**2) ** 2
                prev = curr
                idx += 1

        return log_pdf + self.log_norm_const

    def gradlogpdf(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gradient of log probability density.

        Args:
            X: Input tensor of shape [n_samples, ndim]

        Returns:
            grad: Gradient tensor of shape [n_samples, ndim]
        """
        assert X.shape[1] == self.ndim, f"X must have {self.ndim} columns (dimensions)"

        n_samples = X.shape[0]
        grad = torch.zeros_like(X)

        x1 = X[:, 0]

        # Gradient for x1 has contribution from:
        # 1. First term: -2a(x1 - μ)
        grad[:, 0] = -2 * self.a * (x1 - self.mu)

        # 2. All blocks that condition on x1
        for j in range(self.n2):
            x_ji_plus_1 = X[:, 1 + j * (self.n1 - 1)]  # First variable in block j
            b_j_0 = self.b[j, 0]
            # Add contribution: +4 * b[j,0] * x1 * (x[j,1] - x1^2)
            grad[:, 0] = grad[:, 0] + 4 * b_j_0 * x1 * (x_ji_plus_1 - x1**2)

        # Gradient for variables within blocks
        for j in range(self.n2):
            for i in range(self.n1 - 1):
                idx = 1 + j * (self.n1 - 1) + i
                b_ji = self.b[j, i]

                if i == 0:
                    # First variable in block: x[j,1]
                    prev = x1
                else:
                    # Other variables: x[j,i+1]
                    prev = X[:, idx - 1]

                curr = X[:, idx]

                # Gradient has two parts:
                # 1. From term where this variable appears: -2 * b[j,i] * (x[j,i+1] - x[j,i]^2)
                grad[:, idx] = -2 * b_ji * (curr - prev**2)

                # 2. From next term (if exists) where x[j,i+1] is squared
                if i < self.n1 - 2:
                    next_var = X[:, idx + 1]
                    b_j_i_plus_1 = self.b[j, i + 1]
                    # Add contribution: +4 * b[j,i+1] * x[j,i+1] * (x[j,i+2] - x[j,i+1]^2)
                    grad[:, idx] = (
                        grad[:, idx]
                        + 4 * b_j_i_plus_1 * curr * (next_var - curr**2)
                    )

        return grad


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
        X: Latent variables [batch_size, ndim]
        Y: Observations [batch_size, ndim] or [ndim] (broadcasted)
        rosenbrock_dist: RosenbrockDistribution instance
        sigma_obs: Observation noise standard deviation

    Returns:
        score: ∇log p(x|y), shape [batch_size, ndim]
    """
    # Ensure Y has correct shape
    if Y.ndim == 1:
        Y = Y.unsqueeze(0)  # [1, ndim]

    # Gaussian likelihood: y = x + noise, noise ~ N(0, σ²I)
    # ∇log p(y|x) = -(x - y) / σ²
    grad_log_likelihood = -(X - Y) / (sigma_obs**2)

    # Rosenbrock prior: ∇log p(x)
    grad_log_prior = rosenbrock_dist.gradlogpdf(X)

    return grad_log_likelihood + grad_log_prior
