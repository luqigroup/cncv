"""Gaussian posterior utilities for control variate training.

This module provides utilities for computing score functions and exact posteriors
for Gaussian inference problems.
"""

import torch
from typing import Tuple


def compute_score_posterior_gaussian(
    X: torch.Tensor,
    Y: torch.Tensor,
    mu_prior: torch.Tensor,
    Sigma_prior_inv: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """Compute score function for Gaussian posterior.

    The score function is the gradient of the log posterior:
        ∇log p(x|y) = ∇log p(y|x) + ∇log p(x)

    For Gaussian likelihood p(y|x) = N(x, σ²I) and prior p(x) = N(μ, Σ):
        ∇log p(y|x) = -(x - y) / σ²
        ∇log p(x) = -Σ⁻¹(x - μ)

    Args:
        X: Latent variables [batch_size, n_dim] or [n_dim]
        Y: Observations [batch_size, n_dim] or [n_dim]
        mu_prior: Prior mean [n_dim]
        Sigma_prior_inv: Inverse prior covariance [n_dim, n_dim]
        sigma: Observation noise standard deviation

    Returns:
        score: ∇log p(x|y), same shape as X
    """
    # Likelihood gradient: -(x - y) / σ²
    grad_likelihood = -(X - Y) / (sigma ** 2)

    # Prior gradient: -Σ⁻¹(x - μ)
    if X.ndim == 1:
        # Single sample
        grad_prior = -Sigma_prior_inv @ (X - mu_prior)
    else:
        # Batch of samples [batch_size, n_dim]
        # (X - mu_prior) has shape [batch_size, n_dim]
        # We need Σ⁻¹ @ (X - mu_prior).T, then transpose back
        grad_prior = -(Sigma_prior_inv @ (X - mu_prior).T).T

    return grad_likelihood + grad_prior


def compute_exact_posterior(
    y_obs: torch.Tensor,
    mu_prior: torch.Tensor,
    Sigma_prior: torch.Tensor,
    sigma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact Gaussian posterior analytically.

    For Gaussian prior p(x) = N(μ_prior, Σ_prior) and likelihood
    p(y|x) = N(x, σ²I), the posterior is:
        p(x|y) = N(μ_post, Σ_post)

    where:
        Σ_post = (Σ_prior⁻¹ + σ⁻²I)⁻¹
        μ_post = Σ_post (Σ_prior⁻¹ μ_prior + σ⁻² y)

    Args:
        y_obs: Observation [n_dim]
        mu_prior: Prior mean [n_dim]
        Sigma_prior: Prior covariance [n_dim, n_dim]
        sigma: Observation noise standard deviation

    Returns:
        mu_post: Posterior mean [n_dim]
        Sigma_post: Posterior covariance [n_dim, n_dim]
    """
    n = len(mu_prior)
    I = torch.eye(n, device=y_obs.device, dtype=y_obs.dtype)

    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
    Sigma_post = torch.linalg.inv(Sigma_prior_inv + (1 / sigma**2) * I)
    mu_post = Sigma_post @ (Sigma_prior_inv @ mu_prior + (1 / sigma**2) * y_obs)

    return mu_post, Sigma_post
