"""Utilities for Gaussian posterior inference."""

from .gaussian import compute_score_posterior_gaussian, compute_exact_posterior
from .lr_scheduler import CustomLRScheduler

__all__ = [
    "compute_score_posterior_gaussian",
    "compute_exact_posterior",
    "CustomLRScheduler"
]
