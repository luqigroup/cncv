"""Conditional Neural Control Variates (CNCV) in PyTorch.

This package implements ensemble neural control variates for variance reduction
in Bayesian posterior mean estimation.
"""

from .models import (
    DenseConditionalLayerCV_Reversible,
    EnsembleDenseCV,
    create_forward_reverse_ensemble,
    create_random_split_ensemble,
)
from .utils import (
    compute_score_posterior_gaussian,
    compute_exact_posterior,
)

__version__ = "0.1.0"

__all__ = [
    "DenseConditionalLayerCV_Reversible",
    "EnsembleDenseCV",
    "create_forward_reverse_ensemble",
    "create_random_split_ensemble",
    "compute_score_posterior_gaussian",
    "compute_exact_posterior",
]
