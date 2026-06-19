"""Conditional Neural Control Variates (CNCV) in PyTorch.

This package implements ensemble neural control variates for variance reduction
in Bayesian posterior mean estimation, as well as diffusion models for learning
posterior score functions.
"""

from .models import (
    DenseConditionalLayerCV_Reversible,
    HierarchicalCouplingLayerCV,
    EnsembleDenseCV,
    create_forward_reverse_ensemble,
    create_random_split_ensemble,
    create_hierarchical_ensemble,
    create_shared_hierarchical_ensemble,
    load_ensemble_checkpoint,
    validate_split_config,
    get_split_config,
)
from .models.ddpm_mlp import ConditionalScoreMLPSimple
from .diffusion import NoiseScheduler
from .utils import (
    compute_score_posterior_gaussian,
    compute_exact_posterior,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from .plotting import (
    plot_results,
    plot_training_loss,
    plot_mse_comparison,
    plot_estimator_distributions,
    plot_vrf_vs_sample_size,
    plot_variance_comparison,
    plot_scatter_h_vs_gcv,
    plot_diagonal_correlations,
    plot_posterior_vs_prior,
    compute_h_x,
    PLOT_CONFIG,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "DenseConditionalLayerCV_Reversible",
    "HierarchicalCouplingLayerCV",
    "EnsembleDenseCV",
    "create_forward_reverse_ensemble",
    "create_random_split_ensemble",
    "create_hierarchical_ensemble",
    "create_shared_hierarchical_ensemble",
    "load_ensemble_checkpoint",
    "validate_split_config",
    "get_split_config",
    # DDPM
    "ConditionalScoreMLPSimple",
    "NoiseScheduler",
    # Utils
    "compute_score_posterior_gaussian",
    "compute_exact_posterior",
    "RosenbrockDistribution",
    "compute_score_posterior_rosenbrock",
    # Plotting
    "plot_results",
    "plot_training_loss",
    "plot_mse_comparison",
    "plot_estimator_distributions",
    "plot_vrf_vs_sample_size",
    "plot_variance_comparison",
    "plot_scatter_h_vs_gcv",
    "plot_diagonal_correlations",
    "plot_posterior_vs_prior",
    "compute_h_x",
    "PLOT_CONFIG",
]
