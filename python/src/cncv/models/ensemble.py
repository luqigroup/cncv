"""Ensemble of coupling layers with different split directions.

This module implements an ensemble that combines multiple coupling layers
to overcome the coupling bottleneck.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from .coupling_layer import DenseConditionalLayerCV_Reversible


class EnsembleDenseCV(nn.Module):
    """Ensemble of coupling layers with different split directions.

    Combines control variates from different layers, allowing each to specialize
    on different input components.

    Args:
        layers: List of DenseConditionalLayerCV_Reversible modules
        combination_mode: How to combine CVs ('average' or 'learned_weights')

    Attributes:
        layers: ModuleList of coupling layers
        combination_mode: Combination strategy
        weights: Learnable weights if using learned_weights mode
    """

    def __init__(
        self,
        layers: List[DenseConditionalLayerCV_Reversible],
        combination_mode: str = "average",
    ):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.combination_mode = combination_mode

        if combination_mode == "learned_weights":
            # Initialize learnable weights
            weights = torch.ones(len(layers)) / len(layers)
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer("weights", None)

    def forward(
        self, X: torch.Tensor, C: torch.Tensor, score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute combined control variate.

        Args:
            X: Input [batch_size, n_in]
            C: Conditioning [batch_size, n_cond]
            score: Score function ∇log p(x|y) [batch_size, n_in]

        Returns:
            g_combined: Combined CV [n_cv, batch_size]
            all_g: Individual CVs [n_layers, n_cv, batch_size]
        """
        n_layers = len(self.layers)
        batch_size = X.shape[0]
        n_cv = self.layers[0].n_cv

        all_g = torch.zeros(
            n_layers, n_cv, batch_size, device=X.device, dtype=X.dtype
        )

        # Forward through each layer
        for i, layer in enumerate(self.layers):
            jac_traces, phi_all = layer(X, C)

            # Compute g = trace + φ · score for each CV
            for k in range(n_cv):
                trace_k = jac_traces[k, :]
                phi_k = phi_all[k, :, :]
                phi_dot_score_k = (phi_k * score).sum(dim=1)
                all_g[i, k, :] = trace_k + phi_dot_score_k

        # Combine based on mode
        if self.combination_mode == "average":
            g_combined = all_g.mean(dim=0)
        elif self.combination_mode == "learned_weights":
            # Weighted combination
            g_combined = (all_g * self.weights[:, None, None]).sum(dim=0)
        else:
            raise ValueError(
                f"Unknown combination_mode: {self.combination_mode}"
            )

        return g_combined, all_g


def create_forward_reverse_ensemble(
    n_in: int,
    n_cond: int,
    n_hidden: int,
    n_layers: int = 3,
    activation: nn.Module = None,
    n_cv: int = None,
) -> EnsembleDenseCV:
    """Helper to create a two-layer ensemble with forward and reverse splits.

    This is the simplest ensemble that addresses the coupling bottleneck.

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width
        n_layers: Number of MLP layers (default: 3)
        activation: Activation function (default: nn.Tanh())
        n_cv: Number of control variates (default: None, uses n_in)

    Returns:
        EnsembleDenseCV with two layers (forward and reverse splits)
    """
    if activation is None:
        activation = nn.Tanh()

    # Layer 1: Transform first half (original)
    layer1 = DenseConditionalLayerCV_Reversible(
        n_in,
        n_cond,
        n_hidden,
        n_layers,
        activation=activation,
        n_cv=n_cv,
        reverse_split=False,
    )

    # Layer 2: Transform second half (reversed)
    layer2 = DenseConditionalLayerCV_Reversible(
        n_in,
        n_cond,
        n_hidden,
        n_layers,
        activation=activation,
        n_cv=n_cv,
        reverse_split=True,
    )

    return EnsembleDenseCV([layer1, layer2], combination_mode="average")


def create_random_split_ensemble(
    n_in: int,
    n_cond: int,
    n_hidden: int,
    n_ensemble_members: int,
    n_layers: int = 3,
    activation: nn.Module = None,
    n_cv: int = None,
    seed: int = 1,
) -> EnsembleDenseCV:
    """Create an ensemble with random split directions.

    Each ensemble member will have a randomly chosen reverse_split direction.
    This allows for more diverse coverage of the input space compared to
    just forward/reverse splits.

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width
        n_ensemble_members: Number of ensemble members
        n_layers: Number of MLP layers (default: 3)
        activation: Activation function (default: nn.Tanh())
        n_cv: Number of control variates (default: None, uses n_in)
        seed: Random seed for split directions (default: None)

    Returns:
        EnsembleDenseCV with n_ensemble_members layers
    """
    if activation is None:
        activation = nn.ReLU()

    # Set random seed if provided for reproducible splits
    if seed is not None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)

    # Create ensemble members with random split directions
    ensemble_layers = []
    for i in range(n_ensemble_members):
        # Randomly choose split direction
        reverse_split = bool(torch.rand(1).item() > 0.5)

        layer = DenseConditionalLayerCV_Reversible(
            n_in,
            n_cond,
            n_hidden,
            n_layers,
            activation=activation,
            n_cv=n_cv,
            reverse_split=reverse_split,
        )
        ensemble_layers.append(layer)

    return EnsembleDenseCV(ensemble_layers, combination_mode="average")
