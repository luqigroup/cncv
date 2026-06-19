"""Ensemble of coupling layers with different split directions.

This module implements an ensemble that combines multiple coupling layers
to overcome the coupling bottleneck.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from .coupling_layer import DenseConditionalLayerCV_Reversible
from .hierarchical_coupling import HierarchicalCouplingLayerCV
from .shared_cv import SharedEnsembleCV, SharedHierarchicalCV, SharedEnsembleCVWrapper


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
        activation = nn.ReLU()

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
    """Create an ensemble with random split directions and split sizes.

    Each ensemble member will have:
    - A randomly chosen split_idx (where to split the input)
    - A randomly chosen reverse_split direction (which part to transform)

    This allows for diverse coverage of the input space with different
    split configurations.

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width
        n_ensemble_members: Number of ensemble members
        n_layers: Number of MLP layers (default: 3)
        activation: Activation function (default: nn.Tanh())
        n_cv: Number of control variates (default: None, uses n_in)
        seed: Random seed for reproducibility (default: 1)

    Returns:
        EnsembleDenseCV with n_ensemble_members layers

    Example:
        For n_in=10 and n_ensemble_members=4, might create:
        - Layer 0: split at 3 (3+7), reverse_split=False
        - Layer 1: split at 7 (7+3), reverse_split=True
        - Layer 2: split at 5 (5+5), reverse_split=False
        - Layer 3: split at 2 (2+8), reverse_split=True
    """
    if activation is None:
        activation = nn.Tanh()

    # Set random seed if provided for reproducible splits
    if seed is not None:
        import random

        random.seed(seed)
        torch.manual_seed(seed)

    # Create ensemble members with random split configurations
    ensemble_layers = []
    split_configs = []  # For logging

    for i in range(n_ensemble_members):
        # Randomly choose split index (avoid extremes: use [1, n_in-1])
        # Use uniform distribution over valid split points
        split_idx = torch.randint(1, n_in, (1,)).item()

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
            split_idx=split_idx,
        )
        ensemble_layers.append(layer)

        # Store config for logging
        n_in1 = split_idx
        n_in2 = n_in - split_idx
        split_configs.append((split_idx, n_in1, n_in2, reverse_split))

    # Print split configuration summary
    print("\n=== Ensemble Split Configuration ===")
    for i, (split_idx, n_in1, n_in2, reverse_split) in enumerate(split_configs):
        direction = "reverse" if reverse_split else "forward"
        transformed = f"second {n_in2}" if reverse_split else f"first {n_in1}"
        print(f"Layer {i}: split at {split_idx} ({n_in1}+{n_in2}), "
              f"{direction} (transforms {transformed} components)")

    return EnsembleDenseCV(ensemble_layers, combination_mode="average")


def create_hierarchical_ensemble(
    n_in: int,
    n_cond: int,
    n_hidden: int,
    n_ensemble_members: int,
    n_layers: int = 3,
    depth: int = None,
    activation: nn.Module = None,
    n_cv: int = None,
    seed: int = 1,
) -> EnsembleDenseCV:
    """Create an ensemble of hierarchical coupling layers.

    Each ensemble member is a HierarchicalCouplingLayerCV with its own
    independent set of parameters. The hierarchical tree structure handles
    dimension coverage internally (no random splits needed).

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width for coupling MLPs
        n_ensemble_members: Number of ensemble members
        n_layers: Number of MLP layers in each tree node (default: 3)
        depth: Recursion depth (default: auto = max(1, int(log2(n_in))))
        activation: Activation for MLP hidden layers (default: Tanh)
        n_cv: Number of control variates per member (default: n_in)
        seed: Random seed for reproducibility (default: 1)

    Returns:
        EnsembleDenseCV with n_ensemble_members hierarchical layers
    """
    if activation is None:
        activation = nn.Tanh()

    if seed is not None:
        torch.manual_seed(seed)

    ensemble_layers = []
    for i in range(n_ensemble_members):
        layer = HierarchicalCouplingLayerCV(
            n_in,
            n_cond,
            n_hidden,
            depth=depth,
            n_mlp_layers=n_layers,
            activation=activation,
            n_cv=n_cv,
        )
        ensemble_layers.append(layer)

    print(f"\n=== Hierarchical Ensemble Configuration ===")
    print(f"Members: {n_ensemble_members}, depth: {ensemble_layers[0].depth}, "
          f"n_in: {n_in}, n_hidden: {n_hidden}, n_mlp_layers: {n_layers}")

    return EnsembleDenseCV(ensemble_layers, combination_mode="average")


def create_shared_hierarchical_ensemble(
    n_in: int,
    n_cond: int,
    n_hidden: int,
    n_ensemble_members: int,
    n_layers: int = 3,
    depth: Optional[int] = None,
    activation: nn.Module = None,
    n_cv: int = None,
    seed: int = 1,
) -> SharedEnsembleCVWrapper:
    """Create an ensemble of L hierarchical trees for component-wise CV.

    Each member is an independent tree with a random input permutation.
    Returns a wrapper whose forward() matches EnsembleDenseCV output format:
        (g_combined [n_cv, batch], all_g [1, n_cv, batch])

    Args:
        n_in: Input dimension.
        n_cond: Conditioning dimension.
        n_hidden: Hidden layer width for coupling MLPs.
        n_ensemble_members: Number of ensemble members (L).
        n_layers: Number of MLP layers in each tree node (default: 3).
        depth: Recursion depth (default: auto = max(1, int(log2(n_in)))).
        activation: Activation for MLP hidden layers (default: ReLU).
        n_cv: Ignored (kept for API compatibility; always uses n_in).
        seed: Random seed for reproducibility (default: 1).

    Returns:
        SharedEnsembleCVWrapper with EnsembleDenseCV-compatible output.
    """
    if activation is None:
        activation = nn.ReLU()

    if seed is not None:
        torch.manual_seed(seed)

    layers = []
    for _ in range(n_ensemble_members):
        layer = SharedHierarchicalCV(
            n_in, n_cond, n_hidden,
            depth=depth,
            n_mlp_layers=n_layers,
            activation=activation,
        )
        layers.append(layer)

    shared_ensemble = SharedEnsembleCV(layers)
    wrapper = SharedEnsembleCVWrapper(shared_ensemble)

    n_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)

    print(f"\n=== Hierarchical Ensemble Configuration ===")
    print(f"Ensemble members (L): {n_ensemble_members}")
    print(f"n_in: {n_in}, n_hidden: {n_hidden}, n_mlp_layers: {n_layers}")
    print(f"Depth: {layers[0].depth}")
    print(f"Parameters: {n_params:,}")

    return wrapper
