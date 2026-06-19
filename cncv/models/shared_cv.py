"""Component-wise control variates for vector-valued QoI.

The standard EnsembleDenseCV uses n_cv = d independent transport maps, one per
component of h(x). Each tree k produces a *scalar* CV:

    g_k = tr_k + phi_k . score    (dot product over all d dims -> scalar)

This requires d trees to control d components, which is infeasible at large d.

KEY INSIGHT: Stein's identity holds *per component*. For each component i:

    E[ h_i(x) - g_i(x) ] = 0   where  g_i = dT_i/du_i + T_i * s_i

Here T_i is the i-th output of a *single* transport map T: R^d -> R^d,
dT_i/du_i is the i-th diagonal entry of the Jacobian (NOT the full trace),
and s_i is the i-th component of the score. Everything is scalar * scalar.

So: g = diag + phi * score   (element-wise, shape [B, d])

One tree gives CVs for ALL d components simultaneously. An ensemble of L
trees with random permutations provides diversity.

Comparison:
    EnsembleDenseCV   : n_cv=d trees, g_k = trace_k + phi_k.score (scalar)
    SharedEnsembleCV  : L << d trees, g = diag + phi*score (vector, element-wise)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .hierarchical_coupling import _HierarchicalTree


class SharedHierarchicalCV(nn.Module):
    """Single hierarchical coupling tree for component-wise Stein CV.

    One _HierarchicalTree with a random input permutation produces
    component-wise CVs for all d dimensions simultaneously.

    Output: g = diag + phi * score    (element-wise, shape [B, d])

    Args:
        n_in: Input dimension d.
        n_cond: Conditioning dimension (observation y).
        n_hidden: Hidden layer width for coupling MLPs.
        depth: Recursion depth (default: auto).
        n_mlp_layers: MLP layers per coupling node.
        activation: MLP activation.
        coupling_activation: Activation for scale S.
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_hidden: int,
        depth: Optional[int] = None,
        n_mlp_layers: int = 3,
        activation: nn.Module = None,
        coupling_activation: nn.Module = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        if coupling_activation is None:
            coupling_activation = nn.Identity()

        self.n_in = n_in

        if depth is None:
            depth = max(1, int(math.log2(n_in))) if n_in > 1 else 0
        self.depth = depth

        self.tree = _HierarchicalTree(
            n_in, n_cond, n_hidden, depth,
            n_mlp_layers=n_mlp_layers,
            activation=activation,
            coupling_activation=coupling_activation,
        )

        # Random permutation for input diversity
        perm = torch.randperm(n_in)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)          # [n_in]
        self.register_buffer("inv_perm", inv_perm)   # [n_in]

    def forward(
        self, X: torch.Tensor, C: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Compute component-wise control variate.

        Args:
            X: Input [batch_size, n_in].
            C: Conditioning [batch_size, n_cond].
            score: Posterior score [batch_size, n_in].

        Returns:
            g: Component-wise CV [batch_size, n_in].
        """
        # Permute input
        X_perm = X[:, self.perm]

        # Forward through tree -> (diag, phi) in permuted space
        diag_perm, phi_perm = self.tree(X_perm, C)

        # Un-permute back to original coordinates
        diag = diag_perm[:, self.inv_perm]
        phi = phi_perm[:, self.inv_perm]

        # Component-wise Stein CV: g_i = diag_i + phi_i * score_i
        return diag + phi * score


class SharedEnsembleCV(nn.Module):
    """Ensemble of L SharedHierarchicalCV members.

    Each member is an independent tree with its own random permutation.
    The ensemble averages their outputs.

    The loss is: ||h(x) - g(x)||^2 averaged over batch and components,
    where h(x) = x (posterior mean estimation) and g is the CV.

    Args:
        layers: List of SharedHierarchicalCV modules.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, X: torch.Tensor, C: torch.Tensor, score: torch.Tensor
    ) -> torch.Tensor:
        """Compute averaged component-wise CV.

        Args:
            X: Input [batch_size, n_in].
            C: Conditioning [batch_size, n_cond].
            score: Posterior score [batch_size, n_in].

        Returns:
            g: Component-wise CV [batch_size, n_in].
        """
        g_sum = torch.zeros_like(X)
        for layer in self.layers:
            g_sum = g_sum + layer(X, C, score)
        return g_sum / len(self.layers)


class SharedEnsembleCVWrapper(nn.Module):
    """Wrapper that adapts SharedEnsembleCV output to EnsembleDenseCV interface.

    EnsembleDenseCV.forward() returns (g_combined [n_cv, batch], all_g [n_layers, n_cv, batch]).
    SharedEnsembleCV.forward() returns g [batch, n_in].

    This wrapper transposes so all existing loss computation (h_x.T - g_combined),
    evaluation (g_combined[k, :]), and plotting code works unchanged.

    Args:
        shared_ensemble: A SharedEnsembleCV instance.
    """

    def __init__(self, shared_ensemble: SharedEnsembleCV):
        super().__init__()
        self.shared_ensemble = shared_ensemble

    @property
    def layers(self):
        return self.shared_ensemble.layers

    @property
    def n_cv(self):
        return self.shared_ensemble.layers[0].n_in

    def forward(
        self, X: torch.Tensor, C: torch.Tensor, score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute control variate with EnsembleDenseCV-compatible output shape.

        Args:
            X: Input [batch_size, n_in].
            C: Conditioning [batch_size, n_cond].
            score: Posterior score [batch_size, n_in].

        Returns:
            g_combined: Combined CV [n_in, batch_size] (transposed).
            all_g: Individual CVs [1, n_in, batch_size] (single "layer").
        """
        g = self.shared_ensemble(X, C, score)  # [batch, n_in]
        g_combined = g.T  # [n_in, batch]
        all_g = g_combined.unsqueeze(0)  # [1, n_in, batch]
        return g_combined, all_g


def create_shared_ensemble(
    n_in: int,
    n_cond: int,
    n_hidden: int,
    n_ensemble: int = 16,
    depth: Optional[int] = None,
    n_mlp_layers: int = 3,
    activation: nn.Module = None,
    coupling_activation: nn.Module = None,
    seed: int = 12,
) -> SharedEnsembleCV:
    """Factory for component-wise ensemble.

    Creates L independent trees, each with a random input permutation.

    Args:
        n_in: Input dimension d.
        n_cond: Conditioning dimension.
        n_hidden: Hidden width per coupling MLP.
        n_ensemble: Number of ensemble members (L).
        depth: Recursion depth (default: auto).
        n_mlp_layers: MLP layers per node.
        activation: MLP activation (default: ReLU).
        coupling_activation: Scale activation (default: Identity).
        seed: Random seed.

    Returns:
        SharedEnsembleCV ready for training.

    Example:
        model = create_shared_ensemble(25, 33, 128, n_ensemble=16)
        g = model(X, Y, score)     # [B, 25] -- one CV per component
        loss = ((X - g) ** 2).mean()  # component-wise loss
    """
    if activation is None:
        activation = nn.ReLU()
    if coupling_activation is None:
        coupling_activation = nn.Identity()

    if seed is not None:
        torch.manual_seed(seed)

    layers = []
    for _ in range(n_ensemble):
        layer = SharedHierarchicalCV(
            n_in, n_cond, n_hidden,
            depth=depth,
            n_mlp_layers=n_mlp_layers,
            activation=activation,
            coupling_activation=coupling_activation,
        )
        layers.append(layer)

    ensemble = SharedEnsembleCV(layers)

    n_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)

    print(f"\n=== Ensemble Configuration ===")
    print(f"  n_in={n_in}, n_cond={n_cond}, n_hidden={n_hidden}")
    print(f"  Ensemble members (L): {n_ensemble}")
    print(f"  Depth: {layers[0].depth}")
    print(f"  Parameters: {n_params:,}")

    return ensemble
