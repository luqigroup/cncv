"""Hierarchical coupling layer for control variates (HINT-inspired).

Implements a binary tree of affine coupling operations that produces a dense
triangular Jacobian, where all but one dimension has a learnable diagonal entry.

Each tree uses a random input permutation for diversity (like HINT), and the
trace is invariant under permutations: tr(P^{-1} J P) = tr(J).

Inspired by: Kruse et al., "HINT: Hierarchical Invertible Neural Transport",
arXiv:1905.10687.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class _HierarchicalTree(nn.Module):
    """Binary tree of affine coupling operations for a single control variate.

    At each internal node, the input is split into upper and lower halves.
    The upper half is processed recursively, then used to condition the coupling
    of the lower half, and finally the lower half is processed recursively.

    This produces a dense triangular Jacobian where (for d dimensions, depth
    log2(d)) all but one diagonal entry is learnable.

    The Jacobian diagonal is computed recursively:
        - Leaf: diag = [1, ..., 1]
        - Internal node: diag = [upper_diag; S * lower_diag]

    Args:
        n_in: Input dimension for this subtree
        n_cond: Conditioning dimension (observation y)
        n_hidden: Hidden layer width for coupling MLPs
        depth: Remaining recursion depth (0 = leaf)
        n_mlp_layers: Number of layers in the coupling MLP (default: 3)
        activation: Activation for MLP hidden layers
        coupling_activation: Activation applied to scale S
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_hidden: int,
        depth: int,
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
        self.n_cond = n_cond
        self.depth = depth
        self.coupling_activation = coupling_activation

        if depth <= 0 or n_in <= 1:
            # Leaf node: identity transform
            self.is_leaf = True
            return

        self.is_leaf = False
        self.split_idx = n_in // 2
        upper_dim = self.split_idx
        lower_dim = n_in - self.split_idx

        # Coupling MLP: maps [processed_upper, cond] -> [S, T] for lower half
        coupling_input_dim = upper_dim + n_cond
        coupling_output_dim = 2 * lower_dim  # S and T

        # Build MLP with n_mlp_layers layers (matching flat layer's MLP depth)
        mlp_layers = []
        mlp_layers.append(nn.Linear(coupling_input_dim, n_hidden))
        mlp_layers.append(activation)
        for _ in range(n_mlp_layers - 2):
            mlp_layers.append(nn.Linear(n_hidden, n_hidden))
            mlp_layers.append(activation)
        final_layer = nn.Linear(n_hidden, coupling_output_dim)
        # Small init for last layer so initial g is small (like normalizing flows)
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
        mlp_layers.append(final_layer)
        self.coupling_mlp = nn.Sequential(*mlp_layers)

        # Recursive subtrees
        self.upper_tree = _HierarchicalTree(
            upper_dim, n_cond, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers,
            activation=activation, coupling_activation=coupling_activation,
        )
        self.lower_tree = _HierarchicalTree(
            lower_dim, n_cond, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers,
            activation=activation, coupling_activation=coupling_activation,
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the hierarchical tree.

        Args:
            x: Input tensor [batch, n_in]
            cond: Conditioning tensor [batch, n_cond]

        Returns:
            diag: Jacobian diagonal [batch, n_in]
            phi: Transformed output [batch, n_in]
        """
        if self.is_leaf:
            # Identity: diag = ones, phi = x
            diag = torch.ones_like(x)
            return diag, x

        # Split into upper and lower
        x_upper = x[:, :self.split_idx]
        x_lower = x[:, self.split_idx:]

        # Step 1: Process upper recursively
        upper_diag, phi_upper = self.upper_tree(x_upper, cond)

        # Step 2: Couple lower using processed upper
        coupling_input = torch.cat([phi_upper, cond], dim=1)
        coupling_out = self.coupling_mlp(coupling_input)
        lower_dim = self.n_in - self.split_idx
        S_raw = coupling_out[:, :lower_dim]
        T = coupling_out[:, lower_dim:]
        S = self.coupling_activation(S_raw)
        x_lower_coupled = S * x_lower + T

        # Step 3: Process lower recursively
        lower_diag, phi_lower = self.lower_tree(x_lower_coupled, cond)

        # Combine outputs
        phi = torch.cat([phi_upper, phi_lower], dim=1)

        # Combine diagonals: upper stays, lower gets multiplied by S
        diag = torch.cat([upper_diag, S * lower_diag], dim=1)

        return diag, phi


class HierarchicalCouplingLayerCV(nn.Module):
    """Hierarchical coupling layer for control variates.

    Wraps n_cv independent _HierarchicalTree instances, each with a random
    input permutation for diversity (like HINT). The trace is invariant under
    permutations: tr(P^{-1} J P) = tr(J).

    The phi output is un-permuted back to the original coordinate system
    so that phi · score is computed correctly.

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width for coupling MLPs
        depth: Recursion depth (default: auto = max(1, int(log2(n_in))))
        n_mlp_layers: Number of layers in each node's MLP (default: 3)
        activation: Activation for MLP hidden layers
        coupling_activation: Activation applied to scale S
        n_cv: Number of control variates (default: n_in)
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
        n_cv: Optional[int] = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        if coupling_activation is None:
            coupling_activation = nn.Identity()

        self.n_in = n_in
        self.n_cv = n_cv if n_cv is not None else n_in

        if depth is None:
            depth = max(1, int(math.log2(n_in))) if n_in > 1 else 0

        self.depth = depth

        # Create n_cv independent trees, each with a random input permutation
        self.trees = nn.ModuleList()
        # Store permutations and inverse permutations as buffers (non-trainable)
        perms = []
        inv_perms = []
        for k in range(self.n_cv):
            tree = _HierarchicalTree(
                n_in, n_cond, n_hidden, depth,
                n_mlp_layers=n_mlp_layers,
                activation=activation,
                coupling_activation=coupling_activation,
            )
            self.trees.append(tree)

            # Random permutation for this tree
            perm = torch.randperm(n_in)
            inv_perm = torch.argsort(perm)
            perms.append(perm)
            inv_perms.append(inv_perm)

        # Register as buffer so they move with .to(device) and are saved/loaded
        self.register_buffer("perms", torch.stack(perms))       # [n_cv, n_in]
        self.register_buffer("inv_perms", torch.stack(inv_perms))  # [n_cv, n_in]

    def forward(
        self, X: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing control variates.

        Each tree k operates on permuted input X[:, perm_k] and produces
        phi in permuted space. We un-permute phi back to original coordinates.
        The trace is invariant: tr(P^{-1} J P) = tr(J).

        Args:
            X: Input tensor [batch_size, n_in]
            C: Conditioning tensor [batch_size, n_cond]

        Returns:
            jac_traces: Jacobian traces [n_cv, batch_size]
            phi_all: Transformed outputs [n_cv, batch_size, n_in]
        """
        batch_size = X.shape[0]

        jac_traces = torch.zeros(
            self.n_cv, batch_size, device=X.device, dtype=X.dtype
        )
        phi_all = torch.zeros(
            self.n_cv, batch_size, self.n_in, device=X.device, dtype=X.dtype
        )

        for k, tree in enumerate(self.trees):
            # Permute input for this tree
            X_perm = X[:, self.perms[k]]

            # Forward through tree in permuted space
            diag_k, phi_perm_k = tree(X_perm, C)

            # Trace is invariant under permutation
            jac_traces[k, :] = diag_k.sum(dim=1)

            # Un-permute phi back to original coordinates
            phi_all[k, :, :] = phi_perm_k[:, self.inv_perms[k]]

        return jac_traces, phi_all
