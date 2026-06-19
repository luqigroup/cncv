"""Conditional Normalizing Flow with HINT-style hierarchical coupling layers.

Builds a conditional normalizing flow p(x|y) using the same binary-tree coupling
architecture as the CNCV hierarchical layers, but with:
  - exp(clamp(s)) coupling activation for invertibility (S > 0 always)
  - inverse() method for sampling z -> x
  - log_prob() and score() for density evaluation

The flow is a stack of K trees with random permutations between them.

References:
  Kruse et al., "HINT: Hierarchical Invertible Neural Transport", arXiv:1905.10687
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class _NFTree(nn.Module):
    """Single HINT-style tree for normalizing flow (forward + inverse).

    Same binary-tree architecture as _HierarchicalTree, but with:
      - coupling activation = exp(clamp(s, -5, 5)) for guaranteed S > 0
      - inverse() method for sampling

    Forward: x, cond -> z, log_det  (encoding)
    Inverse: z, cond -> x           (sampling)
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_hidden: int,
        depth: int,
        n_mlp_layers: int = 3,
        activation: nn.Module = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()

        self.n_in = n_in
        self.n_cond = n_cond
        self.depth = depth

        if depth <= 0 or n_in <= 1:
            self.is_leaf = True
            return

        self.is_leaf = False
        self.split_idx = n_in // 2
        upper_dim = self.split_idx
        lower_dim = n_in - self.split_idx

        # Coupling MLP: maps [z_upper, cond] -> [log_S, T] for lower half
        coupling_input_dim = upper_dim + n_cond
        coupling_output_dim = 2 * lower_dim

        mlp_layers = []
        mlp_layers.append(nn.Linear(coupling_input_dim, n_hidden))
        mlp_layers.append(activation)
        for _ in range(n_mlp_layers - 2):
            mlp_layers.append(nn.Linear(n_hidden, n_hidden))
            mlp_layers.append(activation)
        final_layer = nn.Linear(n_hidden, coupling_output_dim)
        nn.init.normal_(final_layer.weight, std=0.01)
        nn.init.zeros_(final_layer.bias)
        mlp_layers.append(final_layer)
        self.coupling_mlp = nn.Sequential(*mlp_layers)

        # Recursive subtrees
        self.upper_tree = _NFTree(
            upper_dim, n_cond, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, activation=activation,
        )
        self.lower_tree = _NFTree(
            lower_dim, n_cond, n_hidden, depth - 1,
            n_mlp_layers=n_mlp_layers, activation=activation,
        )

    def _compute_coupling(self, z_upper, cond):
        """Compute affine coupling parameters S, T from upper output and cond."""
        coupling_input = torch.cat([z_upper, cond], dim=1)
        coupling_out = self.coupling_mlp(coupling_input)
        lower_dim = self.n_in - self.split_idx
        log_S_raw = coupling_out[:, :lower_dim]
        T = coupling_out[:, lower_dim:]
        log_S = torch.clamp(log_S_raw, -5.0, 5.0)
        S = torch.exp(log_S)
        return S, T, log_S

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x -> z with log|det J|.

        Args:
            x: Input tensor [batch, n_in]
            cond: Conditioning tensor [batch, n_cond]

        Returns:
            z: Encoded tensor [batch, n_in]
            log_det: Log absolute determinant of Jacobian [batch]
        """
        if self.is_leaf:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        x_upper = x[:, :self.split_idx]
        x_lower = x[:, self.split_idx:]

        # Step 1: Process upper recursively
        z_upper, ld_upper = self.upper_tree(x_upper, cond)

        # Step 2: Couple lower using processed upper (z_upper)
        S, T, log_S = self._compute_coupling(z_upper, cond)
        x_lower_coupled = S * x_lower + T

        # Step 3: Process lower recursively
        z_lower, ld_lower = self.lower_tree(x_lower_coupled, cond)

        z = torch.cat([z_upper, z_lower], dim=1)
        # log|det J| = sum of log|diag entries|
        # diag = [upper_diag; S * lower_diag]
        # log|det| = log|det_upper| + sum(log S) + log|det_lower|
        log_det = ld_upper + log_S.sum(dim=1) + ld_lower

        return z, log_det

    def inverse(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Inverse pass: z -> x (reverse of forward).

        Args:
            z: Encoded tensor [batch, n_in]
            cond: Conditioning tensor [batch, n_cond]

        Returns:
            x: Decoded tensor [batch, n_in]
        """
        if self.is_leaf:
            return z

        z_upper = z[:, :self.split_idx]
        z_lower = z[:, self.split_idx:]

        # Step 1: Recover x_upper from z_upper
        x_upper = self.upper_tree.inverse(z_upper, cond)

        # Step 2: Compute coupling params (using z_upper, same as forward)
        S, T, _ = self._compute_coupling(z_upper, cond)

        # Step 3: Recover x_lower_coupled from z_lower
        x_lower_coupled = self.lower_tree.inverse(z_lower, cond)

        # Step 4: Invert affine coupling
        x_lower = (x_lower_coupled - T) / S

        return torch.cat([x_upper, x_lower], dim=1)


class ConditionalNF(nn.Module):
    """Conditional normalizing flow = K stacked HINT trees with permutations.

    Learns p(x|y) via maximum likelihood. Each flow layer is an _NFTree with
    a random input permutation for expressiveness.

    Args:
        n_in: Input dimension (x)
        n_cond: Conditioning dimension (y)
        n_hidden: Hidden layer width for coupling MLPs
        n_flow_layers: Number of stacked trees (K)
        depth: Tree recursion depth (default: auto = max(1, int(log2(n_in))))
        n_mlp_layers: Number of layers in each coupling MLP
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_hidden: int,
        n_flow_layers: int = 8,
        depth: Optional[int] = None,
        n_mlp_layers: int = 3,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_cond = n_cond

        if depth is None:
            depth = max(1, int(math.log2(n_in))) if n_in > 1 else 0

        self.depth = depth

        # Create K trees, each with a random permutation
        self.trees = nn.ModuleList()
        perms = []
        inv_perms = []

        for k in range(n_flow_layers):
            tree = _NFTree(
                n_in, n_cond, n_hidden, depth,
                n_mlp_layers=n_mlp_layers,
            )
            self.trees.append(tree)

            perm = torch.randperm(n_in)
            inv_perm = torch.argsort(perm)
            perms.append(perm)
            inv_perms.append(inv_perm)

        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("inv_perms", torch.stack(inv_perms))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode: x, y -> z, log_det (apply trees sequentially).

        Args:
            x: Input [batch, n_in]
            y: Conditioning [batch, n_cond]

        Returns:
            z: Encoded [batch, n_in]
            log_det: Total log|det J| [batch]
        """
        z = x
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for k, tree in enumerate(self.trees):
            # Permute
            z = z[:, self.perms[k]]
            # Tree forward
            z, ld = tree(z, y)
            log_det = log_det + ld

        return z, log_det

    def inverse(
        self, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Sample: z, y -> x (apply trees in reverse order).

        Args:
            z: Latent [batch, n_in]
            y: Conditioning [batch, n_cond]

        Returns:
            x: Decoded [batch, n_in]
        """
        x = z
        for k in reversed(range(len(self.trees))):
            # Inverse tree
            x = self.trees[k].inverse(x, y)
            # Inverse permute
            x = x[:, self.inv_perms[k]]

        return x

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log p(x|y) = log p_z(z) + log|det J|.

        Args:
            x: Input [batch, n_in]
            y: Conditioning [batch, n_cond]

        Returns:
            log_p: Log probability [batch]
        """
        z, log_det = self.forward(x, y)
        # log p_z(z) = -0.5 ||z||^2 - 0.5 * d * log(2*pi)
        log_pz = -0.5 * z.pow(2).sum(dim=1) - 0.5 * self.n_in * math.log(2 * math.pi)
        return log_pz + log_det

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute score: nabla_x log p(x|y) via autograd.

        Uses torch.enable_grad() so this works even inside torch.no_grad() blocks.

        Args:
            x: Input [batch, n_in] (will be detached and requires_grad set)
            y: Conditioning [batch, n_cond]

        Returns:
            grad: Score [batch, n_in]
        """
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            log_p = self.log_prob(x, y)
            grad = torch.autograd.grad(
                log_p.sum(), x, create_graph=self.training
            )[0]
        return grad.detach()

    def sample(self, y: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample x ~ p(x|y) by sampling z ~ N(0,I) then inverting.

        Args:
            y: Conditioning [n_cond] (single observation) or [batch, n_cond]
            n_samples: Number of samples to draw

        Returns:
            x: Samples [n_samples, n_in]
        """
        if y.ndim == 1:
            y = y.unsqueeze(0)
        # Expand y to match n_samples
        y_expanded = y.expand(n_samples, -1)

        z = torch.randn(n_samples, self.n_in, device=y.device, dtype=y.dtype)
        return self.inverse(z, y_expanded)
