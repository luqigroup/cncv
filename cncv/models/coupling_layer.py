"""Coupling layer with reversible split for control variates.

This module implements affine coupling layers that can transform different
components of the input based on the reverse_split parameter.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DenseConditionalLayerCV_Reversible(nn.Module):
    """Coupling layer with reversible split for control variates.

    When reverse_split=False (default):
        φ = [S(x₂, c) ⊙ x₁ + T(x₂, c); x₂]  # Transform x₁, pass x₂

    When reverse_split=True:
        φ = [x₁; S(x₁, c) ⊙ x₂ + T(x₁, c)]  # Pass x₁, transform x₂

    This allows different layers to focus on transforming different components,
    addressing the coupling layer bottleneck.

    Args:
        n_in: Input dimension
        n_cond: Conditioning dimension
        n_hidden: Hidden layer width
        n_layers: Number of MLP layers (default: 3)
        activation: Activation function (default: nn.Tanh())
        coupling_activation: Activation for S (default: nn.Identity())
        n_cv: Number of control variates (default: n_in)
        reverse_split: Whether to reverse which half is transformed (default: False)

    Attributes:
        split_idx: Index where input is split
        n_cv: Number of control variates
        coupling_activation: Activation function applied to S
        reverse_split: Whether using reverse split
        transform_dim: Dimension of the part being transformed
        mlp: The neural network producing S and T parameters
    """

    def __init__(
        self,
        n_in: int,
        n_cond: int,
        n_hidden: int,
        n_layers: int = 3,
        activation: nn.Module = None,
        coupling_activation: nn.Module = None,
        n_cv: int = None,
        reverse_split: bool = False,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        if coupling_activation is None:
            coupling_activation = nn.Identity()

        self.split_idx = n_in // 2
        self.n_cv = n_cv if n_cv is not None else n_in
        self.coupling_activation = coupling_activation
        self.reverse_split = reverse_split

        n_in1 = self.split_idx
        n_in2 = n_in - self.split_idx

        # Determine which part is used as input to MLP
        if reverse_split:
            # MLP input: [X1; C], output: transforms for X2
            input_dim = n_in1 + n_cond
            self.transform_dim = n_in2
        else:
            # MLP input: [X2; C], output: transforms for X1 (original)
            input_dim = n_in2 + n_cond
            self.transform_dim = n_in1

        # Output: n_cv sets of [S, T], each set is transform_dim-dimensional
        n_out = self.n_cv * 2 * self.transform_dim

        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, n_hidden))
        layers.append(activation)

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation)

        layers.append(nn.Linear(n_hidden, n_out))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, X: torch.Tensor, C: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing control variates.

        Args:
            X: Input tensor [batch_size, n_in]
            C: Conditioning tensor [batch_size, n_cond]

        Returns:
            jac_traces: Jacobian traces [n_cv, batch_size]
            phi_all: Transformed outputs [n_cv, batch_size, n_in]
        """
        batch_size = X.shape[0]
        n_in = X.shape[1]

        # Split input
        X1 = X[:, : self.split_idx]
        X2 = X[:, self.split_idx :]

        # Choose which part to use as MLP input
        if self.reverse_split:
            # Use X1 and C to transform X2
            mlp_input = torch.cat([X1, C], dim=1)
        else:
            # Use X2 and C to transform X1 (original behavior)
            mlp_input = torch.cat([X2, C], dim=1)

        # Forward through MLP
        output = self.mlp(mlp_input)

        # Compute control variates for each k
        jac_traces = torch.zeros(
            self.n_cv, batch_size, device=X.device, dtype=X.dtype
        )
        phi_all = torch.zeros(
            self.n_cv, batch_size, n_in, device=X.device, dtype=X.dtype
        )

        for k in range(self.n_cv):
            # Extract S and T for k-th CV
            idx_start = k * 2 * self.transform_dim
            S_raw_k = output[:, idx_start : idx_start + self.transform_dim]
            T_k = output[
                :,
                idx_start + self.transform_dim : idx_start
                + 2 * self.transform_dim,
            ]

            S_k = self.coupling_activation(S_raw_k)

            if self.reverse_split:
                # φ = [X1; S ⊙ X2 + T]
                Y2_k = S_k * X2 + T_k
                phi_k = torch.cat([X1, Y2_k], dim=1)

                # Manually compute Jacobian trace: sum(S_k) + dim(X1)
                jac_traces[k, :] = S_k.sum(dim=1) + self.split_idx
            else:
                # φ = [S ⊙ X1 + T; X2] (original)
                Y1_k = S_k * X1 + T_k
                phi_k = torch.cat([Y1_k, X2], dim=1)

                # Manually compute Jacobian trace: sum(S_k) + dim(X2)
                jac_traces[k, :] = S_k.sum(dim=1) + (n_in - self.split_idx)

            phi_all[k, :, :] = phi_k

        return jac_traces, phi_all
