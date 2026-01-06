"""Tests for Jacobian trace computation in coupling layers."""

import pytest
import torch
import numpy as np

from cncv.models import DenseConditionalLayerCV_Reversible
from tests.utils import finite_diff_jacobian


class TestJacobianTrace:
    """Tests for Jacobian trace computation in coupling layers."""

    def test_coupling_layer_jac_trace_finite_diff_forward(self):
        """Verify computed jac trace matches finite difference (forward split)."""
        # Setup
        torch.manual_seed(42)
        n_in = 4
        n_cond = 2
        n_hidden = 8
        n_layers = 2
        batch_size = 3
        n_cv = 2

        # Create layer (forward split)
        layer = DenseConditionalLayerCV_Reversible(
            n_in, n_cond, n_hidden, n_layers=n_layers, n_cv=n_cv, reverse_split=False
        )
        layer.eval()  # Disable dropout if any

        # Create small test batch
        X = torch.randn(batch_size, n_in, dtype=torch.float32) * 0.1
        C = torch.randn(batch_size, n_cond, dtype=torch.float32) * 0.1

        # Compute Jacobian trace from layer
        jac_traces_computed, phi_all = layer(X, C)

        # For each CV and each sample, verify against finite differences
        eps = 1e-5
        for k in range(n_cv):
            for b in range(batch_size):
                # Define function for this specific CV and sample
                def forward_func(x_vec):
                    """Map flattened x to flattened phi_k."""
                    x_sample = x_vec.view(1, n_in)
                    c_sample = C[b:b+1, :]
                    _, phi_out = layer(x_sample, c_sample)
                    return phi_out[k, 0, :]  # [n_in]

                # Compute explicit Jacobian via finite differences
                J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)  # [n_in, n_in]

                # Compute trace explicitly
                jac_trace_explicit = torch.trace(J)

                # Compare with computed trace
                jac_trace_computed_val = jac_traces_computed[k, b]

                # Use relative tolerance
                assert torch.isclose(
                    jac_trace_computed_val, jac_trace_explicit, rtol=1e-3, atol=1e-4
                ), f"CV {k}, sample {b}: computed={jac_trace_computed_val:.6f}, explicit={jac_trace_explicit:.6f}"

    def test_coupling_layer_jac_trace_finite_diff_reverse(self):
        """Verify computed jac trace matches finite difference (reverse split)."""
        # Setup
        torch.manual_seed(43)
        n_in = 6
        n_cond = 3
        n_hidden = 12
        n_layers = 2
        batch_size = 2
        n_cv = 2

        # Create layer (reverse split)
        layer = DenseConditionalLayerCV_Reversible(
            n_in, n_cond, n_hidden, n_layers=n_layers, n_cv=n_cv, reverse_split=True
        )
        layer.eval()

        # Create small test batch
        X = torch.randn(batch_size, n_in, dtype=torch.float32) * 0.1
        C = torch.randn(batch_size, n_cond, dtype=torch.float32) * 0.1

        # Compute Jacobian trace from layer
        jac_traces_computed, phi_all = layer(X, C)

        # For each CV and each sample, verify against finite differences
        eps = 1e-5
        for k in range(n_cv):
            for b in range(batch_size):
                # Define function for this specific CV and sample
                def forward_func(x_vec):
                    """Map flattened x to flattened phi_k."""
                    x_sample = x_vec.view(1, n_in)
                    c_sample = C[b:b+1, :]
                    _, phi_out = layer(x_sample, c_sample)
                    return phi_out[k, 0, :]  # [n_in]

                # Compute explicit Jacobian via finite differences
                J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)  # [n_in, n_in]

                # Compute trace explicitly
                jac_trace_explicit = torch.trace(J)

                # Compare with computed trace
                jac_trace_computed_val = jac_traces_computed[k, b]

                # Use relative tolerance
                assert torch.isclose(
                    jac_trace_computed_val, jac_trace_explicit, rtol=1e-3, atol=1e-4
                ), f"CV {k}, sample {b}: computed={jac_trace_computed_val:.6f}, explicit={jac_trace_explicit:.6f}"

    def test_jac_trace_manual_formula(self):
        """Test that manual Jacobian trace formula is correct."""
        # Setup
        torch.manual_seed(44)
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batch_size = 1
        n_cv = 1

        # Forward split layer
        layer_forward = DenseConditionalLayerCV_Reversible(
            n_in, n_cond, n_hidden, n_layers=2, n_cv=n_cv, reverse_split=False
        )
        layer_forward.eval()

        X = torch.randn(batch_size, n_in) * 0.1
        C = torch.randn(batch_size, n_cond) * 0.1

        # Get splits
        split_idx = layer_forward.split_idx
        X1 = X[:, :split_idx]
        X2 = X[:, split_idx:]

        # Forward pass
        jac_trace, phi = layer_forward(X, C)

        # Manual computation
        # For forward split: φ = [S ⊙ X1 + T; X2]
        # Jacobian diagonal: [S[0], S[1], ..., S[split_idx-1], 1, 1, ..., 1]
        # Trace = sum(S) + (n_in - split_idx)

        # Extract S from the layer's computation
        mlp_input = torch.cat([X2, C], dim=1)
        output = layer_forward.mlp(mlp_input)
        S_raw = output[:, :split_idx]
        S = layer_forward.coupling_activation(S_raw)

        expected_trace = S.sum() + (n_in - split_idx)

        assert torch.isclose(jac_trace[0, 0], expected_trace, rtol=1e-6)

    def test_ensemble_jac_trace(self):
        """Test that ensemble properly combines Jacobian traces."""
        from cncv.models import create_forward_reverse_ensemble

        torch.manual_seed(45)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        n_layers = 2
        batch_size = 5

        # Create 2-layer ensemble (forward + reverse)
        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=n_layers, n_cv=n_in
        )
        ensemble.eval()

        X = torch.randn(batch_size, n_in) * 0.1
        C = torch.randn(batch_size, n_cond) * 0.1

        # Get individual layer outputs
        jac_trace_1, phi_1 = ensemble.layers[0](X, C)
        jac_trace_2, phi_2 = ensemble.layers[1](X, C)

        # Verify layers have different reverse_split settings
        assert ensemble.layers[0].reverse_split == False
        assert ensemble.layers[1].reverse_split == True

        # Verify they produce different Jacobian traces
        assert not torch.allclose(jac_trace_1, jac_trace_2, rtol=1e-2)

        # Forward through ensemble (should average)
        # Note: ensemble.forward expects score as third argument
        score = torch.randn(batch_size, n_in)
        g_combined, all_g = ensemble(X, C, score)

        # Verify we got outputs from both layers
        assert len(all_g) == 2

        # Note: g_combined includes both jac_trace and phi @ score components
        # We can't easily verify the averaging here without computing the full CV formula
        # But we can verify that all_g has the right structure
        assert all_g[0].shape == (n_in, batch_size)
        assert all_g[1].shape == (n_in, batch_size)

    def test_jac_trace_different_activations(self):
        """Test Jacobian trace with different coupling activations."""
        torch.manual_seed(46)
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batch_size = 2

        X = torch.randn(batch_size, n_in) * 0.1
        C = torch.randn(batch_size, n_cond) * 0.1

        # Test with different activations
        activations = [
            ("identity", torch.nn.Identity()),
            ("tanh", torch.nn.Tanh()),
            ("exp", torch.nn.Softplus()),  # Similar to exp but more stable
        ]

        for name, activation in activations:
            layer = DenseConditionalLayerCV_Reversible(
                n_in,
                n_cond,
                n_hidden,
                n_layers=2,
                n_cv=1,
                coupling_activation=activation,
            )
            layer.eval()

            # Compute trace
            jac_trace, _ = layer(X, C)

            # Verify against finite differences
            eps = 1e-5

            def forward_func(x_vec):
                x_sample = x_vec.view(1, n_in)
                c_sample = C[0:1, :]
                _, phi_out = layer(x_sample, c_sample)
                return phi_out[0, 0, :]

            J = finite_diff_jacobian(forward_func, X[0, :], eps=eps)
            jac_trace_explicit = torch.trace(J)

            assert torch.isclose(
                jac_trace[0, 0], jac_trace_explicit, rtol=1e-2, atol=1e-3
            ), f"Activation {name}: computed={jac_trace[0, 0]:.6f}, explicit={jac_trace_explicit:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
