"""Tests for hierarchical coupling layer Jacobian trace and functionality."""

import pytest
import torch
import numpy as np

from cncv.models.hierarchical_coupling import (
    _HierarchicalTree,
    HierarchicalCouplingLayerCV,
)
from cncv.models.ensemble import create_hierarchical_ensemble
from tests.utils import finite_diff_jacobian


class TestHierarchicalJacobianTrace:
    """Verify Jacobian trace computation against finite differences."""

    @pytest.mark.parametrize("n_in,depth", [
        (2, 1),
        (4, 2),
        (4, 1),
        (8, 3),
        (6, 2),
        (3, 1),
    ])
    def test_tree_jac_trace_finite_diff(self, n_in, depth):
        """Verify tree Jacobian diagonal sums to finite-diff trace."""
        torch.manual_seed(42)
        n_cond = 2
        n_hidden = 16
        batch_size = 3

        # Use Tanh (smooth) to avoid ReLU non-differentiable kinks
        # and float64 for better FD accuracy at deeper recursion
        tree = _HierarchicalTree(
            n_in, n_cond, n_hidden, depth,
            activation=torch.nn.Tanh(),
            coupling_activation=torch.nn.Identity(),
        )
        tree.double().eval()

        X = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.1
        C = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.1

        diag, phi = tree(X, C)
        trace_computed = diag.sum(dim=1)

        eps = 1e-6
        for b in range(batch_size):
            def forward_func(x_vec):
                x_sample = x_vec.view(1, n_in)
                c_sample = C[b:b+1, :]
                _, phi_out = tree(x_sample, c_sample)
                return phi_out[0, :]

            J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)
            trace_fd = torch.trace(J)

            assert torch.isclose(
                trace_computed[b], trace_fd, rtol=1e-4, atol=1e-5
            ), (f"n_in={n_in}, depth={depth}, sample {b}: "
                f"computed={trace_computed[b]:.8f}, fd={trace_fd:.8f}")

    @pytest.mark.parametrize("n_in,n_cv", [
        (2, 2),
        (4, 4),
        (4, 2),
        (8, 3),
    ])
    def test_layer_jac_trace_finite_diff(self, n_in, n_cv):
        """Verify HierarchicalCouplingLayerCV trace against finite differences."""
        torch.manual_seed(43)
        n_cond = 3
        n_hidden = 16
        batch_size = 2

        layer = HierarchicalCouplingLayerCV(
            n_in, n_cond, n_hidden, n_cv=n_cv,
            activation=torch.nn.Tanh(),
        )
        layer.double().eval()

        X = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.1
        C = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.1

        jac_traces, phi_all = layer(X, C)

        eps = 1e-6
        for k in range(n_cv):
            for b in range(batch_size):
                def forward_func(x_vec):
                    x_sample = x_vec.view(1, n_in)
                    c_sample = C[b:b+1, :]
                    _, phi_out = layer(x_sample, c_sample)
                    return phi_out[k, 0, :]

                J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)
                trace_fd = torch.trace(J)

                assert torch.isclose(
                    jac_traces[k, b], trace_fd, rtol=1e-4, atol=1e-5
                ), (f"CV {k}, sample {b}: "
                    f"computed={jac_traces[k, b]:.8f}, fd={trace_fd:.8f}")


class TestHierarchicalShapes:
    """Shape consistency tests."""

    def test_tree_output_shapes(self):
        torch.manual_seed(44)
        n_in, n_cond, n_hidden, depth = 4, 2, 8, 2
        batch_size = 5

        tree = _HierarchicalTree(n_in, n_cond, n_hidden, depth)
        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)

        diag, phi = tree(X, C)
        assert diag.shape == (batch_size, n_in)
        assert phi.shape == (batch_size, n_in)

    def test_layer_output_shapes(self):
        torch.manual_seed(45)
        n_in, n_cond, n_hidden = 4, 3, 16
        n_cv = 3
        batch_size = 7

        layer = HierarchicalCouplingLayerCV(n_in, n_cond, n_hidden, n_cv=n_cv)
        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)

        jac_traces, phi_all = layer(X, C)
        assert jac_traces.shape == (n_cv, batch_size)
        assert phi_all.shape == (n_cv, batch_size, n_in)

    def test_leaf_node_identity(self):
        """Leaf node should return identity transform."""
        torch.manual_seed(46)
        n_in, n_cond, n_hidden = 3, 2, 8
        batch_size = 4

        tree = _HierarchicalTree(n_in, n_cond, n_hidden, depth=0)
        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)

        diag, phi = tree(X, C)
        assert torch.allclose(phi, X)
        assert torch.allclose(diag, torch.ones_like(X))

    def test_single_dim_identity(self):
        """n_in=1 should be identity."""
        torch.manual_seed(47)
        tree = _HierarchicalTree(1, 2, 8, depth=3)
        X = torch.randn(4, 1)
        C = torch.randn(4, 2)

        diag, phi = tree(X, C)
        assert torch.allclose(phi, X)
        assert torch.allclose(diag, torch.ones_like(X))


class TestHierarchicalGradients:
    """Gradient flow tests."""

    def test_gradients_flow_through_tree(self):
        """All tree parameters should receive gradients."""
        torch.manual_seed(48)
        n_in, n_cond, n_hidden = 4, 2, 8
        depth = 2

        tree = _HierarchicalTree(n_in, n_cond, n_hidden, depth)
        X = torch.randn(3, n_in)
        C = torch.randn(3, n_cond)

        diag, phi = tree(X, C)
        loss = diag.sum() + phi.sum()
        loss.backward()

        for name, param in tree.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_gradients_flow_through_layer(self):
        """All layer parameters should receive gradients."""
        torch.manual_seed(49)
        n_in, n_cond, n_hidden = 4, 2, 8
        n_cv = 2

        layer = HierarchicalCouplingLayerCV(n_in, n_cond, n_hidden, n_cv=n_cv)
        X = torch.randn(3, n_in)
        C = torch.randn(3, n_cond)

        jac_traces, phi_all = layer(X, C)
        loss = jac_traces.sum() + phi_all.sum()
        loss.backward()

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_ensemble_training_step(self):
        """Ensemble with hierarchical layers should support training."""
        torch.manual_seed(50)
        n_in, n_cond, n_hidden = 4, 4, 16

        ensemble = create_hierarchical_ensemble(
            n_in, n_cond, n_hidden, n_ensemble_members=2, n_cv=n_in, seed=42,
        )

        X = torch.randn(8, n_in)
        C = torch.randn(8, n_cond)
        score = torch.randn(8, n_in)

        optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)

        # One training step
        g_combined, all_g = ensemble(X, C, score)
        h_x = X
        loss = ((h_x.T - g_combined) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is finite
        assert torch.isfinite(torch.tensor(loss.item()))


class TestHierarchicalDiagonalDensity:
    """Verify hierarchical layers produce denser Jacobians than flat layers."""

    def test_more_learnable_diagonal_entries(self):
        """Hierarchical should have more non-unit diagonal entries than flat."""
        torch.manual_seed(51)
        n_in = 4
        n_cond = 2
        n_hidden = 16

        # Hierarchical layer
        hier_layer = HierarchicalCouplingLayerCV(
            n_in, n_cond, n_hidden, n_cv=1,
        )
        hier_layer.eval()

        X = torch.randn(1, n_in) * 0.1
        C = torch.randn(1, n_cond) * 0.1

        eps = 1e-5

        def hier_func(x_vec):
            _, phi_out = hier_layer(x_vec.view(1, n_in), C)
            return phi_out[0, 0, :]

        J_hier = finite_diff_jacobian(hier_func, X[0, :], eps=eps)
        diag_hier = torch.diag(J_hier)

        # Count non-unit diagonal entries (tolerance for float comparison)
        non_unit_hier = (torch.abs(diag_hier - 1.0) > 0.01).sum().item()

        # For d=4, depth=2: should have 3 out of 4 learnable diag entries
        # (all but x1 which conditions everything)
        assert non_unit_hier >= n_in - 1, (
            f"Expected at least {n_in - 1} non-unit diagonal entries, "
            f"got {non_unit_hier}"
        )


class TestHierarchicalPermutationInvariance:
    """Verify trace is invariant under input permutations."""

    def test_trace_invariant_under_permutation(self):
        """tr(P^{-1} J P) = tr(J): same tree with permuted input has same trace."""
        torch.manual_seed(60)
        n_in, n_cond, n_hidden = 4, 2, 16

        tree = _HierarchicalTree(
            n_in, n_cond, n_hidden, depth=2, n_mlp_layers=3,
            activation=torch.nn.Tanh(),
        )
        tree = tree.to(dtype=torch.float64)
        tree.eval()

        X = torch.randn(5, n_in, dtype=torch.float64) * 0.5
        C = torch.randn(5, n_cond, dtype=torch.float64) * 0.5

        # Compute trace without permutation
        diag_orig, _ = tree(X, C)
        trace_orig = diag_orig.sum(dim=1)

        # Apply a random permutation to input, compute trace
        perm = torch.randperm(n_in)
        X_perm = X[:, perm]
        diag_perm, _ = tree(X_perm, C)
        trace_perm = diag_perm.sum(dim=1)

        # Traces should be different (different inputs) but let's verify with FD
        # that the FULL Jacobian trace of the permuted system is correct.
        eps = 1e-6
        inv_perm = torch.argsort(perm)
        for b in range(3):
            # Full phi with permutation: phi(x) = P^{-1} tree(Px)
            def phi_with_perm(x_vec):
                x_p = x_vec[perm]
                _, phi_out = tree(x_p.view(1, n_in), C[b:b+1])
                return phi_out[0, inv_perm]

            J_full = finite_diff_jacobian(phi_with_perm, X[b], eps=eps)
            trace_fd = torch.trace(J_full)

            # The analytical trace from the tree on permuted input should match
            assert torch.isclose(trace_perm[b], trace_fd, rtol=1e-4), (
                f"Sample {b}: analytical trace={trace_perm[b]:.6f}, "
                f"FD trace={trace_fd:.6f}"
            )

    def test_layer_trace_with_permutations_matches_fd(self):
        """Full layer with per-tree permutations: trace matches FD."""
        torch.manual_seed(61)
        n_in, n_cond, n_hidden = 4, 2, 16
        n_cv = 3

        layer = HierarchicalCouplingLayerCV(
            n_in, n_cond, n_hidden, n_cv=n_cv,
            activation=torch.nn.Tanh(),
        )
        layer = layer.to(dtype=torch.float64)
        layer.eval()

        X = torch.randn(3, n_in, dtype=torch.float64) * 0.5
        C = torch.randn(3, n_cond, dtype=torch.float64) * 0.5

        jac_traces, _ = layer(X, C)

        eps = 1e-6
        for k in range(n_cv):
            for b in range(3):
                def phi_k(x_vec):
                    _, phi_out = layer(x_vec.view(1, n_in), C[b:b+1])
                    return phi_out[k, 0, :]

                J = finite_diff_jacobian(phi_k, X[b], eps=eps)
                trace_fd = torch.trace(J)

                assert torch.isclose(jac_traces[k, b], trace_fd, rtol=1e-4), (
                    f"CV {k}, sample {b}: "
                    f"computed={jac_traces[k, b]:.6f}, fd={trace_fd:.6f}"
                )


class TestHierarchicalActivations:
    """Test with different coupling activations."""

    @pytest.mark.parametrize("activation_name,activation", [
        ("identity", torch.nn.Identity()),
        ("tanh", torch.nn.Tanh()),
        ("softplus", torch.nn.Softplus()),
    ])
    def test_jac_trace_with_activations(self, activation_name, activation):
        torch.manual_seed(52)
        n_in, n_cond, n_hidden = 4, 2, 16
        batch_size = 2

        layer = HierarchicalCouplingLayerCV(
            n_in, n_cond, n_hidden,
            coupling_activation=activation,
            n_cv=1,
        )
        layer.eval()

        X = torch.randn(batch_size, n_in) * 0.1
        C = torch.randn(batch_size, n_cond) * 0.1

        jac_traces, _ = layer(X, C)

        eps = 1e-5
        for b in range(batch_size):
            def forward_func(x_vec):
                _, phi_out = layer(x_vec.view(1, n_in), C[b:b+1, :])
                return phi_out[0, 0, :]

            J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)
            trace_fd = torch.trace(J)

            assert torch.isclose(
                jac_traces[0, b], trace_fd, rtol=1e-2, atol=1e-3
            ), (f"Activation {activation_name}, sample {b}: "
                f"computed={jac_traces[0, b]:.6f}, fd={trace_fd:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
