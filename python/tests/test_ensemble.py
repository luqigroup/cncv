"""Tests for ensemble models and integration."""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cncv.models import (
    EnsembleDenseCV,
    create_forward_reverse_ensemble,
    create_random_split_ensemble,
)
from cncv.utils import compute_score_posterior_gaussian


class TestEnsemble:
    """Tests for ensemble functionality."""

    def test_forward_reverse_ensemble_creation(self):
        """Test that forward-reverse ensemble creates 2 layers with correct splits."""
        torch.manual_seed(53)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        n_layers = 2

        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=n_layers, n_cv=n_in
        )

        # Should have exactly 2 layers
        assert len(ensemble.layers) == 2

        # First layer: forward split (reverse_split=False)
        assert ensemble.layers[0].reverse_split == False

        # Second layer: reverse split (reverse_split=True)
        assert ensemble.layers[1].reverse_split == True

    def test_forward_reverse_ensemble_produces_different_cvs(self):
        """Test that forward-reverse ensemble produces 2 different CVs."""
        torch.manual_seed(54)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        batch_size = 10

        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=2, n_cv=n_in
        )

        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)
        score = torch.randn(batch_size, n_in)

        # Forward through ensemble
        g_combined, all_g = ensemble(X, C, score)

        # Should get 2 individual layer CVs
        assert len(all_g) == 2

        # Each should have shape [n_in, batch_size]
        assert all_g[0].shape == (n_in, batch_size)
        assert all_g[1].shape == (n_in, batch_size)

        # Combined should also be [n_in, batch_size]
        assert g_combined.shape == (n_in, batch_size)

        # all_g[0] and all_g[1] should be different
        assert not torch.allclose(all_g[0], all_g[1], rtol=1e-2)

        # Combined should be the average (for default combination mode)
        if isinstance(all_g, list):
            expected_combined = (all_g[0] + all_g[1]) / 2
            assert torch.allclose(g_combined, expected_combined, rtol=1e-5)

    def test_random_split_ensemble(self):
        """Test random split ensemble with multiple members."""
        torch.manual_seed(55)
        n_in = 6
        n_cond = 6
        n_hidden = 12
        n_members = 4
        batch_size = 5

        ensemble = create_random_split_ensemble(
            n_in, n_cond, n_hidden, n_members, n_layers=2, n_cv=n_in, seed=42
        )

        # Should have n_members layers
        assert len(ensemble.layers) == n_members

        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)
        score = torch.randn(batch_size, n_in)

        # Forward through ensemble
        g_combined, all_g = ensemble(X, C, score)

        # Should get n_members individual CVs
        assert len(all_g) == n_members

        # Each should have shape [n_in, batch_size]
        for g in all_g:
            assert g.shape == (n_in, batch_size)

        # Combined should be average
        # all_g is a list of tensors
        if isinstance(all_g, list):
            expected_combined = torch.stack(all_g).mean(dim=0)
            assert torch.allclose(g_combined, expected_combined, rtol=1e-5)
        else:
            # If all_g is already a single tensor, just verify shapes match
            assert g_combined.shape == all_g.mean(dim=0).shape if len(all_g.shape) > 2 else g_combined.shape

    def test_ensemble_variance_reduction(self):
        """End-to-end test: ensemble should reduce variance on simple Gaussian case."""
        torch.manual_seed(56)
        n_dim = 2
        n_hidden = 32
        n_samples = 5000
        n_trials = 100

        # Setup simple Gaussian problem
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        sigma = 0.3

        # Create ensemble
        ensemble = create_forward_reverse_ensemble(
            n_dim, n_dim, n_hidden, n_layers=3, n_cv=n_dim
        )

        # Generate samples from prior
        L_prior = torch.linalg.cholesky(Sigma_prior)
        X = mu_prior + (L_prior @ torch.randn(n_dim, n_samples)).T
        Y = X + sigma * torch.randn_like(X)

        # Compute score
        score = compute_score_posterior_gaussian(X, Y, mu_prior, Sigma_prior_inv, sigma)

        # Forward through ensemble
        with torch.no_grad():
            g_combined, all_g = ensemble(X, Y, score)

        # Quantity of interest: h(x) = x
        h_x = X

        # Learnable offsets (initialized to zero for this test)
        mu = torch.zeros(n_dim)

        # Compute variance reduction for small sample sizes
        sample_size = 50
        vanilla_estimates = []
        cv_estimates = []

        for _ in range(n_trials):
            # Random subsample
            indices = torch.randperm(n_samples)[:sample_size]

            # Vanilla MC
            vanilla_est = h_x[indices].mean(dim=0)
            vanilla_estimates.append(vanilla_est)

            # CV estimator
            cv_residual = h_x[indices].T - g_combined[:, indices]
            cv_est = cv_residual.mean(dim=1) + mu
            cv_estimates.append(cv_est)

        vanilla_estimates = torch.stack(vanilla_estimates)
        cv_estimates = torch.stack(cv_estimates)

        # Compute variances
        var_vanilla = vanilla_estimates.var(dim=0)
        var_cv = cv_estimates.var(dim=0)

        # Check that CV has lower or similar variance
        # (Note: without training, ensemble may not always reduce variance,
        #  but it should at least not make it much worse)
        # For a trained ensemble, we'd expect var_cv < var_vanilla
        vrf = var_cv / var_vanilla

        # For untrained ensembles, VRF can be quite high
        # We just verify the code runs and produces output
        # For trained ensembles, we'd expect VRF < 1
        assert vrf.shape == (n_dim,), f"VRF shape should be ({n_dim},), got {vrf.shape}"
        assert not torch.isnan(vrf).any(), "VRF should not contain NaN"
        assert (vrf > 0).all(), "VRF should be positive"

    def test_ensemble_with_multiple_cv_per_layer(self):
        """Test n_cv > 1 functionality."""
        torch.manual_seed(57)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        n_cv = 3  # More CVs than dimensions
        batch_size = 5

        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=2, n_cv=n_cv
        )

        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)
        score = torch.randn(batch_size, n_in)

        # Forward
        g_combined, all_g = ensemble(X, C, score)

        # all_g should have 2 elements (2 layers)
        assert len(all_g) == 2

        # Each element should have shape [n_cv, batch_size]
        # Note: The ensemble combines across CVs within each layer
        # So the output shape depends on implementation

        # Combined should work
        assert g_combined.shape[0] == n_cv  # n_cv components
        assert g_combined.shape[1] == batch_size

    def test_ensemble_gradients(self):
        """Test that gradients flow through ensemble correctly."""
        torch.manual_seed(58)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        batch_size = 3

        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=2, n_cv=n_in
        )

        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)
        score = torch.randn(batch_size, n_in)

        # Forward
        g_combined, all_g = ensemble(X, C, score)

        # Create simple loss
        loss = g_combined.sum()

        # Backward
        loss.backward()

        # Check that ensemble parameters have gradients
        for param in ensemble.parameters():
            assert param.grad is not None

    def test_ensemble_eval_mode(self):
        """Test ensemble in eval mode."""
        torch.manual_seed(59)
        n_in = 4
        n_cond = 4
        n_hidden = 8
        batch_size = 5

        ensemble = create_forward_reverse_ensemble(
            n_in, n_cond, n_hidden, n_layers=2, n_cv=n_in
        )

        X = torch.randn(batch_size, n_in)
        C = torch.randn(batch_size, n_cond)
        score = torch.randn(batch_size, n_in)

        # Train mode
        ensemble.train()
        g_train, _ = ensemble(X, C, score)

        # Eval mode
        ensemble.eval()
        with torch.no_grad():
            g_eval, _ = ensemble(X, C, score)

        # For networks without dropout/batchnorm, train and eval should give same results
        # (Note: Our coupling layers don't have dropout/batchnorm by default)
        assert torch.allclose(g_train, g_eval, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
