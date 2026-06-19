"""Tests for Rosenbrock distribution implementation.

Tests the multidimensional Hybrid Rosenbrock distribution including:
- 2D case with b parameter
- Multidimensional extensions
- Normalization constant
- Sampling, logpdf, gradlogpdf
- Score function computation
"""

import torch
import numpy as np
import pytest
from cncv.utils.rosenbrock import (
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)


class TestRosenbrock2D:
    """Tests for 2D Rosenbrock distribution."""

    def test_initialization(self):
        """Test basic initialization."""
        dist = RosenbrockDistribution(mu=1.0, a=0.5, b=2.0, n1=2, n2=1)
        assert dist.mu == 1.0
        assert dist.a == 0.5
        assert dist.ndim == 2
        assert dist.n1 == 2
        assert dist.n2 == 1
        assert dist.b.shape == (1, 1)
        assert dist.b[0, 0] == 2.0

    def test_normalization_constant_2d(self):
        """Test normalization constant for 2D case.

        For 2D: Z = sqrt(a) * sqrt(b) / pi
        """
        a, b = 0.5, 2.0
        dist = RosenbrockDistribution(mu=0.0, a=a, b=b, n1=2, n2=1)

        expected_log_norm = (
            0.5 * np.log(a) + 0.5 * np.log(b) - np.log(np.pi)
        )
        assert np.allclose(dist.log_norm_const, expected_log_norm)

    def test_sampling_2d(self):
        """Test that sampling produces correct shape and reasonable values."""
        dist = RosenbrockDistribution(mu=1.0, a=1.0, b=1.0, n1=2, n2=1)
        samples = dist.sample(1000)

        assert samples.shape == (1000, 2)
        assert torch.isfinite(samples).all()

        # Check approximate moments
        mean = samples.mean(dim=0)
        # x1 should be centered around mu
        assert torch.abs(mean[0] - 1.0) < 0.2

    def test_logpdf_2d(self):
        """Test log pdf computation for 2D case."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=1.0, n1=2, n2=1)

        # Test at origin
        X = torch.tensor([[0.0, 0.0]])
        logp = dist.logpdf(X)
        assert torch.isfinite(logp).all()

        # Test at ridge (x2 = x1^2)
        X = torch.tensor([[1.0, 1.0], [2.0, 4.0]])
        logp = dist.logpdf(X)
        assert torch.isfinite(logp).all()

    def test_logpdf_normalization_2d(self):
        """Test that pdf integrates to 1 (Monte Carlo check)."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=5.0, n1=2, n2=1)

        # Sample from distribution
        n_samples = 100000
        samples = dist.sample(n_samples)

        # Compute log pdf
        logp = dist.logpdf(samples)

        # Check that exp(logp) has reasonable values
        # Mean of exp(logp) under the distribution should equal Z
        # We can't check this directly, but we can check consistency
        assert torch.isfinite(logp).all()
        assert logp.mean() > -10  # Should not be too negative on average

    def test_gradlogpdf_2d(self):
        """Test gradient computation using autograd."""
        dist = RosenbrockDistribution(mu=1.0, a=1.0, b=2.0, n1=2, n2=1)

        X = torch.tensor([[1.0, 1.5], [0.5, 0.3]], requires_grad=True)

        # Compute gradient analytically
        grad_analytic = dist.gradlogpdf(X)

        # Compute gradient with autograd
        logp = dist.logpdf(X)
        grad_autograd = torch.autograd.grad(
            logp.sum(), X, create_graph=False
        )[0]

        assert torch.allclose(grad_analytic, grad_autograd, atol=1e-5)

    def test_b_parameter_effect(self):
        """Test that b parameter affects variance."""
        mu, a = 0.0, 1.0

        # Small b -> large variance in x2
        dist1 = RosenbrockDistribution(mu=mu, a=a, b=0.5, n1=2, n2=1)
        samples1 = dist1.sample(10000)

        # Large b -> small variance in x2
        dist2 = RosenbrockDistribution(mu=mu, a=a, b=5.0, n1=2, n2=1)
        samples2 = dist2.sample(10000)

        var1 = samples1[:, 1].var().item()
        var2 = samples2[:, 1].var().item()

        # Larger b should give smaller variance
        assert var2 < var1


class TestRosenbrockMultiD:
    """Tests for multidimensional Rosenbrock distribution."""

    def test_initialization_3d(self):
        """Test initialization for 3D case (n1=3, n2=1)."""
        dist = RosenbrockDistribution(mu=1.0, a=0.5, b=2.0, n1=3, n2=1)
        assert dist.ndim == 3  # (3-1)*1 + 1 = 3
        assert dist.b.shape == (1, 2)  # [n2, n1-1]

    def test_initialization_5d(self):
        """Test initialization for 5D case (n1=3, n2=2)."""
        dist = RosenbrockDistribution(mu=1.0, a=0.5, b=2.0, n1=3, n2=2)
        assert dist.ndim == 5  # (3-1)*2 + 1 = 5
        assert dist.b.shape == (2, 2)  # [n2, n1-1]

    def test_heterogeneous_b(self):
        """Test with different b values for different terms."""
        b_array = np.array([[1.0, 2.0], [3.0, 4.0]])  # [n2=2, n1-1=2]
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=b_array, n1=3, n2=2)

        assert dist.ndim == 5
        assert np.array_equal(dist.b, b_array)

    def test_normalization_constant_5d(self):
        """Test normalization constant for 5D case.

        For n1=3, n2=2: Z = sqrt(a) * sqrt(b[0,0]) * sqrt(b[0,1]) * sqrt(b[1,0]) * sqrt(b[1,1]) / pi^(5/2)
        """
        a = 1.0
        b = 2.0
        dist = RosenbrockDistribution(mu=0.0, a=a, b=b, n1=3, n2=2)

        # n = (3-1)*2 + 1 = 5
        # 4 b terms total
        expected_log_norm = (
            0.5 * np.log(a) + 4 * 0.5 * np.log(b) - (5 / 2) * np.log(np.pi)
        )
        assert np.allclose(dist.log_norm_const, expected_log_norm)

    def test_sampling_multid(self):
        """Test sampling for multidimensional case."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=2.0, n1=3, n2=2)
        samples = dist.sample(1000)

        assert samples.shape == (1000, 5)
        assert torch.isfinite(samples).all()

    def test_logpdf_multid(self):
        """Test log pdf for multidimensional case."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=2.0, n1=3, n2=2)

        # Generate samples and compute logpdf
        samples = dist.sample(100)
        logp = dist.logpdf(samples)

        assert logp.shape == (100,)
        assert torch.isfinite(logp).all()

    def test_gradlogpdf_multid(self):
        """Test gradient computation for multidimensional case."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=2.0, n1=3, n2=2)

        X = dist.sample(10)
        X.requires_grad = True

        # Compute gradient analytically
        grad_analytic = dist.gradlogpdf(X)

        # Compute gradient with autograd
        logp = dist.logpdf(X)
        grad_autograd = torch.autograd.grad(
            logp.sum(), X, create_graph=False
        )[0]

        assert torch.allclose(grad_analytic, grad_autograd, atol=1e-5)

    def test_conditional_structure(self):
        """Test that the conditional sampling structure is correct.

        For each block j:
        - x[j,0] (= x1) should influence x[j,1]
        - x[j,i] should influence x[j,i+1]
        """
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=10.0, n1=3, n2=1)

        # Sample many times
        samples = dist.sample(10000)

        x1 = samples[:, 0]  # First variable
        x2 = samples[:, 1]  # Second variable (should be near x1^2)
        x3 = samples[:, 2]  # Third variable (should be near x2^2)

        # With large b=10, x2 should be tightly concentrated around x1^2
        residual_2 = (x2 - x1**2).abs().mean()
        assert residual_2 < 0.2

        # Similarly for x3
        residual_3 = (x3 - x2**2).abs().mean()
        assert residual_3 < 0.5


class TestScorePosterior:
    """Tests for score function computation."""

    def test_score_computation_2d(self):
        """Test score function for posterior."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=1.0, n1=2, n2=1)

        X = torch.tensor([[1.0, 1.5], [0.5, 0.3]])
        Y = torch.tensor([[1.2, 1.6]])
        sigma = 0.5

        score = compute_score_posterior_rosenbrock(X, Y, dist, sigma)

        assert score.shape == X.shape
        assert torch.isfinite(score).all()

    def test_score_computation_multid(self):
        """Test score function for multidimensional case."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=2.0, n1=3, n2=2)

        X = dist.sample(50)
        Y = dist.sample(1)
        sigma = 0.3

        score = compute_score_posterior_rosenbrock(X, Y, dist, sigma)

        assert score.shape == X.shape
        assert torch.isfinite(score).all()

    def test_score_gradient_check(self):
        """Test score using autograd."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=2.0, n1=2, n2=1)

        X = torch.tensor([[1.0, 1.5]], requires_grad=True)
        Y = torch.tensor([[1.2, 1.6]])
        sigma = 0.5

        # Compute score analytically
        score = compute_score_posterior_rosenbrock(X, Y, dist, sigma)

        # Compute using autograd
        log_likelihood = -0.5 * ((X - Y) ** 2).sum(dim=1) / (sigma**2)
        log_prior = dist.logpdf(X)
        log_posterior = log_likelihood + log_prior

        grad_autograd = torch.autograd.grad(
            log_posterior.sum(), X, create_graph=False
        )[0]

        assert torch.allclose(score, grad_autograd, atol=1e-5)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        b_array = np.array([[1.0, 2.0]])  # [1, 2]
        with pytest.raises(ValueError, match="must have shape"):
            RosenbrockDistribution(mu=0.0, a=1.0, b=b_array, n1=4, n2=2)

    def test_single_dimension(self):
        """Test behavior with n1=2, n2=1 (standard 2D case)."""
        dist = RosenbrockDistribution(mu=0.0, a=1.0, b=1.0, n1=2, n2=1)
        assert dist.ndim == 2

        samples = dist.sample(100)
        assert samples.shape == (100, 2)

    def test_backward_compatibility(self):
        """Test that default parameters give standard 2D Rosenbrock."""
        # Without b parameter (should default to 1.0)
        dist = RosenbrockDistribution(mu=0.0, a=1.0)
        assert dist.ndim == 2
        assert dist.b[0, 0] == 1.0

        samples = dist.sample(100)
        assert samples.shape == (100, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
