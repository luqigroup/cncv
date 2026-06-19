"""Tests for distribution utilities (Gaussian and Rosenbrock)."""

import pytest
import torch
import numpy as np

from cncv.utils import (
    compute_score_posterior_gaussian,
    compute_exact_posterior,
    RosenbrockDistribution,
    compute_score_posterior_rosenbrock,
)
from tests.utils import finite_diff_grad


class TestGaussianPosterior:
    """Tests for Gaussian posterior utilities."""

    def test_score_computation_single_sample(self):
        """Test score function matches analytical formula for single sample."""
        # Setup
        n_dim = 2
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        sigma = 0.3

        X = torch.tensor([1.0, 2.0])
        Y = torch.tensor([0.5, 1.5])

        # Compute score
        score = compute_score_posterior_gaussian(X, Y, mu_prior, Sigma_prior_inv, sigma)

        # Hand-computed expected score
        # grad_likelihood = -(X - Y) / sigma^2 = -([1,2] - [0.5,1.5]) / 0.09 = -[0.5,0.5] / 0.09
        grad_likelihood_expected = -(X - Y) / (sigma ** 2)

        # grad_prior = -Sigma_inv @ (X - mu) = -Sigma_inv @ [1, 2]
        grad_prior_expected = -Sigma_prior_inv @ (X - mu_prior)

        expected_score = grad_likelihood_expected + grad_prior_expected

        assert torch.allclose(score, expected_score, rtol=1e-5)

    def test_score_computation_batch(self):
        """Test score function for batch of samples."""
        # Setup
        n_dim = 2
        batch_size = 10
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        sigma = 0.3

        X = torch.randn(batch_size, n_dim)
        Y = torch.randn(batch_size, n_dim)

        # Compute score
        score = compute_score_posterior_gaussian(X, Y, mu_prior, Sigma_prior_inv, sigma)

        # Check shape
        assert score.shape == (batch_size, n_dim)

        # Verify each sample matches single-sample computation
        for i in range(batch_size):
            score_i = compute_score_posterior_gaussian(
                X[i], Y[i], mu_prior, Sigma_prior_inv, sigma
            )
            assert torch.allclose(score[i], score_i, rtol=1e-5)

    def test_score_is_gradient_of_logpdf(self):
        """Test that score is the gradient of log posterior."""
        # Setup
        n_dim = 2
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        sigma = 0.3

        Y = torch.tensor([0.5, 1.5])  # Fixed observation

        # Define log posterior as a function of X
        def log_posterior(X):
            """Compute log p(x|y) = log p(y|x) + log p(x)."""
            # Log likelihood: log N(y|x, σ²I)
            log_likelihood = -0.5 * ((Y - X) ** 2).sum() / (sigma ** 2)
            log_likelihood -= n_dim * np.log(sigma) + n_dim * 0.5 * np.log(2 * np.pi)

            # Log prior: log N(x|μ, Σ)
            diff = X - mu_prior
            log_prior = -0.5 * (diff @ Sigma_prior_inv @ diff)
            log_prior -= 0.5 * torch.logdet(Sigma_prior) + n_dim * 0.5 * np.log(2 * np.pi)

            return log_likelihood + log_prior

        # Test point (use float32)
        X = torch.tensor([1.0, 2.0], dtype=torch.float32)

        # Compute score analytically
        score_analytical = compute_score_posterior_gaussian(
            X, Y, mu_prior, Sigma_prior_inv, sigma
        )

        # Compute score via finite differences
        score_fd = finite_diff_grad(log_posterior, X, eps=1e-5)

        # Looser tolerance for finite differences
        # Values like [-5.56, -7.56] vs [-5.63, -7.58] need ~2% tolerance
        assert torch.allclose(score_analytical, score_fd, rtol=2e-2, atol=1e-2)

    def test_exact_posterior(self):
        """Test exact posterior computation."""
        # Setup
        n_dim = 2
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        sigma = 0.3

        Y = torch.tensor([0.5, 1.5])

        # Compute exact posterior
        mu_post, Sigma_post = compute_exact_posterior(Y, mu_prior, Sigma_prior, sigma)

        # Check shapes
        assert mu_post.shape == (n_dim,)
        assert Sigma_post.shape == (n_dim, n_dim)

        # Check posterior covariance is symmetric and positive definite
        assert torch.allclose(Sigma_post, Sigma_post.T, rtol=1e-5)
        eigvals = torch.linalg.eigvalsh(Sigma_post)
        assert (eigvals > 0).all()

        # Verify formula manually
        Sigma_prior_inv = torch.linalg.inv(Sigma_prior)
        I = torch.eye(n_dim)

        Sigma_post_expected = torch.linalg.inv(Sigma_prior_inv + (1 / sigma**2) * I)
        mu_post_expected = Sigma_post_expected @ (
            Sigma_prior_inv @ mu_prior + (1 / sigma**2) * Y
        )

        assert torch.allclose(mu_post, mu_post_expected, rtol=1e-5)
        assert torch.allclose(Sigma_post, Sigma_post_expected, rtol=1e-5)

    def test_posterior_reduces_variance(self):
        """Test that posterior variance is smaller than prior variance."""
        # Setup
        n_dim = 2
        mu_prior = torch.zeros(n_dim)
        Sigma_prior = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        sigma = 0.3

        Y = torch.tensor([0.5, 1.5])

        # Compute exact posterior
        mu_post, Sigma_post = compute_exact_posterior(Y, mu_prior, Sigma_prior, sigma)

        # Check that diagonal elements (variances) decreased
        for i in range(n_dim):
            assert Sigma_post[i, i] < Sigma_prior[i, i]


class TestRosenbrockDistribution:
    """Tests for Rosenbrock distribution."""

    def test_logpdf(self):
        """Test logpdf against known values."""
        # Setup
        rosenbrock = RosenbrockDistribution(mu=0.0, a=0.5)

        # Test points (use float32 to match implementation)
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], dtype=torch.float32)

        # Compute logpdf
        logpdf_vals = rosenbrock.logpdf(X)

        # Check shape
        assert logpdf_vals.shape == (3,)

        # Manual computation for first point (0, 0)
        # log p = -a(x1 - μ)^2 - (x2 - x1^2)^2 + log(sqrt(a)/π)
        # = -0.5(0 - 0)^2 - (0 - 0)^2 + log(sqrt(0.5)/π)
        # = 0 + log(sqrt(0.5)/π) = log(sqrt(0.5)) - log(π)
        expected_0 = np.log(np.sqrt(0.5) / np.pi)
        assert torch.isclose(
            logpdf_vals[0], torch.tensor(expected_0, dtype=torch.float32), rtol=1e-5
        )

        # For point (0.5, 0.25), x2 = x1^2, so second term is zero
        # log p = -0.5(0.5 - 0)^2 - 0 + log(sqrt(0.5)/π)
        # = -0.5 * 0.25 + log(sqrt(0.5)/π) = -0.125 + log(sqrt(0.5)/π)
        expected_2 = -0.125 + np.log(np.sqrt(0.5) / np.pi)
        assert torch.isclose(
            logpdf_vals[2], torch.tensor(expected_2, dtype=torch.float32), rtol=1e-5
        )

    def test_gradlogpdf_finite_diff(self):
        """Test gradlogpdf using finite differences."""
        # Setup
        rosenbrock = RosenbrockDistribution(mu=0.0, a=0.5)

        # Test point (use float32)
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Analytical gradient
        grad_analytical = rosenbrock.gradlogpdf(X)

        # Finite difference gradient
        def logpdf_func(x):
            return rosenbrock.logpdf(x.unsqueeze(0)).squeeze()

        grad_fd = finite_diff_grad(logpdf_func, X[0], eps=1e-5)

        # Looser tolerance for finite differences
        assert torch.allclose(
            grad_analytical[0], grad_fd, rtol=1e-2, atol=1e-3
        )

    def test_gradlogpdf_formula(self):
        """Test gradlogpdf matches manual computation."""
        # Setup
        rosenbrock = RosenbrockDistribution(mu=0.0, a=0.5)

        # Test point (use float32)
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        x1, x2 = 1.0, 2.0

        # Analytical gradient
        grad = rosenbrock.gradlogpdf(X)

        # Manual computation
        # grad_x1 = -2a(x1 - μ) + 4x1(x2 - x1^2)
        # = -2*0.5*(1 - 0) + 4*1*(2 - 1) = -1 + 4 = 3
        expected_grad_x1 = -2 * 0.5 * (x1 - 0.0) + 4 * x1 * (x2 - x1**2)

        # grad_x2 = -2(x2 - x1^2) = -2(2 - 1) = -2
        expected_grad_x2 = -2 * (x2 - x1**2)

        assert torch.isclose(grad[0, 0], torch.tensor(expected_grad_x1, dtype=torch.float32), rtol=1e-5)
        assert torch.isclose(grad[0, 1], torch.tensor(expected_grad_x2, dtype=torch.float32), rtol=1e-5)

    def test_sampling(self):
        """Test that samples have correct properties."""
        torch.manual_seed(60)
        rosenbrock = RosenbrockDistribution(mu=0.0, a=0.5)

        # Generate many samples
        n_samples = 10000
        samples = rosenbrock.sample(n_samples)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Check empirical mean of x1 should be close to μ
        x1_mean = samples[:, 0].mean()
        assert torch.isclose(x1_mean, torch.tensor(0.0), rtol=0.1, atol=0.1)

        # Check empirical variance of x1 should be close to 1/(2a) = 1
        x1_var = samples[:, 0].var()
        expected_var_x1 = 1.0 / (2 * rosenbrock.a)  # = 1.0
        assert torch.isclose(x1_var, torch.tensor(expected_var_x1), rtol=0.2, atol=0.2)

        # For x2, conditional mean is E[x2|x1] = x1^2
        # Marginal mean E[x2] = E[x1^2] = Var[x1] + E[x1]^2 = 1 + 0 = 1
        x2_mean = samples[:, 1].mean()
        assert torch.isclose(x2_mean, torch.tensor(1.0), rtol=0.2, atol=0.2)

    def test_score_posterior_rosenbrock(self):
        """Test full posterior score = likelihood + prior."""
        # Setup
        rosenbrock = RosenbrockDistribution(mu=0.0, a=0.5)
        sigma_obs = 0.3

        # Use float32
        X = torch.tensor([[1.0, 2.0], [0.5, 0.25]], dtype=torch.float32)
        Y = torch.tensor([[0.9, 1.8]], dtype=torch.float32)

        # Compute posterior score
        score = compute_score_posterior_rosenbrock(X, Y, rosenbrock, sigma_obs)

        # Check shape
        assert score.shape == (2, 2)

        # Manual computation for first sample
        # grad_likelihood = -(X - Y) / sigma^2
        grad_likelihood_expected = -(X - Y) / (sigma_obs**2)

        # grad_prior = rosenbrock.gradlogpdf(X)
        grad_prior_expected = rosenbrock.gradlogpdf(X)

        expected_score = grad_likelihood_expected + grad_prior_expected

        assert torch.allclose(score, expected_score, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
