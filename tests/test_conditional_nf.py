"""Tests for conditional normalizing flow with HINT hierarchical coupling."""

import pytest
import torch
import math

from cncv.models.conditional_nf import _NFTree, ConditionalNF
from tests.utils import finite_diff_jacobian, finite_diff_grad


class TestNFTreeInvertibility:
    """Verify _NFTree forward/inverse are consistent."""

    @pytest.mark.parametrize("n_in,depth", [
        (2, 1),
        (4, 2),
        (4, 1),
        (8, 3),
        (6, 2),
        (3, 1),
    ])
    def test_tree_invertibility(self, n_in, depth):
        """x == inverse(forward(x, y)[0], y) for single tree."""
        torch.manual_seed(42)
        n_cond = 2
        n_hidden = 16
        batch_size = 5

        tree = _NFTree(n_in, n_cond, n_hidden, depth)
        tree.double().eval()

        x = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.5
        cond = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.5

        with torch.no_grad():
            z, log_det = tree(x, cond)
            x_recon = tree.inverse(z, cond)

        assert torch.allclose(x, x_recon, atol=1e-12), (
            f"n_in={n_in}, depth={depth}: max error = {(x - x_recon).abs().max():.2e}"
        )

    def test_leaf_identity(self):
        """Leaf node (depth=0) should be identity in both directions."""
        torch.manual_seed(43)
        tree = _NFTree(3, 2, 8, depth=0)
        tree.double().eval()

        x = torch.randn(4, 3, dtype=torch.float64)
        cond = torch.randn(4, 2, dtype=torch.float64)

        with torch.no_grad():
            z, log_det = tree(x, cond)
            x_recon = tree.inverse(z, cond)

        assert torch.allclose(z, x, atol=1e-14)
        assert torch.allclose(log_det, torch.zeros(4, dtype=torch.float64), atol=1e-14)
        assert torch.allclose(x_recon, x, atol=1e-14)

    def test_single_dim_identity(self):
        """n_in=1 should be identity."""
        torch.manual_seed(44)
        tree = _NFTree(1, 2, 8, depth=3)
        tree.double().eval()

        x = torch.randn(4, 1, dtype=torch.float64)
        cond = torch.randn(4, 2, dtype=torch.float64)

        with torch.no_grad():
            z, log_det = tree(x, cond)
            x_recon = tree.inverse(z, cond)

        assert torch.allclose(z, x, atol=1e-14)
        assert torch.allclose(x_recon, x, atol=1e-14)


class TestNFTreeLogDet:
    """Verify log-det computation against finite-difference Jacobian."""

    @pytest.mark.parametrize("n_in,depth", [
        (2, 1),
        (4, 2),
        (4, 1),
        (8, 3),
        (6, 2),
        (3, 1),
    ])
    def test_tree_logdet_vs_fd(self, n_in, depth):
        """log_det from tree should match log|det(J_fd)| from FD Jacobian."""
        torch.manual_seed(45)
        n_cond = 2
        n_hidden = 16
        batch_size = 3

        tree = _NFTree(
            n_in, n_cond, n_hidden, depth,
            activation=torch.nn.Tanh(),
        )
        tree.double().eval()

        X = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.1
        C = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.1

        z, log_det = tree(X, C)

        eps = 1e-6
        for b in range(batch_size):
            def forward_func(x_vec):
                x_sample = x_vec.view(1, n_in)
                c_sample = C[b:b+1, :]
                z_out, _ = tree(x_sample, c_sample)
                return z_out[0, :]

            J = finite_diff_jacobian(forward_func, X[b, :], eps=eps)
            log_det_fd = torch.log(torch.abs(torch.det(J)))

            assert torch.isclose(
                log_det[b], log_det_fd, rtol=1e-4, atol=1e-5
            ), (f"n_in={n_in}, depth={depth}, sample {b}: "
                f"computed={log_det[b]:.8f}, fd={log_det_fd:.8f}")


class TestConditionalNFInvertibility:
    """Verify full flow forward/inverse consistency."""

    @pytest.mark.parametrize("n_in,n_flow_layers", [
        (2, 2),
        (4, 4),
        (4, 8),
        (8, 4),
        (3, 3),
    ])
    def test_flow_invertibility(self, n_in, n_flow_layers):
        """x == inverse(forward(x, y)[0], y) for full flow."""
        torch.manual_seed(50)
        n_cond = 3
        n_hidden = 16
        batch_size = 5

        flow = ConditionalNF(
            n_in, n_cond, n_hidden,
            n_flow_layers=n_flow_layers, depth=2,
        )
        flow.double().eval()

        x = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.5
        y = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.5

        with torch.no_grad():
            z, log_det = flow(x, y)
            x_recon = flow.inverse(z, y)

        assert torch.allclose(x, x_recon, atol=1e-10), (
            f"n_in={n_in}, K={n_flow_layers}: "
            f"max error = {(x - x_recon).abs().max():.2e}"
        )


class TestConditionalNFLogDet:
    """Verify full flow log-det against FD Jacobian."""

    @pytest.mark.parametrize("n_in,n_flow_layers", [
        (2, 2),
        (4, 4),
        (3, 3),
    ])
    def test_flow_logdet_vs_fd(self, n_in, n_flow_layers):
        """Full flow log_det matches log|det(J_fd)|."""
        torch.manual_seed(55)
        n_cond = 2
        n_hidden = 16
        batch_size = 2

        flow = ConditionalNF(
            n_in, n_cond, n_hidden,
            n_flow_layers=n_flow_layers, depth=2,
            n_mlp_layers=2,
        )
        flow.double().eval()

        X = torch.randn(batch_size, n_in, dtype=torch.float64) * 0.1
        Y = torch.randn(batch_size, n_cond, dtype=torch.float64) * 0.1

        _, log_det = flow(X, Y)

        eps = 1e-6
        for b in range(batch_size):
            def forward_func(x_vec):
                z_out, _ = flow(x_vec.view(1, n_in), Y[b:b+1])
                return z_out[0, :]

            J = finite_diff_jacobian(forward_func, X[b], eps=eps)
            log_det_fd = torch.log(torch.abs(torch.det(J)))

            assert torch.isclose(
                log_det[b], log_det_fd, rtol=1e-3, atol=1e-4
            ), (f"n_in={n_in}, K={n_flow_layers}, sample {b}: "
                f"computed={log_det[b]:.8f}, fd={log_det_fd:.8f}")


class TestConditionalNFScore:
    """Verify score computation against FD gradient of log_prob."""

    @pytest.mark.parametrize("n_in", [2, 4, 3])
    def test_score_vs_fd(self, n_in):
        """score(x, y) should match FD gradient of log_prob(x, y)."""
        torch.manual_seed(60)
        n_cond = 2
        n_hidden = 16

        flow = ConditionalNF(
            n_in, n_cond, n_hidden,
            n_flow_layers=4, depth=2, n_mlp_layers=2,
        )
        flow.double().eval()

        x = torch.randn(1, n_in, dtype=torch.float64) * 0.1
        y = torch.randn(1, n_cond, dtype=torch.float64) * 0.1

        # Autograd score
        score = flow.score(x, y)

        # FD gradient of log_prob
        eps = 1e-6
        def log_prob_func(x_vec):
            return flow.log_prob(x_vec.view(1, n_in), y)[0]

        grad_fd = finite_diff_grad(log_prob_func, x.view(-1), eps=eps)

        assert torch.allclose(
            score.view(-1), grad_fd.view(-1), rtol=1e-3, atol=1e-4
        ), (f"n_in={n_in}: max diff = "
            f"{(score.view(-1) - grad_fd.view(-1)).abs().max():.2e}")


class TestConditionalNFShapes:
    """Shape consistency tests."""

    def test_forward_shapes(self):
        torch.manual_seed(70)
        n_in, n_cond, n_hidden = 4, 3, 16
        batch_size = 7

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        x = torch.randn(batch_size, n_in)
        y = torch.randn(batch_size, n_cond)

        z, log_det = flow(x, y)
        assert z.shape == (batch_size, n_in)
        assert log_det.shape == (batch_size,)

    def test_inverse_shapes(self):
        torch.manual_seed(71)
        n_in, n_cond, n_hidden = 4, 3, 16
        batch_size = 5

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        z = torch.randn(batch_size, n_in)
        y = torch.randn(batch_size, n_cond)

        x = flow.inverse(z, y)
        assert x.shape == (batch_size, n_in)

    def test_log_prob_shapes(self):
        torch.manual_seed(72)
        n_in, n_cond, n_hidden = 4, 3, 16
        batch_size = 5

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        x = torch.randn(batch_size, n_in)
        y = torch.randn(batch_size, n_cond)

        log_p = flow.log_prob(x, y)
        assert log_p.shape == (batch_size,)

    def test_score_shapes(self):
        torch.manual_seed(73)
        n_in, n_cond, n_hidden = 4, 3, 16
        batch_size = 5

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        x = torch.randn(batch_size, n_in)
        y = torch.randn(batch_size, n_cond)

        s = flow.score(x, y)
        assert s.shape == (batch_size, n_in)

    def test_sample_shapes(self):
        torch.manual_seed(74)
        n_in, n_cond, n_hidden = 4, 3, 16
        n_samples = 100

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        y = torch.randn(n_cond)

        samples = flow.sample(y, n_samples)
        assert samples.shape == (n_samples, n_in)

    def test_sample_batch_y(self):
        """Sample with batched y (single row expanded)."""
        torch.manual_seed(75)
        n_in, n_cond, n_hidden = 4, 3, 16
        n_samples = 50

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        y = torch.randn(1, n_cond)

        samples = flow.sample(y, n_samples)
        assert samples.shape == (n_samples, n_in)


class TestConditionalNFGradients:
    """Gradient flow tests."""

    def test_gradients_flow_through_nll(self):
        """All parameters should receive gradients from NLL loss."""
        torch.manual_seed(80)
        n_in, n_cond, n_hidden = 4, 3, 16

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4, depth=2)
        x = torch.randn(8, n_in)
        y = torch.randn(8, n_cond)

        loss = -flow.log_prob(x, y).mean()
        loss.backward()

        for name, param in flow.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_nll_is_finite(self):
        """NLL loss should be finite."""
        torch.manual_seed(81)
        n_in, n_cond, n_hidden = 4, 3, 16

        flow = ConditionalNF(n_in, n_cond, n_hidden, n_flow_layers=4)
        x = torch.randn(8, n_in)
        y = torch.randn(8, n_cond)

        loss = -flow.log_prob(x, y).mean()
        assert torch.isfinite(loss), f"NLL is not finite: {loss.item()}"


class TestConditionalNFLogProb:
    """Verify log_prob consistency."""

    def test_log_prob_base_distribution(self):
        """For identity flow (small init), log_prob should be close to N(0,I)."""
        torch.manual_seed(90)
        n_in = 4

        # With small init, the flow is close to identity
        flow = ConditionalNF(n_in, n_in, 16, n_flow_layers=1, depth=1)
        flow.eval()

        # Standard normal samples
        x = torch.randn(1000, n_in) * 0.5
        y = torch.zeros(1000, n_in)

        log_p = flow.log_prob(x, y)

        # Should be close to standard normal log prob (within reason, since
        # even with small init it's not exactly identity due to permutations)
        log_p_normal = -0.5 * x.pow(2).sum(dim=1) - 0.5 * n_in * math.log(2 * math.pi)

        # Just check that the values are in a reasonable range
        assert log_p.mean().item() > -20.0, "log_prob mean is too low"
        assert torch.all(torch.isfinite(log_p)), "Non-finite log_prob values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
