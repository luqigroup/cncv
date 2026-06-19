"""Nonlinear forward model utilities.

Provides the NonlinearForwardModel (F(x) = Ax + sin(x)) and score computation
for reuse across training and evaluation scripts.
"""

import torch


class NonlinearForwardModel:
    """Nonlinear forward model F(x) = Ax + sin(x).

    A is constructed with controlled condition number via SVD.

    Args:
        dim: Problem dimension
        cond_number: Condition number for A (default: 5.0)
        seed: Random seed for A construction (default: 42)
        device: Computation device
    """

    def __init__(self, dim, cond_number=5.0, seed=42, device="cpu"):
        self.dim = dim
        self.device = device

        # Construct A with controlled condition number
        torch.manual_seed(seed)
        U, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        V, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        singular_values = torch.linspace(1.0, cond_number, dim, device=device)
        S = torch.diag(singular_values)
        self.A = (U @ S @ V.T).to(dtype=torch.float32)

    def forward(self, x):
        """F(x) = Ax + sin(x). x: [..., dim]"""
        return x @ self.A.T + torch.sin(x)

    def jacobian(self, x):
        """J_F(x) = A + diag(cos(x)). x: [batch, dim] -> [batch, dim, dim]"""
        batch = x.shape[0]
        cos_x = torch.cos(x)  # [batch, dim]
        J = self.A.unsqueeze(0).expand(batch, -1, -1).clone()  # [batch, dim, dim]
        idx = torch.arange(self.dim, device=x.device)
        J[:, idx, idx] += cos_x
        return J


def compute_score_nonlinear(X, Y_obs, forward_model, mu_prior, Sigma_prior_inv, sigma):
    """Score for nonlinear forward model posterior.

    grad_log_post = -J_F(x)^T (F(x) - y) / sigma^2 - Sigma_prior_inv @ (x - mu_prior)

    Args:
        X: [batch, dim]
        Y_obs: [dim] or [batch, dim]
        forward_model: NonlinearForwardModel
        mu_prior: [dim]
        Sigma_prior_inv: [dim, dim]
        sigma: float

    Returns:
        score: [batch, dim]
    """
    F_x = forward_model.forward(X)  # [batch, dim]
    J_F = forward_model.jacobian(X)  # [batch, dim, dim]

    if Y_obs.ndim == 1:
        residual = F_x - Y_obs.unsqueeze(0)  # [batch, dim]
    else:
        residual = F_x - Y_obs  # [batch, dim]

    # -J_F^T @ residual / sigma^2
    grad_lik = -torch.bmm(J_F.transpose(1, 2), residual.unsqueeze(2)).squeeze(2) / (sigma ** 2)

    # Prior gradient
    grad_prior = -(Sigma_prior_inv @ (X - mu_prior).T).T

    return grad_lik + grad_prior


def setup_nonlinear_problem(dim, sigma=0.3, cond_number=2.0, device="cpu"):
    """Set up the nonlinear forward model problem (prior, forward model, etc.).

    Args:
        dim: Problem dimension
        sigma: Observation noise std
        cond_number: Condition number for forward model A matrix
        device: Computation device

    Returns:
        dict with keys: mu_prior, Sigma_prior, Sigma_prior_inv, L_prior,
                        forward_model, sigma
    """
    mu_prior = torch.zeros(dim, dtype=torch.float32, device=device)

    torch.manual_seed(42)
    diag_values = torch.linspace(1.0, 2.0, dim)
    D_diag = torch.diag(diag_values).to(device=device, dtype=torch.float32)
    if dim > 1:
        A = torch.randn(dim, dim, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        Sigma_prior = Q @ D_diag @ Q.T
    else:
        Sigma_prior = D_diag

    L_prior = torch.linalg.cholesky(Sigma_prior)
    Sigma_prior_inv = torch.linalg.inv(Sigma_prior)

    forward_model = NonlinearForwardModel(dim, cond_number=cond_number, seed=42, device=device)

    return {
        "mu_prior": mu_prior,
        "Sigma_prior": Sigma_prior,
        "Sigma_prior_inv": Sigma_prior_inv,
        "L_prior": L_prior,
        "forward_model": forward_model,
        "sigma": sigma,
    }
