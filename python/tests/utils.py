"""Test utility functions for CNCV.

Provides utilities for:
- Finite difference Jacobian computation
- Adjoint tests (dot product identity)
- Taylor series gradient checks
"""

import torch
import numpy as np


def finite_diff_jacobian(func, x, eps=1e-5):
    """Compute Jacobian matrix using finite differences.

    Args:
        func: Function that maps x -> y where x is [n_in] and y is [n_out]
        x: Input vector [n_in]
        eps: Finite difference step size

    Returns:
        J: Jacobian matrix [n_out, n_in]
    """
    x = x.detach().clone().requires_grad_(False)
    n_in = x.numel()

    # Compute base output
    y_base = func(x)
    n_out = y_base.numel()

    # Initialize Jacobian
    J = torch.zeros(n_out, n_in, dtype=x.dtype, device=x.device)

    # Perturb each input element
    for i in range(n_in):
        x_perturbed = x.clone()
        x_perturbed.view(-1)[i] += eps

        y_perturbed = func(x_perturbed)

        # Finite difference
        J[:, i] = (y_perturbed.view(-1) - y_base.view(-1)) / eps

    return J


def finite_diff_grad(func, x, eps=1e-5):
    """Compute gradient of scalar function using finite differences.

    Args:
        func: Function that maps x -> scalar
        x: Input tensor of any shape
        eps: Finite difference step size

    Returns:
        grad: Gradient tensor with same shape as x
    """
    x = x.detach().clone().requires_grad_(False)
    x_flat = x.view(-1)
    n = x_flat.numel()

    # Compute base value
    f_base = func(x)

    # Initialize gradient
    grad_flat = torch.zeros_like(x_flat)

    # Perturb each element
    for i in range(n):
        x_perturbed = x.clone()
        x_perturbed.view(-1)[i] += eps

        f_perturbed = func(x_perturbed)

        # Finite difference
        grad_flat[i] = (f_perturbed - f_base) / eps

    return grad_flat.view_as(x)


def adjoint_test(jvp_func, vjp_func, x, dy, rtol=1e-5):
    """Test adjoint relationship: <dy, J*dx> = <J^T*dy, dx>.

    Args:
        jvp_func: Function(x, dx) that computes J*dx (Jacobian-vector product)
        vjp_func: Function(x, dy) that computes J^T*dy (vector-Jacobian product)
        x: Input point
        dy: Adjoint vector (same shape as output)
        rtol: Relative tolerance

    Returns:
        passed: True if adjoint test passes
        a: Left dot product <dy, J*dx>
        b: Right dot product <J^T*dy, dx>
    """
    # Random perturbation
    dx = torch.randn_like(x)

    # Forward: dy_dot_Jdx = <dy, J*dx>
    Jdx = jvp_func(x, dx)
    a = torch.dot(dy.view(-1), Jdx.view(-1))

    # Adjoint: JTdy_dot_dx = <J^T*dy, dx>
    JTdy = vjp_func(x, dy)
    b = torch.dot(JTdy.view(-1), dx.view(-1))

    # Check if they're approximately equal
    passed = torch.isclose(a, b, rtol=rtol)

    return passed.item(), a.item(), b.item()


def gradient_check_taylor(func, x, dx=None, steps=[1e-1, 1e-2, 1e-3, 1e-4], rtol_order1=0.1, rtol_order2=0.01):
    """Taylor series gradient check with decreasing step sizes.

    Verifies that:
    - err1 = |f(x + h*dx) - f(x)| / |h| ~ O(1) (zeroth order)
    - err2 = |f(x + h*dx) - f(x) - h*∇f(x)^T*dx| / |h^2| ~ O(1) (first order)

    If gradients are correct, err1 should decrease linearly and err2 should stay constant.

    Args:
        func: Function that computes both f(x) and ∇f(x)
              Should return (f_val, grad_f) when requires_grad=True
        x: Input tensor
        dx: Perturbation direction (random if None)
        steps: List of step sizes to test
        rtol_order1: Relative tolerance for first-order consistency
        rtol_order2: Relative tolerance for second-order consistency

    Returns:
        passed: True if gradient check passes
        errors1: First-order errors (should decrease linearly)
        errors2: Second-order errors (should stay constant)
    """
    if dx is None:
        dx = torch.randn_like(x)

    # Normalize dx
    dx = dx / dx.norm()

    # Compute function and gradient at x
    x.requires_grad_(True)
    f_x = func(x)
    f_x.backward()
    grad_f = x.grad.clone()
    x.grad.zero_()

    # Directional derivative
    directional_deriv = torch.dot(grad_f.view(-1), dx.view(-1))

    errors1 = []
    errors2 = []

    for h in steps:
        # Perturbed point
        x_perturbed = x + h * dx
        x_perturbed = x_perturbed.detach().requires_grad_(False)

        # Function value at perturbed point
        f_perturbed = func(x_perturbed)

        # First-order error: |f(x+h*dx) - f(x)| / h
        err1 = torch.abs(f_perturbed - f_x.detach()) / h

        # Second-order error: |f(x+h*dx) - f(x) - h*∇f^T*dx| / h^2
        err2 = torch.abs(f_perturbed - f_x.detach() - h * directional_deriv) / (h**2)

        errors1.append(err1.item())
        errors2.append(err2.item())

    # Check that err1 decreases and err2 stays roughly constant
    # err1 should decrease by ~factor of 10 for each step
    # err2 should stay within same order of magnitude

    ratio1 = errors1[0] / errors1[-1]  # Should be large (>10)
    ratio2_max = max(errors2) / min(errors2)  # Should be small (<10)

    passed = (ratio1 > 1.0 / rtol_order1) and (ratio2_max < 1.0 / rtol_order2)

    return passed, errors1, errors2


def gradient_check_finite_diff(func, x, eps=1e-5, rtol=1e-3):
    """Simple gradient check using finite differences.

    Args:
        func: Function that computes f(x)
        x: Input tensor
        eps: Finite difference step size
        rtol: Relative tolerance

    Returns:
        passed: True if all gradients match
        grad_autograd: Gradient from autograd
        grad_fd: Gradient from finite differences
    """
    # Autograd gradient
    x.requires_grad_(True)
    f_x = func(x)
    f_x.backward()
    grad_autograd = x.grad.clone()
    x.grad.zero_()
    x.requires_grad_(False)

    # Finite difference gradient
    grad_fd = finite_diff_grad(func, x, eps=eps)

    # Compare
    passed = torch.allclose(grad_autograd, grad_fd, rtol=rtol, atol=1e-6)

    return passed, grad_autograd, grad_fd
