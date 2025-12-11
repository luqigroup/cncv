# Forward pass
Y, jac_trace = network.forward(X)

# Compute your objective: loss = ||target - jac_trace - a'*Y||²
# Get gradient weight: jac_trace_grad_weight = -2 * (target - jac_trace - a'*Y)

# Backward pass with integrated gradients
# Data gradient: ΔY = -2 * (target - jac_trace - a'*Y) * (-a)
# Total gradient = J^T * ΔY + jac_trace_grad_weight * ∇_θ trace(J)
ΔX, X = network.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)


# Compute residual
residual = target - jac_trace - a'*Y  # or: target - jac_trace - dot(a, Y)

# Gradient weight for jacobian trace
jac_trace_grad_weight = -2 * residual  # shape: (batchsize,)

# Data gradient (this is what you pass to backward)
ΔY = -2 * residual * (-a)  # = 2 * residual * a
# ΔY has same shape as Y

# Backward pass
ΔX, X = network.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)
```

The network's `backward` function computes:
- **Data term**: `J^T * ΔY` (where `ΔY = 2 * residual * a`)
- **Trace term**: `jac_trace_grad_weight * ∇_θ trace(J)` (where `jac_trace_grad_weight = -2 * residual`)

So the complete gradient is:
```
∇_θ L = J^T * (2 * residual * a) + (-2 * residual) * ∇_θ trace(J)
      = 2 * residual * (J^T * a - ∇_θ trace(J))
