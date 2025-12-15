# Test squeeze + conditional coupling layer (basic CV version - with jacobian trace)
# Date: January 2025

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, set_params!, tensor_cat, tensor_split, mse, ∇mse, ResidualBlock, squeeze, unsqueeze

using CNCV

# Random seed
Random.seed!(11)

###################################################################################################
# Test invertibility

# Input - note smaller spatial dimensions since we'll be squeezing
nx = 12
ny = 12
k = 4  # Will become 16 after squeezing (4*4 in 2D)
n_cond = k
n_hidden = 4
batchsize = 1

# Input images
X = randn(Float32, nx, ny, k, batchsize)
Cond = randn(Float32, nx, ny, n_cond, batchsize)

# Create squeeze + conditional coupling layer
L = SqueezeConditionalLayerBasicCV(k, n_cond, n_hidden; k1=3, k2=3, p1=1, p2=1, pattern="column")

###################################################################################################
# Test jac_trace per batch element

# Test with larger batchsize
batchsize = 5
X_batch = randn(Float32, nx, ny, k, batchsize)
Cond_batch = randn(Float32, nx, ny, n_cond, batchsize)

# Test jac_trace returns vector with one element per batch
Y, jac_trace_vec = L.forward(X_batch, Cond_batch)
@test jac_trace_vec isa AbstractArray
@test length(jac_trace_vec) == batchsize

# Verify output dimensions after squeeze-coupling-unsqueeze
@test size(Y) == size(X_batch)  # Should be same shape as input

# Test inverse jac_trace
X_inv, jac_trace_inv = L.inverse(Y, Cond_batch)
@test isapprox(norm(X_batch - X_inv)/norm(X_batch), 0f0; atol=1e-2)
@test jac_trace_inv isa AbstractArray
@test length(jac_trace_inv) == batchsize


###################################################################################################
# Invertibility tests

# Reset batchsize to 1
batchsize = 1
X = randn(Float32, nx, ny, k, batchsize)
Cond = randn(Float32, nx, ny, n_cond, batchsize)

# Layer test
Y, jac_trace = L.forward(X, Cond)
X_, _ = L.inverse(Y, Cond)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

Y, jac_trace = L.forward(X, Cond)
_, X_ = L.backward(Y.*0f0, Y, Cond)[1:2]
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)

Y, _ = L.inverse(X, Cond)
X_, jac_trace = L.forward(Y, Cond)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1e-2)


###################################################################################################
# Explicit Jacobian trace verification test
# This is the KEY test: verify numerically that our trace computation is correct

print("\n=== Explicit Jacobian trace verification ===\n")
print("This test verifies that the composed layer (squeeze + coupling) has correct Jacobian trace\n\n")

# Small test case for explicit verification via finite differences
nx_small = 4
ny_small = 4
k_small = 4  # Will become 16 after squeezing
n_cond_small = k_small
n_hidden_small = 4
batchsize_small = 1

X_small = randn(Float32, nx_small, ny_small, k_small, batchsize_small)
Cond_small = randn(Float32, nx_small, ny_small, n_cond_small, batchsize_small)

# Create squeeze + conditional coupling layer for verification
L_verify = SqueezeConditionalLayerBasicCV(k_small, n_cond_small, n_hidden_small;
                                          k1=3, k2=3, p1=1, p2=1, pattern="column")

# Forward pass to get computed trace
Y_small, jac_trace_computed = L_verify.forward(X_small, Cond_small)

print("Input shape (before squeeze):  ", size(X_small), "\n")
print("Output shape (after unsqueeze): ", size(Y_small), "\n")
print("Computed jac_trace: ", jac_trace_computed[1], "\n\n")

# Compute Jacobian explicitly by finite differences
# This computes the full Jacobian matrix J where Y = f(X)
X_concat = vec(X_small)
n_total = length(X_concat)
J = zeros(Float32, n_total, n_total)

ε = 1f-5
print("Computing explicit Jacobian via finite differences ($n_total × $n_total matrix)...\n")
for i = 1:n_total
    X_perturbed = copy(X_concat)
    X_perturbed[i] += ε

    # Reshape back to tensor
    X_p = reshape(X_perturbed, size(X_small))

    Y_p, _ = L_verify.forward(X_p, Cond_small)
    Y_perturbed = vec(Y_p)

    Y_base = vec(Y_small)
    J[:, i] = (Y_perturbed - Y_base) / ε
end

# Compute trace explicitly as sum of diagonal elements
jac_trace_explicit = sum(diag(J))

# For this layer:
# - After squeezing: nx_small/2, ny_small/2, k_small*4
# - Coupling splits into two halves along channel dimension
# - Identity contribution is from the half that passes through unchanged
# - For pattern="column", squeeze is just reshape, so trace is the same as coupling layer alone
identity_contribution = (nx_small/2) * (ny_small/2) * (k_small*4/2)  # Half the channels after squeezing

print("\nResults:\n")
print("  Computed jac_trace: ", jac_trace_computed[1], "\n")
print("  Explicit jac_trace: ", jac_trace_explicit, "\n")
print("  Expected identity contribution: ", identity_contribution, "\n")
print("  Difference: ", abs(jac_trace_computed[1] - jac_trace_explicit), "\n\n")

# Test that they match
@test isapprox(jac_trace_computed[1], jac_trace_explicit; rtol=1f-3)
print("✓ Jacobian trace verification PASSED!\n\n")


###################################################################################################
# Gradient tests with jac_trace_grad_weight

# Target for CV objective
batchsize = 2
X = randn(Float32, nx, ny, k, batchsize)
Cond = randn(Float32, nx, ny, n_cond, batchsize)
X0 = randn(Float32, nx, ny, k, batchsize)
dX = X - X0

target = randn(Float32, batchsize)
a = randn(Float32, nx, ny, k, batchsize)

# Loss Function with CV objective
function loss_cv(L, X, Y, Cond, target, a)
    Y_, jac_trace = L.forward(X, Cond)
    batchsize = size(X)[end]

    # CV residual: target - jac_trace - a'*Y
    residual = target .- jac_trace .- vec(sum(a .* Y_, dims=[1,2,3]))

    # Objective
    f = mse(Y_, Y) + .5f0/batchsize * sum(residual.^2)

    # Gradient weight for jacobian trace
    jac_trace_grad_weight = -residual ./ Float32(batchsize)

    # Data gradient with CV contribution
    ΔY = ∇mse(Y_, Y)
    for i=1:batchsize
        selectdim(ΔY, 4, i) .+= -residual[i] / Float32(batchsize) .* selectdim(a, 4, i)
    end

    ΔX = L.backward(ΔY, Y_, Cond; jac_trace_grad_weight=jac_trace_grad_weight)[1]

    # Pass back gradients w.r.t. input X and from the residual block
    return f, ΔX, L.CL.RB.W1.grad, L.CL.RB.W2.grad, L.CL.RB.W3.grad
end

# Invertible layers
L = SqueezeConditionalLayerBasicCV(k, n_cond, n_hidden; k1=3, k2=3, p1=1, p2=1, pattern="column")
L0 = SqueezeConditionalLayerBasicCV(k, n_cond, n_hidden; k1=3, k2=3, p1=1, p2=1, pattern="column")

# Gradient test w.r.t. input X0
Y = L.forward(X, Cond)[1]
f0, ΔX = loss_cv(L, X0, Y, Cond, target, a)[1:2]
h = 0.1f0
maxiter = 5  # Reduced to avoid numerical precision issues
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test squeeze conditional coupling layer CV (X)\n")
for j=1:maxiter
    f = loss_cv(L, X0 + h*dX, Y, Cond, target, a)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test w.r.t. weights of residual block
Y = L.forward(X, Cond)[1]
Lini = deepcopy(L0)
dW1 = L.CL.RB.W1.data - L0.CL.RB.W1.data
dW2 = L.CL.RB.W2.data - L0.CL.RB.W2.data
dW3 = L.CL.RB.W3.data - L0.CL.RB.W3.data

f0, ΔX, ΔW1, ΔW2, ΔW3 = loss_cv(L0, X, Y, Cond, target, a)
h = 0.5f0  # Larger initial step size
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test squeeze conditional coupling layer CV (params)\n")
for j=1:maxiter
    L0.CL.RB.W1.data = Lini.CL.RB.W1.data + h*dW1
    L0.CL.RB.W2.data = Lini.CL.RB.W2.data + h*dW2
    L0.CL.RB.W3.data = Lini.CL.RB.W3.data + h*dW3
    f = loss_cv(L0, X, Y, Cond, target, a)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

print("\n✓ All tests passed!\n")
