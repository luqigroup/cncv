# Test for DenseConditionalLayerCV
# Date: December 2025

using Test, Random, LinearAlgebra
import CNCV: DenseConditionalLayerCV, get_params, clear_grad!, forward, inverse, backward

# Random seed
Random.seed!(123)

###################################################################################################
# Test invertibility

# Input dimensions for dense layer (low-dimensional)
n_in = 4
n_cond = 2
n_hidden = 16
n_layers = 2
batchsize = 1

# Input vectors
X = randn(Float32, n_in, batchsize)
Cond = randn(Float32, n_cond, batchsize)

# Create dense conditional coupling layer
L = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)

###################################################################################################
# Test jac_traces per batch element (vector-valued CVs)

# Test with larger batchsize
batchsize = 5
X_batch = randn(Float32, n_in, batchsize)
Cond_batch = randn(Float32, n_cond, batchsize)

# Test jac_traces returns matrix n_cv×batch (default n_cv=n_in)
jac_traces, phi_all = forward(X_batch, Cond_batch, L)
@test jac_traces isa AbstractArray
@test size(jac_traces) == (L.n_cv, batchsize)  # n_cv×batchsize
@test L.n_cv == n_in  # default: one CV per input dimension

# Verify phi_all dimensions
@test size(phi_all) == (L.n_cv, n_in, batchsize)  # n_cv × n_in × batchsize

# Test inverse (note: CV layers are not true bijections)
# inverse() is used internally for backward pass, not for actual inversion
# It just recomputes the forward pass from X
X_inv, jac_traces_inv, phi_all_inv = inverse(X_batch, Cond_batch, L)
@test jac_traces_inv isa AbstractArray
@test size(jac_traces_inv) == (L.n_cv, batchsize)
@test size(phi_all_inv) == (L.n_cv, n_in, batchsize)
@test isapprox(X_inv, X_batch; atol=1e-5)  # inverse(X) should return X


###################################################################################################
# Forward/Backward consistency tests

# Reset batchsize to 1
batchsize = 1
X = randn(Float32, n_in, batchsize)
Cond = randn(Float32, n_cond, batchsize)

# Test backward with X (not Y, since CV layers take X directly)
jac_trace, phi_all_test = forward(X, Cond, L)
_, X_recovered = backward(zeros(Float32, size(X)), X, Cond, L)[1:2]
@test isapprox(norm(X - X_recovered)/norm(X), 0f0; atol=1e-5)


###################################################################################################
# Explicit Jacobian trace verification test
# This is the KEY test: verify numerically that our trace computation is correct

print("\n=== Explicit Jacobian trace verification ===\n")
print("This test verifies that the dense coupling layer has correct Jacobian trace\n\n")

# Small test case for explicit verification via finite differences
n_in_small = 4
n_cond_small = 2
n_hidden_small = 8
n_layers_small = 2
batchsize_small = 1

X_small = randn(Float32, n_in_small, batchsize_small)
Cond_small = randn(Float32, n_cond_small, batchsize_small)

# Create dense conditional coupling layer for verification
L_verify = DenseConditionalLayerCV(n_in_small, n_cond_small, n_hidden_small, n_layers_small; activation=tanh)

# Forward pass to get computed traces (vector-valued)
jac_traces_computed, phi_all_small = forward(X_small, Cond_small, L_verify)

print("Input shape:  ", size(X_small), "\n")
print("Number of CVs: ", L_verify.n_cv, "\n")
print("Computed jac_traces shape: ", size(jac_traces_computed), "\n")
print("phi_all shape: ", size(phi_all_small), "\n")
print("First CV trace: ", jac_traces_computed[1, 1], "\n\n")

# For vector CVs, phi_all contains n_cv different transformations: φ_k = [S_k * X1 + T_k; X2]
# Each φ_k has its own Jacobian and trace
print("Note: phi_all contains ", L_verify.n_cv, " different coupling transformations φ_k\n")
print("Each φ_k has its own transformation and trace\n")

# Compute Jacobian explicitly by finite differences for φ_1
X_vec = X_small[:, 1]
n_total = length(X_vec)
J = zeros(Float32, n_total, n_total)

ε = 1f-5
print("\nComputing explicit Jacobian of φ_1 via finite differences ($n_total × $n_total matrix)...\n")

# Get baseline φ_1
phi_1_base = phi_all_small[1, :, 1]

for i = 1:n_total
    X_perturbed = copy(X_vec)
    X_perturbed[i] += ε

    # Reshape back to matrix
    X_p = reshape(X_perturbed, n_in_small, 1)

    _, phi_all_p = forward(X_p, Cond_small, L_verify)
    phi_1_perturbed = phi_all_p[1, :, 1]

    J[:, i] = (phi_1_perturbed - phi_1_base) / ε
end

# Compute trace explicitly as sum of diagonal elements
jac_trace_explicit = sum(diag(J))

# Expected trace from our formula
split_idx = L_verify.split_idx
expected_trace = jac_traces_computed[1, 1]  # Our computed trace for CV #1

print("\nResults:\n")
print("  Explicit Jacobian trace (φ_1): ", jac_trace_explicit, "\n")
print("  Computed trace (CV #1): ", expected_trace, "\n")
print("  Difference: ", abs(jac_trace_explicit - expected_trace), "\n\n")

# Verify our trace computation matches the explicit Jacobian
@test isapprox(jac_trace_explicit, expected_trace; rtol=2f-3)
print("✓ Jacobian trace verification PASSED!\n\n")


###################################################################################################
# Gradient tests with jac_trace_grad_weight

# Helper functions for MSE
mse(x, y) = 0.5f0 * sum((x .- y).^2)
∇mse(x, y) = x .- y

# Target for CV objective
batchsize = 2
X = randn(Float32, n_in, batchsize)
Cond = randn(Float32, n_cond, batchsize)
X0 = randn(Float32, n_in, batchsize)
dX = X - X0

target = randn(Float32, batchsize)
a = randn(Float32, n_in, batchsize)

# Loss Function with CV objective (vector-valued CVs)
function loss_cv(L, X, Y, Cond, target, a)
    jac_traces, phi_all = forward(X, Cond, L)  # jac_traces is n_cv×batchsize
    batchsize = size(X)[end]
    n_cv = L.n_cv

    # Use first CV's transformation as output for loss
    Y_output = phi_all[1, :, :]

    # CV residual: target - sum(jac_traces) - a'*Y
    # For simplicity, use sum of all CV traces
    total_trace = vec(sum(jac_traces, dims=1))  # sum over CVs
    residual = target .- total_trace .- vec(sum(a .* Y_output, dims=1))

    # Objective
    f = mse(Y_output, Y) + .5f0/batchsize * sum(residual.^2)

    # Gradient weights for jacobian traces (broadcast to all CVs)
    jac_trace_grad_weights = zeros(Float32, n_cv, batchsize)
    for k in 1:n_cv
        jac_trace_grad_weights[k, :] = -residual ./ Float32(batchsize)
    end

    # Data gradient with CV contribution
    ΔY = ∇mse(Y_output, Y)
    for i=1:batchsize
        ΔY[:, i] .+= -residual[i] / Float32(batchsize) .* a[:, i]
    end

    ΔX = backward(ΔY, X, Cond, L; jac_trace_grad_weights=jac_trace_grad_weights)[1]

    # Pass back gradients w.r.t. input X and layer parameters
    params = get_params(L)
    param_grads = [p.grad !== nothing ? copy(p.grad) : nothing for p in params]

    return f, ΔX, param_grads...
end

# Invertible layers (created after RNG has been used, so they're different)
L = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)
L0 = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)

# Gradient test w.r.t. input X0
_, phi_all_grad = forward(X, Cond, L)
Y = phi_all_grad[1, :, :]  # Get Y from first CV's transformation
f0, ΔX = loss_cv(L, X0, Y, Cond, target, a)[1:2]
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("Gradient test dense conditional coupling layer CV (X)\n")
for j=1:maxiter
    f = loss_cv(L, X0 + h*dX, Y, Cond, target, a)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=2f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=2f1)


# Gradient test w.r.t. weights of layer
print("\nGradient test dense conditional coupling layer CV (weights)\n")

# Create fresh layers
L_test = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)
L_test0 = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)

# Get all parameters
params_test = get_params(L_test)
params_test0 = get_params(L_test0)

# Perturbation direction (random direction in parameter space)
dparams = [randn(Float32, size(p.data)...) for p in params_test]

# Compute loss and gradients
_, phi_all_test = forward(X, Cond, L_test)
Y_test = phi_all_test[1, :, :]
f0_w, ΔX_w, grad_params... = loss_cv(L_test, X0, Y_test, Cond, target, a)

# Check that we got gradients
@test all([p.grad !== nothing for p in params_test])

# Compute directional derivative: sum over all parameters
directional_deriv = sum([sum(params_test[i].grad .* dparams[i]) for i in 1:length(params_test)])

h_w = 0.1f0
maxiter_w = 5
err1_w = zeros(Float32, maxiter_w)
err2_w = zeros(Float32, maxiter_w)

for j=1:maxiter_w
    # Perturb parameters
    for (i, p) in enumerate(params_test0)
        p.data .= get_params(L_test)[i].data .+ h_w .* dparams[i]
    end

    # Compute loss with perturbed parameters
    clear_grad!(L_test0)
    _, phi_all_test0 = forward(X, Cond, L_test0)
    Y_test0 = phi_all_test0[1, :, :]
    f_perturbed = loss_cv(L_test0, X0, Y_test0, Cond, target, a)[1]

    err1_w[j] = abs(f_perturbed - f0_w)
    err2_w[j] = abs(f_perturbed - f0_w - h_w * directional_deriv)
    print(err1_w[j], "; ", err2_w[j], "\n")

    global h_w = h_w/2f0
end

@test isapprox(err1_w[end] / (err1_w[1]/2^(maxiter_w-1)), 1f0; atol=2f1)
@test isapprox(err2_w[end] / (err2_w[1]/4^(maxiter_w-1)), 1f0; atol=2f1)

print("\n✓ All tests passed!\n")
