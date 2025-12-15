# Test invertible conditional coupling layer (basic CV version - with jacobian trace)
# Date: January 2025

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, set_params!, tensor_cat, tensor_split, mse, ∇mse, ResidualBlock, Sigmoid2Layer

using CNCV

# Random seed
Random.seed!(11)

###################################################################################################
# Test invertibility

# Input
nx = 24
ny = 24
k = 4
n_cond = k
n_hidden = 4
batchsize = 1

# Input images
X = randn(Float32, nx, ny, k, batchsize)
Cond = randn(Float32, nx, ny, n_cond, batchsize)
X0 = randn(Float32, nx, ny, k, batchsize)
dX = X - X0

# Residual block and conditional coupling layer
RB = ResidualBlock(Int(k/2)+n_cond, n_hidden; n_out=k, k1=3, k2=3, p1=1, p2=1, fan=true)
L = ConditionalLayerBasicCV(RB)

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

# Small test case for explicit verification via finite differences
print("\nExplicit Jacobian trace verification\n")
nx_small = 4
ny_small = 4
k_small = 4
n_cond_small = k_small
n_hidden_small = 4
batchsize_small = 1

X_small = randn(Float32, nx_small, ny_small, k_small, batchsize_small)
Cond_small = randn(Float32, nx_small, ny_small, n_cond_small, batchsize_small)

# Create conditional coupling layer for verification
RB_verify = ResidualBlock(Int(k_small/2)+n_cond_small, n_hidden_small; n_out=k_small, k1=3, k2=3, p1=1, p2=1, fan=true)
L_verify = ConditionalLayerBasicCV(RB_verify)

# Forward pass to get computed trace
Y_small, jac_trace_computed = L_verify.forward(X_small, Cond_small)

# Compute Jacobian explicitly by finite differences
X_concat = vec(X_small)
n_total = length(X_concat)
J = zeros(Float32, n_total, n_total)

ε = 1f-5
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

# Compute trace explicitly
jac_trace_explicit = sum(diag(J))
identity_contribution = nx_small * ny_small * Int(k_small/2)  # X2 part that passes through

println("Computed jac_trace: ", jac_trace_computed[1])
println("Explicit jac_trace: ", jac_trace_explicit)
println("Identity contribution (nx*ny*n_in/2): ", identity_contribution)
println("Difference: ", abs(jac_trace_computed[1] - jac_trace_explicit))

# Test that they match
@test isapprox(jac_trace_computed[1], jac_trace_explicit; rtol=1f-3)


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
    return f, ΔX, L.RB.W1.grad, L.RB.W2.grad, L.RB.W3.grad
end

# Invertible layers
RB0 = ResidualBlock(Int(k/2)+n_cond, n_hidden; n_out=k, k1=3, k2=3, p1=1, p2=1, fan=true)
L = ConditionalLayerBasicCV(RB)
L02 = ConditionalLayerBasicCV(RB0)

# Gradient test w.r.t. input X0
Y = L.forward(X, Cond)[1]
f0, ΔX = loss_cv(L, X0, Y, Cond, target, a)[1:2]
h = 0.1f0
maxiter = 5  # Reduced to avoid numerical precision issues
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test conditional coupling layer CV (X)\n")
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
Lini = deepcopy(L02)
dW1 = L.RB.W1.data - L02.RB.W1.data
dW2 = L.RB.W2.data - L02.RB.W2.data
dW3 = L.RB.W3.data - L02.RB.W3.data

f0, ΔX, ΔW1, ΔW2, ΔW3 = loss_cv(L02, X, Y, Cond, target, a)
h = 0.5f0  # Larger initial step size
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test conditional coupling layer CV (params)\n")
for j=1:maxiter
    L02.RB.W1.data = Lini.RB.W1.data + h*dW1
    L02.RB.W2.data = Lini.RB.W2.data + h*dW2
    L02.RB.W3.data = Lini.RB.W3.data + h*dW3
    f = loss_cv(L02, X, Y, Cond, target, a)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
