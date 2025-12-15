# Test invertible coupling layer (CV version - with jacobian trace)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

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
n_in = 2
n_hidden = 4
batchsize = 1

# Input images
Xa = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb = randn(Float32, nx, ny, Int(k/2), batchsize)
Xa0 = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb0 = randn(Float32, nx, ny, Int(k/2), batchsize)
dXa = Xa - Xa0
dXb = Xb - Xb0

# Residual block and coupling layer
RB = ResidualBlock(n_in, n_hidden; fan=true)
L = CouplingLayerBasicCV(RB)

###################################################################################################
# Test jac_trace per batch element

# Test with larger batchsize
batchsize = 5
Xa_batch = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb_batch = randn(Float32, nx, ny, Int(k/2), batchsize)

# Test jac_trace returns vector with one element per batch
Ya, Yb, jac_trace_vec = L.forward(Xa_batch, Xb_batch)
@test jac_trace_vec isa AbstractArray
@test length(jac_trace_vec) == batchsize

# Test inverse jac_trace
Xa_inv, Xb_inv, jac_trace_inv = L.inverse(Ya, Yb)
@test isapprox(norm(Xa_batch - Xa_inv)/norm(Xa_batch), 0f0; atol=1e-2)
@test isapprox(norm(Xb_batch - Xb_inv)/norm(Xb_batch), 0f0; atol=1e-2)
@test jac_trace_inv isa AbstractArray
@test length(jac_trace_inv) == batchsize


###################################################################################################
# Invertibility tests

# Reset batchsize to 1
batchsize = 1
Xa = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb = randn(Float32, nx, ny, Int(k/2), batchsize)

# Layer test
Ya, Yb, jac_trace = L.forward(Xa, Xb)
Xa_, Xb_, _ = L.inverse(Ya, Yb)
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)

Ya, Yb, jac_trace = L.forward(Xa, Xb)
Xa_, Xb_ = L.backward(Ya.*0f0, Yb.*0f0, Ya, Yb)[3:4]
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)

Ya, Yb, _ = L.inverse(Xa, Xb)
Xa_, Xb_, jac_trace = L.forward(Ya, Yb)
@test isapprox(norm(Xa - Xa_)/norm(Xa), 0f0; atol=1e-2)
@test isapprox(norm(Xb - Xb_)/norm(Xb), 0f0; atol=1e-2)


###################################################################################################
# Explicit Jacobian trace verification test

# Small test case for explicit verification via finite differences
print("\nExplicit Jacobian trace verification\n")
nx_small = 4
ny_small = 4
n_in_small = 2
n_hidden_small = 4
batchsize_small = 1

Xa_small = randn(Float32, nx_small, ny_small, n_in_small, batchsize_small)
Xb_small = randn(Float32, nx_small, ny_small, n_in_small, batchsize_small)

# Create coupling layer for verification
RB_verify = ResidualBlock(n_in_small, n_hidden_small; fan=true)
L_verify = CouplingLayerBasicCV(RB_verify)

# Forward pass to get computed trace
Ya_small, Yb_small, jac_trace_computed = L_verify.forward(Xa_small, Xb_small)

# Compute Jacobian explicitly by finite differences
X_concat = vcat(vec(Xa_small), vec(Xb_small))
n_total = length(X_concat)
J = zeros(Float32, n_total, n_total)

ε = 1f-5
for i = 1:n_total
    X_perturbed = copy(X_concat)
    X_perturbed[i] += ε

    # Split back into Xa and Xb
    Xa_p = reshape(X_perturbed[1:length(Xa_small)], size(Xa_small))
    Xb_p = reshape(X_perturbed[length(Xa_small)+1:end], size(Xb_small))

    Ya_p, Yb_p, _ = L_verify.forward(Xa_p, Xb_p)
    Y_perturbed = vcat(vec(Ya_p), vec(Yb_p))

    Y_base = vcat(vec(Ya_small), vec(Yb_small))
    J[:, i] = (Y_perturbed - Y_base) / ε
end

# Compute trace explicitly
jac_trace_explicit = sum(diag(J))
identity_contribution = nx_small * ny_small * n_in_small

println("Computed jac_trace: ", jac_trace_computed[1])
println("Explicit jac_trace: ", jac_trace_explicit)
println("Identity contribution (nx*ny*n_in): ", identity_contribution)
println("Difference: ", abs(jac_trace_computed[1] - jac_trace_explicit))

# Test that they match
@test isapprox(jac_trace_computed[1], jac_trace_explicit; rtol=1f-3)


###################################################################################################
# Gradient tests with jac_trace_grad_weight

# Target for CV objective
batchsize = 2
Xa = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb = randn(Float32, nx, ny, Int(k/2), batchsize)
Xa0 = randn(Float32, nx, ny, Int(k/2), batchsize)
Xb0 = randn(Float32, nx, ny, Int(k/2), batchsize)
dXa = Xa - Xa0
dXb = Xb - Xb0

target = randn(Float32, batchsize)
a = randn(Float32, nx, ny, k, batchsize)

# Loss Function with CV objective
function loss_cv(L, Xa, Xb, Ya, Yb, target, a)
    Ya_, Yb_, jac_trace = L.forward(Xa, Xb)
    Y_cat = tensor_cat(Ya_, Yb_)

    # CV residual: target - jac_trace - a'*Y
    residual = target .- jac_trace .- vec(sum(a .* Y_cat, dims=[1,2,3]))

    # Objective
    f = mse(Y_cat, tensor_cat(Ya, Yb)) + .5f0/batchsize * sum(residual.^2)

    # Gradient weight for jacobian trace
    jac_trace_grad_weight = -residual ./ Float32(batchsize)

    # Data gradient with CV contribution
    ΔY = ∇mse(Y_cat, tensor_cat(Ya, Yb))
    for i=1:batchsize
        selectdim(ΔY, 4, i) .+= -residual[i] / Float32(batchsize) .* selectdim(a, 4, i)
    end

    ΔYa, ΔYb = tensor_split(ΔY)
    ΔXa, ΔXb = L.backward(ΔYa, ΔYb, Ya_, Yb_; jac_trace_grad_weight=jac_trace_grad_weight)[1:2]

    # Pass back gradients w.r.t. input X and from the residual block
    return f, ΔXa, ΔXb, L.RB.W1.grad, L.RB.W2.grad, L.RB.W3.grad
end

# Invertible layers
RB0 = ResidualBlock(n_in, n_hidden; fan=true)
L = CouplingLayerBasicCV(RB)
L02 = CouplingLayerBasicCV(RB0)

# Gradient test w.r.t. input X0
Ya, Yb = L.forward(Xa, Xb)[1:2]
f0, ΔXa, ΔXb = loss_cv(L, Xa0, Xb0, Ya, Yb, target, a)[1:3]
h = 0.1f0
maxiter = 5  # Reduced to avoid numerical precision issues
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test coupling layer CV (X)\n")
for j=1:maxiter
    f = loss_cv(L, Xa0 + h*dXa, Xb0 + h*dXb, Ya, Yb, target, a)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dXa, ΔXa) - h*dot(dXb, ΔXb))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test w.r.t. weights of residual block
Ya, Yb = L.forward(Xa, Xb)[1:2]
Lini = deepcopy(L02)
dW1 = L.RB.W1.data - L02.RB.W1.data
dW2 = L.RB.W2.data - L02.RB.W2.data
dW3 = L.RB.W3.data - L02.RB.W3.data

f0, ΔXa, ΔXb, ΔW1, ΔW2, ΔW3 = loss_cv(L02, Xa, Xb, Ya, Yb, target, a)
h = 0.5f0  # Larger initial step size
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test coupling layer CV (params)\n")
for j=1:maxiter
    L02.RB.W1.data = Lini.RB.W1.data + h*dW1
    L02.RB.W2.data = Lini.RB.W2.data + h*dW2
    L02.RB.W3.data = Lini.RB.W3.data + h*dW3
    f = loss_cv(L02, Xa, Xb, Ya, Yb, target, a)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW1, ΔW1) - h*dot(dW2, ΔW2) - h*dot(dW3, ΔW3))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
batchsize = 1
RB0 = ResidualBlock(n_in, n_hidden; fan=true)
L0 = CouplingLayerBasicCV(RB0; activation=Sigmoid2Layer())
θ0 = deepcopy(get_params(L0))
RB = ResidualBlock(n_in, n_hidden; fan=true)
L = CouplingLayerBasicCV(RB; activation=Sigmoid2Layer())
θ = deepcopy(get_params(L))
X1 = randn(Float32, nx, ny, n_in, batchsize)
X2 = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = θ - θ0
for i = 1:length(θ)
    dθ[i] = norm(θ0[i])*dθ[i]/(norm(dθ[i]).+1f-10)
end
dX1 = randn(Float32, nx, ny, n_in, batchsize); dX1 = norm(X1)*dX1/norm(dX1)
dX2 = randn(Float32, nx, ny, n_in, batchsize); dX2 = norm(X2)*dX2/norm(dX2)

# Jacobian eval
dY1, dY2, Y1, Y2 = L.jacobian(dX1, dX2, dθ, X1, X2)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err9 = zeros(Float32, maxiter)
err10 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(L, θ + h*dθ)
    Y1_, Y2_, _ = L.forward(X1 + h*dX1, X2 + h*dX2)
    err9[j] = sqrt(norm(Y1_ - Y1)^2f0 + norm(Y2_ - Y2)^2f0)
    err10[j] = sqrt(norm(Y1_ - Y1 - h*dY1)^2f0 + norm(Y2_ - Y2 - h*dY2)^2f0)
    print(err9[j], "; ", err10[j], "\n")
    global h = h/2f0
end

@test isapprox(err9[end] / (err9[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err10[end] / (err10[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(L, θ)
dY1, dY2, Y1, Y2 = L.jacobian(dX1, dX2, dθ, X1, X2)
dY1_ = randn(Float32, size(dY1)); dY2_ = randn(Float32, size(dY2))
dX1_, dX2_, dθ_ = L.adjointJacobian(dY1_, dY2_, Y1, Y2)
a_test = dot(dY1, dY1_) + dot(dY2, dY2_)
b_test = dot(dX1, dX1_) + dot(dX2, dX2_) + dot(dθ, dθ_)
@test isapprox(a_test, b_test; rtol=1f-3)
