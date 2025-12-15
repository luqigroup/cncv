# Test activation normalization layer (CV version - with jacobian trace)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, get_grads, clear_grad!, set_params!

using CNCV

Random.seed!(11)
###############################################################################
# Test jac_trace implementation

# Input
nx = 4
ny = 4
nc = 3
batchsize = 1
X = rand(Float32, nx, ny, nc, batchsize)

# ActNormCV and initialize
AN = ActNormCV(nc)
AN.forward(X)

# Explicitely compute jacobian trace through probing
# for small number of dimensions
J = zeros(Float32, Int(nx*ny*nc), Int(nx*ny*nc))
for i=1:nc
    count = 1
    for j=1:nx
        for k=1:ny
            E = zeros(Float32, nx, ny, nc, 1)
            E[k, j, i] = 1f0
            local Y = AN.forward(X)[1]
            # Use inverse to get backward direction
            X_back, _ = AN.inverse(Y)
            ΔX, _ = AN.backward(E, Y)
            J[:, (i-1)*nx*ny + count] = vec(ΔX)
            count += 1
        end
    end
end
jac_trace1 = sum(diag(J))
jac_trace2 = AN.forward(X)[2][1]  # Get first batch element
@test isapprox((jac_trace1 - jac_trace2)/jac_trace1, 0f0; atol=1f-6)


###############################################################################
# Test jac_trace per batch element

# Input with larger batchsize
nx = 4
ny = 4
nc = 3
batchsize = 5
X = rand(Float32, nx, ny, nc, batchsize)

# ActNormCV and initialize
AN = ActNormCV(nc)
AN.forward(X)

# Test jac_trace returns vector with one element per batch
Y, jac_trace_vec = AN.forward(X)
@test jac_trace_vec isa AbstractArray
@test length(jac_trace_vec) == batchsize
# All batch elements should have same trace (since transformation is same for all)
@test all(jac_trace_vec .≈ jac_trace_vec[1])

# Test inverse jac_trace
X_inv, jac_trace_inv_vec = AN.inverse(Y)
@test jac_trace_inv_vec isa AbstractArray
@test length(jac_trace_inv_vec) == batchsize
# Inverse should have same trace as forward (not negated like logdet)
@test all(jac_trace_inv_vec .≈ jac_trace_vec[1])


###############################################################################
# Initialization and invertibility

# Input
nx = 28
ny = 28
nc = 4
batchsize = 1
X = rand(Float32, nx, ny, nc, batchsize)

# Layer and initialization
AN = ActNormCV(nc)
Y, _ = AN.forward(X)

# Test initialization
@test isapprox(mean(Y), 0f0; atol=1f-6)
@test isapprox(var(Y), 1f0; atol=1f-3)

# Test invertibility
@test isapprox(norm(X - AN.inverse(AN.forward(X)[1])[1])/norm(X), 0f0, atol=1f-6)
@test isapprox(norm(X - AN.forward(AN.inverse(X)[1])[1])/norm(X), 0f0, atol=1f-6)

# Test with multiple batches
batchsize = 3
X = rand(Float32, nx, ny, nc, batchsize)
AN = ActNormCV(nc)

Y, jac_trace_vec = AN.forward(X)
X_rec, _ = AN.inverse(Y)
@test isapprox(norm(X - X_rec)/norm(X), 0f0, atol=1f-6)
@test isequal(size(jac_trace_vec), (batchsize,))


###############################################################################
# Gradient Test with jac_trace_grad_weight

AN = ActNormCV(nc)
batchsize = 2
X = randn(Float32, nx, ny, nc, batchsize)
X0 = randn(Float32, nx, ny, nc, batchsize)
dX = X - X0

# Forward pass
Y, _ = AN.forward(X)

# Target for CV objective
target = randn(Float32, batchsize)
a = randn(Float32, size(Y))

function loss_cv(AN, X, Y, target, a)
    # Forward pass
    Y_, jac_trace = AN.forward(X)

    # Compute residual: target - jac_trace - a'*Y
    residual = target .- jac_trace .- vec(sum(a .* Y_, dims=[1,2,3]))

    # Objective: ||residual||^2
    f = .5f0/batchsize * sum(residual.^2)

    # Gradient weight for jacobian trace
    jac_trace_grad_weight = -residual ./ Float32(batchsize)

    # Data gradient: ΔY = -residual/batchsize * a
    ΔY = zeros(Float32, size(Y_))
    for i=1:batchsize
        selectdim(ΔY, 4, i) .= -residual[i] / Float32(batchsize) .* selectdim(a, 4, i)
    end

    # Back propagation with jac_trace_grad_weight
    ΔX, X_ = AN.backward(ΔY, Y_; jac_trace_grad_weight=jac_trace_grad_weight)

    # Check invertibility
    @test isapprox(norm(X - X_)/norm(X), 0f0, atol=1f-6)

    return f, ΔX, get_grads(AN)
end

# Gradient test for X
maxiter = 5  # Reduced to avoid numerical precision issues
print("\nGradient test actnorm_cv (X)\n")
f0, ΔX = loss_cv(AN, X0, Y, target, a)[1:2]
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss_cv(AN, X0 + h*dX, Y, target, a)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for parameters
AN0 = ActNormCV(nc); AN0.forward(randn(Float32, nx, ny, nc, batchsize))
AN_ini = deepcopy(AN0)
θ = get_params(AN_ini)
dθ = get_params(AN) - get_params(AN0)
maxiter = 6
print("\nGradient test actnorm_cv (params)\n")
f0, ΔX, Δθ = loss_cv(AN0, X, Y, target, a)
h = 1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AN0, θ + h*dθ)
    f = loss_cv(AN0, X, Y, target, a)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)


###############################################################################
# Gradient Test with backward_inv

AN = ActNormCV(nc)
batchsize = 2
# Initialize AN with a forward pass
X_init = randn(Float32, nx, ny, nc, batchsize)
AN.forward(X_init)

Y = randn(Float32, nx, ny, nc, batchsize)
Y0 = randn(Float32, nx, ny, nc, batchsize)
dY = Y - Y0

# Target for CV objective
target = randn(Float32, batchsize)
a = randn(Float32, size(Y))

function loss_cv_inv(AN, Y, X_target, target, a)
    # Inverse pass
    X_, jac_trace = AN.inverse(Y)

    # Compute residual
    residual = target .- jac_trace .- vec(sum(a .* X_, dims=[1,2,3]))

    # Objective
    f = .5f0/batchsize * sum(residual.^2)

    # Gradient weight for jacobian trace
    jac_trace_grad_weight = -residual ./ Float32(batchsize)

    # Data gradient
    ΔX = zeros(Float32, size(X_))
    for i=1:batchsize
        selectdim(ΔX, 4, i) .= -residual[i] / Float32(batchsize) .* selectdim(a, 4, i)
    end

    # Backward_inv pass
    ΔY, Y_ = AN.backward_inv(ΔX, X_; jac_trace_grad_weight=jac_trace_grad_weight)

    return f, ΔY, get_grads(AN)
end

# Forward to get X target
X_target, _ = AN.inverse(Y)

# Gradient test for Y
maxiter = 5  # Reduced to avoid numerical precision issues
print("\nGradient test actnorm_cv backward_inv (Y)\n")
f0, ΔY = loss_cv_inv(AN, Y0, X_target, target, a)[1:2]
h = .1f0
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss_cv_inv(AN, Y0 + h*dY, X_target, target, a)[1]
    err5[j] = abs(f - f0)
    err6[j] = abs(f - f0 - h*dot(dY, ΔY))
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)


###################################################################################################
# Jacobian-related tests

# Gradient test

# Initialization
AN = ActNormCV(nc)
batchsize = 1
AN.forward(randn(Float32, nx, ny, nc, batchsize))
θ = deepcopy(get_params(AN))
AN0 = ActNormCV(nc); AN0.forward(randn(Float32, nx, ny, nc, batchsize))
θ0 = deepcopy(get_params(AN0))
X = randn(Float32, nx, ny, nc, batchsize)

# Perturbation (normalized)
dθ = θ - θ0
for i = 1:length(θ)
    dθ[i] = norm(θ0[i])*dθ[i]/(norm(dθ[i]).+1f-10)
end
dX = randn(Float32, nx, ny, nc, batchsize); dX *= norm(X)/norm(dX)

# Jacobian eval
dY, Y = AN.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err9 = zeros(Float32, maxiter)
err10 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(AN, θ + h*dθ)
    Y_loc, _ = AN.forward(X + h*dX)
    err9[j] = norm(Y_loc - Y)
    err10[j] = norm(Y_loc - Y - h*dY)
    print(err9[j], "; ", err10[j], "\n")
    global h = h/2f0
end

@test isapprox(err9[end] / (err9[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err10[end] / (err10[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(AN, θ)
dY, Y = AN.jacobian(dX, dθ, X)
dY_ = randn(Float32, size(dY))
dX_, dθ_, _ = AN.adjointJacobian(dY_, Y)
a_test = dot(dY, dY_)
b_test = dot(dX, dX_) + dot(dθ, dθ_)
@test isapprox(a_test, b_test; rtol=1f-3)
