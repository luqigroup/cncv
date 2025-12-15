# 1x1 convolution operator using Householder matrices (CV version - with jacobian trace and integrated gradients)
# Adapted from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# For Householder reflections, the Jacobian trace is 0 (orthogonal matrices have determinant ±1)

export Conv1x1CV

"""
    C = Conv1x1CV(k)

 or

    C = Conv1x1CV(v1, v2, v3)

 Create network layer for 1x1 convolutions using Householder reflections.

 *Input*:

 - `k`: number of channels

 - `v1`, `v2`, `v3`: Vectors from which to construct matrix.

 *Output*:

 - `C`: Network layer for 1x1 convolutions with Householder reflections.

 *Usage:*

 - Forward mode: `Y, jac_trace = C.forward(X)`

 - Backward mode: `ΔX, X = C.backward((ΔY, Y); jac_trace_grad_weight=nothing)`

 *Trainable parameters:*

 - Householder vectors `C.v1`, `C.v2`, `C.v3`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct Conv1x1CV <: NeuralNetLayer
    k::Integer
    v1::Parameter
    v2::Parameter
    v3::Parameter
    freeze::Bool
end

@Flux.functor Conv1x1CV

# Constructor with random initializations
function Conv1x1CV(k; freeze=false)
    v1 = Parameter(glorot_uniform(k))
    v2 = Parameter(glorot_uniform(k))
    v3 = Parameter(glorot_uniform(k))
    return Conv1x1CV(k, v1, v2, v3, freeze)
end

function Conv1x1CV(v1, v2, v3; freeze=false)
    k = length(v1)
    v1 = Parameter(v1)
    v2 = Parameter(v2)
    v3 = Parameter(v3)
    return Conv1x1CV(k, v1, v2, v3, freeze)
end

## Jacobian trace computation helpers
# Compute k×k Householder matrix: H = I - 2vv^T/(v^Tv)
function householder_matrix(v::AbstractVector{T}) where T
    k = length(v)
    vv = v * v'
    vnorm_sq = v' * v
    return Matrix{T}(I, k, k) - 2 * vv / vnorm_sq
end

# Compute trace of product of three Householder matrices
function compute_householder_trace(v1::AbstractVector{T}, v2::AbstractVector{T}, v3::AbstractVector{T}) where T
    H1 = householder_matrix(v1)
    H2 = householder_matrix(v2)
    H3 = householder_matrix(v3)
    W = H1 * H2 * H3
    return sum(diag(W))
end

## Jacobian trace gradient computation
# For Householder products, trace IS parameter-dependent
# Note: Computing exact gradients would be expensive (requires differentiating matrix products)
# For now, we use finite differences for accuracy
function jac_trace_grad!(C::Conv1x1CV, X::AbstractArray{T, N}) where {T, N}
    k = C.k
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    # Use finite differences to compute gradient
    ε = T(1e-5)
    ∇v1_jac_trace = zeros(T, k)
    ∇v2_jac_trace = zeros(T, k)
    ∇v3_jac_trace = zeros(T, k)

    trace_0 = compute_householder_trace(v1, v2, v3)

    # Gradient w.r.t. v1
    for i = 1:k
        v1_pert = copy(v1)
        v1_pert[i] += ε
        trace_pert = compute_householder_trace(v1_pert, v2, v3)
        ∇v1_jac_trace[i] = (trace_pert - trace_0) / ε
    end

    # Gradient w.r.t. v2
    for i = 1:k
        v2_pert = copy(v2)
        v2_pert[i] += ε
        trace_pert = compute_householder_trace(v1, v2_pert, v3)
        ∇v2_jac_trace[i] = (trace_pert - trace_0) / ε
    end

    # Gradient w.r.t. v3
    for i = 1:k
        v3_pert = copy(v3)
        v3_pert[i] += ε
        trace_pert = compute_householder_trace(v1, v2, v3_pert)
        ∇v3_jac_trace[i] = (trace_pert - trace_0) / ε
    end

    return ∇v1_jac_trace, ∇v2_jac_trace, ∇v3_jac_trace
end


function conv1x1_grad_v(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N},
                        C::Conv1x1CV; adjoint=false, jac_trace_grad_weight::Union{Nothing, AbstractVector{T}}=nothing) where {T, N}

    # Reshape input
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)

    dv1 = cuzeros(X, k)
    dv2 = cuzeros(X, k)
    dv3 = cuzeros(X, k)

    # Do not calculate gradients if layer is frozen
    if C.freeze
        return dv1, dv2, dv3
    end

    V1 = v1*v1'/(v1'*v1)
    V2 = v2*v2'/(v2'*v2)
    V3 = v3*v3'/(v3'*v3)

    dV1 = partial_derivative_outer(v1)
    dV2 = partial_derivative_outer(v2)
    dV3 = partial_derivative_outer(v3)

    M1 = (I - 2 * (V2 + V3) + 4*V2*V3)
    M3 = (I - 2 * (V1 + V2) + 4*V1*V2)
    tmp = cuzeros(X, k, k)
    for i=1:k
        # dV1
        mul!(tmp, dV1[i, :, :], M1)
        @views adjoint ? copyto!(dV1[i, :, :], tmp') : copyto!(dV1[i, :, :], tmp)
        # dV2
        v2 = dV2[i, :, :]
        broadcast!(+, tmp, v2, 4 * V1 * v2 * V3 - 2 * (V1 * v2 + v2 * V3))
        @views adjoint ? copyto!(dV2[i, :, :], tmp') : copyto!(dV2[i, :, :], tmp)
        # dV3
        mul!(tmp, M3, dV3[i, :, :])
        @views adjoint ? copyto!(dV3[i, :, :], tmp') : copyto!(dV3[i, :, :], tmp)
    end

    n_in, batchsize = size(X)[N-1:N]
    prod_res = cuzeros(X, size(dV1, 1))
    for i=1:batchsize
        Xi = -2f0*reshape(selectdim(X, N, i), :, n_in)
        ΔYi = reshape(selectdim(ΔY, N, i), :, n_in)
        broadcast!(+, dv1, dv1, mat_tens_i(prod_res, Xi, dV1, ΔYi))
        broadcast!(+, dv2, dv2, mat_tens_i(prod_res, Xi, dV2, ΔYi))
        broadcast!(+, dv3, dv3, mat_tens_i(prod_res, Xi, dV3, ΔYi))
    end

    # Add jacobian trace gradient if provided
    if !isnothing(jac_trace_grad_weight)
        ∇v1_trace, ∇v2_trace, ∇v3_trace = jac_trace_grad!(C, X)
        # Scale by spatial size (nx * ny)
        spatial_size = prod(size(X)[1:N-2])
        trace_weight_sum = sum(jac_trace_grad_weight)
        dv1 += T(spatial_size) * trace_weight_sum * ∇v1_trace
        dv2 += T(spatial_size) * trace_weight_sum * ∇v2_trace
        dv3 += T(spatial_size) * trace_weight_sum * ∇v3_trace
    end

    return dv1, dv2, dv3
end

# Forward pass
# Compute Jacobian trace: trace(J) = nx * ny * trace(H1 * H2 * H3)
function forward(X::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    Y = cuzeros(X, size(X)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i))...)
    end

    # Compute Jacobian trace: trace(J) = nx * ny * trace(H1 * H2 * H3)
    # The same k×k matrix W is applied at each spatial location
    spatial_size = prod(size(X)[1:N-2])  # nx * ny (or nx * ny * nz for 3D)
    trace_W = compute_householder_trace(v1, v2, v3)
    jac_trace_per_sample = T(spatial_size) * trace_W

    batchsize = size(X, N)
    jac_trace_batch = fill(jac_trace_per_sample, batchsize)

    return Y, jac_trace_batch
end

# Forward pass and update weights
function forward(X_tuple::Tuple, C::Conv1x1CV; set_grad::Bool=true, jac_trace_grad_weight::Union{Nothing, AbstractVector}=nothing)
    ΔX = X_tuple[1]
    X = X_tuple[2]
    ΔY, _ = forward(ΔX, C)    # forward propagate residual
    Y, jac_trace = forward(X, C)  # recompute forward state

    T = eltype(X)
    jtgw = isnothing(jac_trace_grad_weight) ? nothing : convert(AbstractVector{T}, jac_trace_grad_weight)
    Δv1, Δv2, Δv3 = conv1x1_grad_v(Y, ΔX, C; adjoint=true, jac_trace_grad_weight=jtgw)

    if set_grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    else
        Δθ = [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
    end
    set_grad ? (return ΔY, Y, jac_trace) : (return ΔY, Δθ, Y, jac_trace)
end

# Inverse pass
function inverse(Y::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    X = cuzeros(Y, size(Y)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(Y, N)
        Yi = reshape(selectdim(Y, N, i), :, n_in)
        Xi = chain_lr(Yi, v3, v2, v1)
        selectdim(X, N, i) .= reshape(Xi, size(selectdim(X, N, i))...)
    end

    # Compute Jacobian trace: trace(J) = nx * ny * trace(H1 * H2 * H3)
    # For inverse, the transformation is H3 * H2 * H1, which has the same trace
    spatial_size = prod(size(Y)[1:N-2])  # nx * ny (or nx * ny * nz for 3D)
    trace_W = compute_householder_trace(v1, v2, v3)
    jac_trace_per_sample = T(spatial_size) * trace_W

    batchsize = size(Y, N)
    jac_trace_batch = fill(jac_trace_per_sample, batchsize)

    return X, jac_trace_batch
end

# Inverse pass and update weights
function inverse(Y_tuple::Tuple, C::Conv1x1CV; set_grad::Bool=true, jac_trace_grad_weight::Union{Nothing, AbstractVector}=nothing)
    ΔY = Y_tuple[1]
    Y = Y_tuple[2]
    ΔX, _ = inverse(ΔY, C)    # derivative w.r.t. input
    X, jac_trace = inverse(Y, C)  # recompute forward state

    # Gradient w.r.t. weights
    T = eltype(Y)
    jtgw = isnothing(jac_trace_grad_weight) ? nothing : convert(AbstractVector{T}, jac_trace_grad_weight)
    Δv1, Δv2, Δv3 = conv1x1_grad_v(X, ΔY, C; jac_trace_grad_weight=jtgw)

    if set_grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    else
        Δθ = [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
    end

    set_grad ? (return ΔX, X, jac_trace) : (return ΔX, Δθ, X, jac_trace)
end

# Jacobian-related functions
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    Y = cuzeros(X, size(X)...)
    ΔY = cuzeros(ΔX, size(ΔX)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    dv1 = Δθ[1].data
    dv2 = Δθ[2].data
    dv3 = Δθ[3].data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        isa(X, CUDA.CuArray) && (Xi = CUDA.CuArray(Xi))
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i))...)

        ΔXi = reshape(selectdim(ΔX, N, i), :, n_in)
        ΔYi = chain_lr(ΔXi, v1, v2, v3)
        n1 = norm(v1); n2 = norm(v2); n3 = norm(v3);
        c1 = I - 2f0*v1*v1'/n1^2f0; c2 = I - 2f0*v2*v2'/n2^2f0; c3 = I - 2f0*v3*v3'/n3^2f0;
        ΔYi += -2f0*Xi*((dv1*v1'+v1*dv1'-2f0*dot(v1,dv1)*v1*v1'/n1^2f0)/n1^2f0*c2*c3+
                       c1*(dv2*v2'+v2*dv2'-2f0*dot(v2,dv2)*v2*v2'/n2^2f0)/n2^2f0*c3+
                       c1*c2*(dv3*v3'+v3*dv3'-2f0*dot(v3,dv3)*v3*v3'/n3^2f0)/n3^2f0)
        selectdim(ΔY, N, i) .= reshape(ΔYi, size(selectdim(ΔY, N, i))...)
    end

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    return inverse((ΔY, Y), C; set_grad=false)
end

function jacobianInverse(ΔY::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, Y::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    return inverse(C).jacobian(ΔY, Δθ[end:-1:1], Y)
end

function adjointJacobianInverse(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1CV) where {T, N}
    ΔX, Δθinv, X, _ = inverse(C).adjointJacobian(ΔX, X)
    return ΔX, Δθinv[end:-1:1], X
end

function inverse(C::Conv1x1CV)
    return Conv1x1CV(C.k, C.v3, C.v2, C.v1, C.freeze)
end