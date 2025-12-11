# Affine coupling layer from Dinh et al. (2017) - CV version (with jacobian trace and gradients)
# Adapted for compressed sensing applications
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerBasicCV, CouplingLayerBasicCV3D


"""
    CL = CouplingLayerBasicCV(RB::ResidualBlock)

or

    CL = CouplingLayerBasicCV(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2) (2D)

    CL = CouplingLayerBasicCV(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=3) (3D)

    CL = CouplingLayerBasicCV3D(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (3D)

 Create a Real NVP-style invertible coupling layer with a residual block (CV version).

 *Input*:

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s1`)

 - `ndims` : Number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y1, Y2, jac_trace = CL.forward(X1, X2)`

 - Inverse mode: `X1, X2, jac_trace = CL.inverse(Y1, Y2)`

 - Backward mode: `ΔX1, ΔX2, X1, X2 = CL.backward(ΔY1, ΔY2, Y1, Y2)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB`

 See also: [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct CouplingLayerBasicCV <: NeuralNetLayer
    RB::Union{ResidualBlock, FluxBlock}
    activation::ActivationFunction
    is_reversed::Bool
end

@Flux.functor CouplingLayerBasicCV

# Constructor from residual block
function CouplingLayerBasicCV(RB::ResidualBlock; activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerBasicCV(RB, activation, false)
end

CouplingLayerBasicCV(RB::FluxBlock; activation::ActivationFunction=SigmoidLayer()) = CouplingLayerBasicCV(RB, activation, false)

# 2D Constructor from input dimensions
function CouplingLayerBasicCV(n_in::Int64, n_hidden::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # Residual block for invertible layer
    RB = ResidualBlock(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)

    return CouplingLayerBasicCV(RB, activation, false)
end

CouplingLayerBasicCV3D(args...;kw...) = CouplingLayerBasicCV(args...; kw..., ndims=3)

## Jacobian trace computation
# For coupling layer: jac_trace = sum(S)
coupling_jac_trace_forward(S) = dropdims(sum(S; dims=tuple(1:ndims(S)-1...)); dims=tuple(1:ndims(S)-1...))

## Jacobian trace gradient computation
# Gradient of jac_trace w.r.t. S: d(sum(S))/dS = 1 for all elements
coupling_jac_trace_backward(S) = ones(eltype(S), size(S)) ./ size(S)[end]

# 2D/3D Forward pass: Input X, Output Y
function forward(X1::AbstractArray{T, N}, X2::AbstractArray{T, N}, L::CouplingLayerBasicCV; save::Bool=false) where {T, N}

    # Coupling layer
    logS_T1, logS_T2 = tensor_split(L.RB.forward(X1))
    S = L.activation.forward(logS_T1)
    Y2 = S.*X2 + logS_T2

    # Compute jacobian trace per batch element
    jac_trace_batch = coupling_jac_trace_forward(S)

    save ? (return X1, Y2, jac_trace_batch, logS_T1, S) : (return X1, Y2, jac_trace_batch)
end

# 2D/3D Inverse pass: Input Y, Output X
function inverse(Y1::AbstractArray{T, N}, Y2::AbstractArray{T, N}, L::CouplingLayerBasicCV; save::Bool=false) where {T, N}

    # Inverse layer
    logS_T1, logS_T2 = tensor_split(L.RB.forward(Y1))
    S = L.activation.forward(logS_T1)
    X2 = (Y2 - logS_T2) ./ (S .+ eps(T)) # add epsilon to avoid division by 0

    # Compute jacobian trace per batch element
    jac_trace_batch = coupling_jac_trace_forward(S)

    save == true ? (return Y1, X2, jac_trace_batch, logS_T1, S) : (return Y1, X2, jac_trace_batch)
end

# 2D/3D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY1::AbstractArray{T, N}, ΔY2::AbstractArray{T, N}, Y1::AbstractArray{T, N}, Y2::AbstractArray{T, N}, L::CouplingLayerBasicCV; set_grad::Bool=true) where {T, N}

    # Recompute forward state
    X1, X2, _, logS_T1, S = inverse(Y1, Y2, L; save=true)

    # Backpropagate residual
    ΔT = copy(ΔY2)
    ΔS = ΔY2 .* X2
    ΔX2 = ΔY2 .* S

    if set_grad
        ΔX1 = L.RB.backward(tensor_cat(backward(ΔS, logS_T1, S, L.activation), ΔT), X1) + ΔY1
    else
        ΔX1, Δθ = L.RB.backward(tensor_cat(backward(ΔS, logS_T1, S, L.activation), ΔT), X1; set_grad=set_grad)
        ΔX1 += ΔY1
    end

    if set_grad
        return ΔX1, ΔX2, X1, X2
    else
        return ΔX1, ΔX2, Δθ, X1, X2
    end
end

# Compute gradient of jacobian trace and return it
# This computes ∇_θ trace(J) where θ are the parameters of the residual block
function jac_trace_grad(X1::AbstractArray{T, N}, L::CouplingLayerBasicCV) where {T, N}
    # Forward through RB to get S
    logS_T1, _ = tensor_split(L.RB.forward(X1))
    S = L.activation.forward(logS_T1)

    # Gradient of trace w.r.t. S
    ∇S_trace = coupling_jac_trace_backward(S)

    # Backpropagate through activation and RB to get gradient w.r.t. parameters
    ∇logS = backward(∇S_trace, logS_T1, S, L.activation)
    _, Δθ = L.RB.backward(tensor_cat(∇logS, zero(∇logS)), X1; set_grad=false)

    return Δθ
end

# 2D/3D Reverse backward pass: Input (ΔX, X), Output (ΔY, Y)
function backward_inv(ΔX1::AbstractArray{T, N}, ΔX2::AbstractArray{T, N}, X1::AbstractArray{T, N}, X2::AbstractArray{T, N}, L::CouplingLayerBasicCV; set_grad::Bool=true) where {T, N}

    # Recompute inverse state
    Y1, Y2, _, logS_T1, S = forward(X1, X2, L; save=true)

    # Backpropagate residual
    ΔT = -ΔX2 ./ S
    ΔS = X2 .* ΔT