# Conditional coupling layer (basic version) - CV version (with jacobian trace)
# Adapted for compressed sensing applications
# Date: January 2025

export ConditionalLayerBasicCV, ConditionalLayerBasicCV3D


"""
    CL = ConditionalLayerBasicCV(RB::ResidualBlock)

or

    CL = ConditionalLayerBasicCV(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2) (2D)

    CL = ConditionalLayerBasicCV(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=3) (3D)

    CL = ConditionalLayerBasicCV3D(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1) (3D)

 Create a Real NVP-style invertible conditional coupling layer based on a residual block (CV version).

 *Input*:

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 or

 - `n_in`: number of input channels

 - `n_cond`: number of conditioning channels

 - `n_hidden`: number of hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP conditional coupling layer.

 *Usage:*

 - Forward mode: `Y, jac_trace = CL.forward(X, C)`

 - Inverse mode: `X, jac_trace = CL.inverse(Y, C)`

 - Backward mode: `ΔX, X, ΔC = CL.backward(ΔY, Y, C; jac_trace_grad_weight=nothing)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB`

 See also: [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerBasicCV <: NeuralNetLayer
    RB::ResidualBlock
    activation::ActivationFunction
end

@Flux.functor ConditionalLayerBasicCV

# Constructor from residual block
function ConditionalLayerBasicCV(RB::ResidualBlock; activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return ConditionalLayerBasicCV(RB, activation)
end

# Constructor from input dimensions
function ConditionalLayerBasicCV(n_in::Int64, n_cond::Int64, n_hidden::Int64; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, activation::ActivationFunction=SigmoidLayer(), rb_activation::ActivationFunction=ReLUlayer(), ndims=2)

    split_num = Int(round(n_in/2))
    in_split   = n_in-split_num
    out_chan  = 2*split_num

    RB = ResidualBlock(in_split+n_cond, n_hidden; n_out=out_chan, activation=rb_activation, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)

    return ConditionalLayerBasicCV(RB, activation)
end

ConditionalLayerBasicCV3D(args...;kw...) = ConditionalLayerBasicCV(args...; kw..., ndims=3)

## Jacobian trace computation
# For conditional coupling layer: jac_trace = trace(I) + sum(S)
# where trace(I) = nx * ny * n_channels (from the identity part Y2 = X2)
function conditional_coupling_jac_trace_forward(X2, S)
    # Sum over spatial and channel dimensions for each batch element
    trace_S = dropdims(sum(S; dims=tuple(1:ndims(S)-1...)); dims=tuple(1:ndims(S)-1...))
    # Add identity contribution from X2: nx * ny * n_channels of X2 which passes through unchanged
    identity_contribution = prod(size(X2)[1:end-1])  # nx * ny * n_channels of X2
    return trace_S .+ Float32(identity_contribution)
end

## Jacobian trace gradient computation
# Gradient of jac_trace w.r.t. S: d(sum(S))/dS = 1 for all elements
conditional_coupling_jac_trace_backward(S) = ones(eltype(S), size(S))

# Forward pass: Input X and C (conditioning), Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerBasicCV) where {T,N}

    X1, X2 = tensor_split(X)

    Y2 = copy(X2)

    # Cat conditioning variable C into network input
    logS_T = L.RB.forward(tensor_cat(X2, C))
    logS, log_T = tensor_split(logS_T)

    S = L.activation.forward(logS)
    Tm = log_T
    Y1 = S.*X1 + Tm

    Y = tensor_cat(Y1, Y2)

    # Compute jacobian trace
    jac_trace_batch = conditional_coupling_jac_trace_forward(X2, S)

    return Y, jac_trace_batch
end

# Inverse pass: Input Y and C (conditioning), Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerBasicCV; save=false) where {T,N}

    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(tensor_cat(X2, C))
    logS, log_T = tensor_split(logS_T)

    S = L.activation.forward(logS)
    Tm = log_T
    X1 = (Y1 - Tm) ./ (S .+ eps(eltype(S))) # add epsilon to avoid division by 0

    X = tensor_cat(X1, X2)

    # Compute jacobian trace
    jac_trace_batch = conditional_coupling_jac_trace_forward(X2, S)

    save == true ? (return X, jac_trace_batch, X1, X2, logS, S) : (return X, jac_trace_batch)
end

# Backward pass: Input (ΔY, Y, C), Output (ΔX, X, ΔC)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerBasicCV;
                  jac_trace_grad_weight::Union{Nothing, AbstractVector{T}}=nothing) where {T,N}

    # Recompute forward state
    X, _, X1, X2, logS, S = inverse(Y, C, L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1

    # Add jacobian trace gradient if provided
    if !isnothing(jac_trace_grad_weight)
        ∇S_trace = conditional_coupling_jac_trace_backward(S)
        batchsize = size(S)[end]
        weight_expanded = reshape(jac_trace_grad_weight, ntuple(i -> i == ndims(S) ? batchsize : 1, ndims(S))...)
        ΔS += weight_expanded .* ∇S_trace
    end

    ΔX1 = ΔY1 .* S

    # Backpropagate RB
    ΔX2_ΔC = L.RB.backward(tensor_cat(backward(ΔS, logS, S, L.activation), ΔT), (tensor_cat(X2, C)))
    ΔX2, ΔC = tensor_split(ΔX2_ΔC; split_index=size(ΔY2)[N-1])
    ΔX2 += ΔY2

    ΔX = tensor_cat(ΔX1, ΔX2)

    return ΔX, X, ΔC
end

# Compute gradient of jacobian trace and return it
# This computes ∇_θ trace(J) where θ are the parameters
function jac_trace_grad(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::ConditionalLayerBasicCV) where {T,N}
    X1, X2 = tensor_split(X)

    # Forward through RB to get S
    logS_T = L.RB.forward(tensor_cat(X2, C))
    logS, _ = tensor_split(logS_T)
    S = L.activation.forward(logS)

    # Gradient of trace w.r.t. S
    ∇S_trace = conditional_coupling_jac_trace_backward(S)

    # Backpropagate through activation and RB to get gradient w.r.t. RB parameters
    ∇logS = backward(∇S_trace, logS, S, L.activation)
    ΔX2_ΔC = L.RB.backward(tensor_cat(∇logS, zero(∇logS)), tensor_cat(X2, C); set_grad=false)
    # We only care about parameter gradients, not input gradients for trace

    # Note: This would need proper extraction of RB parameter gradients
    # For now, return nothing as placeholder
    return nothing
end
