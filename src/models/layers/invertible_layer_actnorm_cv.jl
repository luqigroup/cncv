# Activation normalization layer (CV version - no logdet)
# Adapted for compressed sensing applications
# Based on original ActNorm from Kingma and Dhariwal (2018)

export ActNormCV, reset!

"""
    AN = ActNormCV(k)

 Create activation normalization layer. The parameters are initialized during
 the first use, such that the output has zero mean and unit variance along
 channels for the current mini-batch size.

 *Input*:

 - `k`: number of channels

 *Output*:

 - `AN`: Network layer for activation normalization.

 *Usage:*

 - Forward mode: `Y = AN.forward(X)`

 - Inverse mode: `X = AN.inverse(Y)`

 - Backward mode: `ΔX, X = AN.backward(ΔY, Y)`

 *Trainable parameters:*

 - Scaling factor `AN.s`

 - Bias `AN.b`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct ActNormCV <: NeuralNetLayer
    k::Integer
    s::Parameter
    b::Parameter
    is_reversed::Bool
end

@Flux.functor ActNormCV

# Constructor: Initialize with nothing
function ActNormCV(k)
    s = Parameter(nothing)
    b = Parameter(nothing)
    return ActNormCV(k, s, b, false)
end

# 2-3D Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, AN::ActNormCV) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s.data) && !AN.is_reversed
        μ = mean(X; dims=dims)[inds...]
        σ_sqr = var(X; dims=dims)[inds...]
        AN.s.data = 1 ./ sqrt.(σ_sqr)
        AN.b.data = -μ ./ sqrt.(σ_sqr)
    end
    Y = X .* reshape(AN.s.data, inds...) .+ reshape(AN.b.data, inds...)

    return Y
end

# 2-3D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, AN::ActNormCV) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s.data) && AN.is_reversed
        μ = mean(Y; dims=dims)[inds...]
        σ_sqr = var(Y; dims=dims)[inds...]
        AN.s.data = sqrt.(σ_sqr)
        AN.b.data = μ
    end
    X = (Y .- reshape(AN.b.data, inds...)) ./ reshape(AN.s.data, inds...)

    return X
end

# 2-3D Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AN::ActNormCV; set_grad::Bool = true) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    X = inverse(Y, AN)
    ΔX = ΔY .* reshape(AN.s.data, inds...)
    Δs = sum(ΔY .* X, dims=dims)[inds...]
    Δb = sum(ΔY, dims=dims)[inds...]

    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
        return ΔX, X
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
        return ΔX, Δθ, X
    end
end

# 2-3D Backward pass (inverse): Input (ΔX, X), Output (ΔY, Y)
function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, AN::ActNormCV; set_grad::Bool = true) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    Y = forward(X, AN)
    ΔY = ΔX ./ reshape(AN.s.data, inds...)
    Δs = -sum(ΔX .* X ./ reshape(AN.s.data, inds...), dims=dims)[inds...]
    Δb = -sum(ΔX ./ reshape(AN.s.data, inds...), dims=dims)[inds...]

    if set_grad
        AN.s.grad = Δs
        AN.b.grad = Δb
        return ΔY, Y
    else
        Δθ = [Parameter(Δs), Parameter(Δb)]
        return ΔY, Δθ, Y
    end
end

# Jacobian-related functions
function jacobian(ΔX::AbstractArray{T, N}, Δθ::AbstractArray{Parameter, 1}, X::AbstractArray{T, N}, AN::ActNormCV) where {T, N}
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    Δs = Δθ[1].data
    Δb = Δθ[2].data

    # Forward evaluation
    Y = forward(X, AN)

    # Jacobian evaluation
    ΔY = ΔX .* reshape(AN.s.data, inds...) .+ X .* reshape(Δs, inds...) .+ reshape(Δb, inds...)

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AN::ActNormCV) where {T, N}
    return backward(ΔY, Y, AN; set_grad=false)
end

# Reverse
function tag_as_reversed!(AN::ActNormCV, tag::Bool)
    AN.is_reversed = tag
    return AN
end