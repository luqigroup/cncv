# File: src/models/utils/activation_functions.jl
# Only extend backward for ActivationFunction - use InvertibleNetworks' type and functions

using InvertibleNetworks: ActivationFunction

# Extend backward for ActivationFunction (not exported by InvertibleNetworks)
function backward(Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}, activation::ActivationFunction) where {T, N}
    backward_activation(activation.backward, activation.inverse, Δy, x, y)
end

function backward_activation(back::Function, inverse::Nothing, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    back(Δy, x)
end

function backward_activation(back::Function, inverse::Function, Δy::AbstractArray{T, N}, x::AbstractArray{T, N}, y::AbstractArray{T, N}) where {T, N}
    back(Δy, y)
end