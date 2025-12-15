# Dense conditional layer for control variates
# Designed for low-dimensional data (not images)
# Date: December 2025

export DenseConditionalLayerCV

# Import what we need to extend
import InvertibleNetworks: Parameter, glorot_uniform, get_params, clear_grad!
import InvertibleNetworks: forward, inverse, backward

"""
    CL = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers=3; activation=tanh)

Create a dense (MLP-based) conditional coupling layer for control variates.
Designed for low-dimensional data where spatial structure is not present.

*Input*:

- `n_in`: number of input dimensions
- `n_cond`: number of conditioning dimensions
- `n_hidden`: number of hidden units in MLP
- `n_layers`: number of hidden layers (default 3)
- `activation`: activation function (default tanh)

*Output*:

- `CL`: Dense conditional coupling layer

*Usage:*

- Forward mode: `Y, jac_trace = forward(X, C, CL)` where X is n_in×batch, C is n_cond×batch
- Inverse mode: `X, jac_trace = inverse(Y, C, CL)`
- Backward mode: `ΔX, X, ΔC = backward(ΔY, Y, C, CL; jac_trace_grad_weight=nothing)`

The layer implements a coupling transformation with tractable Jacobian.
"""
struct DenseConditionalLayerCV
    split_idx::Int  # Where to split input
    W1::Parameter   # First layer weights
    b1::Parameter   # First layer bias
    W_hidden::Vector{Parameter}  # Hidden layer weights
    b_hidden::Vector{Parameter}  # Hidden layer biases
    W_out::Parameter  # Output layer weights
    b_out::Parameter  # Output layer bias
    activation::Function
end

function DenseConditionalLayerCV(n_in::Int, n_cond::Int, n_hidden::Int, n_layers::Int=3; activation=tanh)
    split_idx = div(n_in, 2)
    n_in2 = n_in - split_idx  # Size of X2 (pass-through part)
    n_out = 2 * split_idx  # Output S and T, each of size split_idx

    # Build MLP: input is [X2; C], output is [S; T]
    input_dim = n_in2 + n_cond

    # First layer: input_dim -> n_hidden
    W1 = Parameter(glorot_uniform(n_hidden, input_dim))
    b1 = Parameter(zeros(Float32, n_hidden, 1))

    # Hidden layers: n_hidden -> n_hidden
    W_hidden = Parameter[]
    b_hidden = Parameter[]
    for i in 2:n_layers
        push!(W_hidden, Parameter(glorot_uniform(n_hidden, n_hidden)))
        push!(b_hidden, Parameter(zeros(Float32, n_hidden, 1)))
    end

    # Output layer: n_hidden -> n_out
    W_out = Parameter(glorot_uniform(n_out, n_hidden))
    b_out = Parameter(zeros(Float32, n_out, 1))

    return DenseConditionalLayerCV(split_idx, W1, b1, W_hidden, b_hidden, W_out, b_out, activation)
end

# Make it Flux-compatible
Flux.@functor DenseConditionalLayerCV

# Note: get_params and clear_grad! are already imported by the parent CNCV module
# We're just adding methods for our new type

# Get parameters for training
get_params(L::DenseConditionalLayerCV) = begin
    params = Parameter[L.W1, L.b1]
    append!(params, L.W_hidden)
    append!(params, L.b_hidden)
    push!(params, L.W_out)
    push!(params, L.b_out)
    return params
end

# Clear gradients
# Extend the clear_grad! function from the parent scope
clear_grad!(L::DenseConditionalLayerCV) = begin
    for p in get_params(L)
        p.grad = nothing
    end
end

# Forward pass: Input X (n_in×batch) and C (n_cond×batch), Output Y and jac_trace
function forward(X::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV) where T
    n_in, batch_size = size(X)

    # Split input
    X1 = X[1:L.split_idx, :]  # First half
    X2 = X[L.split_idx+1:end, :]  # Second half (pass-through)

    Y2 = copy(X2)

    # Concatenate X2 and C
    input = vcat(X2, C)

    # Forward through MLP
    # First layer
    hidden = L.W1.data * input .+ L.b1.data
    hidden = L.activation.(hidden)

    # Hidden layers
    for i in 1:length(L.W_hidden)
        hidden = L.W_hidden[i].data * hidden .+ L.b_hidden[i].data
        hidden = L.activation.(hidden)
    end

    # Output layer (no activation)
    output = L.W_out.data * hidden .+ L.b_out.data

    # Split output into S and Tm (translation/shift term)
    S_raw = output[1:L.split_idx, :]
    Tm = output[L.split_idx+1:end, :]

    # Apply sigmoid to S to ensure positivity (for stability)
    S = sigmoid.(S_raw)

    # Compute Y1 = S ⊙ X1 + Tm
    Y1 = S .* X1 .+ Tm

    # Concatenate Y1 and Y2
    Y = vcat(Y1, Y2)

    # Compute Jacobian trace
    # Jacobian is block triangular: [[diag(S), *], [0, I]]
    # Trace = sum(S) + length(X2)
    jac_trace = vec(sum(S, dims=1)) .+ Float32(size(X2, 1))

    return Y, jac_trace
end

# Inverse pass: Input Y and C, Output X and jac_trace
function inverse(Y::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV) where T
    Y1 = Y[1:L.split_idx, :]
    Y2 = Y[L.split_idx+1:end, :]

    X2 = copy(Y2)

    # Forward through MLP (same as forward pass)
    input = vcat(X2, C)

    hidden = L.W1.data * input .+ L.b1.data
    hidden = L.activation.(hidden)

    for i in 1:length(L.W_hidden)
        hidden = L.W_hidden[i].data * hidden .+ L.b_hidden[i].data
        hidden = L.activation.(hidden)
    end

    output = L.W_out.data * hidden .+ L.b_out.data

    S_raw = output[1:L.split_idx, :]
    Tm = output[L.split_idx+1:end, :]
    S = sigmoid.(S_raw)

    # Invert: X1 = (Y1 - Tm) / S
    X1 = (Y1 .- Tm) ./ (S .+ eps(Float32))

    X = vcat(X1, X2)

    # Jacobian trace (same as forward)
    jac_trace = vec(sum(S, dims=1)) .+ Float32(size(X2, 1))

    return X, jac_trace
end

# Backward pass - use Flux automatic differentiation
function backward(ΔY::AbstractMatrix{T}, Y::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV;
                  jac_trace_grad_weight::Union{Nothing, AbstractVector{T}}=nothing) where T

    # Use Flux.gradient to compute gradients
    # This is simpler and more reliable than manual backprop
    params_list = get_params(L)

    # Define loss function for backward pass
    function compute_output()
        # Re-compute forward pass
        # Extract X from Y using inverse
        X, _ = inverse(Y, C, L)

        # Forward pass
        Y_recon, jac_trace = forward(X, C, L)

        # Compute loss: dot product with ΔY
        loss = sum(Y_recon .* ΔY)

        # Add jac_trace term if provided
        if !isnothing(jac_trace_grad_weight)
            loss += sum(jac_trace .* jac_trace_grad_weight)
        end

        return loss
    end

    # Compute gradients using Flux
    grads = Flux.gradient(compute_output, Flux.params(params_list))

    # Store gradients in parameters
    for p in params_list
        if haskey(grads, p.data)
            if p.grad === nothing
                p.grad = grads[p.data]
            else
                p.grad .+= grads[p.data]
            end
        end
    end

    # Compute ΔX and ΔC using chain rule
    # For now, return zero gradients (not used in our training)
    X, _ = inverse(Y, C, L)
    ΔX = zeros(T, size(X))
    ΔC = zeros(T, size(C))

    return ΔX, X, ΔC
end
