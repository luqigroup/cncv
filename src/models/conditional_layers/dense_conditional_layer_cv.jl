# Dense conditional layer for control variates
# Designed for low-dimensional data (not images)
# Date: December 2025

export DenseConditionalLayerCV

# Import what we need from InvertibleNetworks
import InvertibleNetworks: FluxBlock, Parameter, get_params, clear_grad!
import InvertibleNetworks: forward, inverse, backward
import Flux

"""
    CL = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers=3; activation=tanh, coupling_activation=identity, n_cv=nothing)

Create a dense (MLP-based) conditional coupling layer for control variates.
Designed for low-dimensional data where spatial structure is not present.

*Input*:

- `n_in`: number of input dimensions
- `n_cond`: number of conditioning dimensions
- `n_hidden`: number of hidden units in MLP
- `n_layers`: number of hidden layers (default 3)
- `activation`: activation function for hidden layers (default tanh)
- `coupling_activation`: activation function for S parameters (default identity). Use identity for unconstrained trace values.
- `n_cv`: number of control variates to output (default: n_in, one per dimension)

*Output*:

- `CL`: Dense conditional coupling layer that outputs vector-valued control variates

*Usage:*

- Forward mode: `jac_traces, phi_all = forward(X, C, CL)` where X is n_in×batch, C is n_cond×batch
  Returns jac_traces (n_cv×batch) and phi_all (n_cv×n_in×batch) - transformations for all CVs
- Inverse mode: `X, jac_traces, phi_all = inverse(Y, C, CL)` (inverse is for internal use only)
- Backward mode: `ΔX, X, ΔC = backward(ΔY, X, C, CL; jac_trace_grad_weights=nothing, phi_grad_weights=nothing)`
  jac_trace_grad_weights should be n_cv×batch, phi_grad_weights should be n_cv×n_in×batch

The layer implements a coupling transformation with tractable Jacobian.
For vector-valued control variates, we output n_cv separate φ functions,
each giving one control variate via g_k = div(φ_k) + φ_k·∇log p.
"""
struct DenseConditionalLayerCV <: NeuralNetLayer
    split_idx::Int  # Where to split input
    n_cv::Int       # Number of control variates to output
    FB::FluxBlock   # Flux block for MLP
    coupling_activation::Function  # Activation for S (default identity)
end

@Flux.functor DenseConditionalLayerCV

# Constructor from input dimensions
function DenseConditionalLayerCV(n_in::Int, n_cond::Int, n_hidden::Int, n_layers::Int=3;
                                 activation::Function=tanh, coupling_activation::Function=identity,
                                 n_cv::Union{Nothing,Int}=nothing)

    split_idx = div(n_in, 2)
    n_in2 = n_in - split_idx  # Size of X2 (pass-through part)

    # Default: one CV per input dimension
    if n_cv === nothing
        n_cv = n_in
    end

    # Output: n_cv sets of [S, T], each set is split_idx-dimensional
    # So total output is n_cv * 2 * split_idx
    n_out = n_cv * 2 * split_idx

    # Build MLP using Flux: input is [X2; C], output is n_cv sets of [S; T]
    input_dim = n_in2 + n_cond

    # Create layers
    layers = []

    # First layer: input_dim -> n_hidden
    push!(layers, Flux.Dense(input_dim, n_hidden, activation))

    # Hidden layers: n_hidden -> n_hidden
    for i in 2:n_layers
        push!(layers, Flux.Dense(n_hidden, n_hidden, activation))
    end

    # Output layer: n_hidden -> n_out (no activation, we apply coupling_activation separately)
    # Initialize output layer to ZEROS so φ starts small
    W_out = zeros(Float32, n_out, n_hidden)
    b_out = zeros(Float32, n_out)
    push!(layers, Flux.Dense(W_out, b_out))

    # Create Flux Chain
    model = Flux.Chain(layers...)

    # Wrap in FluxBlock
    FB = FluxBlock(model)

    return DenseConditionalLayerCV(split_idx, n_cv, FB, coupling_activation)
end

# Get parameters for training
get_params(L::DenseConditionalLayerCV) = get_params(L.FB)

# Clear gradients
clear_grad!(L::DenseConditionalLayerCV) = clear_grad!(L.FB)

# Forward pass: Input X (n_in×batch) and C (n_cond×batch), Output jac_traces and phi_all
# Returns n_cv different φ_k transformations for computing control variates
function forward(X::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV) where T
    n_in, batch_size = size(X)

    # Split input
    X1 = X[1:L.split_idx, :]  # First half
    X2 = X[L.split_idx+1:end, :]  # Second half

    # Concatenate X2 and C
    input = vcat(X2, C)

    # Forward through FluxBlock (MLP)
    # Output is n_cv sets of [S, T], each of size split_idx
    output = forward(input, L.FB)

    # Compute n_cv control variates (traces) and transformations
    jac_traces = zeros(T, L.n_cv, batch_size)
    phi_all = zeros(T, L.n_cv, n_in, batch_size)  # Store all φ_k transformations

    for k in 1:L.n_cv
        # Extract S and T for k-th control variate
        idx_start = (k-1) * 2 * L.split_idx + 1
        S_raw_k = output[idx_start:idx_start+L.split_idx-1, :]
        T_k = output[idx_start+L.split_idx:idx_start+2*L.split_idx-1, :]

        # Apply coupling activation
        S_k = L.coupling_activation.(S_raw_k)

        # Compute φ_k transformation: φ_k = [S_k ⊙ X1 + T_k; X2]
        Y1_k = S_k .* X1 .+ T_k
        phi_k = vcat(Y1_k, X2)
        phi_all[k, :, :] = phi_k

        # Compute trace for k-th CV: trace = sum(S_k) + dim(X2)
        jac_traces[k, :] = vec(sum(S_k, dims=1)) .+ T(size(X2, 1))
    end

    return jac_traces, phi_all
end

# Inverse pass: For control variates, we don't need true invertibility
# This function is used in backward() to recompute forward state
# We treat the input as X and compute the forward pass
function inverse(Y::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV; save=false) where T
    n_in, batch_size = size(Y)

    # For CV layers, inverse is not well-defined (S can be zero with identity activation)
    # Instead, we just treat Y as X and recompute forward pass
    X = Y

    X1 = X[1:L.split_idx, :]
    X2 = X[L.split_idx+1:end, :]

    # Concatenate X2 and C to get network input
    input = vcat(X2, C)

    # Forward through FluxBlock (MLP)
    output = forward(input, L.FB)

    # Now compute phi_all and jac_traces for all CVs using the recovered X
    jac_traces = zeros(T, L.n_cv, batch_size)
    phi_all = zeros(T, L.n_cv, n_in, batch_size)
    S_all = []

    for k in 1:L.n_cv
        idx_start = (k-1) * 2 * L.split_idx + 1
        S_raw_k = output[idx_start:idx_start+L.split_idx-1, :]
        T_k = output[idx_start+L.split_idx:idx_start+2*L.split_idx-1, :]

        S_k = L.coupling_activation.(S_raw_k)
        push!(S_all, (S_raw_k, S_k, T_k))

        # Compute φ_k transformation using recovered X1
        Y1_k = S_k .* X1 .+ T_k
        phi_k = vcat(Y1_k, X2)
        phi_all[k, :, :] = phi_k

        jac_traces[k, :] = vec(sum(S_k, dims=1)) .+ T(size(X2, 1))
    end

    if save
        return X, jac_traces, phi_all, X1, X2, S_all, input
    else
        return X, jac_traces, phi_all
    end
end

# Backward pass - manual implementation for n_cv control variates
# Note: Second parameter is X (input), not Y (output), despite the signature name
function backward(ΔY::AbstractMatrix{T}, X::AbstractMatrix{T}, C::AbstractMatrix{T}, L::DenseConditionalLayerCV;
                  jac_trace_grad_weights::Union{Nothing, AbstractMatrix{T}}=nothing,
                  phi_grad_weights::Union{Nothing, AbstractArray{T, 3}}=nothing) where T

    # X is provided directly (no need to recover from Y)
    # Compute forward state
    X1 = X[1:L.split_idx, :]
    X2 = X[L.split_idx+1:end, :]

    input = vcat(X2, C)
    output = forward(input, L.FB)

    # Collect S, T for all CVs
    S_all = []
    for k in 1:L.n_cv
        idx_start = (k-1) * 2 * L.split_idx + 1
        S_raw_k = output[idx_start:idx_start+L.split_idx-1, :]
        T_k = output[idx_start+L.split_idx:idx_start+2*L.split_idx-1, :]
        S_k = L.coupling_activation.(S_raw_k)
        push!(S_all, (S_raw_k, S_k, T_k))
    end

    batchsize = size(X, 2)

    # For CVs: Y = X (identity), so ΔX starts with ΔY
    ΔX = copy(ΔY)

    # Initialize gradient for FluxBlock output
    output_dim = L.n_cv * 2 * L.split_idx
    Δoutput = zeros(T, output_dim, batchsize)

    # Backpropagate through each control variate
    for k in 1:L.n_cv
        S_raw_k, S_k, T_k = S_all[k]

        ΔS_k = zeros(T, size(S_k))
        ΔT_k = zeros(T, size(T_k))

        # Gradient from jac_trace if provided
        if !isnothing(jac_trace_grad_weights)
            # jac_trace_k = sum(S_k) + const, so ∂jac_trace_k/∂S_k = 1 for all elements
            for b in 1:batchsize
                ΔS_k[:, b] .+= jac_trace_grad_weights[k, b]
            end
        end

        # Gradient from phi_k if provided
        if !isnothing(phi_grad_weights)
            # phi_k = [S_k .* X1 + T_k; X2]
            # ∂phi_k[1:split_idx]/∂S_k = X1 (element-wise)
            # ∂phi_k[1:split_idx]/∂T_k = 1
            Δphi_k = phi_grad_weights[k, :, :]  # n_in × batchsize

            # Gradient wrt Y1_k part
            ΔY1_k = Δphi_k[1:L.split_idx, :]

            # ∂(S_k .* X1 + T_k)/∂S_k = X1
            ΔS_k .+= ΔY1_k .* X1

            # ∂(S_k .* X1 + T_k)/∂T_k = 1
            ΔT_k .+= ΔY1_k

            # Gradient wrt X2 part (X2 is pass-through in phi_k)
            ΔX2_from_phi = Δphi_k[L.split_idx+1:end, :]
            ΔX[L.split_idx+1:end, :] .+= ΔX2_from_phi
        end

        # Backpropagate through coupling activation
        if L.coupling_activation == sigmoid
            ΔS_raw_k = ΔS_k .* S_k .* (1 .- S_k)
        elseif L.coupling_activation == identity
            # Identity activation: gradient is 1
            ΔS_raw_k = ΔS_k
        else
            # Generic case using Zygote
            ΔS_raw_k = zero(S_raw_k)
            for b in 1:batchsize
                for i in 1:size(S_raw_k, 1)
                    grad = Flux.gradient(x -> L.coupling_activation(x), S_raw_k[i, b])[1]
                    ΔS_raw_k[i, b] = ΔS_k[i, b] * grad
                end
            end
        end

        # Store in Δoutput
        idx_start = (k-1) * 2 * L.split_idx + 1
        Δoutput[idx_start:idx_start+L.split_idx-1, :] = ΔS_raw_k
        Δoutput[idx_start+L.split_idx:idx_start+2*L.split_idx-1, :] = ΔT_k
    end

    # Backpropagate through FluxBlock
    Δinput = backward(Δoutput, input, L.FB; set_grad=true)

    # Split gradient into ΔX2 and ΔC
    n_X2 = size(X2, 1)
    ΔX2_from_mlp = Δinput[1:n_X2, :]
    ΔC = Δinput[n_X2+1:end, :]

    # Add gradient from X2 part (from MLP)
    ΔX[L.split_idx+1:end, :] .+= ΔX2_from_mlp

    return ΔX, X, ΔC
end
