# Ensemble of Dense Conditional CV Layers with Different Splits
# This addresses the coupling layer bottleneck by having multiple layers
# transform different components
# Date: December 2025

export DenseConditionalLayerCV_Reversible, EnsembleDenseCV

import InvertibleNetworks: FluxBlock, Parameter, get_params, clear_grad!
import InvertibleNetworks: forward, inverse, backward
import Flux

"""
    DenseConditionalLayerCV_Reversible

Same as DenseConditionalLayerCV but with `reverse_split` option.

When reverse_split=false (default):
    φ = [S(x₂, c) ⊙ x₁ + T(x₂, c); x₂]  # Transform x₁, pass x₂

When reverse_split=true:
    φ = [x₁; S(x₁, c) ⊙ x₂ + T(x₁, c)]  # Pass x₁, transform x₂

This allows different layers to focus on transforming different components!
"""
struct DenseConditionalLayerCV_Reversible <: NeuralNetLayer
    split_idx::Int
    n_cv::Int
    FB::FluxBlock
    coupling_activation::Function
    reverse_split::Bool  # NEW: reverse which half gets transformed
end

@Flux.functor DenseConditionalLayerCV_Reversible

function DenseConditionalLayerCV_Reversible(n_in::Int, n_cond::Int, n_hidden::Int, n_layers::Int=3;
                                            activation::Function=tanh,
                                            coupling_activation::Function=identity,
                                            n_cv::Union{Nothing,Int}=nothing,
                                            reverse_split::Bool=false)

    split_idx = div(n_in, 2)
    n_in1 = split_idx
    n_in2 = n_in - split_idx

    if n_cv === nothing
        n_cv = n_in
    end

    # Determine which part is used as input to MLP
    if reverse_split
        # MLP input: [X1; C], output: transforms for X2
        input_dim = n_in1 + n_cond
        transform_dim = n_in2
    else
        # MLP input: [X2; C], output: transforms for X1 (original)
        input_dim = n_in2 + n_cond
        transform_dim = n_in1
    end

    # Output: n_cv sets of [S, T], each set is transform_dim-dimensional
    n_out = n_cv * 2 * transform_dim

    # Build MLP
    layers = []
    push!(layers, Flux.Dense(input_dim, n_hidden, activation))
    for i in 2:n_layers
        push!(layers, Flux.Dense(n_hidden, n_hidden, activation))
    end
    push!(layers, Flux.Dense(n_hidden, n_out))

    model = Flux.Chain(layers...)
    FB = FluxBlock(model)

    return DenseConditionalLayerCV_Reversible(split_idx, n_cv, FB, coupling_activation, reverse_split)
end

get_params(L::DenseConditionalLayerCV_Reversible) = get_params(L.FB)
clear_grad!(L::DenseConditionalLayerCV_Reversible) = clear_grad!(L.FB)

function forward(X::AbstractMatrix{T}, C::AbstractMatrix{T},
                 L::DenseConditionalLayerCV_Reversible) where T
    n_in, batch_size = size(X)

    # Split input
    X1 = X[1:L.split_idx, :]
    X2 = X[L.split_idx+1:end, :]

    # Choose which part to use as MLP input
    if L.reverse_split
        # Use X1 and C to transform X2
        mlp_input = vcat(X1, C)
        transform_dim = size(X2, 1)
    else
        # Use X2 and C to transform X1 (original behavior)
        mlp_input = vcat(X2, C)
        transform_dim = size(X1, 1)
    end

    # Forward through MLP
    output = forward(mlp_input, L.FB)

    # Compute control variates
    jac_traces = zeros(T, L.n_cv, batch_size)
    phi_all = zeros(T, L.n_cv, n_in, batch_size)

    for k in 1:L.n_cv
        # Extract S and T for k-th CV
        idx_start = (k-1) * 2 * transform_dim + 1
        S_raw_k = output[idx_start:idx_start+transform_dim-1, :]
        T_k = output[idx_start+transform_dim:idx_start+2*transform_dim-1, :]

        S_k = L.coupling_activation.(S_raw_k)

        if L.reverse_split
            # φ = [X1; S ⊙ X2 + T]
            Y2_k = S_k .* X2 .+ T_k
            phi_k = vcat(X1, Y2_k)

            # Trace: sum(S_k) + dim(X1)
            jac_traces[k, :] = vec(sum(S_k, dims=1)) .+ T(size(X1, 1))
        else
            # φ = [S ⊙ X1 + T; X2] (original)
            Y1_k = S_k .* X1 .+ T_k
            phi_k = vcat(Y1_k, X2)

            # Trace: sum(S_k) + dim(X2)
            jac_traces[k, :] = vec(sum(S_k, dims=1)) .+ T(size(X2, 1))
        end

        phi_all[k, :, :] = phi_k
    end

    return jac_traces, phi_all
end

# Backward pass (similar to original, but handle reverse_split)
function backward(ΔY::AbstractMatrix{T}, X::AbstractMatrix{T}, C::AbstractMatrix{T},
                  L::DenseConditionalLayerCV_Reversible;
                  jac_trace_grad_weights::Union{Nothing,AbstractMatrix{T}}=nothing,
                  phi_grad_weights::Union{Nothing,AbstractArray{T,3}}=nothing) where T

    n_in, batch_size = size(X)
    X1 = X[1:L.split_idx, :]
    X2 = X[L.split_idx+1:end, :]

    if L.reverse_split
        mlp_input = vcat(X1, C)
        transform_dim = size(X2, 1)
    else
        mlp_input = vcat(X2, C)
        transform_dim = size(X1, 1)
    end

    output = forward(mlp_input, L.FB)

    # Accumulate gradients
    Δoutput = zeros(T, size(output))

    for k in 1:L.n_cv
        idx_start = (k-1) * 2 * transform_dim + 1
        S_raw_k = output[idx_start:idx_start+transform_dim-1, :]
        T_k = output[idx_start+transform_dim:idx_start+2*transform_dim-1, :]

        S_k = L.coupling_activation.(S_raw_k)

        # Gradients from trace
        if jac_trace_grad_weights !== nothing
            grad_trace_k = jac_trace_grad_weights[k, :]
            ΔS_from_trace = repeat(reshape(grad_trace_k, 1, batch_size), transform_dim, 1)
        else
            ΔS_from_trace = zeros(T, transform_dim, batch_size)
        end

        # Gradients from phi
        if phi_grad_weights !== nothing
            Δphi_k = phi_grad_weights[k, :, :]

            if L.reverse_split
                # φ = [X1; S ⊙ X2 + T]
                ΔY2 = Δphi_k[L.split_idx+1:end, :]
                ΔS_from_phi = ΔY2 .* X2
                ΔT = ΔY2
            else
                # φ = [S ⊙ X1 + T; X2]
                ΔY1 = Δphi_k[1:L.split_idx, :]
                ΔS_from_phi = ΔY1 .* X1
                ΔT = ΔY1
            end
        else
            ΔS_from_phi = zeros(T, transform_dim, batch_size)
            ΔT = zeros(T, transform_dim, batch_size)
        end

        # Total gradient for S
        ΔS = ΔS_from_trace .+ ΔS_from_phi

        # Chain rule through coupling_activation
        # For identity: ΔS_raw = ΔS
        ΔS_raw = ΔS  # Assuming identity activation

        # Pack into Δoutput
        Δoutput[idx_start:idx_start+transform_dim-1, :] = ΔS_raw
        Δoutput[idx_start+transform_dim:idx_start+2*transform_dim-1, :] = ΔT
    end

    # Backward through MLP
    Δmlp_input = backward(Δoutput, mlp_input, L.FB)

    # Split gradients
    if L.reverse_split
        ΔX1 = Δmlp_input[1:L.split_idx, :]
        ΔC = Δmlp_input[L.split_idx+1:end, :]
        ΔX2 = zeros(T, size(X2))  # X2 was transformed, gradient comes from phi
    else
        ΔX2 = Δmlp_input[1:size(X2, 1), :]
        ΔC = Δmlp_input[size(X2, 1)+1:end, :]
        ΔX1 = zeros(T, size(X1))  # X1 was transformed
    end

    # Add gradients from phi (for transformed part)
    if phi_grad_weights !== nothing
        for k in 1:L.n_cv
            Δphi_k = phi_grad_weights[k, :, :]

            if L.reverse_split
                # X2 was transformed: ΔX2 += S .* Δphi[X2 part]
                idx_start = (k-1) * 2 * transform_dim + 1
                S_raw_k = output[idx_start:idx_start+transform_dim-1, :]
                S_k = L.coupling_activation.(S_raw_k)

                ΔY2 = Δphi_k[L.split_idx+1:end, :]
                ΔX2 .+= S_k .* ΔY2
            else
                # X1 was transformed
                idx_start = (k-1) * 2 * transform_dim + 1
                S_raw_k = output[idx_start:idx_start+transform_dim-1, :]
                S_k = L.coupling_activation.(S_raw_k)

                ΔY1 = Δphi_k[1:L.split_idx, :]
                ΔX1 .+= S_k .* ΔY1
            end
        end
    end

    ΔX = vcat(ΔX1, ΔX2)

    return ΔX, X, ΔC
end


"""
    EnsembleDenseCV

Ensemble of multiple DenseConditionalLayerCV_Reversible layers.
Combines control variates from different layers, allowing each to specialize
on different input components.

Usage:
    ensemble = EnsembleDenseCV([layer1, layer2])
    g_combined, all_g = forward_ensemble(X, C, score, ensemble)
"""
struct EnsembleDenseCV
    layers::Vector{DenseConditionalLayerCV_Reversible}
    combination_mode::Symbol  # :average, :learned_weights
    weights::Union{Nothing, Vector{Float32}}  # For learned_weights mode
end

function EnsembleDenseCV(layers::Vector{DenseConditionalLayerCV_Reversible};
                         combination_mode::Symbol=:average)
    weights = nothing
    if combination_mode == :learned_weights
        # Initialize with equal weights
        weights = ones(Float32, length(layers)) ./ length(layers)
    end
    return EnsembleDenseCV(layers, combination_mode, weights)
end

"""
Compute combined control variate from ensemble.
Returns combined g and individual g's for analysis.
"""
function forward_ensemble(X::AbstractMatrix{T}, C::AbstractMatrix{T},
                          score::AbstractMatrix{T},
                          ensemble::EnsembleDenseCV) where T

    n_layers = length(ensemble.layers)
    batch_size = size(X, 2)
    n_cv = ensemble.layers[1].n_cv

    # Storage for individual CVs
    all_g = zeros(T, n_layers, n_cv, batch_size)

    # Forward through each layer
    for (i, layer) in enumerate(ensemble.layers)
        jac_traces, phi_all = forward(X, C, layer)

        # Compute control variates for this layer
        for k in 1:n_cv
            trace_k = jac_traces[k, :]
            phi_k = phi_all[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score, dims=1))
            all_g[i, k, :] = trace_k .+ phi_dot_score_k
        end
    end

    # Combine based on mode
    if ensemble.combination_mode == :average
        # Simple average
        g_combined = dropdims(mean(all_g, dims=1), dims=1)  # n_cv × batch_size
    elseif ensemble.combination_mode == :learned_weights
        # Weighted combination
        g_combined = zeros(T, n_cv, batch_size)
        for i in 1:n_layers
            w = ensemble.weights[i]
            g_combined .+= w .* all_g[i, :, :]
        end
    else
        error("Unknown combination_mode: $(ensemble.combination_mode)")
    end

    return g_combined, all_g
end

"""
Helper to create a two-layer ensemble with forward and reverse splits.
This is the simplest ensemble that addresses the coupling bottleneck.
"""
function create_forward_reverse_ensemble(n_in::Int, n_cond::Int, n_hidden::Int, n_layers::Int=3;
                                         activation::Function=tanh, n_cv::Union{Nothing,Int}=nothing)

    # Layer 1: Transform first half (original)
    layer1 = DenseConditionalLayerCV_Reversible(n_in, n_cond, n_hidden, n_layers;
                                                activation=activation, n_cv=n_cv,
                                                reverse_split=false)

    # Layer 2: Transform second half (reversed)
    layer2 = DenseConditionalLayerCV_Reversible(n_in, n_cond, n_hidden, n_layers;
                                                activation=activation, n_cv=n_cv,
                                                reverse_split=true)

    return EnsembleDenseCV([layer1, layer2]; combination_mode=:average)
end
