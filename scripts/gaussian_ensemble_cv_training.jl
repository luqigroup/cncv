# Training script for Gaussian prior with ENSEMBLE CV network
# Tests the forward-reverse ensemble to fix the coupling bottleneck
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using InvertibleNetworks: get_params, clear_grad!
using Statistics
using Random
using ProgressMeter
using Flux
using LinearAlgebra
using Distributions

# Import ensemble functions
import CNCV: forward, backward
import CNCV: DenseConditionalLayerCV_Reversible, create_forward_reverse_ensemble

# Random seed
Random.seed!(19)

args = read_config("gaussian_ensemble_cv_training.json")
args = parse_input_args(args)

device = cpu

# Problem setup
n_dim = 2  # 2D problem

# Gaussian prior: X ~ N(μ_prior, Σ_prior)
μ_prior = zeros(Float32, n_dim)
Σ_prior = Float32[2.0 0.5; 0.5 1.0]  # Correlated prior
L_prior = cholesky(Σ_prior).L

# Create ENSEMBLE with forward + reverse splits
println("Creating forward-reverse ensemble...")
ensemble = create_forward_reverse_ensemble(n_dim, n_dim, args["n_hidden"], args["n_layers"];
                                           n_cv=n_dim)

# Get parameters from both layers
layer1 = ensemble.layers[1]
layer2 = ensemble.layers[2]
println("Layer 1 (forward split): reverse_split = ", layer1.reverse_split)
println("Layer 2 (reverse split): reverse_split = ", layer2.reverse_split)
println("Number of layers in ensemble: ", length(ensemble.layers))

# Training data
ntrain = 2^16

# Generate training data from Gaussian prior + Gaussian likelihood
X_train = μ_prior .+ L_prior * randn(Float32, n_dim, ntrain)
Y_train = X_train + args["sigma"] * randn(Float32, n_dim, ntrain)
X_train = X_train |> device
Y_train = Y_train |> device

# Validation data
nval = 2^10
X_val = μ_prior .+ L_prior * randn(Float32, n_dim, nval)
Y_val = X_val + args["sigma"] * randn(Float32, n_dim, nval)
X_val = X_val |> device
Y_val = Y_val |> device

p = Progress(Int(floor(ntrain / args["batchsize"])) * args["max_epoch"])

# Training batch extractor
train_loader = Flux.DataLoader((X_train, Y_train), batchsize = args["batchsize"], shuffle = true)
num_batches = length(train_loader)

# Collect parameters from both layers
all_params = vcat(get_params(layer1), get_params(layer2))

# Optimizer
opt = Flux.Optimiser(
    Flux.ExpDecay(args["lr"], 0.9f0, num_batches * args["lr_step"], 1.0f-6),
    Flux.Adam(args["lr"]),
)

# Learnable offsets (one per component)
μ = zeros(Float32, n_dim) |> device

# Training log
fval = zeros(Float32, num_batches * args["max_epoch"])
fval_eval = zeros(Float32, args["max_epoch"])

# Compute score function
Σ_prior_inv = inv(Σ_prior)
function compute_score_posterior(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                                  μ_prior::AbstractVector{T}, Σ_prior_inv::AbstractMatrix{T},
                                  σ::T) where T
    grad_likelihood = -(X .- Y) ./ (σ^2)
    grad_prior = -Σ_prior_inv * (X .- μ_prior)
    return grad_likelihood .+ grad_prior
end

# Loss function with ENSEMBLE
function loss_ensemble_cv(ensemble, X, Y, μ, σ)
    batchsize = size(X, 2)
    n_cv = n_dim

    # Quantity of interest: h(x) = x
    h_x = X

    # Compute score
    score_term = compute_score_posterior(X, Y, μ_prior, Σ_prior_inv, σ)

    # Forward through BOTH layers and combine
    layer1 = ensemble.layers[1]
    layer2 = ensemble.layers[2]

    # Layer 1
    jac_traces_1, phi_all_1 = forward(X, Y, layer1)
    g_cv_1 = zeros(Float32, n_cv, batchsize)
    for k in 1:n_cv
        trace_k = jac_traces_1[k, :]
        phi_k = phi_all_1[k, :, :]
        phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
        g_cv_1[k, :] = trace_k .+ phi_dot_score_k
    end

    # Layer 2
    jac_traces_2, phi_all_2 = forward(X, Y, layer2)
    g_cv_2 = zeros(Float32, n_cv, batchsize)
    for k in 1:n_cv
        trace_k = jac_traces_2[k, :]
        phi_k = phi_all_2[k, :, :]
        phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
        g_cv_2[k, :] = trace_k .+ phi_dot_score_k
    end

    # AVERAGE the two CVs
    g_cv = (g_cv_1 .+ g_cv_2) ./ 2

    # Controlled estimator: h_i - g_i - μ_i
    residual = zeros(Float32, n_dim, batchsize)
    for i in 1:n_dim
        residual[i, :] = h_x[i, :] .- g_cv[i, :] .- μ[i]
    end

    loss = mean(residual .^ 2)

    return loss, h_x, g_cv, g_cv_1, g_cv_2, jac_traces_1, phi_all_1, jac_traces_2, phi_all_2
end

for epoch = 1:args["max_epoch"]

    # Evaluate on validation set
    fval_eval[epoch] = loss_ensemble_cv(ensemble, X_val, Y_val, μ, args["sigma"])[1]

    for (itr, (X, Y)) in enumerate(train_loader)
        Base.flush(Base.stdout)

        batchsize = size(X, 2)

        # Forward pass
        h_x = X
        score_term = compute_score_posterior(X, Y, μ_prior, Σ_prior_inv, Float32(args["sigma"]))

        # Layer 1
        jac_traces_1, phi_all_1 = forward(X, Y, layer1)
        g_cv_1 = zeros(Float32, n_dim, batchsize)
        for k in 1:n_dim
            trace_k = jac_traces_1[k, :]
            phi_k = phi_all_1[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
            g_cv_1[k, :] = trace_k .+ phi_dot_score_k
        end

        # Layer 2
        jac_traces_2, phi_all_2 = forward(X, Y, layer2)
        g_cv_2 = zeros(Float32, n_dim, batchsize)
        for k in 1:n_dim
            trace_k = jac_traces_2[k, :]
            phi_k = phi_all_2[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
            g_cv_2[k, :] = trace_k .+ phi_dot_score_k
        end

        # Combined CV (average)
        g_cv = (g_cv_1 .+ g_cv_2) ./ 2

        # Residual and loss
        residual = zeros(Float32, n_dim, batchsize)
        for i in 1:n_dim
            residual[i, :] = h_x[i, :] .- g_cv[i, :] .- μ[i]
        end
        loss = mean(residual .^ 2)
        fval[(epoch-1)*num_batches+itr] = loss

        # Backward pass - need to backprop through BOTH layers
        Δresidual = residual ./ Float32(batchsize)

        # Gradients for combined CV: ∂L/∂g_cv = -Δresidual
        Δg_cv = -Δresidual

        # Distribute gradient to both layers (each contributes 1/2 to average)
        Δg_cv_1 = Δg_cv ./ 2
        Δg_cv_2 = Δg_cv ./ 2

        # Backward through Layer 1
        jac_trace_grad_1 = Δg_cv_1
        Δphi_all_1 = zeros(Float32, n_dim, n_dim, batchsize)
        for k in 1:n_dim
            Δphi_all_1[k, :, :] = score_term .* reshape(Δg_cv_1[k, :], 1, batchsize)
        end

        ΔX_1, _, ΔY_1 = backward(zeros(Float32, size(X)), X, Y, layer1;
                                 jac_trace_grad_weights=jac_trace_grad_1,
                                 phi_grad_weights=Δphi_all_1)

        # Backward through Layer 2
        jac_trace_grad_2 = Δg_cv_2
        Δphi_all_2 = zeros(Float32, n_dim, n_dim, batchsize)
        for k in 1:n_dim
            Δphi_all_2[k, :, :] = score_term .* reshape(Δg_cv_2[k, :], 1, batchsize)
        end

        ΔX_2, _, ΔY_2 = backward(zeros(Float32, size(X)), X, Y, layer2;
                                 jac_trace_grad_weights=jac_trace_grad_2,
                                 phi_grad_weights=Δphi_all_2)

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Iteration, itr),
                (:Loss, fval[(epoch-1)*num_batches+itr]),
                (:Loss_eval, fval_eval[epoch]),
                (:μ1, μ[1]),
                (:μ2, μ[2]),
                (:g1_mean, mean(g_cv_1)),
                (:g2_mean, mean(g_cv_2))
            ],
        )

        # Update params for BOTH layers
        for param in get_params(layer1)
            Flux.update!(opt, param.data, param.grad)
        end
        for param in get_params(layer2)
            Flux.update!(opt, param.data, param.grad)
        end

        # Update μ manually
        for i in 1:n_dim
            residual_mean_i = mean(residual[i, :])
            grad_mu_i = -2 * residual_mean_i
            μ[i] -= args["lr"] * grad_mu_i
        end

        clear_grad!(layer1)
        clear_grad!(layer2)
    end

    if epoch % 50 == 0 || epoch == args["max_epoch"]

        save_dict = Dict{String,Any}()
        for (key, val) in args
            save_dict[key] = val
        end

        save_dict = merge(
            save_dict,
            Dict(
                "epoch" => epoch,
                "fval" => fval,
                "fval_eval" => fval_eval,
                "opt" => opt,
                "layer1" => layer1 |> cpu,
                "layer2" => layer2 |> cpu,
                "ensemble" => ensemble |> cpu,
                "mu" => μ |> cpu,
                "mu_prior" => μ_prior,
                "Sigma_prior" => Σ_prior,
            ),
        )
        @tagsave(
            datadir(args["sim_name"], savename(save_dict, "jld2"; digits = 6)),
            save_dict;
            safe = true
        )
    end

end

upload_to_dropbox(args["sim_name"])
