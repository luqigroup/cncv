# Training script for Rosenbrock prior with ENSEMBLE CV network
# Tests the forward-reverse ensemble on Rosenbrock distribution
# Can use either LEARNED or TRUE analytical score function
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using InvertibleNetworks: get_params, clear_grad!
using Rosenbrock
using Statistics
using Random
using ProgressMeter
using Flux
using LinearAlgebra

# Import ensemble functions
import CNCV: forward, backward
import CNCV: DenseConditionalLayerCV_Reversible, create_forward_reverse_ensemble

# Random seed
Random.seed!(19)

args = read_config("rosenbrock_ensemble_cv_training.json")
args = parse_input_args(args)

# Conditionally load pretrained amortized model for learned likelihood
if args["learned_score"] != 0
    println("Loading pretrained amortized model for learned log-likelihood...")
    amortized_args = read_config("rosenbrock_amortized_sampling.json")
    if amortized_args["epoch"] == -1
        amortized_args["epoch"] = amortized_args["max_epoch"]
    end
    loaded_amortized = load_experiment(amortized_args, ["G"])
    G_amortized = loaded_amortized["G"]
    println("Loaded amortized model successfully!")
else
    println("Using TRUE analytical score function (Gaussian + Rosenbrock)")
    G_amortized = nothing
end

device = cpu

# Create ENSEMBLE with forward + reverse splits
println("Creating forward-reverse ensemble for Rosenbrock...")
ensemble = create_forward_reverse_ensemble(2, 2, args["n_hidden"], args["n_layers"];
                                           n_cv=2)

# Get parameters from both layers
layer1 = ensemble.layers[1]
layer2 = ensemble.layers[2]
println("Layer 1 (forward split): reverse_split = ", layer1.reverse_split)
println("Layer 2 (reverse split): reverse_split = ", layer2.reverse_split)
println("Number of layers in ensemble: ", length(ensemble.layers))

# Training data
ntrain = 5120

# Generate training data from Rosenbrock prior + Gaussian likelihood
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_train = rand(RB_dist, ntrain)  # 2×ntrain

# Add Gaussian noise to create observations Y = X + noise
Y_train = X_train + args["sigma"] * randn(Float32, 2, ntrain)
X_train = X_train |> device
Y_train = Y_train |> device

# Validation data
nval = 512
X_val = rand(RB_dist, nval)
Y_val = X_val + args["sigma"] * randn(Float32, 2, nval)
X_val = X_val |> device
Y_val = Y_val |> device

p = Progress(Int(floor(ntrain / args["batchsize"])) * args["max_epoch"])

# Training batch extractor
train_loader = Flux.DataLoader((X_train, Y_train), batchsize = args["batchsize"], shuffle = true)
num_batches = length(train_loader)

# Optimizer
opt = Flux.Optimiser(
    Flux.ExpDecay(args["lr"], 0.9f0, num_batches * args["lr_step"], 1.0f-6),
    Flux.Adam(args["lr"]),
)

# Learnable offsets (one per component)
μ = [0.0f0, 0.0f0] |> device

# Training log
fval = zeros(Float32, num_batches * args["max_epoch"])
fval_eval = zeros(Float32, args["max_epoch"])

# Rosenbrock distribution for analytical score (if needed)
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)

# Score function: switches between learned and true based on args["learned_score"]
function compute_score_posterior(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T
    if args["learned_score"] != 0
        # Use LEARNED score from pretrained amortized model
        batchsize = size(X, 2)
        X_net = reshape(X, 1, 1, 2, batchsize)
        Y_net = reshape(Y, 1, 1, 2, batchsize)

        # Compute score using exact_score function
        ΔX_net = exact_score(G_amortized, X_net, Y_net)

        # Reshape back to 2×batchsize
        grad_log_post = reshape(ΔX_net, 2, batchsize)
    else
        # Use TRUE analytical score: ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
        # Gradient of Gaussian likelihood: ∇ log p(y|x) = -(x-y)/σ²
        grad_likelihood = -(X .- Y) ./ (args["sigma"]^2)

        # Gradient of Rosenbrock prior
        grad_prior = gradlogpdf(RB_dist, X)

        # Total gradient
        grad_log_post = grad_likelihood .+ grad_prior
    end

    return grad_log_post
end

# Loss function with ENSEMBLE
function loss_ensemble_cv(ensemble, X, Y, μ)
    batchsize = size(X, 2)
    n_cv = 2

    # Quantity of interest: h(x) = x
    h_x = X

    # Compute score (learned or true analytical)
    score_term = compute_score_posterior(X, Y)

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
    residual = zeros(Float32, 2, batchsize)
    residual[1, :] = h_x[1, :] .- g_cv[1, :] .- μ[1]
    residual[2, :] = h_x[2, :] .- g_cv[2, :] .- μ[2]

    loss = mean(residual .^ 2)

    return loss, h_x, g_cv, g_cv_1, g_cv_2
end

for epoch = 1:args["max_epoch"]

    # Evaluate on validation set
    fval_eval[epoch] = loss_ensemble_cv(ensemble, X_val, Y_val, μ)[1]

    for (itr, (X, Y)) in enumerate(train_loader)
        Base.flush(Base.stdout)

        batchsize = size(X, 2)

        # Forward pass
        h_x = X
        score_term = compute_score_posterior(X, Y)

        # Layer 1
        jac_traces_1, phi_all_1 = forward(X, Y, layer1)
        g_cv_1 = zeros(Float32, 2, batchsize)
        for k in 1:2
            trace_k = jac_traces_1[k, :]
            phi_k = phi_all_1[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
            g_cv_1[k, :] = trace_k .+ phi_dot_score_k
        end

        # Layer 2
        jac_traces_2, phi_all_2 = forward(X, Y, layer2)
        g_cv_2 = zeros(Float32, 2, batchsize)
        for k in 1:2
            trace_k = jac_traces_2[k, :]
            phi_k = phi_all_2[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
            g_cv_2[k, :] = trace_k .+ phi_dot_score_k
        end

        # Combined CV (average)
        g_cv = (g_cv_1 .+ g_cv_2) ./ 2

        # Residual and loss
        residual = zeros(Float32, 2, batchsize)
        residual[1, :] = h_x[1, :] .- g_cv[1, :] .- μ[1]
        residual[2, :] = h_x[2, :] .- g_cv[2, :] .- μ[2]
        loss = mean(residual .^ 2)
        fval[(epoch-1)*num_batches+itr] = loss

        # Backward pass through BOTH layers
        Δresidual = residual ./ Float32(batchsize)
        Δg_cv = -Δresidual

        # Distribute gradient to both layers
        Δg_cv_1 = Δg_cv ./ 2
        Δg_cv_2 = Δg_cv ./ 2

        # Backward through Layer 1
        jac_trace_grad_1 = Δg_cv_1
        Δphi_all_1 = zeros(Float32, 2, 2, batchsize)
        for k in 1:2
            Δphi_all_1[k, :, :] = score_term .* reshape(Δg_cv_1[k, :], 1, batchsize)
        end

        ΔX_1, _, ΔY_1 = backward(zeros(Float32, size(X)), X, Y, layer1;
                                 jac_trace_grad_weights=jac_trace_grad_1,
                                 phi_grad_weights=Δphi_all_1)

        # Backward through Layer 2
        jac_trace_grad_2 = Δg_cv_2
        Δphi_all_2 = zeros(Float32, 2, 2, batchsize)
        for k in 1:2
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
        residual_mean_1 = mean(residual[1, :])
        residual_mean_2 = mean(residual[2, :])
        grad_mu_1 = -2 * residual_mean_1
        grad_mu_2 = -2 * residual_mean_2
        μ[1] -= args["lr"] * grad_mu_1
        μ[2] -= args["lr"] * grad_mu_2

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
