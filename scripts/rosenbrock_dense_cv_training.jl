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

# Import forward, inverse, backward from CNCV module
import CNCV: forward, inverse, backward

# Random seed
Random.seed!(19)

args = read_config("rosenbrock_dense_cv_training.json")
args = parse_input_args(args)

device = cpu

# Define control variate network using DenseConditionalLayerCV
# This is designed for low-dimensional data (2D Rosenbrock)
CV = DenseConditionalLayerCV(
    2,  # n_in: 2D input
    2,  # n_cond: 2D conditioning
    args["n_hidden"],
    args["n_layers"];
    activation=tanh
)
CV = CV |> device

# Training data number
ntrain = 2^16

# Generate training data from Rosenbrock prior + Gaussian likelihood
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_train = rand(RB_dist, ntrain)  # 2×ntrain

# Add Gaussian noise to create observations Y = X + noise
Y_train = X_train + args["sigma"] * randn(Float32, 2, ntrain)
X_train = X_train |> device
Y_train = Y_train |> device

# Validation data
nval = 2^10
X_val = rand(RB_dist, nval)
Y_val = X_val + args["sigma"] * randn(Float32, 2, nval)
X_val = X_val |> device
Y_val = Y_val |> device

p = Progress(Int(floor(ntrain / args["batchsize"])) * args["max_epoch"])

# Training batch extractor
train_loader =
    Flux.DataLoader((X_train, Y_train), batchsize = args["batchsize"], shuffle = true)
num_batches = length(train_loader)

# Optimizer
opt = Flux.Optimiser(
    Flux.ExpDecay(args["lr"], 0.9f0, num_batches * args["lr_step"], 1.0f-6),
    Flux.Adam(args["lr"]),
)

# Learnable offset
μ = [0.1f0] |> device

# Training log keeper
fval = zeros(Float32, num_batches * args["max_epoch"])
fval_eval = zeros(Float32, args["max_epoch"])

# Helper function: compute gradient of unnormalized log posterior (score function)
# ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
# where:
#   ∇ log p(y|x) = -(x-y)/σ² (Gaussian likelihood)
#   ∇ log p(x) = gradlogpdf from Rosenbrock distribution
function compute_score_posterior(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, sigma::T) where T
    batchsize = size(X, 2)

    # Gradient of log likelihood: ∇ log p(y|x) = -(x-y)/σ²
    grad_likelihood = -(X .- Y) ./ (sigma^2)

    # Gradient of log prior: use gradlogpdf from Rosenbrock
    grad_prior = gradlogpdf(RB_dist, X)

    # Total gradient
    grad_log_post = grad_likelihood .+ grad_prior

    return grad_log_post
end

# Loss function for CV training: minimize E[(h(x) - (div(φ) + φ·∇log p) - μ)²]
# Here h(x) = x (identity, to estimate the mean)
function loss_cv(CV, X, Y, μ, sigma)
    batchsize = size(X, 2)

    # Quantity of interest: h(x) = x (2×batchsize)
    h_x = X

    # Forward through CV network to get jac_trace (which is div(φ))
    phi_X, jac_trace = forward(X, Y, CV)

    # Compute φ(x,y) · ∇log p(x|y)
    score_term = compute_score_posterior(X, Y, sigma)

    # Inner product: φ · ∇log p (sum over dimensions)
    phi_dot_score = vec(sum(phi_X .* score_term, dims=1))  # batchsize

    # Control variate: g(x,y) = div(φ) + φ·∇log p
    g_cv = jac_trace .+ phi_dot_score  # batchsize

    # Controlled estimator for each dimension
    # h_x is 2×batchsize, g_cv is batchsize (scalar CV applied to all dims)
    # Loss: minimize variance of (h - g - μ)
    residual = h_x .- reshape(g_cv .+ μ[1], 1, batchsize)  # 2×batchsize

    loss = mean(residual .^ 2)

    return loss, h_x, g_cv
end

for epoch = 1:args["max_epoch"]

    # Evaluate on validation set
    fval_eval[epoch] = loss_cv(CV, X_val, Y_val, μ, args["sigma"])[1]

    for (itr, (X, Y)) in enumerate(train_loader)
        Base.flush(Base.stdout)

        batchsize = size(X, 2)

        # Forward pass
        h_x = X
        phi_X, jac_trace = forward(X, Y, CV)
        score_term = compute_score_posterior(X, Y, Float32(args["sigma"]))
        phi_dot_score = vec(sum(phi_X .* score_term, dims=1))
        g_cv = jac_trace .+ phi_dot_score

        # Loss
        residual = h_x .- reshape(g_cv .+ μ[1], 1, batchsize)
        loss = mean(residual .^ 2)
        fval[(epoch-1)*num_batches+itr] = loss

        # Backward pass
        # ∂L/∂residual = 2 * residual / (2 * batchsize) = residual / batchsize
        Δresidual = residual ./ Float32(batchsize)

        # ∂residual/∂g_cv = -1 (broadcasted)
        Δg_cv = -vec(sum(Δresidual, dims=1))  # batchsize

        # ∂g_cv/∂jac_trace = 1
        jac_trace_grad_weight = Δg_cv

        # ∂g_cv/∂phi_dot_score = 1, and ∂phi_dot_score/∂phi_X = score_term
        Δphi_X = score_term .* reshape(Δg_cv, 1, batchsize)

        # Backward through CV network
        ΔX, _, ΔY = backward(Δphi_X, phi_X, Y, CV; jac_trace_grad_weight=jac_trace_grad_weight)

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Iteration, itr),
                (:CV_Loss, fval[(epoch-1)*num_batches+itr]),
                (:CV_Loss_eval, fval_eval[epoch]),
            ],
        )

        # Update params
        for param in get_params(CV)
            Flux.update!(opt, param.data, param.grad)
        end

        # Update μ manually
        # ∂L/∂μ = -2 * mean(residual)
        # residual_mean = mean(residual)
        # grad_mu = -2 * residual_mean
        # μ[1] -= args["lr"] * grad_mu

        clear_grad!(CV)
    end

    if epoch % 100 == 0 || epoch == args["max_epoch"]

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
                "CV" => CV |> cpu,
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
