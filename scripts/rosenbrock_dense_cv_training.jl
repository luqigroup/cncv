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
# n_cv=2: output 2 CVs, one for each dimension of h(x) = [x1, x2]
CV = DenseConditionalLayerCV(
    2,  # n_in: 2D input
    2,  # n_cond: 2D conditioning
    args["n_hidden"],
    args["n_layers"];
    activation=tanh,
    n_cv=2  # Vector-valued: 2 CVs (one per component)
)
# Scale down initial weights to make φ small initially
for p in get_params(CV)
    p.data .*= 0.01f0  # Start with small φ
end
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

# Learnable offsets (one per component)
μ = [0.0f0, 0.0f0] |> device

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
# With vector CVs: g_k for component k
function loss_cv(CV, X, Y, μ, sigma)
    batchsize = size(X, 2)
    n_cv = CV.n_cv

    # Quantity of interest: h(x) = x (2×batchsize)
    h_x = X

    # Forward through CV network to get jac_traces and phi_all (n_cv×batchsize)
    # phi_all contains the n_cv different coupling transformations φ_k
    jac_traces, phi_all = forward(X, Y, CV)

    # Compute score: ∇log p(x|y)
    score_term = compute_score_posterior(X, Y, sigma)

    # Compute control variates: g_k(x,y) = div(φ_k) + φ_k·∇log p
    g_cv = zeros(Float32, n_cv, batchsize)
    for k in 1:n_cv
        # Trace for k-th CV: div(φ_k)
        trace_k = jac_traces[k, :]

        # φ_k transformation (n_in × batchsize)
        phi_k = phi_all[k, :, :]

        # φ_k · ∇log p: inner product (sum over dimensions)
        phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))

        # Control variate: g_k = div(φ_k) + φ_k·∇log p
        g_cv[k, :] = trace_k .+ phi_dot_score_k
    end

    # Controlled estimator: h_i - g_i - μ_i
    residual = zeros(Float32, 2, batchsize)
    residual[1, :] = h_x[1, :] .- g_cv[1, :] .- μ[1]
    residual[2, :] = h_x[2, :] .- g_cv[2, :] .- μ[2]

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
        jac_traces, phi_all = forward(X, Y, CV)  # Get traces and phi_k transformations
        score_term = compute_score_posterior(X, Y, Float32(args["sigma"]))

        # Compute control variates (vector-valued)
        n_cv = CV.n_cv
        g_cv = zeros(Float32, n_cv, batchsize)
        for k in 1:n_cv
            trace_k = jac_traces[k, :]
            phi_k = phi_all[k, :, :]
            phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
            g_cv[k, :] = trace_k .+ phi_dot_score_k
        end

        # Loss with component-specific offsets
        residual = zeros(Float32, 2, batchsize)
        residual[1, :] = h_x[1, :] .- g_cv[1, :] .- μ[1]
        residual[2, :] = h_x[2, :] .- g_cv[2, :] .- μ[2]
        loss = mean(residual .^ 2)
        fval[(epoch-1)*num_batches+itr] = loss

        # Backward pass
        # ∂L/∂residual = 2 * residual / (2 * batchsize) = residual / batchsize
        Δresidual = residual ./ Float32(batchsize)

        # ∂residual[i]/∂g_cv[i] = -1 for each component i
        Δg_cv = -Δresidual  # 2×batchsize (n_cv×batch)

        # Now we need to backprop through: g_cv[k] = trace_k + phi_dot_score_k
        # where phi_dot_score_k = sum(phi_k .* score_term, dims=1)

        # ∂g_cv[k]/∂trace_k = 1
        jac_trace_grad_weights = Δg_cv  # n_cv×batchsize

        # ∂g_cv[k]/∂phi_dot_score_k = 1
        # ∂phi_dot_score_k/∂phi_k = score_term
        Δphi_all = zeros(Float32, n_cv, 2, batchsize)
        for k in 1:n_cv
            # Gradient wrt phi_k: ∂phi_dot_score_k/∂phi_k = score_term
            Δphi_all[k, :, :] = score_term .* reshape(Δg_cv[k, :], 1, batchsize)
        end

        # Backward through CV network (with both trace and phi gradients)
        ΔX, _, ΔY = backward(zeros(Float32, size(X)), X, Y, CV;
                             jac_trace_grad_weights=jac_trace_grad_weights,
                             phi_grad_weights=Δphi_all)

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Iteration, itr),
                (:CV_Loss, fval[(epoch-1)*num_batches+itr]),
                (:CV_Loss_eval, fval_eval[epoch]),
                (:μ1, μ[1]),
                (:μ2, μ[2])
            ],
        )

        # Update params
        for param in get_params(CV)
            Flux.update!(opt, param.data, param.grad)
        end

        # Update μ manually (component-wise)
        # ∂L/∂μ_i = -2 * mean(residual[i, :])
        for i in 1:2
            residual_mean_i = mean(residual[i, :])
            grad_mu_i = -2 * residual_mean_i
            μ[i] -= args["lr"] * grad_mu_i
        end

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
