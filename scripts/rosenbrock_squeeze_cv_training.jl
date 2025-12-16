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

# Random seed
Random.seed!(19)

args = read_config("rosenbrock_squeeze_cv_training.json")
args = parse_input_args(args)

device = cpu # InvertibleNetworks.CUDA.functional() ? gpu : cpu

# Define control variate network using SqueezeConditionalLayerBasicCV
# Note: Input is 2D, but we need to reshape to 1×1×2×batchsize for the network
# After squeezing with pattern="column", this becomes 1×1×8×batchsize (since we need 4 channels before squeezing for 2D)
# Actually, we work in 2D, so we can't squeeze. Let's use ConditionalLayerBasicCV directly.
# Wait, the user specifically asked for SqueezeConditionalLayerBasicCV.
# For 2D Rosenbrock, we need at least nx=2, ny=2 spatial dims to squeeze.
# Let me use 2×2×2 input (spatial 2×2, 2 channels), then squeeze to 1×1×8

# Actually, let's keep it simple and use 4×4 spatial dimensions with k=2 channels
# After squeezing: 2×2×8

CV = SqueezeConditionalLayerBasicCV(
    2,  # n_in channels before squeezing
    2,  # n_cond channels
    args["n_hidden"];
    k1=3,
    k2=3,
    p1=1,
    p2=1,
    pattern=args["pattern"]
)
CV = CV |> device

# Training data number
ntrain = 5120

# Generate training data from Rosenbrock prior + Gaussian likelihood
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_train_2d = rand(RB_dist, ntrain)  # 2×ntrain

# Reshape to 4×4×2×ntrain: replicate across spatial dims
X_train = zeros(Float32, 4, 4, 2, ntrain)
for i = 1:ntrain
    for ix = 1:4, iy = 1:4
        X_train[ix, iy, :, i] = X_train_2d[:, i]
    end
end

# Add Gaussian noise to create observations Y = X + noise
Y_train = X_train + args["sigma"] * randn(Float32, 4, 4, 2, ntrain)
X_train = X_train |> device
Y_train = Y_train |> device

# Validation data
nval = 512
X_val_2d = rand(RB_dist, nval)
X_val = zeros(Float32, 4, 4, 2, nval)
for i = 1:nval
    for ix = 1:4, iy = 1:4
        X_val[ix, iy, :, i] = X_val_2d[:, i]
    end
end
Y_val = X_val + args["sigma"] * randn(Float32, 4, 4, 2, nval)
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
μ = [0.0f0] |> device

# Training log keeper
fval = zeros(Float32, num_batches * args["max_epoch"])
fval_eval = zeros(Float32, args["max_epoch"])

# Helper function: compute gradient of unnormalized log posterior (score function)
# ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
# where:
#   ∇ log p(y|x) = -(x-y)/σ² (Gaussian likelihood)
#   ∇ log p(x) = gradlogpdf from Rosenbrock distribution
function compute_score_posterior(X::AbstractArray{T,4}, Y::AbstractArray{T,4}, sigma::T) where T
    batchsize = size(X, 4)
    grad_log_post = zeros(T, size(X))

    # Gradient of log likelihood: ∇ log p(y|x) = -(x-y)/σ²
    grad_likelihood = -(X .- Y) ./ (sigma^2)

    # Gradient of log prior: use gradlogpdf from Rosenbrock
    # Extract 2D points from spatial replication (just take first pixel)
    x_2d = zeros(T, 2, batchsize)
    for i = 1:batchsize
        x_2d[:, i] = X[1, 1, :, i]
    end

    # gradlogpdf returns 2×batchsize
    grad_prior_2d = gradlogpdf(RB_dist, x_2d)

    # Replicate gradient across all spatial locations (since X is replicated)
    for i = 1:batchsize
        for ix = 1:4, iy = 1:4
            grad_log_post[ix, iy, :, i] = grad_likelihood[ix, iy, :, i] + grad_prior_2d[:, i]
        end
    end

    return grad_log_post
end

# Loss function for CV training: minimize E[(h(x) - (div(φ) + φ·∇log p) - μ)²]
# Here h(x) = x (identity, to estimate the mean)
function loss_cv(CV, X, Y, μ, sigma)
    batchsize = size(X, 4)

    # Quantity of interest: h(x) = mean of x across spatial dims
    h_x = dropdims(mean(X, dims=(1,2)), dims=(1,2))  # 2×batchsize

    # Forward through CV network to get jac_trace (which is div(φ))
    phi_X, jac_trace = CV.forward(X, Y)

    # Compute φ(x,y) · ∇log p(x|y)
    score_term = compute_score_posterior(X, Y, sigma)

    # Inner product: φ · ∇log p
    # phi_X has same shape as X: 4×4×2×batchsize
    # score_term has same shape
    phi_dot_score = sum(phi_X .* score_term, dims=(1,2,3))  # 1×1×1×batchsize
    phi_dot_score = dropdims(phi_dot_score, dims=(1,2,3))  # batchsize

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

        batchsize = size(X, 4)

        # Forward pass
        h_x = dropdims(mean(X, dims=(1,2)), dims=(1,2))  # 2×batchsize
        phi_X, jac_trace = CV.forward(X, Y)
        score_term = compute_score_posterior(X, Y, Float32(args["sigma"]))
        phi_dot_score = dropdims(sum(phi_X .* score_term, dims=(1,2,3)), dims=(1,2,3))
        g_cv = jac_trace .+ phi_dot_score

        # Loss
        residual = h_x .- reshape(g_cv .+ μ[1], 1, batchsize)
        loss = mean(residual .^ 2)
        fval[(epoch-1)*num_batches+itr] = loss

        # Backward pass
        # ∂L/∂residual = 2 * residual / (2 * batchsize) = residual / batchsize
        Δresidual = residual ./ Float32(batchsize)

        # ∂residual/∂g_cv = -1 (broadcasted)
        Δg_cv = -dropdims(sum(Δresidual, dims=1), dims=1)  # batchsize

        # ∂g_cv/∂jac_trace = 1
        jac_trace_grad_weight = Δg_cv

        # ∂g_cv/∂phi_dot_score = 1, and ∂phi_dot_score/∂phi_X = score_term
        Δphi_X = score_term .* reshape(Δg_cv, 1, 1, 1, batchsize)

        # Backward through CV network
        ΔX, _, ΔY = CV.backward(Δphi_X, phi_X, Y; jac_trace_grad_weight=jac_trace_grad_weight)

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
        residual_mean = mean(residual)
        grad_mu = -2 * residual_mean
        μ[1] -= args["lr"] * grad_mu

        clear_grad!(CV)
    end

    if epoch % 10 == 0 || epoch == args["max_epoch"]

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
