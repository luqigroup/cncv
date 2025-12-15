# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using InvertibleNetworks: get_params, clear_grad!
using Rosenbrock
using Random
using ProgressMeter
using Flux
using LinearAlgebra
using Zygote

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

# Helper function: compute unnormalized log posterior and its gradient (the "a" vector)
# log p(x|y) ∝ log p(y|x) + log p(x)
#            = -1/(2σ²) ||x-y||² + log p_Rosenbrock(x)
function compute_score_posterior(X::AbstractArray{T,4}, Y::AbstractArray{T,4}, sigma::T) where T
    batchsize = size(X, 4)

    # Compute gradient via Zygote
    function log_posterior_unnorm(x)
        # Data likelihood term: -1/(2σ²) ||x-y||²
        likelihood_term = -sum((x .- Y) .^ 2) / (2 * sigma^2)

        # Prior term: log p_Rosenbrock(x)
        # Extract 2D points from spatial replication (just take first pixel)
        x_2d = zeros(T, 2, batchsize)
        for i = 1:batchsize
            x_2d[:, i] = x[1, 1, :, i]
        end
        prior_term = sum(logpdf(RB_dist, x_2d))

        return likelihood_term + prior_term
    end

    # Compute gradient
    grad_log_post = Zygote.gradient(log_posterior_unnorm, X)[1]

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

        # Compute loss and gradients
        loss, _, _ = loss_cv(CV, X, Y, μ, args["sigma"])
        fval[(epoch-1)*num_batches+itr] = loss

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

        # Update μ (simple gradient descent)
        # ∂L/∂μ = -2 * mean(residual)
        loss_grad = Zygote.gradient(μ_val -> loss_cv(CV, X, Y, μ_val, args["sigma"])[1], μ)[1]
        μ[1] -= args["lr"] * loss_grad[1]

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
