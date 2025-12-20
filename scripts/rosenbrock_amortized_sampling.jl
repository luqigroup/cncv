# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

using DrWatson
@quickactivate :CNCV

using Rosenbrock
using Random
using Distributions
using Statistics
using ProgressMeter
using PyPlot
using Seaborn
using LinearAlgebra
using Flux

font_prop = set_plot_configs()[1]
args = read_config("rosenbrock_amortized_sampling.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Define network
G = NetworkConditionalGlow(
    2,
    2,
    args["n_hidden"],
    args["depth"],
    args["K"];
    freeze_conv = true,
)

# Loading the experiment—only network weights and training loss
loaded_keys = load_experiment(args, ["G", "fval", "fval_eval"])
G = loaded_keys["G"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]

# Testing data
test_size = 10000
test_num = 4
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)

# Generate samples: rand returns 2×n_samples
X_test_2d = rand(RB_dist, test_size)  # 2×test_size
# Reshape to match network format: 1×1×2×test_size
X_test = reshape(X_test_2d, 1, 1, 2, test_size)
Y_test = X_test + args["sigma"] * randn(Float32, 1, 1, 2, test_size)

pair_wise_dist = 1.0f-1
while pair_wise_dist < 4.0f0
    global X_fixed_2d = rand(RB_dist, test_num)  # 2×test_num
    global X_fixed = reshape(X_fixed_2d, 1, 1, 2, test_num)
    global Y_fixed = X_fixed + args["sigma"] * randn(Float32, 1, 1, 2, test_num)
    global pair_wise_dist = 1.0f0
    for j = 1:test_num
        for k = 1:test_num
            if j != k
                global pair_wise_dist *= norm(Y_fixed[1, 1, :, j] - Y_fixed[1, 1, :, k], 1)
            end
        end
    end
    global pair_wise_dist = pair_wise_dist^(1.0f0 / test_num / (test_num - 1))
end

# Predicted samples
X_post = zeros(Float32, 1, 1, 2, test_size, test_num)
Zx = randn(Float32, 1, 1, 2, test_size)
for j = 1:test_num
    Zy_fixed = G.forward(Zx, repeat(Y_fixed[:, :, :, j:j], 1, 1, 1, test_size))[2]
    global X_post[:, :, :, :, j] = G.inverse(Zx, Zy_fixed)
end

X_sgld = zeros(Float32, 1, 1, 2, test_size + 1, test_num)
max_itr = 2 * test_size

# Updated objective function to work with new Rosenbrock format
obj(x, y) = begin
    # Reshape x from 1×1×2×n to 2×n for Rosenbrock
    x_2d = reshape(x, 2, size(x, 4))

    data_term = (1.0f0 / (2.0f0 * args["sigma"]^2.0f0)) * sum((x - y) .^ 2.0f0)
    # logpdf returns a vector, sum it up and negate
    prior_term = -sum(logpdf(RB_dist, x_2d))

    return data_term + prior_term
end

for j = 1:test_num
    f(x) = obj(x, Y_fixed[:, :, :, j:j])
    # Sampling.
    X_sgld[:, :, :, :, j] = MCMC_sampler(
        max_itr,
        randn(Float32, 1, 1, 2, 1),
        f;
        lr = 5.0f0,
        lr_final = 1.0f-1,
        thinning = 1,
    )
end

# Training loss
fig = figure("training logs", figsize = (7, 4))
if args["epoch"] == args["max_epoch"]
    plot(
        range(0, args["epoch"], length = length(fval)),
        fval,
        color = "#4a4a4a",
        label = "training loss",
    )
    plot(
        range(0, args["epoch"], length = length(fval_eval)),
        fval_eval,
        color = "#a1a1a1",
        label = "validation loss",
    )
else
    plot(
        range(0, args["epoch"], length = length(fval[1:findfirst(fval .== 0.0f0)-1])),
        fval[1:findfirst(fval .== 0.0f0)-1],
        color = "#4a4a4a",
        label = "training loss",
    )
    plot(
        range(
            0,
            args["epoch"],
            length = length(fval_eval[1:findfirst(fval_eval .== 0.0f0)-1]),
        ),
        fval_eval[1:findfirst(fval_eval .== 0.0f0)-1],
        color = "#a1a1a1",
        label = "validation loss",
    )
end
legend()
title("Training objective")
ylabel(L"KL divergence + $const.$")
xlabel("Epochs")
xlim([0.0, args["epoch"]])
wsave(joinpath(save_path, "training-obj.png"), fig)
close(fig)

fig = figure("rosenbrock samples", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(X_test[1, 1, 1, :], X_test[1, 1, 2, :], s = 0.5, color = "#000000", alpha = 0.15)
scatter(
    X_fixed[1, 1, 1, :],
    X_fixed[1, 1, 2, :],
    s = 50.0,
    color = "#000000",
    marker = "^",
    label = "Testing model instances",
)
grid(false)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.legend(loc = "upper right", ncol = 2)
ax.set_title("Prior distribution")
wsave(joinpath(save_path, "model.png"), fig)
close(fig)

fig = figure("data samples", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(Y_test[1, 1, 1, :], Y_test[1, 1, 2, :], s = 0.5, color = "#000000", alpha = 0.15)
scatter(
    Y_fixed[1, 1, 1, :],
    Y_fixed[1, 1, 2, :],
    s = 50.0,
    color = "#000000",
    marker = "v",
    label = "Testing data instances",
)
grid(false)
ax.legend(loc = "upper right", ncol = 2)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$y_2$")
ax.set_xlabel(L"$y_1$")
ax.set_title("Data distribution")
wsave(joinpath(save_path, "data.png"), fig)
close(fig)

fig = figure("rosenbrock samples", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
label3 = scatter(
    X_sgld[1, 1, 1, :, :],
    X_sgld[1, 1, 2, :, :],
    s = 0.5,
    color = "#000000",
    alpha = 0.1,
    label = "MCMC",
)
grid(true)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("MCMC")
wsave(joinpath(save_path, "mcmc.png"), fig)
close(fig)

fig = figure("rosenbrock samples", figsize = (5, 5))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
label3 = scatter(
    X_post[1, 1, 1, :, :],
    X_post[1, 1, 2, :, :],
    s = 0.5,
    color = "#000000",
    alpha = 0.1,
    label = "Predicted",
)
grid(true)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("Amortized variational inference")
wsave(joinpath(save_path, "avi-samples.png"), fig)
close(fig)

# Likelihood comparison plot
# Compute log-likelihood using the network (exact_likelihood function)
# For conditional network, we need to provide both X and Y
# We'll use one of the fixed test cases for this comparison
j_test = 1
Y_test_fixed = repeat(Y_fixed[:, :, :, j_test:j_test], 1, 1, 1, test_size)
loglike_G = exact_likelihood(G, X_post[:, :, :, :, j_test], Y_test_fixed)

# Compute true log-likelihood: log p(x|y) = log p(y|x) + log p(x) - log p(y)
# Since log p(y) is constant for a fixed y, we can ignore it
# log p(x|y) ∝ log p(y|x) + log p(x)
# log p(y|x) = -0.5 * ||y - x||^2 / sigma^2 - log(sqrt(2*pi)*sigma) * dim
# log p(x) = logpdf(RB_dist, x)

# Reshape for Rosenbrock distribution
X_post_2d = reshape(X_post[:, :, :, :, j_test], 2, test_size)

# Compute log p(x) - prior
loglike_prior = logpdf(RB_dist, X_post_2d)

# Compute log p(y|x) - likelihood
data_diff = X_post[:, :, :, :, j_test] - Y_test_fixed
loglike_data = -0.5f0 .* sum(data_diff .^ 2, dims = [1, 2, 3])[1, 1, 1, :] ./ (args["sigma"]^2)
# Add normalization constant
dim = 2
loglike_data = loglike_data .- dim * log(sqrt(2.0f0 * π) * args["sigma"])

# Total true log-likelihood (up to constant)
loglike_true = loglike_prior + loglike_data

# Compute KL divergence + constant
kl_divergence = mean(loglike_true - loglike_G)
println("KL divergence + const. (test case $j_test): ", kl_divergence)

# Create histogram comparison plot
fig = figure("histogram", figsize = (7, 2.5))
ax = histplot(
    loglike_true,
    kde = true,
    bins = 50,
    label = "true log-likelihood",
    alpha = 0.8,
    color = "#ff8800",
)
histplot(
    loglike_G,
    kde = true,
    bins = 50,
    label = "predicted log-likelihood",
    alpha = 0.8,
    color = "#00b4ba",
)
for label in ax.get_xticklabels()
    label.set_fontproperties(font_prop)
end
for label in ax.get_yticklabels()
    label.set_fontproperties(font_prop)
end
ax.set_xlabel("Log-likelihood", fontproperties = font_prop)
ax.legend(prop = font_prop)
wsave(joinpath(save_path, "log-like-hist.png"), fig)
close(fig)

# Gradient comparison plot (score function sanity check)
# Compare learned gradlogpdf from network vs true analytical gradlogpdf
println("Computing score function comparison...")

# Use the same test case for gradient comparison
j_test = 1
Y_test_fixed = repeat(Y_fixed[:, :, :, j_test:j_test], 1, 1, 1, test_size)
X_test_subset = X_post[:, :, :, :, j_test]

# Compute LEARNED gradlogpdf using network's exact_score function
ΔX_learned = exact_score(G, X_test_subset, Y_test_fixed)  # This is ∇ log p(x|y)

# Compute TRUE gradlogpdf using analytical formulas
# ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
X_test_2d = reshape(X_test_subset, 2, test_size)
Y_test_2d = reshape(Y_test_fixed, 2, test_size)

# Gradient of Gaussian likelihood: ∇ log p(y|x) = -(x-y)/σ²
grad_likelihood = -(X_test_2d .- Y_test_2d) ./ (args["sigma"]^2)

# Gradient of Rosenbrock prior
grad_prior = gradlogpdf(RB_dist, X_test_2d)

# Total true gradient
grad_true = grad_likelihood .+ grad_prior

# Reshape learned gradient for comparison
grad_learned = reshape(ΔX_learned, 2, test_size)

# Create component-wise scatter plots comparing learned vs true gradients
fig = figure("gradient comparison", figsize = (12, 5))

# Component 1
subplot(1, 2, 1)
scatter(grad_true[1, :], grad_learned[1, :], s = 2, alpha = 0.5, color = "#00b4ba")
ref_line_vals = range(minimum(grad_true[1, :]), maximum(grad_true[1, :]), length = 100)
plot(ref_line_vals, ref_line_vals, "k--", linewidth = 2, label = "y=x")
xlabel("True ∇log p(x₁|y)", fontproperties = font_prop)
ylabel("Learned ∇log p(x₁|y)", fontproperties = font_prop)
title("Component 1: Score Function", fontproperties = font_prop)
legend(prop = font_prop)
grid(true, alpha = 0.3)
axis("equal")

# Component 2
subplot(1, 2, 2)
scatter(grad_true[2, :], grad_learned[2, :], s = 2, alpha = 0.5, color = "#ff8800")
ref_line_vals = range(minimum(grad_true[2, :]), maximum(grad_true[2, :]), length = 100)
plot(ref_line_vals, ref_line_vals, "k--", linewidth = 2, label = "y=x")
xlabel("True ∇log p(x₂|y)", fontproperties = font_prop)
ylabel("Learned ∇log p(x₂|y)", fontproperties = font_prop)
title("Component 2: Score Function", fontproperties = font_prop)
legend(prop = font_prop)
grid(true, alpha = 0.3)
axis("equal")

tight_layout()
wsave(joinpath(save_path, "gradient-comparison.png"), fig)
close(fig)

# Create 2D overlay plot: gradient vectors in 2D space
fig = figure("gradient field 2D", figsize = (7, 7))
ax = fig.add_subplot(111)

# Plot true gradients
scatter(grad_true[1, :], grad_true[2, :], s = 3, alpha = 0.4,
        color = "#ff8800", marker = "o", label = "True ∇log p(x|y)")

# Plot learned gradients
scatter(grad_learned[1, :], grad_learned[2, :], s = 3, alpha = 0.4,
        color = "#00b4ba", marker = "^", label = "Learned ∇log p(x|y)")

xlabel("∇log p(x₁|y)", fontproperties = font_prop)
ylabel("∇log p(x₂|y)", fontproperties = font_prop)
title("Score Function: True vs Learned", fontproperties = font_prop)
legend(prop = font_prop)
grid(true, alpha = 0.3)
axis("equal")

tight_layout()
wsave(joinpath(save_path, "gradient-field-2d.png"), fig)
close(fig)

# Print correlation between learned and true gradients
for i in 1:2
    corr = cor(grad_true[i, :], grad_learned[i, :])
    println("Component $i gradient correlation: ", round(corr, digits=4))
end

rc("font", family = "serif", size = 16)
font_prop =
    matplotlib.font_manager.FontProperties(family = "serif", style = "normal", size = 18)
dq = 1.0f-2
for j = 1:test_num
    mcmc_quantile = quantile(X_sgld[1, 1, 1, :, j], dq:dq:1-dq)
    avi_quantile = quantile(X_post[1, 1, 1, :, j], dq:dq:1-dq)
    fig = figure("qq", figsize = (5, 5))
    plot(mcmc_quantile, avi_quantile, "o", lw = 2.0, color = "k", alpha = 0.6)
    ref_line = range(mcmc_quantile[1], mcmc_quantile[end], length = 100)
    plot(ref_line, ref_line, lw = 1.0, color = "k", alpha = 1)
    title("Q-Q plot for " * L"$x_1$" * " component")
    xlabel("MCMC quantiles")
    ylabel("Predicted quantiles")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    wsave(joinpath(save_path, "qq1" * string(j) * ".png"), fig)
    close(fig)

    mcmc_quantile = quantile(X_sgld[1, 1, 2, :, j], dq:dq:1-dq)
    avi_quantile = quantile(X_post[1, 1, 2, :, j], dq:dq:1-dq)

    fig = figure("qq2", figsize = (5, 5))
    plot(mcmc_quantile, avi_quantile, "o", lw = 2.0, color = "k", alpha = 0.6)
    ref_line = range(mcmc_quantile[1], mcmc_quantile[end], length = 100)
    plot(ref_line, ref_line, lw = 1.0, color = "k", alpha = 1)
    title("Q-Q plot for " * L"$x_2$" * " component")
    xlabel("MCMC quantiles")
    ylabel("Predicted quantiles")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    wsave(joinpath(save_path, "qq2" * string(j) * ".png"), fig)
    close(fig)
end

upload_to_dropbox(args["sim_name"])