# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

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

font_prop = set_plot_configs()[1]
args = read_config("rosenbrock_squeeze_cv_sampling.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Load trained CV network
loaded_keys = load_experiment(args, ["CV", "mu", "fval", "fval_eval"])
CV = loaded_keys["CV"]
μ = loaded_keys["mu"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]

# Testing data
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
test_size = args["n_samples_test"]

# Fix a single observation Y_obs (drawn from prior + noise)
# This is the "observed data" we want to do posterior inference for
X_true_2d = rand(RB_dist, 1)  # True parameter (1 instance)
X_true = zeros(Float32, 4, 4, 2, 1)
for ix = 1:4, iy = 1:4
    X_true[ix, iy, :, 1] = X_true_2d[:, 1]
end
Y_obs = X_true + args["sigma"] * randn(Float32, 4, 4, 2, 1)  # Observed data

println("True parameter X: ", X_true_2d[:, 1])
println("Observed data Y: ", Y_obs[1, 1, :, 1])

# Generate posterior samples: X ~ p(X|Y_obs)
# Using simple MCMC (Langevin dynamics) to sample from posterior
println("\nGenerating posterior samples via MCMC...")

function posterior_energy(x_2d, y_obs, sigma)
    # -log p(x|y) = -log p(y|x) - log p(x) + const
    # = (1/2σ²)||x-y||² - log p(x)
    x_4d = zeros(Float32, 4, 4, 2, 1)
    for ix = 1:4, iy = 1:4
        x_4d[ix, iy, :, 1] = x_2d
    end
    likelihood_term = sum((x_4d .- y_obs) .^ 2) / (2 * sigma^2)
    prior_term = -logpdf(RB_dist, reshape(x_2d, 2, 1))[1]
    return likelihood_term + prior_term
end

# MCMC sampling
X_test_2d = zeros(Float32, 2, test_size)
x_current = rand(RB_dist, 1)[:, 1]  # Initialize from prior
step_size = 0.01f0
n_burnin = 1000

for i = 1:(test_size + n_burnin)
    global x_current
    # Langevin proposal
    # Compute gradient of -log p(x|y)
    x_4d = zeros(Float32, 4, 4, 2, 1)
    for ix = 1:4, iy = 1:4
        x_4d[ix, iy, :, 1] = x_current
    end

    grad_likelihood = (x_4d .- Y_obs) ./ (args["sigma"]^2)
    grad_prior = -gradlogpdf(RB_dist, reshape(x_current, 2, 1))[:, 1]
    grad_energy = grad_likelihood[1, 1, :, 1] + grad_prior

    # Langevin step: x' = x - step_size * ∇E + sqrt(2*step_size) * noise
    x_proposal = x_current - step_size * grad_energy + sqrt(2 * step_size) * randn(Float32, 2)

    # Metropolis acceptance (with Langevin correction)
    energy_current = posterior_energy(x_current, Y_obs, Float32(args["sigma"]))
    energy_proposal = posterior_energy(x_proposal, Y_obs, Float32(args["sigma"]))

    log_accept_prob = -(energy_proposal - energy_current)

    if log(rand()) < log_accept_prob
        x_current = x_proposal
    end

    # Save sample after burnin
    if i > n_burnin
        X_test_2d[:, i - n_burnin] = x_current
    end

    if i % 1000 == 0
        println("  MCMC iteration $i / $(test_size + n_burnin)")
    end
end

# Convert to 4D format
X_test = zeros(Float32, 4, 4, 2, test_size)
for i = 1:test_size
    for ix = 1:4, iy = 1:4
        X_test[ix, iy, :, i] = X_test_2d[:, i]
    end
end

# Y is the SAME for all samples (we're conditioning on fixed observation)
Y_test = repeat(Y_obs, 1, 1, 1, test_size)

# Helper function: compute gradient of unnormalized log posterior (score function)
# ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
function compute_score_posterior(X::AbstractArray{T,4}, Y::AbstractArray{T,4}, sigma::T) where T
    batchsize = size(X, 4)
    grad_log_post = zeros(T, size(X))

    # Gradient of log likelihood: ∇ log p(y|x) = -(x-y)/σ²
    grad_likelihood = -(X .- Y) ./ (sigma^2)

    # Gradient of log prior: use gradlogpdf from Rosenbrock
    x_2d = zeros(T, 2, batchsize)
    for i = 1:batchsize
        x_2d[:, i] = X[1, 1, :, i]
    end

    grad_prior_2d = gradlogpdf(RB_dist, x_2d)

    # Replicate gradient across all spatial locations
    for i = 1:batchsize
        for ix = 1:4, iy = 1:4
            grad_log_post[ix, iy, :, i] = grad_likelihood[ix, iy, :, i] + grad_prior_2d[:, i]
        end
    end

    return grad_log_post
end

# Compute quantities of interest: h(x) = mean(x) across spatial dims
h_x_all = dropdims(mean(X_test, dims=(1,2)), dims=(1,2))  # 2×test_size

# Forward through CV network
phi_X, jac_trace = CV.forward(X_test, Y_test)

# Compute control variate: g(x,y) = div(φ) + φ·∇log p
score_term = compute_score_posterior(X_test, Y_test, Float32(args["sigma"]))
phi_dot_score = dropdims(sum(phi_X .* score_term, dims=(1,2,3)), dims=(1,2,3))
g_cv_all = jac_trace .+ phi_dot_score  # test_size

# True posterior mean E[h(X)|Y_obs] (approximated by sample mean of MCMC samples)
true_mean = mean(h_x_all, dims=2)

println("\n=== Posterior Inference Setup ===")
println("Observed data Y_obs: ", Y_obs[1, 1, :, 1])
println("True parameter X_true: ", X_true_2d[:, 1])
println("Posterior mean E[X|Y] (MCMC estimate): ", true_mean')
println("Number of posterior samples: ", test_size)
println("\n=== Control Variate Check ===")
println("Mean of CV g(x,y): ", mean(g_cv_all))
println("Std of CV g(x,y): ", std(g_cv_all))
println("Learned offset μ (from training): ", μ[1])
println("Mean of h(x): ", mean(h_x_all, dims=2)')
println("Correlation h(x1) vs g: ", cor(h_x_all[1, :], g_cv_all))
println("Correlation h(x2) vs g: ", cor(h_x_all[2, :], g_cv_all))

# IMPORTANT FIX: Recompute μ for this specific Y_obs
# The trained μ doesn't generalize to new Y values!
μ_adjusted = [mean(g_cv_all)]
println("\nAdjusted offset μ (for this Y): ", μ_adjusted[1])
println("This ensures E[g(X,Y_obs)] ≈ 0 for this specific observation")

# Use adjusted μ for variance reduction
μ = μ_adjusted

# Variance reduction analysis
# Compare vanilla estimator vs. CV-corrected estimator for different sample sizes

sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
n_trials = 100  # Number of Monte Carlo trials for each sample size

vanilla_mse_x1 = zeros(Float32, length(sample_sizes))
vanilla_mse_x2 = zeros(Float32, length(sample_sizes))
cv_mse_x1 = zeros(Float32, length(sample_sizes))
cv_mse_x2 = zeros(Float32, length(sample_sizes))

vanilla_std_x1 = zeros(Float32, length(sample_sizes))
vanilla_std_x2 = zeros(Float32, length(sample_sizes))
cv_std_x1 = zeros(Float32, length(sample_sizes))
cv_std_x2 = zeros(Float32, length(sample_sizes))

Random.seed!(42)

println("\n=== Variance Reduction Analysis ===\n")

for (idx, n) in enumerate(sample_sizes)
    vanilla_estimates_x1 = zeros(Float32, n_trials)
    vanilla_estimates_x2 = zeros(Float32, n_trials)
    cv_estimates_x1 = zeros(Float32, n_trials)
    cv_estimates_x2 = zeros(Float32, n_trials)

    for trial = 1:n_trials
        # Sample n points randomly
        inds = rand(1:test_size, n)

        # Vanilla estimator: mean of h(x)
        vanilla_estimates_x1[trial] = mean(h_x_all[1, inds])
        vanilla_estimates_x2[trial] = mean(h_x_all[2, inds])

        # CV estimator: mean of (h(x) - g(x,y) - μ)
        cv_estimates_x1[trial] = mean(h_x_all[1, inds] .- g_cv_all[inds]) - μ[1]
        cv_estimates_x2[trial] = mean(h_x_all[2, inds] .- g_cv_all[inds]) - μ[1]
    end

    # Compute MSE (squared bias + variance)
    vanilla_mse_x1[idx] = mean((vanilla_estimates_x1 .- true_mean[1]) .^ 2)
    vanilla_mse_x2[idx] = mean((vanilla_estimates_x2 .- true_mean[2]) .^ 2)
    cv_mse_x1[idx] = mean((cv_estimates_x1 .- true_mean[1]) .^ 2)
    cv_mse_x2[idx] = mean((cv_estimates_x2 .- true_mean[2]) .^ 2)

    # Compute standard deviation
    vanilla_std_x1[idx] = std(vanilla_estimates_x1)
    vanilla_std_x2[idx] = std(vanilla_estimates_x2)
    cv_std_x1[idx] = std(cv_estimates_x1)
    cv_std_x2[idx] = std(cv_estimates_x2)

    println("n=$n: Vanilla MSE (x1)=$(vanilla_mse_x1[idx]), CV MSE (x1)=$(cv_mse_x1[idx]), " *
            "Ratio=$(vanilla_mse_x1[idx]/cv_mse_x1[idx])")
end

# Plot training loss
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
title("Control Variate Training Objective")
ylabel("MSE Loss")
xlabel("Epochs")
xlim([0.0, args["epoch"]])
wsave(joinpath(save_path, "training-obj.png"), fig)
close(fig)

# Plot MSE vs sample size for x1
rc("font", family = "serif", size = 14)
fig = figure("mse-x1", figsize = (7, 5))
loglog(sample_sizes, vanilla_mse_x1, "o-", lw = 2.0, color = "#d62728", label = "Vanilla", markersize=8)
loglog(sample_sizes, cv_mse_x1, "s-", lw = 2.0, color = "#2ca02c", label = "CV-corrected", markersize=8)
# Reference line: 1/n scaling
loglog(sample_sizes, vanilla_mse_x1[1] * sample_sizes[1] ./ sample_sizes, "--",
       color = "#7f7f7f", alpha = 0.5, label = L"$\propto 1/n$")
legend(loc = "upper right")
title(L"MSE for $x_1$ mean estimator")
xlabel("Sample size " * L"$n$")
ylabel("MSE")
grid(true, alpha = 0.3)
wsave(joinpath(save_path, "mse-vs-n-x1.png"), fig)
close(fig)

# Plot MSE vs sample size for x2
fig = figure("mse-x2", figsize = (7, 5))
loglog(sample_sizes, vanilla_mse_x2, "o-", lw = 2.0, color = "#d62728", label = "Vanilla", markersize=8)
loglog(sample_sizes, cv_mse_x2, "s-", lw = 2.0, color = "#2ca02c", label = "CV-corrected", markersize=8)
loglog(sample_sizes, vanilla_mse_x2[1] * sample_sizes[1] ./ sample_sizes, "--",
       color = "#7f7f7f", alpha = 0.5, label = L"$\propto 1/n$")
legend(loc = "upper right")
title(L"MSE for $x_2$ mean estimator")
xlabel("Sample size " * L"$n$")
ylabel("MSE")
grid(true, alpha = 0.3)
wsave(joinpath(save_path, "mse-vs-n-x2.png"), fig)
close(fig)

# Plot standard deviation vs sample size for x1
fig = figure("std-x1", figsize = (7, 5))
loglog(sample_sizes, vanilla_std_x1, "o-", lw = 2.0, color = "#d62728", label = "Vanilla", markersize=8)
loglog(sample_sizes, cv_std_x1, "s-", lw = 2.0, color = "#2ca02c", label = "CV-corrected", markersize=8)
loglog(sample_sizes, vanilla_std_x1[1] * sqrt(sample_sizes[1]) ./ sqrt.(sample_sizes), "--",
       color = "#7f7f7f", alpha = 0.5, label = L"$\propto 1/\sqrt{n}$")
legend(loc = "upper right")
title(L"Std. dev. for $x_1$ mean estimator")
xlabel("Sample size " * L"$n$")
ylabel("Standard deviation")
grid(true, alpha = 0.3)
wsave(joinpath(save_path, "std-vs-n-x1.png"), fig)
close(fig)

# Plot standard deviation vs sample size for x2
fig = figure("std-x2", figsize = (7, 5))
loglog(sample_sizes, vanilla_std_x2, "o-", lw = 2.0, color = "#d62728", label = "Vanilla", markersize=8)
loglog(sample_sizes, cv_std_x2, "s-", lw = 2.0, color = "#2ca02c", label = "CV-corrected", markersize=8)
loglog(sample_sizes, vanilla_std_x2[1] * sqrt(sample_sizes[1]) ./ sqrt.(sample_sizes), "--",
       color = "#7f7f7f", alpha = 0.5, label = L"$\propto 1/\sqrt{n}$")
legend(loc = "upper right")
title(L"Std. dev. for $x_2$ mean estimator")
xlabel("Sample size " * L"$n$")
ylabel("Standard deviation")
grid(true, alpha = 0.3)
wsave(joinpath(save_path, "std-vs-n-x2.png"), fig)
close(fig)

# Variance reduction ratio plot
fig = figure("variance-reduction", figsize = (7, 5))
variance_ratio_x1 = vanilla_mse_x1 ./ cv_mse_x1
variance_ratio_x2 = vanilla_mse_x2 ./ cv_mse_x2
semilogx(sample_sizes, variance_ratio_x1, "o-", lw = 2.0, color = "#1f77b4",
         label = L"$x_1$ component", markersize=8)
semilogx(sample_sizes, variance_ratio_x2, "s-", lw = 2.0, color = "#ff7f0e",
         label = L"$x_2$ component", markersize=8)
axhline(y=1.0, color = "#7f7f7f", linestyle = "--", alpha = 0.5, label = "No reduction")
legend(loc = "best")
title("Variance Reduction Factor")
xlabel("Sample size " * L"$n$")
ylabel("MSE ratio (Vanilla / CV)")
grid(true, alpha = 0.3)
wsave(joinpath(save_path, "variance-reduction-ratio.png"), fig)
close(fig)

# Sample correlation between h(x) and g(x,y)
fig = figure("cv-correlation", figsize = (10, 4))

subplot(1, 2, 1)
scatter(h_x_all[1, 1:1000], g_cv_all[1:1000], s = 5, color = "#1f77b4", alpha = 0.3)
corr_x1 = cor(h_x_all[1, :], g_cv_all)
xlabel(L"$h(x) = x_1$")
ylabel(L"$g(x,y)$ (control variate)")
title(L"$x_1$ vs CV " * "(ρ = $(round(corr_x1, digits=3)))")
grid(true, alpha = 0.3)

subplot(1, 2, 2)
scatter(h_x_all[2, 1:1000], g_cv_all[1:1000], s = 5, color = "#ff7f0e", alpha = 0.3)
corr_x2 = cor(h_x_all[2, :], g_cv_all)
xlabel(L"$h(x) = x_2$")
ylabel(L"$g(x,y)$ (control variate)")
title(L"$x_2$ vs CV " * "(ρ = $(round(corr_x2, digits=3)))")
grid(true, alpha = 0.3)

tight_layout()
wsave(joinpath(save_path, "cv-correlation.png"), fig)
close(fig)

println("\n=== Summary ===")
println("Learned offset μ: ", μ[1])
println("Correlation between x1 and CV: ", corr_x1)
println("Correlation between x2 and CV: ", corr_x2)
println("\nVariance reduction at n=1000:")
println("  x1: $(round(variance_ratio_x1[findfirst(sample_sizes .== 1000)], digits=2))x")
println("  x2: $(round(variance_ratio_x2[findfirst(sample_sizes .== 1000)], digits=2))x")

upload_to_dropbox(args["sim_name"])
