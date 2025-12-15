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
using Zygote

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

# Generate test samples: X ~ prior, Y = X + noise
X_test_2d = rand(RB_dist, test_size)  # 2×test_size
X_test = zeros(Float32, 4, 4, 2, test_size)
for i = 1:test_size
    for ix = 1:4, iy = 1:4
        X_test[ix, iy, :, i] = X_test_2d[:, i]
    end
end
Y_test = X_test + args["sigma"] * randn(Float32, 4, 4, 2, test_size)

# Helper function: compute score of unnormalized posterior
function compute_score_posterior(X::AbstractArray{T,4}, Y::AbstractArray{T,4}, sigma::T) where T
    batchsize = size(X, 4)

    function log_posterior_unnorm(x)
        likelihood_term = -sum((x .- Y) .^ 2) / (2 * sigma^2)
        x_2d = zeros(T, 2, batchsize)
        for i = 1:batchsize
            x_2d[:, i] = x[1, 1, :, i]
        end
        prior_term = sum(logpdf(RB_dist, x_2d))
        return likelihood_term + prior_term
    end

    grad_log_post = Zygote.gradient(log_posterior_unnorm, X)[1]
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

# True mean of h(x) under the posterior (approximated by sample mean)
true_mean = mean(h_x_all, dims=2)

println("True mean (x1, x2): ", true_mean')

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
