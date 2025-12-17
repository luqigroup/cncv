# Sampling and evaluation script for Gaussian ENSEMBLE CV
# Tests if forward-reverse ensemble fixes the variance problem
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using Statistics
using Random
using LinearAlgebra
using Distributions
using JLD2
using PyPlot
using Seaborn

# Import ensemble functions
import CNCV: forward

# Random seed
Random.seed!(123)

font_prop = set_plot_configs()[1]
args = read_config("gaussian_ensemble_cv_sampling.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Load trained ensemble model
loaded_keys = load_experiment(args, ["layer1", "layer2", "mu", "mu_prior", "Sigma_prior", "fval", "fval_eval"])
layer1 = loaded_keys["layer1"]
layer2 = loaded_keys["layer2"]
μ = loaded_keys["mu"]
μ_prior = loaded_keys["mu_prior"]
Σ_prior = loaded_keys["Sigma_prior"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]

n_dim = length(μ_prior)
σ_obs = Float32(args["sigma"])

println("\n=== Ensemble Model Info ===")
println("Layer 1 reverse_split: ", layer1.reverse_split)
println("Layer 2 reverse_split: ", layer2.reverse_split)
println("n_hidden: ", args["n_hidden"])
println("n_layers: ", args["n_layers"])
println("Observation noise σ: ", σ_obs)

# Compute exact posterior
function compute_exact_posterior(Y_obs, μ_prior, Σ_prior, σ)
    n = length(μ_prior)
    Σ_prior_inv = inv(Σ_prior)
    Σ_post = inv(Σ_prior_inv + (1/σ^2) * I)
    μ_post = Σ_post * (Σ_prior_inv * μ_prior + (1/σ^2) * Y_obs)
    return μ_post, Σ_post
end

# Generate test observation
println("\n=== Generating Test Data ===")
X_true = μ_prior .+ cholesky(Σ_prior).L * randn(Float32, n_dim)
Y_obs = X_true + σ_obs * randn(Float32, n_dim)

println("True parameter X: ", X_true')
println("Observation Y: ", Y_obs')

# Compute exact posterior
μ_post_exact, Σ_post_exact = compute_exact_posterior(Y_obs, μ_prior, Σ_prior, σ_obs)
println("\n=== Exact Posterior (Analytical) ===")
println("Posterior mean: ", μ_post_exact')
println("Posterior covariance:")
display(Σ_post_exact)
println()

# Sample from exact posterior
test_size = 10000
L_post = cholesky(Σ_post_exact).L
X_test = μ_post_exact .+ L_post * randn(Float32, n_dim, test_size)
Y_test = repeat(Y_obs, 1, test_size)

# Compute score function
Σ_prior_inv = inv(Σ_prior)
function compute_score_posterior(X, Y, μ_prior, Σ_prior_inv, σ)
    grad_likelihood = -(X .- Y) ./ (σ^2)
    grad_prior = -Σ_prior_inv * (X .- μ_prior)
    return grad_likelihood .+ grad_prior
end

score_term = compute_score_posterior(X_test, Y_test, μ_prior, Σ_prior_inv, σ_obs)

# Quantity of interest
h_x_all = X_test

# Forward through BOTH layers
println("\n=== Computing Control Variates from Both Layers ===")

# Layer 1
jac_traces_1, phi_all_1 = forward(X_test, Y_test, layer1)
g_cv_1 = zeros(Float32, n_dim, test_size)
for k in 1:n_dim
    trace_k = jac_traces_1[k, :]
    phi_k = phi_all_1[k, :, :]
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
    g_cv_1[k, :] = trace_k .+ phi_dot_score_k
end

# Layer 2
jac_traces_2, phi_all_2 = forward(X_test, Y_test, layer2)
g_cv_2 = zeros(Float32, n_dim, test_size)
for k in 1:n_dim
    trace_k = jac_traces_2[k, :]
    phi_k = phi_all_2[k, :, :]
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
    g_cv_2[k, :] = trace_k .+ phi_dot_score_k
end

# Combined (average)
g_cv_combined = (g_cv_1 .+ g_cv_2) ./ 2

println("Ensemble Control Variates:")
for k in 1:n_dim
    println("  CV $k: mean = ", mean(g_cv_combined[k, :]), ", std = ", std(g_cv_combined[k, :]))
    corr = cor(h_x_all[k, :], g_cv_combined[k, :])
    println("    Correlation with h_$k: ", corr)
end

# True posterior mean
true_mean = μ_post_exact

println("\n=== Posterior Inference Setup ===")
println("Analytical posterior mean E[X|Y]: ", true_mean')
println("Number of posterior samples: ", test_size)

println("\n=== Stein's Identity Check ===")
println("Ensemble CV should have E[g_k|Y] ≈ 0")
for k in 1:n_dim
    ratio = abs(mean(g_cv_combined[k, :])) / std(g_cv_combined[k, :])
    println("  Component $k: E[g_$k] / Std[g_$k] = ", round(ratio, digits=4), " (should be << 1)")
end

# Variance reduction analysis
println("\n=== Variance Reduction Analysis ===\n")

sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

vanilla_mse = zeros(Float32, length(sample_sizes), n_dim)
cv_combined_mse = zeros(Float32, length(sample_sizes), n_dim)

num_trials = 100

for (idx, n) in enumerate(sample_sizes)
    vanilla_estimates = zeros(Float32, num_trials, n_dim)
    cv_combined_estimates = zeros(Float32, num_trials, n_dim)

    for trial in 1:num_trials
        inds = randperm(test_size)[1:n]

        # Vanilla Monte Carlo
        for i in 1:n_dim
            vanilla_estimates[trial, i] = mean(h_x_all[i, inds])
        end

        # Combined CV
        for i in 1:n_dim
            cv_combined_estimates[trial, i] = mean(h_x_all[i, inds] .- g_cv_combined[i, inds])
        end
    end

    # Compute MSE
    for i in 1:n_dim
        vanilla_mse[idx, i] = mean((vanilla_estimates[:, i] .- true_mean[i]).^2)
        cv_combined_mse[idx, i] = mean((cv_combined_estimates[:, i] .- true_mean[i]).^2)
    end
end

println("\n=== Variance Reduction Summary ===")
for i in 1:n_dim
    vrf_comb = mean(cv_combined_mse[:, i]) / mean(vanilla_mse[:, i])

    println("Component $i: VRF = $(round(vrf_comb, digits=4))")
    if vrf_comb < 1.0
        improvement = (1.0 - vrf_comb) * 100
        println("  → Variance reduced by $(round(improvement, digits=1))%")
    else
        println("  → WARNING: Variance increased!")
    end
end

# ============= PLOTTING =============

# 1. Training loss plot
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
ylabel("MSE Loss")
xlabel("Epochs")
xlim([0.0, args["epoch"]])
wsave(joinpath(save_path, "training-obj.png"), fig)
close(fig)

# 2. MSE comparison plot across sample sizes
fig = figure("mse comparison", figsize = (10, 5))

for i in 1:n_dim
    subplot(1, n_dim, i)
    plot(sample_sizes, vanilla_mse[:, i], "o-", label = "Vanilla MC", linewidth = 2, markersize = 6, color = "#7f7f7f")
    plot(sample_sizes, cv_combined_mse[:, i], "d-", label = "Ensemble CV", linewidth = 2, markersize = 6, color = "#d62728")

    # Add 1/n reference line
    reference_1_over_n = vanilla_mse[1, i] * (sample_sizes[1] ./ sample_sizes)
    plot(sample_sizes, reference_1_over_n, "--", label = "1/n", linewidth = 1.5, color = "k", alpha = 0.5)

    xlabel("Sample size")
    ylabel("MSE")
    title("Component $i")
    xscale("log")
    yscale("log")
    legend()
    grid(true, alpha = 0.3)
end
tight_layout()
wsave(joinpath(save_path, "mse-comparison.png"), fig)
close(fig)

# 3. Violin plots of estimator distributions
# Use a moderate sample size to show variance
n_violin = 100
num_violin_trials = 200

fig = figure("estimator distributions", figsize = (10, 5))

for comp in 1:n_dim
    subplot(1, n_dim, comp)

    # Collect estimates from multiple trials
    vanilla_ests = zeros(Float32, num_violin_trials)
    combined_ests = zeros(Float32, num_violin_trials)

    for trial in 1:num_violin_trials
        inds = randperm(test_size)[1:n_violin]

        vanilla_ests[trial] = mean(h_x_all[comp, inds])
        combined_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_combined[comp, inds])
    end

    # Create violin plots
    positions = [1, 2]
    data_to_plot = [vanilla_ests, combined_ests]

    parts = plt.violinplot(data_to_plot, positions=positions, showmeans=true, showmedians=true)

    # Mark ground truth
    axhline(y = true_mean[comp], color = "k", linestyle = "--", linewidth = 2, label = "Ground truth")

    xticks(positions, ["Vanilla MC", "Ensemble CV"])
    ylabel("Estimate of E[X_$comp|Y]")
    title("Component $comp (n=$n_violin, $(num_violin_trials) trials)")
    legend()
    grid(true, alpha = 0.3, axis = "y")

    # Add variance text
    var_vanilla = var(vanilla_ests)
    var_combined = var(combined_ests)
    vrf = var_combined / var_vanilla
    reduction_pct = (1 - vrf) * 100
    text(0.5, 0.95, "VRF = $(round(vrf, digits=3))\nReduction = $(round(reduction_pct, digits=1))%",
         transform=gca().transAxes, horizontalalignment="center", verticalalignment="top",
         bbox=Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5))
end

tight_layout()
wsave(joinpath(save_path, "estimator-distributions.png"), fig)
close(fig)

# 3b. Variance Reduction Factor across sample sizes
fig = figure("vrf vs sample size", figsize = (10, 5))

for comp in 1:n_dim
    subplot(1, n_dim, comp)

    # Compute variance for each sample size by re-running trials
    vanilla_var_vs_n = zeros(Float32, length(sample_sizes))
    combined_var_vs_n = zeros(Float32, length(sample_sizes))
    vrf_vs_n = zeros(Float32, length(sample_sizes))

    for (idx, n) in enumerate(sample_sizes)
        vanilla_ests = zeros(Float32, num_trials)
        combined_ests = zeros(Float32, num_trials)

        for trial in 1:num_trials
            inds = randperm(test_size)[1:n]
            vanilla_ests[trial] = mean(h_x_all[comp, inds])
            combined_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_combined[comp, inds])
        end

        vanilla_var_vs_n[idx] = var(vanilla_ests)
        combined_var_vs_n[idx] = var(combined_ests)
        vrf_vs_n[idx] = combined_var_vs_n[idx] / vanilla_var_vs_n[idx]
    end

    plot(sample_sizes, vrf_vs_n, "o-", linewidth = 2, markersize = 8, label = "VRF (Combined CV)", color = "#d62728")
    axhline(y = 1.0, color = "k", linestyle = "--", linewidth = 1, label = "No reduction")

    xlabel("Sample size")
    ylabel("Variance Reduction Factor")
    title("Component $comp: VRF vs Sample Size")
    xscale("log")
    legend()
    grid(true, alpha = 0.3)
    ylim([0, max(1.5, maximum(vrf_vs_n) * 1.1)])
end

tight_layout()
wsave(joinpath(save_path, "vrf-vs-sample-size.png"), fig)
close(fig)

# 4. Variance comparison across methods
fig = figure("variance comparison", figsize = (10, 5))

for comp in 1:n_dim
    subplot(1, n_dim, comp)

    # Compute actual variances for a fixed sample size
    n_var = 100
    num_var_trials = 500

    vanilla_ests = zeros(Float32, num_var_trials)
    combined_ests = zeros(Float32, num_var_trials)

    for trial in 1:num_var_trials
        inds = randperm(test_size)[1:n_var]
        vanilla_ests[trial] = mean(h_x_all[comp, inds])
        combined_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_combined[comp, inds])
    end

    vars = [var(vanilla_ests), var(combined_ests)]
    labels = ["Vanilla MC", "Ensemble CV"]
    colors = ["#7f7f7f", "#d62728"]

    bar(1:2, vars, color = colors, alpha = 0.8, width = 0.6)
    xticks(1:2, labels)
    ylabel("Var[estimator]")
    title("Component $comp: Estimator Variance (n=$n_var)")
    grid(true, alpha = 0.3, axis = "y")

    # Add VRF annotation
    vrf = vars[2] / vars[1]
    reduction_pct = (1 - vrf) * 100
    text(0.5, 0.95, "VRF = $(round(vrf, digits=3))\nReduction = $(round(reduction_pct, digits=1))%",
         transform=gca().transAxes, horizontalalignment="center", verticalalignment="top",
         bbox=Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5))
end

tight_layout()
wsave(joinpath(save_path, "variance-comparison.png"), fig)
close(fig)

# 5. Scatter plots showing h vs g_cv
fig = figure("scatter plots", figsize = (12, 8))

for i in 1:n_dim
    # Layer 1
    subplot(n_dim, 3, (i-1)*3 + 1)
    scatter(h_x_all[i, :], g_cv_1[i, :], s = 1, alpha = 0.3)
    xlabel("h_$i(x)")
    ylabel("g_$i^(L1)")
    title("Layer 1: Component $i")
    grid(true, alpha = 0.3)

    # Layer 2
    subplot(n_dim, 3, (i-1)*3 + 2)
    scatter(h_x_all[i, :], g_cv_2[i, :], s = 1, alpha = 0.3)
    xlabel("h_$i(x)")
    ylabel("g_$i^(L2)")
    title("Layer 2: Component $i")
    grid(true, alpha = 0.3)

    # Combined
    subplot(n_dim, 3, (i-1)*3 + 3)
    scatter(h_x_all[i, :], g_cv_combined[i, :], s = 1, alpha = 0.3, color = "#d62728")
    xlabel("h_$i(x)")
    ylabel("g_$i^(combined)")
    title("Combined: Component $i")
    grid(true, alpha = 0.3)
end
tight_layout()
wsave(joinpath(save_path, "scatter-h-vs-gcv.png"), fig)
close(fig)

upload_to_dropbox(args["sim_name"])
