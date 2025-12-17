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

# Import ensemble functions
import CNCV: forward

# Random seed
Random.seed!(123)

# Load trained ensemble model
results_dir = datadir("gaussian_ensemble_cv")
files = readdir(results_dir, join=true)
latest_file = sort(files, by=mtime)[end]
println("Loading trained ensemble from: ", basename(latest_file))

data = load(latest_file)
layer1 = data["layer1"]
layer2 = data["layer2"]
μ = data["mu"]
args = Dict(key => data[key] for key in ["sigma", "n_hidden", "n_layers"])
μ_prior = data["mu_prior"]
Σ_prior = data["Sigma_prior"]

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

println("Layer 1 (forward split, transforms x₁):")
for k in 1:n_dim
    println("  CV $k: mean = ", mean(g_cv_1[k, :]), ", std = ", std(g_cv_1[k, :]))
    corr = cor(h_x_all[k, :], g_cv_1[k, :])
    println("    Correlation with h_$k: ", corr)
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

println("\nLayer 2 (reverse split, transforms x₂):")
for k in 1:n_dim
    println("  CV $k: mean = ", mean(g_cv_2[k, :]), ", std = ", std(g_cv_2[k, :]))
    corr = cor(h_x_all[k, :], g_cv_2[k, :])
    println("    Correlation with h_$k: ", corr)
end

# Combined (average)
g_cv_combined = (g_cv_1 .+ g_cv_2) ./ 2

println("\nCombined (averaged):")
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
println("Each CV should have E[g_k|Y] ≈ 0")
println("\nLayer 1:")
for k in 1:n_dim
    ratio = abs(mean(g_cv_1[k, :])) / std(g_cv_1[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

println("\nLayer 2:")
for k in 1:n_dim
    ratio = abs(mean(g_cv_2[k, :])) / std(g_cv_2[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

println("\nCombined:")
for k in 1:n_dim
    ratio = abs(mean(g_cv_combined[k, :])) / std(g_cv_combined[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

# Variance reduction analysis
println("\n=== Variance Reduction Analysis ===\n")

sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

vanilla_mse = zeros(Float32, length(sample_sizes), n_dim)
cv_layer1_mse = zeros(Float32, length(sample_sizes), n_dim)
cv_layer2_mse = zeros(Float32, length(sample_sizes), n_dim)
cv_combined_mse = zeros(Float32, length(sample_sizes), n_dim)

num_trials = 100

for (idx, n) in enumerate(sample_sizes)
    vanilla_estimates = zeros(Float32, num_trials, n_dim)
    cv_layer1_estimates = zeros(Float32, num_trials, n_dim)
    cv_layer2_estimates = zeros(Float32, num_trials, n_dim)
    cv_combined_estimates = zeros(Float32, num_trials, n_dim)

    for trial in 1:num_trials
        inds = randperm(test_size)[1:n]

        # Vanilla Monte Carlo
        for i in 1:n_dim
            vanilla_estimates[trial, i] = mean(h_x_all[i, inds])
        end

        # Layer 1 CV
        for i in 1:n_dim
            cv_layer1_estimates[trial, i] = mean(h_x_all[i, inds] .- g_cv_1[i, inds])
        end

        # Layer 2 CV
        for i in 1:n_dim
            cv_layer2_estimates[trial, i] = mean(h_x_all[i, inds] .- g_cv_2[i, inds])
        end

        # Combined CV
        for i in 1:n_dim
            cv_combined_estimates[trial, i] = mean(h_x_all[i, inds] .- g_cv_combined[i, inds])
        end
    end

    # Compute MSE
    for i in 1:n_dim
        vanilla_mse[idx, i] = mean((vanilla_estimates[:, i] .- true_mean[i]).^2)
        cv_layer1_mse[idx, i] = mean((cv_layer1_estimates[:, i] .- true_mean[i]).^2)
        cv_layer2_mse[idx, i] = mean((cv_layer2_estimates[:, i] .- true_mean[i]).^2)
        cv_combined_mse[idx, i] = mean((cv_combined_estimates[:, i] .- true_mean[i]).^2)
    end

    println("n=$n:")
    for i in 1:n_dim
        ratio_l1 = cv_layer1_mse[idx, i] / vanilla_mse[idx, i]
        ratio_l2 = cv_layer2_mse[idx, i] / vanilla_mse[idx, i]
        ratio_comb = cv_combined_mse[idx, i] / vanilla_mse[idx, i]
        println("  Component $i:")
        println("    Vanilla MSE: $(vanilla_mse[idx, i])")
        println("    Layer 1 CV MSE: $(cv_layer1_mse[idx, i]) (ratio: $ratio_l1)")
        println("    Layer 2 CV MSE: $(cv_layer2_mse[idx, i]) (ratio: $ratio_l2)")
        println("    Combined CV MSE: $(cv_combined_mse[idx, i]) (ratio: $ratio_comb)")
    end
end

println("\n=== Variance Reduction Summary ===")
for i in 1:n_dim
    vrf_l1 = mean(cv_layer1_mse[:, i]) / mean(vanilla_mse[:, i])
    vrf_l2 = mean(cv_layer2_mse[:, i]) / mean(vanilla_mse[:, i])
    vrf_comb = mean(cv_combined_mse[:, i]) / mean(vanilla_mse[:, i])

    println("\nComponent $i:")
    println("  Layer 1 VRF: $vrf_l1")
    if vrf_l1 < 1.0
        println("    → Reduces variance by $(round((1-vrf_l1)*100, digits=1))%")
    else
        println("    → Increases variance by $(round((vrf_l1-1)*100, digits=1))%")
    end

    println("  Layer 2 VRF: $vrf_l2")
    if vrf_l2 < 1.0
        println("    → Reduces variance by $(round((1-vrf_l2)*100, digits=1))%")
    else
        println("    → Increases variance by $(round((vrf_l2-1)*100, digits=1))%")
    end

    println("  COMBINED VRF: $vrf_comb")
    if vrf_comb < 1.0
        improvement = (1.0 - vrf_comb) * 100
        println("    → ✓ ENSEMBLE reduces variance by $(round(improvement, digits=1))%")
    else
        println("    → ✗ WARNING: Ensemble increases variance!")
    end
end

println("\n=== Comparison with Single-Layer Results ===")
println("(From previous gaussian_dense_cv run)")
println("Single layer Component 1: VRF ≈ 6.5 (549% increase)")
println("Single layer Component 2: VRF ≈ 12.5 (1146% increase)")
println("\nEnsemble improvement over single layer:")
for i in 1:n_dim
    vrf_comb = mean(cv_combined_mse[:, i]) / mean(vanilla_mse[:, i])
    single_vrf = i == 1 ? 6.5 : 12.5
    if vrf_comb < single_vrf
        println("Component $i: $(round((single_vrf - vrf_comb)/single_vrf * 100, digits=1))% better than single layer")
    end
end
