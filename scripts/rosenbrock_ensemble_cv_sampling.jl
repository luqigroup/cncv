# Sampling and evaluation script for Rosenbrock ENSEMBLE CV
# Tests if forward-reverse ensemble fixes the variance problem for Rosenbrock
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using Statistics
using Random
using LinearAlgebra
using Rosenbrock
using JLD2

# Import ensemble functions
import CNCV: forward

# Random seed
Random.seed!(123)

# Load trained ensemble model
results_dir = datadir("rosenbrock-ensemble-cv")
files = readdir(results_dir, join=true)
latest_file = sort(files, by=mtime)[end]
println("Loading trained ensemble from: ", basename(latest_file))

data = load(latest_file)
layer1 = data["layer1"]
layer2 = data["layer2"]
μ = data["mu"]
args = Dict(key => data[key] for key in ["sigma", "n_hidden", "n_layers", "n_samples_test"])

σ_obs = Float32(args["sigma"])
test_size = args["n_samples_test"]

println("\n=== Ensemble Model Info ===")
println("Layer 1 reverse_split: ", layer1.reverse_split)
println("Layer 2 reverse_split: ", layer2.reverse_split)
println("n_hidden: ", args["n_hidden"])
println("n_layers: ", args["n_layers"])
println("Observation noise σ: ", σ_obs)

# Generate test data from Rosenbrock distribution
println("\n=== Generating Test Data ===")
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)

# Sample from Rosenbrock prior
X_test = rand(RB_dist, test_size)

# Generate observation (using first sample as "true" parameter)
X_true = X_test[:, 1]
Y_obs = X_true + σ_obs * randn(Float32, 2)

println("True parameter X: ", X_true')
println("Observation Y: ", Y_obs')

# For inference, we need samples from the POSTERIOR p(X|Y)
# For Rosenbrock, we don't have analytical posterior, so we use MCMC samples
# For now, use prior samples conditioned on the observation via likelihood weighting
# (This is approximate but allows us to test the CV performance)

# Create conditioning: all samples observe the same Y
Y_test = repeat(Y_obs, 1, test_size)

# Compute score function
function compute_score_posterior(X, Y, sigma)
    grad_likelihood = -(X .- Y) ./ (sigma^2)
    grad_prior = gradlogpdf(RB_dist, X)
    return grad_likelihood .+ grad_prior
end

score_term = compute_score_posterior(X_test, Y_test, σ_obs)

# Quantity of interest: h(x) = x
h_x_all = X_test

# Forward through BOTH layers
println("\n=== Computing Control Variates from Both Layers ===")

# Layer 1
jac_traces_1, phi_all_1 = forward(X_test, Y_test, layer1)
g_cv_1 = zeros(Float32, 2, test_size)
for k in 1:2
    trace_k = jac_traces_1[k, :]
    phi_k = phi_all_1[k, :, :]
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
    g_cv_1[k, :] = trace_k .+ phi_dot_score_k
end

println("Layer 1 (forward split, transforms x₁):")
for k in 1:2
    println("  CV $k: mean = ", mean(g_cv_1[k, :]), ", std = ", std(g_cv_1[k, :]))
    corr = cor(h_x_all[k, :], g_cv_1[k, :])
    println("    Correlation with h_$k: ", corr)
end

# Layer 2
jac_traces_2, phi_all_2 = forward(X_test, Y_test, layer2)
g_cv_2 = zeros(Float32, 2, test_size)
for k in 1:2
    trace_k = jac_traces_2[k, :]
    phi_k = phi_all_2[k, :, :]
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
    g_cv_2[k, :] = trace_k .+ phi_dot_score_k
end

println("\nLayer 2 (reverse split, transforms x₂):")
for k in 1:2
    println("  CV $k: mean = ", mean(g_cv_2[k, :]), ", std = ", std(g_cv_2[k, :]))
    corr = cor(h_x_all[k, :], g_cv_2[k, :])
    println("    Correlation with h_$k: ", corr)
end

# Combined (average)
g_cv_combined = (g_cv_1 .+ g_cv_2) ./ 2

println("\nCombined (averaged):")
for k in 1:2
    println("  CV $k: mean = ", mean(g_cv_combined[k, :]), ", std = ", std(g_cv_combined[k, :]))
    corr = cor(h_x_all[k, :], g_cv_combined[k, :])
    println("    Correlation with h_$k: ", corr)
end

println("\n=== Stein's Identity Check ===")
println("Each CV should have E[g_k|Y] ≈ 0")
println("\nLayer 1:")
for k in 1:2
    ratio = abs(mean(g_cv_1[k, :])) / std(g_cv_1[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

println("\nLayer 2:")
for k in 1:2
    ratio = abs(mean(g_cv_2[k, :])) / std(g_cv_2[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

println("\nCombined:")
for k in 1:2
    ratio = abs(mean(g_cv_combined[k, :])) / std(g_cv_combined[k, :])
    println("  E[g_$k] / Std[g_$k] = ", ratio, " (should be << 1)")
end

# Variance analysis (relative variance comparison)
println("\n=== Variance Comparison ===")

# We don't know true posterior mean for Rosenbrock, so we compare VARIANCES
# Var[h - g] vs Var[h]

println("\nComponent 1:")
var_h_1 = var(h_x_all[1, :])
var_h_minus_g_l1_1 = var(h_x_all[1, :] .- g_cv_1[1, :])
var_h_minus_g_l2_1 = var(h_x_all[1, :] .- g_cv_2[1, :])
var_h_minus_g_comb_1 = var(h_x_all[1, :] .- g_cv_combined[1, :])

println("  Var[h₁]: ", var_h_1)
println("  Var[h₁ - g₁^(layer1)]: ", var_h_minus_g_l1_1, " (VRF: ", var_h_minus_g_l1_1/var_h_1, ")")
println("  Var[h₁ - g₁^(layer2)]: ", var_h_minus_g_l2_1, " (VRF: ", var_h_minus_g_l2_1/var_h_1, ")")
println("  Var[h₁ - g₁^(combined)]: ", var_h_minus_g_comb_1, " (VRF: ", var_h_minus_g_comb_1/var_h_1, ")")

if var_h_minus_g_comb_1 < var_h_1
    reduction = (1 - var_h_minus_g_comb_1/var_h_1) * 100
    println("  → ✓ Combined reduces variance by $(round(reduction, digits=1))%")
else
    increase = (var_h_minus_g_comb_1/var_h_1 - 1) * 100
    println("  → ✗ Combined increases variance by $(round(increase, digits=1))%")
end

println("\nComponent 2:")
var_h_2 = var(h_x_all[2, :])
var_h_minus_g_l1_2 = var(h_x_all[2, :] .- g_cv_1[2, :])
var_h_minus_g_l2_2 = var(h_x_all[2, :] .- g_cv_2[2, :])
var_h_minus_g_comb_2 = var(h_x_all[2, :] .- g_cv_combined[2, :])

println("  Var[h₂]: ", var_h_2)
println("  Var[h₂ - g₂^(layer1)]: ", var_h_minus_g_l1_2, " (VRF: ", var_h_minus_g_l1_2/var_h_2, ")")
println("  Var[h₂ - g₂^(layer2)]: ", var_h_minus_g_l2_2, " (VRF: ", var_h_minus_g_l2_2/var_h_2, ")")
println("  Var[h₂ - g₂^(combined)]: ", var_h_minus_g_comb_2, " (VRF: ", var_h_minus_g_comb_2/var_h_2, ")")

if var_h_minus_g_comb_2 < var_h_2
    reduction = (1 - var_h_minus_g_comb_2/var_h_2) * 100
    println("  → ✓ Combined reduces variance by $(round(reduction, digits=1))%")
else
    increase = (var_h_minus_g_comb_2/var_h_2 - 1) * 100
    println("  → ✗ Combined increases variance by $(round(increase, digits=1))%")
end

# Summary
println("\n=== Summary ===")
vrf_comb_1 = var_h_minus_g_comb_1 / var_h_1
vrf_comb_2 = var_h_minus_g_comb_2 / var_h_2

println("\nOverall Variance Reduction Factors:")
println("  Component 1: VRF = ", vrf_comb_1)
println("  Component 2: VRF = ", vrf_comb_2)

if vrf_comb_1 < 1.0 && vrf_comb_2 < 1.0
    avg_reduction = ((1 - vrf_comb_1) + (1 - vrf_comb_2)) / 2 * 100
    println("\n✓ SUCCESS! Average variance reduction: $(round(avg_reduction, digits=1))%")
else
    println("\n✗ WARNING: Ensemble did not achieve variance reduction on both components")
end

println("\n=== Comparison with Single-Layer Results ===")
println("(From previous rosenbrock_dense_cv run)")
println("Single layer Component 1: VRF ≈ 26.5 (2554% increase)")
println("Single layer Component 2: VRF ≈ 12.7 (1171% increase)")

if vrf_comb_1 < 26.5 && vrf_comb_2 < 12.7
    println("\n✓ Ensemble is MUCH better than single layer!")
    println("  Component 1: $(round((26.5 - vrf_comb_1)/26.5 * 100, digits=1))% improvement")
    println("  Component 2: $(round((12.7 - vrf_comb_2)/12.7 * 100, digits=1))% improvement")
end

# Detailed correlation analysis
println("\n=== Detailed Correlation Analysis ===")
println("\nCross-correlations (should show specialization):")
println("  Layer 1: Corr(h₁, g₁) vs Corr(h₂, g₁)")
corr_l1_11 = cor(h_x_all[1, :], g_cv_1[1, :])
corr_l1_21 = cor(h_x_all[2, :], g_cv_1[1, :])
println("    Corr(h₁, g₁^L1) = ", corr_l1_11)
println("    Corr(h₂, g₁^L1) = ", corr_l1_21)
if abs(corr_l1_11) > abs(corr_l1_21)
    println("    ✓ Layer 1 CV₁ specializes on component 1")
end

println("\n  Layer 2: Corr(h₁, g₁) vs Corr(h₂, g₁)")
corr_l2_11 = cor(h_x_all[1, :], g_cv_2[1, :])
corr_l2_21 = cor(h_x_all[2, :], g_cv_2[1, :])
println("    Corr(h₁, g₁^L2) = ", corr_l2_11)
println("    Corr(h₂, g₁^L2) = ", corr_l2_21)
if abs(corr_l2_11) > abs(corr_l2_21)
    println("    ✓ Layer 2 CV₁ specializes on component 1")
end
