# Sampling and evaluation script for Rosenbrock ENSEMBLE CV
# Tests if forward-reverse ensemble fixes the variance problem for Rosenbrock
# Can use either LEARNED or TRUE analytical score function
# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: December 2025

using DrWatson
@quickactivate :CNCV

using Statistics
using Random
using LinearAlgebra
using Rosenbrock
using JLD2
using PyPlot
using Seaborn
using Distributions

# Import ensemble functions
import CNCV: forward

# Random seed
# Random.seed!(123)

font_prop = set_plot_configs()[1]
args = read_config("rosenbrock_ensemble_cv_sampling.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Conditionally load pretrained amortized model for learned likelihood
if args["learned_score"] != 0
    println("Loading pretrained amortized model for learned log-likelihood...")
    amortized_args = read_config("rosenbrock_amortized_sampling.json")
    if amortized_args["epoch"] == -1
        amortized_args["epoch"] = amortized_args["max_epoch"]
    end
    loaded_amortized = load_experiment(amortized_args, ["G"])
    G_amortized = loaded_amortized["G"]
    println("Loaded amortized model successfully!")
else
    println("Using TRUE analytical score function (Gaussian + Rosenbrock)")
    G_amortized = nothing
end

# Load trained ensemble model
loaded_keys = load_experiment(args, ["layer1", "layer2", "mu", "fval", "fval_eval"])
layer1 = loaded_keys["layer1"]
layer2 = loaded_keys["layer2"]
μ = loaded_keys["mu"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]

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

# Sample one true parameter from prior
X_true = rand(RB_dist, 1)[:, 1]
Y_obs = X_true + σ_obs * randn(Float32, 2)

println("True parameter X: ", X_true')
println("Observation Y: ", Y_obs')

# Generate POSTERIOR samples p(X|Y) conditioned on Y_obs
println("\n=== Generating Posterior Samples ===")

if args["learned_score"] != 0
    println("Using pretrained amortized model to generate posterior samples...")
    # Use amortized variational inference to generate posterior samples
    Y_obs_net = reshape(Y_obs, 1, 1, 2, 1)
    Y_test_net = repeat(Y_obs_net, 1, 1, 1, test_size)

    # Sample from standard Gaussian in latent space
    Zx = randn(Float32, 1, 1, 2, test_size)

    # Transform through the network conditioned on Y_obs
    Zy_fixed = G_amortized.forward(Zx, Y_test_net)[2]
    X_test_net = G_amortized.inverse(Zx, Zy_fixed)

    # Reshape to 2×test_size
    X_test = reshape(X_test_net, 2, test_size)
    println("Generated ", test_size, " posterior samples using amortized model")
else
    println("Using MCMC to generate posterior samples...")
    # Use MCMC (SGLD) to sample from posterior
    # Objective function for MCMC: -log p(x|y) = -log p(y|x) - log p(x) + const
    obj(x, y) = begin
        # Reshape x from 1×1×2×n to 2×n for Rosenbrock
        x_2d = reshape(x, 2, size(x, 4))

        data_term = (1.0f0 / (2.0f0 * σ_obs^2)) * sum((x - y) .^ 2.0f0)
        # logpdf returns a vector, sum it up and negate
        prior_term = -sum(logpdf(RB_dist, x_2d))

        return data_term + prior_term
    end

    Y_obs_net = reshape(Y_obs, 1, 1, 2, 1)
    f(x) = obj(x, Y_obs_net)

    # Run MCMC sampler
    max_itr = 20 * test_size
    X_sgld = MCMC_sampler(
        max_itr,
        randn(Float32, 1, 1, 2, 1),
        f;
        lr = 5.0f0,
        lr_final = 1.0f-1,
        thinning = 10,
    )

    # Reshape to 2×test_size (remove first sample and extra dimension)
    X_test = reshape(X_sgld[1, 1, :, 2:end], 2, test_size)
    println("Generated ", test_size, " posterior samples using MCMC")
end

# Create conditioning: all samples observe the same Y
Y_test = repeat(Y_obs, 1, test_size)

# Score function: switches between learned and true based on args["learned_score"]
function compute_score_posterior(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T
    if args["learned_score"] != 0
        # Use LEARNED score from pretrained amortized model
        batchsize = size(X, 2)
        X_net = reshape(X, 1, 1, 2, batchsize)
        Y_net = reshape(Y, 1, 1, 2, batchsize)

        # Compute score using exact_score function
        ΔX_net = exact_score(G_amortized, X_net, Y_net)

        # Reshape back to 2×batchsize
        grad_log_post = reshape(ΔX_net, 2, batchsize)
    else
        # Use TRUE analytical score: ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
        # Gradient of Gaussian likelihood: ∇ log p(y|x) = -(x-y)/σ²
        grad_likelihood = -(X .- Y) ./ (σ_obs^2)

        # Gradient of Rosenbrock prior
        grad_prior = gradlogpdf(RB_dist, X)

        # Total gradient
        grad_log_post = grad_likelihood .+ grad_prior
    end

    return grad_log_post
end

score_term = compute_score_posterior(X_test, Y_test)

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

# Layer 2
jac_traces_2, phi_all_2 = forward(X_test, Y_test, layer2)
g_cv_2 = zeros(Float32, 2, test_size)
for k in 1:2
    trace_k = jac_traces_2[k, :]
    phi_k = phi_all_2[k, :, :]
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
    g_cv_2[k, :] = trace_k .+ phi_dot_score_k
end

# Combined (average)
g_cv_combined = (g_cv_1 .+ g_cv_2) ./ 2

println("Ensemble Control Variates:")
for k in 1:2
    println("  CV $k: mean = ", mean(g_cv_combined[k, :]), ", std = ", std(g_cv_combined[k, :]))
    corr = cor(h_x_all[k, :], g_cv_combined[k, :])
    println("    Correlation with h_$k: ", corr)
end

println("\n=== Stein's Identity Check ===")
println("Ensemble CV should have E[g_k|Y] ≈ 0")
for k in 1:2
    ratio = abs(mean(g_cv_combined[k, :])) / std(g_cv_combined[k, :])
    println("  Component $k: E[g_$k] / Std[g_$k] = ", round(ratio, digits=4), " (should be << 1)")
end

# Variance analysis (relative variance comparison)
# Compute numerical ground truth using all samples (approximation of true posterior mean)
numerical_ground_truth = vec(mean(X_test, dims=2))

println("\n=== Variance Reduction Summary ===")
println("Numerical ground truth E[X|Y] ≈ ", numerical_ground_truth')

var_h_1 = var(h_x_all[1, :])
var_h_minus_g_comb_1 = var(h_x_all[1, :] .- g_cv_combined[1, :])
vrf_comb_1 = var_h_minus_g_comb_1 / var_h_1

var_h_2 = var(h_x_all[2, :])
var_h_minus_g_comb_2 = var(h_x_all[2, :] .- g_cv_combined[2, :])
vrf_comb_2 = var_h_minus_g_comb_2 / var_h_2

println("\nComponent 1: VRF = $(round(vrf_comb_1, digits=4))")
if vrf_comb_1 < 1.0
    reduction = (1 - vrf_comb_1) * 100
    println("  → Variance reduced by $(round(reduction, digits=1))%")
else
    increase = (vrf_comb_1 - 1) * 100
    println("  → WARNING: Variance increased by $(round(increase, digits=1))%")
end

println("\nComponent 2: VRF = $(round(vrf_comb_2, digits=4))")
if vrf_comb_2 < 1.0
    reduction = (1 - vrf_comb_2) * 100
    println("  → Variance reduced by $(round(reduction, digits=1))%")
else
    increase = (vrf_comb_2 - 1) * 100
    println("  → WARNING: Variance increased by $(round(increase, digits=1))%")
end

if vrf_comb_1 < 1.0 && vrf_comb_2 < 1.0
    avg_reduction = ((1 - vrf_comb_1) + (1 - vrf_comb_2)) / 2 * 100
    println("\n✓ Average variance reduction: $(round(avg_reduction, digits=1))%")
end

# ============= MSE ANALYSIS ACROSS SAMPLE SIZES =============

sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
vanilla_mse = zeros(Float32, length(sample_sizes), 2)
cv_combined_mse = zeros(Float32, length(sample_sizes), 2)

num_trials = 100

println("\n=== Computing MSE across sample sizes ===")
for (idx, n) in enumerate(sample_sizes)
    vanilla_estimates = zeros(Float32, num_trials, 2)
    cv_combined_estimates = zeros(Float32, num_trials, 2)

    for trial in 1:num_trials
        inds = randperm(test_size)[1:n]

        # Vanilla Monte Carlo
        for i in 1:2
            vanilla_estimates[trial, i] = mean(h_x_all[i, inds])
        end

        # Combined CV
        for i in 1:2
            cv_combined_estimates[trial, i] = mean(h_x_all[i, inds] .- g_cv_combined[i, inds])
        end
    end

    # Compute MSE relative to numerical ground truth
    for i in 1:2
        vanilla_mse[idx, i] = mean((vanilla_estimates[:, i] .- numerical_ground_truth[i]).^2)
        cv_combined_mse[idx, i] = mean((cv_combined_estimates[:, i] .- numerical_ground_truth[i]).^2)
    end
end

println("MSE analysis complete")

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

for i in 1:2
    subplot(1, 2, i)
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

# 3. Prior distribution samples
fig = figure("rosenbrock samples", figsize = (6, 6))
ax = fig.add_subplot(111)
ax.patch.set_facecolor("white")
scatter(X_test[1, :], X_test[2, :], s = 0.5, color = "#000000", alpha = 0.15, label = "Prior samples")
grid(false)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.legend(loc = "upper right")
ax.set_title("Rosenbrock Prior Distribution")
wsave(joinpath(save_path, "prior-samples.png"), fig)
close(fig)

# 4. Violin plots of estimator distributions
# Use a moderate sample size to show variance
n_violin = 200
num_violin_trials = 200

fig = figure("estimator distributions", figsize = (10, 5))

for comp in 1:2
    subplot(1, 2, comp)

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

    # Mark numerical ground truth
    axhline(y = numerical_ground_truth[comp], color = "k", linestyle = "--", linewidth = 2, label = "Numerical ground truth")

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

# 5. Variance Reduction Factor across sample sizes
sample_sizes_vrf = [50, 100, 200, 500, 1000, 2000]
num_vrf_trials = 200

fig = figure("vrf vs sample size", figsize = (10, 5))

for comp in 1:2
    subplot(1, 2, comp)

    # Compute variance for each sample size by running trials
    vanilla_var_vs_n = zeros(Float32, length(sample_sizes_vrf))
    layer1_var_vs_n = zeros(Float32, length(sample_sizes_vrf))
    layer2_var_vs_n = zeros(Float32, length(sample_sizes_vrf))
    combined_var_vs_n = zeros(Float32, length(sample_sizes_vrf))

    vrf_layer1_vs_n = zeros(Float32, length(sample_sizes_vrf))
    vrf_layer2_vs_n = zeros(Float32, length(sample_sizes_vrf))
    vrf_combined_vs_n = zeros(Float32, length(sample_sizes_vrf))

    for (idx, n) in enumerate(sample_sizes_vrf)
        vanilla_ests = zeros(Float32, num_vrf_trials)
        layer1_ests = zeros(Float32, num_vrf_trials)
        layer2_ests = zeros(Float32, num_vrf_trials)
        combined_ests = zeros(Float32, num_vrf_trials)

        for trial in 1:num_vrf_trials
            inds = randperm(test_size)[1:n]
            vanilla_ests[trial] = mean(h_x_all[comp, inds])
            layer1_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_1[comp, inds])
            layer2_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_2[comp, inds])
            combined_ests[trial] = mean(h_x_all[comp, inds] .- g_cv_combined[comp, inds])
        end

        vanilla_var_vs_n[idx] = var(vanilla_ests)
        layer1_var_vs_n[idx] = var(layer1_ests)
        layer2_var_vs_n[idx] = var(layer2_ests)
        combined_var_vs_n[idx] = var(combined_ests)

        vrf_layer1_vs_n[idx] = layer1_var_vs_n[idx] / vanilla_var_vs_n[idx]
        vrf_layer2_vs_n[idx] = layer2_var_vs_n[idx] / vanilla_var_vs_n[idx]
        vrf_combined_vs_n[idx] = combined_var_vs_n[idx] / vanilla_var_vs_n[idx]
    end

    plot(sample_sizes_vrf, vrf_layer1_vs_n, "s-", linewidth = 2, markersize = 6, label = "VRF (Layer 1)", alpha = 0.7)
    plot(sample_sizes_vrf, vrf_layer2_vs_n, "^-", linewidth = 2, markersize = 6, label = "VRF (Layer 2)", alpha = 0.7)
    plot(sample_sizes_vrf, vrf_combined_vs_n, "o-", linewidth = 2, markersize = 8, label = "VRF (Combined)", color = "#d62728")
    axhline(y = 1.0, color = "k", linestyle = "--", linewidth = 1, label = "No reduction")

    # Add single layer baseline as reference
    axhline(y = (comp == 1 ? 26.5 : 12.7), color = "#bcbd22", linestyle = ":", linewidth = 2, label = "Single layer baseline")

    xlabel("Sample size")
    ylabel("Variance Reduction Factor")
    title("Component $comp: VRF vs Sample Size")
    xscale("log")
    legend()
    grid(true, alpha = 0.3)
    ylim([0, max(2.0, maximum(vrf_combined_vs_n) * 1.2)])
end

tight_layout()
wsave(joinpath(save_path, "vrf-vs-sample-size.png"), fig)
close(fig)

# 6. Variance comparison across methods
fig = figure("variance comparison", figsize = (10, 5))

for comp in 1:2
    subplot(1, 2, comp)

    # Compute actual variances for a fixed sample size
    n_var = 200
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
    yscale("log")

    # Add VRF annotation
    vrf = vars[2] / vars[1]
    reduction_pct = (1 - vrf) * 100

    # Single layer baseline VRF
    single_vrf = comp == 1 ? 26.5 : 12.7
    improvement = (single_vrf - vrf) / single_vrf * 100

    text(0.5, 0.95, "VRF = $(round(vrf, digits=3))\nReduction = $(round(reduction_pct, digits=1))%\nImprovement over single = $(round(improvement, digits=1))%",
         transform=gca().transAxes, horizontalalignment="center", verticalalignment="top",
         bbox=Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5))
end

tight_layout()
wsave(joinpath(save_path, "variance-comparison.png"), fig)
close(fig)

# 7. Scatter plots showing h vs g_cv
fig = figure("scatter plots", figsize = (12, 8))

for i in 1:2
    # Layer 1
    subplot(2, 3, (i-1)*3 + 1)
    scatter(h_x_all[i, :], g_cv_1[i, :], s = 1, alpha = 0.3)
    xlabel("h_$i(x)")
    ylabel("g_$i^(L1)")
    title("Layer 1: Component $i")
    grid(true, alpha = 0.3)

    # Layer 2
    subplot(2, 3, (i-1)*3 + 2)
    scatter(h_x_all[i, :], g_cv_2[i, :], s = 1, alpha = 0.3)
    xlabel("h_$i(x)")
    ylabel("g_$i^(L2)")
    title("Layer 2: Component $i")
    grid(true, alpha = 0.3)

    # Combined
    subplot(2, 3, (i-1)*3 + 3)
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
