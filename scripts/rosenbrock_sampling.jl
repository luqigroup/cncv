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
args = read_config("rosenbrock_sampling.json")
args = parse_input_args(args)

if args["epoch"] == -1
    args["epoch"] = args["max_epoch"]
end

save_path = plotsdir(args["sim_name"], savename(args))

# Define network
G = NetworkGlow(2, args["n_hidden"], args["depth"], args["K"], freeze_conv = true)

# Loading the experiment—only network weights and training loss
loaded_keys = load_experiment(args, ["G", "fval", "fval_eval"])
G = loaded_keys["G"]
fval = loaded_keys["fval"]
fval_eval = loaded_keys["fval_eval"]

# Testing data
test_size = 2^13
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_test_2d = rand(RB_dist, test_size)  # 2×test_size
X_test = reshape(X_test_2d, 1, 1, 2, test_size)  # 1×1×2×test_size

# Predicted samples
Zx = randn(Float32, 1, 1, 2, test_size)
X_ = G.inverse(Zx)

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

# True samples from Rosenbrock distribution.
fig, ax = subplots(1, 1, figsize = (5, 5))
ax.scatter(X_test[1, 1, 1, :], X_test[1, 1, 2, :], s = 0.5, color = "#819FB3", alpha = 0.5)
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-2.5, 8])
ax.grid(false)
ax.set_title("True samples")
wsave(joinpath(save_path, "true-samples.png"), fig)
close(fig)

fig, ax = subplots(1, 1, figsize = (5, 5))
ax.scatter(X_[1, 1, 1, :], X_[1, 1, 2, :], s = 0.5, color = "#D68D96", alpha = 0.5)

ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-2.5, 8])
ax.grid(false)
ax.set_title("Predicted samples")
wsave(joinpath(save_path, "nf-samples.png"), fig)
close(fig)

# Likelihood comparison plot

# Compute log-likelihood using the network (exact_likelihood function)
loglike_G = exact_likelihood(G, X_test)

# Compute true log-likelihood using Rosenbrock distribution
# Reshape X_test from 1×1×2×test_size to 2×test_size for Rosenbrock
loglike_true = Rosenbrock.logpdf(RB_dist, X_test_2d)

# Compute KL divergence + constant
kl_divergence = mean(loglike_true - loglike_G)
println("KL divergence + const.: ", kl_divergence)

# Create histogram comparison plot
fig = figure("histogram", figsize=(7, 2.5))
ax = histplot(
    loglike_true,
    kde=true,
    bins=50,
    label = "true log-likelihood",
    alpha= 0.8,
    color="#ff8800"
)
histplot(
    loglike_G,
    kde=true,
    bins=50,
    label = "predicted log-likelihood",
    alpha= 0.6,
    color="#00b4ba")
for label in ax.get_xticklabels()
    label.set_fontproperties(font_prop)
end
for label in ax.get_yticklabels()
    label.set_fontproperties(font_prop)
end
ax.set_xlabel("Log-likelihood", fontproperties=font_prop)
ax.legend(prop=font_prop)
wsave(joinpath(save_path, "log-like-hist.png"), fig)
close(fig)

upload_to_dropbox(args["sim_name"])