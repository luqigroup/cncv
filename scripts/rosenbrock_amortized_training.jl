# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

using DrWatson
@quickactivate :CNCV

using InvertibleNetworks
using Rosenbrock
using Random
using ProgressMeter
using Flux

# Random seed
Random.seed!(19)

args = read_config("rosenbrock_amortized_training.json")
args = parse_input_args(args)

device = cpu # InvertibleNetworks.CUDA.functional() ? gpu : cpu

# Define network.
G = NetworkConditionalGlowCV(
    2,
    2,
    args["n_hidden"],
    args["depth"],
    args["K"];
    freeze_conv = false,
)
G = G |> device

# Training data number.
ntrain = 5120

# Generate training data: rand returns 2×ntrain, reshape to 1×1×2×ntrain
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_train_2d = rand(RB_dist, ntrain)  # 2×ntrain
X_train = reshape(X_train_2d, 1, 1, 2, ntrain)
Y_train = X_train + args["sigma"] * randn(Float32, 1, 1, 2, ntrain)
X_train = X_train |> device
Y_train = Y_train |> device

# Validation data number.
nval = 512
X_val_2d = rand(RB_dist, nval)  # 2×nval
X_val = reshape(X_val_2d, 1, 1, 2, nval)
Y_val = X_val + args["sigma"] * randn(Float32, 1, 1, 2, nval)
X_val = X_val |> device
Y_val = Y_val |> device

p = Progress(Int(floor(ntrain / args["batchsize"])) * args["max_epoch"])

# Training Batch extractor.
train_loader =
    Flux.DataLoader((X_train, Y_train), batchsize = args["batchsize"], shuffle = true)
num_batches = length(train_loader)

# Optimizer.
opt = Flux.Optimiser(
    Flux.ExpDecay(args["lr"], 0.9f0, num_batches * args["lr_step"], 1.0f-6),
    Flux.Adam(args["lr"]),
)

# Training log keeper.
fval = zeros(Float32, num_batches * args["max_epoch"])
fval_eval = zeros(Float32, args["max_epoch"])

for epoch = 1:args["max_epoch"]

    fval_eval[epoch] = negative_log_likelihood(G, X_val, Y_val; grad = false)

    for (itr, (X, Y)) in enumerate(train_loader)
        Base.flush(Base.stdout)

        fval[(epoch-1)*num_batches+itr] = negative_log_likelihood(G, X, Y)[1]

        ProgressMeter.next!(
            p;
            showvalues = [
                (:Epoch, epoch),
                (:Iteration, itr),
                (:NLL, fval[(epoch-1)*num_batches+itr]),
                (:NLL_eval, fval_eval[epoch]),
            ],
        )

        # Update params
        for p in get_params(G)
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(G)
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
                "G" => G |> cpu,
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