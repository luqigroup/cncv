# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

module CNCV

using DrWatson
using Flux
using Flux: Dense, sigmoid
using JLD2
using JSON
using HDF5
using ArgParse
using Random
using DataFrames
using LinearAlgebra
using Distributions
using Statistics
using ProgressMeter
using PyPlot
using Seaborn
using InvertibleNetworks:
    ActivationFunction,
    InvertibleNetwork,
    NeuralNetLayer,
    Parameter,
    Squeezer,
    FluxBlock,
    ResidualBlock,
    ReLUlayer,
    SigmoidLayer,
    ShuffleLayer,
    tensor_cat,
    tensor_split,
    squeeze,
    unsqueeze,
    cuzeros,
    glorot_uniform,
    chain_lr,
    array_of_array,
    cat_states,
    split_states,
    clear_grad!,
    get_params,
    reverse
using CUDA

import Base.*
import Base.-
import Base.adjoint
import DrWatson: _wsave
import Random: rand
import Base.getindex
import Distributions: logpdf, gradlogpdf
import InvertibleNetworks:
    forward,
    inverse,
    backward,
    backward_inv,
    jacobian,
    adjointJacobian,
    jacobianInverse,
    adjointJacobianInverse,
    tag_as_reversed!

# Utils
include("./models/utils/activation_functions.jl")

# Models
# Layers
include("./models/layers/invertible_layer_actnorm.jl")
include("./models/layers/invertible_layer_actnorm_cv.jl")
include("./models/layers/invertible_layer_conv1x1.jl")
include("./models/layers/invertible_layer_conv1x1_cv.jl")
include("./models/layers/invertible_layer_basic.jl")
include("./models/layers/invertible_layer_glow.jl")
include("./models/conditional_layers/conditional_layer_glow.jl")
include("./models/conditional_layers/conditional_layer_basic_cv.jl")
include("./models/conditional_layers/dense_conditional_layer_cv.jl")
include("./models/conditional_layers/dense_conditional_layer_cv_ensemble.jl")


# Invertible network architectures
include("./models/networks/invertible_network_glow.jl")  # Glow: Dinh et al. (2017), Kingma and Dhariwal (2018)
include("./models/networks/invertible_network_conditional_glow.jl")

# Utilities.
include("./utils/load_experiment.jl")
include("./utils/upload_to_dropbox.jl")
include("./utils/data_loader.jl")
include("./utils/savefig.jl")
include("./utils/logpdf.jl")
include("./utils/config.jl")
include("./utils/cs_op.jl")

# Objective functions.
include("./objectives/objectives.jl")
include("./objectives/exact_likelihood.jl")

# Sampling
include("./sampling/pSGLD.jl")
include("./sampling/sample.jl")


end

