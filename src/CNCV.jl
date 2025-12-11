# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

module CNCV

using DrWatson
using Flux
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
    # Core types
    ActivationFunction,
    InvertibleNetwork,
    NeuralNetLayer,
    Parameter,
    Squeezer,
    # Layer types
    FluxBlock,
    ResidualBlock,
    # Activation functions
    ReLUlayer,
    SigmoidLayer,
    ShuffleLayer,
    # Tensor operations
    tensor_cat,
    tensor_split,
    # Initialization utilities
    cuzeros,
    glorot_uniform,
    # Conv1x1 utilities
    chain_lr,
    # Multi-scale network utilities
    array_of_array,
    cat_states,
    split_states,
    # Parameter management
    clear_grad!,
    get_params
using CUDA

import Base.*
import Base.-
import Base.adjoint
import DrWatson: _wsave
import Random: rand
import Base.getindex
import Distributions: logpdf, gradlogpdf
import InvertibleNetworks: forward, inverse, backward

# Utils
include("./models/utils/activation_functions.jl")

# Models
# Layers
# include("./models/layers/layer_affine.jl")
include("./models/layers/invertible_layer_actnorm.jl")
include("./models/layers/invertible_layer_conv1x1.jl")
include("./models/layers/invertible_layer_basic.jl")
include("./models/layers/invertible_layer_glow.jl")
include("./models/conditional_layers/conditional_layer_glow.jl")

# Invertible network architectures
include("./models/networks/invertible_network_glow.jl")  # Glow: Dinh et al. (2017), Kingma and Dhariwal (2018)

# Conditional layers and nets
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

