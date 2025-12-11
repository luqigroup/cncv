# Invertible network based on Glow (Kingma and Dhariwal, 2018) - CV version (no logdet)
# Includes 1x1 convolution and residual block
# Adapted for compressed sensing applications
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkConditionalGlowCV, NetworkConditionalGlowCV3D

"""
    G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

    G = NetworkConditionalGlowCV3D(n_in, n_cond, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

 Create a conditional invertible network based on the Glow architecture (CV version). Each flow step in the inner loop
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior
 to the inner loop, and a splitting operation afterwards.

 *Input*:

 - 'n_in': number of input channels of variable to sample

 - 'n_cond': number of input channels of condition

 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `split_scales`: if true, perform squeeze operation which halves spatial dimensions and duplicates channel dimensions
    then split output in half along channel dimension after each scale. Feed one half through the next layers,
    while saving the remaining channels for the output.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
 operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 - `squeeze_type` : squeeze type that happens at each multiscale level

 *Output*:

 - `G`: invertible Glow network.

 *Usage:*

 - Forward mode: `ZX, ZC = G.forward(X, C)`

 - Backward mode: `ΔX, X, ΔC = G.backward(ΔZX, ZX, ZC)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in activation normalizations `G.AN[i,j]` and coupling layers `G.CL[i,j]`,
   where `i` and `j` range from `1` to `L` and `K` respectively.

 See also: [`ActNormCV`](@ref), [`CouplingLayerGlowCV`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkConditionalGlowCV <: InvertibleNetwork
    AN::AbstractArray{ActNormCV, 2}
    AN_C::ActNormCV
    CL::AbstractArray{ConditionalLayerGlowCV, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkConditionalGlowCV

# Constructor
function NetworkConditionalGlowCV(n_in, n_cond, n_hidden, L, K; freeze_conv=false, split_scales=false, rb_activation::ActivationFunction=ReLUlayer(), k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    AN = Array{ActNormCV}(undef, L, K)    # activation normalization
    AN_C = ActNormCV(n_cond)    # activation normalization for condition
    CL = Array{ConditionalLayerGlowCV}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block

    if split_scales
        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 2^(ndims)
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        n_cond *= channel_factor # squeeze if split_scales is turned on
        for j=1:K
            AN[i, j] = ActNormCV(n_in)
            CL[i, j] = ConditionalLayerGlowCV(n_in, n_cond, n_hidden; freeze_conv=freeze_conv, rb_activation=rb_activation, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, activation=activation, ndims=ndims)
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2)) # split
    end

    return NetworkConditionalGlowCV(AN, AN_C, CL, Z_dims, L, K, squeezer, split_scales)
end

NetworkConditionalGlowCV3D(args; kw...) = NetworkConditionalGlowCV(args...; kw..., ndims=3)

# Forward pass
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalGlowCV) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, G.L-1))
    orig_shape = size(X)

    C = G.AN_C.forward(C)

    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        (G.split_scales) && (C = G.squeezer.forward(C))
        for j=1:G.K
            X = G.AN[i, j].forward(X)
            X = G.CL[i, j].forward(X, C)
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = reshape(cat_states(Z_save, X),orig_shape))
    return X, C
end

# Inverse pass
function inverse(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalGlowCV) where {T, N}
    G.split_scales && ((Z_save, X) = split_states(X[:], G.Z_dims))
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X = G.CL[i, j].inverse(X, C)
            X = G.AN[i, j].inverse(X)
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
        (G.split_scales) && (C = G.squeezer.inverse(C))
    end
    return X
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkConditionalGlowCV) where {T, N}
    # Split data and gradients
    if G.split_scales
        ΔZ_save, ΔX = split_states(ΔX[:], G.Z_dims)
        Z_save, X = split_states(X[:], G.Z_dims)
    end

    ΔC = T(0) .* C
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X  = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            ΔX, X, ΔC_ = G.CL[i, j].backward(ΔX, X, C)
            ΔX, X = G.AN[i, j].backward(ΔX, X)
            ΔC += ΔC_
        end

        if G.split_scales
            C = G.squeezer.inverse(C)
            ΔC = G.squeezer.inverse(ΔC)
            X = G.squeezer.inverse(X)
            ΔX = G.squeezer.inverse(ΔX)
        end
    end

    ΔC, C = G.AN_C.backward(ΔC, C)
    return ΔX, X, ΔC
end