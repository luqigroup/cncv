# Invertible network based on Glow (Kingma and Dhariwal, 2018) - CV version (with jacobian trace and gradients)
# Includes 1x1 convolution and residual block
# Adapted for compressed sensing applications
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkGlowCV, NetworkGlowCV3D

"""
    G = NetworkGlowCV(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

    G = NetworkGlowCV3D(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

 Create an invertible network based on the Glow architecture (CV version). Each flow step in the inner loop
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior
 to the inner loop, and a splitting operation afterwards.

 *Input*:

 - 'n_in': number of input channels

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

 - Forward mode: `Y, jac_trace = G.forward(X)`

 - Backward mode: `ΔX, X = G.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in activation normalizations `G.AN[i,j]` and coupling layers `G.CL[i,j]`,
   where `i` and `j` range from `1` to `L` and `K` respectively.

 See also: [`ActNormCV`](@ref), [`CouplingLayerGlowCV`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkGlowCV <: InvertibleNetwork
    AN::AbstractArray{ActNormCV, 2}
    CL::AbstractArray{CouplingLayerGlowCV, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkGlowCV

# Constructor
function NetworkGlowCV(n_in, n_hidden, L, K; nx=nothing, dense=false, freeze_conv=false, split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    (n_in == 1) && (split_scales = true) # Need extra channels for coupling layer
    (dense && isnothing(nx)) && error("Dense network needs nx as kwarg input")

    AN = Array{ActNormCV}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlowCV}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block

    if split_scales
        Z_dims = fill!(Array{Array}(undef, max(L-1,1)), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 2^(ndims)
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        (dense && split_scales) && (nx = Int64(nx/2))
        for j=1:K
            AN[i, j] = ActNormCV(n_in)
            CL[i, j] = CouplingLayerGlowCV(n_in, n_hidden; nx=nx, dense=dense, freeze_conv=freeze_conv, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, activation=activation, ndims=ndims)
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2); ) # split
    end

    return NetworkGlowCV(AN, CL, Z_dims, L, K, squeezer, split_scales)
end

NetworkGlowCV3D(args; kw...) = NetworkGlowCV(args...; kw..., ndims=3)

# Forward pass and compute jacobian trace
function forward(X::AbstractArray{T, N}, G::NetworkGlowCV) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, max(G.L-1,1)))

    # Initialize jacobian trace accumulator
    batchsize = size(X)[N]
    jac_trace_total = zeros(T, batchsize)

    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        for j=1:G.K
            X, jac_trace_an = G.AN[i, j].forward(X)
            X, jac_trace_cl = G.CL[i, j].forward(X)
            jac_trace_total .+= jac_trace_an .+ jac_trace_cl
        end
        if G.split_scales && (i < G.L || i == 1)    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = cat_states(Z_save, X))

    return X, jac_trace_total
end

# Inverse pass
function inverse(Z::AbstractArray{T, N}, G::NetworkGlowCV) where {T, N}
    X = Z
    G.split_scales && ((Z_save, X) = split_states(X, G.Z_dims;L_net=G.L))

    # Initialize jacobian trace accumulator
    batchsize = size(X)[N]
    jac_trace_total = zeros(T, batchsize)

    for i=G.L:-1:1
        if G.split_scales && (i < G.L || G.L == 1)
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X, jac_trace_cl = G.CL[i, j].inverse(X)
            X, jac_trace_an = G.AN[i, j].inverse(X)
            jac_trace_total .+= jac_trace_an .+ jac_trace_cl
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
    end
    return X, jac_trace_total
end

# Backward pass and compute gradients
function backward(ΔZ::AbstractArray{T, N}, Z::AbstractArray{T, N}, G::NetworkGlowCV; set_grad::Bool=true) where {T, N}
    ΔX = ΔZ
    X = Z
    # Split data and gradients
    if G.split_scales
        ΔX_save, ΔX = split_states(ΔX, G.Z_dims;L_net=G.L)
        X_save, X = split_states(X, G.Z_dims;L_net=G.L)
    end

    if ~set_grad
        ΔθAN = Vector{Parameter}(undef, 0)
        ΔθCL = Vector{Parameter}(undef, 0)
    end

    for i=G.L:-1:1
        if G.split_scales && (i < G.L || G.L == 1)
            X  = tensor_cat(X, X_save[i])
            ΔX = tensor_cat(ΔX, ΔX_save[i])
        end
        for j=G.K:-1:1
            if set_grad
                ΔX, X = G.CL[i, j].backward(ΔX, X)
                ΔX, X = G.AN[i, j].backward(ΔX, X)
            else
                ΔX, Δθcl_ij, X = G.CL[i, j].backward(ΔX, X; set_grad=set_grad)
                ΔX, Δθan_ij, X = G.AN[i, j].backward(ΔX, X; set_grad=set_grad)
                prepend!(ΔθAN, Δθan_ij)
                prepend!(ΔθCL, Δθcl_ij)
            end
        end

        if G.split_scales
            X = G.squeezer.inverse(X)
            ΔX = G.squeezer.inverse(ΔX)
        end
    end
    set_grad ? (return ΔX, X) : (return ΔX, vcat(ΔθAN, ΔθCL), X)
end

# Compute gradient of jacobian trace w.r.t. all network parameters
function jac_trace_grad(X::AbstractArray{T, N}, G::NetworkGlowCV) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, max(G.L-1,1)))

    ∇θAN_trace = Vector{Any}(undef, 0)
    ∇θCL_trace = Vector{Any}(undef, 0)

    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        for j=1:G.K
            # Get gradient of trace w.r.t. ActNorm parameters
            ∇s_an = jac_trace_grad!(G.AN[i, j], X)
            push!(∇θAN_trace, ∇s_an)

            X, _ = G.AN[i, j].forward(X)

            # Get gradient of trace w.r.t. CouplingLayer parameters
            ∇θ_cl = jac_trace_grad(X, G.CL[i, j])
            push!(∇θCL_trace, ∇θ_cl)

            X, _ = G.CL[i, j].forward(X)
        end
        if G.split_scales && (i < G.L || i == 1)
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end

    return ∇θAN_trace, ∇θCL_trace
end


## Jacobian-related utils
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Vector{Parameter}, X, G::NetworkGlowCV) where {T, N}

    if G.split_scales
        Z_save = array_of_array(ΔX, G.L-1)
        ΔZ_save = array_of_array(ΔX, G.L-1)
    end

    cls = 2*G.K*G.L
    ΔθAN = Vector{Parameter}(undef, 0)
    ΔθCL = Vector{Parameter}(undef, 0)

    for i=1:G.L
        if G.split_scales
            X = G.squeezer.forward(X)
            ΔX = G.squeezer.forward(ΔX)
        end

        for j=1:G.K
            as = length(ΔθAN)+1
            cs = cls + length(ΔθCL) + 1
            ΔX, X = G.AN[i, j].jacobian(ΔX, Δθ[as:as+1], X)
            ΔX, X = G.CL[i, j].jacobian(ΔX, Δθ[cs:cs+7], X)
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZ = tensor_split(ΔX)
            Z_save[i] = Z
            ΔZ_save[i] = ΔZ
            G.Z_dims[i] = collect(size(Z))
        end
    end
    if G.split_scales
        X = cat_states(Z_save, X)
        ΔX = cat_states(ΔZ_save, ΔX)
    end

    return ΔX, X
end

adjointJacobian(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, G::NetworkGlowCV) where {T, N} = backward(ΔX, X, G; set_grad=false)