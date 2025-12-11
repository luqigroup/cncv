# Affine coupling layer from Dinh et al. (2017) - CV version (no logdet)
# Includes 1x1 convolution from Putzky and Welling (2019)
# Adapted for compressed sensing applications
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerGlowCV, CouplingLayerGlowCV3D


"""
    CL = CouplingLayerGlowCV(C::Conv1x1CV, RB::ResidualBlock)

or

    CL = CouplingLayerGlowCV(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2) (2D)

    CL = CouplingLayerGlowCV(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=3) (3D)

    CL = CouplingLayerGlowCV3D(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1) (3D)

 Create a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block (CV version).

 *Input*:

 - `C::Conv1x1CV`: 1x1 convolution layer

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y = CL.forward(X)`

 - Inverse mode: `X = CL.inverse(Y)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1CV`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerGlowCV <: NeuralNetLayer
    C::Conv1x1CV
    RB::Union{ResidualBlock, FluxBlock}
    activation::ActivationFunction
end

@Flux.functor CouplingLayerGlowCV

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlowCV(C::Conv1x1CV, RB::ResidualBlock; activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlowCV(C, RB, activation)
end

# Constructor from 1x1 convolution and residual Flux block
CouplingLayerGlowCV(C::Conv1x1CV, RB::FluxBlock; activation::ActivationFunction=SigmoidLayer()) = CouplingLayerGlowCV(C, RB, activation)

# Constructor from input dimensions
function CouplingLayerGlowCV(n_in::Int64, n_hidden::Int64; nx=nothing, dense=false, freeze_conv=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1CV(n_in; freeze=freeze_conv)

    split_num = Int(round(n_in/2))
    in_chan   = n_in-split_num
    out_chan  = 2*split_num

    if dense
        isnothing(nx) && error("Dense network needs nx as kwarg input")
        RB = FluxBlock(Chain(x->reshape(x,nx*in_chan,:),Dense(nx*in_chan,n_in*n_hidden,relu),Dense(n_in*n_hidden,n_in*n_hidden,relu),Dense(n_in*n_hidden,nx*out_chan,relu),x->reshape(x,nx,out_chan,:)))
    else
        RB = ResidualBlock(in_chan, n_hidden;n_out=out_chan, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)
    end

    return CouplingLayerGlowCV(C, RB, activation)
end

CouplingLayerGlowCV3D(args...;kw...) = CouplingLayerGlowCV(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, L::CouplingLayerGlowCV) where {T,N}
    X_ = L.C.forward(X)
    X1, X2 = tensor_split(X_)

    Y2 = copy(X2)
    logS_T = L.RB.forward(X2)
    logSm, Tm = tensor_split(logS_T)
    Sm = L.activation.forward(logSm)
    Y1 = Sm.*X1 + Tm

    Y = tensor_cat(Y1, Y2)

    return Y
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, L::CouplingLayerGlowCV; save=false) where {T,N}
    Y1, Y2 = tensor_split(Y)

    X2 = copy(Y2)
    logS_T = L.RB.forward(X2)
    logSm, Tm = tensor_split(logS_T)
    Sm = L.activation.forward(logSm)
    X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0

    X_ = tensor_cat(X1, X2)
    X = L.C.inverse(X_)

    save == true ? (return X, X1, X2, logSm, Sm) : (return X)
end

# Backward pass: Input (ΔY, Y), Output (ΔX, X)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlowCV; set_grad::Bool=true) where {T,N}

    # Recompute forward state
    X, X1, X2, logS, S = inverse(Y, L; save=true)

    # Backpropagate residual
    ΔY1, ΔY2 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1

    ΔX1 = ΔY1 .* S
    if set_grad
        ΔX2 = L.RB.backward(tensor_cat(backward(ΔS, logS, S, L.activation), ΔT), X2) + ΔY2
    else
        ΔX2, Δθrb = L.RB.backward(tensor_cat(backward(ΔS, logS, S, L.activation), ΔT), X2; set_grad=set_grad)
        ΔX2 += ΔY2
    end
    ΔX_ = tensor_cat(ΔX1, ΔX2)
    if set_grad
        ΔX = L.C.inverse((ΔX_, tensor_cat(X1, X2)))[1]
    else
        ΔX, Δθc = L.C.inverse((ΔX_, tensor_cat(X1, X2)); set_grad=set_grad)[1:2]
        Δθ = cat(Δθc, Δθrb; dims=1)
    end

    if set_grad
        return ΔX, X
    else
        return ΔX, Δθ, X
    end
end

# Jacobian-related functions
function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X, L::CouplingLayerGlowCV) where {T,N}

    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1, X2 = tensor_split(X_)
    ΔX1, ΔX2 = tensor_split(ΔX_)

    Y2 = copy(X2)
    ΔY2 = copy(ΔX2)
    ΔlogS_T, logS_T = L.RB.jacobian(ΔX2, Δθ[4:end], X2)
    ΔlogS, ΔlogT = tensor_split(ΔlogS_T)
    logS, logT = tensor_split(logS_T)
    Sm = L.activation.forward(logS)
    ΔS = backward(ΔlogS, logS, Sm, L.activation)
    Tm = logT
    ΔT = ΔlogT
    Y1 = Sm.*X1 + Tm
    ΔY1 = ΔS.*X1 + Sm.*ΔX1 + ΔT
    Y = tensor_cat(Y1, Y2)
    ΔY = tensor_cat(ΔY1, ΔY2)

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlowCV) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end