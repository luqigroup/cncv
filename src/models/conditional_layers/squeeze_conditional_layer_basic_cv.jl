# Squeeze + Conditional coupling layer (basic version) - CV version (with jacobian trace)
# Combines squeezing/unsqueezing with conditional coupling layer
# Date: January 2025

export SqueezeConditionalLayerBasicCV, SqueezeConditionalLayerBasicCV3D


"""
    SCL = SqueezeConditionalLayerBasicCV(CL::ConditionalLayerBasicCV; pattern="column")

or

    SCL = SqueezeConditionalLayerBasicCV(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, pattern="column", ndims=2) (2D)

    SCL = SqueezeConditionalLayerBasicCV(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, pattern="column", ndims=3) (3D)

    SCL = SqueezeConditionalLayerBasicCV3D(n_in, n_cond, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, pattern="column") (3D)

 Create an invertible layer that combines squeezing with a conditional coupling layer (CV version).

 The layer performs: X → squeeze → ConditionalLayerBasicCV → Y

 *Input*:

 - `CL::ConditionalLayerBasicCV`: Conditional coupling layer

 or

 - `n_in`: number of input channels (BEFORE squeezing, will be multiplied by 4 in 2D or 8 in 3D after squeezing)

 - `n_cond`: number of conditioning channels (BEFORE squeezing)

 - `n_hidden`: number of hidden channels in residual block

 - `k1`, `k2`: kernel size of convolutions in residual block

 - `p1`, `p2`: padding for convolutions

 - `s1`, `s2`: stride for convolutions

 - `pattern`: Squeezing pattern ("column", "patch", or "checkerboard"). Default: "column"

 - `ndims` : number of dimensions (2 or 3)

 *Output*:

 - `SCL`: Invertible squeeze + conditional coupling layer.

 *Usage:*

 - Forward mode: `Y, jac_trace = SCL.forward(X, C)`

 - Inverse mode: `X, jac_trace = SCL.inverse(Y, C)`

 - Backward mode: `ΔX, X, ΔC = SCL.backward(ΔY, Y, C; jac_trace_grad_weight=nothing)`

 *Trainable parameters:*

 - None in `SCL` itself

 - Trainable parameters in the conditional coupling layer `SCL.CL`

 *Note on Jacobian trace:*

 For pattern="column", the squeezing operation is just a reshape (identity permutation in vectorized form),
 so trace(J_total) = trace(J_coupling). The trace computation remains tractable.

 See also: [`ConditionalLayerBasicCV`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
struct SqueezeConditionalLayerBasicCV <: NeuralNetLayer
    CL::ConditionalLayerBasicCV
    pattern::String
end

@Flux.functor SqueezeConditionalLayerBasicCV

# Constructor from ConditionalLayerBasicCV
function SqueezeConditionalLayerBasicCV(CL::ConditionalLayerBasicCV; pattern::String="column")
    return SqueezeConditionalLayerBasicCV(CL, pattern)
end

# Constructor from input dimensions
function SqueezeConditionalLayerBasicCV(n_in::Int64, n_cond::Int64, n_hidden::Int64;
                                        k1=3, k2=1, p1=1, p2=0, s1=1, s2=1,
                                        pattern::String="column",
                                        activation::ActivationFunction=SigmoidLayer(),
                                        rb_activation::ActivationFunction=ReLUlayer(),
                                        ndims=2)

    # After squeezing: channels multiply by 2^(ndims) (4 for 2D, 8 for 3D)
    n_in_squeezed = n_in * 2^ndims
    n_cond_squeezed = n_cond * 2^ndims

    # Create conditional coupling layer with squeezed dimensions
    CL = ConditionalLayerBasicCV(n_in_squeezed, n_cond_squeezed, n_hidden;
                                 k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2,
                                 activation=activation, rb_activation=rb_activation, ndims=ndims)

    return SqueezeConditionalLayerBasicCV(CL, pattern)
end

SqueezeConditionalLayerBasicCV3D(args...;kw...) = SqueezeConditionalLayerBasicCV(args...; kw..., ndims=3)

# Forward pass: Input X and C (conditioning), Output Y
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, L::SqueezeConditionalLayerBasicCV) where {T,N}

    # Squeeze inputs
    X_squeezed = squeeze(X; pattern=L.pattern)
    C_squeezed = squeeze(C; pattern=L.pattern)

    # Apply conditional coupling layer
    Y_squeezed, jac_trace = L.CL.forward(X_squeezed, C_squeezed)

    # Unsqueeze output
    Y = unsqueeze(Y_squeezed; pattern=L.pattern)

    # For pattern="column", the squeeze/unsqueeze are just reshapes,
    # so the Jacobian trace is unchanged
    return Y, jac_trace
end

# Inverse pass: Input Y and C (conditioning), Output X
function inverse(Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::SqueezeConditionalLayerBasicCV; save=false) where {T,N}

    # Squeeze inputs
    Y_squeezed = squeeze(Y; pattern=L.pattern)
    C_squeezed = squeeze(C; pattern=L.pattern)

    # Apply inverse of conditional coupling layer
    if save
        X_squeezed, jac_trace, X1, X2, logS, S = L.CL.inverse(Y_squeezed, C_squeezed; save=true)
        X = unsqueeze(X_squeezed; pattern=L.pattern)
        return X, jac_trace, X1, X2, logS, S
    else
        X_squeezed, jac_trace = L.CL.inverse(Y_squeezed, C_squeezed)
        X = unsqueeze(X_squeezed; pattern=L.pattern)
        return X, jac_trace
    end
end

# Backward pass: Input (ΔY, Y, C), Output (ΔX, X, ΔC)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::AbstractArray{T, N}, L::SqueezeConditionalLayerBasicCV;
                  jac_trace_grad_weight::Union{Nothing, AbstractVector{T}}=nothing) where {T,N}

    # Squeeze inputs
    ΔY_squeezed = squeeze(ΔY; pattern=L.pattern)
    Y_squeezed = squeeze(Y; pattern=L.pattern)
    C_squeezed = squeeze(C; pattern=L.pattern)

    # Backpropagate through conditional coupling layer
    ΔX_squeezed, X_squeezed, ΔC_squeezed = L.CL.backward(ΔY_squeezed, Y_squeezed, C_squeezed;
                                                         jac_trace_grad_weight=jac_trace_grad_weight)

    # Unsqueeze outputs
    ΔX = unsqueeze(ΔX_squeezed; pattern=L.pattern)
    X = unsqueeze(X_squeezed; pattern=L.pattern)
    ΔC = unsqueeze(ΔC_squeezed; pattern=L.pattern)

    return ΔX, X, ΔC
end
