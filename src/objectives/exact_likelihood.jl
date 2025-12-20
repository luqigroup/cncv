# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export exact_likelihood, exact_score


"""
Compute p(X) given X and Net
"""
function exact_likelihood(Net::NetworkGlow, X::AbstractArray{Float32,4})

    Zx, logdet = Net.forward(X; logdet_per_batch=true)
    loglike = sum(Distributions.logpdf(Normal(0f0, 1f0), Zx), dims = [1, 2, 3])[1, 1, 1, :]
    loglike = loglike + logdet

    return loglike

end


"""
Compute p(x|y) given:
"""
function exact_likelihood(
    Net::NetworkConditionalGlow,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4},
)

    Zx, Zy, logdet = Net.forward(X, Y; logdet_per_batch=true)
    loglike = sum(Distributions.logpdf(Normal(0f0, 1f0), Zx), dims = [1, 2, 3])[1, 1, 1, :]
    loglike = loglike + logdet

end


"""
Compute ∇_X log p(X) given X and Net (score function for unconditional network)
"""
function exact_score(Net::NetworkGlow, X::AbstractArray{Float32,4})

    # Forward pass through the network
    Zx, logdet = Net.forward(X)

    # Gradient of log p(x) w.r.t. Zx
    # log p(x) = sum(log N(Zx; 0, 1)) + logdet
    # ∂log p/∂Zx = ∂log N(Zx; 0, 1)/∂Zx
    ΔZx = gradlogpdf(0.0f0, 1.0f0, Zx)

    # Backward pass to get gradient w.r.t. X
    ΔX = Net.backward(ΔZx, Zx)[1]

    return ΔX

end


"""
Compute ∇_X log p(X|Y) given X, Y and Net (score function for conditional network)
"""
function exact_score(
    Net::NetworkConditionalGlow,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4},
)

    # Forward pass through the network
    Zx, Zy, logdet = Net.forward(X, Y)

    # Gradient of log p(x|y) w.r.t. Zx
    # log p(x|y) = sum(log N(Zx; 0, 1)) + logdet
    # ∂log p/∂Zx = ∂log N(Zx; 0, 1)/∂Zx
    ΔZx = gradlogpdf(0.0f0, 1.0f0, Zx)

    # Backward pass to get gradient w.r.t. X
    ΔX = Net.backward(ΔZx, Zx, Zy)[1]

    return ΔX

end


# function kl_divergance(
#     Net::NetworkConditionalGlowCV,
#     nlog_density::Function,
#     X_sgld::AbstractArray{Float32,4},
#     Y_obs::AbstractArray{Float32,4},
# )

#     loglike_G = exact_likelihood(G, X_sgld, repeat(Y_obs[:, :, :, 1:1], 1, 1, 1, size(X_sgld)[4] + 1))


#     loglike_true = sum(
#         logpdf(1f0, 2.5f0, X_sgld),
#         dims=[1, 2, 3]
#     )[1, 1, 1, :]

#     return sum(loglike_true - loglike_G) / size(X_sgld)[4]

# end