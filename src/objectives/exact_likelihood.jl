# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export exact_likelihood


"""
Compute p(X) given X and Net
"""
function exact_likelihood(Net::NetworkGlow, X::AbstractArray{Float32,4})

    Zx, logdet = Net.forward(X)

    loglike = sum(logpdf(0.0f0, 1.0f0, Zx), dims = [1, 2, 3])[1, 1, 1, :]
    loglike = loglike .+ logdet

    return loglike

end


"""
Compute p(x|y) given:
Zx, _, logdet = Net.forward(X, repeat(Y, 1, 1, 1, size(X, 4)); x_lane=true)
"""
function exact_likelihood(
    Net::NetworkConditionalGlow,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4},
)

    Zx, Zy, logdet = Net.forward(X, Y)

    loglike = sum(logpdf(0.0f0, 1.0f0, Zx), dims = [1, 2, 3])[1, 1, 1, :]
    loglike = loglike .+ logdet

    return loglike

end


# function kl_divergance(
#     Net::NetworkConditionalGlow,
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