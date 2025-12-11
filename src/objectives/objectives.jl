# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export negative_log_likelihood


function negative_log_likelihood(Net::NetworkGlow, X::AbstractArray{Float32,4}; grad::Bool = true)

    Zx, logdet = Net.forward(X)
    z_size = size(Zx)

    f = sum(logpdf(0.0f0, 1.0f0, Zx)) + logdet * z_size[4]

    if grad
        ΔZx = -gradlogpdf(0.0f0, 1.0f0, Zx) / z_size[4]
        ΔX = Net.backward(ΔZx, Zx)[1]

        return -f / z_size[4], ΔX
    else
        return -f / z_size[4]
    end
end



function negative_log_likelihood(
    Net::NetworkConditionalGlow,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4};
    grad::Bool = true,
)

    Zx, Zy, logdet = Net.forward(X, Y)
    if CUDA.functional()
        CUDA.reclaim()
    end
    z_size = size(Zx)

    f = sum(logpdf(0.0f0, 1.0f0, Zx))
    f = f + logdet * z_size[4]

    if grad
        ΔZx = -gradlogpdf(0.0f0, 1.0f0, Zx) / z_size[4]

        ΔX = Net.backward(ΔZx, Zx, Zy)[1]
        if CUDA.functional()
            CUDA.reclaim()
        end
        GC.gc()

        return -f / z_size[4], ΔX
    else
        return -f / z_size[4]
    end
end
