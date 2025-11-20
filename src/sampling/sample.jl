# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025

export MCMC_sampler

function MCMC_sampler(
    max_itr::Int,
    x0::AbstractArray{Float32,4},
    nlog_density::Function;
    lr::Float32 = 5.0f-2,
    lr_final::Float32 = 1.0f-4,
    lr_step::Int = 1,
    thinning::Int = 10,
)
    θ = Flux.Params([x0])
    opt = pSGLD(lr)
    x_samples = zeros(Float32, size(x0, 1), size(x0, 2), size(x0, 3), max_itr)
    lr_fun = construct_custom_lr_decay(lr, lr_final, max_itr)

    p = Progress(max_itr)
    for j = 1:max_itr

        # Evaluate objective and gradients
        obj, back = Flux.pullback(θ) do
            nlog_density(x0)
        end
        grads = back(1.0f0)

        if j % lr_step == 0
            opt.eta = lr_fun(j)
        end
        Flux.update!(opt, x0, grads[x0])


        ProgressMeter.next!(
            p;
            showvalues = [("SGLD itreration", j), ("NLL", obj), ("SGLD stepsize", opt.eta)],
        )

        x_samples[:, :, :, j:j] = x0
    end

    # Warm-up phase
    x_samples = x_samples[:, :, :, fld(max_itr, 2):end]

    # Thinning
    x_samples = x_samples[:, :, :, 1:thinning:end]

    return x_samples
end


function construct_custom_lr_decay(
    initial_lr::Float32,
    final_lr::Float32,
    max_itr::Int;
    gamma::Float32 = -1.0f0 / 3.0f0,
)

    b = max_itr / ((final_lr / initial_lr)^(1.0f0 / gamma) - 1.0f0)
    a = initial_lr / (b^gamma)

    lr(t) = a * (b + t - 1)^gamma

    return lr

end
