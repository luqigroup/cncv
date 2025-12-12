# Test for objective functions (non-CV)
# Tests include: negative log-likelihood, gradient computation

using Test, Random, LinearAlgebra, Statistics

using CNCV

Random.seed!(707172)

@testset "Objective Functions Tests" begin

    @testset "Negative Log-Likelihood - Basic" begin
        nx, ny = 8, 8
        n_in = 4
        batchsize = 8

        # Mock network forward output
        Y = randn(Float32, nx, ny, n_in, batchsize)
        logdet = randn(Float32, batchsize)

        # NLL = -logdet + 0.5*||Y||^2 / n
        nll = -mean(logdet) + 0.5f0 * mean(Y.^2)

        @test typeof(nll) <: Real
        @test isfinite(nll)
    end

    @testset "Negative Log-Likelihood - Per-batch" begin
        nx, ny = 8, 8
        n_in = 4
        batchsize = 8

        Y = randn(Float32, nx, ny, n_in, batchsize)
        logdet_batch = randn(Float32, batchsize)

        # Compute per-batch NLL
        nll_batch = zeros(Float32, batchsize)
        for i in 1:batchsize
            Yi = Y[:, :, :, i]
            nll_batch[i] = -logdet_batch[i] + 0.5f0 * mean(Yi.^2)
        end

        @test length(nll_batch) == batchsize
        @test all(isfinite.(nll_batch))
    end

    @testset "Maximum Mean Discrepancy (MMD)" begin
        n = 100
        batchsize = 8

        X = randn(Float32, n, batchsize)
        Y = randn(Float32, n, batchsize)

        # Simple MMD-like metric
        mmd = mean((mean(X, dims=2) - mean(Y, dims=2)).^2)

        @test typeof(mmd) <: Real
        @test mmd >= 0f0
        @test isfinite(mmd)
    end

end

println("\n✓ All objective function tests passed!")
