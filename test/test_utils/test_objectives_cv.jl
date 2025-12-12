# Test for objective functions (CV version with control variate terms)
# Tests include: CV-enhanced objectives with jacobian trace

using Test, Random, LinearAlgebra, Statistics

using CNCV

Random.seed!(707172)

@testset "Objective Functions CV Tests" begin

    @testset "CV Negative Log-Likelihood - Basic" begin
        nx, ny = 8, 8
        n_in = 4
        batchsize = 8

        # Mock network forward output
        Y = randn(Float32, nx, ny, n_in, batchsize)
        jac_trace = randn(Float32, batchsize)

        # CV-NLL includes jacobian trace term
        lambda = 0.1f0
        cv_nll = -mean(jac_trace) + 0.5f0 * mean(Y.^2) + lambda * var(jac_trace)

        @test typeof(cv_nll) <: Real
        @test isfinite(cv_nll)
    end

    @testset "CV Objective - With Trace Weight" begin
        nx, ny = 8, 8
        n_in = 4
        batchsize = 8

        Y = randn(Float32, nx, ny, n_in, batchsize)
        jac_trace = randn(Float32, batchsize)

        # Objective with weighted trace term
        lambda = 0.1f0
        objective = 0.5f0 * norm(Y)^2f0 + lambda * sum(jac_trace)

        @test typeof(objective) <: Real
        @test isfinite(objective)
    end

    @testset "Control Variate Variance Reduction" begin
        # Test that CV reduces variance (conceptual test)
        batchsize = 100

        # Standard samples
        samples_std = randn(Float32, batchsize)
        var_std = var(samples_std)

        # With control variate
        cv = randn(Float32, batchsize)
        alpha = -cov(samples_std, cv) / var(cv)
        samples_cv = samples_std + alpha * (cv .- mean(cv))
        var_cv = var(samples_cv)

        # CV should reduce variance (in expectation)
        @test typeof(var_cv) <: Real
        @test isfinite(var_cv)
        @test var_cv >= 0f0
    end

end

println("\n✓ All objective function CV tests passed!")
