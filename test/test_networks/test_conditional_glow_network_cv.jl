# Test for NetworkConditionalGlowCV (CV version with jacobian trace)
# Full conditional Glow network with control variate support
# Tests include: invertibility, gradient tests, jacobian trace tests, conditioning

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(404142)

@testset "NetworkConditionalGlowCV Tests" begin

    @testset "NetworkConditionalGlowCV 2D - Basic and Jacobian Trace" begin
        nx, ny, n_in, n_cond, n_hidden, depth, K, batchsize = 16, 16, 4, 2, 8, 2, 2, 8

        G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        Y, jac_trace_batch = G.forward(X, C)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "NetworkConditionalGlowCV 2D - Invertibility with Jacobian Trace" begin
        nx, ny, n_in, n_cond, n_hidden, depth, K, batchsize = 16, 16, 4, 2, 8, 2, 2, 8

        G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        Y, jac_trace_fwd = G.forward(X, C)
        X_recon, jac_trace_inv = G.inverse(Y, C)

        @test isapprox(X, X_recon; rtol=1e-3)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-4)
    end

    @testset "NetworkConditionalGlowCV 2D - Conditioning Effect" begin
        nx, ny, n_in, n_cond, n_hidden, depth, K, batchsize = 8, 8, 4, 2, 8, 1, 2, 4

        G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C1 = randn(Float32, nx, ny, n_cond, batchsize)
        C2 = randn(Float32, nx, ny, n_cond, batchsize)

        Y1, _ = G.forward(X, C1)
        Y2, _ = G.forward(X, C2)

        @test !isapprox(Y1, Y2; rtol=0.1)
    end

    @testset "NetworkConditionalGlowCV 2D - Gradient Test with Trace Weight" begin
        nx, ny, n_in, n_cond, n_hidden, depth, K, batchsize = 8, 8, 4, 2, 8, 1, 2, 4

        G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, depth, K)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        function loss_with_trace(G, X, C)
            Y, jac_trace = G.forward(X, C)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            ΔX, _ = G.backward(ΔY, Y, C; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔX
        end

        f0, ΔX = loss_with_trace(G, X0, C)

        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test NetworkConditionalGlowCV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(G, X0 + h*dX, C)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "NetworkConditionalGlowCV - Conditional Sampling with Trace" begin
        nx, ny, n_in, n_cond, n_hidden, depth, K, batchsize = 8, 8, 4, 2, 8, 1, 2, 4

        G = NetworkConditionalGlowCV(n_in, n_cond, n_hidden, depth, K)

        Z = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)
        X, jac_trace = G.inverse(Z, C)

        @test size(X) == size(Z)
        @test length(jac_trace) == batchsize
        @test all(isfinite.(X))
        @test all(isfinite.(jac_trace))
    end

end

println("\n✓ All NetworkConditionalGlowCV tests passed!")
