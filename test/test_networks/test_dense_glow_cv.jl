# Test for DenseGlowCV (CV version with jacobian trace)
# Dense Glow network with control variate support
# Tests include: basic functionality, invertibility, jacobian trace

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(505152)

@testset "DenseGlowCV Tests" begin

    @testset "DenseGlowCV - Basic and Jacobian Trace" begin
        n_in = 4
        n_hidden = 8
        K = 3
        batchsize = 8

        G = DenseGlowCV(n_in, n_hidden, K)
        X = randn(Float32, 1, 1, n_in, batchsize)

        Y, jac_trace_batch = G.forward(X)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "DenseGlowCV - Invertibility with Jacobian Trace" begin
        n_in = 4
        n_hidden = 8
        K = 3
        batchsize = 8

        G = DenseGlowCV(n_in, n_hidden, K)
        X = randn(Float32, 1, 1, n_in, batchsize)

        Y, jac_trace_fwd = G.forward(X)
        X_recon, jac_trace_inv = G.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-4)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "DenseGlowCV - Gradient Test with Trace Weight" begin
        n_in = 4
        n_hidden = 8
        K = 2
        batchsize = 4

        G = DenseGlowCV(n_in, n_hidden, K)
        X0 = randn(Float32, 1, 1, n_in, batchsize)
        dX = randn(Float32, 1, 1, n_in, batchsize)

        function loss_with_trace(G, X)
            Y, jac_trace = G.forward(X)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            ΔX = G.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)[1]

            return f, ΔX
        end

        f0, ΔX = loss_with_trace(G, X0)

        h = 0.1f0
        err2 = abs(loss_with_trace(G, X0 + h*dX)[1] - f0 - h*dot(dX, ΔX))
        h /= 2f0
        err2_half = abs(loss_with_trace(G, X0 + h*dX)[1] - f0 - h*dot(dX, ΔX))

        @test err2_half < err2  # Second-order convergence
    end

end

println("\n✓ All DenseGlowCV tests passed!")
