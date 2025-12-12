# Test for DenseGlow (non-CV version)
# Dense Glow network (no spatial squeezing)
# Tests include: basic functionality, invertibility, per-batch logdet

using Test, Random, LinearAlgebra, Statistics

using CNCV

Random.seed!(505152)

@testset "DenseGlow Tests" begin

    @testset "DenseGlow - Basic" begin
        n_in = 4
        n_hidden = 8
        K = 3
        batchsize = 8

        G = DenseGlow(n_in, n_hidden, K; logdet=true)
        X = randn(Float32, 1, 1, n_in, batchsize)

        Y, logdet = G.forward(X)

        @test size(Y) == size(X)
        @test typeof(logdet) <: Real
    end

    @testset "DenseGlow - Per-batch logdet" begin
        n_in = 4
        n_hidden = 8
        K = 3
        batchsize = 8

        G = DenseGlow(n_in, n_hidden, K; logdet=true)
        X = randn(Float32, 1, 1, n_in, batchsize)

        Y1, logdet_avg = G.forward(X; logdet_per_batch=false)
        Y2, logdet_batch = G.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2
        @test typeof(logdet_avg) <: Real
        @test length(logdet_batch) == batchsize
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "DenseGlow - Invertibility" begin
        n_in = 4
        n_hidden = 8
        K = 3
        batchsize = 8

        G = DenseGlow(n_in, n_hidden, K; logdet=false)
        X = randn(Float32, 1, 1, n_in, batchsize)

        Y = G.forward(X)
        X_recon = G.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-4)
    end

end

println("\n✓ All DenseGlow tests passed!")
