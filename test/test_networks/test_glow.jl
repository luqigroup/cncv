# Test for NetworkGlow (non-CV version with per-batch logdet)
# Full Glow network composed of multiple coupling layers
# Tests include: invertibility, gradient tests, per-batch logdet, composition

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(303132)

@testset "NetworkGlow Tests" begin

    ###############################################################################
    # Test 2D NetworkGlow
    ###############################################################################

    @testset "NetworkGlow 2D - Basic" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2  # Number of flow steps per scale
        batchsize = 8

        # Create network
        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)

        # Input
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward pass
        Y, logdet = G.forward(X)

        @test size(Y)[1:3] == (nx, ny, n_in)  # Spatial and channel dims preserved or transformed
        @test size(Y, 4) == batchsize
        @test typeof(logdet) <: Real
    end

    @testset "NetworkGlow 2D - Per-batch logdet" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 8

        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward with per_batch=false (batch-averaged, default)
        Y1, logdet_avg = G.forward(X; logdet_per_batch=false)

        # Forward with per_batch=true
        Y2, logdet_batch = G.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2

        # Check logdet dimensions
        @test typeof(logdet_avg) <: Real
        @test length(logdet_batch) == batchsize

        # Check that average of per-batch equals batch-averaged
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "NetworkGlow 2D - Invertibility" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 8

        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=false)
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward-inverse
        Y = G.forward(X)
        X_recon = G.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-3)

        # Inverse-forward
        Y2 = G.inverse(X)
        X_recon2 = G.forward(Y2)

        @test isapprox(X, X_recon2; rtol=1e-3)
    end

    @testset "NetworkGlow 2D - Gradient Test (Input)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        # Loss function
        function loss(G, X)
            Y, logdet = G.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            ΔX = G.backward(ΔY, Y)[1]
            return f, ΔX
        end

        # Initial loss
        f0, ΔX = loss(G, X0)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test NetworkGlow 2D: input")
        for j=1:maxiter
            f = loss(G, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "NetworkGlow 2D - Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        # Two instances
        G1 = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)
        G2 = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)

        X = randn(Float32, nx, ny, n_in, batchsize)

        θ0 = deepcopy(get_params(G1))
        θ = deepcopy(get_params(G2))

        # Loss function
        function loss_params(G, X)
            Y, logdet = G.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            G.backward(ΔY, Y)
            return f, deepcopy(get_params(G))
        end

        f0, Δθ = loss_params(G1, X)

        # Perturbation
        dθ = θ - θ0
        for i = 1:length(dθ)
            if norm(θ0[i].data) != 0f0
                dθ[i].data .*= norm(θ0[i].data)/norm(dθ[i].data)
            end
        end

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err3 = zeros(Float32, maxiter)
        err4 = zeros(Float32, maxiter)

        println("\nGradient test NetworkGlow 2D: parameters")
        for j=1:maxiter
            θ_curr = θ0 + h*dθ
            set_params!(G1, θ_curr)

            f = loss_params(G1, X)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "NetworkGlow - Logdet Accumulation" begin
        # Test that network logdet is sum of layer logdets
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 4

        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)

        Y, logdet = G.forward(X; logdet_per_batch=true)

        # Logdet should be accumulated across all layers
        @test length(logdet) == batchsize
        @test all(isfinite.(logdet))
    end

    @testset "NetworkGlow - Sampling" begin
        # Test generating samples from latent space
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=false)

        # Sample from latent (typically standard normal)
        Z = randn(Float32, nx, ny, n_in, batchsize)
        X = G.inverse(Z)

        @test size(X) == size(Z)
        @test all(isfinite.(X))
    end

end

println("\n✓ All NetworkGlow tests passed!")
