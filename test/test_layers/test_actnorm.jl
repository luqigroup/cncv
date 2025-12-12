# Test for ActNorm layer (non-CV version)
# Tests include: invertibility, gradient tests, per-batch logdet

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(123)

@testset "ActNorm Layer Tests" begin

    ###############################################################################
    # Test 2D ActNorm
    ###############################################################################

    @testset "ActNorm 2D - Basic" begin
        # Parameters
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        # Create layer
        AN = ActNorm(nc; logdet=true)

        # Input
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward pass
        Y, logdet = AN.forward(X)

        @test size(Y) == size(X)
        @test !isnothing(AN.s.data)
        @test !isnothing(AN.b.data)

        # Test that parameters are initialized (output has zero mean and unit variance per channel)
        Y_mean = mean(Y; dims=(1,2,4))
        Y_std = std(Y; dims=(1,2,4))
        @test isapprox(Y_mean, zeros(Float32, 1, 1, nc, 1); atol=1e-5)
        @test isapprox(Y_std, ones(Float32, 1, 1, nc, 1); atol=1e-1)
    end

    @testset "ActNorm 2D - Per-batch logdet" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward with per_batch=false (batch-averaged, default)
        Y1, logdet_avg = AN.forward(X; logdet_per_batch=false)

        # Forward with per_batch=true
        Y2, logdet_batch = AN.forward(X; logdet_per_batch=true)

        @test size(Y1) == size(Y2)
        @test Y1 ≈ Y2  # Output should be the same

        # Check logdet dimensions
        @test typeof(logdet_avg) <: Real  # Single value
        @test length(logdet_batch) == batchsize  # Vector of length batchsize

        # Check that average of per-batch equals batch-averaged
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)

        # Verify logdet value
        expected_logdet = nx * ny * sum(log.(abs.(AN.s.data)))
        @test isapprox(logdet_avg, expected_logdet; rtol=1e-5)
        @test all(isapprox.(logdet_batch, expected_logdet; rtol=1e-5))
    end

    @testset "ActNorm 2D - Invertibility" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward-inverse
        Y, _ = AN.forward(X)
        X_reconstructed = AN.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-5)

        # Inverse-forward
        Y2 = AN.inverse(X)
        X_reconstructed2, _ = AN.forward(Y2)

        @test isapprox(X, X_reconstructed2; rtol=1e-5)
    end

    @testset "ActNorm 2D - Gradient Test (Input)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nc, batchsize)
        X0 = randn(Float32, nx, ny, nc, batchsize)
        dX = randn(Float32, nx, ny, nc, batchsize)

        # Loss function
        function loss(AN, X)
            Y, logdet = AN.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            ΔX = AN.backward(ΔY, Y)[1]
            return f, ΔX
        end

        # Initial loss
        f0, ΔX = loss(AN, X0)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test ActNorm 2D: input")
        for j=1:maxiter
            f = loss(AN, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ActNorm 2D - Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        AN = ActNorm(nc; logdet=true)
        AN.forward(randn(Float32, nx, ny, nc, batchsize))
        X = randn(Float32, nx, ny, nc, batchsize)

        # Initialize parameters
        θ0 = deepcopy(get_params(AN))

        # Create another instance with different params
        AN2 = ActNorm(nc; logdet=true)
        AN2.forward(randn(Float32, nx, ny, nc, batchsize))

        θ = deepcopy(get_params(AN2))

        # Loss function
        function loss_params(AN, X)
            Y, logdet = AN.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0 / batchsize
            ΔY = Y
            AN.backward(ΔY, Y)
            return f, deepcopy(get_params(AN))
        end

        # Initial loss
        f0, Δθ = loss_params(AN, X)

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

        println("\nGradient test ActNorm 2D: parameters")
        for j=1:maxiter
            # Set parameters
            for i = 1:length(θ0)
                AN.s.data .= (θ0[1].data + h*dθ[1].data)
                AN.b.data .= (θ0[2].data + h*dθ[2].data)
            end

            f = loss_params(AN, X)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    ###############################################################################
    # Test 3D ActNorm
    ###############################################################################

    @testset "ActNorm 3D - Basic" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, logdet = AN.forward(X)

        @test size(Y) == size(X)
        @test !isnothing(AN.s.data)
        @test !isnothing(AN.b.data)
    end

    @testset "ActNorm 3D - Per-batch logdet" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        # Test per-batch logdet
        Y1, logdet_avg = AN.forward(X; logdet_per_batch=false)
        Y2, logdet_batch = AN.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2
        @test length(logdet_batch) == batchsize
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "ActNorm 3D - Invertibility" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        AN = ActNorm(nc; logdet=true)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, _ = AN.forward(X)
        X_reconstructed = AN.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-5)
    end

end

println("\n✓ All ActNorm tests passed!")
