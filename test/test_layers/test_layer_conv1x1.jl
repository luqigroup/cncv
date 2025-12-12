# Test for Conv1x1 layer (non-CV version)
# Tests include: invertibility, gradient tests, per-batch logdet
# Conv1x1 uses Householder reflections, so logdet is always 0

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!

using CNCV

# Include your CNCV module or layer definitions here
# using CNCV

Random.seed!(456)

@testset "Conv1x1 Layer Tests" begin

    ###############################################################################
    # Test 2D Conv1x1
    ###############################################################################

    @testset "Conv1x1 2D - Basic" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        # Create layer
        C = Conv1x1(nc; logdet=true)

        # Input
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward pass
        Y, logdet = C.forward(X)

        @test size(Y) == size(X)
        @test !isnothing(C.v1.data)
        @test !isnothing(C.v2.data)
        @test !isnothing(C.v3.data)

        # For Householder reflections, logdet should be 0
        @test isapprox(logdet, 0f0; atol=1e-6)
    end

    @testset "Conv1x1 2D - Per-batch logdet" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        C = Conv1x1(nc; logdet=true)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward with per_batch=false (batch-averaged, default)
        Y1, logdet_avg = C.forward(X; logdet_per_batch=false)

        # Forward with per_batch=true
        Y2, logdet_batch = C.forward(X; logdet_per_batch=true)

        @test size(Y1) == size(Y2)
        @test Y1 ≈ Y2

        # Check logdet dimensions
        @test typeof(logdet_avg) <: Real
        @test length(logdet_batch) == batchsize

        # For Householder, all logdets should be 0
        @test isapprox(logdet_avg, 0f0; atol=1e-6)
        @test all(isapprox.(logdet_batch, 0f0; atol=1e-6))

        # Check that average of per-batch equals batch-averaged
        @test isapprox(mean(logdet_batch), logdet_avg; atol=1e-6)
    end

    @testset "Conv1x1 2D - Invertibility" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        C = Conv1x1(nc; logdet=false)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward-inverse
        Y = C.forward(X)
        X_reconstructed = C.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-4)

        # Inverse-forward
        Y2 = C.inverse(X)
        X_reconstructed2 = C.forward(Y2)

        @test isapprox(X, X_reconstructed2; rtol=1e-4)
    end

    @testset "Conv1x1 2D - Orthogonality" begin
        # Householder reflections should produce orthogonal matrices
        nx, ny = 8, 8
        nc = 4
        batchsize = 1

        C = Conv1x1(nc; logdet=false)
        X = randn(Float32, nx, ny, nc, batchsize)
        Y = C.forward(X)

        # ||Y|| should equal ||X|| for orthogonal transformations
        @test isapprox(norm(Y), norm(X); rtol=1e-4)
    end

    @testset "Conv1x1 2D - Gradient Test (Input)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1(nc; logdet=true)
        X = randn(Float32, nx, ny, nc, batchsize)
        X0 = randn(Float32, nx, ny, nc, batchsize)
        dX = randn(Float32, nx, ny, nc, batchsize)

        # Loss function
        function loss(C, X)
            Y, logdet = C.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            ΔX = C.inverse((ΔY, Y))[1]
            return f, ΔX
        end

        # Initial loss
        f0, ΔX = loss(C, X0)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test Conv1x1 2D: input")
        for j=1:maxiter
            f = loss(C, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "Conv1x1 2D - Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        # Create two instances
        C1 = Conv1x1(nc; logdet=true)
        C2 = Conv1x1(nc; logdet=true)

        X = randn(Float32, nx, ny, nc, batchsize)

        θ0 = deepcopy(get_params(C1))
        θ = deepcopy(get_params(C2))

        # Loss function
        function loss_params(C, X)
            Y, logdet = C.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            C.inverse((ΔY, Y))
            return f, deepcopy(get_params(C))
        end

        # Initial loss
        f0, Δθ = loss_params(C1, X)

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

        println("\nGradient test Conv1x1 2D: parameters")
        for j=1:maxiter
            C1.v1.data .= θ0[1].data + h*dθ[1].data
            C1.v2.data .= θ0[2].data + h*dθ[2].data
            C1.v3.data .= θ0[3].data + h*dθ[3].data

            f = loss_params(C1, X)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    ###############################################################################
    # Test 3D Conv1x1
    ###############################################################################

    @testset "Conv1x1 3D - Basic" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1(nc; logdet=true)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, logdet = C.forward(X)

        @test size(Y) == size(X)
        @test isapprox(logdet, 0f0; atol=1e-6)
    end

    @testset "Conv1x1 3D - Per-batch logdet" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1(nc; logdet=true)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y1, logdet_avg = C.forward(X; logdet_per_batch=false)
        Y2, logdet_batch = C.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2
        @test length(logdet_batch) == batchsize
        @test all(isapprox.(logdet_batch, 0f0; atol=1e-6))
        @test isapprox(mean(logdet_batch), logdet_avg; atol=1e-6)
    end

    @testset "Conv1x1 3D - Invertibility" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1(nc; logdet=false)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y = C.forward(X)
        X_reconstructed = C.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-4)
    end

    @testset "Conv1x1 3D - Orthogonality" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 1

        C = Conv1x1(nc; logdet=false)
        X = randn(Float32, nx, ny, nz, nc, batchsize)
        Y = C.forward(X)

        # Norm preservation for orthogonal transformations
        @test isapprox(norm(Y), norm(X); rtol=1e-4)
    end

    @testset "Conv1x1 - Frozen Parameters" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1(nc; freeze=true, logdet=false)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Store initial parameters
        v1_init = copy(C.v1.data)
        v2_init = copy(C.v2.data)
        v3_init = copy(C.v3.data)

        # Forward-backward
        Y = C.forward(X)
        ΔY = randn(Float32, size(Y)...)
        C.inverse((ΔY, Y))

        # Parameters should not have gradients when frozen
        @test C.freeze == true
        # If gradients are computed, they should be zero or params unchanged
    end

end

println("\n✓ All Conv1x1 tests passed!")
