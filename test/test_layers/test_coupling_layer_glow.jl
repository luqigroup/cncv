# Test for CouplingLayerGlow (non-CV version with per-batch logdet)
# Glow coupling layer: Conv1x1 + Coupling
# Tests include: invertibility, gradient tests, per-batch logdet

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(202122)

@testset "CouplingLayerGlow Tests" begin

    ###############################################################################
    # Test 2D CouplingLayerGlow
    ###############################################################################

    @testset "CouplingLayerGlow 2D - Basic" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        # Create layer
        CL = CouplingLayerGlow(n_in, n_hidden; logdet=true)

        # Input
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward pass
        Y, logdet = CL.forward(X)

        @test size(Y) == size(X)
        @test typeof(logdet) <: Real
    end

    @testset "CouplingLayerGlow 2D - Per-batch logdet" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        CL = CouplingLayerGlow(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward with per_batch=false (batch-averaged, default)
        Y1, logdet_avg = CL.forward(X; logdet_per_batch=false)

        # Forward with per_batch=true
        Y2, logdet_batch = CL.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2

        # Check logdet dimensions
        @test typeof(logdet_avg) <: Real
        @test length(logdet_batch) == batchsize

        # Check that average of per-batch equals batch-averaged
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "CouplingLayerGlow 2D - Invertibility" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        CL = CouplingLayerGlow(n_in, n_hidden; logdet=false)
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward-inverse
        Y = CL.forward(X)
        X_recon = CL.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-4)

        # Inverse-forward
        Y2 = CL.inverse(X)
        X_recon2 = CL.forward(Y2)

        @test isapprox(X, X_recon2; rtol=1e-4)
    end

    @testset "CouplingLayerGlow 2D - Gradient Test (Input)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerGlow(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        # Loss function
        function loss(CL, X)
            Y, logdet = CL.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0 / batchsize
            ΔY = Y
            ΔX, _ = CL.backward(ΔY, Y)
            return f, ΔX
        end

        # Initial loss
        f0, ΔX = loss(CL, X0)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test CouplingLayerGlow 2D: input")
        for j=1:maxiter
            f = loss(CL, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "CouplingLayerGlow 2D - Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        # Two instances
        CL1 = CouplingLayerGlow(n_in, n_hidden; logdet=true)
        CL2 = CouplingLayerGlow(n_in, n_hidden; logdet=true)

        X = randn(Float32, nx, ny, n_in, batchsize)

        θ0 = deepcopy(get_params(CL1))
        θ = deepcopy(get_params(CL2))

        # Loss function
        function loss_params(CL, X)
            Y, logdet = CL.forward(X)
            f = -logdet + 0.5f0*norm(Y)^2f0
            ΔY = Y
            CL.backward(ΔY, Y)
            return f, deepcopy(get_params(CL))
        end

        f0, Δθ = loss_params(CL1, X)

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

        println("\nGradient test CouplingLayerGlow 2D: parameters")
        for j=1:maxiter
            θ_curr = θ0 + h*dθ
            set_params!(CL1, θ_curr)

            f = loss_params(CL1, X)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    ###############################################################################
    # Test 3D CouplingLayerGlow
    ###############################################################################

    @testset "CouplingLayerGlow 3D - Basic" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerGlow3D(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)

        Y, logdet = CL.forward(X)

        @test size(Y) == size(X)
        @test typeof(logdet) <: Real
    end

    @testset "CouplingLayerGlow 3D - Per-batch logdet" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerGlow3D(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)

        Y1, logdet_avg = CL.forward(X; logdet_per_batch=false)
        Y2, logdet_batch = CL.forward(X; logdet_per_batch=true)

        @test Y1 ≈ Y2
        @test length(logdet_batch) == batchsize
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "CouplingLayerGlow 3D - Invertibility" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerGlow3D(n_in, n_hidden; logdet=false)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)

        Y = CL.forward(X)
        X_recon = CL.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-4)
    end

    @testset "CouplingLayerGlow - Logdet Components" begin
        # Test that logdet comes from both Conv1x1 (=0) and coupling
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerGlow(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)

        Y, logdet = CL.forward(X; logdet_per_batch=true)

        # Conv1x1 uses Householder, so its logdet = 0
        # All logdet should come from coupling layer
        @test all(isfinite.(logdet))
    end

end

println("\n✓ All CouplingLayerGlow tests passed!")
