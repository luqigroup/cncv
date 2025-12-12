# Test for CouplingLayerBasic (non-CV version)
# Tests include: invertibility, gradient tests, per-batch logdet

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!, ResidualBlock

using CNCV

Random.seed!(789)

@testset "CouplingLayerBasic Tests" begin

    ###############################################################################
    # Test 2D CouplingLayerBasic
    ###############################################################################

    @testset "CouplingLayerBasic 2D - Basic" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        # Create layer
        CL = CouplingLayerBasic(n_in, n_hidden; logdet=true)

        # Split input
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # Forward pass
        Y1, Y2, logdet = CL.forward(X1, X2)

        @test size(Y1) == size(X1)
        @test size(Y2) == size(X2)
        @test typeof(logdet) <: Real
    end

    @testset "CouplingLayerBasic 2D - Per-batch logdet" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        CL = CouplingLayerBasic(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # Forward with per_batch=false (batch-averaged, default)
        Y1a, Y2a, logdet_avg = CL.forward(X1, X2; logdet_per_batch=false)

        # Forward with per_batch=true
        Y1b, Y2b, logdet_batch = CL.forward(X1, X2; logdet_per_batch=true)

        @test Y1a ≈ Y1b
        @test Y2a ≈ Y2b

        # Check logdet dimensions
        @test typeof(logdet_avg) <: Real
        @test length(logdet_batch) == batchsize

        # Check that average of per-batch equals batch-averaged
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "CouplingLayerBasic 2D - Invertibility" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        CL = CouplingLayerBasic(n_in, n_hidden; logdet=false)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # Forward-inverse
        Y1, Y2 = CL.forward(X1, X2)
        X1_recon, X2_recon = CL.inverse(Y1, Y2)

        @test isapprox(X1, X1_recon; rtol=1e-4)
        @test isapprox(X2, X2_recon; rtol=1e-4)

        # Inverse-forward
        Y1b, Y2b = CL.inverse(X1, X2)
        X1_recon2, X2_recon2 = CL.forward(Y1b, Y2b)

        @test isapprox(X1, X1_recon2; rtol=1e-4)
        @test isapprox(X2, X2_recon2; rtol=1e-4)
    end

    @testset "CouplingLayerBasic 2D - Gradient Test (Input)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasic(n_in, n_hidden; logdet=true)

        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        X1, X2 = tensor_split(X)
        X01, X02 = tensor_split(X0)
        dX1, dX2 = tensor_split(dX)

        # Loss function
        function loss(CL, X1, X2)
            Y1, Y2, logdet = CL.forward(X1, X2)
            Y = tensor_cat(Y1, Y2)
            f = -logdet + 0.5f0*norm(Y)^2f0

            ΔY = Y
            ΔY1, ΔY2 = tensor_split(ΔY)
            ΔX1, ΔX2, _, _ = CL.backward(ΔY1, ΔY2, Y1, Y2)

            return f, ΔX1, ΔX2
        end

        # Initial loss
        f0, ΔX1, ΔX2 = loss(CL, X01, X02)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test CouplingLayerBasic 2D: input")
        for j=1:maxiter
            f = loss(CL, X01 + h*dX1, X02 + h*dX2)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*(dot(dX1, ΔX1) + dot(dX2, ΔX2)))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "CouplingLayerBasic 2D - Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        # Two instances
        CL1 = CouplingLayerBasic(n_in, n_hidden; logdet=true)
        CL2 = CouplingLayerBasic(n_in, n_hidden; logdet=true)

        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        θ0 = deepcopy(get_params(CL1))
        θ = deepcopy(get_params(CL2))

        # Loss function
        function loss_params(CL, X1, X2)
            Y1, Y2, logdet = CL.forward(X1, X2)
            Y = tensor_cat(Y1, Y2)
            f = -logdet + 0.5f0*norm(Y)^2f0

            ΔY = Y
            ΔY1, ΔY2 = tensor_split(ΔY)
            CL.backward(ΔY1, ΔY2, Y1, Y2)

            return f, deepcopy(get_params(CL))
        end

        f0, Δθ = loss_params(CL1, X1, X2)

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

        println("\nGradient test CouplingLayerBasic 2D: parameters")
        for j=1:maxiter
            # Set perturbed parameters
            θ_curr = θ0 + h*dθ
            set_params!(CL1, θ_curr)

            f = loss_params(CL1, X1, X2)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    ###############################################################################
    # Test 3D CouplingLayerBasic
    ###############################################################################

    @testset "CouplingLayerBasic 3D - Basic" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasic3D(n_in, n_hidden; logdet=true)

        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, logdet = CL.forward(X1, X2)

        @test size(Y1) == size(X1)
        @test size(Y2) == size(X2)
        @test typeof(logdet) <: Real
    end

    @testset "CouplingLayerBasic 3D - Per-batch logdet" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasic3D(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1a, Y2a, logdet_avg = CL.forward(X1, X2; logdet_per_batch=false)
        Y1b, Y2b, logdet_batch = CL.forward(X1, X2; logdet_per_batch=true)

        @test Y1a ≈ Y1b
        @test Y2a ≈ Y2b
        @test length(logdet_batch) == batchsize
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)
    end

    @testset "CouplingLayerBasic 3D - Invertibility" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasic3D(n_in, n_hidden; logdet=false)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2 = CL.forward(X1, X2)
        X1_recon, X2_recon = CL.inverse(Y1, Y2)

        @test isapprox(X1, X1_recon; rtol=1e-4)
        @test isapprox(X2, X2_recon; rtol=1e-4)
    end

    @testset "CouplingLayerBasic - Logdet Consistency" begin
        # Test that logdet is computed correctly
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasic(n_in, n_hidden; logdet=true)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, logdet_avg = CL.forward(X1, X2; logdet_per_batch=false)
        _, _, logdet_batch = CL.forward(X1, X2; logdet_per_batch=true)

        # Manually compute expected logdet for verification
        # logdet = sum(log(|S|)) / batchsize for averaged
        # We can't compute exact value without knowing S, but we can check consistency
        @test isapprox(mean(logdet_batch), logdet_avg; rtol=1e-5)

        # All batch elements should have same logdet (deterministic transformation)
        # Actually, logdet can vary per batch if S varies, so we just check they're reasonable
        @test all(isfinite.(logdet_batch))
        @test isfinite(logdet_avg)
    end

end

println("\n✓ All CouplingLayerBasic tests passed!")
