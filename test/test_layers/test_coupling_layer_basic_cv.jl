# Test for CouplingLayerBasicCV (CV version with jacobian trace)
# Tests include: invertibility, gradient tests, jacobian trace tests, integrated gradients

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!, ResidualBlock

using CNCV

Random.seed!(789)

@testset "CouplingLayerBasicCV Tests" begin

    ###############################################################################
    # Test 2D CouplingLayerBasicCV
    ###############################################################################

    @testset "CouplingLayerBasicCV 2D - Basic and Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        # Create layer
        CL = CouplingLayerBasicCV(n_in, n_hidden)

        # Split input
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # Forward pass
        Y1, Y2, jac_trace_batch = CL.forward(X1, X2)

        @test size(Y1) == size(X1)
        @test size(Y2) == size(X2)
        @test length(jac_trace_batch) == batchsize

        # Jacobian trace should be sum of S for coupling layer
        # We can't verify exact value without knowing S, but check it's finite
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "CouplingLayerBasicCV 2D - Invertibility with Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        batchsize = 8

        CL = CouplingLayerBasicCV(n_in, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # Forward-inverse
        Y1, Y2, jac_trace_fwd = CL.forward(X1, X2)
        X1_recon, X2_recon, jac_trace_inv = CL.inverse(Y1, Y2)

        @test isapprox(X1, X1_recon; rtol=1e-4)
        @test isapprox(X2, X2_recon; rtol=1e-4)

        # Jacobian traces should be equal (same S in forward and inverse)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "CouplingLayerBasicCV 2D - Gradient Test with Jacobian Trace Weight" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasicCV(n_in, n_hidden)

        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        X1, X2 = tensor_split(X)
        X01, X02 = tensor_split(X0)
        dX1, dX2 = tensor_split(dX)

        # Loss function with jacobian trace term
        function loss_with_trace(CL, X1, X2)
            Y1, Y2, jac_trace = CL.forward(X1, X2)
            Y = tensor_cat(Y1, Y2)

            # Loss = ||Y||^2 + lambda * sum(jac_trace)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            # Compute gradients
            ΔY = Y
            ΔY1, ΔY2 = tensor_split(ΔY)
            jac_trace_grad_weight = fill(lambda, batchsize)

            ΔX1, ΔX2, _, _ = CL.backward(ΔY1, ΔY2, Y1, Y2;
                                         jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔX1, ΔX2, deepcopy(get_params(CL))
        end

        # Base point
        f0, ΔX1, ΔX2, Δθ = loss_with_trace(CL, X01, X02)

        # Gradient test (input)
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test CouplingLayerBasicCV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(CL, X01 + h*dX1, X02 + h*dX2)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*(dot(dX1, ΔX1) + dot(dX2, ΔX2)))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "CouplingLayerBasicCV 2D - Jacobian Trace Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        # Two instances
        CL1 = CouplingLayerBasicCV(n_in, n_hidden)
        CL2 = CouplingLayerBasicCV(n_in, n_hidden)

        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        θ0 = deepcopy(get_params(CL1))
        θ = deepcopy(get_params(CL2))

        # Loss function
        function loss_params(CL, X1, X2)
            Y1, Y2, jac_trace = CL.forward(X1, X2)
            Y = tensor_cat(Y1, Y2)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            ΔY1, ΔY2 = tensor_split(ΔY)
            jac_trace_grad_weight = fill(lambda, batchsize)
            CL.backward(ΔY1, ΔY2, Y1, Y2; jac_trace_grad_weight=jac_trace_grad_weight)

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

        # Gradient test (parameters)
        h = 0.1f0
        maxiter = 5
        err3 = zeros(Float32, maxiter)
        err4 = zeros(Float32, maxiter)

        println("\nGradient test CouplingLayerBasicCV 2D with jac_trace: parameters")
        for j=1:maxiter
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

    @testset "CouplingLayerBasicCV 2D - Jacobian Trace Consistency" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasicCV(n_in, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, jac_trace, logS_T1, S = CL.forward(X1, X2; save=true)

        # Verify trace calculation manually
        # For coupling: jac_trace = sum(S) per batch
        manual_trace = dropdims(sum(S; dims=tuple(1:ndims(S)-1...)); dims=tuple(1:ndims(S)-1...))

        @test isapprox(jac_trace, manual_trace; rtol=1e-5)
    end

    ###############################################################################
    # Test 3D CouplingLayerBasicCV
    ###############################################################################

    @testset "CouplingLayerBasicCV 3D - Basic and Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasicCV3D(n_in, n_hidden)

        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, jac_trace_batch = CL.forward(X1, X2)

        @test size(Y1) == size(X1)
        @test size(Y2) == size(X2)
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "CouplingLayerBasicCV 3D - Invertibility with Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasicCV3D(n_in, n_hidden)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, jac_trace_fwd = CL.forward(X1, X2)
        X1_recon, X2_recon, jac_trace_inv = CL.inverse(Y1, Y2)

        @test isapprox(X1, X1_recon; rtol=1e-4)
        @test isapprox(X2, X2_recon; rtol=1e-4)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "CouplingLayerBasicCV - Comparison with Non-CV Version" begin
        # Verify that without jac_trace_grad_weight, behavior is similar
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        # CV version
        CL_cv = CouplingLayerBasicCV(n_in, n_hidden)
        Y1_cv, Y2_cv, jac_trace = CL_cv.forward(X1, X2)

        # Non-CV version
        CL = CouplingLayerBasic(n_in, n_hidden; logdet=true)
        Y1, Y2, logdet = CL.forward(X1, X2; logdet_per_batch=true)

        # Structure should be the same (though values differ due to different RB initialization)
        @test size(Y1_cv) == size(Y1)
        @test size(Y2_cv) == size(Y2)
        @test length(jac_trace) == length(logdet)
    end

    @testset "CouplingLayerBasicCV - Backward without Trace Weight" begin
        # Test that backward works correctly when jac_trace_grad_weight is nothing
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        batchsize = 4

        CL = CouplingLayerBasicCV(n_in, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X1, X2 = tensor_split(X)

        Y1, Y2, _ = CL.forward(X1, X2)
        ΔY = randn(Float32, size(tensor_cat(Y1, Y2))...)
        ΔY1, ΔY2 = tensor_split(ΔY)

        # Backward without jac_trace_grad_weight
        ΔX1, ΔX2, X1_recon, X2_recon = CL.backward(ΔY1, ΔY2, Y1, Y2;
                                                    jac_trace_grad_weight=nothing)

        @test size(ΔX1) == size(X1)
        @test size(ΔX2) == size(X2)
        @test isapprox(X1, X1_recon; rtol=1e-5)
        @test isapprox(X2, X2_recon; rtol=1e-5)
    end

end

println("\n✓ All CouplingLayerBasicCV tests passed!")
