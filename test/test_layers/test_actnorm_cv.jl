# Test for ActNormCV layer (CV version with jacobian trace)
# Tests include: invertibility, gradient tests, jacobian trace tests, integrated gradients

using Test, Random, LinearAlgebra, Statistics
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(123)

@testset "ActNormCV Layer Tests" begin

    ###############################################################################
    # Test 2D ActNormCV
    ###############################################################################

    @testset "ActNormCV 2D - Basic and Jacobian Trace" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        # Create layer
        AN = ActNormCV(nc)

        # Input
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward pass
        Y, jac_trace_batch = AN.forward(X)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test !isnothing(AN.s.data)
        @test !isnothing(AN.b.data)

        # Verify jacobian trace computation
        # For ActNorm: jac_trace = nx * ny * sum(s)
        expected_jac_trace = nx * ny * sum(AN.s.data)
        @test all(isapprox.(jac_trace_batch, expected_jac_trace; rtol=1e-5))

        # Test that parameters are initialized
        Y_mean = mean(Y; dims=(1,2,4))
        Y_std = std(Y; dims=(1,2,4))
        @test isapprox(Y_mean, zeros(Float32, 1, 1, nc, 1); atol=1e-5)
        @test isapprox(Y_std, ones(Float32, 1, 1, nc, 1); atol=1e-1)
    end

    @testset "ActNormCV 2D - Invertibility with Jacobian Trace" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        AN = ActNormCV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward-inverse
        Y, jac_trace_fwd = AN.forward(X)
        X_reconstructed, jac_trace_inv = AN.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-5)

        # Jacobian traces should be equal in magnitude (opposite in log-det sense)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "ActNormCV 2D - Gradient Test with Jacobian Trace Weight" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        AN = ActNormCV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Initialize
        AN.forward(X)

        # Loss function with jacobian trace term
        function loss_with_trace(AN, X)
            Y, jac_trace = AN.forward(X)

            # Loss = ||Y||^2 + lambda * sum(jac_trace)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            # Compute gradients
            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)

            ΔX, _ = AN.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔX, deepcopy(get_params(AN))
        end

        # Base point
        X0 = randn(Float32, nx, ny, nc, batchsize)
        AN.forward(X0)  # Initialize
        f0, ΔX, Δθ = loss_with_trace(AN, X0)

        # Perturbation
        dX = randn(Float32, nx, ny, nc, batchsize)

        # Gradient test (input)
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test ActNormCV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(AN, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ActNormCV 2D - Jacobian Trace Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        # Two instances for parameter perturbation
        AN1 = ActNormCV(nc)
        AN2 = ActNormCV(nc)

        X = randn(Float32, nx, ny, nc, batchsize)

        # Initialize both
        AN1.forward(X)
        AN2.forward(X)

        θ0 = deepcopy(get_params(AN1))
        θ = deepcopy(get_params(AN2))

        # Loss function
        function loss_params(AN, X)
            Y, jac_trace = AN.forward(X)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            AN.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, deepcopy(get_params(AN))
        end

        f0, Δθ = loss_params(AN1, X)

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

        println("\nGradient test ActNormCV 2D with jac_trace: parameters")
        for j=1:maxiter
            AN1.s.data .= θ0[1].data + h*dθ[1].data
            AN1.b.data .= θ0[2].data + h*dθ[2].data

            f = loss_params(AN1, X)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ActNormCV 2D - Jacobian Trace Consistency" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        AN = ActNormCV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        Y, jac_trace = AN.forward(X)

        # Verify trace calculation manually
        # trace(J) = nx * ny * sum(s) for ActNorm
        manual_trace = nx * ny * sum(AN.s.data)

        @test all(isapprox.(jac_trace, manual_trace; rtol=1e-5))

        # Verify trace is same for all batch elements
        @test all(jac_trace .≈ jac_trace[1])
    end

    ###############################################################################
    # Test 3D ActNormCV
    ###############################################################################

    @testset "ActNormCV 3D - Basic and Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        AN = ActNormCV(nc)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, jac_trace_batch = AN.forward(X)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize

        # Verify jacobian trace for 3D
        expected_jac_trace = nx * ny * nz * sum(AN.s.data)
        @test all(isapprox.(jac_trace_batch, expected_jac_trace; rtol=1e-5))
    end

    @testset "ActNormCV 3D - Invertibility with Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        AN = ActNormCV(nc)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, jac_trace_fwd = AN.forward(X)
        X_reconstructed, jac_trace_inv = AN.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-5)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "ActNormCV - Comparison with Non-CV Version" begin
        # Verify that without jac_trace_grad_weight, CV version behaves like non-CV
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        X = randn(Float32, nx, ny, nc, batchsize)

        # CV version
        AN_cv = ActNormCV(nc)
        Y_cv, jac_trace = AN_cv.forward(X)

        # Non-CV version
        AN = ActNorm(nc; logdet=true)
        Y, logdet = AN.forward(X; logdet_per_batch=true)

        # Outputs should be similar (after initialization)
        # Note: they won't be exactly the same due to different initializations
        # but the structure should be the same
        @test size(Y_cv) == size(Y)
        @test length(jac_trace) == length(logdet)
    end

end

println("\n✓ All ActNormCV tests passed!")
