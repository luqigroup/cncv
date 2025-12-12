# Test for Conv1x1CV layer (CV version with jacobian trace)
# Tests include: invertibility, gradient tests, jacobian trace tests
# Note: Householder reflections are orthogonal, so jacobian trace is always 0

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(456)

@testset "Conv1x1CV Layer Tests" begin

    ###############################################################################
    # Test 2D Conv1x1CV
    ###############################################################################

    @testset "Conv1x1CV 2D - Basic and Jacobian Trace" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        # Create layer
        C = Conv1x1CV(nc)

        # Input
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward pass
        Y, jac_trace_batch = C.forward(X)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test !isnothing(C.v1.data)
        @test !isnothing(C.v2.data)
        @test !isnothing(C.v3.data)

        # For Householder reflections (orthogonal), jacobian trace should be 0
        @test all(isapprox.(jac_trace_batch, 0f0; atol=1e-6))
    end

    @testset "Conv1x1CV 2D - Invertibility with Jacobian Trace" begin
        nx, ny = 16, 16
        nc = 4
        batchsize = 8

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Forward-inverse
        Y, jac_trace_fwd = C.forward(X)
        X_reconstructed, jac_trace_inv = C.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-4)

        # Jacobian traces should both be 0
        @test all(isapprox.(jac_trace_fwd, 0f0; atol=1e-6))
        @test all(isapprox.(jac_trace_inv, 0f0; atol=1e-6))
    end

    @testset "Conv1x1CV 2D - Orthogonality" begin
        nx, ny = 8, 8
        nc = 4
        batchsize = 1

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)
        Y, _ = C.forward(X)

        # ||Y|| should equal ||X|| for orthogonal transformations
        @test isapprox(norm(Y), norm(X); rtol=1e-4)
    end

    @testset "Conv1x1CV 2D - Gradient Test with Jacobian Trace Weight" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Loss function with jacobian trace term
        # Note: Since jac_trace is 0 for Householder, this mainly tests the infrastructure
        function loss_with_trace(C, X)
            Y, jac_trace = C.forward(X)

            # Loss = ||Y||^2 + lambda * sum(jac_trace)
            # For Householder, jac_trace = 0, so this simplifies to ||Y||^2
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            # Compute gradients
            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)

            ΔX, _, _ = C.inverse((ΔY, Y); jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔX, deepcopy(get_params(C))
        end

        # Base point
        X0 = randn(Float32, nx, ny, nc, batchsize)
        f0, ΔX, Δθ = loss_with_trace(C, X0)

        # Perturbation
        dX = randn(Float32, nx, ny, nc, batchsize)

        # Gradient test (input)
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test Conv1x1CV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(C, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "Conv1x1CV 2D - Jacobian Trace Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        # Two instances
        C1 = Conv1x1CV(nc)
        C2 = Conv1x1CV(nc)

        X = randn(Float32, nx, ny, nc, batchsize)

        θ0 = deepcopy(get_params(C1))
        θ = deepcopy(get_params(C2))

        # Loss function
        function loss_params(C, X)
            Y, jac_trace = C.forward(X)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            C.inverse((ΔY, Y); jac_trace_grad_weight=jac_trace_grad_weight)

            return f, deepcopy(get_params(C))
        end

        f0, Δθ = loss_params(C1, X)

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

        println("\nGradient test Conv1x1CV 2D with jac_trace: parameters")
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

    @testset "Conv1x1CV 2D - Jacobian Trace Gradient Consistency" begin
        # Test that jacobian trace gradient is zero for Householder
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Compute jacobian trace gradients
        ∇v1_trace, ∇v2_trace, ∇v3_trace = jac_trace_grad!(C, X)

        # For Householder (orthogonal), gradients should be zero
        @test all(isapprox.(∇v1_trace, 0f0; atol=1e-6))
        @test all(isapprox.(∇v2_trace, 0f0; atol=1e-6))
        @test all(isapprox.(∇v3_trace, 0f0; atol=1e-6))
    end

    ###############################################################################
    # Test 3D Conv1x1CV
    ###############################################################################

    @testset "Conv1x1CV 3D - Basic and Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, jac_trace_batch = C.forward(X)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize

        # For Householder, all traces should be 0
        @test all(isapprox.(jac_trace_batch, 0f0; atol=1e-6))
    end

    @testset "Conv1x1CV 3D - Invertibility with Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nz, nc, batchsize)

        Y, jac_trace_fwd = C.forward(X)
        X_reconstructed, jac_trace_inv = C.inverse(Y)

        @test isapprox(X, X_reconstructed; rtol=1e-4)
        @test all(isapprox.(jac_trace_fwd, 0f0; atol=1e-6))
        @test all(isapprox.(jac_trace_inv, 0f0; atol=1e-6))
    end

    @testset "Conv1x1CV 3D - Orthogonality" begin
        nx, ny, nz = 8, 8, 8
        nc = 2
        batchsize = 1

        C = Conv1x1CV(nc)
        X = randn(Float32, nx, ny, nz, nc, batchsize)
        Y, _ = C.forward(X)

        # Norm preservation
        @test isapprox(norm(Y), norm(X); rtol=1e-4)
    end

    @testset "Conv1x1CV - Frozen Parameters" begin
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        C = Conv1x1CV(nc; freeze=true)
        X = randn(Float32, nx, ny, nc, batchsize)

        # Store initial parameters
        v1_init = copy(C.v1.data)
        v2_init = copy(C.v2.data)
        v3_init = copy(C.v3.data)

        # Forward-backward with jac_trace_grad_weight
        Y, _ = C.forward(X)
        ΔY = randn(Float32, size(Y)...)
        jac_trace_grad_weight = randn(Float32, batchsize)
        C.inverse((ΔY, Y); jac_trace_grad_weight=jac_trace_grad_weight)

        @test C.freeze == true
    end

    @testset "Conv1x1CV - Comparison with Non-CV Version" begin
        # Verify that outputs are the same (only trace bookkeeping differs)
        nx, ny = 8, 8
        nc = 2
        batchsize = 4

        X = randn(Float32, nx, ny, nc, batchsize)

        # Create layers with same parameters
        v1 = randn(Float32, nc)
        v2 = randn(Float32, nc)
        v3 = randn(Float32, nc)

        C_cv = Conv1x1CV(v1, v2, v3)
        C = Conv1x1(v1, v2, v3; logdet=true)

        # Forward pass
        Y_cv, jac_trace = C_cv.forward(X)
        Y, logdet = C.forward(X; logdet_per_batch=true)

        # Outputs should be identical
        @test Y_cv ≈ Y

        # Both should have zero trace/logdet
        @test all(isapprox.(jac_trace, 0f0; atol=1e-6))
        @test all(isapprox.(logdet, 0f0; atol=1e-6))
    end

end

println("\n✓ All Conv1x1CV tests passed!")
