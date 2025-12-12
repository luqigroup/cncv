# Test for ConditionalLayerGlowCV (CV version with jacobian trace)
# Tests include: invertibility, gradient tests, jacobian trace tests, conditioning

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(101112)

@testset "ConditionalLayerGlowCV Tests" begin

    ###############################################################################
    # Test 2D ConditionalLayerGlowCV
    ###############################################################################

    @testset "ConditionalLayerGlowCV 2D - Basic and Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 8

        # Create layer
        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)

        # Input and conditioning
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        # Forward pass
        Y, jac_trace_batch = CL.forward(X, C)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "ConditionalLayerGlowCV 2D - Invertibility with Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 8

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        # Forward-inverse
        Y, jac_trace_fwd = CL.forward(X, C)
        X_recon, jac_trace_inv = CL.inverse(Y, C)

        @test isapprox(X, X_recon; rtol=1e-4)

        # Jacobian traces should be equal
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "ConditionalLayerGlowCV 2D - Conditioning Effect" begin
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C1 = randn(Float32, nx, ny, n_cond, batchsize)
        C2 = randn(Float32, nx, ny, n_cond, batchsize)

        Y1, _ = CL.forward(X, C1)
        Y2, _ = CL.forward(X, C2)

        # Different conditioning should produce different outputs
        @test !isapprox(Y1, Y2; rtol=0.1)
    end

    @testset "ConditionalLayerGlowCV 2D - Gradient Test with Jacobian Trace Weight" begin
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)

        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        # Loss function with jacobian trace term
        function loss_with_trace(CL, X, C)
            Y, jac_trace = CL.forward(X, C)

            # Loss = ||Y||^2 + lambda * sum(jac_trace)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            # Compute gradients
            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)

            ΔX, _, _ = CL.backward(ΔY, Y, C; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔX, deepcopy(get_params(CL))
        end

        # Base point
        f0, ΔX, Δθ = loss_with_trace(CL, X0, C)

        # Gradient test (input)
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test ConditionalLayerGlowCV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(CL, X0 + h*dX, C)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ConditionalLayerGlowCV 2D - Gradient Test (Conditioning)" begin
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)

        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)
        C0 = randn(Float32, nx, ny, n_cond, batchsize)
        dC = randn(Float32, nx, ny, n_cond, batchsize)

        # Loss function
        function loss_cond(CL, X, C)
            Y, jac_trace = CL.forward(X, C)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            _, _, ΔC = CL.backward(ΔY, Y, C; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, ΔC
        end

        # Initial loss
        f0, ΔC = loss_cond(CL, X, C0)

        # Gradient test
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test ConditionalLayerGlowCV 2D with jac_trace: conditioning")
        for j=1:maxiter
            f = loss_cond(CL, X, C0 + h*dC)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dC, ΔC))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ConditionalLayerGlowCV 2D - Jacobian Trace Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        # Two instances
        CL1 = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        CL2 = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)

        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        θ0 = deepcopy(get_params(CL1))
        θ = deepcopy(get_params(CL2))

        # Loss function
        function loss_params(CL, X, C)
            Y, jac_trace = CL.forward(X, C)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            CL.backward(ΔY, Y, C; jac_trace_grad_weight=jac_trace_grad_weight)

            return f, deepcopy(get_params(CL))
        end

        f0, Δθ = loss_params(CL1, X, C)

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

        println("\nGradient test ConditionalLayerGlowCV 2D with jac_trace: parameters")
        for j=1:maxiter
            θ_curr = θ0 + h*dθ
            set_params!(CL1, θ_curr)

            f = loss_params(CL1, X, C)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
            println("  Iter $j: err3=$(err3[j]), err4=$(err4[j])")
            h /= 2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "ConditionalLayerGlowCV 2D - Jacobian Trace Consistency" begin
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        Y, jac_trace = CL.forward(X, C)

        # Verify trace is per-batch
        @test length(jac_trace) == batchsize
        @test all(isfinite.(jac_trace))

        # For conditional glow: jac_trace = jac_trace_conv + jac_trace_coupling
        # Conv1x1 (Householder) has trace 0, so trace should come from coupling only
    end

    ###############################################################################
    # Test 3D ConditionalLayerGlowCV
    ###############################################################################

    @testset "ConditionalLayerGlowCV 3D - Basic and Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV3D(n_in, n_cond, n_hidden)

        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        C = randn(Float32, nx, ny, nz, n_cond, batchsize)

        Y, jac_trace_batch = CL.forward(X, C)

        @test size(Y) == size(X)
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "ConditionalLayerGlowCV 3D - Invertibility with Jacobian Trace" begin
        nx, ny, nz = 8, 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV3D(n_in, n_cond, n_hidden)
        X = randn(Float32, nx, ny, nz, n_in, batchsize)
        C = randn(Float32, nx, ny, nz, n_cond, batchsize)

        Y, jac_trace_fwd = CL.forward(X, C)
        X_recon, jac_trace_inv = CL.inverse(Y, C)

        @test isapprox(X, X_recon; rtol=1e-4)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-5)
    end

    @testset "ConditionalLayerGlowCV - Comparison with Non-CV Version" begin
        # Verify structural compatibility
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        # CV version
        CL_cv = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        Y_cv, jac_trace = CL_cv.forward(X, C)

        # Non-CV version
        CL = ConditionalLayerGlow(n_in, n_cond, n_hidden; logdet=true)
        Y, logdet = CL.forward(X, C; logdet_per_batch=true)

        # Structure should be compatible
        @test size(Y_cv) == size(Y)
        @test length(jac_trace) == length(logdet)
    end

    @testset "ConditionalLayerGlowCV - Backward without Trace Weight" begin
        # Test backward works correctly when jac_trace_grad_weight is nothing
        nx, ny = 8, 8
        n_in = 4
        n_cond = 2
        n_hidden = 8
        batchsize = 4

        CL = ConditionalLayerGlowCV(n_in, n_cond, n_hidden)
        X = randn(Float32, nx, ny, n_in, batchsize)
        C = randn(Float32, nx, ny, n_cond, batchsize)

        Y, _ = CL.forward(X, C)
        ΔY = randn(Float32, size(Y)...)

        # Backward without jac_trace_grad_weight
        ΔX, X_recon, ΔC = CL.backward(ΔY, Y, C; jac_trace_grad_weight=nothing)

        @test size(ΔX) == size(X)
        @test size(ΔC) == size(C)
        @test isapprox(X, X_recon; rtol=1e-5)
    end

end

println("\n✓ All ConditionalLayerGlowCV tests passed!")
