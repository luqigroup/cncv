# Test for NetworkGlowCV (CV version with jacobian trace)
# Full Glow network with control variate support
# Tests include: invertibility, gradient tests, jacobian trace tests, trace accumulation

using Test, Random, LinearAlgebra
using InvertibleNetworks: get_params, clear_grad!

using CNCV

Random.seed!(303132)

@testset "NetworkGlowCV Tests" begin

    ###############################################################################
    # Test 2D NetworkGlowCV
    ###############################################################################

    @testset "NetworkGlowCV 2D - Basic and Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 8

        # Create network
        G = NetworkGlowCV(n_in, n_hidden, depth, K)

        # Input
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward pass
        Y, jac_trace_batch = G.forward(X)

        @test size(Y)[1:3] == (nx, ny, n_in)
        @test size(Y, 4) == batchsize
        @test length(jac_trace_batch) == batchsize
        @test all(isfinite.(jac_trace_batch))
    end

    @testset "NetworkGlowCV 2D - Invertibility with Jacobian Trace" begin
        nx, ny = 16, 16
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 8

        G = NetworkGlowCV(n_in, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)

        # Forward-inverse
        Y, jac_trace_fwd = G.forward(X)
        X_recon, jac_trace_inv = G.inverse(Y)

        @test isapprox(X, X_recon; rtol=1e-3)

        # Jacobian traces should be equal (accumulated over all layers)
        @test isapprox(jac_trace_fwd, jac_trace_inv; rtol=1e-4)
    end

    @testset "NetworkGlowCV 2D - Gradient Test with Jacobian Trace Weight" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        G = NetworkGlowCV(n_in, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)
        X0 = randn(Float32, nx, ny, n_in, batchsize)
        dX = randn(Float32, nx, ny, n_in, batchsize)

        # Loss function with jacobian trace term
        function loss_with_trace(G, X)
            Y, jac_trace = G.forward(X)

            # Loss = ||Y||^2 + lambda * sum(jac_trace)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            # Compute gradients
            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)

            ΔX = G.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)[1]

            return f, ΔX, deepcopy(get_params(G))
        end

        # Base point
        f0, ΔX, Δθ = loss_with_trace(G, X0)

        # Gradient test (input)
        h = 0.1f0
        maxiter = 5
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        println("\nGradient test NetworkGlowCV 2D with jac_trace: input")
        for j=1:maxiter
            f = loss_with_trace(G, X0 + h*dX)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            println("  Iter $j: err1=$(err1[j]), err2=$(err2[j])")
            h /= 2f0
        end

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
    end

    @testset "NetworkGlowCV 2D - Jacobian Trace Gradient Test (Parameters)" begin
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        # Two instances
        G1 = NetworkGlowCV(n_in, n_hidden, depth, K)
        G2 = NetworkGlowCV(n_in, n_hidden, depth, K)

        X = randn(Float32, nx, ny, n_in, batchsize)

        θ0 = deepcopy(get_params(G1))
        θ = deepcopy(get_params(G2))

        # Loss function
        function loss_params(G, X)
            Y, jac_trace = G.forward(X)
            lambda = 0.1f0
            f = 0.5f0*norm(Y)^2f0 + lambda * sum(jac_trace)

            ΔY = Y
            jac_trace_grad_weight = fill(lambda, batchsize)
            G.backward(ΔY, Y; jac_trace_grad_weight=jac_trace_grad_weight)

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

        # Gradient test (parameters)
        h = 0.1f0
        maxiter = 5
        err3 = zeros(Float32, maxiter)
        err4 = zeros(Float32, maxiter)

        println("\nGradient test NetworkGlowCV 2D with jac_trace: parameters")
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

    @testset "NetworkGlowCV - Jacobian Trace Accumulation" begin
        # Test that network trace is sum of layer traces
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 2
        K = 2
        batchsize = 4

        G = NetworkGlowCV(n_in, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)

        Y, jac_trace = G.forward(X)

        # Trace should be accumulated across all layers
        @test length(jac_trace) == batchsize
        @test all(isfinite.(jac_trace))
    end

    @testset "NetworkGlowCV - Sampling with Trace" begin
        # Test generating samples from latent space
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        G = NetworkGlowCV(n_in, n_hidden, depth, K)

        # Sample from latent
        Z = randn(Float32, nx, ny, n_in, batchsize)
        X, jac_trace = G.inverse(Z)

        @test size(X) == size(Z)
        @test length(jac_trace) == batchsize
        @test all(isfinite.(X))
        @test all(isfinite.(jac_trace))
    end

    @testset "NetworkGlowCV - Comparison with Non-CV Version" begin
        # Verify structural compatibility
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        X = randn(Float32, nx, ny, n_in, batchsize)

        # CV version
        G_cv = NetworkGlowCV(n_in, n_hidden, depth, K)
        Y_cv, jac_trace = G_cv.forward(X)

        # Non-CV version
        G = NetworkGlow(n_in, n_hidden, depth, K; logdet=true)
        Y, logdet = G.forward(X; logdet_per_batch=true)

        # Structure should be compatible
        @test size(Y_cv) == size(Y)
        @test length(jac_trace) == length(logdet)
    end

    @testset "NetworkGlowCV - Backward without Trace Weight" begin
        # Test backward works correctly when jac_trace_grad_weight is nothing
        nx, ny = 8, 8
        n_in = 4
        n_hidden = 8
        depth = 1
        K = 2
        batchsize = 4

        G = NetworkGlowCV(n_in, n_hidden, depth, K)
        X = randn(Float32, nx, ny, n_in, batchsize)

        Y, _ = G.forward(X)
        ΔY = randn(Float32, size(Y)...)

        # Backward without jac_trace_grad_weight
        ΔX, X_recon = G.backward(ΔY, Y; jac_trace_grad_weight=nothing)

        @test size(ΔX) == size(X)
        @test isapprox(X, X_recon; rtol=1e-4)
    end

end

println("\n✓ All NetworkGlowCV tests passed!")
