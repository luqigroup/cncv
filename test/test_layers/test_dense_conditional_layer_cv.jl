# Test for DenseConditionalLayerCV
# Date: December 2025

using Test
using Random
using LinearAlgebra
import CNCV: DenseConditionalLayerCV, get_params, clear_grad!, forward, inverse, backward

@testset "DenseConditionalLayerCV Tests" begin

    Random.seed!(123)

    # Test dimensions
    n_in = 4
    n_cond = 2
    n_hidden = 16
    n_layers = 2
    batch_size = 8

    # Create layer
    L = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh)

    @testset "Layer construction" begin
        @test L.split_idx == 2
        params = get_params(L)
        @test length(params) > 0
    end

    @testset "Forward pass" begin
        X = randn(Float32, n_in, batch_size)
        C = randn(Float32, n_cond, batch_size)

        Y, jac_trace = forward(X, C, L)

        @test size(Y) == size(X)
        @test length(jac_trace) == batch_size
        @test all(jac_trace .>= 1.0f0)  # Should be at least 1 from identity part
    end

    @testset "Inverse correctness" begin
        X = randn(Float32, n_in, batch_size)
        C = randn(Float32, n_cond, batch_size)

        Y, jac_trace_fwd = forward(X, C, L)
        X_rec, jac_trace_inv = inverse(Y, C, L)

        @test isapprox(X, X_rec, rtol=1e-5)
        @test isapprox(jac_trace_fwd, jac_trace_inv, rtol=1e-5)
    end

    @testset "Jacobian trace - explicit verification" begin
        # Use small dimensions for explicit Jacobian computation
        n_in_small = 4
        n_cond_small = 2
        batch_small = 2

        L_small = DenseConditionalLayerCV(n_in_small, n_cond_small, 8, 2; activation=tanh)

        X_small = randn(Float32, n_in_small, batch_small)
        C_small = randn(Float32, n_cond_small, batch_small)

        Y_small, jac_trace_computed = forward(X_small, C_small, L_small)

        # Compute explicit Jacobian for first sample
        ε = 1f-4
        n_total = n_in_small
        J = zeros(Float32, n_total, n_total)

        X_vec = X_small[:, 1]
        C_vec = C_small[:, 1:1]  # Keep as matrix

        for i = 1:n_total
            X_perturbed = copy(X_vec)
            X_perturbed[i] += ε

            Y_base, _ = forward(reshape(X_vec, n_in_small, 1), C_vec, L_small)
            Y_pert, _ = forward(reshape(X_perturbed, n_in_small, 1), C_vec, L_small)

            J[:, i] = (Y_pert[:, 1] - Y_base[:, 1]) / ε
        end

        jac_trace_explicit = sum(diag(J))

        println("  Computed jac_trace: ", jac_trace_computed[1])
        println("  Explicit jac_trace: ", jac_trace_explicit)
        println("  Difference: ", abs(jac_trace_computed[1] - jac_trace_explicit))

        @test isapprox(jac_trace_computed[1], jac_trace_explicit, rtol=1e-2)
    end

    @testset "Backward pass gradients" begin
        X = randn(Float32, n_in, batch_size)
        C = randn(Float32, n_cond, batch_size)

        Y, jac_trace = forward(X, C, L)

        # Random upstream gradients
        ΔY = randn(Float32, n_in, batch_size)
        jac_grad = randn(Float32, batch_size)

        # Clear gradients
        clear_grad!(L)

        # Backward pass
        ΔX, X_rec, ΔC = backward(ΔY, Y, C, L; jac_trace_grad_weight=jac_grad)

        # Check that gradients were stored
        params = get_params(L)
        n_with_grad = sum([p.grad !== nothing for p in params])
        @test n_with_grad == length(params)

        # Check gradient magnitudes are reasonable
        max_grad = maximum([maximum(abs.(p.grad)) for p in params])
        @test max_grad < 100.0f0  # Shouldn't explode
        @test max_grad > 0.0f0    # Should be non-zero
    end

    @testset "Zero initialization verification" begin
        # Output layer should start at zero
        L_new = DenseConditionalLayerCV(4, 2, 16, 2; activation=tanh)

        X = randn(Float32, 4, 4)
        C = randn(Float32, 2, 4)

        Y, jac_trace = forward(X, C, L_new)

        # With zero output weights, Y should be close to X
        # (only modified by the identity part of coupling)
        # Actually, half passes through unchanged, half gets modified
        # So check that the passed-through half is exactly X
        @test isapprox(Y[3:4, :], X[3:4, :], rtol=1e-6)

        # The other half should be small (near zero from zero-init output layer)
        # Actually it goes through S*X1 + T where S comes from sigmoid of zero
        # sigmoid(0) = 0.5, so Y1 ≈ 0.5*X1
        @test maximum(abs.(Y[1:2, :] - 0.5f0 * X[1:2, :])) < 0.1f0
    end

    @testset "Gradient check with finite differences" begin
        # Small test
        L_grad = DenseConditionalLayerCV(2, 2, 8, 1; activation=tanh)

        X = randn(Float32, 2, 2)
        C = randn(Float32, 2, 2)

        # Define a simple loss
        function test_loss(L_test, X_test, C_test)
            Y, jac = forward(X_test, C_test, L_test)
            return sum(Y .^ 2) + sum(jac .^ 2)
        end

        # Compute loss
        loss0 = test_loss(L_grad, X, C)

        # Compute gradient via backward
        Y, jac_trace = forward(X, C, L_grad)
        ΔY = 2 .* Y
        jac_grad = 2 .* jac_trace

        clear_grad!(L_grad)
        backward(ΔY, Y, C, L_grad; jac_trace_grad_weight=jac_grad)

        # Check gradient on first parameter using finite differences
        params = get_params(L_grad)
        p1 = params[1]
        grad_computed = copy(p1.grad)

        # Finite difference
        ε = 1f-4
        grad_fd = zeros(Float32, size(p1.data))

        for i in 1:min(5, length(p1.data))  # Just check a few elements
            p1.data[i] += ε
            loss_plus = test_loss(L_grad, X, C)
            p1.data[i] -= 2*ε
            loss_minus = test_loss(L_grad, X, C)
            p1.data[i] += ε  # Restore

            grad_fd[i] = (loss_plus - loss_minus) / (2*ε)
        end

        # Compare (only first few elements)
        println("\n  Gradient check (first 5 elements):")
        for i in 1:min(5, length(p1.data))
            println("    [$i] Computed: $(grad_computed[i]), FD: $(grad_fd[i]), Diff: $(abs(grad_computed[i] - grad_fd[i]))")
        end

        # Relative error should be small
        rel_err = maximum(abs.(grad_computed[1:5] .- grad_fd[1:5]) ./ (abs.(grad_fd[1:5]) .+ 1f-6))
        @test rel_err < 0.1  # 10% relative error tolerance
    end

end

println("\n✓ All DenseConditionalLayerCV tests passed!")
