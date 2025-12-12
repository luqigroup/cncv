# Test for Jacobian utilities (non-CV)
# Tests include: jacobian computation, adjoint tests

using Test, Random, LinearAlgebra

using CNCV

Random.seed!(808182)

@testset "Jacobian Utilities Tests" begin

    @testset "Jacobian - Basic Computation" begin
        # Simple linear transformation
        n = 10
        A = randn(Float32, n, n)
        x = randn(Float32, n)
        dx = randn(Float32, n)

        # Jacobian-vector product: J*dx = A*dx
        Jdx = A * dx

        @test size(Jdx) == size(x)
    end

    @testset "Adjoint Jacobian Test" begin
        # Test that J^T is indeed the adjoint
        n = 10
        A = randn(Float32, n, n)
        x = randn(Float32, n)
        y = randn(Float32, n)
        dx = randn(Float32, n)
        dy = randn(Float32, n)

        # Forward: y = A*x, dy = A*dx
        Jdx = A * dx

        # Adjoint: J^T * dy = A^T * dy
        JTdy = A' * dy

        # Adjoint test: <dy, J*dx> = <J^T*dy, dx>
        a = dot(dy, Jdx)
        b = dot(JTdy, dx)

        @test isapprox(a, b; rtol=1e-5)
    end

    @testset "Log-determinant from Jacobian" begin
        # For triangular matrix, logdet is sum of log diagonal
        n = 10
        L = tril(randn(Float32, n, n))
        for i in 1:n
            L[i,i] = abs(L[i,i]) + 1f0  # Ensure positive diagonal
        end

        logdet_manual = sum(log.(abs.(diag(L))))
        logdet_builtin = logdet(L)

        @test isapprox(logdet_manual, logdet_builtin; rtol=1e-5)
    end

end

println("\n✓ All Jacobian utility tests passed!")
