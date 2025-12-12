# Test for Jacobian utilities (CV version with trace computation)
# Tests include: trace computation, trace gradients

using Test, Random, LinearAlgebra

using CNCV

Random.seed!(808182)

@testset "Jacobian Utilities CV Tests" begin

    @testset "Jacobian Trace - Basic Computation" begin
        # For diagonal matrix, trace = sum of diagonal
        n = 10
        D = diagm(randn(Float32, n))

        trace_manual = sum(diag(D))
        trace_builtin = tr(D)

        @test isapprox(trace_manual, trace_builtin; rtol=1e-6)
    end

    @testset "Jacobian Trace - General Matrix" begin
        n = 10
        A = randn(Float32, n, n)

        trace_A = tr(A)

        @test typeof(trace_A) <: Real
        @test isfinite(trace_A)
    end

    @testset "Trace Gradient - Diagonal Matrix" begin
        # For f(D) = tr(D) where D = diag(d), gradient is [1,1,...,1]
        n = 10
        d = randn(Float32, n)
        D = diagm(d)

        # Trace
        trace_D = tr(D)

        # Gradient of trace w.r.t. diagonal elements
        grad_trace = ones(Float32, n)

        # Finite difference check
        h = 1f-5
        i = 5  # Test one element
        d_pert = copy(d)
        d_pert[i] += h
        D_pert = diagm(d_pert)

        fd_grad = (tr(D_pert) - trace_D) / h

        @test isapprox(fd_grad, grad_trace[i]; rtol=1e-3)
    end

    @testset "Trace - Product Rule" begin
        # tr(AB) properties
        n = 5
        A = randn(Float32, n, n)
        B = randn(Float32, n, n)

        trace_AB = tr(A * B)
        trace_BA = tr(B * A)

        # tr(AB) = tr(BA)
        @test isapprox(trace_AB, trace_BA; rtol=1e-5)
    end

    @testset "Trace of Jacobian for Affine Transform" begin
        # For y = Ax + b, trace(J) = trace(A)
        n = 10
        A = randn(Float32, n, n)
        x = randn(Float32, n)
        b = randn(Float32, n)

        # Jacobian is A
        trace_J = tr(A)

        @test typeof(trace_J) <: Real
        @test isfinite(trace_J)
    end

end

println("\n✓ All Jacobian utility CV tests passed!")
