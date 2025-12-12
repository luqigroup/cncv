# Test for activation functions (non-CV)
# Tests include: forward/backward, gradient tests

using Test, Random, LinearAlgebra

using CNCV

Random.seed!(606162)

@testset "Activation Functions Tests" begin

    @testset "SigmoidLayer - Forward/Backward" begin
        n = 100
        batchsize = 8

        act = SigmoidLayer()
        X = randn(Float32, n, batchsize)

        # Forward
        Y = act.forward(X)
        @test size(Y) == size(X)
        @test all(Y .>= 0f0)  # Sigmoid output is positive
        @test all(Y .<= 1f0)  # Sigmoid output is <= 1

        # Backward
        ΔY = randn(Float32, n, batchsize)
        ΔX = act.backward(ΔY, X, Y)
        @test size(ΔX) == size(X)
    end

    @testset "SigmoidLayer - Gradient Test" begin
        n = 10
        batchsize = 4

        act = SigmoidLayer()
        X0 = randn(Float32, n, batchsize)
        dX = randn(Float32, n, batchsize)

        function loss(act, X)
            Y = act.forward(X)
            f = 0.5f0 * norm(Y)^2f0
            ΔY = Y
            ΔX = act.backward(ΔY, X, Y)
            return f, ΔX
        end

        f0, ΔX = loss(act, X0)
        h = 0.1f0
        f1 = loss(act, X0 + h*dX)[1]

        err1 = abs(f1 - f0)
        err2 = abs(f1 - f0 - h*dot(dX, ΔX))

        h /= 2f0
        f2 = loss(act, X0 + h*dX)[1]
        err2_half = abs(f2 - f0 - h*dot(dX, ΔX))

        @test err2_half < err2  # Second-order convergence
    end

    @testset "RELUlayer - Forward/Backward" begin
        n = 100
        batchsize = 8

        act = RELUlayer()
        X = randn(Float32, n, batchsize)

        Y = act.forward(X)
        @test size(Y) == size(X)
        @test all(Y .>= 0f0)  # ReLU output is non-negative

        ΔY = randn(Float32, n, batchsize)
        ΔX = act.backward(ΔY, X, Y)
        @test size(ΔX) == size(X)
    end

    @testset "LeakyRELU - Forward/Backward" begin
        n = 100
        batchsize = 8
        slope = 0.01f0

        act = LeakyRELU(slope)
        X = randn(Float32, n, batchsize)

        Y = act.forward(X)
        @test size(Y) == size(X)

        ΔY = randn(Float32, n, batchsize)
        ΔX = act.backward(ΔY, X, Y)
        @test size(ΔX) == size(X)
    end

end

println("\n✓ All activation function tests passed!")
