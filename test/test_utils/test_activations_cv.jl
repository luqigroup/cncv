# Test for activation functions (CV version)
# CV versions have same interface as non-CV, tests are identical

using Test, Random, LinearAlgebra

using CNCV

Random.seed!(606162)

@testset "Activation Functions CV Tests" begin

    # Note: CV versions of activations typically have same interface
    # The difference is in how they're used in layers with jac_trace_grad_weight

    @testset "SigmoidLayer CV - Forward/Backward" begin
        n = 100
        batchsize = 8

        act = SigmoidLayer()
        X = randn(Float32, n, batchsize)

        Y = act.forward(X)
        @test size(Y) == size(X)
        @test all(Y .>= 0f0)
        @test all(Y .<= 1f0)

        ΔY = randn(Float32, n, batchsize)
        ΔX = act.backward(ΔY, X, Y)
        @test size(ΔX) == size(X)
    end

    # Additional CV-specific tests can be added if activation functions
    # have CV-specific behavior beyond the layer level

end

println("\n✓ All activation function CV tests passed!")
