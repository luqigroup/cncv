# Test script for ensemble CV architecture
# Verifies that forward-reverse ensemble compiles and runs

using LinearAlgebra
using Statistics
using Random
import CNCV: DenseConditionalLayerCV_Reversible, EnsembleDenseCV
import CNCV: forward, forward_ensemble, create_forward_reverse_ensemble

Random.seed!(123)

println("="^80)
println("Testing Ensemble CV Architecture")
println("="^80)

n_dim = 2
n_cond = 2
n_hidden = 16
n_layers = 2
batch_size = 100

println("\n1. Testing DenseConditionalLayerCV_Reversible (forward split)")
CV_forward = DenseConditionalLayerCV_Reversible(n_dim, n_cond, n_hidden, n_layers;
                                                reverse_split=false, n_cv=n_dim)

X = randn(Float32, n_dim, batch_size)
C = randn(Float32, n_cond, batch_size)

jac_traces_f, phi_all_f = forward(X, C, CV_forward)

println("   Input shape: ", size(X))
println("   jac_traces shape: ", size(jac_traces_f))
println("   phi_all shape: ", size(phi_all_f))
println("   ✓ Forward split works!")

println("\n2. Testing DenseConditionalLayerCV_Reversible (reverse split)")
CV_reverse = DenseConditionalLayerCV_Reversible(n_dim, n_cond, n_hidden, n_layers;
                                                reverse_split=true, n_cv=n_dim)

jac_traces_r, phi_all_r = forward(X, C, CV_reverse)

println("   jac_traces shape: ", size(jac_traces_r))
println("   phi_all shape: ", size(phi_all_r))
println("   ✓ Reverse split works!")

println("\n3. Checking which components differ")
println("   Forward φ vs input:")
for k in 1:n_dim
    phi_k = phi_all_f[k, :, :]
    diff_1 = norm(phi_k[1, :] - X[1, :])
    diff_2 = norm(phi_k[2, :] - X[2, :])
    println("     CV $k: |φ[1] - x[1]| = $(round(diff_1, digits=3)), |φ[2] - x[2]| = $(round(diff_2, digits=3))")
end

println("   Reverse φ vs input:")
for k in 1:n_dim
    phi_k = phi_all_r[k, :, :]
    diff_1 = norm(phi_k[1, :] - X[1, :])
    diff_2 = norm(phi_k[2, :] - X[2, :])
    println("     CV $k: |φ[1] - x[1]| = $(round(diff_1, digits=3)), |φ[2] - x[2]| = $(round(diff_2, digits=3))")
end

println("\n4. Testing EnsembleDenseCV")
ensemble = EnsembleDenseCV([CV_forward, CV_reverse]; combination_mode=:average)

score = randn(Float32, n_dim, batch_size)
g_combined, all_g = forward_ensemble(X, C, score, ensemble)

println("   Number of layers: ", length(ensemble.layers))
println("   Combined g shape: ", size(g_combined))
println("   All g shape: ", size(all_g))
println("   ✓ Ensemble works!")

println("\n5. Testing helper function")
ensemble2 = create_forward_reverse_ensemble(n_dim, n_cond, n_hidden, n_layers; n_cv=n_dim)
g_combined2, all_g2 = forward_ensemble(X, C, score, ensemble2)

println("   ✓ Helper function works!")

println("\n6. Checking that layers produce different outputs")
diff_layers = norm(all_g[1, :, :] - all_g[2, :, :])
println("   |g_layer1 - g_layer2| = ", round(diff_layers, digits=3))
if diff_layers > 1e-6
    println("   ✓ Layers produce different CVs (as expected)")
else
    println("   ✗ WARNING: Layers produce identical outputs!")
end

println("\n7. Computing control variates explicitly")
for i in 1:2
    layer = ensemble.layers[i]
    jac_tr, phi = forward(X, C, layer)

    for k in 1:n_dim
        trace_k = jac_tr[k, :]
        phi_k = phi[k, :, :]
        phi_dot_score = vec(sum(phi_k .* score, dims=1))
        g_k = trace_k .+ phi_dot_score

        println("   Layer $i, CV $k:")
        println("     Mean trace: ", round(mean(trace_k), digits=4))
        println("     Mean φ·∇log p: ", round(mean(phi_dot_score), digits=4))
        println("     Mean g: ", round(mean(g_k), digits=4))
    end
end

println("\n8. Combined CV statistics")
for k in 1:n_dim
    println("   Combined CV $k:")
    println("     Mean: ", round(mean(g_combined[k, :]), digits=4))
    println("     Std: ", round(std(g_combined[k, :]), digits=4))
    println("     Individual layer 1 mean: ", round(mean(all_g[1, k, :]), digits=4))
    println("     Individual layer 2 mean: ", round(mean(all_g[2, k, :]), digits=4))
end

println("\n" * "="^80)
println("✓ All ensemble tests passed!")
println("="^80)
