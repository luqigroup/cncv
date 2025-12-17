# Diagnostic tests for DenseConditionalLayerCV
# Verify that E[g_k(X,Y)|Y] = 0 (Stein's identity)

using Random, Statistics, LinearAlgebra
import CNCV: DenseConditionalLayerCV, forward, get_params, clear_grad!, backward
using Rosenbrock

Random.seed!(123)

println("="^80)
println("DIAGNOSTIC TEST: Control Variate E[g_k|Y] = 0 Verification")
println("="^80)

# Small network for testing
n_in = 2
n_cond = 2
n_hidden = 16
n_layers = 2
n_cv = 2

# Create network
CV = DenseConditionalLayerCV(n_in, n_cond, n_hidden, n_layers; activation=tanh, n_cv=n_cv)

# DON'T scale weights - test with default Flux initialization
# for p in get_params(CV)
#     p.data .*= 0.01f0
# end
println("   Testing with DEFAULT Flux initialization (no 0.01 scaling)")

println("\n1. Network initialization check:")
println("   Weights scaled by 0.01")
params = get_params(CV)
println("   Number of parameter arrays: ", length(params))
println("   Output layer weight norm: ", norm(params[end].data))
println("   Output layer bias norm: ", norm(params[end-1].data))

# Generate test data
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
sigma = 0.1f0
n_samples = 1000

X_test = rand(RB_dist, n_samples)
Y_test = X_test + sigma * randn(Float32, 2, n_samples)

println("\n2. Test data:")
println("   Number of samples: ", n_samples)
println("   X mean: ", mean(X_test, dims=2)')
println("   X std: ", std(X_test, dims=2)')
println("   Y mean: ", mean(Y_test, dims=2)')
println("   Y std: ", std(Y_test, dims=2)')

# Compute score function
function compute_score(X, Y, sigma, RB_dist)
    grad_likelihood = -(X .- Y) ./ (sigma^2)
    grad_prior = gradlogpdf(RB_dist, X)
    return grad_likelihood .+ grad_prior
end

score_term = compute_score(X_test, Y_test, sigma, RB_dist)

println("\n3. Score function check:")
println("   Score mean: ", mean(score_term, dims=2)')
println("   Score std: ", std(score_term, dims=2)')

# Forward through network
jac_traces, phi_all = forward(X_test, Y_test, CV)

println("\n4. Network outputs:")
println("   jac_traces shape: ", size(jac_traces))
println("   phi_all shape: ", size(phi_all))
println("   jac_traces mean (CV 1): ", mean(jac_traces[1, :]))
println("   jac_traces mean (CV 2): ", mean(jac_traces[2, :]))
println("   jac_traces std (CV 1): ", std(jac_traces[1, :]))
println("   jac_traces std (CV 2): ", std(jac_traces[2, :]))
println("   jac_traces range (CV 1): [", minimum(jac_traces[1, :]), ", ", maximum(jac_traces[1, :]), "]")
println("   jac_traces range (CV 2): [", minimum(jac_traces[2, :]), ", ", maximum(jac_traces[2, :]), "]")

# Check phi transformations
println("\n5. φ transformations:")
for k in 1:n_cv
    phi_k = phi_all[k, :, :]
    println("   φ_$k mean: ", mean(phi_k, dims=2)')
    println("   φ_$k std: ", std(phi_k, dims=2)')

    # Check if φ ≈ X (should be for small weights)
    diff_from_X = phi_k .- X_test
    println("   φ_$k - X mean: ", mean(diff_from_X, dims=2)')
    println("   φ_$k - X std: ", std(diff_from_X, dims=2)')
end

# Compute control variates
g_cv_all = zeros(Float32, n_cv, n_samples)
for k in 1:n_cv
    trace_k = jac_traces[k, :]
    phi_k = phi_all[k, :, :]

    # φ_k · ∇log p: inner product
    phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))

    # Control variate: g_k = div(φ_k) + φ_k·∇log p
    g_cv_all[k, :] = trace_k .+ phi_dot_score_k
end

println("\n6. Control variates:")
for k in 1:n_cv
    println("   g_$k mean: ", mean(g_cv_all[k, :]))
    println("   g_$k std: ", std(g_cv_all[k, :]))

    # By Stein's identity, E[g_k|Y] should be 0
    # Since we're averaging over samples with different Y, we expect small but non-zero mean
    # But it should be MUCH smaller than std
    ratio = abs(mean(g_cv_all[k, :])) / std(g_cv_all[k, :])
    println("   |E[g_$k]| / Std[g_$k]: ", ratio, " (should be << 1)")
end

# Compute correlations
h_x = X_test
println("\n7. Correlations with h(x) = x:")
for k in 1:n_cv
    corr = cor(h_x[k, :], g_cv_all[k, :])
    println("   Correlation(h_$k, g_$k): ", corr)
end

# Test gradient computation
println("\n8. Gradient check:")
# Simple loss: minimize variance of g_k
loss_val = mean(g_cv_all.^2)
println("   Initial loss (mean of g_k^2): ", loss_val)

# Compute gradients
Δg_cv = 2 .* g_cv_all ./ n_samples  # ∂L/∂g_k

# Backprop through g_k = trace_k + phi_dot_score_k
jac_trace_grad_weights = Δg_cv  # ∂L/∂trace_k

# ∂L/∂phi_k via chain rule
Δphi_all = zeros(Float32, n_cv, n_in, n_samples)
for k in 1:n_cv
    # ∂g_k/∂phi_k = score_term (from phi_dot_score_k)
    Δphi_all[k, :, :] = score_term .* reshape(Δg_cv[k, :], 1, n_samples)
end

# Backward through network
clear_grad!(CV)
ΔX, _, ΔY = backward(zeros(Float32, size(X_test)), X_test, Y_test, CV;
                     jac_trace_grad_weights=jac_trace_grad_weights,
                     phi_grad_weights=Δphi_all)

# Check if gradients are computed
has_grads = all([p.grad !== nothing for p in get_params(CV)])
println("   All parameters have gradients: ", has_grads)

if has_grads
    grad_norms = [norm(p.grad) for p in get_params(CV)]
    println("   Gradient norms: ", grad_norms)
    println("   Max gradient norm: ", maximum(grad_norms))
    println("   Min gradient norm: ", minimum(grad_norms))

    # Check if gradients are non-zero
    non_zero_grads = sum(grad_norms .> 1f-10)
    println("   Non-zero gradients: $non_zero_grads / $(length(grad_norms))")
end

# Test with a single update step
println("\n9. Test gradient step:")
old_params = [copy(p.data) for p in get_params(CV)]

# Try multiple learning rates
lrs = [1f-6, 1f-5, 1f-4, 1f-3]
for lr in lrs
    # Apply gradient step
    for (i, p) in enumerate(get_params(CV))
        p.data .= old_params[i] .- lr .* p.grad
    end

    # Recompute loss
    jac_traces_new, phi_all_new = forward(X_test, Y_test, CV)
    g_cv_new = zeros(Float32, n_cv, n_samples)
    for k in 1:n_cv
        trace_k = jac_traces_new[k, :]
        phi_k = phi_all_new[k, :, :]
        phi_dot_score_k = vec(sum(phi_k .* score_term, dims=1))
        g_cv_new[k, :] = trace_k .+ phi_dot_score_k
    end
    loss_new = mean(g_cv_new.^2)

    println("   lr=$lr: Loss change=$(loss_new - loss_val), Decreased=$(loss_new < loss_val)")
end

# Restore old parameters
for (i, p) in enumerate(get_params(CV))
    p.data .= old_params[i]
end

println("\n10. SUMMARY:")
println("   ✓ Network outputs reasonable values: ", all(isfinite.(jac_traces)) && all(isfinite.(phi_all)))
println("   ✓ Control variates computed: ", all(isfinite.(g_cv_all)))
println("   ✓ Gradients flow: ", has_grads && maximum(grad_norms) > 1f-10)
println("   ✓ Gradient step can decrease loss: Check lr sweep above")

# Check for potential issues
println("\n11. POTENTIAL ISSUES:")
issues_found = false

# Check if E[g_k] is too large
for k in 1:n_cv
    ratio = abs(mean(g_cv_all[k, :])) / std(g_cv_all[k, :])
    if ratio > 0.1
        println("   ⚠ WARNING: E[g_$k] / Std[g_$k] = $ratio > 0.1")
        println("      This suggests E[g_$k|Y] ≠ 0, violating Stein's identity!")
        issues_found = true
    end
end

# Check if φ is too different from X
for k in 1:n_cv
    phi_k = phi_all[k, :, :]
    max_diff = maximum(abs.(phi_k .- X_test))
    if max_diff > 1.0
        println("   ⚠ WARNING: max|φ_$k - X| = $max_diff > 1.0")
        println("      φ should be close to X with small initial weights")
        issues_found = true
    end
end

# Check gradient magnitude
if has_grads && maximum(grad_norms) < 1f-6
    println("   ⚠ WARNING: Maximum gradient norm = $(maximum(grad_norms)) < 1e-6")
    println("      Gradients might be too small for learning")
    issues_found = true
end

if !issues_found
    println("   ✓ No obvious issues detected")
end

println("\n" * "="^80)
