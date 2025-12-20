# Test to verify exact_score computes the correct gradient for the control variate formula
# Formula: g(x, y) = ∇x · ϕ(x, y) + ϕ(x, y) · ∇x log p(x|y)

using DrWatson
@quickactivate :CNCV

using Random
using Rosenbrock
using Statistics
Random.seed!(123)

# Load pretrained model
amortized_args = read_config("rosenbrock_amortized_sampling.json")
if amortized_args["epoch"] == -1
    amortized_args["epoch"] = amortized_args["max_epoch"]
end
loaded = load_experiment(amortized_args, ["G"])
G = loaded["G"]

println("\n=== Verifying Score Function for Control Variates ===\n")

# Create test data
sigma = 0.4f0
RB_dist = RosenbrockDistribution(0.0f0, 5.0f-1)
X_2d = rand(RB_dist, 5)
Y_2d = X_2d + sigma * randn(Float32, 2, 5)

# Reshape for network
X_test = reshape(X_2d, 1, 1, 2, 5)
Y_test = reshape(Y_2d, 1, 1, 2, 5)

# Compute learned score using exact_score
score_learned = exact_score(G, X_test, Y_test)
score_learned_2d = reshape(score_learned, 2, 5)

# Compute TRUE analytical score: ∇ log p(x|y) = ∇ log p(y|x) + ∇ log p(x)
grad_likelihood = -(X_2d .- Y_2d) ./ (sigma^2)  # ∇ log p(y|x)
grad_prior = gradlogpdf(RB_dist, X_2d)  # ∇ log p(x)
score_true = grad_likelihood .+ grad_prior

println("Learned score (component 1, first sample): ", score_learned_2d[1, 1])
println("True score    (component 1, first sample): ", score_true[1, 1])
println()
println("Learned score (component 2, first sample): ", score_learned_2d[2, 1])
println("True score    (component 2, first sample): ", score_true[2, 1])
println()

# Check correlation (should be high if signs are correct)
corr1 = cor(score_learned_2d[1, :], score_true[1, :])
corr2 = cor(score_learned_2d[2, :], score_true[2, :])

println("Correlation between learned and true score:")
println("  Component 1: ", round(corr1, digits=4))
println("  Component 2: ", round(corr2, digits=4))
println()

if corr1 > 0.85 && corr2 > 0.85
    println("✓ PASS: exact_score returns POSITIVE gradient ∇x log p(x|y)")
    println("✓ High correlation confirms signs are CORRECT")
    println("✓ Signs match for control variate formula:")
    println("  g(x, y) = ∇x · ϕ(x, y) + ϕ(x, y) · ∇x log p(x|y)")
else
    println("✗ WARNING: Low correlation - check implementation")
end

# Also check that signs match for all samples
signs_match_1 = all(sign.(score_learned_2d[1, :]) .== sign.(score_true[1, :]))
signs_match_2 = all(sign.(score_learned_2d[2, :]) .== sign.(score_true[2, :]))
println()
println("Sign agreement:")
println("  Component 1: ", signs_match_1 ? "✓ All signs match" : "✗ Signs differ")
println("  Component 2: ", signs_match_2 ? "✓ All signs match" : "✗ Signs differ")
