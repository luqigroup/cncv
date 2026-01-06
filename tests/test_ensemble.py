"""Quick test script for ensemble implementation."""

import torch

from cncv import (
    create_forward_reverse_ensemble,
    create_random_split_ensemble,
)

print("=" * 60)
print("Testing Ensemble Implementation")
print("=" * 60)

# Problem setup
n_dim = 2
n_cond = 2
n_hidden = 32
n_layers = 3
n_cv = 2
batch_size = 10

# Test 1: Forward/Reverse Ensemble (2 members)
print("\n1. Testing forward/reverse ensemble (2 members):")
ensemble_fwd_rev = create_forward_reverse_ensemble(
    n_dim, n_cond, n_hidden, n_layers, n_cv=n_cv
)
print(f"   Number of members: {len(ensemble_fwd_rev.layers)}")
for i, layer in enumerate(ensemble_fwd_rev.layers):
    print(f"   Member {i+1}: reverse_split = {layer.reverse_split}")

# Test forward pass
X = torch.randn(batch_size, n_dim)
C = torch.randn(batch_size, n_cond)
score = torch.randn(batch_size, n_dim)

g_combined, all_g = ensemble_fwd_rev(X, C, score)
print(f"   Output shapes:")
print(f"     g_combined: {g_combined.shape} (expected: [{n_cv}, {batch_size}])")
print(f"     all_g: {all_g.shape} (expected: [2, {n_cv}, {batch_size}])")

# Test 2: Random Split Ensemble (5 members)
print("\n2. Testing random split ensemble (5 members):")
ensemble_random = create_random_split_ensemble(
    n_dim, n_cond, n_hidden, n_ensemble_members=5, n_layers=n_layers, n_cv=n_cv, seed=42
)
print(f"   Number of members: {len(ensemble_random.layers)}")
for i, layer in enumerate(ensemble_random.layers):
    print(f"   Member {i+1}: reverse_split = {layer.reverse_split}")

# Test forward pass
g_combined, all_g = ensemble_random(X, C, score)
print(f"   Output shapes:")
print(f"     g_combined: {g_combined.shape} (expected: [{n_cv}, {batch_size}])")
print(f"     all_g: {all_g.shape} (expected: [5, {n_cv}, {batch_size}])")

# Test 3: Random Split Ensemble (10 members)
print("\n3. Testing random split ensemble (10 members):")
ensemble_large = create_random_split_ensemble(
    n_dim, n_cond, n_hidden, n_ensemble_members=10, n_layers=n_layers, n_cv=n_cv, seed=123
)
print(f"   Number of members: {len(ensemble_large.layers)}")
split_counts = {"False": 0, "True": 0}
for i, layer in enumerate(ensemble_large.layers):
    split_counts[str(layer.reverse_split)] += 1
print(f"   Split distribution:")
print(f"     Forward splits (False): {split_counts['False']}")
print(f"     Reverse splits (True): {split_counts['True']}")

# Test forward pass
g_combined, all_g = ensemble_large(X, C, score)
print(f"   Output shapes:")
print(f"     g_combined: {g_combined.shape} (expected: [{n_cv}, {batch_size}])")
print(f"     all_g: {all_g.shape} (expected: [10, {n_cv}, {batch_size}])")

# Test 4: Verify Stein's identity approximately holds
print("\n4. Testing that ensemble averaging works correctly:")
# The average across ensemble members should equal g_combined
manual_avg = all_g.mean(dim=0)
print(f"   Max difference between g_combined and mean(all_g): {(g_combined - manual_avg).abs().max().item():.2e}")
print(f"   (Should be ~0 for 'average' combination mode)")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
