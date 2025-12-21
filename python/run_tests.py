#!/usr/bin/env python
"""Simple test runner to avoid pytest plugin issues."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

# Import test modules
print("=" * 80)
print("Running test_distributions.py")
print("=" * 80)
try:
    exec(open("tests/test_distributions.py").read())
    print("\n✓ test_distributions.py completed")
except Exception as e:
    print(f"\n✗ test_distributions.py failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Running test_ensemble.py")
print("=" * 80)
try:
    exec(open("tests/test_ensemble.py").read())
    print("\n✓ test_ensemble.py completed")
except Exception as e:
    print(f"\n✗ test_ensemble.py failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Running test_jacobian.py")
print("=" * 80)
try:
    exec(open("tests/test_jacobian.py").read())
    print("\n✓ test_jacobian.py completed")
except Exception as e:
    print(f"\n✗ test_jacobian.py failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Running test_coupling_layer.py")
print("=" * 80)
try:
    exec(open("tests/test_coupling_layer.py").read())
    print("\n✓ test_coupling_layer.py completed")
except Exception as e:
    print(f"\n✗ test_coupling_layer.py failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
