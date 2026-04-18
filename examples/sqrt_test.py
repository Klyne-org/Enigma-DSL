#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma


@enigma.kernel
def sqrt_test(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.sqrt(A[tid])

print("Compiling sqrt_test kernel...")
compiled = enigma.compile(sqrt_test)

print("\n--- Generated Metal source ---")
print(compiled.metal_source)

assert "sqrt" in compiled.metal_source, "expected 'sqrt' in MSL output"

N = 1024
A = np.abs(np.random.randn(N).astype(np.float32)) + 1e-3

runtime = enigma.MetalRuntime()
out_bytes = runtime.execute(
    compiled,
    [A],
    N * 4,
    grid=(N, 1, 1),
    threads=(min(N, 256), 1, 1),
)
result = np.frombuffer(out_bytes, dtype=np.float32)

expected = np.sqrt(A)
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

print(f"\nOK — sqrt matches np.sqrt on {N} elements")
print(f"  first 5 inputs : {A[:5]}")
print(f"  first 5 outputs: {result[:5]}")
print(f"  first 5 expect : {expected[:5]}")

