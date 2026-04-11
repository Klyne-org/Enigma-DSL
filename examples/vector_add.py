#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma

@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]

print("Compiling vector_add kernel...")
compiled = enigma.compile(vector_add, dump_ir=True)

print("Generated Metal source:")
print("-" * 60)
print(compiled.metal_source)
print("-" * 60)

N = 1024
A = np.random.randn(N).astype(np.float32)
B = np.random.randn(N).astype(np.float32)

print(f"\nDispatching kernel with N={N} elements...")
runtime = enigma.MetalRuntime()
result_bytes = runtime.execute(
    compiled, inputs=[A, B], output_size=N * 4,
    grid=(N, 1, 1), threads=(min(N, 256), 1, 1),
)

result = np.frombuffer(result_bytes, dtype=np.float32)
expected = A + B
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)
print(f"PASSED: vector_add({N} elements)")
print(f"  max |error| = {np.max(np.abs(result - expected)):.2e}")
