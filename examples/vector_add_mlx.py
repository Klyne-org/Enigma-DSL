#!/usr/bin/env python3
"""vector_add using mlx.core.array end-to-end — no numpy in user code.

Mirrors examples/vector_add.py but demonstrates the Triton/cuTile-style API
where the runtime consumes and produces framework tensors directly. On Apple
Silicon's unified memory the mlx buffer is handed straight to Metal — no copy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

import enigma


@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]


print("Compiling vector_add kernel...")
compiled = enigma.compile(vector_add)

N = 1024
A = mx.random.normal((N,)).astype(mx.float32)
B = mx.random.normal((N,)).astype(mx.float32)

print(f"Dispatching kernel with N={N} elements (mlx inputs/outputs)...")
runtime = enigma.MetalRuntime()
C = runtime.execute(
    compiled,
    inputs=[A, B],
    output_shapes=[(N,)],
    output_dtypes=[mx.float32],
    grid=(N, 1, 1),
    threads=(min(N, 256), 1, 1),
)

expected = A + B
max_err = mx.max(mx.abs(C - expected)).item()
assert max_err < 1e-5, f"max error {max_err} too large"
print(f"PASSED: vector_add({N} elements) via mlx")
print(f"  max |error| = {max_err:.2e}")
print(f"  output type: {type(C).__name__}, dtype: {C.dtype}, shape: {C.shape}")
