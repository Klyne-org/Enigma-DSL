#!/usr/bin/env python3
"""GEMM end-to-end test: C = A * B + C (accumulate pattern).

K=4 inner dimension, 2D grid, float4 dot product for the inner product.
This demonstrates the standard GEMM accumulate: C += A @ B.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N_DIM = 64
K_DIM = 4

runtime = enigma.MetalRuntime()


@enigma.kernel
def gemm_k4(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    row = enigma.thread_position_in_grid_xyz("y")
    col = enigma.thread_position_in_grid_xyz("x")

    n = enigma.metal_cast(N_DIM, "uint")
    k = enigma.metal_cast(K_DIM, "uint")

    base_a = row * k
    base_b = col
    out_idx = row * n + col

    a0 = A[base_a]
    a1 = A[base_a + 1]
    a2 = A[base_a + 2]
    a3 = A[base_a + 3]

    b0 = B[base_b]
    b1 = B[base_b + n]
    b2 = B[base_b + n * 2]
    b3 = B[base_b + n * 3]

    avec = enigma.make_float4(a0, a1, a2, a3)
    bvec = enigma.make_float4(b0, b1, b2, b3)

    C[out_idx] = enigma.dot(avec, bvec) + C[out_idx]


# --- Test 1: simple accumulate ---
M = 32
compiled = enigma.compile(gemm_k4)
msl = compiled.metal_source
print("--- gemm_k4 MSL ---")
print(msl)
assert "dot(" in msl, msl

A = np.random.randn(M, K_DIM).astype(np.float32)
B = np.random.randn(K_DIM, N_DIM).astype(np.float32)
C_init = np.random.randn(M, N_DIM).astype(np.float32)

raw = runtime.execute(
    compiled,
    [A.ravel(), B.ravel()],
    M * N_DIM * 4,
    grid=(N_DIM, M, 1),
    threads=(min(N_DIM, 16), min(M, 16), 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy().reshape(M, N_DIM)
expected = A @ B
np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
print(f"OK  gemm {M}x{K_DIM} @ {K_DIM}x{N_DIM} (C starts zeroed)")


# --- Test 2: larger dimensions ---
M2 = 128
A2 = np.random.randn(M2, K_DIM).astype(np.float32)
B2 = np.random.randn(K_DIM, N_DIM).astype(np.float32)
raw2 = runtime.execute(
    compiled,
    [A2.ravel(), B2.ravel()],
    M2 * N_DIM * 4,
    grid=(N_DIM, M2, 1),
    threads=(min(N_DIM, 16), min(M2, 16), 1),
)
out2 = np.frombuffer(raw2, dtype=np.float32).copy().reshape(M2, N_DIM)
np.testing.assert_allclose(out2, A2 @ B2, rtol=1e-4, atol=1e-4)
print(f"OK  gemm {M2}x{K_DIM} @ {K_DIM}x{N_DIM}")

print("\nAll GEMM tests passed.")
