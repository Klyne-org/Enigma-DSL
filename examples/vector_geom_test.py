#!/usr/bin/env python3
"""Vector + geometry ops end-to-end test.

Two kernels:
 1. dot_k:    loads 3 floats per thread from each of Ax/Ay/Az, By/... into
              a float3 pair, computes dot, stores scalar.
 2. length_k: loads 3 floats into float3, computes length, stores.

Buffers are still scalar `device float*`. We assemble vecs from scalar loads.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


# ---- dot product of two float3 vectors per thread ----
@enigma.kernel
def dot_k(
    Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32,
    Bx: enigma.f32, By: enigma.f32, Bz: enigma.f32,
    Out: enigma.f32,
):
    tid = enigma.thread_position_in_grid
    a = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    b = enigma.make_float3(Bx[tid], By[tid], Bz[tid])
    Out[tid] = enigma.dot(a, b)


compiled = enigma.compile(dot_k)
msl = compiled.metal_source
print("--- dot_k MSL ---")
print(msl)
assert "dot(" in msl, msl

Ax = np.random.randn(N).astype(np.float32)
Ay = np.random.randn(N).astype(np.float32)
Az = np.random.randn(N).astype(np.float32)
Bx = np.random.randn(N).astype(np.float32)
By = np.random.randn(N).astype(np.float32)
Bz = np.random.randn(N).astype(np.float32)

raw = runtime.execute(
    compiled, [Ax, Ay, Az, Bx, By, Bz], N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy()
expected = Ax * Bx + Ay * By + Az * Bz
np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
print("OK  dot (float3)")


# ---- length of a float3 vector per thread ----
@enigma.kernel
def length_k(
    Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32, Out: enigma.f32,
):
    tid = enigma.thread_position_in_grid
    v = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    Out[tid] = enigma.length(v)


compiled = enigma.compile(length_k)
assert "length(" in compiled.metal_source, compiled.metal_source

raw = runtime.execute(
    compiled, [Ax, Ay, Az], N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy()
expected = np.sqrt(Ax*Ax + Ay*Ay + Az*Az)
np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
print("OK  length (float3)")


# ---- distance between two float3 points ----
@enigma.kernel
def distance_k(
    Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32,
    Bx: enigma.f32, By: enigma.f32, Bz: enigma.f32,
    Out: enigma.f32,
):
    tid = enigma.thread_position_in_grid
    a = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    b = enigma.make_float3(Bx[tid], By[tid], Bz[tid])
    Out[tid] = enigma.distance(a, b)


compiled = enigma.compile(distance_k)
assert "distance(" in compiled.metal_source
raw = runtime.execute(
    compiled, [Ax, Ay, Az, Bx, By, Bz], N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy()
expected = np.sqrt((Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2)
np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
print("OK  distance (float3)")


# ---- normalize + extract .x ----
@enigma.kernel
def normalize_x_k(
    Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32, Out: enigma.f32,
):
    tid = enigma.thread_position_in_grid
    v = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    n = enigma.normalize(v)
    Out[tid] = n.x


compiled = enigma.compile(normalize_x_k)
assert "normalize(" in compiled.metal_source
# ensure inputs are non-zero norm
A = np.random.randn(3, N).astype(np.float32)
A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-3)
Ax2, Ay2, Az2 = A[0].copy(), A[1].copy(), A[2].copy()
raw = runtime.execute(
    compiled, [Ax2, Ay2, Az2], N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy()
norm = np.sqrt(Ax2*Ax2 + Ay2*Ay2 + Az2*Az2)
expected = Ax2 / norm
np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)
print("OK  normalize + vec.x extraction")


print("\nAll vector + geometry tests passed.")
