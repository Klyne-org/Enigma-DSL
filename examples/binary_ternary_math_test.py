#!/usr/bin/env python3
"""Smoke-test binary + ternary float math ops."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


def run_binary(op_name: str, msl_token: str, ref, A, B):
    fn = getattr(enigma, op_name)

    @enigma.kernel
    def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
        tid = enigma.thread_position_in_grid
        C[tid] = fn(A[tid], B[tid])

    compiled = enigma.compile(k)
    assert msl_token in compiled.metal_source, f"{op_name}: '{msl_token}' missing"
    out = np.frombuffer(
        runtime.execute(compiled, [A, B], N*4, grid=(N,1,1), threads=(256,1,1)),
        dtype=np.float32,
    ).copy()
    np.testing.assert_allclose(out, ref(A, B), rtol=1e-3, atol=1e-3)
    print(f"OK  {op_name}")


A = np.random.randn(N).astype(np.float32)
B = np.random.randn(N).astype(np.float32)
Apos = np.abs(A) + 1e-2

run_binary("fmin", "fmin", np.fmin, A, B)
run_binary("fmax", "fmax", np.fmax, A, B)
run_binary("pow",  "pow",  np.power, Apos, B * 0.5)
run_binary("fmod", "fmod", np.fmod, A, np.abs(B) + 1e-2)
run_binary("atan2","atan2", np.arctan2, A, B)
run_binary("copysign","copysign", np.copysign, A, B)


# Ternary: fma
@enigma.kernel
def fma_k(A: enigma.f32, B: enigma.f32, C: enigma.f32, D: enigma.f32):
    tid = enigma.thread_position_in_grid
    D[tid] = enigma.fma(A[tid], B[tid], C[tid])

compiled = enigma.compile(fma_k)
assert "fma" in compiled.metal_source
C = np.random.randn(N).astype(np.float32)
out = np.frombuffer(
    runtime.execute(compiled, [A, B, C], N*4, grid=(N,1,1), threads=(256,1,1)),
    dtype=np.float32,
).copy()
np.testing.assert_allclose(out, A*B + C, rtol=1e-3, atol=1e-3)
print("OK  fma")


# clamp
@enigma.kernel
def clamp_k(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.clamp(A[tid], A[tid]*0, A[tid]*0 + 1)  # clamp to [0,1] via expressions

# simpler: use mix with t=0.5 constants would need constant plumbing; skip for now.

print("\nBinary + ternary math ops passed.")
