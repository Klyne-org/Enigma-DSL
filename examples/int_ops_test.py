#!/usr/bin/env python3
"""End-to-end numerical tests for integer / bit / simd / cast / predicate ops."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


def _run(kernel, inputs, out_nbytes, out_dtype):
    compiled = enigma.compile(kernel)
    raw = runtime.execute(compiled, inputs, out_nbytes,
                          grid=(N, 1, 1), threads=(256, 1, 1))
    return np.frombuffer(raw, dtype=out_dtype).copy(), compiled.metal_source


# ---------------- Unary int: popcount / clz / ctz / reverse_bits ----------------
@enigma.kernel
def popcount_k(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.popcount(A[tid])

A = np.random.randint(0, 2**31 - 1, size=N, dtype=np.uint32)
out, msl = _run(popcount_k, [A], N * 4, np.uint32)
expected = np.array([bin(int(x)).count("1") for x in A], dtype=np.uint32)
assert "popcount" in msl
np.testing.assert_array_equal(out, expected)
print("OK  popcount")


@enigma.kernel
def clz_k(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.clz(A[tid])

A = np.random.randint(1, 2**31 - 1, size=N, dtype=np.uint32)
out, msl = _run(clz_k, [A], N * 4, np.uint32)
expected = np.array([32 - int(x).bit_length() for x in A], dtype=np.uint32)
assert "clz" in msl
np.testing.assert_array_equal(out, expected)
print("OK  clz")


@enigma.kernel
def ctz_k(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.ctz(A[tid])

A = np.random.randint(1, 2**31 - 1, size=N, dtype=np.uint32)
out, msl = _run(ctz_k, [A], N * 4, np.uint32)
expected = np.array([(int(x) & -int(x)).bit_length() - 1 for x in A], dtype=np.uint32)
assert "ctz" in msl
np.testing.assert_array_equal(out, expected)
print("OK  ctz")


@enigma.kernel
def rev_k(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.reverse_bits(A[tid])

A = np.random.randint(0, 2**31 - 1, size=N, dtype=np.uint32)
out, msl = _run(rev_k, [A], N * 4, np.uint32)
expected = np.array([int(bin(int(x) & 0xFFFFFFFF)[2:].zfill(32)[::-1], 2) for x in A], dtype=np.uint32)
assert "reverse_bits" in msl
np.testing.assert_array_equal(out, expected)
print("OK  reverse_bits")


# ---------------- Binary int: imin / imax / add_sat / sub_sat ----------------
@enigma.kernel
def imin_k(A: enigma.u32, B: enigma.u32, C: enigma.u32):
    tid = enigma.thread_position_in_grid
    C[tid] = enigma.imin(A[tid], B[tid])

A = np.random.randint(0, 10000, size=N, dtype=np.uint32)
B = np.random.randint(0, 10000, size=N, dtype=np.uint32)
out, msl = _run(imin_k, [A, B], N * 4, np.uint32)
assert "min" in msl
np.testing.assert_array_equal(out, np.minimum(A, B))
print("OK  imin")


@enigma.kernel
def imax_k(A: enigma.u32, B: enigma.u32, C: enigma.u32):
    tid = enigma.thread_position_in_grid
    C[tid] = enigma.imax(A[tid], B[tid])

out, msl = _run(imax_k, [A, B], N * 4, np.uint32)
assert "max" in msl
np.testing.assert_array_equal(out, np.maximum(A, B))
print("OK  imax")


@enigma.kernel
def addsat_k(A: enigma.u32, B: enigma.u32, C: enigma.u32):
    tid = enigma.thread_position_in_grid
    C[tid] = enigma.add_sat(A[tid], B[tid])

INT_MAX = np.int64(2**31 - 1)
INT_MIN = np.int64(-(2**31))
A = np.random.randint(INT_MIN, INT_MAX, size=N, dtype=np.int32).astype(np.uint32)
B = np.random.randint(INT_MIN, INT_MAX, size=N, dtype=np.int32).astype(np.uint32)
out, msl = _run(addsat_k, [A, B], N * 4, np.uint32)
As, Bs = A.view(np.int32).astype(np.int64), B.view(np.int32).astype(np.int64)
expected = np.clip(As + Bs, INT_MIN, INT_MAX).astype(np.int32).view(np.uint32)
assert "addsat" in msl
np.testing.assert_array_equal(out, expected)
print("OK  add_sat")


@enigma.kernel
def subsat_k(A: enigma.u32, B: enigma.u32, C: enigma.u32):
    tid = enigma.thread_position_in_grid
    C[tid] = enigma.sub_sat(A[tid], B[tid])

out, msl = _run(subsat_k, [A, B], N * 4, np.uint32)
expected = np.clip(As - Bs, INT_MIN, INT_MAX).astype(np.int32).view(np.uint32)
assert "subsat" in msl
np.testing.assert_array_equal(out, expected)
print("OK  sub_sat")


# ---------------- Ternary int: iclamp ----------------
@enigma.kernel
def iclamp_k(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.iclamp(A[tid], A[tid]*0 + 100, A[tid]*0 + 500)

A = np.random.randint(0, 1000, size=N, dtype=np.uint32)
out, msl = _run(iclamp_k, [A], N * 4, np.uint32)
assert "clamp" in msl
np.testing.assert_array_equal(out, np.clip(A, 100, 500))
print("OK  iclamp")


# ---------------- Casts: metal_cast (u32 -> f32) ----------------
@enigma.kernel
def cast_k(A: enigma.u32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.metal_cast(A[tid], "f32")

A = np.random.randint(0, 10000, size=N, dtype=np.uint32)
out, msl = _run(cast_k, [A], N * 4, np.float32)
assert "metal_cast" in msl or "static_cast" in msl or "float" in msl
np.testing.assert_allclose(out, A.astype(np.float32), rtol=0, atol=0)
print("OK  metal_cast u32->f32")


# ---------------- SIMD reduction: simd_sum ----------------
# Each 32-lane SIMD group will get the same sum of the 32 lane values.
@enigma.kernel
def simdsum_k(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.simd_sum(A[tid])

A = np.random.randn(N).astype(np.float32)
out, msl = _run(simdsum_k, [A], N * 4, np.float32)
assert "simd_sum" in msl
# Apple SIMD group width = 32
expected = np.repeat(A.reshape(-1, 32).sum(axis=1), 32).astype(np.float32)
np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
print("OK  simd_sum")


# ---------------- Float predicates: isnan -> store as int ----------------
@enigma.kernel
def isnan_k(A: enigma.f32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.metal_cast(enigma.isnan(A[tid]), "u32")

A = np.random.randn(N).astype(np.float32)
A[::7] = np.nan
out, msl = _run(isnan_k, [A], N * 4, np.uint32)
assert "isnan" in msl
np.testing.assert_array_equal(out.astype(bool), np.isnan(A))
print("OK  isnan")


print("\nAll int/simd/cast/predicate numerical tests passed.")
