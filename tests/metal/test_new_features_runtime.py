# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Runtime tests for the metal-only DSL additions.

These exercise the GPU side of the new features:
  * `enigma.gemm` (scalar fallback) end-to-end correctness
  * `enigma.copy(..., coalesced_width=4)` produces correct results
  * Quantization helpers (uint8x4 pack/unpack) round-trip on device
  * `enigma.benchmark.bench` measures GPU dispatch wall-clock time
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma
from enigma import testing


@testing.requires_metal
class TestGemmScalarRuntime(unittest.TestCase):
    """Scalar gemm path on a single thread — correctness, not perf."""

    def test_3x4x2_gemm_correctness(self):
        M, N, K = 3, 4, 2

        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, Out: enigma.f32):
            A_s = enigma.threadgroup_alloc("float", M * K)
            B_s = enigma.threadgroup_alloc("float", K * N)
            for i in range(M * K):
                A_s[i] = A[i]
            for i in range(K * N):
                B_s[i] = B[i]
            enigma.barrier("mem_threadgroup")
            C = enigma.register_tensor((M, N), dtype="float", fill=0.0)
            enigma.gemm(A_s, B_s, C, M=M, N=N, K=K)
            for m in range(M):
                for n in range(N):
                    Out[m * N + n] = C[m, n]

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        b = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.float32)
        raw = rt.execute(
            compiled, [a.ravel(), b.ravel()], M * N * 4,
            grid=(1, 1, 1), threads=(1, 1, 1),
        )
        out = np.frombuffer(raw, dtype=np.float32).copy().reshape(M, N)
        np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-5)


@testing.requires_metal
class TestCopyCoalescedRuntime(unittest.TestCase):
    def test_coalesced_4_round_trip(self):
        N = 256

        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            shared = enigma.threadgroup_alloc("float", N)
            enigma.copy(A, shared, count=N, coalesced_width=4)
            enigma.barrier("mem_threadgroup")
            tid = enigma.thread_position_in_grid
            B[tid] = shared[tid]

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        a = np.arange(N, dtype=np.float32)
        raw = rt.execute(
            compiled, [a], N * 4,
            grid=(N, 1, 1), threads=(N, 1, 1),
        )
        out = np.frombuffer(raw, dtype=np.float32).copy()
        np.testing.assert_array_equal(out, a)


@testing.requires_metal
class TestQuantHelpersRuntime(unittest.TestCase):
    def test_pack_unpack_uint8x4(self):
        N = 64

        @enigma.kernel
        def k(In: enigma.u32, Out: enigma.u32):
            tid = enigma.thread_position_in_grid
            packed = In[tid]
            l0, l1, l2, l3 = enigma.unpack_uint8x4(packed)
            re = enigma.pack_uint8x4(l0, l1, l2, l3)
            Out[tid] = re

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2**32, size=N, dtype=np.uint32)
        raw = rt.execute(
            compiled, [a], N * 4,
            grid=(N, 1, 1), threads=(N, 1, 1),
        )
        out = np.frombuffer(raw, dtype=np.uint32).copy()
        np.testing.assert_array_equal(out, a)


@testing.requires_metal
class TestBenchmarkBenchOnDevice(unittest.TestCase):
    def test_bench_returns_positive_times(self):
        from enigma import benchmark

        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        n = 4096
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)

        def run():
            rt.execute(
                compiled, [a, b], n * 4,
                grid=(n, 1, 1), threads=(64, 1, 1),
            )

        result = benchmark.bench(run, repeat=5, warmup=1, label="vector_add")
        self.assertEqual(result.n, 5)
        self.assertGreater(result.median_us, 0.0)


if __name__ == "__main__":
    unittest.main()
