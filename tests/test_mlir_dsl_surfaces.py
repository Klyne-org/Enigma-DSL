"""End-to-end tests for DSL surfaces routed through the MLIR emitter:

- Comparisons (cmp_eq/ne/lt/gt/le/ge on int and float) via `where`
- Grid/thread query accessors with x/y/z dims
- bf16 / i8 / i16 / i64 scalar types
- vec_width > 0 buffer rewriting
- TV-layout (tv_load / tv_add / tv_store) lowered to per-element MLIR
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide


class TestComparisonAndWhere(unittest.TestCase):
    def test_cmp_gt_where_float(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, Out: enigma.f32):
            tid = enigma.thread_position_in_grid
            a = A[tid]
            b = B[tid]
            # enigma.where signature: (false_val, true_val, cond) -> result
            Out[tid] = enigma.where(b, a, enigma.cmp_gt(a, b))

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        N = 1024
        A = np.random.randn(N).astype(np.float32)
        B = np.random.randn(N).astype(np.float32)
        out = np.frombuffer(
            rt.execute(compiled, [A, B], N * 4, grid=(N, 1, 1), threads=(256, 1, 1)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, np.maximum(A, B), rtol=1e-5, atol=1e-7)

    def test_cmp_lt_int(self):
        @enigma.kernel
        def k(A: enigma.i32, B: enigma.i32, Out: enigma.i32):
            tid = enigma.thread_position_in_grid
            Out[tid] = enigma.where(B[tid], A[tid], enigma.cmp_lt(A[tid], B[tid]))

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        N = 1024
        A = np.random.randint(-1000, 1000, size=N, dtype=np.int32)
        B = np.random.randint(-1000, 1000, size=N, dtype=np.int32)
        out = np.frombuffer(
            rt.execute(compiled, [A, B], N * 4, grid=(N, 1, 1), threads=(256, 1, 1)),
            dtype=np.int32,
        )
        np.testing.assert_array_equal(out, np.minimum(A, B))


class TestGridQueries(unittest.TestCase):
    def test_all_query_ops_emit(self):
        @enigma.kernel
        def k(Out: enigma.i32):
            tid = enigma.thread_position_in_grid
            a = enigma.thread_index_in_threadgroup()
            b = enigma.thread_index_in_simdgroup()
            c = enigma.threads_per_simdgroup()
            d = enigma.simdgroup_index_in_threadgroup()
            Out[tid] = enigma.metal_cast(a + b + c + d, enigma.i32)

        compiled = enigma.compile(k)
        self.assertIn("kernel void k", compiled.metal_source)


class TestExtendedDtypes(unittest.TestCase):
    def test_bf16_add(self):
        @enigma.kernel
        def k(A: enigma.bf16, B: enigma.bf16, Out: enigma.bf16):
            tid = enigma.thread_position_in_grid
            Out[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k)
        self.assertIn("device bfloat*", compiled.metal_source)

    def test_i64_add(self):
        @enigma.kernel
        def k(A: enigma.i64, B: enigma.i64, Out: enigma.i64):
            tid = enigma.thread_position_in_grid
            Out[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k)
        self.assertIn("device long*", compiled.metal_source)


class TestVecWidth(unittest.TestCase):
    def test_float4_emits_vec_buffer(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k, vec_width=4)
        self.assertIn("device float4*", compiled.metal_source)
        # MLIR source is always present now (no fallback).
        self.assertIsNotNone(compiled.mlir_source)
        self.assertIn("vector<4xf32>", compiled.mlir_source)

    def test_float4_correctness(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k, vec_width=4)
        rt = enigma.MetalRuntime()
        N = 4096
        A = np.random.randn(N).astype(np.float32)
        B = np.random.randn(N).astype(np.float32)
        out = np.frombuffer(
            rt.execute(compiled, [A, B], N * 4, grid=(N // 4, 1, 1), threads=(64, 1, 1)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, A + B, rtol=1e-5, atol=1e-7)


class TestTVLayoutThroughMLIR(unittest.TestCase):
    def test_tv_add_via_mlir(self):
        @enigma.kernel
        def add_tv(gA, gB, gC, tv_layout, tiler):
            tidx, _, _ = enigma.arch.thread_idx()
            bidx, _, _ = enigma.arch.block_idx()
            blkA = gA[((None, None), bidx)]
            blkB = gB[((None, None), bidx)]
            blkC = gC[((None, None), bidx)]
            tA = tensor_composition(blkA, tv_layout, tiler)
            tB = tensor_composition(blkB, tv_layout, tiler)
            tC = tensor_composition(blkC, tv_layout, tiler)
            thrA = tA[(tidx, None)]
            thrB = tB[(tidx, None)]
            thrC = tC[(tidx, None)]
            thrC.store(thrA.load() + thrB.load())

        @enigma.jit
        def launch(mA, mB, mC):
            thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
            val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
            tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)
            gA = tensor_zipped_divide(mA, tiler_mn)
            gB = tensor_zipped_divide(mB, tiler_mn)
            gC = tensor_zipped_divide(mC, tiler_mn)
            num_blocks = enigma.size(gA, mode=[1])
            threads = enigma.size(tv_layout, mode=[0])
            add_tv(gA, gB, gC, tv_layout, tiler_mn).launch(
                grid=(num_blocks * threads, 1, 1),
                block=(threads, 1, 1),
            )

        M, N = 256, 512
        mA = Tensor("A", 0, "float", enigma.Layout((M, N), (N, 1)))
        mB = Tensor("B", 1, "float", enigma.Layout((M, N), (N, 1)))
        mC = Tensor("C", 2, "float", enigma.Layout((M, N), (N, 1)))

        compiled = enigma.compile(launch, mA, mB, mC)
        # MLIR source should be present; TV path is now in MLIR.
        self.assertIsNotNone(compiled.mlir_source)

        A = np.random.randn(M, N).astype(np.float32)
        B = np.random.randn(M, N).astype(np.float32)
        rt = enigma.MetalRuntime()
        out = np.frombuffer(
            rt.execute(
                compiled, inputs=[A.ravel(), B.ravel()],
                output_size=M * N * 4, grid=compiled.grid, threads=compiled.block,
            ),
            dtype=np.float32,
        ).reshape(M, N)
        np.testing.assert_allclose(out, A + B, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
