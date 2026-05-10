# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Runtime tests for DSL surfaces routed through the MLIR emitter."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide


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


class TestVecWidthCompile(unittest.TestCase):
    def test_float4_emits_vec_buffer(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k, vec_width=4)
        self.assertIn("device float4*", compiled.metal_source)
        self.assertIsNotNone(compiled.mlir_source)
        self.assertIn("vector<4xf32>", compiled.mlir_source)


class TestTVLayoutCompile(unittest.TestCase):
    def test_tv_add_via_mlir_compiles(self):
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

        m, n = 256, 512
        mA = Tensor("A", 0, "float", enigma.Layout((m, n), (n, 1)))
        mB = Tensor("B", 1, "float", enigma.Layout((m, n), (n, 1)))
        mC = Tensor("C", 2, "float", enigma.Layout((m, n), (n, 1)))

        compiled = enigma.compile(launch, mA, mB, mC)
        self.assertIsNotNone(compiled.mlir_source)


class TestComparisonAndWhereRuntime(unittest.TestCase):
    def test_cmp_gt_where_float(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, Out: enigma.f32):
            tid = enigma.thread_position_in_grid
            a = A[tid]
            b = B[tid]
            Out[tid] = enigma.where(b, a, enigma.cmp_gt(a, b))

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        out = np.frombuffer(
            rt.execute(compiled, [a, b], n * 4, grid=(n, 1, 1), threads=(256, 1, 1)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, np.maximum(a, b), rtol=1e-5, atol=1e-7)

    def test_cmp_lt_int(self):
        @enigma.kernel
        def k(A: enigma.i32, B: enigma.i32, Out: enigma.i32):
            tid = enigma.thread_position_in_grid
            Out[tid] = enigma.where(B[tid], A[tid], enigma.cmp_lt(A[tid], B[tid]))

        compiled = enigma.compile(k)
        rt = enigma.MetalRuntime()
        n = 1024
        a = np.random.randint(-1000, 1000, size=n, dtype=np.int32)
        b = np.random.randint(-1000, 1000, size=n, dtype=np.int32)
        out = np.frombuffer(
            rt.execute(compiled, [a, b], n * 4, grid=(n, 1, 1), threads=(256, 1, 1)),
            dtype=np.int32,
        )
        np.testing.assert_array_equal(out, np.minimum(a, b))


class TestVecWidthRuntime(unittest.TestCase):
    def test_float4_correctness(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k, vec_width=4)
        rt = enigma.MetalRuntime()
        n = 4096
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        out = np.frombuffer(
            rt.execute(compiled, [a, b], n * 4, grid=(n // 4, 1, 1), threads=(64, 1, 1)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)


class TestTVLayoutRuntime(unittest.TestCase):
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

        m, n = 256, 512
        mA = Tensor("A", 0, "float", enigma.Layout((m, n), (n, 1)))
        mB = Tensor("B", 1, "float", enigma.Layout((m, n), (n, 1)))
        mC = Tensor("C", 2, "float", enigma.Layout((m, n), (n, 1)))

        compiled = enigma.compile(launch, mA, mB, mC)
        a = np.random.randn(m, n).astype(np.float32)
        b = np.random.randn(m, n).astype(np.float32)
        rt = enigma.MetalRuntime()
        out = np.frombuffer(
            rt.execute(
                compiled,
                inputs=[a.ravel(), b.ravel()],
                output_size=m * n * 4,
                grid=compiled.grid,
                threads=compiled.block,
            ),
            dtype=np.float32,
        ).reshape(m, n)
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
