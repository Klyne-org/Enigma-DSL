# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Runtime tests for GEMM and matmul kernels."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma

N_DIM = 64
K_DIM = 4


@enigma.kernel
def matmul_k4(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    row = enigma.thread_position_in_grid_xyz("y")
    col = enigma.thread_position_in_grid_xyz("x")
    n = enigma.metal_cast(N_DIM, "uint")
    k = enigma.metal_cast(K_DIM, "uint")
    base_a = row * k
    base_b = col
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
    C[row * n + col] = enigma.dot(avec, bvec)


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


class TestMatMulCompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = enigma.compile(matmul_k4)

    def test_metal_source_has_dot(self):
        self.assertIn("dot(", self.compiled.metal_source)

    def test_metal_source_has_float4(self):
        self.assertIn("float4(", self.compiled.metal_source)

    def test_2d_grid(self):
        src = self.compiled.metal_source
        self.assertIn("_tpg.y", src)
        self.assertIn("_tpg.x", src)


class TestGEMMCompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = enigma.compile(gemm_k4)

    def test_metal_source_has_accumulate(self):
        src = self.compiled.metal_source
        self.assertIn("dot(", src)


class TestSimdgroupMatrixOpsMLIR(unittest.TestCase):
    def test_simdgroup_gemm_traces(self):
        from enigma.compiler.kernel import trace_kernel
        from enigma.compiler.mlir_emitter import emit_mlir

        @enigma.kernel
        def simd_gemm(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            a_mat = enigma.simdgroup_matrix_load(A, 8)
            b_mat = enigma.simdgroup_matrix_load(B, 8)
            zero_val = enigma.metal_cast(0, "float")
            c_mat = enigma.make_filled_simdgroup_matrix(zero_val)
            result = enigma.simdgroup_multiply_accumulate(a_mat, b_mat, c_mat)
            enigma.simdgroup_matrix_store(result, C, 8)

        builder = trace_kernel(simd_gemm)
        op_types = [op.op_type for op in builder.ops]
        self.assertIn("simdgroup_matrix_load", op_types)
        self.assertIn("simdgroup_matrix_store", op_types)
        self.assertIn("simdgroup_multiply_accumulate", op_types)
        self.assertIn("make_filled_simdgroup_matrix", op_types)

        mlir = emit_mlir(builder)
        self.assertIn("enigma.simdgroup_matrix_load", mlir)
        self.assertIn("enigma.simdgroup_matrix_store", mlir)
        self.assertIn("enigma.simdgroup_multiply_accumulate", mlir)
        self.assertIn("enigma.make_filled_simdgroup_matrix", mlir)
        self.assertIn("vector<8x8xf32>", mlir)


class TestMatMulRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = enigma.compile(matmul_k4)
        cls.runtime = enigma.MetalRuntime()

    def _run_matmul(self, m):
        a = np.random.randn(m, K_DIM).astype(np.float32)
        b = np.random.randn(K_DIM, N_DIM).astype(np.float32)
        raw = self.runtime.execute(
            self.compiled,
            [a.ravel(), b.ravel()],
            m * N_DIM * 4,
            grid=(N_DIM, m, 1),
            threads=(min(N_DIM, 16), min(m, 16), 1),
        )
        out = np.frombuffer(raw, dtype=np.float32).copy().reshape(m, N_DIM)
        np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)

    def test_small(self):
        self._run_matmul(4)

    def test_medium(self):
        self._run_matmul(32)

    def test_large(self):
        self._run_matmul(128)


class TestGEMMRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = enigma.compile(gemm_k4)
        cls.runtime = enigma.MetalRuntime()

    def _run_gemm(self, m):
        a = np.random.randn(m, K_DIM).astype(np.float32)
        b = np.random.randn(K_DIM, N_DIM).astype(np.float32)
        raw = self.runtime.execute(
            self.compiled,
            [a.ravel(), b.ravel()],
            m * N_DIM * 4,
            grid=(N_DIM, m, 1),
            threads=(min(N_DIM, 16), min(m, 16), 1),
        )
        out = np.frombuffer(raw, dtype=np.float32).copy().reshape(m, N_DIM)
        expected = a @ b
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_small(self):
        self._run_gemm(4)

    def test_medium(self):
        self._run_gemm(32)

    def test_large(self):
        self._run_gemm(128)


if __name__ == "__main__":
    unittest.main()
