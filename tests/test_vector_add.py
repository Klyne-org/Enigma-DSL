import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma


class TestVectorAdd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @enigma.kernel
        def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        cls.compiled = enigma.compile(vector_add)
        cls.runtime = enigma.MetalRuntime()

    def _run_vector_add(self, N):
        A = np.random.randn(N).astype(np.float32)
        B = np.random.randn(N).astype(np.float32)
        result = np.frombuffer(
            self.runtime.execute(
                self.compiled, [A, B], N * 4, grid=(N, 1, 1), threads=(min(N, 256), 1, 1)
            ),
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, A + B, rtol=1e-5, atol=1e-7)

    def test_small(self):
        self._run_vector_add(4)

    def test_single_threadgroup(self):
        self._run_vector_add(256)

    def test_multiple_threadgroups(self):
        self._run_vector_add(1024)

    def test_large(self):
        self._run_vector_add(65536)

    def test_metal_source_generation(self):
        src = self.compiled.metal_source
        self.assertIn("kernel void vector_add", src)
        self.assertIn("device const float* A [[buffer(0)]]", src)
        self.assertIn("device const float* B [[buffer(1)]]", src)
        self.assertIn("device float* C [[buffer(2)]]", src)
        self.assertIn("uint tid [[thread_position_in_grid]]", src)
        self.assertIn("A[tid]", src)
        self.assertIn("B[tid]", src)
        self.assertIn("C[tid]", src)


class TestTracingIR(unittest.TestCase):
    def test_trace_records_ops(self):
        from enigma.compiler.kernel import trace_kernel

        @enigma.kernel
        def add_kernel(X: enigma.f32, Y: enigma.f32, Z: enigma.f32):
            tid = enigma.thread_position_in_grid
            Z[tid] = X[tid] + Y[tid]

        builder = trace_kernel(add_kernel)
        op_types = [op.op_type for op in builder.ops]
        self.assertIn("thread_position_in_grid", op_types)
        self.assertIn("load", op_types)
        self.assertIn("add", op_types)
        self.assertIn("store", op_types)
        self.assertEqual(builder.kernel_name, "add_kernel")
        self.assertEqual(len(builder.args), 3)

    def test_metal_emission(self):
        from enigma.compiler.kernel import trace_kernel
        from enigma.compiler.metal_emitter import emit_metal

        @enigma.kernel
        def my_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        builder = trace_kernel(my_add)
        source = emit_metal(builder)
        self.assertIn("#include <metal_stdlib>", source)
        self.assertIn("kernel void my_add", source)


class TestExportMetal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @enigma.kernel
        def add_export(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        cls.compiled = enigma.compile(add_export)

    def test_export_metal_default_path(self):
        import tempfile

        path = self.compiled.export_metal(os.path.join(tempfile.mkdtemp(), "add_export.metal"))
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            src = f.read()
        self.assertIn("kernel void add_export", src)
        self.assertEqual(src, self.compiled.metal_source)
        os.remove(path)

    def test_export_metal_custom_path(self):
        import tempfile

        out = os.path.join(tempfile.mkdtemp(), "my_kernel.metal")
        path = self.compiled.export_metal(out)
        self.assertEqual(path, out)
        self.assertTrue(os.path.exists(out))
        with open(out) as f:
            self.assertIn("#include <metal_stdlib>", f.read())
        os.remove(out)

    def test_keep_metal_source_with_work_dir(self):
        import tempfile

        work = tempfile.mkdtemp()

        @enigma.kernel
        def kept(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        enigma.compile(kept, keep_metal_source=True, work_dir=work)
        metal_path = os.path.join(work, "kept.metal")
        self.assertTrue(os.path.exists(metal_path))
        with open(metal_path) as f:
            self.assertIn("kernel void kept", f.read())


if __name__ == "__main__":
    unittest.main()
