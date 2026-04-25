import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma


class TestVectorAddCompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @enigma.kernel
        def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        cls.compiled = enigma.compile(vector_add)

    def test_metal_source_generation(self):
        src = self.compiled.metal_source
        self.assertIn("kernel void vector_add", src)
        self.assertIn("device float* v0 [[buffer(0)]]", src)
        self.assertIn("device float* v1 [[buffer(1)]]", src)
        self.assertIn("device float* v2 [[buffer(2)]]", src)
        self.assertIn("[[thread_position_in_grid]]", src)


class TestTracingCompile(unittest.TestCase):
    def test_metal_emission(self):
        from enigma.compiler.kernel import trace_kernel
        from enigma.compiler.mlir_emitter import emit_msl

        @enigma.kernel
        def my_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        builder = trace_kernel(my_add)
        source = emit_msl(builder)
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


class TestVectorAddRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        @enigma.kernel
        def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        cls.compiled = enigma.compile(vector_add)
        cls.runtime = enigma.MetalRuntime()

    def _run_vector_add(self, n):
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        result = np.frombuffer(
            self.runtime.execute(
                self.compiled, [a, b], n * 4, grid=(n, 1, 1), threads=(min(n, 256), 1, 1)
            ),
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, a + b, rtol=1e-5, atol=1e-7)

    def test_small(self):
        self._run_vector_add(4)

    def test_single_threadgroup(self):
        self._run_vector_add(256)

    def test_multiple_threadgroups(self):
        self._run_vector_add(1024)

    def test_large(self):
        self._run_vector_add(65536)


if __name__ == "__main__":
    unittest.main()
