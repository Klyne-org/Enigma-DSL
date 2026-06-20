# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma
from enigma import testing


def _has_mlir_bindings():
    try:
        import mlir  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_mlir_bindings(), "requires MLIR Python bindings")
@testing.requires_metal
class TestProfilerRuntime(unittest.TestCase):
    def test_execute_records_gpu_dispatch_event(self):
        @enigma.kernel
        def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(vector_add)
        runtime = enigma.MetalRuntime()
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)

        with enigma.profile(profile_memory=True) as prof:
            raw = runtime.execute(
                compiled,
                [a, b],
                n * 4,
                grid=(n, 1, 1),
                threads=(256, 1, 1),
            )

        out = np.frombuffer(raw, dtype=np.float32)
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)

        gpu_events = [e for e in prof.events() if e.name == "gpu_dispatch"]
        self.assertEqual(len(gpu_events), 1)
        event = gpu_events[0]
        self.assertEqual(event.kernel_name, "vector_add")
        self.assertEqual(event.grid, (n, 1, 1))
        self.assertEqual(event.threads, (256, 1, 1))
        self.assertGreater(event.gpu_time_us, 0.0)
        self.assertEqual(event.input_bytes, n * 4 * 2)
        self.assertEqual(event.output_bytes, n * 4)
        self.assertIn("scheduling_us", event.metadata)
        self.assertIn("queue_wait_us", event.metadata)
        if runtime.supports_stage_counters():
            self.assertGreater(event.metadata["encoder_gpu_us"], 0.0)
        self.assertGreaterEqual(event.metadata["max_threads_per_threadgroup"], 256)
        self.assertGreater(event.metadata["thread_execution_width"], 0)
        self.assertGreater(event.metadata["threadgroup_occupancy"], 0.0)
        self.assertLessEqual(event.metadata["threadgroup_occupancy"], 1.0)

    def test_scope_and_hook_attach_call_path_and_metrics(self):
        @enigma.kernel
        def vector_scale(A: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] * 2.0

        compiled = enigma.compile(vector_scale)
        runtime = enigma.MetalRuntime()
        n = 1024
        a = np.random.randn(n).astype(np.float32)

        enigma.register_kernel_hook(
            "vector_scale",
            lambda *, kernel_name, grid, threads: {"flops": float(grid[0]), "bytes": 8.0 * grid[0]},
        )
        try:
            with enigma.profile() as prof:
                with enigma.scope("layer0"):
                    runtime.execute(
                        compiled, [a], n * 4, grid=(n, 1, 1), threads=(256, 1, 1)
                    )
        finally:
            enigma.unregister_kernel_hook("vector_scale")

        gpu_event = next(e for e in prof.events() if e.name == "gpu_dispatch")
        self.assertEqual(gpu_event.call_path, ("layer0",))
        self.assertEqual(gpu_event.metrics["flops"], float(n))
        row = prof.key_averages().by_name("gpu_dispatch")
        self.assertGreater(row.gflops_per_s, 0.0)
        self.assertIn("layer0", prof.tree())

    def test_benchmark_kernel_reports_steady_state_gpu_time(self):
        @enigma.kernel
        def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(vector_add)
        runtime = enigma.MetalRuntime()
        n = 1 << 16
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        prepared = runtime.prepare(compiled, [a, b], n * 4)
        try:
            bench = enigma.benchmark_kernel(
                prepared,
                grid=(n, 1, 1),
                threads=(256, 1, 1),
                repeat=20,
                warmup=5,
                bytes_moved=12.0 * n,
            )
        finally:
            prepared.release()

        self.assertEqual(bench.calls, 20)
        self.assertGreater(bench.median_us, 0.0)
        self.assertLessEqual(bench.min_us, bench.median_us)
        self.assertLessEqual(bench.median_us, bench.max_us)
        self.assertGreater(bench.gbytes_per_s, 0.0)

    def test_gputrace_capture(self):
        import tempfile

        @enigma.kernel
        def vector_copy(A: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid]

        compiled = enigma.compile(vector_copy)
        runtime = enigma.MetalRuntime()
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        path = os.path.join(tempfile.mkdtemp(), "enigma.gputrace")

        try:
            with enigma.profile(capture=path):
                runtime.execute(compiled, [a], n * 4, grid=(n, 1, 1), threads=(256, 1, 1))
        except RuntimeError:
            # Capture needs MTL_CAPTURE_ENABLED=1; without it the runtime
            # must fail loudly, never silently skip.
            if os.environ.get("MTL_CAPTURE_ENABLED") == "1":
                raise
            return
        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
