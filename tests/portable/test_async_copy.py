# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Unit tests for the AIR-backed async-copy DSL surface.

Tests verify three layers:
  1) The Python tracer records the right ``async_copy_*`` IR ops.
  2) The MLIR emitter lowers them to ``enigma.async_copy_*`` dialect ops.
  3) The MSL emitter writes the exact ``__asm("air.*")`` extern decls.

These tests are portable (no Metal device required); they only exercise
the trace -> MLIR -> MSL pipeline.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import enigma  # noqa: E402


class TestAsyncCopyTracing(unittest.TestCase):
    def test_trace_records_1d_d2t_and_wait(self):
        from enigma.compiler.kernel import trace_kernel

        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tile = enigma.threadgroup_alloc("float", 64)
            c0 = enigma.metal_cast(0, "uint")
            cnt = enigma.metal_cast(64, "uint")
            ev = enigma.async_copy_1d_d2t(tile, c0, A, c0, cnt)
            enigma.async_copy_wait(ev)

        b = trace_kernel(k)
        op_types = [op.op_type for op in b.ops]
        self.assertIn("async_copy_1d_d2t", op_types)
        self.assertIn("async_copy_wait", op_types)

    def test_trace_records_all_directions(self):
        from enigma.compiler.kernel import trace_kernel

        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tile = enigma.threadgroup_alloc("float", 64)
            c0 = enigma.metal_cast(0, "uint")
            cnt = enigma.metal_cast(64, "uint")
            epr = enigma.metal_cast(8, "uint")
            tc = enigma.metal_cast(8, "uint")
            tr = enigma.metal_cast(8, "uint")

            e0 = enigma.async_copy_1d_d2t(tile, c0, A, c0, cnt)
            e1 = enigma.async_copy_2d_d2t(tile, c0, epr, A, c0, cnt, tc, tr)
            enigma.async_copy_wait(e0, e1)
            enigma.barrier()
            e2 = enigma.async_copy_1d_t2d(B, c0, tile, c0, cnt)
            e3 = enigma.async_copy_2d_t2d(B, c0, cnt, tile, c0, epr, tc, tr)
            enigma.async_copy_wait(e2, e3)

        op_types = [op.op_type for op in trace_kernel(k).ops]
        for needle in (
            "async_copy_1d_d2t",
            "async_copy_1d_t2d",
            "async_copy_2d_d2t",
            "async_copy_2d_t2d",
            "async_copy_wait",
        ):
            self.assertIn(needle, op_types, f"missing tracer op {needle}")


class TestAsyncCopyMSL(unittest.TestCase):
    """End-to-end DSL -> MSL emission. Verifies the AIR intrinsic strings."""

    def _emit(self, fn):
        from enigma.compiler.kernel import trace_kernel
        from enigma.compiler.mlir_emitter import emit_msl

        return emit_msl(trace_kernel(fn))

    def test_msl_includes_all_five_air_intrinsics(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tile = enigma.threadgroup_alloc("float", 64)
            c0 = enigma.metal_cast(0, "uint")
            cnt = enigma.metal_cast(64, "uint")
            epr = enigma.metal_cast(8, "uint")
            tc = enigma.metal_cast(8, "uint")
            tr = enigma.metal_cast(8, "uint")
            e0 = enigma.async_copy_1d_d2t(tile, c0, A, c0, cnt)
            e1 = enigma.async_copy_1d_t2d(B, c0, tile, c0, cnt)
            e2 = enigma.async_copy_2d_d2t(tile, c0, epr, A, c0, cnt, tc, tr)
            e3 = enigma.async_copy_2d_t2d(B, c0, cnt, tile, c0, epr, tc, tr)
            enigma.async_copy_wait(e0, e1, e2, e3)

        msl = self._emit(k)

        # All five AIR intrinsic asm strings must appear exactly once at file
        # scope (before the kernel).
        for intrinsic in (
            "air.simdgroup_async_copy_1d.p3i8.p1i8",
            "air.simdgroup_async_copy_1d.p1i8.p3i8",
            "air.simdgroup_async_copy_2d.p3i8.p1i8",
            "air.simdgroup_async_copy_2d.p1i8.p3i8",
            "air.wait_simdgroup_events",
        ):
            self.assertIn(intrinsic, msl, f"missing AIR intrinsic in MSL: {intrinsic}")

        # The preamble must come before the kernel definition.
        kernel_pos = msl.find("kernel void")
        preamble_pos = msl.find("struct _enigma_async_event_t")
        self.assertGreater(kernel_pos, preamble_pos, "async-copy preamble must precede kernel")

    def test_msl_does_not_emit_preamble_when_unused(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tid = enigma.thread_position_in_grid
            B[tid] = A[tid]

        msl = self._emit(k)
        self.assertNotIn("air.simdgroup_async_copy", msl)
        self.assertNotIn("_enigma_async_event_t", msl)


if __name__ == "__main__":
    unittest.main()
