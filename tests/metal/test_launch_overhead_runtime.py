# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Runtime tests for the kernel-launch-overhead optimizations.

These exercise the fast paths added to reduce per-launch cost:
  * MetalRuntime library cache — repeated execute() reuses one MTLLibrary
  * PreparedKernel.set_input() — re-run a prepared kernel on changing data
  * MetalRuntime.dispatch_batch() — many kernels, one command buffer / sync

All checks are correctness, not timing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import enigma
from enigma import testing


def _make_vector_add():
    @enigma.kernel
    def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
        tid = enigma.thread_position_in_grid
        C[tid] = A[tid] + B[tid]

    return enigma.compile(vector_add)


@testing.requires_metal
class TestLibraryCache(unittest.TestCase):
    """The runtime should reuse one MTLLibrary across execute() calls."""

    @classmethod
    def setUpClass(cls):
        cls.compiled = _make_vector_add()

    def test_repeated_execute_uses_cached_library(self):
        rt = enigma.MetalRuntime()
        n = 256
        for _ in range(5):
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            out = np.frombuffer(
                rt.execute(self.compiled, [a, b], n * 4,
                           grid=(n, 1, 1), threads=(n, 1, 1)),
                dtype=np.float32,
            )
            np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)
        # One library handle, cached under the metallib path.
        self.assertEqual(len(rt._lib_cache), 1)
        self.assertIn(self.compiled.metallib_path, rt._lib_cache)
        rt.close()
        # close() is idempotent.
        rt.close()
        self.assertEqual(len(rt._lib_cache), 0)

    def test_execute_correct_after_close_reload(self):
        rt = enigma.MetalRuntime()
        n = 64
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        rt.execute(self.compiled, [a, b], n * 4, grid=(n, 1, 1), threads=(n, 1, 1))
        rt.close()
        # After close() the cache is empty; the next call must reload cleanly.
        out = np.frombuffer(
            rt.execute(self.compiled, [a, b], n * 4, grid=(n, 1, 1), threads=(n, 1, 1)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)
        rt.close()


@testing.requires_metal
class TestSetInput(unittest.TestCase):
    """PreparedKernel.set_input() re-runs on new data without realloc."""

    @classmethod
    def setUpClass(cls):
        cls.compiled = _make_vector_add()

    def test_set_input_changes_result(self):
        rt = enigma.MetalRuntime()
        n = 256
        a0 = np.random.randn(n).astype(np.float32)
        b0 = np.random.randn(n).astype(np.float32)
        prepared = rt.prepare(self.compiled, [a0, b0], n * 4)

        prepared.dispatch((n, 1, 1), (n, 1, 1))
        first = np.frombuffer(prepared.read_output(), dtype=np.float32)
        np.testing.assert_allclose(first, a0 + b0, rtol=1e-5, atol=1e-7)

        # Swap in fresh inputs and re-dispatch — result must track new data.
        a1 = np.random.randn(n).astype(np.float32)
        b1 = np.random.randn(n).astype(np.float32)
        prepared.set_input(0, a1)
        prepared.set_input(1, b1)
        prepared.dispatch((n, 1, 1), (n, 1, 1))
        second = np.frombuffer(prepared.read_output(), dtype=np.float32)
        np.testing.assert_allclose(second, a1 + b1, rtol=1e-5, atol=1e-7)

        prepared.release()
        rt.close()

    def test_set_input_rejects_oversized_array(self):
        rt = enigma.MetalRuntime()
        n = 64
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        prepared = rt.prepare(self.compiled, [a, b], n * 4)
        too_big = np.random.randn(n * 2).astype(np.float32)
        with self.assertRaises(RuntimeError):
            prepared.set_input(0, too_big)
        prepared.release()
        rt.close()

    def test_set_input_rejects_bad_index(self):
        rt = enigma.MetalRuntime()
        n = 64
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        prepared = rt.prepare(self.compiled, [a, b], n * 4)
        with self.assertRaises(RuntimeError):
            prepared.set_input(2, a)        # index 2 is the output buffer
        with self.assertRaises(RuntimeError):
            prepared.set_input(-1, a)
        prepared.release()
        rt.close()


@testing.requires_metal
class TestDispatchBatch(unittest.TestCase):
    """dispatch_batch() runs many kernels in one command buffer."""

    @classmethod
    def setUpClass(cls):
        cls.compiled = _make_vector_add()

    def test_batch_results_match_individual_dispatch(self):
        rt = enigma.MetalRuntime()
        n = 256
        batch = 6
        preps, expected = [], []
        for _ in range(batch):
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            preps.append(rt.prepare(self.compiled, [a, b], n * 4))
            expected.append(a + b)

        jobs = [(p, (n, 1, 1), (n, 1, 1)) for p in preps]
        rt.dispatch_batch(jobs)

        for prepared, exp in zip(preps, expected):
            out = np.frombuffer(prepared.read_output(), dtype=np.float32)
            np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-7)

        for p in preps:
            p.release()
        rt.close()

    def test_empty_batch_is_noop(self):
        rt = enigma.MetalRuntime()
        rt.dispatch_batch([])  # must not raise
        rt.close()

    def test_batch_rejects_non_prepared_job(self):
        rt = enigma.MetalRuntime()
        with self.assertRaises(RuntimeError):
            rt.dispatch_batch([("not a prepared kernel", (1, 1, 1), (1, 1, 1))])
        rt.close()

    def test_single_element_batch(self):
        rt = enigma.MetalRuntime()
        n = 128
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        prepared = rt.prepare(self.compiled, [a, b], n * 4)
        rt.dispatch_batch([(prepared, (n, 1, 1), (n, 1, 1))])
        out = np.frombuffer(prepared.read_output(), dtype=np.float32)
        np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-7)
        prepared.release()
        rt.close()


if __name__ == "__main__":
    unittest.main()
