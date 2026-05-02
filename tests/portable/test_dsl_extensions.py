"""Tests for DSL extensions added after R1: Carry/iter_args, Scalar args,
RegisterTensor, copy, load_if/store_if, Pipeline, capability queries.

These tests only exercise tracing / compile-time behavior and the Python
runtime-side packing logic. End-to-end GPU dispatch is covered by
examples/ and is not required for CI."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import enigma
from enigma._tracing import KernelBuilder, RegisterTensor, Tensor


def _make_builder(*bufs):
    builder = KernelBuilder("test_kernel")
    for i, name in enumerate(bufs):
        builder.args.append((name, i, "float"))
    return builder


class TestForRangeIterArgs(unittest.TestCase):
    def test_init_produces_iter_args_and_results(self):
        b = _make_builder("A", "Out")
        with b:
            A = Tensor("A", 0, "float")
            Out = Tensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            zero = b.make_const("float", 0.0)
            with enigma.for_range(0, 8, init=[zero]) as (i, carry):
                carry[0] = carry[0] + A[i]
            Out[tid] = carry[0]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)
        for_op = for_ops[0]
        self.assertIn("iter_args", for_op.attrs)
        self.assertIn("yield_vals", for_op.attrs)
        self.assertIn("results", for_op.attrs)
        self.assertEqual(len(for_op.attrs["iter_args"]), 1)
        self.assertEqual(len(for_op.attrs["results"]), 1)

    def test_carry_rebound_after_exit(self):
        """After the with-block, carry[i] must equal the for_op's i-th result."""
        b = _make_builder("A", "Out")
        with b:
            A = Tensor("A", 0, "float")
            zero = b.make_const("float", 0.0)
            with enigma.for_range(0, 4, init=[zero]) as (i, carry):
                carry[0] = carry[0] + A[i]
            final = carry[0]

        for_op = [op for op in b.ops if op.op_type == "scf_for"][0]
        self.assertIs(final, for_op.attrs["results"][0])


class TestScalarAnnotation(unittest.TestCase):
    def test_scalar_holds_dtype(self):
        s = enigma.Scalar(enigma.f32)
        self.assertEqual(s.metal_name, "float")
        # width is in bits for Numeric dtypes (f32 → 32).
        self.assertEqual(s.width, 32)


class TestRegisterTensor(unittest.TestCase):
    def test_register_tensor_shape_and_dtype(self):
        b = _make_builder("Out")
        with b:
            rt = enigma.register_tensor(shape=(4, 4), dtype="float", fill=0.0)
            self.assertIsInstance(rt, RegisterTensor)
            self.assertEqual(rt.shape, (4, 4))
            self.assertEqual(rt.dtype, "float")

    def test_register_tensor_indexing_emits_ops(self):
        b = _make_builder("A", "Out")
        with b:
            A = Tensor("A", 0, "float")
            reg = enigma.register_tensor(shape=(2, 2), dtype="float", fill=0.0)
            tid = enigma.thread_position_in_grid
            reg[0, 0] = A[tid]
            _ = reg[0, 0]
        # register_tensor should not leak scf_for or scf_if ops.
        self.assertEqual([op for op in b.ops if op.op_type.startswith("scf_")], [])


class TestCopy(unittest.TestCase):
    def test_copy_emits_for_loop(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float")
            B = Tensor("B", 1, "float")
            enigma.copy(A, B, count=16)
        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)


class TestLoadStoreIf(unittest.TestCase):
    def test_store_if_wraps_store_in_scf_if(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float")
            B = Tensor("B", 1, "float")
            tid = enigma.thread_position_in_grid
            ten = b.make_const("int", 10)
            mask = enigma.cmp_lt(tid, enigma.metal_cast(ten, "int"))
            val = A[tid]
            enigma.store_if(B, tid, val, mask)
        if_ops = [op for op in b.ops if op.op_type == "scf_if"]
        self.assertEqual(len(if_ops), 1)

    def test_load_if_uses_select(self):
        b = _make_builder("A")
        with b:
            A = Tensor("A", 0, "float")
            tid = enigma.thread_position_in_grid
            ten = b.make_const("int", 10)
            mask = enigma.cmp_lt(tid, enigma.metal_cast(ten, "int"))
            _ = enigma.load_if(A, tid, mask, default=0.0)
        # Should produce a select op, not an scf_if.
        select_ops = [op for op in b.ops if op.op_type == "select"]
        self.assertTrue(len(select_ops) >= 1)


class TestPipeline(unittest.TestCase):
    def test_pipeline_has_front_back(self):
        b = _make_builder("A")
        with b:
            pipe = enigma.pipeline(dtype="float", size=8, stages=2)
            self.assertEqual(len(pipe._buffers), 2)
            f = pipe.front()
            back = pipe.back()
            self.assertIsNotNone(f)
            self.assertIsNotNone(back)
            self.assertIsNot(f, back)
            pipe.swap()
            self.assertIs(pipe.front(), back)


class TestDeviceCapabilities(unittest.TestCase):
    """Pure dataclass tests — do NOT require a live Metal device."""

    def test_m3_gating(self):
        from enigma.runtime_dispatch.runtime import DeviceCapabilities

        m1 = DeviceCapabilities("apple7", 1007, False, True, 32, 32768, 1024, "M1")
        m3 = DeviceCapabilities("apple9", 1009, True, True, 32, 32768, 1024, "M3")
        m4 = DeviceCapabilities("apple10", 1010, True, True, 32, 32768, 1024, "M4")
        self.assertFalse(m1.is_m3_or_newer)
        self.assertTrue(m3.is_m3_or_newer)
        self.assertTrue(m4.is_m3_or_newer)

        with self.assertRaises(RuntimeError):
            m1.require_m3("async_copy")
        m3.require_m3("async_copy")  # should not raise


class TestConstantPacking(unittest.TestCase):
    """Bug 3 runtime helper: _pack_constants."""

    def test_pack_constants_empty(self):
        from enigma.runtime_dispatch.runtime import _pack_constants

        inds, tags, vals, n = _pack_constants(None)
        self.assertEqual(n, 0)
        inds, tags, vals, n = _pack_constants({})
        self.assertEqual(n, 0)

    def test_pack_constants_float(self):
        from enigma.runtime_dispatch.runtime import _pack_constants

        inds, tags, vals, n = _pack_constants({0: ("float", 2.5)})
        self.assertEqual(n, 1)
        self.assertEqual(inds[0], 0)
        self.assertEqual(tags[0], 0)  # 0 == float
        self.assertAlmostEqual(vals[0], 2.5)

    def test_pack_constants_unknown_type_raises(self):
        from enigma.runtime_dispatch.runtime import _pack_constants

        with self.assertRaises(RuntimeError):
            _pack_constants({0: ("badtype", 1)})


if __name__ == "__main__":
    unittest.main()
