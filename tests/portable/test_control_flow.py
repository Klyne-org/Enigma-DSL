"""Tests for control flow tracing (for_range, if_, while_).

These tests verify the DSL-side tracing infrastructure only (no MLIR
dialect wheel required).  The MLIR emission of scf.for/scf.if/scf.while
is tested separately once the dialect wheel is installed.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import enigma
from enigma._tracing import KernelBuilder, TracingTensor


def _make_builder(*bufs):
    """Create a KernelBuilder with named float buffers."""
    builder = KernelBuilder("test_kernel")
    for i, name in enumerate(bufs):
        builder.args.append((name, i, "float"))
    return builder


class TestForRange(unittest.TestCase):
    """Tests for enigma.for_range()."""

    def test_basic_for_traces(self):
        """for_range produces a scf_for op with a body region."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 10):
                Out[tid] = A[tid]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)
        self.assertEqual(len(for_ops[0].regions), 1)
        self.assertGreater(len(for_ops[0].regions[0]), 0)

    def test_for_iv_is_irvalue(self):
        """The induction variable yielded by for_range is an IRValue."""
        from enigma._tracing import IRValue

        b = _make_builder("A")
        with b:
            A = TracingTensor("A", 0, "float")
            with enigma.for_range(0, 4) as i:
                _ = A[i]
            self.assertIsInstance(i, IRValue)

    def test_for_body_ops_not_in_top_level(self):
        """Ops inside for_range body are nested, not in builder.ops."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 8):
                Out[tid] = A[tid]

        top_types = [op.op_type for op in b.ops]
        # load and store should be inside the for body, not top-level
        self.assertNotIn("load", top_types)
        self.assertNotIn("store", top_types)

    def test_for_with_irvalue_bounds(self):
        """for_range accepts IRValue bounds (not just ints)."""
        b = _make_builder("A")
        with b:
            A = TracingTensor("A", 0, "float")
            lo = enigma.metal_cast(0, "int")
            hi = enigma.metal_cast(10, "int")
            step = enigma.metal_cast(2, "int")
            with enigma.for_range(lo, hi, step) as i:
                _ = A[i]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)

    def test_for_custom_dtype(self):
        """for_range supports custom induction variable dtype."""
        b = _make_builder("A")
        with b:
            A = TracingTensor("A", 0, "float")
            with enigma.for_range(0, 10, dtype="uint") as i:
                _ = A[i]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(for_ops[0].attrs["dtype"], "uint")


class TestIf(unittest.TestCase):
    """Tests for enigma.if_()."""

    def test_if_simple(self):
        """Simple if (no else) traces correctly."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            cond = enigma.cmp_gt(tid, 5)
            with enigma.if_(cond):
                Out[tid] = A[tid]

        if_ops = [op for op in b.ops if op.op_type == "scf_if"]
        self.assertEqual(len(if_ops), 1)
        self.assertEqual(len(if_ops[0].regions), 1)
        self.assertFalse(if_ops[0].attrs["has_else"])

    def test_if_else(self):
        """If/else traces both regions."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            val = A[tid]
            cond = enigma.cmp_gt(val, enigma.metal_cast(0, "float"))
            with enigma.if_(cond) as (then_b, else_b):
                with then_b:
                    Out[tid] = val
                with else_b:
                    Out[tid] = enigma.metal_cast(0, "float")

        if_ops = [op for op in b.ops if op.op_type == "scf_if"]
        self.assertEqual(len(if_ops), 1)
        self.assertTrue(if_ops[0].attrs["has_else"])
        self.assertEqual(len(if_ops[0].regions), 2)
        self.assertGreater(len(if_ops[0].regions[0]), 0)  # then
        self.assertGreater(len(if_ops[0].regions[1]), 0)  # else

    def test_if_body_not_in_top_level(self):
        """Ops inside if_ body are nested, not in builder.ops."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            cond = enigma.cmp_gt(tid, 0)
            with enigma.if_(cond):
                Out[tid] = A[tid]

        top_types = [op.op_type for op in b.ops]
        self.assertNotIn("store", top_types)


class TestWhile(unittest.TestCase):
    """Tests for enigma.while_()."""

    def test_while_traces(self):
        """while_ produces a scf_while op with before+after regions."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            tid = enigma.thread_position_in_grid
            i = enigma.metal_cast(0, "int")
            n = enigma.metal_cast(10, "int")

            with enigma.while_(lambda: enigma.cmp_lt(i, n)):
                _ = A[tid]

        while_ops = [op for op in b.ops if op.op_type == "scf_while"]
        self.assertEqual(len(while_ops), 1)
        self.assertEqual(len(while_ops[0].regions), 2)
        before = while_ops[0].regions[0]
        after = while_ops[0].regions[1]
        self.assertGreater(len(before), 0)  # condition ops
        self.assertGreater(len(after), 0)  # body ops


class TestNestedControlFlow(unittest.TestCase):
    """Tests for nested control flow."""

    def test_for_with_nested_if(self):
        """for_range containing an if_ nests correctly."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 8) as i:
                cond = enigma.cmp_lt(i, 4)
                with enigma.if_(cond):
                    Out[tid] = A[tid]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)
        for_body = for_ops[0].regions[0]
        if_in_for = [op for op in for_body if op.op_type == "scf_if"]
        self.assertEqual(len(if_in_for), 1)

    def test_nested_for_loops(self):
        """Two levels of for_range nest correctly."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 4):
                with enigma.for_range(0, 4):
                    _ = A[tid]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)  # outer for
        outer_body = for_ops[0].regions[0]
        inner_for = [op for op in outer_body if op.op_type == "scf_for"]
        self.assertEqual(len(inner_for), 1)  # inner for

    def test_sequential_for_loops(self):
        """Two sequential for_range ops at the same level."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 4):
                _ = A[tid]
            with enigma.for_range(0, 8):
                Out[tid] = A[tid]

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 2)

    def test_if_inside_while(self):
        """if_ inside while_ nests correctly."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            i = enigma.metal_cast(0, "int")
            n = enigma.metal_cast(10, "int")

            with enigma.while_(lambda: enigma.cmp_lt(i, n)):
                cond = enigma.cmp_gt(tid, 0)
                with enigma.if_(cond):
                    Out[tid] = A[tid]

        while_ops = [op for op in b.ops if op.op_type == "scf_while"]
        self.assertEqual(len(while_ops), 1)
        after = while_ops[0].regions[1]
        if_in_while = [op for op in after if op.op_type == "scf_if"]
        self.assertEqual(len(if_in_while), 1)


class TestRegionStack(unittest.TestCase):
    """Tests for the KernelBuilder region stack invariants."""

    def test_region_stack_balanced(self):
        """After tracing, region stack should be back to root."""
        b = _make_builder("A")
        with b:
            A = TracingTensor("A", 0, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 4):
                _ = A[tid]
        self.assertEqual(len(b._region_stack), 1)
        self.assertIs(b._region_stack[0], b.ops)

    def test_ops_after_for_at_top_level(self):
        """Ops recorded after a for_range go to the top level."""
        b = _make_builder("A", "Out")
        with b:
            A = TracingTensor("A", 0, "float")
            Out = TracingTensor("Out", 1, "float")
            tid = enigma.thread_position_in_grid
            with enigma.for_range(0, 4):
                _ = A[tid]
            # This store should be at top level, not inside the for
            Out[tid] = A[tid]

        top_types = [op.op_type for op in b.ops]
        # The for and the store should both be at top level
        self.assertIn("scf_for", top_types)
        self.assertIn("store", top_types)


if __name__ == "__main__":
    unittest.main()
