# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import enigma
from enigma.compiler.kernel import trace_kernel


# Kernels at module level so inspect.getsource() can find them.

@enigma.kernel
def acc_kernel(A: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    total = enigma.metal_cast(0.0, "float")
    for i in enigma.range(10):
        total = total + A[i]
    Out[tid] = total


@enigma.kernel
def multi_kernel(A: enigma.f32, B: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    a = enigma.metal_cast(0.0, "float")
    b = enigma.metal_cast(1.0, "float")
    for i in enigma.range(5):
        a = a + A[i]
        b = b * B[i]
    Out[tid] = a + b


@enigma.kernel
def nocarry_kernel(A: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    for i in enigma.range(10):
        Out[i] = A[i]


@enigma.kernel
def unroll_kernel(Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    for h in enigma.range_constexpr(4):
        Out[tid + h] = enigma.metal_cast(float(h), "float")


class TestEnigmaRange(unittest.TestCase):
    def test_simple_accumulator(self):
        builder = trace_kernel(acc_kernel)
        op_types = [op.op_type for op in builder.ops]
        self.assertIn("scf_for", op_types)
        scf = next(op for op in builder.ops if op.op_type == "scf_for")
        self.assertIn("iter_args", scf.attrs)
        self.assertEqual(len(scf.attrs["iter_args"]), 1)

    def test_multiple_carries(self):
        builder = trace_kernel(multi_kernel)
        scf = next(op for op in builder.ops if op.op_type == "scf_for")
        self.assertEqual(len(scf.attrs["iter_args"]), 2)

    def test_no_carry(self):
        builder = trace_kernel(nocarry_kernel)
        scf = next(op for op in builder.ops if op.op_type == "scf_for")
        self.assertNotIn("iter_args", scf.attrs)


class TestRangeConstexpr(unittest.TestCase):
    def test_unrolls(self):
        builder = trace_kernel(unroll_kernel)
        stores = [op for op in builder.ops if op.op_type == "store"]
        self.assertEqual(len(stores), 4)
        scf_ops = [op for op in builder.ops if op.op_type == "scf_for"]
        self.assertEqual(len(scf_ops), 0)


if __name__ == "__main__":
    unittest.main()
