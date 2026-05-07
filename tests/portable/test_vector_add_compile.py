# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import enigma


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


if __name__ == "__main__":
    unittest.main()
