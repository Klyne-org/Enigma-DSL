"""Tests for the metal-only DSL additions inspired by tilelang_metal_analysis.md.

These cover (Tier 1 / Tier 2):
  * `enigma.testing` helpers (requires_metal, is_metal_available, ...)
  * Compiler emit_only / kernel_source / has_metallib
  * `enigma.benchmark` utilities
  * `enigma.gemm` tile op (scalar + simdgroup paths) — trace-only
  * Multi-stage `enigma.pipeline`
  * `enigma.copy(..., coalesced_width=k)`
  * Quantization helpers (pack/unpack uint8x4, int4x2, dequantize_int8)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import enigma
from enigma._tracing import KernelBuilder, Tensor


def _make_builder(*bufs, dtype="float"):
    builder = KernelBuilder("test_kernel")
    for i, name in enumerate(bufs):
        builder.args.append((name, i, dtype))
    return builder


class TestTesting(unittest.TestCase):
    def test_helpers_are_callable(self):
        from enigma import testing

        self.assertIsInstance(testing.is_darwin(), bool)
        self.assertIsInstance(testing.is_apple_silicon(), bool)
        self.assertIsInstance(testing.is_metal_available(), bool)

    def test_requires_metal_attaches_skip_attr(self):
        from enigma import testing

        @testing.requires_metal
        def _decorated():
            return 42

        # unittest.skipUnless sets __unittest_skip__ when condition fails,
        # leaves it unset (False) when condition holds.
        skipped = getattr(_decorated, "__unittest_skip__", False)
        self.assertEqual(skipped, not testing.is_metal_available())

    def test_requires_apple_silicon_attaches_skip_attr(self):
        from enigma import testing

        @testing.requires_apple_silicon
        def _decorated():
            return "asi"

        skipped = getattr(_decorated, "__unittest_skip__", False)
        self.assertEqual(skipped, not testing.is_apple_silicon())

    def test_skip_alias_exists(self):
        from enigma import testing

        self.assertIs(testing.skip_if_no_metal, testing.requires_metal)


class TestCompilerEmitOnly(unittest.TestCase):
    def test_kernel_source_alias_and_emit_only(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32, C: enigma.f32):
            tid = enigma.thread_position_in_grid
            C[tid] = A[tid] + B[tid]

        compiled = enigma.compile(k, emit_only=True)
        self.assertEqual(compiled.kernel_source, compiled.metal_source)
        self.assertIn("kernel void k", compiled.kernel_source)
        self.assertIsNotNone(compiled.mlir_source)
        self.assertFalse(compiled.has_metallib)
        self.assertIsNone(compiled.metallib_path)
        self.assertIsNone(compiled.metallib_bytes)

    def test_env_emit_only_respected(self):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tid = enigma.thread_position_in_grid
            B[tid] = A[tid]

        old = os.environ.get("ENIGMA_EMIT_ONLY")
        try:
            os.environ["ENIGMA_EMIT_ONLY"] = "1"
            compiled = enigma.compile(k)
            self.assertFalse(compiled.has_metallib)
        finally:
            if old is None:
                os.environ.pop("ENIGMA_EMIT_ONLY", None)
            else:
                os.environ["ENIGMA_EMIT_ONLY"] = old


class TestBenchmarkUtilities(unittest.TestCase):
    def test_bench_basic(self):
        from enigma import benchmark

        counter = {"n": 0}

        def fn():
            counter["n"] += 1

        result = benchmark.bench(fn, repeat=10, warmup=2, label="noop")
        self.assertEqual(counter["n"], 10 + 2)
        self.assertEqual(result.label, "noop")
        self.assertEqual(result.n, 10)
        self.assertGreaterEqual(result.median_us, 0.0)
        self.assertGreaterEqual(result.max_us, result.min_us)

    def test_bench_format(self):
        from enigma.benchmark import BenchResult, format_bench_result

        r = BenchResult(label="test", samples_us=[1.5, 1.7, 1.9], warmup=0)
        line = format_bench_result(r)
        self.assertIn("test", line)
        self.assertIn("median", line)
        line2 = format_bench_result(r, throughput_gbps=12.5)
        self.assertIn("BW=12.50GB/s", line2)


class TestGemmTraceOnly(unittest.TestCase):
    def test_gemm_simdgroup_path_emits_simdgroup_ops(self):
        b = _make_builder("A", "B", "C")
        with b:
            A = Tensor("A", 0, "float", address_space="threadgroup", shape=64)
            B = Tensor("B", 1, "float", address_space="threadgroup", shape=64)
            C = Tensor("C", 2, "float", address_space="threadgroup", shape=64)
            enigma.gemm(A, B, C, M=8, N=8, K=8)

        op_types = [op.op_type for op in b.ops]
        self.assertIn("simdgroup_matrix_load", op_types)
        self.assertIn("simdgroup_multiply_accumulate", op_types)
        self.assertIn("simdgroup_matrix_store", op_types)
        self.assertIn("threadgroup_barrier", op_types)
        self.assertNotIn("scf_for", op_types)

    def test_gemm_scalar_path_emits_for_loops(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float", address_space="threadgroup", shape=12)
            B = Tensor("B", 1, "float", address_space="threadgroup", shape=12)
            C = enigma.register_tensor((3, 4), dtype="float", fill=0.0)
            enigma.gemm(A, B, C, M=3, N=4, K=2)

        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 12)  # one per (m, n) cell
        op_types = [op.op_type for op in b.ops]
        self.assertNotIn("simdgroup_matrix_load", op_types)

    def test_gemm_non_8x8x8_falls_back_to_scalar(self):
        # 16x16x16 doesn't fit the single-tile simdgroup path either —
        # multi-tile chaining isn't supported until simdgroup_matrix_load
        # gains an offset operand.
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float", address_space="threadgroup", shape=256)
            B = Tensor("B", 1, "float", address_space="threadgroup", shape=256)
            C = enigma.register_tensor((16, 16), dtype="float", fill=0.0)
            enigma.gemm(A, B, C, M=16, N=16, K=16)
        op_types = [op.op_type for op in b.ops]
        self.assertNotIn("simdgroup_matrix_load", op_types)
        self.assertIn("scf_for", op_types)

    def test_gemm_use_simdgroup_true_invalid_shape_raises(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float", address_space="threadgroup", shape=12)
            B = Tensor("B", 1, "float", address_space="threadgroup", shape=12)
            C = enigma.register_tensor((3, 4), dtype="float", fill=0.0)
            with self.assertRaises(enigma.EnigmaError):
                enigma.gemm(A, B, C, M=3, N=4, K=2, use_simdgroup=True)


class TestMultistagePipeline(unittest.TestCase):
    def test_pipeline_three_stages(self):
        b = _make_builder("A")
        with b:
            pipe = enigma.pipeline(dtype="float", size=8, stages=3)
            self.assertEqual(pipe.stages, 3)
            self.assertEqual(len(pipe._buffers), 3)
            f0 = pipe.front()
            s1 = pipe.stage(1)
            s2 = pipe.stage(2)
            self.assertIsNot(f0, s1)
            self.assertIsNot(s1, s2)
            self.assertIs(pipe.back(), s2)
            pipe.advance()
            self.assertIs(pipe.front(), s1)
            pipe.advance()
            self.assertIs(pipe.front(), s2)
            pipe.advance()
            self.assertIs(pipe.front(), f0)

    def test_pipeline_two_stage_swap_alias(self):
        b = _make_builder("A")
        with b:
            pipe = enigma.pipeline("float", 4, stages=2)
            f = pipe.front()
            pipe.swap()
            self.assertIs(pipe.front(), pipe._buffers[1])
            self.assertIs(pipe.back(), f)

    def test_pipeline_swap_only_for_two_stages(self):
        b = _make_builder("A")
        with b:
            pipe = enigma.pipeline("float", 4, stages=3)
            with self.assertRaises(enigma.EnigmaError):
                pipe.swap()

    def test_pipeline_invalid_stages(self):
        b = _make_builder("A")
        with b:
            with self.assertRaises(enigma.EnigmaError):
                enigma.pipeline("float", 4, stages=1)


class TestCopyCoalescedWidth(unittest.TestCase):
    def test_coalesced_width_iterates_count_div_width(self):
        # With coalesced_width=4, the loop body unrolls 4 elements per
        # iteration, so the trip count is count // 4.
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float")
            B = Tensor("B", 1, "float")
            enigma.copy(A, B, count=16, coalesced_width=4)
        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)
        # 4 unrolled loads + 4 unrolled stores per iteration.
        inner = for_ops[0].regions[0]
        self.assertEqual(sum(1 for o in inner if o.op_type == "load"), 4)
        self.assertEqual(sum(1 for o in inner if o.op_type == "store"), 4)

    def test_coalesced_width_invalid_raises(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float")
            B = Tensor("B", 1, "float")
            with self.assertRaises(enigma.EnigmaError):
                enigma.copy(A, B, count=8, coalesced_width=3)

    def test_coalesced_width_one_iterates_count_times(self):
        b = _make_builder("A", "B")
        with b:
            A = Tensor("A", 0, "float")
            B = Tensor("B", 1, "float")
            enigma.copy(A, B, count=8)  # default coalesced_width=1
        for_ops = [op for op in b.ops if op.op_type == "scf_for"]
        self.assertEqual(len(for_ops), 1)
        inner = for_ops[0].regions[0]
        self.assertEqual(sum(1 for o in inner if o.op_type == "load"), 1)
        self.assertEqual(sum(1 for o in inner if o.op_type == "store"), 1)


class TestQuantizationHelpers(unittest.TestCase):
    def test_pack_uint8x4_emits_insert_bits(self):
        b = _make_builder("Out")
        with b:
            v0 = b.make_const("uint", 1)
            v1 = b.make_const("uint", 2)
            v2 = b.make_const("uint", 3)
            v3 = b.make_const("uint", 4)
            enigma.pack_uint8x4(v0, v1, v2, v3)
        op_types = [op.op_type for op in b.ops]
        self.assertEqual(op_types.count("insert_bits"), 4)

    def test_unpack_uint8x4_emits_extract_bits(self):
        b = _make_builder("Out")
        with b:
            packed = b.make_const("uint", 0xDEADBEEF)
            lanes = enigma.unpack_uint8x4(packed)
            self.assertEqual(len(lanes), 4)
        op_types = [op.op_type for op in b.ops]
        self.assertEqual(op_types.count("extract_bits"), 4)

    def test_pack_int4x2_round_trip_traces(self):
        b = _make_builder("Out")
        with b:
            lo = b.make_const("int", 5)
            hi = b.make_const("int", -3)
            packed = enigma.pack_int4x2(lo, hi)
            lo2, hi2 = enigma.unpack_int4x2(packed)
            self.assertIsNotNone(lo2)
            self.assertIsNotNone(hi2)
        op_types = [op.op_type for op in b.ops]
        self.assertIn("insert_bits", op_types)
        self.assertIn("extract_bits", op_types)
        self.assertIn("select", op_types)

    def test_dequantize_int8(self):
        b = _make_builder("X")
        with b:
            x = b.make_const("int", 64)
            scale = b.make_const("float", 0.0625)
            zero = b.make_const("int", 0)
            res = enigma.dequantize_int8(x, scale, zero_point=zero)
            self.assertEqual(res.dtype, "float")
        op_types = [op.op_type for op in b.ops]
        self.assertIn("metal_cast", op_types)


class TestNoTargetString(unittest.TestCase):
    def test_compile_signature_has_no_target_kwarg(self):
        import inspect

        sig = inspect.signature(enigma.compile)
        self.assertNotIn("target", sig.parameters)


if __name__ == "__main__":
    unittest.main()
