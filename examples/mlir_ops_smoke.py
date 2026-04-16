#!/usr/bin/env python3
"""MLIR-level smoke test for all newly wired ops.

Just checks that tracing + _build_module succeed and the expected
enigma.<op> mnemonic appears in the MLIR text. No Metal execution.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import enigma
from enigma.compiler.mlir_emitter import emit_mlir
from enigma._tracing import KernelBuilder


def _trace(fn):
    builder = KernelBuilder(fn.__name__)
    with builder:
        fn(builder)
    return emit_mlir(builder)


# --- Unary int ops (use uint tid) ---
def int_ops(b):
    tid = b.get_thread_position_in_grid()
    _ = enigma.popcount(tid)
    _ = enigma.clz(tid)
    _ = enigma.ctz(tid)
    _ = enigma.reverse_bits(tid)

txt = _trace(int_ops)
for m in ["popcount", "clz", "ctz", "reverse_bits"]:
    assert f"enigma.{m}" in txt, f"missing enigma.{m} in MLIR"
print("OK  int unary ops")


# --- Int binary ops ---
def int_bin(b):
    tid = b.get_thread_position_in_grid()
    _ = enigma.imin(tid, tid)
    _ = enigma.imax(tid, tid)
    _ = enigma.add_sat(tid, tid)
    _ = enigma.sub_sat(tid, tid)
    _ = enigma.mul_hi(tid, tid)
    _ = enigma.rotate(tid, tid)
    _ = enigma.abs_diff(tid, tid)
    _ = enigma.iclamp(tid, tid, tid)
    _ = enigma.mad_sat(tid, tid, tid)

txt = _trace(int_bin)
for m in ["imin", "imax", "add_sat", "sub_sat", "mul_hi", "rotate",
          "abs_diff", "iclamp", "mad_sat"]:
    assert f"enigma.{m}" in txt, f"missing enigma.{m}"
print("OK  int binary/ternary ops")


# --- Bit extract/insert ---
def bit_ops(b):
    tid = b.get_thread_position_in_grid()
    x = enigma.extract_bits(tid, 4, 8)
    _ = enigma.insert_bits(tid, x, 4, 8)

txt = _trace(bit_ops)
assert "enigma.extract_bits" in txt
assert "enigma.insert_bits" in txt
print("OK  extract_bits / insert_bits")


# --- SIMD ---
def simd_ops(b):
    tid = b.get_thread_position_in_grid()
    _ = enigma.simd_sum(tid)
    _ = enigma.simd_product(tid)
    _ = enigma.simd_min(tid)
    _ = enigma.simd_max(tid)
    _ = enigma.simd_and(tid)
    _ = enigma.simd_or(tid)
    _ = enigma.simd_xor(tid)
    _ = enigma.simd_prefix_inclusive_sum(tid)
    _ = enigma.simd_shuffle(tid, tid)
    _ = enigma.simd_shuffle_up(tid, tid)
    _ = enigma.simd_shuffle_down(tid, tid)
    _ = enigma.simd_shuffle_xor(tid, tid)
    _ = enigma.simd_broadcast(tid, tid)

txt = _trace(simd_ops)
for m in ["simd_sum", "simd_product", "simd_min", "simd_max",
          "simd_and", "simd_or", "simd_xor",
          "simd_prefix_inclusive_sum",
          "simd_shuffle", "simd_shuffle_up", "simd_shuffle_down",
          "simd_shuffle_xor", "simd_broadcast"]:
    assert f"enigma.{m}" in txt, f"missing enigma.{m}"
print("OK  simd ops")


# --- Barriers ---
def barriers(b):
    enigma.barrier()
    enigma.barrier("mem_device")
    enigma.simd_barrier()

txt = _trace(barriers)
assert "enigma.threadgroup_barrier" in txt
assert "enigma.simdgroup_barrier" in txt
print("OK  barriers")


# --- Casts ---
def casts(b):
    tid = b.get_thread_position_in_grid()
    _ = enigma.metal_cast(tid, "f32")
    _ = enigma.as_type(tid, "f32")

txt = _trace(casts)
assert "enigma.metal_cast" in txt
assert "enigma.as_type" in txt
print("OK  casts")


print("\nAll MLIR-level smoke checks passed.")
