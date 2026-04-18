#!/usr/bin/env python3
"""Quad group ops — each quad = 4 threads (2x2 pixel quad on Apple GPUs).

quad_sum(tid)  → every lane in a 4-thread quad gets sum of all 4 tids
quad_max(tid)  → every lane gets the max tid of its quad
quad_broadcast(tid, 0) → every lane gets tid of lane 0 of its quad
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


def _run(kernel, out_dtype=np.uint32):
    compiled = enigma.compile(kernel)
    A = np.zeros(1, dtype=np.uint32)  # dummy input
    raw = runtime.execute(
        compiled, [A], N * 4, grid=(N, 1, 1), threads=(256, 1, 1),
    )
    return np.frombuffer(raw, dtype=out_dtype).copy(), compiled.metal_source


@enigma.kernel
def quad_sum_k(_dummy: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.quad_sum(tid)

out, msl = _run(quad_sum_k)
assert "quad_sum" in msl, msl
tids = np.arange(N, dtype=np.uint32)
expected = tids.reshape(-1, 4).sum(axis=1, keepdims=True).repeat(4, axis=1).reshape(-1).astype(np.uint32)
np.testing.assert_array_equal(out, expected)
print("OK  quad_sum")


@enigma.kernel
def quad_max_k(_dummy: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.quad_max(tid)

out, msl = _run(quad_max_k)
assert "quad_max" in msl
expected = tids.reshape(-1, 4).max(axis=1, keepdims=True).repeat(4, axis=1).reshape(-1).astype(np.uint32)
np.testing.assert_array_equal(out, expected)
print("OK  quad_max")


@enigma.kernel
def quad_broadcast_k(_dummy: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.quad_broadcast(tid, 0)  # pull lane 0 of each quad

out, msl = _run(quad_broadcast_k)
assert "quad_broadcast" in msl
expected = tids.reshape(-1, 4)[:, 0:1].repeat(4, axis=1).reshape(-1).astype(np.uint32)
np.testing.assert_array_equal(out, expected)
print("OK  quad_broadcast")


print("\nAll quad group ops passed.")
