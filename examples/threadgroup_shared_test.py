#!/usr/bin/env python3
"""Threadgroup shared memory smoke test.

Each thread writes its local index into shared memory, barriers, then reads
back from the slot (block_dim - 1 - tid_in_block) and writes to output.
This exercises:
  - threadgroup_alloc (static-sized shared buffer)
  - store / load on a threadgroup-space buffer
  - threadgroup_barrier to order writes before reads
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

BLOCK = 256
GRID = 1024
runtime = enigma.MetalRuntime()


@enigma.kernel
def shared_reverse(A: enigma.u32, B: enigma.u32):
    tid = enigma.thread_position_in_grid
    tidx, _y, _z = enigma.arch.thread_idx()
    bdim, _b, _c = enigma.arch.block_dim()

    shared = enigma.threadgroup_alloc("uint", BLOCK)
    shared[tidx] = A[tid]
    enigma.barrier("mem_threadgroup")

    mirror = bdim - 1 - tidx
    B[tid] = shared[mirror]


compiled = enigma.compile(shared_reverse)
msl = compiled.metal_source
assert "threadgroup" in msl, msl
assert "threadgroup_barrier" in msl, msl
print("--- generated MSL ---")
print(msl)

A = np.arange(GRID, dtype=np.uint32)
raw = runtime.execute(
    compiled, [A], GRID * 4, grid=(GRID, 1, 1), threads=(BLOCK, 1, 1),
)
out = np.frombuffer(raw, dtype=np.uint32).copy()

expected = A.reshape(-1, BLOCK)[:, ::-1].reshape(-1)
np.testing.assert_array_equal(out, expected)
print(f"OK  threadgroup_alloc + barrier: {GRID} elements reversed in {GRID//BLOCK} blocks")
