#!/usr/bin/env python3
"""Global atomic counter: every thread adds 1 to counter[0]; expect counter[0] == N.

Also checks atomic_fetch_max on a scoreboard buffer.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


# MetalRuntime.execute() creates a fresh zero-initialized output buffer
# appended as the LAST kernel arg and returns its contents after dispatch.
# So we make the atomic target the last kernel arg: a 1-element "dummy"
# input arg + the atomic target as the runtime-managed output.


# ---- Global counter via atomic_fetch_add ----
@enigma.kernel
def atomic_counter(_dummy: enigma.u32, counter: enigma.u32):
    tid = enigma.thread_position_in_grid  # noqa: F841
    _ = counter.atomic_fetch_add(0, 1, order="relaxed")


compiled = enigma.compile(atomic_counter)
print("--- atomic_counter MSL ---")
print(compiled.metal_source)
assert "atomic_fetch_add" in compiled.metal_source, compiled.metal_source

dummy = np.zeros(1, dtype=np.uint32)
raw = runtime.execute(
    compiled, [dummy], 4, grid=(N, 1, 1), threads=(min(N, 256), 1, 1),
)
result = np.frombuffer(raw, dtype=np.uint32)
assert result[0] == N, f"expected {N}, got {result[0]}"
print(f"OK  atomic_fetch_add: counter == {N}")


# ---- atomic_fetch_max: each thread tries to push its tid into slot[0] ----
@enigma.kernel
def atomic_max(_dummy: enigma.u32, slot: enigma.u32):
    tid = enigma.thread_position_in_grid
    _ = slot.atomic_fetch_max(0, tid, order="relaxed")


compiled = enigma.compile(atomic_max)
assert "atomic_fetch_max" in compiled.metal_source
raw = runtime.execute(
    compiled, [dummy], 4, grid=(N, 1, 1), threads=(min(N, 256), 1, 1),
)
result = np.frombuffer(raw, dtype=np.uint32)
assert result[0] == N - 1, f"expected {N-1}, got {result[0]}"
print(f"OK  atomic_fetch_max: slot == {N-1}")


print("\nAtomic tests passed.")
