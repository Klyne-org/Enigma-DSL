#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide

# --- Naive scalar kernel ---


@enigma.kernel
def vector_add_naive(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]


naive_compiled = enigma.compile(vector_add_naive)

# --- Same kernel but emitted with float4* buffer types ---

vec4_compiled = enigma.compile(vector_add_naive, vec_width=4)

# --- TV layout kernel ---


@enigma.kernel
def add_kernel_tv(gA, gB, gC, tv_layout, tiler):
    tidx, _, _ = enigma.arch.thread_idx()
    bidx, _, _ = enigma.arch.block_idx()
    blkA = gA[((None, None), bidx)]
    blkB = gB[((None, None), bidx)]
    blkC = gC[((None, None), bidx)]
    tidfrgA = tensor_composition(blkA, tv_layout, tiler)
    tidfrgB = tensor_composition(blkB, tv_layout, tiler)
    tidfrgC = tensor_composition(blkC, tv_layout, tiler)
    thrA = tidfrgA[(tidx, None)]
    thrB = tidfrgB[(tidx, None)]
    thrC = tidfrgC[(tidx, None)]
    thrC.store(thrA.load() + thrB.load())


M, N = 1024, 1024
TOTAL = M * N


@enigma.jit
def elementwise_add_tv(mA, mB, mC):
    thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)
    gA = tensor_zipped_divide(mA, tiler_mn)
    gB = tensor_zipped_divide(mB, tiler_mn)
    gC = tensor_zipped_divide(mC, tiler_mn)
    num_blocks = enigma.size(gA, mode=[1])
    threads = enigma.size(tv_layout, mode=[0])
    add_kernel_tv(gA, gB, gC, tv_layout, tiler_mn).launch(
        grid=(num_blocks * threads, 1, 1), block=(threads, 1, 1)
    )


mA = Tensor("A", 0, "float", enigma.Layout((M, N), (N, 1)))
mB = Tensor("B", 1, "float", enigma.Layout((M, N), (N, 1)))
mC = Tensor("C", 2, "float", enigma.Layout((M, N), (N, 1)))
tv_compiled = enigma.compile(elementwise_add_tv, mA, mB, mC)

# --- Print generated Metal for vec4 ---

print("float4 kernel (generated):")
print(vec4_compiled.metal_source)

# --- Verify ---

print(f"Tensor: {M}x{N} float32 ({TOTAL * 4 / 1e6:.0f} MB)")

A_np = np.random.randn(TOTAL).astype(np.float32)
B_np = np.random.randn(TOTAL).astype(np.float32)
expected = A_np + B_np
runtime = enigma.MetalRuntime()

naive_out = np.frombuffer(
    runtime.execute(
        naive_compiled, [A_np, B_np], TOTAL * 4, grid=(TOTAL, 1, 1), threads=(256, 1, 1)
    ),
    dtype=np.float32,
)
np.testing.assert_allclose(naive_out, expected, rtol=1e-5)
print("Naive:    correct")

vec4_out = np.frombuffer(
    runtime.execute(
        vec4_compiled, [A_np, B_np], TOTAL * 4, grid=(TOTAL // 4, 1, 1), threads=(256, 1, 1)
    ),
    dtype=np.float32,
)
np.testing.assert_allclose(vec4_out, expected, rtol=1e-5)
print("float4:   correct")

tv_out = np.frombuffer(
    runtime.execute(
        tv_compiled, [A_np, B_np], TOTAL * 4, grid=tv_compiled.grid, threads=tv_compiled.block
    ),
    dtype=np.float32,
)
np.testing.assert_allclose(tv_out, expected, rtol=1e-5)
print("TV:       correct")

# --- Benchmark ---

WARMUP, ITERS = 10, 100


def bench(name, compiled_k, grid, threads):
    prep = runtime.prepare(compiled_k, [A_np, B_np], TOTAL * 4)
    for _ in range(WARMUP):
        prep.dispatch(grid=grid, threads=threads)
    times = []
    for _ in range(ITERS):
        gpu_us = prep.dispatch_timed(grid=grid, threads=threads)
        times.append(gpu_us)
    prep.release()
    med = np.median(times)
    bw = 3 * TOTAL * 4 / (med * 1e-6) / 1e9
    print(f"  {name:40s}  {med:8.3f} us  {bw:6.1f} GB/s")
    return med


print(f"\n{'─' * 72}")
t_naive = bench("float  (scalar)", naive_compiled, grid=(TOTAL, 1, 1), threads=(256, 1, 1))
t_vec4 = bench("float4 (vec)", vec4_compiled, grid=(TOTAL // 4, 1, 1), threads=(256, 1, 1))
t_tv = bench(
    "TV layout (float4, 16 elem/thread)",
    tv_compiled,
    grid=tv_compiled.grid,
    threads=tv_compiled.block,
)
print(f"{'─' * 72}")
print(f"  float4 vs float:  {t_naive / t_vec4:.2f}x")
print(f"  TV     vs float:  {t_naive / t_tv:.2f}x")
