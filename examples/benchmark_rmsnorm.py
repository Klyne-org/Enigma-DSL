#!/usr/bin/env python3
"""RMSNorm benchmark: Enigma DSL (layout algebra) vs handwritten Metal."""
import os
import sys
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide

ROWS = 4096
N = 4096
EPS = 1e-5
THREADS_PER_GROUP = 256
ELEMS_PER_THREAD = N // THREADS_PER_GROUP

thr_layout = enigma.make_ordered_layout((THREADS_PER_GROUP,), order=(0,))
val_layout = enigma.make_ordered_layout((ELEMS_PER_THREAD,), order=(0,))
tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

print(f"TV layout:")
print(f"  tiler  = {tiler_mn}")
print(f"  TV     = {tv_layout}")
print(f"  threads = {enigma.size(tv_layout, mode=[0])}")
print(f"  values  = {enigma.size(tv_layout, mode=[1])}")


@enigma.kernel
def rmsnorm_kernel(gX, gW, gOut, tv_layout, tiler):
    tidx, _, _ = enigma.arch.thread_idx()
    row_idx, _, _ = enigma.arch.block_idx()

    blkX = gX[((None, None), row_idx)][(0, None)]
    blkOut = gOut[((None, None), row_idx)][(0, None)]
    blkW = gW[(None,)]

    tvX = tensor_composition(blkX, tv_layout, tiler)
    tvOut = tensor_composition(blkOut, tv_layout, tiler)
    tvW = tensor_composition(blkW, tv_layout, tiler)

    my_X = tvX[(tidx, None)]
    my_Out = tvOut[(tidx, None)]
    my_W = tvW[(tidx, None)]

    x_vals = my_X.load()
    w_vals = my_W.load()
    sumsq = x_vals * x_vals

    partial = enigma.simd_sum(sumsq)

    shared = enigma.threadgroup_alloc("float", 32)
    simd_lane = enigma.thread_index_in_simdgroup()
    simd_idx = enigma.simdgroup_index_in_threadgroup()

    is_first = enigma.cmp_eq(simd_lane, 0)
    enigma.store_if(shared, simd_idx, partial, is_first)
    enigma.barrier()

    num_sg = enigma.simdgroups_per_threadgroup()
    in_range = enigma.cmp_ult(simd_lane, num_sg)
    val = enigma.load_if(shared, simd_lane, in_range, default=0.0)
    total_sumsq = enigma.simd_sum(val)

    mean_sq = total_sumsq / float(N)
    scale = enigma.rsqrt(mean_sq + EPS)

    result = x_vals * scale * w_vals
    my_Out.store(result)


@enigma.jit
def rmsnorm_jit(mX, mW, mOut):
    tiler_2d = (1, tiler_mn[0]) if isinstance(tiler_mn, tuple) else (1, tiler_mn)
    gX = tensor_zipped_divide(mX, tiler_2d)
    gOut = tensor_zipped_divide(mOut, tiler_2d)
    gW = tensor_zipped_divide(mW, tiler_mn)
    threads = enigma.size(tv_layout, mode=[0])
    rmsnorm_kernel(gX, gW, gOut, tv_layout, tiler_mn).launch(
        grid=(ROWS * threads, 1, 1),
        block=(threads, 1, 1),
    )


mX = Tensor("X", 0, "float", enigma.Layout((ROWS, N), (N, 1)))
mW = Tensor("W", 1, "float", enigma.Layout((N,), (1,)))
mOut = Tensor("Out", 2, "float", enigma.Layout((ROWS, N), (N, 1)))

print("\nCompiling Enigma RMSNorm...")
enigma_compiled = enigma.compile(rmsnorm_jit, mX, mW, mOut)
enigma_compiled.export_metal(os.path.join(os.path.dirname(__file__), "rmsnorm_enigma.metal"))

print("Compiling handwritten RMSNorm...")
hw_dir = tempfile.mkdtemp(prefix="rmsnorm_hw_")
hw_metal = os.path.join(os.path.dirname(__file__), "rmsnorm_handwritten.metal")
hw_air = os.path.join(hw_dir, "rmsnorm.air")
hw_metallib = os.path.join(hw_dir, "rmsnorm.metallib")
subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", hw_metal, "-o", hw_air],
               check=True, capture_output=True)
subprocess.run(["xcrun", "-sdk", "macosx", "metallib", hw_air, "-o", hw_metallib],
               check=True, capture_output=True)
from enigma.compiler.compiler import CompiledKernel
from pathlib import Path

hw_compiled = CompiledKernel(
    kernel_name="rmsnorm",
    metallib_path=hw_metallib,
    metallib_bytes=Path(hw_metallib).read_bytes(),
    metal_source=open(hw_metal).read(),
)

np.random.seed(42)
X = np.random.randn(ROWS, N).astype(np.float32)
W = np.random.randn(N).astype(np.float32)
rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True) + EPS)
expected = (X / rms) * W

runtime = enigma.MetalRuntime()
print(f"\nRMSNorm: {ROWS}×{N} f32, eps={EPS}, threads={THREADS_PER_GROUP}")

enigma_out = np.frombuffer(
    runtime.execute(enigma_compiled, [X.ravel(), W], ROWS * N * 4,
                    grid=enigma_compiled.grid, threads=enigma_compiled.block),
    dtype=np.float32).reshape(ROWS, N)
err_e = np.max(np.abs(enigma_out - expected))
print(f"  Enigma:      max|err| = {err_e:.2e}  {'PASS' if err_e < 1e-3 else 'FAIL'}")

hw_out = np.frombuffer(
    runtime.execute(hw_compiled, [X.ravel(), W], ROWS * N * 4,
                    grid=(ROWS * THREADS_PER_GROUP, 1, 1),
                    threads=(THREADS_PER_GROUP, 1, 1)),
    dtype=np.float32).reshape(ROWS, N)
err_h = np.max(np.abs(hw_out - expected))
print(f"  Handwritten: max|err| = {err_h:.2e}  {'PASS' if err_h < 1e-3 else 'FAIL'}")

WARMUP, ITERS = 20, 200

def bench(name, compiled_k, inputs, grid, threads):
    prep = runtime.prepare(compiled_k, inputs, ROWS * N * 4)
    for _ in range(WARMUP):
        prep.dispatch(grid=grid, threads=threads)
    times = []
    for _ in range(ITERS):
        times.append(prep.dispatch_timed(grid=grid, threads=threads))
    prep.release()
    med = np.median(times)
    bw = (ROWS * N * 4 * 2 + N * 4 + ROWS * N * 4) / (med * 1e-6) / 1e9
    print(f"  {name:40s} {med:8.2f} us  {bw:6.1f} GB/s")
    return med

print(f"\n{'─' * 80}")
t_e = bench("Enigma DSL (layout algebra)", enigma_compiled,
            [X.ravel(), W], enigma_compiled.grid, enigma_compiled.block)
t_h = bench("Handwritten Metal (float4)", hw_compiled,
            [X.ravel(), W], (ROWS * THREADS_PER_GROUP, 1, 1), (THREADS_PER_GROUP, 1, 1))
print(f"{'─' * 80}")
print(f"  Enigma / Handwritten = {t_h / t_e:.2f}x")
