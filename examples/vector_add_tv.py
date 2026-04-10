#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide


@enigma.kernel
def add_kernel(gA, gB, gC, tv_layout, tiler):
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


@enigma.jit
def elementwise_add(mA, mB, mC):
    thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

    gA = tensor_zipped_divide(mA, tiler_mn)
    gB = tensor_zipped_divide(mB, tiler_mn)
    gC = tensor_zipped_divide(mC, tiler_mn)

    num_blocks = enigma.size(gA, mode=[1])
    threads = enigma.size(tv_layout, mode=[0])

    add_kernel(gA, gB, gC, tv_layout, tiler_mn).launch(
        grid=(num_blocks * threads, 1, 1),
        block=(threads, 1, 1),
    )


M, N = 256, 512

mA = Tensor("A", 0, "float", enigma.Layout((M, N), (N, 1)))
mB = Tensor("B", 1, "float", enigma.Layout((M, N), (N, 1)))
mC = Tensor("C", 2, "float", enigma.Layout((M, N), (N, 1)))

print("Compiling with @jit + @kernel...")
compiled = enigma.compile(elementwise_add, mA, mB, mC)

print(f"\nGenerated Metal source:")
print("=" * 60)
print(compiled.metal_source)
print("=" * 60)

A = np.random.randn(M, N).astype(np.float32)
B = np.random.randn(M, N).astype(np.float32)

runtime = enigma.MetalRuntime()
result_bytes = runtime.execute(
    compiled, inputs=[A.ravel(), B.ravel()], output_size=M * N * 4,
    grid=compiled.grid, threads=compiled.block,
)

result = np.frombuffer(result_bytes, dtype=np.float32).reshape(M, N)
expected = A + B
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-7)
print(f"\nPASSED: TV-layout element-wise add ({M}x{N})")
print(f"  max |error| = {np.max(np.abs(result - expected)):.2e}")
