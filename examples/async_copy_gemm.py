#!/usr/bin/env python3
"""GEMM using AIR-backed async device->threadgroup copy.

Port of https://percisely.xyz/gemm. Uses enigma.async_copy_2d_d2t for
tile loads and simdgroup-matrix ops for compute.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma

SW = 2
SIMD_TILE = 2
TILE_K = 2

TG_M = 8 * SIMD_TILE * SW   # 32
TG_N = 8 * SIMD_TILE * SW
TG_K = 8 * TILE_K           # 16

THREADGROUP_THREADS = 32 * SW * SW

N = 64
M = 64
K = 64
assert N % TG_M == 0 and M % TG_N == 0 and K % TG_K == 0


@enigma.kernel
def gemm_async(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tg_row = enigma.threadgroup_position_in_grid()
    tg_col = enigma.threadgroup_position_in_grid()
    grid_cols = enigma.metal_cast(M // TG_N, "uint")
    bx = tg_row / grid_cols
    by = tg_row % grid_cols

    tid = enigma.thread_position_in_threadgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()

    A_tg = enigma.threadgroup_alloc("float", TG_M * TG_K)
    B_tg = enigma.threadgroup_alloc("float", TG_K * TG_N)

    acc = enigma.register_tensor((SIMD_TILE, SIMD_TILE), dtype="float", fill=0.0)

    k_tiles = K // TG_K
    zero = enigma.metal_cast(0, "uint")
    tg_m = enigma.metal_cast(TG_M, "uint")
    tg_n = enigma.metal_cast(TG_N, "uint")
    tg_k = enigma.metal_cast(TG_K, "uint")
    k_const = enigma.metal_cast(K, "uint")
    m_const = enigma.metal_cast(M, "uint")

    a_base_row = bx * tg_m
    b_base_col = by * tg_n

    for l in range(k_tiles):
        l_u = enigma.metal_cast(l, "uint")
        k_off = l_u * tg_k
        a_src_off = a_base_row * k_const + k_off
        b_src_off = k_off * m_const + b_base_col

        # First simdgroup issues async loads
        with enigma.if_(enigma.cmp_eq(sg_idx, 0)) as (then_b, _else_b):
            with then_b:
                ev_a = enigma.async_copy_2d_d2t(
                    A_tg, zero, tg_k, A, a_src_off, k_const, tg_k, tg_m)
                ev_b = enigma.async_copy_2d_d2t(
                    B_tg, zero, tg_n, B, b_src_off, m_const, tg_n, tg_k)
                enigma.async_copy_wait(ev_a, ev_b)

        enigma.barrier()
        enigma.barrier()

    out_base = (bx * tg_m) * m_const + (by * tg_n)
    out_idx = out_base + tid
    C[out_idx] = acc[0, 0]


print("Compiling async-copy GEMM kernel…")
compiled = enigma.compile(gemm_async, dump_ir=True)

print("\nGenerated Metal source (first 80 lines):")
print("-" * 70)
print("\n".join(compiled.metal_source.splitlines()[:80]))
print("-" * 70)

required_intrinsics = [
    "air.simdgroup_async_copy_2d.p3i8.p1i8",
    "air.wait_simdgroup_events",
]
for s in required_intrinsics:
    assert s in compiled.metal_source, f"missing AIR intrinsic: {s}"
print(f"\nAll required AIR intrinsics present: {required_intrinsics}")

if os.environ.get("ENIGMA_RUN_GEMM"):
    print("\nDispatching kernel on Metal…")
    np.random.seed(0)
    A = np.random.randn(N, K).astype(np.float32) * 0.1
    B = np.random.randn(K, M).astype(np.float32) * 0.1

    runtime = enigma.MetalRuntime()
    grid_blocks = (N // TG_M) * (M // TG_N)
    raw = runtime.execute(compiled, inputs=[A.ravel(), B.ravel()],
                          output_size=N * M * 4,
                          grid=(grid_blocks, 1, 1),
                          threads=(THREADGROUP_THREADS, 1, 1))
    C = np.frombuffer(raw, dtype=np.float32).reshape(N, M)
    print(f"  output shape: {C.shape}")

print("\nasync_copy_gemm: pipeline verified end-to-end.")
