#!/usr/bin/env python3
"""FlashAttention forward — naive vs TV-layout launcher, benchmarked.

Online-softmax FA-1: no S/P materialised, (m, l, O) carried in scf.for iter_args.
D=4 uses float4 dot path. Both versions share the same kernel body.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma
from enigma.tensor import Tensor, tensor_zipped_divide


runtime = enigma.MetalRuntime()

M = 1024
N = 1024
D = 4
BC = 16
BR = 32
assert N % BC == 0 and M % BR == 0
NT = N // BC


def _fa_body(Q, K, V, O, row):
    d = enigma.metal_cast(D, "uint")
    bc = enigma.metal_cast(BC, "uint")
    nt = enigma.metal_cast(NT, "int")

    qb = row * d
    qvec = enigma.make_float4(Q[qb], Q[qb + 1], Q[qb + 2], Q[qb + 3])

    inv_sqrt_d = enigma.rsqrt(enigma.metal_cast(D, "float"))
    neg_big = enigma.metal_cast(-1e30, "float")
    zero_f = enigma.metal_cast(0, "float")

    with enigma.for_range(0, nt,
            init=[neg_big, zero_f, zero_f, zero_f, zero_f, zero_f]) as (t, c):
        tile_base = enigma.metal_cast(t, "uint") * bc * d

        def _score(j):
            kb = tile_base + enigma.metal_cast(j * D, "uint")
            kv = enigma.make_float4(K[kb], K[kb + 1], K[kb + 2], K[kb + 3])
            return enigma.dot(qvec, kv) * inv_sqrt_d

        scores = [_score(j) for j in range(BC)]

        m_tile = scores[0]
        for s in scores[1:]:
            m_tile = enigma.fmax(m_tile, s)

        m_new = enigma.fmax(c[0], m_tile)
        alpha = enigma.exp(c[0] - m_new)

        ps = [enigma.exp(s - m_new) for s in scores]
        p_sum = ps[0]
        for p in ps[1:]:
            p_sum = p_sum + p

        l_new = alpha * c[1] + p_sum
        o0 = alpha * c[2]
        o1 = alpha * c[3]
        o2 = alpha * c[4]
        o3 = alpha * c[5]
        for j in range(BC):
            vb = tile_base + enigma.metal_cast(j * D, "uint")
            pj = ps[j]
            o0 = enigma.fma(pj, V[vb],     o0)
            o1 = enigma.fma(pj, V[vb + 1], o1)
            o2 = enigma.fma(pj, V[vb + 2], o2)
            o3 = enigma.fma(pj, V[vb + 3], o3)

        c[0] = m_new
        c[1] = l_new
        c[2] = o0
        c[3] = o1
        c[4] = o2
        c[5] = o3

    inv_l = enigma.metal_cast(1, "float") / c[1]
    ob = row * d
    O[ob]     = c[2] * inv_l
    O[ob + 1] = c[3] * inv_l
    O[ob + 2] = c[4] * inv_l
    O[ob + 3] = c[5] * inv_l


@enigma.kernel
def fa_naive(Q: enigma.f32, K: enigma.f32, V: enigma.f32, O: enigma.f32):
    row = enigma.thread_position_in_grid
    _fa_body(Q, K, V, O, row)


fa_naive_compiled = enigma.compile(fa_naive)


@enigma.kernel
def fa_tv_kernel(Q, K, V, O):
    from enigma._tracing import Tensor
    Qt = Tensor(Q.name, Q.buffer_index, Q.metal_dtype)
    Kt = Tensor(K.name, K.buffer_index, K.metal_dtype)
    Vt = Tensor(V.name, V.buffer_index, V.metal_dtype)
    Ot = Tensor(O.name, O.buffer_index, O.metal_dtype)

    tidx, _, _ = enigma.arch.thread_idx()
    bidx, _, _ = enigma.arch.block_idx()
    bdim, _, _ = enigma.arch.block_dim()
    row = bidx * bdim + tidx
    _fa_body(Qt, Kt, Vt, Ot, row)


@enigma.jit
def fa_tv(mQ, mK, mV, mO):
    thr_layout = enigma.make_ordered_layout((BR, 1), order=(1, 0))
    val_layout = enigma.make_ordered_layout((1, D), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

    gQ = tensor_zipped_divide(mQ, tiler_mn)
    num_blocks = enigma.size(gQ, mode=[1])
    threads = enigma.size(tv_layout, mode=[0])

    fa_tv_kernel(mQ, mK, mV, mO).launch(
        grid=(num_blocks * threads, 1, 1),
        block=(threads, 1, 1),
    )


mQ = Tensor("Q", 0, "float", enigma.Layout((M, D), (D, 1)))
mK = Tensor("K", 1, "float", enigma.Layout((N, D), (D, 1)))
mV = Tensor("V", 2, "float", enigma.Layout((N, D), (D, 1)))
mO = Tensor("O", 3, "float", enigma.Layout((M, D), (D, 1)))
fa_tv_compiled = enigma.compile(fa_tv, mQ, mK, mV, mO)

# -- Correctness --
print("=" * 70)
print(f"FlashAttention forward   —   M={M}  N={N}  D={D}  BC={BC}  BR={BR}")
print("=" * 70)

np.random.seed(1234)
Qh = np.random.randn(M, D).astype(np.float32)
Kh = np.random.randn(N, D).astype(np.float32)
Vh = np.random.randn(N, D).astype(np.float32)

S_ref = (Qh @ Kh.T) / np.sqrt(D)
P_ref = np.exp(S_ref - S_ref.max(axis=1, keepdims=True))
P_ref = P_ref / P_ref.sum(axis=1, keepdims=True)
O_ref = P_ref @ Vh

raw_naive = runtime.execute(fa_naive_compiled, [Qh.ravel(), Kh.ravel(), Vh.ravel()],
                            M * D * 4, grid=(M, 1, 1), threads=(min(BR, 32), 1, 1))
O_naive = np.frombuffer(raw_naive, dtype=np.float32).copy().reshape(M, D)
err_naive = np.max(np.abs(O_naive - O_ref))
print(f"  naive kernel :  max |err| = {err_naive:.2e}   "
      f"{'OK' if err_naive < 1e-3 else 'FAIL'}")

raw_tv = runtime.execute(fa_tv_compiled, [Qh.ravel(), Kh.ravel(), Vh.ravel()],
                         M * D * 4, grid=fa_tv_compiled.grid, threads=fa_tv_compiled.block)
O_tv = np.frombuffer(raw_tv, dtype=np.float32).copy().reshape(M, D)
err_tv = np.max(np.abs(O_tv - O_ref))
print(f"  TV    kernel :  max |err| = {err_tv:.2e}   "
      f"{'OK' if err_tv < 1e-3 else 'FAIL'}")

assert err_naive < 1e-3
assert err_tv < 1e-3

# -- Benchmark --
FLOPS_PER_RUN = 4.0 * M * N * D
WARMUP = 50
ITERS = 500


def bench(label, compiled, grid, threads):
    prep = runtime.prepare(compiled, [Qh.ravel(), Kh.ravel(), Vh.ravel()], M * D * 4)
    for _ in range(WARMUP):
        prep.dispatch(grid=grid, threads=threads)
    times_us = []
    for _ in range(ITERS):
        times_us.append(prep.dispatch_timed(grid=grid, threads=threads))
    prep.release()
    arr = np.asarray(times_us)
    return float(arr.min()), float(np.median(arr))


print(f"\nBenchmark   —   warmup={WARMUP}   iters={ITERS}")
print("-" * 70)

results = []
for name, compiled, grid, threads in [
    ("naive  (f32, 1-D dispatch)", fa_naive_compiled, (M, 1, 1), (min(BR, 32), 1, 1)),
    ("TV     (layout-driven grid)", fa_tv_compiled, fa_tv_compiled.grid, fa_tv_compiled.block),
]:
    mn_us, med_us = bench(name, compiled, grid, threads)
    tflops_min = FLOPS_PER_RUN / (mn_us * 1e-6) / 1e12
    tflops_med = FLOPS_PER_RUN / (med_us * 1e-6) / 1e12
    print(f"  {name:32s}  min {mn_us:8.2f} us ({tflops_min:5.3f} TFLOP/s)   "
          f"med {med_us:8.2f} us ({tflops_med:5.3f} TFLOP/s)")
    results.append((name, mn_us))

print("-" * 70)
print(f"  TV vs naive speedup: {results[0][1] / results[1][1]:.2f}x")
