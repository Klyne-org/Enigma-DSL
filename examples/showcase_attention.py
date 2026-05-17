#!/usr/bin/env python3
"""FlashAttention forward pass — two implementations, benchmarked.

This example shows a fused FlashAttention forward kernel written in the
Enigma DSL, in two flavours:

  1) ``fa_naive``
        Raw @enigma.kernel.  We compute the launch grid ourselves
        (M threads in 1-D).  Straightforward reference implementation.

  2) ``fa_tv``
        Same kernel body, but launched through @enigma.jit + a TV
        (Thread-Value) layout.  The TV layout decides how query rows
        are partitioned across thread-blocks, so the host side does
        not hard-code grid / block dimensions.

Both kernels implement the standard FlashAttention-1 online-softmax
algorithm with an outer tile loop over K/V:

    m_i = -inf, l_i = 0, O_i = 0                   # running stats
    for tile t in 0..N_TILES:
        s_j      = (Q_i . K_j) / sqrt(D)          for j in tile
        m_new    = max(m_i, max_j s_j)
        alpha    = exp(m_i - m_new)               # rescale factor
        p_j      = exp(s_j - m_new)               # softmax numerator
        l_new    = alpha * l_i  +  sum_j p_j
        O_new    = alpha * O_i  +  sum_j p_j * V_j
        m_i, l_i, O_i = m_new, l_new, O_new
    O_i /= l_i

No S (attention scores) or P (softmax) tensors are ever materialised in
device memory — the whole thing is a single kernel with an scf.for loop
carrying (m, l, O) in registers.

At the end we run a warm-up + timed sweep on both kernels and report
GPU-side microseconds and an effective-TFLOPS number.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma
from enigma.tensor import Tensor, tensor_zipped_divide


runtime = enigma.MetalRuntime()


# =========================================================================
# Shapes
# -------------------------------------------------------------------------
# D is fixed to 4 so we can use ``enigma.make_float4`` + ``enigma.dot``
# for the Q.K and P.V inner products.  M and N can be anything divisible
# by BR / BC respectively.
#
# Layout of the three input buffers (all float32, row-major):
#     Q : [M, D]
#     K : [N, D]
#     V : [N, D]
# Output:
#     O : [M, D]
# =========================================================================
M = 1024  # number of query rows
N = 1024  # number of key / value rows
D = 4     # head dimension (must be 4: float4 path)
BC = 16   # KV tile width — 16 keys processed per outer iteration
BR = 32   # query rows per thread-block (only used by the TV launcher)
assert N % BC == 0, "N must be a multiple of BC"
assert M % BR == 0, "M must be a multiple of BR"
NT = N // BC  # number of outer (K/V) tiles per query row


# =========================================================================
# Shared FA kernel body
# -------------------------------------------------------------------------
# Both versions share the exact same kernel body.  The only difference is
# how each version chooses its (grid, block) dispatch shape.  We factor
# the body out into a helper so it is obvious that the math is identical.
#
# Per-thread contract: one thread owns one query row (row = tidx + bidx*bdim).
# For every outer tile of BC keys we:
#     (a) compute BC scaled scores  s_j = (Q_i . K_j) / sqrt(D)
#     (b) find the tile max, rescale the running (l, O) by alpha
#     (c) accumulate p_j and p_j * V_j into (l_new, O_new)
# The scf.for loop carries six scalars:   [m, l, O0, O1, O2, O3].
# =========================================================================
def _fa_body(Q, K, V, O, row):
    d = enigma.metal_cast(D, "uint")
    bc = enigma.metal_cast(BC, "uint")
    nt = enigma.metal_cast(NT, "int")

    # ---- Load the query row once into 4 scalars (D == 4). --------------
    # qb is the element offset of row `row` inside the flat Q buffer.
    qb = row * d
    q0 = Q[qb]
    q1 = Q[qb + 1]
    q2 = Q[qb + 2]
    q3 = Q[qb + 3]
    qvec = enigma.make_float4(q0, q1, q2, q3)

    # ---- Constants used inside the loop. -------------------------------
    # Softmax scaling factor 1 / sqrt(D).
    inv_sqrt_d = enigma.rsqrt(enigma.metal_cast(D, "float"))
    # Very negative number used as the initial running max (m_0 = -inf).
    neg_big = enigma.metal_cast(-1e30, "float")
    zero_f = enigma.metal_cast(0, "float")

    # ---- Outer loop: iterate over NT tiles of BC keys each. ------------
    # init=[m, l, O0, O1, O2, O3] is the online-softmax running state,
    # carried as iter_args across iterations of the scf.for.
    with enigma.for_range(
        0,
        nt,
        init=[neg_big, zero_f, zero_f, zero_f, zero_f, zero_f],
    ) as (t, c):
        # Byte/element offset of the first K (and V) element in this tile.
        tile_base = enigma.metal_cast(t, "uint") * bc * d

        # ---- Pass 1: compute all BC scaled scores for this tile. -------
        # Each score is a float4 dot between the cached qvec and one
        # key row K[tile_base + j*D : tile_base + j*D + D], scaled.
        def _score(j):
            kb = tile_base + enigma.metal_cast(j * D, "uint")
            kv = enigma.make_float4(K[kb], K[kb + 1], K[kb + 2], K[kb + 3])
            return enigma.dot(qvec, kv) * inv_sqrt_d

        scores = [_score(j) for j in range(BC)]

        # Tile-wise maximum via a tree of fmax (BC = 16 → 15 fmax calls).
        m_tile = scores[0]
        for s in scores[1:]:
            m_tile = enigma.fmax(m_tile, s)

        # Global running max update.
        m_new = enigma.fmax(c[0], m_tile)
        # Rescale factor for prior (l, O): alpha = exp(m_old - m_new).
        alpha = enigma.exp(c[0] - m_new)

        # ---- Pass 2: exponentiate scores, accumulate p_sum and p.V. ----
        ps = [enigma.exp(s - m_new) for s in scores]

        # Sum of this tile's softmax numerators.
        p_sum = ps[0]
        for p in ps[1:]:
            p_sum = p_sum + p

        # Running normaliser: l_new = alpha * l_old + sum_j p_j.
        l_new = alpha * c[1] + p_sum

        # Running output: O_new = alpha * O_old + sum_j p_j * V_j.
        # Start by rescaling the prior accumulator...
        o0 = alpha * c[2]
        o1 = alpha * c[3]
        o2 = alpha * c[4]
        o3 = alpha * c[5]
        # ...then fma in each (p_j, V_j) pair.
        for j in range(BC):
            vb = tile_base + enigma.metal_cast(j * D, "uint")
            pj = ps[j]
            o0 = enigma.fma(pj, V[vb],     o0)
            o1 = enigma.fma(pj, V[vb + 1], o1)
            o2 = enigma.fma(pj, V[vb + 2], o2)
            o3 = enigma.fma(pj, V[vb + 3], o3)

        # Write the new running state back into the iter_args carry.
        c[0] = m_new
        c[1] = l_new
        c[2] = o0
        c[3] = o1
        c[4] = o2
        c[5] = o3

    # ---- Final normalisation: O_i /= l_i, then store the output row. --
    inv_l = enigma.metal_cast(1, "float") / c[1]
    ob = row * d
    O[ob]     = c[2] * inv_l
    O[ob + 1] = c[3] * inv_l
    O[ob + 2] = c[4] * inv_l
    O[ob + 3] = c[5] * inv_l


# =========================================================================
# Version 1 — naive launcher: raw @enigma.kernel, host picks grid/block.
# -------------------------------------------------------------------------
# One thread per query row.  Flat 1-D grid of M threads.  The host side
# chooses a block size (BR) manually.
# =========================================================================
@enigma.kernel
def fa_naive(Q: enigma.f32, K: enigma.f32, V: enigma.f32, O: enigma.f32):
    # thread_position_in_grid gives us the global thread id which, for a
    # 1-D dispatch, is exactly the query-row index.
    row = enigma.thread_position_in_grid
    _fa_body(Q, K, V, O, row)


fa_naive_compiled = enigma.compile(fa_naive)


# =========================================================================
# Version 2 — TV-layout launcher.
# -------------------------------------------------------------------------
# Identical kernel body.  We go through @enigma.jit: the JIT wrapper uses
# a Thread-Value layout to decide how to partition the [M, D] query tile
# across thread-blocks.  The layout computes (num_blocks, threads_per_block)
# for us instead of us hard-coding them.
#
#     thr_layout = (BR, 1)       → BR threads per block along the row dim
#     val_layout = (1,  D)       → each thread handles 1 row, all D cols
#
# The TV wrapper also walks the Q tensor with ``tensor_zipped_divide`` to
# figure out how many row-blocks exist — this is the pattern used by
# examples/benchmark_naive_vs_tv.py for element-wise kernels.
# =========================================================================
@enigma.kernel
def fa_tv_kernel(Q, K, V, O):
    # When invoked through @enigma.jit, Q/K/V/O arrive as layout-aware
    # ``Tensor`` objects (not the flat TracingTensor the naive path uses),
    # so we re-wrap them as scalar-indexable proxies for the FA body.
    # This keeps the body helper unchanged between the two versions.
    from enigma._tracing import TracingTensor
    Qt = TracingTensor(Q.name, Q.buffer_index, Q.metal_dtype)
    Kt = TracingTensor(K.name, K.buffer_index, K.metal_dtype)
    Vt = TracingTensor(V.name, V.buffer_index, V.metal_dtype)
    Ot = TracingTensor(O.name, O.buffer_index, O.metal_dtype)

    # Reconstruct the global query-row index from block / thread IDs.
    # This is equivalent to `thread_position_in_grid` for our 1-D launch,
    # but expressed explicitly so the TV launch layer can pick its own
    # (grid, block) decomposition.
    tidx, _, _ = enigma.arch.thread_idx()
    bidx, _, _ = enigma.arch.block_idx()
    bdim, _, _ = enigma.arch.block_dim()
    row = bidx * bdim + tidx
    _fa_body(Qt, Kt, Vt, Ot, row)


@enigma.jit
def fa_tv(mQ, mK, mV, mO):
    # Build a TV layout that says "one thread per query row, each thread
    # owns all D columns of that row".  We only use it to derive
    # (num_blocks, threads_per_block) — the kernel itself does raw
    # buffer indexing because the FA math is inherently scalar.
    thr_layout = enigma.make_ordered_layout((BR, 1), order=(1, 0))
    val_layout = enigma.make_ordered_layout((1, D), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

    # zipped_divide partitions Q's [M, D] shape by the tiler so that
    # gQ.shape[1] (the block-count dimension) tells us the number of
    # row-blocks we need to dispatch.
    gQ = tensor_zipped_divide(mQ, tiler_mn)
    num_blocks = enigma.size(gQ, mode=[1])
    threads = enigma.size(tv_layout, mode=[0])

    fa_tv_kernel(mQ, mK, mV, mO).launch(
        grid=(num_blocks * threads, 1, 1),
        block=(threads, 1, 1),
    )


# Declare the buffer layouts for the JIT compiler's symbolic trace.
mQ = Tensor("Q", 0, "float", enigma.Layout((M, D), (D, 1)))
mK = Tensor("K", 1, "float", enigma.Layout((N, D), (D, 1)))
mV = Tensor("V", 2, "float", enigma.Layout((N, D), (D, 1)))
mO = Tensor("O", 3, "float", enigma.Layout((M, D), (D, 1)))
fa_tv_compiled = enigma.compile(fa_tv, mQ, mK, mV, mO)


# =========================================================================
# Correctness check
# -------------------------------------------------------------------------
# Generate a random (Q, K, V), run both kernels, and compare against a
# NumPy reference softmax(Q K^T / sqrt(D)) @ V.
# =========================================================================
print("=" * 70)
print(f"FlashAttention forward   —   M={M}  N={N}  D={D}  BC={BC}  BR={BR}")
print("=" * 70)

np.random.seed(1234)
Qh = np.random.randn(M, D).astype(np.float32)
Kh = np.random.randn(N, D).astype(np.float32)
Vh = np.random.randn(N, D).astype(np.float32)

# NumPy reference for correctness.
S_ref = (Qh @ Kh.T) / np.sqrt(D)
P_ref = np.exp(S_ref - S_ref.max(axis=1, keepdims=True))
P_ref = P_ref / P_ref.sum(axis=1, keepdims=True)
O_ref = P_ref @ Vh

# --- naive run ---
raw_naive = runtime.execute(
    fa_naive_compiled,
    [Qh.ravel(), Kh.ravel(), Vh.ravel()],
    M * D * 4,
    grid=(M, 1, 1),
    threads=(min(BR, 32), 1, 1),
)
O_naive = np.frombuffer(raw_naive, dtype=np.float32).copy().reshape(M, D)
err_naive = np.max(np.abs(O_naive - O_ref))
print(f"  naive kernel :  max |err| = {err_naive:.2e}   "
      f"{'OK' if err_naive < 1e-3 else 'FAIL'}")

# --- TV run ---
raw_tv = runtime.execute(
    fa_tv_compiled,
    [Qh.ravel(), Kh.ravel(), Vh.ravel()],
    M * D * 4,
    grid=fa_tv_compiled.grid,
    threads=fa_tv_compiled.block,
)
O_tv = np.frombuffer(raw_tv, dtype=np.float32).copy().reshape(M, D)
err_tv = np.max(np.abs(O_tv - O_ref))
print(f"  TV    kernel :  max |err| = {err_tv:.2e}   "
      f"{'OK' if err_tv < 1e-3 else 'FAIL'}")

assert err_naive < 1e-3, "naive FA kernel is numerically broken"
assert err_tv    < 1e-3, "TV FA kernel is numerically broken"


# =========================================================================
# Benchmark
# -------------------------------------------------------------------------
# FLOP accounting for standard attention:
#     QK^T :  M * N * D  multiplies  +  M * N * (D - 1)  adds   ≈ 2 * M * N * D
#     softmax (exp + divide)         : dominated by N * M  exps + N*M adds
#     P  V :  M * D * N  multiplies  +  M * D * (N - 1)  adds   ≈ 2 * M * N * D
# For large D the 4*M*N*D term dominates.  FlashAttention does exactly
# the same number of useful FLOPs (it just saves *memory traffic*), so
# this is the number we report.
# =========================================================================
FLOPS_PER_RUN = 4.0 * M * N * D  # QK^T + PV dominate
WARMUP = 50
ITERS = 500


def bench(label, compiled, grid, threads):
    """Prepare + warmup + time a kernel.  Returns (min_us, med_us)."""
    prep = runtime.prepare(
        compiled,
        [Qh.ravel(), Kh.ravel(), Vh.ravel()],
        M * D * 4,
    )
    # Warm-up: ramp GPU clocks so the timed window isn't dominated by
    # frequency transitions.
    for _ in range(WARMUP):
        prep.dispatch(grid=grid, threads=threads)

    times_us = []
    for _ in range(ITERS):
        times_us.append(prep.dispatch_timed(grid=grid, threads=threads))

    prep.release()
    arr = np.asarray(times_us)
    return float(arr.min()), float(np.median(arr))


print()
print(f"Benchmark   —   warmup={WARMUP}   iters={ITERS}")
print(f"FLOPs per call ≈ {FLOPS_PER_RUN:,.0f}   (4 * M * N * D)")
print("-" * 70)

results = []
for name, compiled, grid, threads in [
    ("naive  (f32, 1-D dispatch)",
        fa_naive_compiled, (M, 1, 1), (min(BR, 32), 1, 1)),
    ("TV     (layout-driven grid)",
        fa_tv_compiled, fa_tv_compiled.grid, fa_tv_compiled.block),
]:
    mn_us, med_us = bench(name, compiled, grid, threads)
    # GPU timestamps are microseconds → TFLOPS = FLOPs / (time_s * 1e12)
    tflops_min = FLOPS_PER_RUN / (mn_us * 1e-6) / 1e12
    tflops_med = FLOPS_PER_RUN / (med_us * 1e-6) / 1e12
    print(f"  {name:32s}  min {mn_us:8.2f} us ({tflops_min:5.3f} TFLOP/s)   "
          f"med {med_us:8.2f} us ({tflops_med:5.3f} TFLOP/s)")
    results.append((name, mn_us, tflops_min))

print("-" * 70)

# Relative speedup (based on min times).
t_naive = results[0][1]
t_tv = results[1][1]
print(f"  TV vs naive speedup: {t_naive / t_tv:.2f}x   "
      f"(min-time basis)")

print()
print("All kernels correct.   FA forward pass fused (no S/P in global mem),")
print("online-softmax state carried via scf.for iter_args.")
