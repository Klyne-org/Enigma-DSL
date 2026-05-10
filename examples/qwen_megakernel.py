#!/usr/bin/env python3
"""Qwen-style decode "megakernel" — Enigma DSL port.

Reference:
    https://github.com/ighoshsubho/metal-shader-kernels/blob/main/metal/qwen_megakernel.metal

The reference Metal kernel fuses a *full* Qwen3-0.6B decode step into a single
persistent dispatch using grid-wide barriers (a seq_cst fence + relaxed atomic
counter) to march all threadgroups through phases:

    Phase 1: RMSNorm(input) + fused QKV-proj           (bf16 weights)
    Phase 2: per-head RMSNorm + RoPE + KV-cache write
    Phase 3: online-softmax SDPA  (grouped-query)
    Phase 4: O-proj + residual
    Phase 5: post-RMSNorm + SwiGLU MLP (gate, up fused)
    Phase 6: down-proj + residual
    Final  : RMSNorm

Enigma does not expose a grid-wide barrier primitive (`grid_sync`) — that
would need persistent CTAs and a seq-cst fence on `atomic_uint`, neither of
which is wired through the dialect today. So this port keeps the *math*
identical, including the exact RMSNorm tree-reduction, online-softmax
recurrence, and SwiGLU formulation, but dispatches each phase as its own
kernel. The CPU-side launch loop plays the role of `grid_sync`.

Everything else maps 1-to-1 to the megakernel:

    bf16_dot_tg              →  threadgroup-reduction matvec via simd_sum
    rmsnorm_f32 / _bf16      →  identical sum-of-squares + rsqrt + scale
    head_norm_rope           →  per-head RMSNorm + interleaved-half RoPE
    online softmax           →  m / l / O recurrence with rescale
    SwiGLU                   →  silu(gate) * up

Shapes match Qwen3-0.6B: H=1024, INTER=3072, QH=16, KVH=8, HD=128.
Weights are kept in float32 here (the DSL's elementwise pipeline is f32-clean
for the matvec; bf16 quantisation is orthogonal to what we're showcasing).

Run:
    python examples/qwen_megakernel.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma


# =========================================================================
# Qwen3-0.6B-ish shape constants (matched to the reference kernel)
# -------------------------------------------------------------------------
# We keep H, INTER, HD identical.  Head counts (QH/KVH) and cache length
# are kept modest so the single-step decode runs comfortably on any M-series
# GPU and the NumPy reference stays fast.
# =========================================================================
H = 1024                  # hidden dim
INTER = 3072              # MLP intermediate dim
QH = 16                   # number of query heads
KVH = 8                   # number of KV heads
HD = H // QH              # head dim — 64 here (Qwen3-0.6B uses 128, same path)
assert H == QH * HD
GQA = QH // KVH           # grouped-query ratio
QSZ = QH * HD             # = H
KVSZ = KVH * HD
CACHE_LEN = 32            # number of past tokens already in KV cache
RMS_EPS = 1e-6
THREADS = 128             # threadgroup size for the reduction kernels


# =========================================================================
# Kernel 1 — RMSNorm  (one threadgroup per row, simd-sum reduction)
# -------------------------------------------------------------------------
# Direct DSL transcription of `rmsnorm_f32` from the megakernel:
#
#     sq = sum_i x_i * x_i
#     rstd = rsqrt(sq / H + eps)
#     y_i = (x_i * rstd) * w_i
#
# Launch:   grid = (H, 1, 1),   threads = (THREADS, 1, 1)   per row
# We launch one threadgroup; the host loops if there were multiple rows.
# =========================================================================
@enigma.kernel
def rmsnorm_kernel(X: enigma.f32, W: enigma.f32, Y: enigma.f32):
    tid = enigma.thread_position_in_threadgroup()
    sg_lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    n_sg = enigma.simdgroups_per_threadgroup()

    elems_per_thread = H // THREADS  # = 8 for H=1024, THREADS=128

    # ---- Pass 1: sum of squares (per-thread partial). -------------------
    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, elems_per_thread, init=[zero]) as (j, c):
        idx = tid + enigma.metal_cast(j, "uint") * enigma.metal_cast(THREADS, "uint")
        v = X[idx]
        c[0] = c[0] + v * v

    # Simd-level reduction.
    sg_part = enigma.simd_sum(c[0])

    # Cross-simdgroup reduction via threadgroup memory (1 slot per simdgroup).
    scratch = enigma.threadgroup_alloc("float", 32)
    is_first_lane = enigma.cmp_eq(sg_lane, 0)
    enigma.store_if(scratch, sg_idx, sg_part, is_first_lane)
    enigma.barrier()

    in_range = enigma.cmp_ult(sg_lane, n_sg)
    v_lane = enigma.load_if(scratch, sg_lane, in_range, default=0.0)
    total = enigma.simd_sum(v_lane)

    rstd = enigma.rsqrt(total / float(H) + RMS_EPS)

    # ---- Pass 2: scale and apply weight. --------------------------------
    with enigma.for_range(0, elems_per_thread) as j:
        idx = tid + enigma.metal_cast(j, "uint") * enigma.metal_cast(THREADS, "uint")
        Y[idx] = (X[idx] * rstd) * W[idx]


# =========================================================================
# Kernel 2 — Q / K / V projection (one kernel per matrix)
# -------------------------------------------------------------------------
# Reference: each output row is a dot of one weight row with the normalised
# hidden vector.  In the megakernel this is `bf16_dot_tg` — one simdgroup
# per output row, fused into a single dispatch.  Enigma's scf.if doesn't
# expose multi-arm "else if", so for legibility we keep the math identical
# but launch three Q/K/V dispatches; each is a flat matvec.
#
# Layout (row-major):
#     Wq : [QSZ,  H]      Wk : [KVSZ, H]      Wv : [KVSZ, H]
#     X  : [H]
# =========================================================================
@enigma.kernel
def matvec_kernel(X: enigma.f32, W: enigma.f32, Out: enigma.f32):
    row = enigma.thread_position_in_grid
    h = enigma.metal_cast(H, "uint")
    base = row * h
    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, H, init=[zero]) as (i, c):
        ui = enigma.metal_cast(i, "uint")
        c[0] = enigma.fma(W[base + ui], X[ui], c[0])
    Out[row] = c[0]


# =========================================================================
# Kernel 3 — per-head RMSNorm + RoPE  (interleaved-half rotary)
# -------------------------------------------------------------------------
# `head_norm_rope` from the megakernel, but expressed with one threadgroup
# per head, HD threads per group.  RoPE partner is at i + HD/2 (or i - HD/2
# for the upper half), so each thread can load itself + its partner from
# threadgroup memory without simd shuffles.
#
# Launch:  grid = (n_heads, 1, 1),  threads = (HD, 1, 1)
#
# `data` is contiguous: head `h` lives at data[h*HD : (h+1)*HD].
# =========================================================================
@enigma.kernel
def head_rmsnorm_rope_kernel(data_in: enigma.f32, w: enigma.f32,
                              cos_pos: enigma.f32, sin_pos: enigma.f32,
                              data_out: enigma.f32):
    head_idx = enigma.threadgroup_position_in_grid()
    tid = enigma.thread_position_in_threadgroup()
    sg_lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    n_sg = enigma.simdgroups_per_threadgroup()

    half = enigma.metal_cast(HD // 2, "uint")
    base = head_idx * enigma.metal_cast(HD, "uint")

    # RMSNorm over the head: sum of squares then rsqrt.
    x = data_in[base + tid]
    sq = x * x
    sg_part = enigma.simd_sum(sq)
    scratch = enigma.threadgroup_alloc("float", 32)
    is_first_lane = enigma.cmp_eq(sg_lane, 0)
    enigma.store_if(scratch, sg_idx, sg_part, is_first_lane)
    enigma.barrier()
    in_range = enigma.cmp_ult(sg_lane, n_sg)
    v_lane = enigma.load_if(scratch, sg_lane, in_range, default=0.0)
    total = enigma.simd_sum(v_lane)
    sc = enigma.rsqrt(total / float(HD) + RMS_EPS)
    x_n = x * sc * w[tid]

    # Publish the normalised value so the partner lookup sees it.
    tile = enigma.threadgroup_alloc("float", HD)
    tile[tid] = x_n
    enigma.barrier()

    # RoPE — interleaved-half:
    #   if i <  HD/2:  out = x_n * cos  -  partner * sin
    #   else        :  out = partner * sin  +  x_n * cos
    # The partner index is (i + HD/2) mod HD.
    cv = cos_pos[tid]
    sv = sin_pos[tid]
    with enigma.if_(enigma.cmp_ult(tid, half)) as (then_b, else_b):
        with then_b:
            partner = tile[tid + half]
            data_out[base + tid] = x_n * cv - partner * sv
        with else_b:
            partner = tile[tid - half]
            data_out[base + tid] = partner * sv + x_n * cv


# =========================================================================
# Kernel 4 — online-softmax SDPA (grouped-query)
# -------------------------------------------------------------------------
# Reproduces the SDPA phase of the megakernel: a numerically-stable
# online softmax over the past `cache_len` keys, then weighted sum of V.
# One threadgroup per Q-head.
#
# Per-thread contract: one thread owns one element of the output head
# (tid in [0, HD)).  All HD threads cooperate to:
#   - load q (one element per thread)
#   - loop over cache positions:
#       partial qk = q_local * k_local
#       reduce qk across the head via simd_sum + threadgroup memory
#       broadcast the scaled score
#       update (m, l) and rescale o_local
#       fma p * v_local into o_local
#   - write o_local / l_global
#
# Layout:
#     Q : [QH,  HD]
#     K : [KVH, cache_len_total, HD]      (cache_len_total = CACHE_LEN + 1)
#     V : [KVH, cache_len_total, HD]
#     O : [QH,  HD]
# K and V already have the *current* token written into slot
# `position = CACHE_LEN` by a preceding step on the host.
# =========================================================================
CACHE_TOTAL = CACHE_LEN + 1  # past + this token
ATTN_SCALE = 1.0 / float(HD) ** 0.5


@enigma.kernel
def sdpa_kernel(Q: enigma.f32, K: enigma.f32, V: enigma.f32, O: enigma.f32):
    qh = enigma.threadgroup_position_in_grid()
    tid = enigma.thread_position_in_threadgroup()
    sg_lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    n_sg = enigma.simdgroups_per_threadgroup()

    hd = enigma.metal_cast(HD, "uint")
    kvh_stride = enigma.metal_cast(CACHE_TOTAL * HD, "uint")
    gqa = enigma.metal_cast(GQA, "uint")

    # Which KV head this Q head reads from.
    kvh = qh / gqa

    # My element of q (one float per thread, broadcast across head).
    q_elem = Q[qh * hd + tid]

    # Online-softmax running state per thread (each thread owns one o
    # element, but m / l are scalars duplicated across the head — we keep
    # them as per-thread floats and only the broadcast value matters).
    neg_big = enigma.metal_cast(-1e30, "float")
    zero = enigma.metal_cast(0, "float")

    scratch = enigma.threadgroup_alloc("float", 32)   # for cross-simdgroup reduce
    bcast = enigma.threadgroup_alloc("float", 1)      # for score broadcast

    with enigma.for_range(0, CACHE_TOTAL, init=[neg_big, zero, zero]) as (pos, c):
        upos = enigma.metal_cast(pos, "uint")
        k_off = kvh * kvh_stride + upos * hd + tid
        v_off = k_off

        k_el = K[k_off]
        v_el = V[v_off]

        # qk dot across the head: simd_sum + cross-simdgroup reduce.
        prod = q_elem * k_el
        sg_part = enigma.simd_sum(prod)
        is_first_lane = enigma.cmp_eq(sg_lane, 0)
        enigma.store_if(scratch, sg_idx, sg_part, is_first_lane)
        enigma.barrier()
        in_range = enigma.cmp_ult(sg_lane, n_sg)
        lane_v = enigma.load_if(scratch, sg_lane, in_range, default=0.0)
        qk_full = enigma.simd_sum(lane_v)

        # Tid 0 publishes the scaled score; everyone else picks it up.
        scaled = qk_full * ATTN_SCALE
        is_tid0 = enigma.cmp_eq(tid, 0)
        enigma.store_if(bcast, 0, scaled, is_tid0)
        enigma.barrier()
        score = bcast[0]

        # Online softmax update.
        m_old = c[0]
        m_new = enigma.fmax(m_old, score)
        rescale = enigma.exp(m_old - m_new)
        p = enigma.exp(score - m_new)
        l_new = c[1] * rescale + p
        o_new = c[2] * rescale + p * v_el

        c[0] = m_new
        c[1] = l_new
        c[2] = o_new
        enigma.barrier()

    inv_l = enigma.metal_cast(1, "float") / c[1]
    O[qh * hd + tid] = c[2] * inv_l


# =========================================================================
# Kernel 5 — O projection + residual
# -------------------------------------------------------------------------
# y[r] = sum_c W_o[r, c] * attn[c]  +  res[r]
# One thread per output row of size H, dot of length QSZ.
# =========================================================================
@enigma.kernel
def o_proj_kernel(attn: enigma.f32, Wo: enigma.f32, res: enigma.f32,
                  out: enigma.f32):
    row = enigma.thread_position_in_grid
    qsz_u = enigma.metal_cast(QSZ, "uint")
    base = row * qsz_u

    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, QSZ, init=[zero]) as (i, c):
        ui = enigma.metal_cast(i, "uint")
        c[0] = enigma.fma(Wo[base + ui], attn[ui], c[0])
    out[row] = c[0] + res[row]


# =========================================================================
# Kernel 6 — SwiGLU MLP, gate+up fused
# -------------------------------------------------------------------------
# Reference: `g_mlp[r] = (gate * sigmoid(gate)) * up`
# One thread per intermediate row.
# =========================================================================
@enigma.kernel
def swiglu_kernel(X: enigma.f32, Wg: enigma.f32, Wu: enigma.f32,
                  Out: enigma.f32):
    row = enigma.thread_position_in_grid
    h = enigma.metal_cast(H, "uint")
    base = row * h

    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, H, init=[zero, zero]) as (i, c):
        ui = enigma.metal_cast(i, "uint")
        xv = X[ui]
        c[0] = enigma.fma(Wg[base + ui], xv, c[0])
        c[1] = enigma.fma(Wu[base + ui], xv, c[1])
    gate = c[0]
    up = c[1]
    one = enigma.metal_cast(1, "float")
    sig = one / (one + enigma.exp(-gate))
    Out[row] = (gate * sig) * up


# =========================================================================
# Kernel 7 — down projection + residual
# -------------------------------------------------------------------------
# y[r] = sum_c W_d[r, c] * mlp[c]  +  res[r]
# =========================================================================
@enigma.kernel
def down_proj_kernel(mlp: enigma.f32, Wd: enigma.f32, res: enigma.f32,
                     out: enigma.f32):
    row = enigma.thread_position_in_grid
    inter_u = enigma.metal_cast(INTER, "uint")
    base = row * inter_u

    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, INTER, init=[zero]) as (i, c):
        ui = enigma.metal_cast(i, "uint")
        c[0] = enigma.fma(Wd[base + ui], mlp[ui], c[0])
    out[row] = c[0] + res[row]


# =========================================================================
# Compile everything
# =========================================================================
print("Compiling Enigma Qwen megakernel phases…")
rmsnorm_c   = enigma.compile(rmsnorm_kernel)
matvec_c    = enigma.compile(matvec_kernel)
rope_c      = enigma.compile(head_rmsnorm_rope_kernel)
sdpa_c      = enigma.compile(sdpa_kernel)
oproj_c     = enigma.compile(o_proj_kernel)
swiglu_c    = enigma.compile(swiglu_kernel)
downproj_c  = enigma.compile(down_proj_kernel)
print("  done.\n")


# =========================================================================
# Host harness — runs one decode step against a NumPy reference
# =========================================================================
runtime = enigma.MetalRuntime()


def rmsnorm_np(x, w, eps=RMS_EPS):
    rstd = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rstd * w


def rope_np(x, cos_p, sin_p):
    # x shape: [..., HD], interleaved-half rotary like the megakernel.
    half = HD // 2
    lo = x[..., :half]
    hi = x[..., half:]
    new_lo = lo * cos_p[:half] - hi * sin_p[:half]
    new_hi = hi * cos_p[half:] + lo * sin_p[half:]
    return np.concatenate([new_lo, new_hi], axis=-1)


def run_buf(compiled, inputs, output_size, grid, threads):
    raw = runtime.execute(compiled, inputs, output_size,
                          grid=grid, threads=threads)
    return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------- Build random weights + cache + input ------------------
np.random.seed(0)
x_in = np.random.randn(H).astype(np.float32) * 0.1
w_in_norm = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
w_post_norm = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
w_q_norm = np.random.randn(HD).astype(np.float32) * 0.1 + 1.0
w_k_norm = np.random.randn(HD).astype(np.float32) * 0.1 + 1.0

# Use small-scale weights so the activations stay tame.
Wq = (np.random.randn(QSZ, H) / np.sqrt(H)).astype(np.float32)
Wk = (np.random.randn(KVSZ, H) / np.sqrt(H)).astype(np.float32)
Wv = (np.random.randn(KVSZ, H) / np.sqrt(H)).astype(np.float32)
Wo = (np.random.randn(H, QSZ) / np.sqrt(QSZ)).astype(np.float32)
Wg = (np.random.randn(INTER, H) / np.sqrt(H)).astype(np.float32)
Wu = (np.random.randn(INTER, H) / np.sqrt(H)).astype(np.float32)
Wd = (np.random.randn(H, INTER) / np.sqrt(INTER)).astype(np.float32)

# Pretend RoPE tables — just sin/cos of fixed angles for our position.
angles = (np.arange(HD).astype(np.float32) / HD) * (np.pi / 2)
cos_pos = np.cos(angles).astype(np.float32)
sin_pos = np.sin(angles).astype(np.float32)

# KV cache: past CACHE_LEN tokens already encoded (random), current slot empty.
k_cache = (np.random.randn(KVH, CACHE_TOTAL, HD) * 0.1).astype(np.float32)
v_cache = (np.random.randn(KVH, CACHE_TOTAL, HD) * 0.1).astype(np.float32)

print("=" * 72)
print(f"Qwen single-layer decode   —   H={H}  INTER={INTER}  "
      f"QH={QH}  KVH={KVH}  HD={HD}  cache_len={CACHE_LEN}")
print("=" * 72)


# ---------------- Phase 1: RMSNorm ----------------
norm_out = run_buf(rmsnorm_c, [x_in, w_in_norm], H * 4,
                   grid=(THREADS, 1, 1), threads=(THREADS, 1, 1))
norm_ref = rmsnorm_np(x_in, w_in_norm)
err = np.max(np.abs(norm_out - norm_ref))
print(f"  [1] input RMSNorm        max|err| = {err:.2e}   "
      f"{'OK' if err < 1e-4 else 'FAIL'}")
assert err < 1e-4, "RMSNorm divergence"


# ---------------- Phase 2: Q / K / V projections ----------------
q = run_buf(matvec_c, [norm_out, Wq.ravel()], QSZ * 4,
            grid=(QSZ, 1, 1), threads=(min(QSZ, 256), 1, 1))
k = run_buf(matvec_c, [norm_out, Wk.ravel()], KVSZ * 4,
            grid=(KVSZ, 1, 1), threads=(min(KVSZ, 256), 1, 1))
v = run_buf(matvec_c, [norm_out, Wv.ravel()], KVSZ * 4,
            grid=(KVSZ, 1, 1), threads=(min(KVSZ, 256), 1, 1))
err = max(np.max(np.abs(q - Wq @ norm_out)),
          np.max(np.abs(k - Wk @ norm_out)),
          np.max(np.abs(v - Wv @ norm_out)))
print(f"  [2] Q / K / V proj       max|err| = {err:.2e}   "
      f"{'OK' if err < 5e-4 else 'FAIL'}")
assert err < 5e-4, "QKV divergence"


# ---------------- Phase 3: per-head RMSNorm + RoPE on Q and K -----------
# Launch one threadgroup per head (= HD threads each); grid passes *total*
# threads to the runtime, so KVH heads × HD threads = KVSZ total.
k_after = run_buf(rope_c, [k.copy(), w_k_norm, cos_pos, sin_pos],
                  KVSZ * 4,
                  grid=(KVSZ, 1, 1), threads=(HD, 1, 1))
# Reference.
k_view = k.reshape(KVH, HD)
k_norm_ref = rmsnorm_np(k_view, w_k_norm)
k_ref_rope = rope_np(k_norm_ref, cos_pos, sin_pos).ravel()
err = np.max(np.abs(k_after - k_ref_rope))
print(f"  [3a] K head-norm + RoPE  max|err| = {err:.2e}   "
      f"{'OK' if err < 5e-4 else 'FAIL'}")
assert err < 5e-4, "K RoPE divergence"

q_after = run_buf(rope_c, [q.copy(), w_q_norm, cos_pos, sin_pos],
                  QSZ * 4,
                  grid=(QSZ, 1, 1), threads=(HD, 1, 1))
q_view = q.reshape(QH, HD)
q_norm_ref = rmsnorm_np(q_view, w_q_norm)
q_ref_rope = rope_np(q_norm_ref, cos_pos, sin_pos).ravel()
err = np.max(np.abs(q_after - q_ref_rope))
print(f"  [3b] Q head-norm + RoPE  max|err| = {err:.2e}   "
      f"{'OK' if err < 5e-4 else 'FAIL'}")
assert err < 5e-4, "Q RoPE divergence"


# ---------------- Phase 4: write current K/V into cache + SDPA ---------
# Slot CACHE_LEN holds the freshly-projected token.
k_cache[:, CACHE_LEN, :] = k_after.reshape(KVH, HD)
v_cache[:, CACHE_LEN, :] = v.reshape(KVH, HD)

attn_out = run_buf(sdpa_c,
                   [q_after, k_cache.ravel(), v_cache.ravel()],
                   QSZ * 4,
                   grid=(QSZ, 1, 1), threads=(HD, 1, 1))

# NumPy reference for grouped-query attention.
attn_ref = np.empty((QH, HD), dtype=np.float32)
qmat = q_after.reshape(QH, HD)
for qh in range(QH):
    kvh = qh // GQA
    scores = (k_cache[kvh] @ qmat[qh]) * ATTN_SCALE      # [cache_total]
    m = scores.max()
    p = np.exp(scores - m)
    p = p / p.sum()
    attn_ref[qh] = p @ v_cache[kvh]
attn_ref_flat = attn_ref.ravel()
err = np.max(np.abs(attn_out - attn_ref_flat))
print(f"  [4] SDPA (online softmax) max|err| = {err:.2e}   "
      f"{'OK' if err < 1e-3 else 'FAIL'}")
assert err < 1e-3, "SDPA divergence"


# ---------------- Phase 5: O projection + residual ----------------
oproj_out = run_buf(oproj_c, [attn_out, Wo.ravel(), x_in], H * 4,
                    grid=(H, 1, 1), threads=(min(H, 256), 1, 1))
oproj_ref = Wo @ attn_out + x_in
err = np.max(np.abs(oproj_out - oproj_ref))
print(f"  [5] O-proj + residual    max|err| = {err:.2e}   "
      f"{'OK' if err < 5e-4 else 'FAIL'}")
assert err < 5e-4, "O-proj divergence"


# ---------------- Phase 6: post-RMSNorm + SwiGLU MLP ----------------
post_norm = run_buf(rmsnorm_c, [oproj_out, w_post_norm], H * 4,
                    grid=(THREADS, 1, 1), threads=(THREADS, 1, 1))
post_norm_ref = rmsnorm_np(oproj_out, w_post_norm)
err = np.max(np.abs(post_norm - post_norm_ref))
print(f"  [6a] post-RMSNorm        max|err| = {err:.2e}   "
      f"{'OK' if err < 1e-4 else 'FAIL'}")
assert err < 1e-4, "post-norm divergence"

mlp_out = run_buf(swiglu_c, [post_norm, Wg.ravel(), Wu.ravel()], INTER * 4,
                  grid=(INTER, 1, 1), threads=(min(INTER, 256), 1, 1))
gate_ref = Wg @ post_norm
up_ref = Wu @ post_norm
sig = 1.0 / (1.0 + np.exp(-gate_ref))
mlp_ref = (gate_ref * sig) * up_ref
err = np.max(np.abs(mlp_out - mlp_ref))
print(f"  [6b] SwiGLU (gate+up)    max|err| = {err:.2e}   "
      f"{'OK' if err < 1e-3 else 'FAIL'}")
assert err < 1e-3, "SwiGLU divergence"


# ---------------- Phase 7: down projection + residual ----------------
final_out = run_buf(downproj_c, [mlp_out, Wd.ravel(), oproj_out], H * 4,
                    grid=(H, 1, 1), threads=(min(H, 256), 1, 1))
final_ref = Wd @ mlp_out + oproj_out
err = np.max(np.abs(final_out - final_ref))
print(f"  [7] down-proj + residual max|err| = {err:.2e}   "
      f"{'OK' if err < 1e-3 else 'FAIL'}")
assert err < 1e-3, "down-proj divergence"


print()
print("All phases match the NumPy reference. One full Qwen-style decode")
print("step has been executed by Enigma-emitted Metal kernels.")
