#!/usr/bin/env python3
"""Qwen3-0.6B single-layer decode — fused single-dispatch megakernel.

All 7 phases (RMSNorm, QKV proj, head-norm+RoPE, SDPA, O-proj, SwiGLU,
down-proj) run in ONE kernel using plain helper functions that trace
inline via the thread-local builder. Single threadgroup, no grid_sync.

Reference: metal-shader-kernels/metal/qwen_megakernel.metal

Run:
    python examples/qwen_megakernel.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma

# Qwen3-0.6B shape constants
H = 1024
INTER = 3072
QH = 16
KVH = 8
HD = H // QH        # 64
GQA = QH // KVH     # 2
QSZ = QH * HD
KVSZ = KVH * HD     # 512
CACHE_LEN = 32
CACHE_TOTAL = CACHE_LEN + 1
RMS_EPS = 1e-6
ATTN_SCALE = 1.0 / float(HD) ** 0.5
TG = 256


# -- Helpers: plain functions, traced inline when called from @enigma.kernel --

def _rmsnorm(X, W, Y, dim, tid):
    sg_lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    n_sg = enigma.simdgroups_per_threadgroup()
    elems_per_thread = dim // TG

    zero = enigma.metal_cast(0, "float")
    with enigma.for_range(0, elems_per_thread, init=[zero]) as (j, c):
        idx = tid + enigma.metal_cast(j, "uint") * enigma.metal_cast(TG, "uint")
        v = X[idx]
        c[0] = c[0] + v * v

    sg_part = enigma.simd_sum(c[0])
    scratch = enigma.threadgroup_alloc("float", 32)
    is_first = enigma.cmp_eq(sg_lane, 0)
    enigma.store_if(scratch, sg_idx, sg_part, is_first)
    enigma.barrier()
    in_range = enigma.cmp_ult(sg_lane, n_sg)
    v_lane = enigma.load_if(scratch, sg_lane, in_range, default=0.0)
    total = enigma.simd_sum(v_lane)
    rstd = enigma.rsqrt(total / float(dim) + RMS_EPS)

    with enigma.for_range(0, elems_per_thread) as j:
        idx = tid + enigma.metal_cast(j, "uint") * enigma.metal_cast(TG, "uint")
        Y[idx] = (X[idx] * rstd) * W[idx]


def _matvec(X, W, Out, out_rows, in_cols, tid):
    h_u = enigma.metal_cast(in_cols, "uint")
    tg_u = enigma.metal_cast(TG, "uint")
    n_rows = (out_rows + TG - 1) // TG

    with enigma.for_range(0, n_rows) as chunk:
        row = tid + enigma.metal_cast(chunk, "uint") * tg_u
        in_bounds = enigma.cmp_ult(row, enigma.metal_cast(out_rows, "uint"))
        with enigma.if_(in_bounds) as (then_b, else_b):
            with then_b:
                base = row * h_u
                zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, in_cols, init=[zero]) as (i, c):
                    ui = enigma.metal_cast(i, "uint")
                    c[0] = enigma.fma(W[base + ui], X[ui], c[0])
                Out[row] = c[0]


def _head_norm_rope(data_in, w, cos_pos, sin_pos, data_out, n_heads, tid):
    # data_out MUST differ from data_in (aliasing breaks RoPE partner reads)
    hd_u = enigma.metal_cast(HD, "uint")
    half_val = HD // 2
    half_u = enigma.metal_cast(half_val, "uint")
    tg_u = enigma.metal_cast(TG, "uint")

    n_iters = (n_heads + TG - 1) // TG
    with enigma.for_range(0, n_iters) as chunk:
        head_idx = tid + enigma.metal_cast(chunk, "uint") * tg_u
        in_bounds = enigma.cmp_ult(head_idx, enigma.metal_cast(n_heads, "uint"))
        with enigma.if_(in_bounds) as (then_b, else_b):
            with then_b:
                base = head_idx * hd_u

                zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, HD, init=[zero]) as (k, c):
                    uk = enigma.metal_cast(k, "uint")
                    v = data_in[base + uk]
                    c[0] = c[0] + v * v

                sc = enigma.rsqrt(c[0] / float(HD) + RMS_EPS)

                # Normalise into data_out.
                with enigma.for_range(0, HD) as k:
                    uk = enigma.metal_cast(k, "uint")
                    data_out[base + uk] = data_in[base + uk] * sc * w[uk]

                # RoPE: process (lo, hi) pairs so both reads precede writes.
                with enigma.for_range(0, half_val) as k:
                    uk = enigma.metal_cast(k, "uint")
                    lo_idx = base + uk
                    hi_idx = base + uk + half_u
                    lo_val = data_out[lo_idx]
                    hi_val = data_out[hi_idx]
                    cv_lo = cos_pos[uk]
                    sv_lo = sin_pos[uk]
                    cv_hi = cos_pos[uk + half_u]
                    sv_hi = sin_pos[uk + half_u]
                    data_out[lo_idx] = lo_val * cv_lo - hi_val * sv_lo
                    data_out[hi_idx] = hi_val * cv_hi + lo_val * sv_hi


def _sdpa_serial(Q, K, V, O, tid):
    """One thread per query head, serial over cache positions."""
    hd_u = enigma.metal_cast(HD, "uint")
    gqa_u = enigma.metal_cast(GQA, "uint")
    kvh_stride = enigma.metal_cast(CACHE_TOTAL * HD, "uint")
    is_active = enigma.cmp_ult(tid, enigma.metal_cast(QH, "uint"))

    with enigma.if_(is_active) as (then_b, else_b):
        with then_b:
            qh = tid
            kvh = qh / gqa_u
            neg_big = enigma.metal_cast(-1e30, "float")
            zero = enigma.metal_cast(0, "float")

            # Pass 1: online softmax — find m and l.
            with enigma.for_range(0, CACHE_TOTAL, init=[neg_big, zero]) as (pos, c):
                upos = enigma.metal_cast(pos, "uint")
                dot_zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, HD, init=[dot_zero]) as (d, dc):
                    ud = enigma.metal_cast(d, "uint")
                    q_el = Q[qh * hd_u + ud]
                    k_off = kvh * kvh_stride + upos * hd_u + ud
                    dc[0] = enigma.fma(q_el, K[k_off], dc[0])

                score = dc[0] * ATTN_SCALE
                m_old = c[0]
                m_new = enigma.fmax(m_old, score)
                c[0] = m_new
                rescale = enigma.exp(m_old - m_new)
                p = enigma.exp(score - m_new)
                c[1] = c[1] * rescale + p

            # Pass 2: weighted V sum with final m, l.
            m_final = c[0]
            inv_l = enigma.metal_cast(1, "float") / c[1]

            with enigma.for_range(0, HD) as d:
                ud = enigma.metal_cast(d, "uint")
                acc_zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, CACHE_TOTAL, init=[acc_zero]) as (pos, vc):
                    upos = enigma.metal_cast(pos, "uint")
                    dot_z = enigma.metal_cast(0, "float")
                    with enigma.for_range(0, HD, init=[dot_z]) as (d2, dc2):
                        ud2 = enigma.metal_cast(d2, "uint")
                        q_el = Q[qh * hd_u + ud2]
                        k_off = kvh * kvh_stride + upos * hd_u + ud2
                        dc2[0] = enigma.fma(q_el, K[k_off], dc2[0])
                    score = dc2[0] * ATTN_SCALE
                    p = enigma.exp(score - m_final) * inv_l
                    v_off = kvh * kvh_stride + upos * hd_u + ud
                    vc[0] = vc[0] + p * V[v_off]

                O[qh * hd_u + ud] = vc[0]


def _matvec_add(X, W, res, Out, out_rows, in_cols, tid):
    h_u = enigma.metal_cast(in_cols, "uint")
    tg_u = enigma.metal_cast(TG, "uint")
    n_rows = (out_rows + TG - 1) // TG

    with enigma.for_range(0, n_rows) as chunk:
        row = tid + enigma.metal_cast(chunk, "uint") * tg_u
        in_bounds = enigma.cmp_ult(row, enigma.metal_cast(out_rows, "uint"))
        with enigma.if_(in_bounds) as (then_b, else_b):
            with then_b:
                base = row * h_u
                zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, in_cols, init=[zero]) as (i, c):
                    ui = enigma.metal_cast(i, "uint")
                    c[0] = enigma.fma(W[base + ui], X[ui], c[0])
                Out[row] = c[0] + res[row]


def _swiglu(X, Wg, Wu, Out, out_rows, in_cols, tid):
    h_u = enigma.metal_cast(in_cols, "uint")
    tg_u = enigma.metal_cast(TG, "uint")
    n_rows = (out_rows + TG - 1) // TG

    with enigma.for_range(0, n_rows) as chunk:
        row = tid + enigma.metal_cast(chunk, "uint") * tg_u
        in_bounds = enigma.cmp_ult(row, enigma.metal_cast(out_rows, "uint"))
        with enigma.if_(in_bounds) as (then_b, else_b):
            with then_b:
                base = row * h_u
                zero = enigma.metal_cast(0, "float")
                with enigma.for_range(0, in_cols, init=[zero, zero]) as (i, c):
                    ui = enigma.metal_cast(i, "uint")
                    xv = X[ui]
                    c[0] = enigma.fma(Wg[base + ui], xv, c[0])
                    c[1] = enigma.fma(Wu[base + ui], xv, c[1])
                gate = c[0]
                up = c[1]
                one = enigma.metal_cast(1, "float")
                sig = one / (one + enigma.exp(-gate))
                Out[row] = (gate * sig) * up


# -- The fused kernel: one dispatch, all phases --

@enigma.kernel
def qwen_fused(
    X: enigma.f32, w_norm: enigma.f32,
    Wq: enigma.f32, Wk: enigma.f32, Wv: enigma.f32,
    w_q_norm: enigma.f32, w_k_norm: enigma.f32,
    cos_pos: enigma.f32, sin_pos: enigma.f32,
    K_cache: enigma.f32, V_cache: enigma.f32,
    Wo: enigma.f32, w_post: enigma.f32,
    Wg: enigma.f32, Wu: enigma.f32, Wd: enigma.f32,
    norm_buf: enigma.f32, q_buf: enigma.f32,
    k_buf: enigma.f32, v_buf: enigma.f32,
    attn_buf: enigma.f32, mlp_buf: enigma.f32,
    post_buf: enigma.f32,
    q_rope_buf: enigma.f32, k_rope_buf: enigma.f32,
    Out: enigma.f32,
):
    tid = enigma.thread_position_in_threadgroup()

    _rmsnorm(X, w_norm, norm_buf, H, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _matvec(norm_buf, Wq, q_buf, QSZ, H, tid)
    _matvec(norm_buf, Wk, k_buf, KVSZ, H, tid)
    _matvec(norm_buf, Wv, v_buf, KVSZ, H, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _head_norm_rope(q_buf, w_q_norm, cos_pos, sin_pos, q_rope_buf, QH, tid)
    _head_norm_rope(k_buf, w_k_norm, cos_pos, sin_pos, k_rope_buf, KVH, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _sdpa_serial(q_rope_buf, K_cache, V_cache, attn_buf, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _matvec_add(attn_buf, Wo, X, norm_buf, H, QSZ, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _rmsnorm(norm_buf, w_post, post_buf, H, tid)
    enigma.barrier("mem_device_and_threadgroup")
    _swiglu(post_buf, Wg, Wu, mlp_buf, INTER, H, tid)
    enigma.barrier("mem_device_and_threadgroup")

    _matvec_add(mlp_buf, Wd, norm_buf, Out, H, INTER, tid)


# -- Compile --

print("Compiling fused megakernel…")
t0 = time.perf_counter()
fused_c = enigma.compile(qwen_fused)
t_compile = time.perf_counter() - t0
print(f"  compiled in {t_compile:.2f}s  ({len(fused_c.metal_source)} chars of Metal)\n")

# -- Test data --

runtime = enigma.MetalRuntime()
np.random.seed(0)

x_in = np.random.randn(H).astype(np.float32) * 0.1
w_in_norm = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
w_post_norm = np.random.randn(H).astype(np.float32) * 0.1 + 1.0
w_q_norm = np.random.randn(HD).astype(np.float32) * 0.1 + 1.0
w_k_norm = np.random.randn(HD).astype(np.float32) * 0.1 + 1.0

Wq = (np.random.randn(QSZ, H) / np.sqrt(H)).astype(np.float32)
Wk = (np.random.randn(KVSZ, H) / np.sqrt(H)).astype(np.float32)
Wv = (np.random.randn(KVSZ, H) / np.sqrt(H)).astype(np.float32)
Wo = (np.random.randn(H, QSZ) / np.sqrt(QSZ)).astype(np.float32)
Wg = (np.random.randn(INTER, H) / np.sqrt(H)).astype(np.float32)
Wu = (np.random.randn(INTER, H) / np.sqrt(H)).astype(np.float32)
Wd = (np.random.randn(H, INTER) / np.sqrt(INTER)).astype(np.float32)

angles = (np.arange(HD).astype(np.float32) / HD) * (np.pi / 2)
cos_pos = np.cos(angles).astype(np.float32)
sin_pos = np.sin(angles).astype(np.float32)

k_cache = (np.random.randn(KVH, CACHE_TOTAL, HD) * 0.1).astype(np.float32)
v_cache = (np.random.randn(KVH, CACHE_TOTAL, HD) * 0.1).astype(np.float32)

# -- NumPy reference --

def rmsnorm_np(x, w, eps=RMS_EPS):
    rstd = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rstd * w


def rope_np(x, cos_p, sin_p):
    half = HD // 2
    lo, hi = x[..., :half], x[..., half:]
    return np.concatenate([lo * cos_p[:half] - hi * sin_p[:half],
                           hi * cos_p[half:] + lo * sin_p[half:]], axis=-1)


def full_decode_np(x_in):
    norm1 = rmsnorm_np(x_in, w_in_norm)
    q, k, v = Wq @ norm1, Wk @ norm1, Wv @ norm1

    q_roped = rope_np(rmsnorm_np(q.reshape(QH, HD), w_q_norm), cos_pos, sin_pos)
    k_roped = rope_np(rmsnorm_np(k.reshape(KVH, HD), w_k_norm), cos_pos, sin_pos)

    kc, vc = k_cache.copy(), v_cache.copy()
    kc[:, CACHE_LEN, :] = k_roped
    vc[:, CACHE_LEN, :] = v.reshape(KVH, HD)

    attn = np.empty((QH, HD), dtype=np.float32)
    for qh in range(QH):
        kvh = qh // GQA
        scores = (kc[kvh] @ q_roped[qh]) * ATTN_SCALE
        p = np.exp(scores - scores.max())
        attn[qh] = (p / p.sum()) @ vc[kvh]

    oproj = Wo @ attn.ravel() + x_in
    post = rmsnorm_np(oproj, w_post_norm)
    gate, up = Wg @ post, Wu @ post
    sig = 1.0 / (1.0 + np.exp(-gate))
    return Wd @ ((gate * sig) * up) + oproj


ref_out = full_decode_np(x_in)


def run_buf(compiled, inputs, output_size, grid, threads):
    raw = runtime.execute(compiled, inputs, output_size,
                          grid=grid, threads=threads)
    return np.frombuffer(raw, dtype=np.float32).copy()


# -- Correctness --

print("=" * 72)
print("Correctness test")
print("=" * 72)

# Pre-fill KV cache with host-computed projections.
norm1_ref = rmsnorm_np(x_in, w_in_norm)
k_roped_ref = rope_np(rmsnorm_np((Wk @ norm1_ref).reshape(KVH, HD), w_k_norm),
                       cos_pos, sin_pos)
kc_fused, vc_fused = k_cache.copy(), v_cache.copy()
kc_fused[:, CACHE_LEN, :] = k_roped_ref
vc_fused[:, CACHE_LEN, :] = (Wv @ norm1_ref).reshape(KVH, HD)

zeros_h = np.zeros(H, dtype=np.float32)
zeros_qsz = np.zeros(QSZ, dtype=np.float32)
zeros_kvsz = np.zeros(KVSZ, dtype=np.float32)
zeros_inter = np.zeros(INTER, dtype=np.float32)

fused_out = run_buf(
    fused_c,
    [x_in, w_in_norm, Wq.ravel(), Wk.ravel(), Wv.ravel(),
     w_q_norm, w_k_norm, cos_pos, sin_pos,
     kc_fused.ravel(), vc_fused.ravel(), Wo.ravel(), w_post_norm,
     Wg.ravel(), Wu.ravel(), Wd.ravel(),
     zeros_h.copy(), zeros_qsz.copy(), zeros_kvsz.copy(), zeros_kvsz.copy(),
     zeros_qsz.copy(), zeros_inter.copy(), zeros_h.copy(),
     zeros_qsz.copy(), zeros_kvsz.copy()],
    H * 4, grid=(TG, 1, 1), threads=(TG, 1, 1),
)

err = np.max(np.abs(fused_out - ref_out))
status = "OK" if err < 1e-3 else "FAIL"
print(f"  fused vs NumPy  max|err| = {err:.2e}   {status}")
assert err < 1e-3, f"Fused megakernel divergence: {err}"

# -- Throughput --

print()
print("=" * 72)
print("Throughput benchmark")
print("=" * 72)

WARMUP = 5
ITERS = 100

fused_inputs = [
    x_in, w_in_norm, Wq.ravel(), Wk.ravel(), Wv.ravel(),
    w_q_norm, w_k_norm, cos_pos, sin_pos,
    kc_fused.ravel(), vc_fused.ravel(), Wo.ravel(), w_post_norm,
    Wg.ravel(), Wu.ravel(), Wd.ravel(),
    zeros_h.copy(), zeros_qsz.copy(), zeros_kvsz.copy(), zeros_kvsz.copy(),
    zeros_qsz.copy(), zeros_inter.copy(), zeros_h.copy(),
    zeros_qsz.copy(), zeros_kvsz.copy(),
]

for _ in range(WARMUP):
    run_buf(fused_c, fused_inputs, H * 4, grid=(TG, 1, 1), threads=(TG, 1, 1))

t0 = time.perf_counter()
for _ in range(ITERS):
    run_buf(fused_c, fused_inputs, H * 4, grid=(TG, 1, 1), threads=(TG, 1, 1))
dt = time.perf_counter() - t0
tps = ITERS / dt

print(f"  {tps:.1f} tok/s  ({dt / ITERS * 1000:.2f} ms/tok)")
print()
