#!/usr/bin/env python3
"""SDPA benchmark: Enigma v4 vs gpt-oss-style handwritten Metal.

Same data layout: interleaved KV, scale factors, args struct.
4 simdgroups, 8 Q heads per KV head, head_dim=64.
"""
import os
import sys
import struct
import subprocess
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import enigma
from enigma.tensor import Tensor, tensor_composition, tensor_zipped_divide
from enigma.compiler.compiler import CompiledKernel

QMUL = 8
HEAD_DIM = 64
NUM_Q_HEADS = 64
NUM_KV_HEADS = NUM_Q_HEADS // QMUL  # = 8
NUM_SG = 4
THREADS = NUM_SG * 32
NUM_TOKENS = 128
NUM_KV_TOKENS = 512
TOKEN_STRIDE = 2 * HEAD_DIM  # interleaved K/V
QKV_DIM = NUM_Q_HEADS * HEAD_DIM


thr = enigma.make_ordered_layout((32,), order=(0,))
val = enigma.make_ordered_layout((2,), order=(0,))
tiler_d, tv_d = enigma.make_layout_tv(thr, val)


def _tv_load_head(buf_name, buf_idx, base_offset, lane, tv_layout, tiler):
    """Layout algebra TV load: Tensor → zipped_divide → tensor_composition → load.
    Returns a float2 vec (single-group TV load auto-promotes to vec)."""
    t = Tensor(buf_name, buf_idx, "float",
               enigma.Layout(HEAD_DIM, 1), base_offset=base_offset)
    tiled = tensor_zipped_divide(t, tiler)
    tv = tensor_composition(tiled[(None,)], tv_layout, tiler)
    return tv[(lane, None)].load()


@enigma.kernel
def sdpa_enigma(mQ, mKV, mScale, mO):
    qt = enigma.threadgroup_position_in_grid("x")
    h = enigma.threadgroup_position_in_grid("y")
    lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    num_sg = enigma.simdgroups_per_threadgroup()
    f2 = lane * 2

    # Q: TV load per head → float2 (layout algebra vectorized)
    q_group_base = qt * QKV_DIM + h * (QMUL * HEAD_DIM)
    q = []
    for hh in enigma.range_constexpr(QMUL):
        q.append(_tv_load_head("Q", 0, q_group_base + hh * HEAD_DIM,
                               lane, tv_d, tiler_d))

    m = []
    for hh in enigma.range_constexpr(QMUL):
        m.append(mScale[h * QMUL + hh])

    l = []
    for hh in enigma.range_constexpr(QMUL):
        l.append(enigma.where(enigma.metal_cast(0.0, "float"),
                              enigma.metal_cast(1.0, "float"),
                              enigma.cmp_eq(sg_idx, 0)))

    ox = [enigma.metal_cast(0.0, "float") for _ in range(QMUL)]
    oy = [enigma.metal_cast(0.0, "float") for _ in range(QMUL)]

    kv_head_base = h * NUM_KV_TOKENS * TOKEN_STRIDE

    kv_iters = NUM_KV_TOKENS // NUM_SG
    for kt_iter in enigma.range(kv_iters):
        actual_kt = kt_iter * NUM_SG + sg_idx
        kt_base = kv_head_base + actual_kt * TOKEN_STRIDE

        k_addr = kt_base + f2
        kval = enigma.make_float2(mKV[k_addr], mKV[k_addr + 1])
        v_addr = kt_base + HEAD_DIM + f2
        vx = mKV[v_addr]
        vy = mKV[v_addr + 1]

        for hh in enigma.range_constexpr(QMUL):
            qk = enigma.simd_sum(enigma.dot(q[hh], kval))
            new_m = enigma.fmax(m[hh], qk)
            alpha = enigma.exp(m[hh] - new_m)
            p = enigma.exp(qk - new_m)
            l[hh] = enigma.fma(l[hh], alpha, p)
            m[hh] = new_m
            ox[hh] = enigma.fma(vx, p, ox[hh] * alpha)
            oy[hh] = enigma.fma(vy, p, oy[hh] * alpha)

    # Cross-sg m/l reduction
    shared_ml = enigma.threadgroup_alloc("float", 16 * NUM_SG)
    is_first = enigma.cmp_eq(lane, 0)
    for hh in enigma.range_constexpr(QMUL):
        enigma.store_if(shared_ml, hh * NUM_SG + sg_idx, m[hh], is_first)
        enigma.store_if(shared_ml, (QMUL + hh) * NUM_SG + sg_idx, l[hh], is_first)
    enigma.barrier()

    gm = []
    for hh in enigma.range_constexpr(QMUL):
        sg_m = enigma.load_if(shared_ml, hh * NUM_SG + lane,
                               enigma.cmp_ult(lane, num_sg), default=-1e30)
        gm.append(enigma.simd_max(sg_m))

    for hh in enigma.range_constexpr(QMUL):
        rescale = enigma.exp(m[hh] - gm[hh])
        ox[hh] = ox[hh] * rescale
        oy[hh] = oy[hh] * rescale

    shared_gl = enigma.threadgroup_alloc("float", QMUL)
    with enigma.if_(enigma.cmp_eq(sg_idx, 0)):
        for hh in enigma.range_constexpr(QMUL):
            sg_l = enigma.load_if(shared_ml, (QMUL + hh) * NUM_SG + lane,
                                   enigma.cmp_ult(lane, num_sg), default=0.0)
            sg_m_v = enigma.load_if(shared_ml, hh * NUM_SG + lane,
                                     enigma.cmp_ult(lane, num_sg), default=-1e30)
            gl_h = enigma.simd_sum(sg_l * enigma.exp(sg_m_v - gm[hh]))
            enigma.store_if(shared_gl, hh, gl_h, enigma.cmp_eq(lane, 0))
    enigma.barrier()

    # Output reduction
    shared_ox = enigma.threadgroup_alloc("float", QMUL * THREADS)
    shared_oy = enigma.threadgroup_alloc("float", QMUL * THREADS)
    tid = enigma.thread_position_in_threadgroup("x")
    for hh in enigma.range_constexpr(QMUL):
        shared_ox[hh * THREADS + tid] = ox[hh]
        shared_oy[hh * THREADS + tid] = oy[hh]
    enigma.barrier()

    o_base = qt * (NUM_Q_HEADS * HEAD_DIM) + h * (QMUL * HEAD_DIM)
    with enigma.if_(enigma.cmp_eq(sg_idx, 0)):
        for hh in enigma.range_constexpr(QMUL):
            sx = shared_ox[hh * THREADS + lane]
            sy = shared_oy[hh * THREADS + lane]
            for s in enigma.range_constexpr(1, NUM_SG):
                sx = sx + shared_ox[hh * THREADS + s * 32 + lane]
                sy = sy + shared_oy[hh * THREADS + s * 32 + lane]
            gl_h = shared_gl[hh]
            addr = o_base + hh * HEAD_DIM + f2
            mO[addr] = sx / gl_h
            mO[addr + 1] = sy / gl_h


@enigma.jit
def sdpa_jit(mQ, mKV, mScale, mO):
    sdpa_enigma(mQ, mKV, mScale, mO).launch(
        grid=(NUM_TOKENS * THREADS, NUM_KV_HEADS, 1),
        block=(THREADS, 1, 1),
    )


mQ = Tensor("Q", 0, "float", enigma.Layout(NUM_TOKENS * QKV_DIM, 1))
mKV = Tensor("KV", 1, "float", enigma.Layout(NUM_KV_HEADS * NUM_KV_TOKENS * TOKEN_STRIDE, 1))
mScale = Tensor("Scale", 2, "float", enigma.Layout(NUM_Q_HEADS, 1))
mO = Tensor("O", 3, "float", enigma.Layout(NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM, 1))

print("Compiling Enigma SDPA (gpt-oss layout)...")
enigma_compiled = enigma.compile(sdpa_jit, mQ, mKV, mScale, mO)
enigma_compiled.export_metal(os.path.join(os.path.dirname(__file__), "sdpa_enigma.metal"))

print("Compiling gpt-oss SDPA...")
hw_dir = tempfile.mkdtemp()
hw_metal = os.path.join(os.path.dirname(__file__), "sdpa_handwritten.metal")
hw_air = os.path.join(hw_dir, "sdpa.air")
hw_metallib = os.path.join(hw_dir, "sdpa.metallib")
subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", hw_metal, "-o", hw_air],
               check=True, capture_output=True)
subprocess.run(["xcrun", "-sdk", "macosx", "metallib", hw_air, "-o", hw_metallib],
               check=True, capture_output=True)
hw_compiled = CompiledKernel("sdpa_gptoss", hw_metallib,
                             Path(hw_metallib).read_bytes(), "")

np.random.seed(42)
Q = np.random.randn(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM).astype(np.float32) * 0.1
K = np.random.randn(NUM_KV_HEADS, NUM_KV_TOKENS, HEAD_DIM).astype(np.float32) * 0.1
V = np.random.randn(NUM_KV_HEADS, NUM_KV_TOKENS, HEAD_DIM).astype(np.float32) * 0.1
scale = np.zeros(NUM_Q_HEADS, dtype=np.float32) - 1e30  # init m to -inf

# Interleave KV: (num_kv_heads, num_kv_tokens, 2, head_dim)
KV = np.zeros((NUM_KV_HEADS, NUM_KV_TOKENS, TOKEN_STRIDE), dtype=np.float32)
KV[:, :, :HEAD_DIM] = K
KV[:, :, HEAD_DIM:] = V

# Args struct: qkv_dim, num_kv_tokens, kv_stride, window
args = struct.pack("<IIII", QKV_DIM, NUM_KV_TOKENS, NUM_KV_TOKENS * TOKEN_STRIDE, NUM_KV_TOKENS)
args_f32 = np.frombuffer(args, dtype=np.float32)

# Reference
def attention_ref(Q, K, V):
    T, H, D = Q.shape
    O = np.zeros_like(Q)
    for t in range(T):
        for qh in range(H):
            kvh = qh // QMUL
            scores = Q[t, qh] @ K[kvh].T
            scores = np.exp(scores - np.max(scores))
            scores /= np.sum(scores)
            O[t, qh] = scores @ V[kvh]
    return O

expected = attention_ref(Q, K, V)
runtime = enigma.MetalRuntime()
O_size = NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM * 4

# Enigma (buffers: Q=0, KV=1, Scale=2, O=3)
e_out = np.frombuffer(
    runtime.execute(enigma_compiled,
                    [Q.reshape(-1), KV.reshape(-1), scale],
                    O_size, grid=enigma_compiled.grid, threads=enigma_compiled.block),
    dtype=np.float32).reshape(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
err_e = np.max(np.abs(e_out - expected))
print(f"  Enigma:      max|err|={err_e:.2e}  {'PASS' if err_e < 0.05 else 'FAIL'}")

# gpt-oss style (buffers: args=0, Q=1, KV=2, Scale=3, O=4)
hw_grid = (NUM_TOKENS * THREADS, NUM_KV_HEADS, 1)
hw_threads = (THREADS, 1, 1)
h_out = np.frombuffer(
    runtime.execute(hw_compiled,
                    [args_f32, Q.reshape(-1), KV.reshape(-1), scale],
                    O_size, grid=hw_grid, threads=hw_threads),
    dtype=np.float32).reshape(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
err_h = np.max(np.abs(h_out - expected))
print(f"  gpt-oss:     max|err|={err_h:.2e}  {'PASS' if err_h < 0.05 else 'FAIL'}")

flops_per_token = NUM_Q_HEADS * NUM_KV_TOKENS * (4 * HEAD_DIM + 5)
total_flops = NUM_TOKENS * flops_per_token

WARMUP, ITERS = 20, 200
def bench(name, compiled_k, inputs, grid, threads):
    prep = runtime.prepare(compiled_k, inputs, O_size)
    for _ in range(WARMUP):
        prep.dispatch(grid=grid, threads=threads)
    times = []
    for _ in range(ITERS):
        times.append(prep.dispatch_timed(grid=grid, threads=threads))
    prep.release()
    med = np.median(times)
    tflops = total_flops / (med * 1e-6) / 1e12
    print(f"  {name:45s} {med:8.2f} us  {tflops:.3f} TFLOPS")
    return med

inputs_enigma = [Q.reshape(-1), KV.reshape(-1), scale]
inputs_hw = [args_f32, Q.reshape(-1), KV.reshape(-1), scale]
print(f"\nSDPA: {NUM_TOKENS} tok × {NUM_KV_TOKENS} KV × {NUM_Q_HEADS} Qh × {NUM_KV_HEADS} KVh × d={HEAD_DIM}, {NUM_SG} sg")
print(f"  FLOPs: {total_flops/1e9:.2f} GFLOPs")
print(f"\n{'─' * 80}")
t_e = bench("Enigma DSL (gpt-oss layout, 4sg)", enigma_compiled, inputs_enigma,
            enigma_compiled.grid, enigma_compiled.block)
t_h = bench("gpt-oss handwritten (4sg)", hw_compiled, inputs_hw, hw_grid, hw_threads)
print(f"{'─' * 80}")
print(f"  Enigma / gpt-oss = {t_h / t_e:.2f}x")
