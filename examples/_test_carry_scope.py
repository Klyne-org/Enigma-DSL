import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import enigma
from enigma.tensor import Tensor
import numpy as np

@enigma.kernel
def test_k(mK, mO):
    lane = enigma.thread_index_in_simdgroup()
    sg_idx = enigma.simdgroup_index_in_threadgroup()
    num_sg = enigma.simdgroups_per_threadgroup()

    m0 = enigma.metal_cast(-1e30, "float")
    l0 = enigma.metal_cast(0.0, "float")
    ox0 = enigma.metal_cast(0.0, "float")

    for kt in enigma.range(4):
        kx = mK[kt * 64 + lane * 2]
        new_m = enigma.fmax(m0, kx)
        alpha = enigma.exp(m0 - new_m)
        p = enigma.exp(kx - new_m)
        l0 = enigma.fma(l0, alpha, p)
        m0 = new_m
        ox0 = enigma.fma(kx, p, ox0 * alpha)

    # Cross-sg reduction
    shared = enigma.threadgroup_alloc("float", 8)
    is_first = enigma.cmp_eq(lane, 0)
    enigma.store_if(shared, sg_idx, m0, is_first)
    enigma.store_if(shared, 4 + sg_idx, l0, is_first)
    enigma.barrier()

    gm = enigma.load_if(shared, lane, enigma.cmp_ult(lane, num_sg), default=-1e30)
    gm = enigma.simd_max(gm)
    ox0 = ox0 * enigma.exp(m0 - gm)

    shared_gl = enigma.threadgroup_alloc("float", 1)
    with enigma.if_(enigma.cmp_eq(sg_idx, 0)):
        sg_l = enigma.load_if(shared, 4 + lane, enigma.cmp_ult(lane, num_sg), default=0.0)
        sg_m = enigma.load_if(shared, lane, enigma.cmp_ult(lane, num_sg), default=-1e30)
        gl = enigma.simd_sum(sg_l * enigma.exp(sg_m - gm))
        enigma.store_if(shared_gl, 0, gl, enigma.cmp_eq(lane, 0))
    enigma.barrier()

    # Output reduction
    shared_ox = enigma.threadgroup_alloc("float", 128)
    shared_ox[lane + sg_idx * 32] = ox0
    enigma.barrier()

    with enigma.if_(enigma.cmp_eq(sg_idx, 0)):
        total = shared_ox[lane]
        for s in enigma.range_constexpr(1, 4):
            total = total + shared_ox[lane + s * 32]
        gl_val = shared_gl[0]
        mO[lane] = total / gl_val

@enigma.jit
def test_jit(mK, mO):
    test_k(mK, mO).launch(grid=(128, 1, 1), block=(128, 1, 1))

mK = Tensor("K", 0, "float", enigma.Layout(256, 1))
mO = Tensor("O", 1, "float", enigma.Layout(32, 1))

print("Compiling...")
compiled = enigma.compile(test_jit, mK, mO, keep_metal_source=True,
                          work_dir=os.path.join(os.path.dirname(__file__), "_build"))
print("COMPILED OK")
