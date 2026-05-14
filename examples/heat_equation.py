#!/usr/bin/env python3
"""2D Heat Equation (Jacobi) — Enigma DSL.

∂T/∂t = α∇²T → T_new[y,x] = 0.25 * (T[y-1,x] + T[y+1,x] + T[y,x-1] + T[y,x+1])

Shared memory with 1-cell halo for stencil neighbors.
Boundary condition: zero (Dirichlet).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import enigma
from enigma.tensor import Tensor

H, W = 512, 512
TILE_H, TILE_W = 16, 16
NUM_STEPS = 100
SH_H = TILE_H + 2
SH_W = TILE_W + 2


@enigma.kernel
def jacobi_step(mIn, mOut):
    bx = enigma.threadgroup_position_in_grid("x")
    by = enigma.threadgroup_position_in_grid("y")
    tx = enigma.thread_position_in_threadgroup("x")
    ty = enigma.thread_position_in_threadgroup("y")

    gx = bx * TILE_W + tx
    gy = by * TILE_H + ty
    sx = tx + 1
    sy = ty + 1

    shared = enigma.threadgroup_alloc("float", SH_H * SH_W)

    shared[sy * SH_W + sx] = mIn[gy * W + gx]

    with enigma.if_(enigma.cmp_eq(ty, 0)):
        hgy = gy - 1
        shared[0 * SH_W + sx] = enigma.where(
            enigma.metal_cast(0.0, "float"), mIn[hgy * W + gx],
            enigma.cmp_ge(hgy, 0))

    with enigma.if_(enigma.cmp_eq(ty, TILE_H - 1)):
        hgy = gy + 1
        shared[(TILE_H + 1) * SH_W + sx] = enigma.where(
            enigma.metal_cast(0.0, "float"), mIn[hgy * W + gx],
            enigma.cmp_ult(hgy, H))

    with enigma.if_(enigma.cmp_eq(tx, 0)):
        hgx = gx - 1
        shared[sy * SH_W + 0] = enigma.where(
            enigma.metal_cast(0.0, "float"), mIn[gy * W + hgx],
            enigma.cmp_ge(hgx, 0))

    with enigma.if_(enigma.cmp_eq(tx, TILE_W - 1)):
        hgx = gx + 1
        shared[sy * SH_W + (TILE_W + 1)] = enigma.where(
            enigma.metal_cast(0.0, "float"), mIn[gy * W + hgx],
            enigma.cmp_ult(hgx, W))

    enigma.barrier()

    n = shared[(sy - 1) * SH_W + sx]
    s = shared[(sy + 1) * SH_W + sx]
    w = shared[sy * SH_W + (sx - 1)]
    e = shared[sy * SH_W + (sx + 1)]

    mOut[gy * W + gx] = (n + s + e + w) * 0.25


@enigma.jit
def heat_step(mIn, mOut):
    jacobi_step(mIn, mOut).launch(
        grid=((W // TILE_W) * TILE_W, (H // TILE_H) * TILE_H, 1),
        block=(TILE_W, TILE_H, 1),
    )


mIn = Tensor("In", 0, "float", enigma.Layout(H * W, 1))
mOut = Tensor("Out", 1, "float", enigma.Layout(H * W, 1))

print(f"Heat equation: {H}×{W}, {TILE_H}×{TILE_W} tiles, {NUM_STEPS} steps")
print("Compiling...")
compiled = enigma.compile(heat_step, mIn, mOut)
compiled.export_metal(os.path.join(os.path.dirname(__file__), "metal", "heat_equation.metal"))

grid_init = np.zeros((H, W), dtype=np.float32)
grid_init[H // 4:3 * H // 4, W // 4:3 * W // 4] = 100.0

def jacobi_ref(g, steps):
    g = g.copy()
    for _ in range(steps):
        gn = g.copy()
        gn[1:-1, 1:-1] = 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] +
                                   g[1:-1, :-2] + g[1:-1, 2:])
        g = gn
    return g

expected = jacobi_ref(grid_init, NUM_STEPS)
runtime = enigma.MetalRuntime()
buf_size = H * W * 4

cur = grid_init.ravel().copy()
for step in range(NUM_STEPS):
    nxt = np.frombuffer(
        runtime.execute(compiled, [cur], buf_size,
                        grid=compiled.grid, threads=compiled.block),
        dtype=np.float32).copy()
    cur = nxt

err = np.max(np.abs(cur.reshape(H, W) - expected))
print(f"  max|err| = {err:.2e}  {'PASS' if err < 0.1 else 'FAIL'}")

WARMUP, ITERS = 10, 100
prep = runtime.prepare(compiled, [grid_init.ravel()], buf_size)
for _ in range(WARMUP):
    prep.dispatch(grid=compiled.grid, threads=compiled.block)
times = []
for _ in range(ITERS):
    times.append(prep.dispatch_timed(grid=compiled.grid, threads=compiled.block))
prep.release()
med = np.median(times)
print(f"  {med:.2f} us/step  {H * W / (med * 1e-6) / 1e9:.2f} Gcells/s")
