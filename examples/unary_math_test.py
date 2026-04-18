#!/usr/bin/env python3
"""Smoke-test every unary float math op end-to-end: trace -> MLIR -> MSL."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

UNARY_OPS = [
    ("abs",      "abs",   lambda A: np.abs(A),         lambda: np.random.randn(1024).astype(np.float32)),
    ("ceil",     "ceil",  lambda A: np.ceil(A),        lambda: np.random.randn(1024).astype(np.float32)),
    ("floor",    "floor", lambda A: np.floor(A),       lambda: np.random.randn(1024).astype(np.float32)),
    ("trunc",    "trunc", lambda A: np.trunc(A),       lambda: np.random.randn(1024).astype(np.float32)),
    ("sqrt",     "sqrt",  lambda A: np.sqrt(A),        lambda: np.abs(np.random.randn(1024).astype(np.float32)) + 1e-3),
    ("rsqrt",    "rsqrt", lambda A: 1.0/np.sqrt(A),    lambda: np.abs(np.random.randn(1024).astype(np.float32)) + 1e-2),
    ("exp",      "exp",   lambda A: np.exp(A),         lambda: (np.random.randn(1024).astype(np.float32))),
    ("exp2",     "exp2",  lambda A: np.exp2(A),        lambda: np.random.randn(1024).astype(np.float32)),
    ("log",      "log",   lambda A: np.log(A),         lambda: np.abs(np.random.randn(1024).astype(np.float32)) + 1e-2),
    ("log2",     "log2",  lambda A: np.log2(A),        lambda: np.abs(np.random.randn(1024).astype(np.float32)) + 1e-2),
    ("log10",    "log10", lambda A: np.log10(A),       lambda: np.abs(np.random.randn(1024).astype(np.float32)) + 1e-2),
    ("sin",      "sin",   lambda A: np.sin(A),         lambda: np.random.randn(1024).astype(np.float32)),
    ("cos",      "cos",   lambda A: np.cos(A),         lambda: np.random.randn(1024).astype(np.float32)),
    ("tan",      "tan",   lambda A: np.tan(A),         lambda: np.random.randn(1024).astype(np.float32) * 0.5),
    ("tanh",     "tanh",  lambda A: np.tanh(A),        lambda: np.random.randn(1024).astype(np.float32)),
    ("sinh",     "sinh",  lambda A: np.sinh(A),        lambda: np.random.randn(1024).astype(np.float32)),
    ("cosh",     "cosh",  lambda A: np.cosh(A),        lambda: np.random.randn(1024).astype(np.float32)),
    ("asin",     "asin",  lambda A: np.arcsin(A),      lambda: (np.random.rand(1024).astype(np.float32)*2-1)*0.99),
    ("acos",     "acos",  lambda A: np.arccos(A),      lambda: (np.random.rand(1024).astype(np.float32)*2-1)*0.99),
    ("atan",     "atan",  lambda A: np.arctan(A),      lambda: np.random.randn(1024).astype(np.float32)),
]

N = 1024
runtime = enigma.MetalRuntime()

for name, msl_token, ref, gen in UNARY_OPS:
    fn = getattr(enigma, name)

    def make(fn):
        @enigma.kernel
        def k(A: enigma.f32, B: enigma.f32):
            tid = enigma.thread_position_in_grid
            B[tid] = fn(A[tid])
        return k

    k = make(fn)
    compiled = enigma.compile(k)
    assert msl_token in compiled.metal_source, f"{name}: '{msl_token}' not in MSL\n{compiled.metal_source}"

    A = gen()
    out = np.frombuffer(
        runtime.execute(compiled, [A], N*4, grid=(N,1,1), threads=(256,1,1)),
        dtype=np.float32,
    ).copy()
    np.testing.assert_allclose(out, ref(A), rtol=1e-3, atol=1e-3)
    print(f"OK  {name:10s}")

print("\nAll unary math ops passed.")
