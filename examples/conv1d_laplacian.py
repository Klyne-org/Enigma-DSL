#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""1D Laplacian stencil — Enigma DSL vs handwritten Metal vs numpy.

The discrete 1D Laplacian is the workhorse of finite-difference PDE solvers
(heat equation, wave equation, diffusion). For a 1D field f sampled on a
uniform grid with spacing h, the second derivative is approximated by the
3-point stencil:

    (∇²f)[i] ≈ (f[i-1] - 2*f[i] + f[i+1]) / h²

Boundary cells are kept at zero (Dirichlet BC) for simplicity. This script
runs three implementations on the same input and checks they agree:

  * `laplacian_dsl`     - written in Enigma's @kernel DSL (this is the demo)
  * handwritten Metal   - shipped as `conv1d_laplacian_handwritten.metal`,
                          shown for comparison only
  * numpy reference     - ground truth for correctness

All three should produce identical results to within float32 precision.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import enigma


N = 4096
H = 1.0  # unit grid spacing; rescale if you want a physical domain
INV_H2 = np.float32(1.0 / (H * H))


@enigma.kernel
def laplacian_dsl(F: enigma.f32, Out: enigma.f32):
    i = enigma.thread_position_in_grid
    zero_u = enigma.metal_cast(0, "uint")
    n_minus_1 = enigma.metal_cast(N - 1, "uint")
    two = enigma.metal_cast(2.0, "float")
    inv_h2 = enigma.metal_cast(float(INV_H2), "float")

    Out[i] = enigma.metal_cast(0.0, "float")
    with enigma.if_(enigma.cmp_ult(zero_u, i) & enigma.cmp_ult(i, n_minus_1)):
        left = F[i - 1]
        center = F[i]
        right = F[i + 1]
        Out[i] = (left - two * center + right) * inv_h2


def numpy_reference(f: np.ndarray, inv_h2: float) -> np.ndarray:
    out = np.zeros_like(f)
    out[1:-1] = (f[:-2] - 2.0 * f[1:-1] + f[2:]) * inv_h2
    return out


def main() -> None:
    print("Compiling laplacian_dsl kernel...")
    compiled = enigma.compile(laplacian_dsl)

    print("Generated Metal source:")
    print("-" * 60)
    print(compiled.metal_source)
    print("-" * 60)

    rng = np.random.default_rng(0)
    f = rng.standard_normal(N).astype(np.float32)

    runtime = enigma.MetalRuntime()
    raw = runtime.execute(
        compiled,
        inputs=[f],
        output_size=N * 4,
        grid=(N, 1, 1),
        threads=(min(N, 256), 1, 1),
    )
    out_gpu = np.frombuffer(raw, dtype=np.float32)
    out_ref = numpy_reference(f, float(INV_H2))

    np.testing.assert_allclose(out_gpu, out_ref, rtol=1e-4, atol=1e-4)
    abs_err = np.max(np.abs(out_gpu - out_ref))
    rel_err = np.max(np.abs(out_gpu - out_ref) / (np.abs(out_ref) + 1e-30))
    print(f"PASSED: 1D Laplacian on {N} samples")
    print(f"  max |abs error| = {abs_err:.2e}")
    print(f"  max |rel error| = {rel_err:.2e}")


if __name__ == "__main__":
    main()
