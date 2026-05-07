// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Klyne Research
//
// Handwritten Metal companion for conv1d_laplacian.py.
// 1D Laplacian on a uniform grid with Dirichlet (zero) boundaries:
//   out[i] = (f[i-1] - 2*f[i] + f[i+1]) / h^2
// This file exists for comparison with the Enigma-generated source.

#include <metal_stdlib>
using namespace metal;

constant uint N = 4096;

kernel void laplacian(
    device const float* f       [[buffer(0)]],
    device       float* out     [[buffer(1)]],
    constant     float& inv_h2  [[buffer(2)]],
    uint                i       [[thread_position_in_grid]]
) {
    if (i == 0 || i >= N - 1) {
        out[i] = 0.0f;
        return;
    }
    out[i] = (f[i - 1] - 2.0f * f[i] + f[i + 1]) * inv_h2;
}
