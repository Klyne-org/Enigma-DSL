#!/usr/bin/env python3
"""Pack/unpack round-trip: unorm4x8 pack(float4) then unpack back to float4.

Round-trip loses precision (quantized to 8 bits/channel), so we compare with
a generous tolerance against the expected quantized values.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma

N = 1024
runtime = enigma.MetalRuntime()


@enigma.kernel
def pack_roundtrip_k(
    Rx: enigma.f32, Gy: enigma.f32, Bz: enigma.f32, Aw: enigma.f32,
    Out: enigma.f32,
):
    tid = enigma.thread_position_in_grid
    v = enigma.make_float4(Rx[tid], Gy[tid], Bz[tid], Aw[tid])
    packed = enigma.pack_float_to_unorm4x8(v)
    unp = enigma.unpack_unorm4x8_to_float(packed)
    Out[tid] = unp.x  # read back the R channel


compiled = enigma.compile(pack_roundtrip_k)
msl = compiled.metal_source
assert "pack_float_to_unorm4x8" in msl, msl
assert "unpack_unorm4x8_to_float" in msl, msl

R = np.random.rand(N).astype(np.float32)      # in [0, 1]
G = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)
A = np.random.rand(N).astype(np.float32)

raw = runtime.execute(
    compiled, [R, G, B, A], N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
out = np.frombuffer(raw, dtype=np.float32).copy()

# unorm8 quantization: round(x*255)/255
expected = np.round(R * 255.0) / 255.0
np.testing.assert_allclose(out, expected, rtol=0, atol=1.0/255.0 + 1e-6)
print("OK  pack_unorm4x8 / unpack_unorm4x8 round-trip (R channel)")


print("\nPack/unpack tests passed.")
