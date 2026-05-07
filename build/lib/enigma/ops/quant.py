# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Pack / unpack helpers for low-bit quantization.

Apple Metal does not have native ``fp8`` or ``fp4`` storage, and ``int4``
is also unsupported. The standard workaround (used by tilelang PR #2130)
is to store quantized values in a packed ``uint`` buffer and dequantize
on the fly. The helpers below wrap Enigma's existing ``extract_bits`` /
``insert_bits`` intrinsics so user code can avoid hand-rolling bit
manipulation per kernel.

  * ``pack_uint8x4`` / ``unpack_uint8x4`` — four 8-bit lanes in a 32-bit ``uint``.
  * ``pack_int4x2``  / ``unpack_int4x2``  — two signed 4-bit lanes in a byte.
  * ``dequantize_int8``                   — ``scale * (x - zero_point)`` as float.
"""

from .._tracing import (
    _ensure_ir,
    _require_builder,
    cmp_ge,
    extract_bits,
    insert_bits,
    metal_cast,
    select,
)

__all__ = [
    "pack_uint8x4",
    "unpack_uint8x4",
    "pack_int4x2",
    "unpack_int4x2",
    "dequantize_int8",
]


def _as_uint(v):
    v = _ensure_ir(v)
    return v if v.dtype == "uint" else metal_cast(v, "uint")


def pack_uint8x4(b0, b1, b2, b3):
    """Pack four ``uchar``-sized lanes into a single ``uint`` (LSB-first).

    Only the low 8 bits of each input are kept.
    """
    _require_builder("pack_uint8x4")
    out = metal_cast(0, "uint")
    for shift, val in zip((0, 8, 16, 24), (b0, b1, b2, b3)):
        out = insert_bits(out, _as_uint(val), shift, 8)
    return out


def unpack_uint8x4(packed):
    """Unpack a ``uint`` into a 4-tuple of ``uint`` lanes (LSB-first)."""
    _require_builder("unpack_uint8x4")
    v = _as_uint(packed)
    return tuple(extract_bits(v, sh, 8) for sh in (0, 8, 16, 24))


def pack_int4x2(lo, hi):
    """Pack two signed 4-bit ints (low/high nibble) into a ``uint`` byte.

    Negative values are stored as two's-complement nibbles, which round
    through :func:`unpack_int4x2` without loss.
    """
    _require_builder("pack_int4x2")
    out = metal_cast(0, "uint")
    out = insert_bits(out, _as_uint(lo), 0, 4)
    out = insert_bits(out, _as_uint(hi), 4, 4)
    return out


def unpack_int4x2(packed):
    """Unpack a packed byte into two signed ``int`` 4-bit values.

    Sign extension: read the nibble unsigned, subtract 16 if the high
    bit is set.
    """
    _require_builder("unpack_int4x2")
    v = _as_uint(packed)
    eight = metal_cast(8, "int")
    sixteen = metal_cast(16, "int")

    def _sign_extend(nib_uint):
        nib = metal_cast(nib_uint, "int")
        return select(nib, nib - sixteen, cmp_ge(nib, eight))

    return _sign_extend(extract_bits(v, 0, 4)), _sign_extend(extract_bits(v, 4, 4))


def dequantize_int8(x, scale, zero_point=0):
    """Return ``scale * (x - zero_point)`` as a ``float``.

    Common helper for INT8 dequant — handy for fused-dequant GEMM kernels.
    """
    _require_builder("dequantize_int8")
    v = _ensure_ir(x)
    s = _ensure_ir(scale)
    z = _ensure_ir(zero_point)
    if v.dtype != "int":
        v = metal_cast(v, "int")
    if z.dtype != "int":
        z = metal_cast(z, "int")
    if s.dtype != "float":
        s = metal_cast(s, "float")
    return metal_cast(v - z, "float") * s
