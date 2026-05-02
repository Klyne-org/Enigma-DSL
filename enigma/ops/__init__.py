"""Higher-level tile ops built on top of the Enigma tracing primitives.

These are user-facing helpers that lower to combinations of intrinsics
already exposed by ``enigma._tracing`` — no extra dialect changes needed.
"""

from .gemm import gemm as gemm
from .quant import (
    pack_uint8x4 as pack_uint8x4,
    unpack_uint8x4 as unpack_uint8x4,
    pack_int4x2 as pack_int4x2,
    unpack_int4x2 as unpack_int4x2,
    dequantize_int8 as dequantize_int8,
)
