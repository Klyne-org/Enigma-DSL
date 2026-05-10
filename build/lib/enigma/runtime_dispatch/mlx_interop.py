# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""MLX <-> Metal zero-copy interop.

Apple Silicon's unified memory means an mlx.array's backing buffer is already
reachable by the GPU — we just need its base pointer and byte length to feed
enigma_create_buffer. No DLPack dance, no copy. This mirrors how Triton and
cuTile accept torch.Tensor directly.
"""

from __future__ import annotations

import ctypes
from typing import Any, Tuple

try:
    import mlx.core as _mx
    _HAS_MLX = True
except ImportError:
    _mx = None
    _HAS_MLX = False


def is_mlx_array(obj: Any) -> bool:
    return _HAS_MLX and isinstance(obj, _mx.array)


_MLX_DTYPE_TO_METAL: dict = {}
_MLX_DTYPE_TO_ITEMSIZE: dict = {}
if _HAS_MLX:
    _MLX_DTYPE_TO_METAL = {
        _mx.float32: "float",
        _mx.float16: "half",
        _mx.bfloat16: "bfloat",
        _mx.int8: "char",
        _mx.int16: "short",
        _mx.int32: "int",
        _mx.int64: "long",
        _mx.uint8: "uchar",
        _mx.uint16: "ushort",
        _mx.uint32: "uint",
        _mx.uint64: "ulong",
        _mx.bool_: "bool",
    }
    _MLX_DTYPE_TO_ITEMSIZE = {
        _mx.float32: 4, _mx.float16: 2, _mx.bfloat16: 2,
        _mx.int8: 1, _mx.int16: 2, _mx.int32: 4, _mx.int64: 8,
        _mx.uint8: 1, _mx.uint16: 2, _mx.uint32: 4, _mx.uint64: 8,
        _mx.bool_: 1,
    }


def mlx_dtype_to_metal(dtype) -> str:
    name = _MLX_DTYPE_TO_METAL.get(dtype)
    if name is None:
        raise TypeError(f"unsupported mlx dtype for Metal interop: {dtype}")
    return name


def mlx_buffer_ptr_and_size(arr) -> Tuple[int, int]:
    """Force materialization and return (pointer, nbytes) of the unified-memory buffer.

    The pointer stays valid as long as ``arr`` is alive. Callers must hold a
    reference to ``arr`` for the duration of the GPU dispatch.
    """
    if not is_mlx_array(arr):
        raise TypeError(f"expected mlx.core.array, got {type(arr).__name__}")
    # Force the computation graph to realize so the buffer exists in memory.
    _mx.eval(arr)
    mv = memoryview(arr)
    if mv.nbytes == 0:
        # Empty array — create a dummy aligned pointer (Metal dislikes NULL).
        return 0, 0
    # memoryview handles bfloat16 (which numpy can't). Pin the address via ctypes.
    raw = (ctypes.c_ubyte * mv.nbytes).from_buffer(mv)
    return ctypes.addressof(raw), mv.nbytes


def make_mlx_output(shape, dtype):
    """Allocate an mlx.array whose buffer will be handed to the GPU as an output."""
    if not _HAS_MLX:
        raise RuntimeError("mlx is not installed")
    arr = _mx.zeros(shape, dtype=dtype)
    _mx.eval(arr)
    return arr


def mlx_nbytes(shape, dtype) -> int:
    itemsize = _MLX_DTYPE_TO_ITEMSIZE.get(dtype)
    if itemsize is None:
        raise TypeError(f"unsupported mlx dtype: {dtype}")
    n = 1
    for d in shape:
        n *= int(d)
    return n * itemsize
