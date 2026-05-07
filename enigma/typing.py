# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

from __future__ import annotations


class Numeric:
    width: int = 0
    metal_name: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __repr__(self):
        return self.metal_name


class Float32(Numeric):
    width = 32
    metal_name = "float"


class Float16(Numeric):
    width = 16
    metal_name = "half"


class BFloat16(Numeric):
    width = 16
    metal_name = "bfloat"


class Int8(Numeric):
    width = 8
    metal_name = "char"


class UInt8(Numeric):
    width = 8
    metal_name = "uchar"


class Int16(Numeric):
    width = 16
    metal_name = "short"


class UInt16(Numeric):
    width = 16
    metal_name = "ushort"


class Int32(Numeric):
    width = 32
    metal_name = "int"


class UInt32(Numeric):
    width = 32
    metal_name = "uint"


class Int64(Numeric):
    width = 64
    metal_name = "long"


class UInt64(Numeric):
    width = 64
    metal_name = "ulong"


class Bool(Numeric):
    width = 1
    metal_name = "bool"


f32 = Float32()
f16 = Float16()
bf16 = BFloat16()
i8 = Int8()
u8 = UInt8()
i16 = Int16()
u16 = UInt16()
i32 = Int32()
u32 = UInt32()
i64 = Int64()
u64 = UInt64()
b1 = Bool()


class Scalar:
    """Kernel-argument annotation for a scalar (non-buffer) parameter.

    Usage::

        @enigma.kernel
        def gemm(A: enigma.f32, B: enigma.f32, C: enigma.f32,
                 M: enigma.Scalar(enigma.u32),
                 alpha: enigma.Scalar(enigma.f32)):
            ...

    A ``Scalar`` parameter is passed by value at runtime and is available
    inside the kernel as an ``IRValue`` of the annotated dtype.  It is
    packed into a 1-element device buffer under the hood (so no dialect
    changes are needed); the runtime handles packing.
    """

    def __init__(self, dtype: Numeric):
        if not isinstance(dtype, Numeric):
            raise TypeError(
                f"Scalar(dtype): dtype must be an enigma numeric (e.g. enigma.f32), got {dtype!r}"
            )
        self.dtype = dtype
        self.metal_name = dtype.metal_name
        self.width = dtype.width

    def __repr__(self):
        return f"Scalar({self.metal_name})"
