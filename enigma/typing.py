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


class Int32(Numeric):
    width = 32
    metal_name = "int"


class UInt32(Numeric):
    width = 32
    metal_name = "uint"


f32 = Float32()
f16 = Float16()
bf16 = BFloat16()
i32 = Int32()
u32 = UInt32()
