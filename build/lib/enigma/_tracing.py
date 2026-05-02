"""Tracing IR for Enigma kernel compilation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_local = threading.local()


class EnigmaError(Exception):
    """Structured error raised by the DSL tracer / emitter."""


def _require_builder(op_name: str) -> "KernelBuilder":
    builder = get_builder()
    if builder is None:
        raise EnigmaError(
            f"enigma.{op_name} can only be used inside @enigma.kernel / @enigma.jit"
        )
    return builder


def get_builder() -> Optional[KernelBuilder]:
    return getattr(_local, "builder", None)


@dataclass
class IRValue:
    """An SSA value produced during tracing."""

    name: str
    dtype: str
    _tv_groups: Any = field(default=None, repr=False)

    def __add__(self, other) -> IRValue:
        if isinstance(other, int) and other == 0:
            return self
        return _tv_aware_binop("add", "tv_add", self, other)

    def __radd__(self, other) -> IRValue:
        if isinstance(other, int) and other == 0:
            return self
        return _tv_aware_binop("add", "tv_add", other, self)

    def __sub__(self, other) -> IRValue:
        return _tv_aware_binop("sub", "tv_sub", self, other)

    def __rsub__(self, other) -> IRValue:
        return _tv_aware_binop("sub", "tv_sub", other, self)

    def __mul__(self, other) -> IRValue:
        if isinstance(other, int) and other == 1:
            return self
        return _tv_aware_binop("mul", "tv_mul", self, other)

    def __rmul__(self, other) -> IRValue:
        if isinstance(other, int) and other == 1:
            return self
        return _tv_aware_binop("mul", "tv_mul", other, self)

    def __truediv__(self, other) -> IRValue:
        return _tv_aware_binop("div", "tv_div", self, other)

    def __floordiv__(self, other) -> IRValue:
        return _tv_aware_binop("div", "tv_div", self, other)

    def __mod__(self, other) -> IRValue:
        return _tv_aware_binop("mod", "tv_mod", self, other)

    def __neg__(self) -> IRValue:
        builder = get_builder()
        assert builder is not None
        result = builder.new_value(self.dtype)
        builder.record(IROp("neg", result, [self]))
        return result

    def __or__(self, other) -> IRValue:
        return _binop("bitor", self, other)

    def __ror__(self, other) -> IRValue:
        return _binop("bitor", other, self)

    def __and__(self, other) -> IRValue:
        return _binop("bitand", self, other)

    def __rand__(self, other) -> IRValue:
        return _binop("bitand", other, self)

    def __xor__(self, other) -> IRValue:
        return _binop("bitxor", self, other)

    def __rxor__(self, other) -> IRValue:
        return _binop("bitxor", other, self)

    def __lshift__(self, other) -> IRValue:
        return _binop("shl", self, other)

    def __rshift__(self, other) -> IRValue:
        return _binop("shr", self, other)

    def __invert__(self) -> IRValue:
        builder = get_builder()
        assert builder is not None
        result = builder.new_value(self.dtype)
        builder.record(IROp("bitnot", result, [self]))
        return result


def _ensure_ir(x) -> IRValue:
    """Wrap a Python scalar as an IR constant if needed."""
    if isinstance(x, IRValue):
        return x
    if isinstance(x, bool):
        builder = get_builder()
        assert builder is not None
        return builder.make_const("i1", int(x))
    if isinstance(x, int):
        builder = get_builder()
        assert builder is not None
        return builder.make_const("uint", x)
    if isinstance(x, float):
        builder = get_builder()
        assert builder is not None
        # Float constants aren't cached; emit a fresh one each time.
        val = builder.new_value("float")
        builder.record(IROp("const", val, [], attrs={"value": float(x), "dtype": "float"}))
        return val
    raise TypeError(f"Cannot convert {type(x).__name__} to IRValue")


def _binop(op_type: str, lhs, rhs) -> IRValue:
    builder = get_builder()
    assert builder is not None, "Binary op outside tracing context"
    lhs, rhs = _ensure_ir(lhs), _ensure_ir(rhs)
    result = builder.new_value(lhs.dtype)
    builder.record(IROp(op_type, result, [lhs, rhs]))
    return result

def _unary(op_type: str, x: IRValue, result_dtype: Optional[str] = None) -> IRValue:
    builder = get_builder()
    assert builder is not None, f"{op_type} outside tracing context"
    result = builder.new_value(result_dtype or x.dtype)
    builder.record(IROp(op_type, result, [x]))
    return result


def _ternary(op_type: str, a, b, c, result_dtype: Optional[str] = None) -> IRValue:
    builder = get_builder()
    assert builder is not None
    a, b, c = _ensure_ir(a), _ensure_ir(b), _ensure_ir(c)
    result = builder.new_value(result_dtype or a.dtype)
    builder.record(IROp(op_type, result, [a, b, c]))
    return result


# --- Unary float math ---
def sqrt(x: IRValue) -> IRValue: return _unary("sqrt", x)
def abs(x: IRValue) -> IRValue: return _unary("abs", x)
def ceil(x: IRValue) -> IRValue: return _unary("ceil", x)
def floor(x: IRValue) -> IRValue: return _unary("floor", x)
def round(x: IRValue) -> IRValue: return _unary("round", x)
def trunc(x: IRValue) -> IRValue: return _unary("trunc", x)
def sign(x: IRValue) -> IRValue: return _unary("sign", x)
def saturate(x: IRValue) -> IRValue: return _unary("saturate", x)
def fract(x: IRValue) -> IRValue: return _unary("fract", x)
def rsqrt(x: IRValue) -> IRValue: return _unary("rsqrt", x)
def exp(x: IRValue) -> IRValue: return _unary("exp", x)
def exp2(x: IRValue) -> IRValue: return _unary("exp2", x)
def log(x: IRValue) -> IRValue: return _unary("log", x)
def log2(x: IRValue) -> IRValue: return _unary("log2", x)
def log10(x: IRValue) -> IRValue: return _unary("log10", x)
def sin(x: IRValue) -> IRValue: return _unary("sin", x)
def cos(x: IRValue) -> IRValue: return _unary("cos", x)
def tan(x: IRValue) -> IRValue: return _unary("tan", x)
def asin(x: IRValue) -> IRValue: return _unary("asin", x)
def acos(x: IRValue) -> IRValue: return _unary("acos", x)
def atan(x: IRValue) -> IRValue: return _unary("atan", x)
def sinh(x: IRValue) -> IRValue: return _unary("sinh", x)
def cosh(x: IRValue) -> IRValue: return _unary("cosh", x)
def tanh(x: IRValue) -> IRValue: return _unary("tanh", x)


# --- Binary float math ---
def fmin(a, b): return _binop("fmin", a, b)
def fmax(a, b): return _binop("fmax", a, b)
def pow(a, b): return _binop("pow", a, b)
def fmod(a, b): return _binop("fmod", a, b)
def atan2(a, b): return _binop("atan2", a, b)
def step(edge, x): return _binop("step", edge, x)
def copysign(a, b): return _binop("copysign", a, b)


# --- Ternary float math ---
def clamp(x, lo, hi): return _ternary("clamp", x, lo, hi)
def fma(a, b, c): return _ternary("fma", a, b, c)
def mix(a, b, t): return _ternary("mix", a, b, t)
def smoothstep(e0, e1, x): return _ternary("smoothstep", e0, e1, x)


# --- Float predicates (return i1) ---
def isnan(x): return _unary("isnan", x, result_dtype="i1")
def isinf(x): return _unary("isinf", x, result_dtype="i1")
def isfinite(x): return _unary("isfinite", x, result_dtype="i1")
def signbit(x): return _unary("signbit", x, result_dtype="i1")
def isnormal(x): return _unary("isnormal", x, result_dtype="i1")


# --- Select + int min/max/clamp ---
def select(false_val, true_val, condition) -> IRValue:
    builder = get_builder()
    assert builder is not None
    false_val = _ensure_ir(false_val)
    true_val = _ensure_ir(true_val)
    condition = _ensure_ir(condition)
    result = builder.new_value(true_val.dtype)
    builder.record(IROp("select", result, [false_val, true_val, condition]))
    return result

def imin(a, b): return _binop("imin", a, b)
def imax(a, b): return _binop("imax", a, b)
def iclamp(x, lo, hi): return _ternary("iclamp", x, lo, hi)


# --- Integer bit ops ---
def popcount(x): return _unary("popcount", x)
def clz(x): return _unary("clz", x)
def ctz(x): return _unary("ctz", x)
def reverse_bits(x): return _unary("reverse_bits", x)
def abs_diff_unary(x): return _unary("abs_diff_unary", x)
def abs_diff(a, b): return _binop("abs_diff", a, b)
def add_sat(a, b): return _binop("add_sat", a, b)
def sub_sat(a, b): return _binop("sub_sat", a, b)
def mul_hi(a, b): return _binop("mul_hi", a, b)
def rotate(a, b): return _binop("rotate", a, b)
def mad_sat(a, b, c): return _ternary("mad_sat", a, b, c)

def extract_bits(value, offset: int, bits: int) -> IRValue:
    builder = get_builder()
    assert builder is not None
    value = _ensure_ir(value)
    result = builder.new_value(value.dtype)
    builder.record(IROp("extract_bits", result, [value],
                        attrs={"offset": int(offset), "bits": int(bits)}))
    return result

def insert_bits(base, insert, offset: int, bits: int) -> IRValue:
    builder = get_builder()
    assert builder is not None
    base = _ensure_ir(base); insert = _ensure_ir(insert)
    result = builder.new_value(base.dtype)
    builder.record(IROp("insert_bits", result, [base, insert],
                        attrs={"offset": int(offset), "bits": int(bits)}))
    return result


# --- SIMD group ops ---
def simd_sum(x): return _unary("simd_sum", x)
def simd_product(x): return _unary("simd_product", x)
def simd_min(x): return _unary("simd_min", x)
def simd_max(x): return _unary("simd_max", x)
def simd_and(x): return _unary("simd_and", x)
def simd_or(x): return _unary("simd_or", x)
def simd_xor(x): return _unary("simd_xor", x)
def simd_prefix_exclusive_sum(x): return _unary("simd_prefix_exclusive_sum", x)
def simd_prefix_inclusive_sum(x): return _unary("simd_prefix_inclusive_sum", x)
def simd_prefix_exclusive_product(x): return _unary("simd_prefix_exclusive_product", x)
def simd_prefix_inclusive_product(x): return _unary("simd_prefix_inclusive_product", x)

def _simd_shuffle(op_type: str, value, index) -> IRValue:
    builder = get_builder()
    assert builder is not None
    value = _ensure_ir(value); index = _ensure_ir(index)
    result = builder.new_value(value.dtype)
    builder.record(IROp(op_type, result, [value, index]))
    return result

def simd_shuffle(value, lane): return _simd_shuffle("simd_shuffle", value, lane)
def simd_shuffle_up(value, delta): return _simd_shuffle("simd_shuffle_up", value, delta)
def simd_shuffle_down(value, delta): return _simd_shuffle("simd_shuffle_down", value, delta)
def simd_shuffle_xor(value, mask): return _simd_shuffle("simd_shuffle_xor", value, mask)
def simd_broadcast(value, lane): return _simd_shuffle("simd_broadcast", value, lane)


# --- Quad group ops (4-thread pixel quads) ---
def quad_sum(x): return _unary("quad_sum", x)
def quad_product(x): return _unary("quad_product", x)
def quad_min(x): return _unary("quad_min", x)
def quad_max(x): return _unary("quad_max", x)
def quad_and(x): return _unary("quad_and", x)
def quad_or(x): return _unary("quad_or", x)
def quad_xor(x): return _unary("quad_xor", x)
def quad_prefix_exclusive_sum(x): return _unary("quad_prefix_exclusive_sum", x)
def quad_prefix_inclusive_sum(x): return _unary("quad_prefix_inclusive_sum", x)

def quad_shuffle(value, lane): return _simd_shuffle("quad_shuffle", value, lane)
def quad_shuffle_up(value, delta): return _simd_shuffle("quad_shuffle_up", value, delta)
def quad_shuffle_down(value, delta): return _simd_shuffle("quad_shuffle_down", value, delta)
def quad_shuffle_xor(value, mask): return _simd_shuffle("quad_shuffle_xor", value, mask)
def quad_broadcast(value, lane): return _simd_shuffle("quad_broadcast", value, lane)


# --- Comparison ops (arith.cmpi / arith.cmpf) ---
# Result dtype is always "i1" (Metal's `bool`). Integer predicates are signed;
# Metal int is signed by default. Use u_* variants for unsigned semantics.

def _cmp(op_type: str, a, b) -> IRValue:
    builder = _require_builder(op_type)
    a, b = _ensure_ir(a), _ensure_ir(b)
    result = builder.new_value("i1")
    builder.record(IROp(op_type, result, [a, b]))
    return result

def cmp_eq(a, b): return _cmp("cmp_eq", a, b)
def cmp_ne(a, b): return _cmp("cmp_ne", a, b)
def cmp_lt(a, b): return _cmp("cmp_lt", a, b)
def cmp_le(a, b): return _cmp("cmp_le", a, b)
def cmp_gt(a, b): return _cmp("cmp_gt", a, b)
def cmp_ge(a, b): return _cmp("cmp_ge", a, b)
def cmp_ult(a, b): return _cmp("cmp_ult", a, b)
def cmp_ule(a, b): return _cmp("cmp_ule", a, b)
def cmp_ugt(a, b): return _cmp("cmp_ugt", a, b)
def cmp_uge(a, b): return _cmp("cmp_uge", a, b)


# --- Grid / thread query ops (x/y/z selector) ---
# Each returns an IRValue of dtype "uint" carrying the chosen dimension as
# an attr. The emitter maps to the corresponding dialect op.

_DIM_VALID = ("x", "y", "z")

def _grid_query(op_type: str, dim: str) -> IRValue:
    builder = _require_builder(op_type)
    if dim not in _DIM_VALID:
        raise EnigmaError(f"{op_type}: dim must be 'x'/'y'/'z', got {dim!r}")
    result = builder.new_value("uint")
    builder.record(IROp(op_type, result, [], attrs={"dim": dim}))
    return result

def thread_position_in_grid_xyz(dim: str = "x"):
    return _grid_query("thread_position_in_grid", dim)
def thread_position_in_threadgroup(dim: str = "x"):
    return _grid_query("thread_position_in_threadgroup", dim)
def threadgroup_position_in_grid(dim: str = "x"):
    return _grid_query("threadgroup_position_in_grid", dim)
def threads_per_threadgroup(dim: str = "x"):
    return _grid_query("threads_per_threadgroup", dim)
def threads_per_grid(dim: str = "x"):
    return _grid_query("threads_per_grid", dim)
def threadgroups_per_grid(dim: str = "x"):
    return _grid_query("threadgroups_per_grid", dim)
def grid_size(dim: str = "x"):
    return _grid_query("grid_size", dim)
def thread_index_in_threadgroup():
    return _grid_query("thread_index_in_threadgroup", "x")
def thread_index_in_simdgroup():
    return _grid_query("thread_index_in_simdgroup", "x")
def simdgroup_index_in_threadgroup():
    return _grid_query("simdgroup_index_in_threadgroup", "x")
def threads_per_simdgroup():
    return _grid_query("threads_per_simdgroup", "x")
def simdgroups_per_threadgroup():
    return _grid_query("simdgroups_per_threadgroup", "x")


# --- Function constants (Metal specialization constants) ---
# function_constant(dtype, index) produces an IRValue bound to
# `[[function_constant(index)]]` at pipeline creation time.

def function_constant(dtype: str, index: int) -> IRValue:
    builder = _require_builder("function_constant")
    result = builder.new_value(dtype)
    builder.record(IROp("function_constant", result, [],
                        attrs={"index": int(index), "dtype": dtype}))
    return result


# --- Regular matrix ops on vector<CxRxT> ---
# The dialect models MSL matrix types (float4x4 etc) as multi-dim vector types.
# These ops operate on those types; construction is via IR constants / loads.

def matmul(a: IRValue, b: IRValue, result_dtype: Optional[str] = None) -> IRValue:
    builder = _require_builder("matmul")
    result = builder.new_value(result_dtype or a.dtype)
    builder.record(IROp("matmul", result, [a, b]))
    return result

def transpose(m: IRValue, result_dtype: Optional[str] = None) -> IRValue:
    builder = _require_builder("transpose")
    result = builder.new_value(result_dtype or m.dtype)
    builder.record(IROp("transpose", result, [m]))
    return result

def determinant(m: IRValue, scalar_dtype: Optional[str] = None) -> IRValue:
    builder = _require_builder("determinant")
    # Determinant returns a scalar; caller usually knows the element dtype.
    dt = scalar_dtype
    if dt is None:
        parsed = parse_vec_dtype(m.dtype)
        dt = parsed[1] if parsed is not None else m.dtype
    result = builder.new_value(dt)
    builder.record(IROp("determinant", result, [m]))
    return result


# --- Simdgroup matrix ops (hardware 8x8 matrix units) ---

def _simdgroup_mat_dtype(elem: str, rows: int = 8, cols: int = 8) -> str:
    return f"simdgroup_matrix<{rows},{cols},{elem}>"


def _parse_simdgroup_mat_dtype(dt: str):
    """Return (rows, cols, elem) if dt is a simdgroup_matrix dtype, else None."""
    if not (isinstance(dt, str) and dt.startswith("simdgroup_matrix<") and dt.endswith(">")):
        return None
    body = dt[len("simdgroup_matrix<"):-1]
    parts = body.split(",")
    return int(parts[0]), int(parts[1]), parts[2].strip()


def simdgroup_matrix_load(
    buf: "Tensor", elements_per_row: int, elem: str = "float",
    rows: int = 8, cols: int = 8,
) -> IRValue:
    builder = _require_builder("simdgroup_matrix_load")
    dt = _simdgroup_mat_dtype(elem, rows, cols)
    result = builder.new_value(dt)
    builder.record(IROp("simdgroup_matrix_load", result, [],
                        attrs={**buf._attrs(), "elements_per_row": int(elements_per_row),
                               "elem": elem, "rows": rows, "cols": cols}))
    return result


def simdgroup_matrix_store(
    matrix: IRValue, buf: "Tensor", elements_per_row: int,
) -> None:
    builder = _require_builder("simdgroup_matrix_store")
    parsed = _parse_simdgroup_mat_dtype(matrix.dtype)
    assert parsed is not None, f"Expected simdgroup_matrix dtype, got {matrix.dtype}"
    builder.record(IROp("simdgroup_matrix_store", None, [matrix],
                        attrs={**buf._attrs(), "elements_per_row": int(elements_per_row),
                               "rows": parsed[0], "cols": parsed[1], "elem": parsed[2]}))


def simdgroup_multiply_accumulate(
    a: IRValue, b: IRValue, c: IRValue,
) -> IRValue:
    builder = _require_builder("simdgroup_multiply_accumulate")
    result = builder.new_value(c.dtype)
    builder.record(IROp("simdgroup_multiply_accumulate", result, [a, b, c]))
    return result


def make_filled_simdgroup_matrix(
    value: IRValue, elem: str = "float", rows: int = 8, cols: int = 8,
) -> IRValue:
    builder = _require_builder("make_filled_simdgroup_matrix")
    value = _ensure_ir(value)
    dt = _simdgroup_mat_dtype(elem, rows, cols)
    result = builder.new_value(dt)
    builder.record(IROp("make_filled_simdgroup_matrix", result, [value],
                        attrs={"elem": elem, "rows": rows, "cols": cols}))
    return result


# --- Cast ops ---
def metal_cast(x, dtype: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    x = _ensure_ir(x)
    result = builder.new_value(dtype)
    builder.record(IROp("metal_cast", result, [x], attrs={"target_dtype": dtype}))
    return result

def as_type(x, dtype: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    x = _ensure_ir(x)
    result = builder.new_value(dtype)
    builder.record(IROp("as_type", result, [x], attrs={"target_dtype": dtype}))
    return result


# --- Barriers ---
def barrier(mem_flags: str = "mem_threadgroup") -> None:
    builder = get_builder()
    assert builder is not None
    builder.record(IROp("threadgroup_barrier", None, [], attrs={"mem_flags": mem_flags}))

def simd_barrier(mem_flags: str = "mem_threadgroup") -> None:
    builder = get_builder()
    assert builder is not None
    builder.record(IROp("simdgroup_barrier", None, [], attrs={"mem_flags": mem_flags}))


# ============================================================================
# Vector / SIMD values — float2/3/4, half2/3/4, int2/3/4, uint2/3/4
# ============================================================================
# Represented by an IRValue whose dtype is a string of the form
# "vec<N,elem>" where N in {2,3,4} and elem in {float, half, int, uint}.
# Elementwise + - * / are inherited from IRValue and lowered to arith ops on
# the vector MLIR type (MLIR arith supports vector-of-scalar natively).

def _vec_dtype(elem: str, n: int) -> str:
    return f"vec<{n},{elem}>"


def parse_vec_dtype(dt: str):
    """Return (N, elem) if dt is a vec dtype, else None."""
    if not (isinstance(dt, str) and dt.startswith("vec<") and dt.endswith(">")):
        return None
    body = dt[4:-1]
    n_s, elem = body.split(",", 1)
    return int(n_s), elem.strip()


def make_vec(*components: IRValue) -> IRValue:
    """Assemble a vector from 2, 3, or 4 scalars (must have same dtype)."""
    assert len(components) in (2, 3, 4), "make_vec expects 2, 3, or 4 scalars"
    builder = get_builder()
    assert builder is not None
    components = tuple(_ensure_ir(c) for c in components)
    elem = components[0].dtype
    for c in components:
        assert c.dtype == elem, f"make_vec: mixed dtypes {elem} vs {c.dtype}"
    result = builder.new_value(_vec_dtype(elem, len(components)))
    builder.record(IROp("vec_make", result, list(components),
                        attrs={"elem": elem, "n": len(components)}))
    return result


def vec_extract(v: IRValue, lane: int) -> IRValue:
    """Extract one scalar element from a vec value."""
    builder = get_builder()
    assert builder is not None
    parsed = parse_vec_dtype(v.dtype)
    assert parsed is not None, f"vec_extract expects a vec dtype, got {v.dtype}"
    n, elem = parsed
    assert 0 <= lane < n
    result = builder.new_value(elem)
    builder.record(IROp("vec_extract", result, [v],
                        attrs={"lane": int(lane), "elem": elem, "n": n}))
    return result


class _VecAccessor:
    """Descriptor-style .x/.y/.z/.w on IRValue for vec dtypes."""
    __slots__ = ("_lane",)
    def __init__(self, lane): self._lane = lane
    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return vec_extract(obj, self._lane)


IRValue.x = _VecAccessor(0)
IRValue.y = _VecAccessor(1)
IRValue.z = _VecAccessor(2)
IRValue.w = _VecAccessor(3)


def make_float2(x, y): return make_vec(x, y)
def make_float3(x, y, z): return make_vec(x, y, z)
def make_float4(x, y, z, w): return make_vec(x, y, z, w)


# --- Pack / Unpack (vec <-> packed int) ---

_UNPACK_OPS = {
    "unpack_snorm4x8_to_float":     ("float", 4),
    "unpack_unorm4x8_to_float":     ("float", 4),
    "unpack_snorm2x16_to_float":    ("float", 2),
    "unpack_unorm2x16_to_float":    ("float", 2),
    "unpack_srgb_unorm4x8_to_float":("float", 4),
    "unpack_unorm10a2_to_float":    ("float", 4),
}


def _pack(op_type: str, v: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    result = builder.new_value("uint")
    builder.record(IROp(op_type, result, [v]))
    return result


def _unpack(op_type: str, x: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    _elem, n = _UNPACK_OPS[op_type]
    x = _ensure_ir(x)
    result = builder.new_value(_vec_dtype(_elem, n))
    builder.record(IROp(op_type, result, [x], attrs={"elem": _elem, "n": n}))
    return result


def pack_float_to_snorm4x8(v): return _pack("pack_float_to_snorm4x8", v)
def pack_float_to_unorm4x8(v): return _pack("pack_float_to_unorm4x8", v)
def pack_float_to_snorm2x16(v): return _pack("pack_float_to_snorm2x16", v)
def pack_float_to_unorm2x16(v): return _pack("pack_float_to_unorm2x16", v)
def pack_float_to_srgb_unorm4x8(v): return _pack("pack_float_to_srgb_unorm4x8", v)
def pack_float_to_unorm10a2(v): return _pack("pack_float_to_unorm10a2", v)

def unpack_snorm4x8_to_float(x): return _unpack("unpack_snorm4x8_to_float", x)
def unpack_unorm4x8_to_float(x): return _unpack("unpack_unorm4x8_to_float", x)
def unpack_snorm2x16_to_float(x): return _unpack("unpack_snorm2x16_to_float", x)
def unpack_unorm2x16_to_float(x): return _unpack("unpack_unorm2x16_to_float", x)
def unpack_srgb_unorm4x8_to_float(x): return _unpack("unpack_srgb_unorm4x8_to_float", x)
def unpack_unorm10a2_to_float(x): return _unpack("unpack_unorm10a2_to_float", x)


# --- Geometry ops ---

def dot(a: IRValue, b: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    parsed = parse_vec_dtype(a.dtype)
    assert parsed is not None
    _n, elem = parsed
    result = builder.new_value(elem)
    builder.record(IROp("dot", result, [a, b]))
    return result


def length(v: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    parsed = parse_vec_dtype(v.dtype)
    assert parsed is not None
    _n, elem = parsed
    result = builder.new_value(elem)
    builder.record(IROp("length", result, [v]))
    return result


def distance(a: IRValue, b: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    parsed = parse_vec_dtype(a.dtype)
    assert parsed is not None
    _n, elem = parsed
    result = builder.new_value(elem)
    builder.record(IROp("distance", result, [a, b]))
    return result


def cross(a: IRValue, b: IRValue) -> IRValue:
    return _unary("cross", a) if False else _vec_binop("cross", a, b)


def normalize(v: IRValue) -> IRValue:
    return _unary("normalize", v)


def reflect(incident: IRValue, normal: IRValue) -> IRValue:
    return _vec_binop("reflect", incident, normal)


def refract(incident: IRValue, normal: IRValue, eta) -> IRValue:
    builder = get_builder()
    assert builder is not None
    eta = _ensure_ir(eta)
    result = builder.new_value(incident.dtype)
    builder.record(IROp("refract", result, [incident, normal, eta]))
    return result


def faceforward(n: IRValue, incident: IRValue, nref: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    result = builder.new_value(n.dtype)
    builder.record(IROp("faceforward", result, [n, incident, nref]))
    return result


def _vec_binop(op_type: str, a: IRValue, b: IRValue) -> IRValue:
    builder = get_builder()
    assert builder is not None
    result = builder.new_value(a.dtype)
    builder.record(IROp(op_type, result, [a, b]))
    return result

def _tv_aware_binop(scalar_op: str, tv_op: str, lhs, rhs) -> IRValue:
    """Route to TV or scalar binop: TV×TV, TV×scalar (broadcast), or scalar×scalar."""
    builder = get_builder()
    assert builder is not None
    lhs, rhs = _ensure_ir(lhs), _ensure_ir(rhs)

    l_tv = isinstance(lhs, IRValue) and lhs._tv_groups is not None
    r_tv = isinstance(rhs, IRValue) and rhs._tv_groups is not None

    if l_tv and r_tv:
        return _tv_binop(tv_op, lhs, rhs)
    if l_tv and not r_tv:
        return _tv_scalar_binop(tv_op, lhs, rhs)
    if not l_tv and r_tv:
        return _tv_scalar_binop(tv_op, rhs, lhs, scalar_on_left=True)
    return _binop(scalar_op, lhs, rhs)


def _tv_binop(op_type: str, lhs: IRValue, rhs: IRValue) -> IRValue:
    """Binary op on two TV-grouped values, preserving group structure."""
    builder = get_builder()
    assert builder is not None
    result = builder.new_value(lhs.dtype)
    result._tv_groups = lhs._tv_groups
    builder.record(
        IROp(
            op_type,
            result,
            [lhs, rhs],
            attrs={
                "groups": lhs._tv_groups,
                "dtype": lhs.dtype,
            },
        )
    )
    return result


def _tv_scalar_binop(
    op_type: str, tv_val: IRValue, scalar_val: IRValue,
    scalar_on_left: bool = False,
) -> IRValue:
    """TV × scalar broadcast. scalar_on_left=True for scalar OP tv (e.g. scalar / tv)."""
    builder = get_builder()
    assert builder is not None
    result = builder.new_value(tv_val.dtype)
    result._tv_groups = tv_val._tv_groups
    builder.record(
        IROp(
            op_type,
            result,
            [tv_val, scalar_val],
            attrs={
                "groups": tv_val._tv_groups,
                "dtype": tv_val.dtype,
                "broadcast_scalar": True,
                "scalar_on_left": scalar_on_left,
            },
        )
    )
    return result


@dataclass
class IROp:
    op_type: str
    result: Optional[IRValue]
    operands: List[Any] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    regions: List[List["IROp"]] = field(default_factory=list)


from .tensor import Tensor


# --- Atomic helpers (free functions + Tensor methods use these) ---

def _atomic_load(buf: "Tensor", index, order: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index)
    result = builder.new_value(buf.metal_dtype)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp("atomic_load", result, [index], attrs=attrs))
    return result

def _atomic_store(buf: "Tensor", index, value, order: str) -> None:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index); value = _ensure_ir(value)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp("atomic_store", None, [index, value], attrs=attrs))

def _atomic_rmw(op_type: str, buf: "Tensor", index, value, order: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index); value = _ensure_ir(value)
    result = builder.new_value(buf.metal_dtype)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp(op_type, result, [index, value], attrs=attrs))
    return result

def _atomic_cas(buf: "Tensor", index, expected, desired,
                success_order: str, failure_order: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index)
    expected = _ensure_ir(expected); desired = _ensure_ir(desired)
    result = builder.new_value("i1")
    attrs = buf._attrs()
    attrs["success_order"] = success_order
    attrs["failure_order"] = failure_order
    builder.record(IROp("atomic_compare_exchange_weak", result,
                        [index, expected, desired], attrs=attrs))
    return result


def atomic_load(buf, index, order="relaxed"): return buf.atomic_load(index, order)
def atomic_store(buf, index, value, order="relaxed"): buf.atomic_store(index, value, order)
def atomic_exchange(buf, index, value, order="relaxed"): return buf.atomic_exchange(index, value, order)
def atomic_fetch_add(buf, index, value, order="relaxed"): return buf.atomic_fetch_add(index, value, order)
def atomic_fetch_sub(buf, index, value, order="relaxed"): return buf.atomic_fetch_sub(index, value, order)
def atomic_fetch_min(buf, index, value, order="relaxed"): return buf.atomic_fetch_min(index, value, order)
def atomic_fetch_max(buf, index, value, order="relaxed"): return buf.atomic_fetch_max(index, value, order)
def atomic_fetch_and(buf, index, value, order="relaxed"): return buf.atomic_fetch_and(index, value, order)
def atomic_fetch_or(buf, index, value, order="relaxed"): return buf.atomic_fetch_or(index, value, order)
def atomic_fetch_xor(buf, index, value, order="relaxed"): return buf.atomic_fetch_xor(index, value, order)
def atomic_compare_exchange_weak(buf, index, expected, desired,
                                 success_order="relaxed", failure_order="relaxed"):
    return buf.atomic_compare_exchange_weak(index, expected, desired,
                                            success_order, failure_order)


# --- Threadgroup shared memory ---
_shared_counter = 0

def threadgroup_alloc(dtype: str, size: int) -> "Tensor":
    """Allocate threadgroup-shared memory. Returns a Tensor you can
    load/store into and pass to atomics. Must be called inside @enigma.kernel.
    """
    global _shared_counter
    builder = get_builder()
    assert builder is not None, "threadgroup_alloc() only inside @enigma.kernel"
    _shared_counter += 1
    name = f"_shared{_shared_counter}"
    builder.record(IROp("threadgroup_alloc", None, [],
                        attrs={"buffer": name, "dtype": dtype, "size": int(size)}))
    return Tensor(name, -1, dtype, address_space="threadgroup", shape=int(size))


class KernelBuilder:
    """Accumulates traced IR operations for one kernel.

    Supports nested regions for control flow: ``_region_stack`` holds a stack
    of op-lists.  ``record()`` always appends to the top of the stack.
    Control-flow context managers push a new list when entering a body and
    pop it when leaving (attaching it as a region on the parent op).
    """

    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.ops: List[IROp] = []
        self.args: List[Tuple[str, int, str]] = []
        self._counter = 0
        self._tid_value: Optional[IRValue] = None
        self._const_cache: Dict[tuple, IRValue] = {}
        # Region stack: top element is where record() appends.
        # Starts with self.ops as the root region.
        self._region_stack: List[List[IROp]] = [self.ops]

    def new_value(self, dtype: str) -> IRValue:
        name = f"_v{self._counter}"
        self._counter += 1
        return IRValue(name, dtype)

    def record(self, op: IROp) -> None:
        self._region_stack[-1].append(op)

    def _push_region(self) -> List[IROp]:
        """Push a new region onto the stack and return it."""
        region: List[IROp] = []
        self._region_stack.append(region)
        return region

    def _pop_region(self) -> List[IROp]:
        """Pop the current region from the stack and return it."""
        assert len(self._region_stack) > 1, "Cannot pop root region"
        return self._region_stack.pop()

    def get_thread_position_in_grid(self) -> IRValue:
        if self._tid_value is None:
            self._tid_value = IRValue("tid", "uint")
            self.record(IROp("thread_position_in_grid", self._tid_value, []))
        return self._tid_value

    _tidx: Optional[IRValue] = None
    _bidx: Optional[IRValue] = None
    _bdim: Optional[IRValue] = None

    def get_thread_idx(self) -> IRValue:
        if self._tidx is None:
            self._tidx = IRValue("tidx", "uint")
            self.record(IROp("thread_position_in_threadgroup", self._tidx, []))
        return self._tidx

    def get_block_idx(self) -> IRValue:
        if self._bidx is None:
            self._bidx = IRValue("bidx", "uint")
            self.record(IROp("threadgroup_position_in_grid", self._bidx, []))
        return self._bidx

    def get_block_dim(self) -> IRValue:
        if self._bdim is None:
            self._bdim = IRValue("bdim", "uint")
            self.record(IROp("threads_per_threadgroup", self._bdim, []))
        return self._bdim

    def make_const(self, dtype: str, value: int) -> IRValue:
        key = (dtype, value)
        if key in self._const_cache:
            return self._const_cache[key]
        val = IRValue(f"_c{value}", dtype)
        self.record(IROp("const", val, [], attrs={"value": value}))
        self._const_cache[key] = val
        return val

    def __enter__(self):
        _local.builder = self
        return self

    def __exit__(self, *exc):
        _local.builder = None


# ============================================================================
# Control flow — for_range / if_ / while_
# ============================================================================
# These trace to structured IR ops with nested regions, which the MLIR
# emitter lowers to scf.for / scf.if / scf.while.


class Carry:
    """Mutable container of loop-carried IRValues for ``enigma.for_range(init=[...])``.

    Inside the loop body, ``carry[i]`` reads the current per-iteration SSA
    value (the iter_arg).  Assigning ``carry[i] = new_val`` updates the
    slot, and the last-written value is yielded at the end of the iteration.

    After the loop closes, ``carry[i]`` reads the loop's *result* (the
    value after the final iteration), so you can use the carried value
    in code following the ``with`` block.
    """

    __slots__ = ("_slots", "_closed")

    def __init__(self, initial: List[IRValue]):
        self._slots: List[IRValue] = list(initial)
        self._closed = False

    def __len__(self) -> int:
        return len(self._slots)

    def __getitem__(self, i: int) -> IRValue:
        return self._slots[i]

    def __setitem__(self, i: int, value) -> None:
        if self._closed:
            raise EnigmaError(
                "Cannot assign to carry[] outside the for_range body"
            )
        self._slots[i] = _ensure_ir(value)

    def _close_and_rebind(self, new_slots: List[IRValue]) -> None:
        self._slots = list(new_slots)
        self._closed = True


class _ForRangeContext:
    """Context manager for ``enigma.for_range(lo, hi, step, init=[...])``.

    Without ``init`` — no loop-carried values::

        with enigma.for_range(0, K, step=1) as i:
            Out[i] = A[i] + B[i]

    With ``init`` — accumulator-style loops::

        with enigma.for_range(0, K, init=[acc0]) as (i, carry):
            carry[0] = carry[0] + A[i] * B[i]
        acc_final = carry[0]  # scf.for result after the loop

    The induction variable ``i`` is an ``IRValue`` of dtype ``"int"``
    (index-typed in MLIR).  The body ops are captured in a region and
    attached to a ``scf_for`` IROp.
    """

    def __init__(self, lo, hi, step=1, dtype: str = "int", init=None):
        self._builder = _require_builder("for_range")
        self._lo = _ensure_ir(lo) if not isinstance(lo, IRValue) else lo
        self._hi = _ensure_ir(hi) if not isinstance(hi, IRValue) else hi
        self._step = _ensure_ir(step) if not isinstance(step, IRValue) else step
        self._dtype = dtype
        # The induction variable — available to user code inside the body.
        self._iv = self._builder.new_value(dtype)
        self._iv.name = f"_iv{self._builder._counter}"

        # Loop-carried values: the caller-supplied initial SSA values live
        # outside the loop; we generate a fresh IRValue for each to serve
        # as the iter_arg (body-local name).
        if init is None:
            self._init_vals: List[IRValue] = []
            self._iter_args: List[IRValue] = []
            self._carry: Optional[Carry] = None
        else:
            self._init_vals = [_ensure_ir(v) for v in init]
            self._iter_args = []
            for iv in self._init_vals:
                arg = self._builder.new_value(iv.dtype)
                arg.name = f"_ia{self._builder._counter}"
                self._iter_args.append(arg)
            self._carry = Carry(self._iter_args)
            # Placeholder IRValues that will become the for-op results
            # after __exit__ rebinds them. They're reserved up-front so
            # the scf_for op has a stable list of result names.
            self._result_vals: List[IRValue] = [
                self._builder.new_value(iv.dtype) for iv in self._init_vals
            ]

    def __enter__(self):
        self._body = self._builder._push_region()
        if self._carry is None:
            return self._iv
        return (self._iv, self._carry)

    def __exit__(self, *exc):
        self._builder._pop_region()
        if self._carry is None:
            op = IROp(
                "scf_for", None,
                [self._lo, self._hi, self._step],
                attrs={"iv": self._iv, "dtype": self._dtype},
                regions=[self._body],
            )
            self._builder.record(op)
            return

        # With iter_args: final slot values are the scf.yield operands.
        # The op has one result per init value; rebind the Carry to the
        # results so user code outside the loop reads the right SSA.
        op = IROp(
            "scf_for", None,
            [self._lo, self._hi, self._step] + list(self._init_vals),
            attrs={
                "iv": self._iv,
                "dtype": self._dtype,
                "iter_args": list(self._iter_args),
                "yield_vals": list(self._carry._slots),
                "results": list(self._result_vals),
            },
            regions=[self._body],
        )
        self._builder.record(op)
        self._carry._close_and_rebind(self._result_vals)


def for_range(lo, hi, step=1, dtype: str = "int", init=None) -> _ForRangeContext:
    """Trace a ``for`` loop.

    Parameters
    ----------
    lo : int or IRValue
        Loop lower bound (inclusive).
    hi : int or IRValue
        Loop upper bound (exclusive).
    step : int or IRValue
        Loop step (default 1).
    dtype : str
        Type of the induction variable (default ``"int"``).
    init : list of IRValue or int, optional
        Initial values for loop-carried variables. If given, the context
        manager returns ``(iv, carry)`` where ``carry`` is a mutable
        ``Carry`` slot list. Write ``carry[k] = new_val`` inside the body
        to update; read ``carry[k]`` outside the loop for the final value.

    Returns a context manager::

        # No carries:
        with enigma.for_range(0, K) as i:
            ...

        # With carries (accumulator):
        with enigma.for_range(0, K, init=[zero]) as (i, carry):
            carry[0] = carry[0] + A[i]
        total = carry[0]
    """
    return _ForRangeContext(lo, hi, step, dtype, init=init)


class _IfContext:
    """Context manager for ``enigma.if_(condition)``.

    Usage — if-only::

        with enigma.if_(cond):
            Out[tid] = a

    Usage — if/else::

        with enigma.if_(cond) as (then_block, else_block):
            with then_block:
                Out[tid] = a
            with else_block:
                Out[tid] = b

    When used without unpacking (no ``as``), the body is the then-branch
    with no else.  When unpacked into two blocks, each block is a context
    manager for its region.
    """

    def __init__(self, condition: IRValue):
        self._builder = _require_builder("if_")
        self._condition = _ensure_ir(condition) if not isinstance(condition, IRValue) else condition
        self._then_body: List[IROp] = []
        self._else_body: List[IROp] = []
        self._used_blocks = False

    def __enter__(self):
        # Return (then_block, else_block) so the user can choose.
        # If they ignore the return value, we treat the whole `with` body as
        # the then-branch by pushing a region immediately.
        self._then_ctx = _RegionBlock(self._builder, self._then_body)
        self._else_ctx = _RegionBlock(self._builder, self._else_body)
        # Push the then region immediately — if the user unpacks and uses
        # the block context managers, they'll push/pop themselves.
        # We detect the pattern in __exit__: if _then_ctx was never
        # explicitly entered, we treat the whole body as the then-branch.
        self._auto_then = self._builder._push_region()
        return (self._then_ctx, self._else_ctx)

    def __exit__(self, *exc):
        # If the user used `with enigma.if_(c):` without unpacking blocks,
        # everything recorded went into self._auto_then.
        auto_ops = self._builder._pop_region()

        if self._then_ctx._entered:
            # User used explicit blocks — auto_then should be empty.
            then_ops = self._then_body
            else_ops = self._else_body
        else:
            # Simple `with enigma.if_(c):` — auto_then is the then-branch.
            then_ops = auto_ops
            else_ops = []

        regions = [then_ops]
        if else_ops:
            regions.append(else_ops)

        op = IROp(
            "scf_if", None,
            [self._condition],
            attrs={"has_else": bool(else_ops)},
            regions=regions,
        )
        self._builder.record(op)


class _RegionBlock:
    """Helper context manager for an individual then/else block."""

    def __init__(self, builder: KernelBuilder, target: List[IROp]):
        self._builder = builder
        self._target = target
        self._entered = False

    def __enter__(self):
        self._entered = True
        self._builder._region_stack.append(self._target)
        return self

    def __exit__(self, *exc):
        self._builder._pop_region()


def if_(condition) -> _IfContext:
    """Trace a conditional (if/else).

    Parameters
    ----------
    condition : IRValue
        Boolean condition (i1 dtype).

    Two usage patterns::

        # If-only (no else):
        with enigma.if_(cond):
            Out[tid] = a

        # If/else:
        with enigma.if_(cond) as (then_b, else_b):
            with then_b:
                Out[tid] = a
            with else_b:
                Out[tid] = b
    """
    return _IfContext(condition)


class _WhileContext:
    """Context manager for ``enigma.while_(cond_fn)``.

    Usage::

        def cond():
            return enigma.cmp_lt(i_val, n)

        with enigma.while_(cond) as loop:
            # loop body
            ...

    The condition function is called once at trace time to capture the
    condition ops into the ``before`` region.  The body of the ``with``
    block becomes the ``after`` region.
    """

    def __init__(self, cond_fn):
        self._builder = _require_builder("while_")
        self._cond_fn = cond_fn
        self._before_body: List[IROp] = []
        self._after_body: List[IROp] = []

    def __enter__(self):
        # Trace the condition function into the "before" region.
        self._builder._region_stack.append(self._before_body)
        self._cond_result = self._cond_fn()
        self._builder._pop_region()

        # Now push the "after" (body) region for the with-block.
        self._builder._region_stack.append(self._after_body)
        return self

    def __exit__(self, *exc):
        self._builder._pop_region()

        op = IROp(
            "scf_while", None,
            [],
            attrs={"cond_result": self._cond_result},
            regions=[self._before_body, self._after_body],
        )
        self._builder.record(op)


def while_(cond_fn) -> _WhileContext:
    """Trace a ``while`` loop.

    Parameters
    ----------
    cond_fn : callable
        A zero-argument function that, when called, traces the condition
        ops and returns an ``IRValue`` of dtype ``"i1"`` (boolean).

    Returns a context manager::

        with enigma.while_(lambda: enigma.cmp_lt(i, n)):
            # body
    """
    return _WhileContext(cond_fn)


# ============================================================================
# R5 — Predicated loads / stores
# ============================================================================
# Lower to scf.if around a load/store. load_if returns a default if the
# predicate is false; store_if is a no-op if the predicate is false.


def load_if(buf: "Tensor", index, mask, default=0) -> IRValue:
    """Load ``buf[index]`` if ``mask`` is true, else return ``default``.

    Implementation note: Metal has no masked-load intrinsic.  This emits
    a single unconditional load followed by ``select(default, val, mask)``.
    The load uses ``index`` as-is, so callers must ensure the buffer
    tolerates that read when ``mask`` is false (pad the buffer, or clamp
    ``index`` to a safe value before calling).
    """
    _require_builder("load_if")
    mask = _ensure_ir(mask)
    default_v = _ensure_ir(default)
    result_dtype = buf.metal_dtype
    if default_v.dtype != result_dtype:
        default_v = metal_cast(default_v, result_dtype)
    val = buf[index]
    return select(default_v, val, mask)


def store_if(buf: "Tensor", index, value, mask) -> None:
    """Store ``buf[index] = value`` only when ``mask`` is true.

    Lowers to ``scf.if`` wrapping the store.
    """
    builder = _require_builder("store_if")
    mask = _ensure_ir(mask)
    with _IfContext(mask) as (then_b, _):
        with then_b:
            buf[index] = value


# ============================================================================
# R3 — Tiled copy primitive
# ============================================================================
# Copies `count` elements from src[src_offset + i] to dst[dst_offset + i]
# using an enigma.for_range loop. Respects an optional predicate.


def copy(src: "Tensor", dst: "Tensor", count: int,
         src_offset=0, dst_offset=0, mask_fn=None, coalesced_width: int = 1) -> None:
    """Copy ``count`` elements from ``src`` to ``dst``.

    Parameters
    ----------
    src, dst : Tensor
        Source and destination buffers (device or threadgroup).
    count : int
        Number of elements to copy.
    src_offset, dst_offset : int or IRValue
        Per-buffer base offsets (default 0).
    mask_fn : callable, optional
        Optional ``fn(i) -> i1`` predicate per element. When provided,
        elements with ``mask_fn(i) == false`` are skipped.
    coalesced_width : int
        When > 1 and the dtype + ``count`` permit it, unroll
        ``coalesced_width`` adjacent elements per loop iteration so the
        compiler can fuse them into a single wider load/store. For full
        ``device float4*`` coalescing, also pass ``vec_width=k`` to
        :func:`enigma.compile` (which changes the buffer signature
        itself). Honoured for ``2`` and ``4`` on float/half buffers;
        anything else falls back to scalar.

    Example — copy one tile from device to shared::

        tile = enigma.threadgroup_alloc('float', 256)
        enigma.copy(A, tile, count=256, src_offset=block_start)
        enigma.barrier()
    """
    if coalesced_width not in (1, 2, 4):
        raise EnigmaError(
            f"copy: coalesced_width must be 1, 2, or 4, got {coalesced_width}"
        )

    use_vec = (
        coalesced_width > 1
        and isinstance(count, int)
        and count % coalesced_width == 0
        and src.metal_dtype in ("float", "f32", "half", "f16")
        and dst.metal_dtype in ("float", "f32", "half", "f16")
        and mask_fn is None
        and isinstance(src_offset, int)
        and isinstance(dst_offset, int)
        and src_offset % coalesced_width == 0
        and dst_offset % coalesced_width == 0
    )

    if use_vec:
        n_groups = int(count) // coalesced_width
        with for_range(0, n_groups) as gi:
            base = gi * int(coalesced_width)
            for k in range(coalesced_width):
                dst[base + dst_offset + k] = src[base + src_offset + k]
        return

    with for_range(0, int(count)) as i:
        src_idx = i if isinstance(src_offset, int) and src_offset == 0 else (i + src_offset)
        dst_idx = i if isinstance(dst_offset, int) and dst_offset == 0 else (i + dst_offset)
        if mask_fn is None:
            dst[dst_idx] = src[src_idx]
        else:
            store_if(dst, dst_idx, src[src_idx], mask_fn(i))


# ============================================================================
# R4 — Register-level tensor abstraction
# ============================================================================


class RegisterTensor:
    """A small fixed-size tensor backed by per-thread register SSA values.

    Unlike ``Tensor`` (which sits in device/threadgroup memory),
    a ``RegisterTensor`` is a bag of IRValues kept in scope. Reads return
    the current SSA value; writes rebind the slot.

    Shape is a tuple. Indices must be compile-time ints — registers can't
    be addressed dynamically in MSL without spilling.

    Example::

        acc = enigma.register_tensor((4, 4), dtype='float', fill=0.0)
        with enigma.for_range(0, K) as k:
            ...
            for i in range(4):
                for j in range(4):
                    acc[i, j] = enigma.fma(a[i], b[j], acc[i, j])
    """

    __slots__ = ("shape", "dtype", "_slots", "_strides")

    def __init__(self, shape, dtype: str = "float", fill=0):
        self.shape = tuple(int(s) for s in shape)
        if not self.shape:
            raise EnigmaError("register_tensor: shape cannot be empty")
        self.dtype = dtype
        # Row-major strides.
        strides = []
        acc = 1
        for s in reversed(self.shape):
            strides.append(acc)
            acc *= s
        self._strides = tuple(reversed(strides))
        n = acc
        fill_v = _ensure_ir(fill)
        if fill_v.dtype != dtype:
            fill_v = metal_cast(fill_v, dtype)
        self._slots = [fill_v] * n

    def _flat_index(self, key) -> int:
        if isinstance(key, int):
            key = (key,)
        if not isinstance(key, tuple) or len(key) != len(self.shape):
            raise EnigmaError(
                f"register_tensor indexing: expected {len(self.shape)}-tuple of ints, "
                f"got {key!r}"
            )
        for k, s in zip(key, self.shape):
            if not isinstance(k, int):
                raise EnigmaError(
                    "register_tensor: indices must be Python ints (static); "
                    f"got {type(k).__name__}"
                )
            if k < 0 or k >= s:
                raise EnigmaError(f"register_tensor: index {k} out of range for shape {self.shape}")
        off = 0
        for k, st in zip(key, self._strides):
            off += k * st
        return off

    def __getitem__(self, key) -> IRValue:
        return self._slots[self._flat_index(key)]

    def __setitem__(self, key, value) -> None:
        v = _ensure_ir(value)
        if v.dtype != self.dtype:
            v = metal_cast(v, self.dtype)
        self._slots[self._flat_index(key)] = v


def register_tensor(shape, dtype: str = "float", fill=0) -> RegisterTensor:
    """Create a per-thread register-resident tensor. See :class:`RegisterTensor`."""
    return RegisterTensor(shape, dtype=dtype, fill=fill)


# ============================================================================
# R6 — Async copy (M3+ only)
# ============================================================================
# The dialect ops for `enigma.async_copy_to_threadgroup / commit / wait` are
# not yet present in the wheel. The DSL surface is provided so user code can
# be written today; a clear error guides the user to run on M3+ hardware and
# to wait for dialect support to land.


class _AsyncCopyUnavailable(Exception):
    pass


def _require_m3_runtime(feature: str) -> None:
    """Check device capabilities at trace time (best-effort).

    If a MetalRuntime happens to have been instantiated, use it; otherwise
    skip the check (runtime dispatch will catch it when the kernel is
    launched).
    """
    try:
        from .runtime_dispatch.runtime import MetalRuntime  # noqa: WPS433
    except Exception:
        return
    # Cached capability probe: cheap to re-query, but only do it once.
    global _CACHED_CAPS
    try:
        rt = MetalRuntime()
        caps = rt.device_capabilities()
    except Exception:
        return
    caps.require_m3(feature)


def async_copy_to_threadgroup(
    src: "Tensor", dst: "Tensor",
    count: int, src_offset=0, dst_offset=0,
) -> IRValue:
    """Schedule an async copy from device to threadgroup memory (M3+ only).

    Returns an opaque token ``IRValue`` that must be passed to
    :func:`async_copy_wait`. Calling this on M1/M2 raises at runtime
    when the kernel is launched.
    """
    _require_m3_runtime("async_copy_to_threadgroup")
    builder = _require_builder("async_copy_to_threadgroup")
    tok = builder.new_value("async_token")
    builder.record(IROp(
        "async_copy_to_threadgroup", tok,
        [_ensure_ir(src_offset), _ensure_ir(dst_offset)],
        attrs={
            "src": src._attrs(), "dst": dst._attrs(),
            "count": int(count),
        },
    ))
    return tok


def async_copy_commit(token: IRValue) -> None:
    """Commit a group of previously-issued async copies (M3+ only)."""
    builder = _require_builder("async_copy_commit")
    builder.record(IROp("async_copy_commit", None, [token]))


def async_copy_wait(token: IRValue) -> None:
    """Block until the async copy group completes (M3+ only)."""
    builder = _require_builder("async_copy_wait")
    builder.record(IROp("async_copy_wait", None, [token]))


# ============================================================================
# R7 — Pipeline / double-buffering helper
# ============================================================================
# A lightweight wrapper that alternates between two shared tiles. On M1/M2
# it relies on barriers; on M3+ it can use async_copy for an overlapped
# compute/load pipeline.


class Pipeline:
    """Multi-stage ring buffer of threadgroup tiles.

    Allocates ``stages`` shared buffers and rotates them across loop
    iterations. Index 0 is what the current iteration consumes
    (``front()``); index ``stages-1`` is the buffer being filled for the
    most-distant future iteration (``back()`` / ``stage(stages-1)``).

    Call :meth:`advance` once at the bottom of each iteration to rotate.
    The rotation is purely a Python-side index update; no MSL code is
    emitted by the rotation itself, so this is essentially free.

    Example — three-stage prefetched copy ``compute -> load`` overlap::

        pipe = enigma.pipeline('float', 256, stages=3)
        for stage_id in range(2):
            enigma.copy(A_tile_loader(stage_id), pipe.stage(stage_id), count=256)
        enigma.barrier()

        with enigma.for_range(0, NUM_TILES) as it:
            enigma.barrier()
            consume(pipe.front())                          # iteration `it`
            enigma.copy(A_tile_loader(it + 2),
                        pipe.stage(2), count=256)          # prefetch
            pipe.advance()
    """

    def __init__(self, dtype: str, size: int, stages: int = 2):
        if stages < 2:
            raise EnigmaError(f"Pipeline: stages must be >= 2, got {stages}")
        self.dtype = dtype
        self.size = int(size)
        self.stages = int(stages)
        self._buffers = [threadgroup_alloc(dtype, size) for _ in range(stages)]
        self._phase = 0

    def stage(self, k: int) -> "Tensor":
        """Return the buffer for offset ``k`` from the current front (0..stages-1)."""
        if not (0 <= k < self.stages):
            raise EnigmaError(
                f"Pipeline.stage: k={k} out of range for stages={self.stages}"
            )
        return self._buffers[(self._phase + k) % self.stages]

    def front(self) -> "Tensor":
        return self.stage(0)

    def back(self) -> "Tensor":
        return self.stage(self.stages - 1)

    def advance(self) -> None:
        """Rotate the ring forward by one slot (front becomes back)."""
        self._phase = (self._phase + 1) % self.stages

    # Backwards-compat alias for previous two-stage API.
    def swap(self) -> None:
        if self.stages != 2:
            raise EnigmaError(
                "Pipeline.swap() is only valid for stages=2. Use advance() for >=3."
            )
        self.advance()


def pipeline(dtype: str, size: int, stages: int = 2) -> Pipeline:
    """Create a :class:`Pipeline` for multi-stage tile loads."""
    return Pipeline(dtype=dtype, size=size, stages=stages)
