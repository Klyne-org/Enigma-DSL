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
        if (
            isinstance(other, IRValue)
            and self._tv_groups is not None
            and other._tv_groups is not None
        ):
            return _tv_binop("tv_add", self, other)
        return _binop("add", self, other)

    def __radd__(self, other) -> IRValue:
        if isinstance(other, int) and other == 0:
            return self
        return _binop("add", other, self)

    def __sub__(self, other) -> IRValue:
        return _binop("sub", self, other)

    def __rsub__(self, other) -> IRValue:
        return _binop("sub", other, self)

    def __mul__(self, other) -> IRValue:
        if isinstance(other, int) and other == 1:
            return self
        return _binop("mul", self, other)

    def __rmul__(self, other) -> IRValue:
        if isinstance(other, int) and other == 1:
            return self
        return _binop("mul", other, self)

    def __truediv__(self, other) -> IRValue:
        return _binop("div", self, other)

    def __floordiv__(self, other) -> IRValue:
        return _binop("div", self, other)

    def __mod__(self, other) -> IRValue:
        return _binop("mod", self, other)

    def __neg__(self) -> IRValue:
        builder = get_builder()
        assert builder is not None
        result = builder.new_value(self.dtype)
        builder.record(IROp("neg", result, [self]))
        return result


def _ensure_ir(x) -> IRValue:
    """Wrap a Python int as an IR constant if needed."""
    if isinstance(x, IRValue):
        return x
    if isinstance(x, int):
        builder = get_builder()
        assert builder is not None
        return builder.make_const("uint", x)
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
    buf: "TracingTensor", elements_per_row: int, elem: str = "float",
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
    matrix: IRValue, buf: "TracingTensor", elements_per_row: int,
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

def _tv_binop(op_type: str, lhs: IRValue, rhs: IRValue) -> IRValue:
    """Binary op on TV-vectorized values, preserving group structure."""
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


@dataclass
class IROp:
    op_type: str
    result: Optional[IRValue]
    operands: List[Any] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


class TracingTensor:
    """Proxy tensor for naive kernel tracing (flat 1D indexing).

    ``address_space`` is "device" for kernel-arg buffers and "threadgroup"
    for buffers returned by ``threadgroup_alloc``. ``shape`` is None for
    device buffers (dynamic) and an int for threadgroup allocs (static).
    """

    def __init__(
        self,
        name: str,
        buffer_index: int,
        metal_dtype: str,
        address_space: str = "device",
        shape: Optional[int] = None,
    ):
        self.name = name
        self.buffer_index = buffer_index
        self.metal_dtype = metal_dtype
        self.address_space = address_space
        self.shape = shape

    def _attrs(self) -> Dict[str, Any]:
        return {
            "buffer": self.name,
            "buffer_index": self.buffer_index,
            "address_space": self.address_space,
            "dtype": self.metal_dtype,
            "shape": self.shape,
        }

    def __getitem__(self, index) -> IRValue:
        builder = get_builder()
        assert builder is not None
        index = _ensure_ir(index)
        result = builder.new_value(self.metal_dtype)
        builder.record(IROp("load", result, [index], attrs=self._attrs()))
        return result

    def __setitem__(self, index, value: IRValue) -> None:
        builder = get_builder()
        assert builder is not None
        index = _ensure_ir(index)
        builder.record(IROp("store", None, [index, value], attrs=self._attrs()))

    # --- Atomics (method-style) ---
    def atomic_load(self, index, order: str = "relaxed") -> IRValue:
        return _atomic_load(self, index, order)

    def atomic_store(self, index, value, order: str = "relaxed") -> None:
        _atomic_store(self, index, value, order)

    def atomic_exchange(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_exchange", self, index, value, order)

    def atomic_fetch_add(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_add", self, index, value, order)

    def atomic_fetch_sub(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_sub", self, index, value, order)

    def atomic_fetch_min(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_min", self, index, value, order)

    def atomic_fetch_max(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_max", self, index, value, order)

    def atomic_fetch_and(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_and", self, index, value, order)

    def atomic_fetch_or(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_or", self, index, value, order)

    def atomic_fetch_xor(self, index, value, order: str = "relaxed") -> IRValue:
        return _atomic_rmw("atomic_fetch_xor", self, index, value, order)

    def atomic_compare_exchange_weak(
        self, index, expected, desired,
        success_order: str = "relaxed", failure_order: str = "relaxed",
    ) -> IRValue:
        return _atomic_cas(self, index, expected, desired, success_order, failure_order)


# --- Atomic helpers (free functions + TracingTensor methods use these) ---

def _atomic_load(buf: "TracingTensor", index, order: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index)
    result = builder.new_value(buf.metal_dtype)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp("atomic_load", result, [index], attrs=attrs))
    return result

def _atomic_store(buf: "TracingTensor", index, value, order: str) -> None:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index); value = _ensure_ir(value)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp("atomic_store", None, [index, value], attrs=attrs))

def _atomic_rmw(op_type: str, buf: "TracingTensor", index, value, order: str) -> IRValue:
    builder = get_builder()
    assert builder is not None
    index = _ensure_ir(index); value = _ensure_ir(value)
    result = builder.new_value(buf.metal_dtype)
    attrs = buf._attrs()
    attrs["memory_order"] = order
    builder.record(IROp(op_type, result, [index, value], attrs=attrs))
    return result

def _atomic_cas(buf: "TracingTensor", index, expected, desired,
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

def threadgroup_alloc(dtype: str, size: int) -> "TracingTensor":
    """Allocate threadgroup-shared memory. Returns a TracingTensor you can
    load/store into and pass to atomics. Must be called inside @enigma.kernel.
    """
    global _shared_counter
    builder = get_builder()
    assert builder is not None, "threadgroup_alloc() only inside @enigma.kernel"
    _shared_counter += 1
    name = f"_shared{_shared_counter}"
    builder.record(IROp("threadgroup_alloc", None, [],
                        attrs={"buffer": name, "dtype": dtype, "size": int(size)}))
    return TracingTensor(name, -1, dtype, address_space="threadgroup", shape=int(size))


class KernelBuilder:
    """Accumulates traced IR operations for one kernel."""

    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.ops: List[IROp] = []
        self.args: List[Tuple[str, int, str]] = []
        self._counter = 0
        self._tid_value: Optional[IRValue] = None
        self._const_cache: Dict[tuple, IRValue] = {}

    def new_value(self, dtype: str) -> IRValue:
        name = f"_v{self._counter}"
        self._counter += 1
        return IRValue(name, dtype)

    def record(self, op: IROp) -> None:
        self.ops.append(op)

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
