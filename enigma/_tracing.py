"""Tracing IR for Enigma kernel compilation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_local = threading.local()


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
        if (isinstance(other, IRValue)
                and self._tv_groups is not None
                and other._tv_groups is not None):
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


def _tv_binop(op_type: str, lhs: IRValue, rhs: IRValue) -> IRValue:
    """Binary op on TV-vectorized values, preserving group structure."""
    builder = get_builder()
    assert builder is not None
    result = builder.new_value(lhs.dtype)
    result._tv_groups = lhs._tv_groups
    builder.record(IROp(op_type, result, [lhs, rhs], attrs={
        "groups": lhs._tv_groups, "dtype": lhs.dtype,
    }))
    return result


@dataclass
class IROp:
    op_type: str
    result: Optional[IRValue]
    operands: List[Any] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


class TracingTensor:
    """Proxy tensor for naive kernel tracing (flat 1D indexing)."""

    def __init__(self, name: str, buffer_index: int, metal_dtype: str):
        self.name = name
        self.buffer_index = buffer_index
        self.metal_dtype = metal_dtype

    def __getitem__(self, index: IRValue) -> IRValue:
        builder = get_builder()
        assert builder is not None
        result = builder.new_value(self.metal_dtype)
        builder.record(IROp("load", result, [index],
                            attrs={"buffer": self.name, "buffer_index": self.buffer_index}))
        return result

    def __setitem__(self, index: IRValue, value: IRValue) -> None:
        builder = get_builder()
        assert builder is not None
        builder.record(IROp("store", None, [index, value],
                            attrs={"buffer": self.name, "buffer_index": self.buffer_index}))


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
