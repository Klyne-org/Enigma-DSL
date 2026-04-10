"""Emit Metal C++ source from traced IR."""

from __future__ import annotations
from typing import List, Set

from .._tracing import KernelBuilder, IROp


def _metal_vec_type(base_dtype: str, width: int) -> str:
    return f"{base_dtype}{width}"


def _best_vec_width(num_contiguous: int, dtype: str) -> int:
    for w in (4, 2):
        if num_contiguous >= w:
            return w
    return 1


def _base_expr(base_offset) -> str:
    from .._tracing import IRValue
    return base_offset.name if isinstance(base_offset, IRValue) else str(base_offset)


def emit_metal(builder: KernelBuilder, vec_width: int = 0) -> str:
    """Convert traced IR ops into Metal source text.

    vec_width: if > 0, emit buffer pointers as vector types (e.g. float4*)
               and use tid to index vector elements directly.
    """
    lines = ["#include <metal_stdlib>", "using namespace metal;", ""]

    written_buffers: Set[str] = set()
    for op in builder.ops:
        if op.op_type in ("store", "vec_store", "tv_store"):
            written_buffers.add(op.attrs["buffer"])

    op_types = {op.op_type for op in builder.ops}

    # Kernel signature — use vector pointer types if vec_width > 0
    params: List[str] = []
    for name, buf_idx, metal_dtype in builder.args:
        const = "" if name in written_buffers else "const "
        dtype_str = _metal_vec_type(metal_dtype, vec_width) if vec_width > 0 else metal_dtype
        params.append(f"    device {const}{dtype_str}* {name} [[buffer({buf_idx})]]")

    if "thread_position_in_grid" in op_types:
        params.append("    uint tid [[thread_position_in_grid]]")
    if "thread_position_in_threadgroup" in op_types:
        params.append("    uint tidx [[thread_position_in_threadgroup]]")
    if "threadgroup_position_in_grid" in op_types:
        params.append("    uint bidx [[threadgroup_position_in_grid]]")
    if "threads_per_threadgroup" in op_types:
        params.append("    uint bdim [[threads_per_threadgroup]]")

    lines.append(f"kernel void {builder.kernel_name}(")
    lines.append(",\n".join(params))
    lines.append(") {")

    # When vec_width > 0, the buffer type handles vectorization,
    # so scalar load/add/store ops just work on vector elements.
    for op in builder.ops:
        _emit_op(op, lines, vec_width)

    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _emit_op(op: IROp, lines: List[str], vec_width: int = 0) -> None:
    t = op.op_type

    if t in ("thread_position_in_grid", "thread_position_in_threadgroup",
             "threadgroup_position_in_grid", "threads_per_threadgroup"):
        return

    if t == "const":
        lines.append(f"    {op.result.dtype} {op.result.name} = {op.attrs['value']};")

    elif t == "load":
        buf = op.attrs["buffer"]
        idx = op.operands[0].name
        dtype = _metal_vec_type(op.result.dtype, vec_width) if vec_width > 0 else op.result.dtype
        lines.append(f"    {dtype} {op.result.name} = {buf}[{idx}];")

    elif t == "store":
        lines.append(f"    {op.attrs['buffer']}[{op.operands[0].name}] = {op.operands[1].name};")

    elif t == "tv_load":
        _emit_tv_load(op, lines)

    elif t == "tv_store":
        _emit_tv_store(op, lines)

    elif t in ("add", "sub", "mul", "div", "mod"):
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%"}[t]
        a, b = op.operands
        dtype = op.result.dtype
        if vec_width > 0 and dtype in ("float", "half", "bfloat"):
            dtype = _metal_vec_type(dtype, vec_width)
        lines.append(f"    {dtype} {op.result.name} = {a.name} {sym} {b.name};")

    elif t == "neg":
        lines.append(f"    {op.result.dtype} {op.result.name} = -{op.operands[0].name};")

    elif t == "tv_add":
        a, b, res = op.operands[0], op.operands[1], op.result
        for gi, (start, count) in enumerate(op.attrs["groups"]):
            vec_w = _best_vec_width(count, op.attrs["dtype"])
            vtype = _metal_vec_type(op.attrs["dtype"], vec_w)
            for vi in range(count // vec_w):
                lines.append(f"    {vtype} {res.name}_g{gi}_v{vi} = "
                             f"{a.name}_g{gi}_v{vi} + {b.name}_g{gi}_v{vi};")
            for ri in range(count % vec_w):
                lines.append(f"    {op.attrs['dtype']} {res.name}_g{gi}_s{ri} = "
                             f"{a.name}_g{gi}_s{ri} + {b.name}_g{gi}_s{ri};")


def _emit_tv_load(op: IROp, lines: List[str]) -> None:
    buf, base, groups, dtype, res = (
        op.attrs["buffer"], op.attrs["base_offset"],
        op.attrs["groups"], op.attrs["dtype"], op.result)
    base_e = _base_expr(base)

    for gi, (start, count) in enumerate(groups):
        vec_w = _best_vec_width(count, dtype)
        vtype = _metal_vec_type(dtype, vec_w)
        for vi in range(count // vec_w):
            off = start + vi * vec_w
            off_expr = f"{base_e} + {off}" if off != 0 else base_e
            lines.append(f"    {vtype} {res.name}_g{gi}_v{vi} = "
                         f"*reinterpret_cast<device const {vtype}*>(&{buf}[{off_expr}]);")
        for ri in range(count % vec_w):
            off = start + (count // vec_w) * vec_w + ri
            off_expr = f"{base_e} + {off}" if off != 0 else base_e
            lines.append(f"    {dtype} {res.name}_g{gi}_s{ri} = {buf}[{off_expr}];")


def _emit_tv_store(op: IROp, lines: List[str]) -> None:
    buf, base, groups, dtype, val = (
        op.attrs["buffer"], op.attrs["base_offset"],
        op.attrs["groups"], op.attrs["dtype"], op.operands[0])
    base_e = _base_expr(base)

    for gi, (start, count) in enumerate(groups):
        vec_w = _best_vec_width(count, dtype)
        vtype = _metal_vec_type(dtype, vec_w)
        for vi in range(count // vec_w):
            off = start + vi * vec_w
            off_expr = f"{base_e} + {off}" if off != 0 else base_e
            lines.append(f"    *reinterpret_cast<device {vtype}*>(&{buf}[{off_expr}]) = "
                         f"{val.name}_g{gi}_v{vi};")
        for ri in range(count % vec_w):
            off = start + (count // vec_w) * vec_w + ri
            off_expr = f"{base_e} + {off}" if off != 0 else base_e
            lines.append(f"    {buf}[{off_expr}] = {val.name}_g{gi}_s{ri};")
