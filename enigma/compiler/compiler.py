"""Compilation pipeline: trace -> MLIR (enigma dialect) -> MSL -> metallib."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .kernel import JitDef, KernelDef, _JitContext, trace_kernel
from .mlir_emitter import emit_mlir, emit_msl


@dataclass
class CompiledKernel:
    kernel_name: str
    metallib_path: str
    metallib_bytes: bytes
    metal_source: str
    mlir_source: Optional[str] = None
    grid: Optional[Tuple[int, ...]] = None
    block: Optional[Tuple[int, ...]] = None

    def export_metal(self, path: str = None) -> str:
        if path is None:
            path = f"{self.kernel_name}.metal"
        with open(path, "w") as f:
            f.write(self.metal_source)
        return path


def compile(
    fn, *args, keep_metal_source=False, dump_ir=False, dump_mlir=False,
    work_dir=None, vec_width=0,
) -> CompiledKernel:
    """Compile @enigma.kernel (naive) or @enigma.jit (TV layout) to .metallib."""
    if isinstance(fn, JitDef):
        return _compile_jit(
            fn, args, dump_ir=dump_ir, dump_mlir=dump_mlir,
            keep_metal_source=keep_metal_source, work_dir=work_dir, vec_width=vec_width,
        )
    elif isinstance(fn, KernelDef):
        return _compile_naive(
            fn, dump_ir=dump_ir, dump_mlir=dump_mlir,
            keep_metal_source=keep_metal_source, work_dir=work_dir, vec_width=vec_width,
        )
    raise TypeError(f"Expected @enigma.kernel or @enigma.jit, got {type(fn).__name__}")


def _compile_naive(kernel_fn, *, dump_ir, dump_mlir, keep_metal_source, work_dir, vec_width):
    builder = trace_kernel(kernel_fn)
    return _emit_and_build(
        builder, dump_ir=dump_ir, dump_mlir=dump_mlir,
        keep_metal_source=keep_metal_source, work_dir=work_dir, vec_width=vec_width,
    )


def _compile_jit(jit_fn, tensor_args, *, dump_ir, dump_mlir, keep_metal_source, work_dir, vec_width):
    with _JitContext() as ctx:
        jit_fn.fn(*tensor_args)
    if ctx.builder is None:
        raise RuntimeError(
            f"@jit function '{jit_fn.name}' did not launch any kernel. "
            f"Call kernel_fn(...).launch(grid=..., block=...) inside it."
        )
    compiled = _emit_and_build(
        ctx.builder, dump_ir=dump_ir, dump_mlir=dump_mlir,
        keep_metal_source=keep_metal_source, work_dir=work_dir, vec_width=vec_width,
    )
    compiled.grid = ctx.grid
    compiled.block = ctx.block
    return compiled


def _lower_to_msl(builder, vec_width: int) -> tuple[str, Optional[str]]:
    """Return (msl_source, mlir_source).

    Single path: trace -> MLIR (enigma dialect) -> MSL via dialect translator.
    """
    mlir_text = emit_mlir(builder, vec_width=vec_width)
    msl = emit_msl(builder, vec_width=vec_width)
    return msl, mlir_text


def _emit_and_build(
    builder, *, dump_ir, keep_metal_source, work_dir, vec_width=0, dump_mlir=False
) -> CompiledKernel:
    if dump_ir:
        print(f"=== IR: {builder.kernel_name} ({len(builder.ops)} ops) ===")
        for op in builder.ops:
            res = op.result.name if op.result else "(void)"
            operands = ", ".join(getattr(o, "name", str(o)) for o in op.operands)
            print(f"  {res} = {op.op_type}({operands})")

    metal_source, mlir_source = _lower_to_msl(builder, vec_width)

    if dump_mlir:
        print(f"=== MLIR (enigma dialect): {builder.kernel_name} ===\n{mlir_source}")

    if dump_ir:
        print(f"=== Metal ===\n{metal_source}")

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="enigma_")
    else:
        os.makedirs(work_dir, exist_ok=True)

    metal_path = os.path.join(work_dir, f"{builder.kernel_name}.metal")
    air_path = os.path.join(work_dir, f"{builder.kernel_name}.air")
    metallib_path = os.path.join(work_dir, f"{builder.kernel_name}.metallib")

    with open(metal_path, "w") as f:
        f.write(metal_source)

    _run_xcrun(["xcrun", "-sdk", "macosx", "metal", "-c", metal_path, "-o", air_path])
    _run_xcrun(["xcrun", "-sdk", "macosx", "metallib", air_path, "-o", metallib_path])

    metallib_bytes = Path(metallib_path).read_bytes()
    if not keep_metal_source:
        for p in (metal_path, air_path):
            if os.path.exists(p):
                os.remove(p)

    return CompiledKernel(
        kernel_name=builder.kernel_name,
        metallib_path=metallib_path,
        metallib_bytes=metallib_bytes,
        metal_source=metal_source,
        mlir_source=mlir_source,
    )


def _run_xcrun(cmd: list[str]) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            f"Command not found: {cmd[0]}. Ensure Xcode Command Line Tools are installed."
        )
    if result.returncode != 0:
        raise RuntimeError(f"xcrun failed: {' '.join(cmd)}\n{result.stderr.strip()}")
