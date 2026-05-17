"""Metal GPU dispatch via ctypes -> Swift dylib."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..compiler.compiler import CompiledKernel

_SWIFT_DIR = Path(__file__).parent / "swift"
_SWIFT_SRC = _SWIFT_DIR / "libenigma_runtime.swift"
_DYLIB_PATH = _SWIFT_DIR / "libenigma_runtime.dylib"

_DISPATCH_ERRORS = {
    -1: "failed to create command buffer or compute encoder",
    -2: "GPU execution error (check Metal validation layer)",
}


def _gpu_error(msg: str, **ctx):
    """Format a verbose GPU error with context."""
    parts = [f"enigma GPU error: {msg}"]
    for k, v in ctx.items():
        parts.append(f"  {k}: {v}")
    return RuntimeError("\n".join(parts))


def _check_dispatch(rc: int, kernel_name: str = "", grid=None, threads=None):
    if rc == 0:
        return
    detail = _DISPATCH_ERRORS.get(rc, f"unknown error code {rc}")
    raise _gpu_error(
        f"dispatch failed: {detail}",
        kernel=kernel_name,
        grid=grid,
        threads=threads,
        return_code=rc,
    )


def _ensure_runtime_built() -> Path:
    """Build the Swift runtime dylib if needed."""
    if _DYLIB_PATH.exists() and _DYLIB_PATH.stat().st_mtime > _SWIFT_SRC.stat().st_mtime:
        return _DYLIB_PATH
    print("enigma: building Swift runtime...", file=sys.stderr)
    result = subprocess.run(
        ["swiftc", "-O", "-emit-library", "-o", str(_DYLIB_PATH), str(_SWIFT_SRC)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise _gpu_error("failed to compile Swift runtime", stderr=result.stderr.strip())
    print(f"enigma: runtime built -> {_DYLIB_PATH}", file=sys.stderr)
    return _DYLIB_PATH


# Type tags for enigma_create_pipeline_with_constants.
# Must match switch statement in libenigma_runtime.swift.
_FC_TYPE_TAGS = {
    "float": 0, "f32": 0,
    "int": 1, "i32": 1,
    "uint": 2, "u32": 2,
    "bool": 3, "i1": 3,
    "half": 4, "f16": 4,
}


def _pack_constants(constants):
    """Turn {index: (type_name, value)} into the parallel arrays the Swift side wants.

    Returns (indices_arr, tags_arr, values_arr, count).
    """
    if not constants:
        return None, None, None, 0
    items = sorted(constants.items(), key=lambda x: x[0])
    n = len(items)
    IndArr = (ctypes.c_int32 * n)
    TagArr = (ctypes.c_int32 * n)
    ValArr = (ctypes.c_double * n)
    inds, tags, vals = IndArr(), TagArr(), ValArr()
    for i, (idx, payload) in enumerate(items):
        if not (isinstance(payload, tuple) and len(payload) == 2):
            raise RuntimeError(
                f"constants[{idx}] must be (type_name, value) tuple, got {payload!r}"
            )
        type_name, value = payload
        tag = _FC_TYPE_TAGS.get(type_name)
        if tag is None:
            raise RuntimeError(
                f"constants[{idx}]: unknown type {type_name!r}. "
                f"Expected one of {sorted(set(_FC_TYPE_TAGS))}"
            )
        inds[i] = int(idx)
        tags[i] = tag
        vals[i] = float(value) if not isinstance(value, bool) else float(bool(value))
    return inds, tags, vals, n


_SCALAR_DTYPE_TO_NUMPY = {
    "float": "float32", "f32": "float32",
    "half": "float16", "f16": "float16",
    "bfloat": "float32", "bf16": "float32",  # no native bf16 numpy
    "char": "int8", "int8": "int8", "i8": "int8",
    "uchar": "uint8", "uint8": "uint8", "u8": "uint8",
    "short": "int16", "int16": "int16", "i16": "int16",
    "ushort": "uint16", "uint16": "uint16", "u16": "uint16",
    "int": "int32", "int32": "int32", "i32": "int32",
    "uint": "uint32", "uint32": "uint32", "u32": "uint32",
    "long": "int64", "int64": "int64", "i64": "int64",
    "ulong": "uint64", "uint64": "uint64", "u64": "uint64",
    "bool": "uint8", "i1": "uint8",
}


def _merge_scalar_buffers(inputs, scalars, scalar_params, num_outputs=0):
    """Interleave scalar 1-element buffers among the input buffer slots.

    scalar_params: list of (name, buffer_index, metal_dtype) from the trace.
    num_outputs:   number of output buffers that will be appended AFTER this
                   merged input list. Used only to validate buffer_index bounds.

    Returns a flat list of np.ndarray for the input+scalar slots (outputs are
    appended by the caller).
    """
    if not scalar_params:
        if scalars:
            raise RuntimeError(
                f"execute: got {len(scalars)} scalars but kernel has no Scalar params"
            )
        return list(inputs)

    if len(scalars) != len(scalar_params):
        raise RuntimeError(
            f"execute: expected {len(scalar_params)} scalars "
            f"(one per Scalar param), got {len(scalars)}"
        )

    total_slots = len(inputs) + len(scalar_params) + num_outputs
    # Only the first (total_slots - num_outputs) slots belong to the merged list.
    merged_slot_count = total_slots - num_outputs
    out: list = [None] * merged_slot_count
    scalar_indices = set()
    # Place scalar values by their declared buffer_index (sorted for determinism).
    sp_sorted = sorted(enumerate(scalar_params), key=lambda x: x[1][1])
    for orig_idx, (_name, bi, metal_dtype) in sp_sorted:
        np_dtype = _SCALAR_DTYPE_TO_NUMPY.get(metal_dtype, "float32")
        arr = np.asarray([scalars[orig_idx]], dtype=np_dtype)
        if bi >= total_slots:
            raise RuntimeError(
                f"scalar buffer_index {bi} exceeds total slots {total_slots}"
            )
        if bi >= merged_slot_count:
            raise RuntimeError(
                f"scalar buffer_index {bi} lands in output region "
                f"(merged_slots={merged_slot_count}, total={total_slots})"
            )
        out[bi] = arr
        scalar_indices.add(bi)
    # Fill remaining slots with buffer inputs in order.
    input_iter = iter(inputs)
    for i in range(merged_slot_count):
        if i in scalar_indices:
            continue
        out[i] = next(input_iter)
    return out


@dataclass
class DeviceCapabilities:
    """Metal device capabilities relevant for Enigma kernel tuning."""

    gpu_family: str            # e.g. "apple7" (M1), "apple8" (M2), "apple9" (M3), "apple10" (M4)
    gpu_family_raw: int        # MTLGPUFamily integer
    supports_async_copy: bool  # true on M3+ (apple9 and later)
    supports_simdgroup_matrix: bool
    simdgroup_size: int
    max_threadgroup_memory: int
    max_threads_per_threadgroup: int
    device_name: str

    @property
    def is_m3_or_newer(self) -> bool:
        """True on M3 / A17 Pro / M4 or later."""
        return self.gpu_family_raw >= 1009  # MTLGPUFamilyApple9

    def require_m3(self, feature: str) -> None:
        """Raise if this device is not M3 or newer."""
        if not self.is_m3_or_newer:
            raise RuntimeError(
                f"enigma: {feature} requires Apple GPU family 9 (M3/A17) or newer. "
                f"Current device: {self.device_name} ({self.gpu_family})"
            )


# MTLGPUFamily integer constants (from Metal headers).
_GPU_FAMILY_NAMES = {
    1001: "apple1", 1002: "apple2", 1003: "apple3", 1004: "apple4",
    1005: "apple5", 1006: "apple6", 1007: "apple7",   # M1
    1008: "apple8",                                    # M2 / A15-A16
    1009: "apple9",                                    # M3 / A17
    1010: "apple10",                                   # M4
    2001: "mac1", 2002: "mac2",
    3001: "common1", 3002: "common2", 3003: "common3",
    4001: "metal3",
}


def _query_device_capabilities(lib, device_ptr) -> DeviceCapabilities:
    """Query capabilities of the Metal device."""
    # Find highest supported apple family.
    family_raw = 0
    for code in (1010, 1009, 1008, 1007, 1006, 1005, 1004, 1003, 1002, 1001):
        if lib.enigma_device_supports_family(device_ptr, code) != 0:
            family_raw = code
            break
    name_buf = ctypes.create_string_buffer(128)
    lib.enigma_device_name(device_ptr, name_buf, 128)
    name = name_buf.value.decode(errors="replace")

    return DeviceCapabilities(
        gpu_family=_GPU_FAMILY_NAMES.get(family_raw, f"unknown({family_raw})"),
        gpu_family_raw=family_raw,
        supports_async_copy=(family_raw >= 1009),
        supports_simdgroup_matrix=(family_raw >= 1007),
        simdgroup_size=32,  # fixed on Apple Silicon
        max_threadgroup_memory=lib.enigma_device_max_threadgroup_memory(device_ptr),
        max_threads_per_threadgroup=lib.enigma_device_max_threads_per_threadgroup(device_ptr),
        device_name=name,
    )


class MetalRuntime:
    """Metal device + command queue, dispatches kernels via ctypes."""

    def __init__(self, dylib_path: Optional[str] = None):
        path = dylib_path or str(_ensure_runtime_built())
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as e:
            raise _gpu_error("failed to load Swift runtime dylib", path=path, error=str(e))
        self._setup_signatures()
        self._device = self._lib.enigma_create_device()
        if not self._device:
            raise _gpu_error("no Metal device found (Apple Silicon required)")
        self._queue = self._lib.enigma_create_queue(self._device)
        if not self._queue:
            raise _gpu_error("failed to create Metal command queue")

    def _setup_signatures(self):
        L = self._lib
        vp, sz, i32, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_char_p

        L.enigma_create_device.restype = vp
        L.enigma_create_device.argtypes = []
        L.enigma_create_queue.restype = vp
        L.enigma_create_queue.argtypes = [vp]
        L.enigma_load_library.restype = vp
        L.enigma_load_library.argtypes = [vp, cp]
        L.enigma_create_pipeline.restype = vp
        L.enigma_create_pipeline.argtypes = [vp, vp, cp]
        L.enigma_create_buffer.restype = vp
        L.enigma_create_buffer.argtypes = [vp, vp, sz]
        L.enigma_create_buffer_empty.restype = vp
        L.enigma_create_buffer_empty.argtypes = [vp, sz]
        L.enigma_buffer_contents.restype = vp
        L.enigma_buffer_contents.argtypes = [vp]
        L.enigma_buffer_length.restype = sz
        L.enigma_buffer_length.argtypes = [vp]
        L.enigma_dispatch.restype = i32
        L.enigma_dispatch.argtypes = [vp, vp, ctypes.POINTER(vp), sz, sz, sz, sz, sz, sz, sz]
        L.enigma_dispatch_timed.restype = i32
        L.enigma_dispatch_timed.argtypes = [
            vp,
            vp,
            ctypes.POINTER(vp),
            sz,
            sz,
            sz,
            sz,
            sz,
            sz,
            sz,
            ctypes.POINTER(ctypes.c_double),
        ]
        L.enigma_release.restype = None
        L.enigma_release.argtypes = [vp]

        # Capability queries (added for R10 / R6 M3+ gating).
        L.enigma_device_supports_family.restype = i32
        L.enigma_device_supports_family.argtypes = [vp, i32]
        L.enigma_device_name.restype = None
        L.enigma_device_name.argtypes = [vp, ctypes.c_char_p, sz]
        L.enigma_device_max_threadgroup_memory.restype = sz
        L.enigma_device_max_threadgroup_memory.argtypes = [vp]
        L.enigma_device_max_threads_per_threadgroup.restype = sz
        L.enigma_device_max_threads_per_threadgroup.argtypes = [vp]

        # Pipeline creation with function constants (Bug 3 runtime).
        L.enigma_create_pipeline_with_constants.restype = vp
        L.enigma_create_pipeline_with_constants.argtypes = [
            vp, vp, cp,
            ctypes.POINTER(i32),     # indices
            ctypes.POINTER(i32),     # type tags (0=float, 1=int, 2=uint, 3=bool, 4=half)
            ctypes.POINTER(ctypes.c_double),  # packed values (stored as double; cast by tag)
            sz,                      # count
        ]

    def execute(
        self,
        compiled: CompiledKernel,
        inputs: List[np.ndarray],
        output_size=None,
        grid: Tuple[int, int, int] = (1, 1, 1),
        threads: Tuple[int, int, int] = (1, 1, 1),
        *,
        scalars: Optional[List] = None,
        output_sizes: Optional[List[int]] = None,
        constants: Optional[dict] = None,
    ):
        """One-shot: create resources, dispatch, read back, cleanup.

        Extra parameters (all optional):
            scalars:       values for ``enigma.Scalar`` params, in order.
                           Packed into 1-element buffers at the right slots.
            output_sizes:  list of byte sizes for multiple output buffers.
                           If given, returns ``list[bytes]``; otherwise
                           returns a single ``bytes`` of ``output_size``.
            constants:     dict of ``{index: (type_name, value)}`` to set
                           Metal function constants before pipeline creation.
                           type_name is one of float/half/int/uint/bool.
        """
        mtl_lib = self._lib.enigma_load_library(self._device, compiled.metallib_path.encode())
        if not mtl_lib:
            raise _gpu_error(
                "failed to load metallib", path=compiled.metallib_path, kernel=compiled.kernel_name
            )

        if constants:
            inds, tags, vals, n = _pack_constants(constants)
            pso = self._lib.enigma_create_pipeline_with_constants(
                self._device, mtl_lib, compiled.kernel_name.encode(), inds, tags, vals, n
            )
        else:
            pso = self._lib.enigma_create_pipeline(self._device, mtl_lib, compiled.kernel_name.encode())
        if not pso:
            self._lib.enigma_release(mtl_lib)
            raise _gpu_error(
                "failed to create compute pipeline",
                kernel=compiled.kernel_name,
                hint="function name may not match metallib contents",
            )

        if output_sizes is not None:
            out_sizes = [int(s) for s in output_sizes]
            multi_out = True
        else:
            if output_size is None:
                raise _gpu_error("execute: must pass output_size or output_sizes")
            out_sizes = [int(output_size)]
            multi_out = False
        scalar_params = getattr(compiled, "scalar_params", None) or []

        gpu_bufs = []
        try:
            merged_inputs = _merge_scalar_buffers(
                inputs, scalars or [], scalar_params, num_outputs=len(out_sizes)
            )
            for i, arr in enumerate(merged_inputs):
                arr = np.ascontiguousarray(arr)
                buf = self._lib.enigma_create_buffer(self._device, arr.ctypes.data, arr.nbytes)
                if not buf:
                    raise _gpu_error(
                        "failed to create input buffer",
                        buffer_index=i,
                        size_bytes=arr.nbytes,
                        dtype=str(arr.dtype),
                        shape=arr.shape,
                    )
                gpu_bufs.append(buf)

            out_bufs = []
            for sz in out_sizes:
                ob = self._lib.enigma_create_buffer_empty(self._device, sz)
                if not ob:
                    raise _gpu_error("failed to create output buffer", size_bytes=sz)
                out_bufs.append(ob)
                gpu_bufs.append(ob)

            BufArr = ctypes.c_void_p * len(gpu_bufs)
            buf_arr = BufArr(*gpu_bufs)

            rc = self._lib.enigma_dispatch(
                pso,
                self._queue,
                buf_arr,
                len(gpu_bufs),
                grid[0],
                grid[1],
                grid[2],
                threads[0],
                threads[1],
                threads[2],
            )
            _check_dispatch(rc, compiled.kernel_name, grid, threads)

            outs: list[bytes] = []
            for ob, sz in zip(out_bufs, out_sizes):
                out_ptr = self._lib.enigma_buffer_contents(ob)
                outs.append(bytes((ctypes.c_char * sz).from_address(out_ptr)))
            return outs if multi_out else outs[0]
        finally:
            for buf in gpu_bufs:
                self._lib.enigma_release(buf)
            self._lib.enigma_release(pso)
            self._lib.enigma_release(mtl_lib)

    def device_capabilities(self) -> "DeviceCapabilities":
        """Return capability flags for the current Metal device."""
        return _query_device_capabilities(self._lib, self._device)

    def prepare(
        self, compiled: CompiledKernel, inputs: List[np.ndarray], output_size: int
    ) -> PreparedKernel:
        """Pre-allocate GPU resources for fast repeated dispatch."""
        mtl_lib = self._lib.enigma_load_library(self._device, compiled.metallib_path.encode())
        if not mtl_lib:
            raise _gpu_error("failed to load metallib", path=compiled.metallib_path)

        pso = self._lib.enigma_create_pipeline(self._device, mtl_lib, compiled.kernel_name.encode())
        if not pso:
            self._lib.enigma_release(mtl_lib)
            raise _gpu_error("failed to create compute pipeline", kernel=compiled.kernel_name)

        gpu_bufs = []
        for i, arr in enumerate(inputs):
            arr = np.ascontiguousarray(arr)
            buf = self._lib.enigma_create_buffer(self._device, arr.ctypes.data, arr.nbytes)
            if not buf:
                for b in gpu_bufs:
                    self._lib.enigma_release(b)
                self._lib.enigma_release(pso)
                self._lib.enigma_release(mtl_lib)
                raise _gpu_error(
                    "failed to create input buffer", buffer_index=i, size_bytes=arr.nbytes
                )
            gpu_bufs.append(buf)

        out_buf = self._lib.enigma_create_buffer_empty(self._device, output_size)
        if not out_buf:
            for b in gpu_bufs:
                self._lib.enigma_release(b)
            self._lib.enigma_release(pso)
            self._lib.enigma_release(mtl_lib)
            raise _gpu_error("failed to create output buffer", size_bytes=output_size)
        gpu_bufs.append(out_buf)

        BufArr = ctypes.c_void_p * len(gpu_bufs)
        return PreparedKernel(
            self,
            pso,
            mtl_lib,
            gpu_bufs,
            BufArr(*gpu_bufs),
            out_buf,
            output_size,
            compiled.kernel_name,
        )


class PreparedKernel:
    """Pre-allocated resources for fast repeated dispatch."""

    def __init__(
        self, runtime, pso, mtl_lib, gpu_bufs, buf_arr, out_buf, output_size, kernel_name=""
    ):
        self._rt = runtime
        self._pso, self._mtl_lib = pso, mtl_lib
        self._gpu_bufs, self._buf_arr = gpu_bufs, buf_arr
        self._out_buf, self._output_size = out_buf, output_size
        self._kernel_name = kernel_name

    def dispatch(self, grid: Tuple[int, int, int], threads: Tuple[int, int, int]) -> None:
        rc = self._rt._lib.enigma_dispatch(
            self._pso,
            self._rt._queue,
            self._buf_arr,
            len(self._gpu_bufs),
            grid[0],
            grid[1],
            grid[2],
            threads[0],
            threads[1],
            threads[2],
        )
        _check_dispatch(rc, self._kernel_name, grid, threads)

    def dispatch_timed(self, grid: Tuple[int, int, int], threads: Tuple[int, int, int]) -> float:
        """Dispatch and return GPU execution time in microseconds (Metal timestamps)."""
        gpu_time = ctypes.c_double(0.0)
        rc = self._rt._lib.enigma_dispatch_timed(
            self._pso,
            self._rt._queue,
            self._buf_arr,
            len(self._gpu_bufs),
            grid[0],
            grid[1],
            grid[2],
            threads[0],
            threads[1],
            threads[2],
            ctypes.byref(gpu_time),
        )
        _check_dispatch(rc, self._kernel_name, grid, threads)
        return gpu_time.value

    def read_output(self) -> bytes:
        out_ptr = self._rt._lib.enigma_buffer_contents(self._out_buf)
        return bytes((ctypes.c_char * self._output_size).from_address(out_ptr))

    def release(self):
        for buf in self._gpu_bufs:
            self._rt._lib.enigma_release(buf)
        self._rt._lib.enigma_release(self._pso)
        self._rt._lib.enigma_release(self._mtl_lib)
