"""Metal GPU dispatch via ctypes -> Swift dylib."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from ..compiler.compiler import CompiledKernel

_SWIFT_DIR = Path(__file__).parent / "swift"
_SWIFT_SRC = _SWIFT_DIR / "libenigma_runtime.swift"
_DYLIB_PATH = _SWIFT_DIR / "libenigma_runtime.dylib"


def _ensure_runtime_built() -> Path:
    """Build the Swift runtime dylib if needed."""
    if _DYLIB_PATH.exists() and _DYLIB_PATH.stat().st_mtime > _SWIFT_SRC.stat().st_mtime:
        return _DYLIB_PATH
    print("enigma: building Swift runtime...")
    result = subprocess.run(
        ["swiftc", "-O", "-emit-library", "-o", str(_DYLIB_PATH), str(_SWIFT_SRC)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile Swift runtime:\n{result.stderr}")
    print(f"enigma: runtime built -> {_DYLIB_PATH}")
    return _DYLIB_PATH


class MetalRuntime:
    """Metal device + command queue, dispatches kernels via ctypes."""

    def __init__(self, dylib_path: Optional[str] = None):
        path = dylib_path or str(_ensure_runtime_built())
        self._lib = ctypes.CDLL(path)
        self._setup_signatures()
        self._device = self._lib.enigma_create_device()
        assert self._device, "no Metal device available"
        self._queue = self._lib.enigma_create_queue(self._device)
        assert self._queue, "failed to create command queue"

    def _setup_signatures(self):
        L = self._lib
        vp, sz, i32, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_char_p

        L.enigma_create_device.restype = vp; L.enigma_create_device.argtypes = []
        L.enigma_create_queue.restype = vp; L.enigma_create_queue.argtypes = [vp]
        L.enigma_load_library.restype = vp; L.enigma_load_library.argtypes = [vp, cp]
        L.enigma_create_pipeline.restype = vp; L.enigma_create_pipeline.argtypes = [vp, vp, cp]
        L.enigma_create_buffer.restype = vp; L.enigma_create_buffer.argtypes = [vp, vp, sz]
        L.enigma_create_buffer_empty.restype = vp; L.enigma_create_buffer_empty.argtypes = [vp, sz]
        L.enigma_buffer_contents.restype = vp; L.enigma_buffer_contents.argtypes = [vp]
        L.enigma_buffer_length.restype = sz; L.enigma_buffer_length.argtypes = [vp]
        L.enigma_dispatch.restype = i32
        L.enigma_dispatch.argtypes = [vp, vp, ctypes.POINTER(vp), sz, sz, sz, sz, sz, sz, sz]
        L.enigma_dispatch_timed.restype = i32
        L.enigma_dispatch_timed.argtypes = [
            vp, vp, ctypes.POINTER(vp), sz, sz, sz, sz, sz, sz, sz,
            ctypes.POINTER(ctypes.c_double)]
        L.enigma_release.restype = None; L.enigma_release.argtypes = [vp]

    def execute(self, compiled: CompiledKernel, inputs: List[np.ndarray],
                output_size: int, grid: Tuple[int, int, int],
                threads: Tuple[int, int, int]) -> bytes:
        """One-shot: create resources, dispatch, read back, cleanup."""
        mtl_lib = self._lib.enigma_load_library(self._device, compiled.metallib_path.encode())
        assert mtl_lib, f"Failed to load metallib: {compiled.metallib_path}"
        pso = self._lib.enigma_create_pipeline(self._device, mtl_lib, compiled.kernel_name.encode())
        assert pso, f"Failed to create pipeline for '{compiled.kernel_name}'"

        gpu_bufs = []
        for arr in inputs:
            arr = np.ascontiguousarray(arr)
            buf = self._lib.enigma_create_buffer(self._device, arr.ctypes.data, arr.nbytes)
            assert buf, "Failed to create input buffer"
            gpu_bufs.append(buf)

        out_buf = self._lib.enigma_create_buffer_empty(self._device, output_size)
        assert out_buf, "Failed to create output buffer"
        gpu_bufs.append(out_buf)

        BufArr = ctypes.c_void_p * len(gpu_bufs)
        buf_arr = BufArr(*gpu_bufs)

        rc = self._lib.enigma_dispatch(
            pso, self._queue, buf_arr, len(gpu_bufs),
            grid[0], grid[1], grid[2], threads[0], threads[1], threads[2])
        assert rc == 0, f"GPU dispatch failed: {rc}"

        out_ptr = self._lib.enigma_buffer_contents(out_buf)
        data = bytes((ctypes.c_char * output_size).from_address(out_ptr))

        for buf in gpu_bufs:
            self._lib.enigma_release(buf)
        self._lib.enigma_release(pso)
        self._lib.enigma_release(mtl_lib)
        return data

    def prepare(self, compiled: CompiledKernel, inputs: List[np.ndarray],
                output_size: int) -> PreparedKernel:
        """Pre-allocate GPU resources for fast repeated dispatch."""
        mtl_lib = self._lib.enigma_load_library(self._device, compiled.metallib_path.encode())
        assert mtl_lib
        pso = self._lib.enigma_create_pipeline(self._device, mtl_lib, compiled.kernel_name.encode())
        assert pso

        gpu_bufs = []
        for arr in inputs:
            arr = np.ascontiguousarray(arr)
            buf = self._lib.enigma_create_buffer(self._device, arr.ctypes.data, arr.nbytes)
            assert buf
            gpu_bufs.append(buf)
        out_buf = self._lib.enigma_create_buffer_empty(self._device, output_size)
        assert out_buf
        gpu_bufs.append(out_buf)

        BufArr = ctypes.c_void_p * len(gpu_bufs)
        return PreparedKernel(self, pso, mtl_lib, gpu_bufs, BufArr(*gpu_bufs), out_buf, output_size)


class PreparedKernel:
    """Pre-allocated resources for fast repeated dispatch (benchmarking)."""

    def __init__(self, runtime, pso, mtl_lib, gpu_bufs, buf_arr, out_buf, output_size):
        self._rt = runtime
        self._pso, self._mtl_lib = pso, mtl_lib
        self._gpu_bufs, self._buf_arr = gpu_bufs, buf_arr
        self._out_buf, self._output_size = out_buf, output_size

    def dispatch(self, grid: Tuple[int, int, int], threads: Tuple[int, int, int]) -> None:
        rc = self._rt._lib.enigma_dispatch(
            self._pso, self._rt._queue, self._buf_arr, len(self._gpu_bufs),
            grid[0], grid[1], grid[2], threads[0], threads[1], threads[2])
        assert rc == 0, f"GPU dispatch failed: {rc}"

    def dispatch_timed(self, grid: Tuple[int, int, int], threads: Tuple[int, int, int]) -> float:
        """Dispatch and return GPU execution time in microseconds (Metal timestamps)."""
        gpu_time = ctypes.c_double(0.0)
        rc = self._rt._lib.enigma_dispatch_timed(
            self._pso, self._rt._queue, self._buf_arr, len(self._gpu_bufs),
            grid[0], grid[1], grid[2], threads[0], threads[1], threads[2],
            ctypes.byref(gpu_time))
        assert rc == 0, f"GPU dispatch failed: {rc}"
        return gpu_time.value

    def read_output(self) -> bytes:
        out_ptr = self._rt._lib.enigma_buffer_contents(self._out_buf)
        return bytes((ctypes.c_char * self._output_size).from_address(out_ptr))

    def release(self):
        for buf in self._gpu_bufs:
            self._rt._lib.enigma_release(buf)
        self._rt._lib.enigma_release(self._pso)
        self._rt._lib.enigma_release(self._mtl_lib)
