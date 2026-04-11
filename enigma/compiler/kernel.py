"""@enigma.kernel and @enigma.jit decorators."""

from __future__ import annotations

import inspect
import threading
from typing import Callable, Optional, Tuple

from .._tracing import KernelBuilder, TracingTensor
from ..typing import Numeric

_jit_local = threading.local()


class _JitContext:
    """Captures kernel trace + launch config from a @jit run."""
    def __init__(self):
        self.builder: Optional[KernelBuilder] = None
        self.grid: Optional[Tuple[int, ...]] = None
        self.block: Optional[Tuple[int, ...]] = None

    def __enter__(self):
        _jit_local.ctx = self
        return self

    def __exit__(self, *exc):
        _jit_local.ctx = None


def _get_jit_context() -> Optional[_JitContext]:
    return getattr(_jit_local, "ctx", None)


class KernelDef:
    """A decorated kernel function. Call inside @jit to get a KernelHandle."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self._sig = inspect.signature(fn)

    def __call__(self, *args, **kwargs) -> KernelHandle:
        return KernelHandle(self, args, kwargs)

    def __repr__(self):
        return f"<enigma.kernel {self.name}>"


def kernel(fn: Callable) -> KernelDef:
    return KernelDef(fn)


class JitDef:
    """A host-side JIT function that does layout algebra + launches kernels."""
    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__

    def __repr__(self):
        return f"<enigma.jit {self.name}>"


def jit(fn: Callable) -> JitDef:
    return JitDef(fn)


class KernelHandle:
    """Returned by calling @kernel inside @jit. .launch() triggers tracing."""

    def __init__(self, kernel_def: KernelDef, args: tuple, kwargs: dict):
        self.kernel_def = kernel_def
        self.args = args
        self.kwargs = kwargs

    def launch(self, grid: Tuple[int, ...], block: Tuple[int, ...]):
        from ..tensor import Tensor

        builder = KernelBuilder(self.kernel_def.name)

        seen: set[str] = set()
        for arg in self.args:
            if isinstance(arg, Tensor) and arg.buffer_index >= 0 and arg.name not in seen:
                builder.args.append((arg.name, arg.buffer_index, arg.metal_dtype))
                seen.add(arg.name)

        with builder:
            self.kernel_def.fn(*self.args, **self.kwargs)

        ctx = _get_jit_context()
        if ctx is not None:
            ctx.builder = builder
            ctx.grid = grid
            ctx.block = block
        else:
            raise RuntimeError("kernel.launch() called outside enigma.compile() context")


def _metal_dtype_from_annotation(ann) -> str:
    if isinstance(ann, Numeric):
        return ann.metal_name
    if isinstance(ann, type) and issubclass(ann, Numeric):
        return ann.metal_name
    raise TypeError(f"Unsupported kernel parameter annotation: {ann!r}")


def trace_kernel(kdef: KernelDef) -> KernelBuilder:
    """Trace a @kernel with type-annotated params (naive path)."""
    builder = KernelBuilder(kdef.name)
    params = list(kdef._sig.parameters.values())

    proxies = []
    for idx, param in enumerate(params):
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            raise TypeError(f"Kernel parameter '{param.name}' needs a type annotation")
        metal_dtype = _metal_dtype_from_annotation(ann)
        proxy = TracingTensor(param.name, idx, metal_dtype)
        builder.args.append((param.name, idx, metal_dtype))
        proxies.append(proxy)

    with builder:
        kdef.fn(*proxies)
    return builder
