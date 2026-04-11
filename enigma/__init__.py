"""Enigma — A Python DSL for Apple Metal GPU kernels."""

from enigma._version import __version__

from .typing import Float32, Float16, BFloat16, Int32, UInt32, f32, f16, bf16, i32, u32
from .core import (
    Layout, make_layout, make_ordered_layout, make_identity_layout,
    size, coalesce, complement, composition,
    zipped_divide, logical_divide, blocked_product,
    recast_layout, make_layout_tv,
)
from .tuple import select, repeat_like, product
from .tensor import Tensor, tensor_composition, tensor_zipped_divide, make_identity_tensor
from .compiler.kernel import kernel, jit
from .compiler.compiler import compile, CompiledKernel
from .runtime_dispatch.runtime import MetalRuntime, PreparedKernel


class arch:
    """Metal thread/block index accessors for use inside @enigma.kernel."""

    @staticmethod
    def thread_idx():
        from ._tracing import get_builder
        b = get_builder()
        assert b is not None, "arch.thread_idx() only inside @enigma.kernel"
        return (b.get_thread_idx(), 0, 0)

    @staticmethod
    def block_idx():
        from ._tracing import get_builder
        b = get_builder()
        assert b is not None, "arch.block_idx() only inside @enigma.kernel"
        return (b.get_block_idx(), 0, 0)

    @staticmethod
    def block_dim():
        from ._tracing import get_builder
        b = get_builder()
        assert b is not None, "arch.block_dim() only inside @enigma.kernel"
        return (b.get_block_dim(), 0, 0)


def __getattr__(name: str):
    if name == "thread_position_in_grid":
        from ._tracing import get_builder
        builder = get_builder()
        if builder is None:
            raise RuntimeError(
                "enigma.thread_position_in_grid can only be used inside @enigma.kernel"
            )
        return builder.get_thread_position_in_grid()
    raise AttributeError(f"module 'enigma' has no attribute {name!r}")
