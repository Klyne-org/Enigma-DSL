"""Enigma — A Python DSL for Apple Metal GPU kernels."""

try:
    from ._version import __version__ as __version__
except ImportError:
    __version__ = "unknown"
from .compiler.compiler import CompiledKernel as CompiledKernel, compile as compile
from .compiler.kernel import jit as jit, kernel as kernel
from .core import (
    Layout as Layout,
    blocked_product as blocked_product,
    coalesce as coalesce,
    complement as complement,
    composition as composition,
    logical_divide as logical_divide,
    make_identity_layout as make_identity_layout,
    make_layout as make_layout,
    make_layout_tv as make_layout_tv,
    make_ordered_layout as make_ordered_layout,
    recast_layout as recast_layout,
    size as size,
    zipped_divide as zipped_divide,
)
from .runtime_dispatch.runtime import MetalRuntime as MetalRuntime, PreparedKernel as PreparedKernel
from .tensor import (
    Tensor as Tensor,
    make_identity_tensor as make_identity_tensor,
    tensor_composition as tensor_composition,
    tensor_zipped_divide as tensor_zipped_divide,
)
from .tuple import product as product, repeat_like as repeat_like, select as select
from .typing import (
    BFloat16 as BFloat16,
    Float16 as Float16,
    Float32 as Float32,
    Int32 as Int32,
    UInt32 as UInt32,
    bf16 as bf16,
    f16 as f16,
    f32 as f32,
    i32 as i32,
    u32 as u32,
)


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
