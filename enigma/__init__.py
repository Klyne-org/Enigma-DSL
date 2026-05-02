"""Enigma — A Python DSL for Apple Metal GPU kernels."""

try:
    from ._version import __version__ as __version__
except ImportError:
    __version__ = "unknown"
from .compiler.compiler import CompiledKernel as CompiledKernel, compile as compile
from .compiler.kernel import jit as jit, kernel as kernel
from .core import (
    Layout as Layout,
    Swizzle as Swizzle,
    SwizzledLayout as SwizzledLayout,
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
    swizzle as swizzle,
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
    Bool as Bool,
    Float16 as Float16,
    Float32 as Float32,
    Int8 as Int8,
    Int16 as Int16,
    Int32 as Int32,
    Int64 as Int64,
    Scalar as Scalar,
    UInt8 as UInt8,
    UInt16 as UInt16,
    UInt32 as UInt32,
    UInt64 as UInt64,
    b1 as b1,
    bf16 as bf16,
    f16 as f16,
    f32 as f32,
    i8 as i8,
    i16 as i16,
    i32 as i32,
    i64 as i64,
    u8 as u8,
    u16 as u16,
    u32 as u32,
    u64 as u64,
)
from ._tracing import (
    sqrt as sqrt,
    abs as abs,
    ceil as ceil,
    floor as floor,
    round as round,
    trunc as trunc,
    sign as sign,
    saturate as saturate,
    fract as fract,
    rsqrt as rsqrt,
    exp as exp,
    exp2 as exp2,
    log as log,
    log2 as log2,
    log10 as log10,
    sin as sin,
    cos as cos,
    tan as tan,
    asin as asin,
    acos as acos,
    atan as atan,
    sinh as sinh,
    cosh as cosh,
    tanh as tanh,
    fmin as fmin,
    fmax as fmax,
    pow as pow,
    fmod as fmod,
    atan2 as atan2,
    step as step,
    copysign as copysign,
    clamp as clamp,
    fma as fma,
    mix as mix,
    smoothstep as smoothstep,
    isnan as isnan,
    isinf as isinf,
    isfinite as isfinite,
    signbit as signbit,
    isnormal as isnormal,
    select as where,  # noqa: F401  — re-exported as enigma.where (avoids enigma.select collision)
    imin as imin,
    imax as imax,
    iclamp as iclamp,
    popcount as popcount,
    clz as clz,
    ctz as ctz,
    reverse_bits as reverse_bits,
    abs_diff as abs_diff,
    abs_diff_unary as abs_diff_unary,
    add_sat as add_sat,
    sub_sat as sub_sat,
    mul_hi as mul_hi,
    rotate as rotate,
    mad_sat as mad_sat,
    extract_bits as extract_bits,
    insert_bits as insert_bits,
    simd_sum as simd_sum,
    simd_product as simd_product,
    simd_min as simd_min,
    simd_max as simd_max,
    simd_and as simd_and,
    simd_or as simd_or,
    simd_xor as simd_xor,
    simd_prefix_exclusive_sum as simd_prefix_exclusive_sum,
    simd_prefix_inclusive_sum as simd_prefix_inclusive_sum,
    simd_prefix_exclusive_product as simd_prefix_exclusive_product,
    simd_prefix_inclusive_product as simd_prefix_inclusive_product,
    simd_shuffle as simd_shuffle,
    simd_shuffle_up as simd_shuffle_up,
    simd_shuffle_down as simd_shuffle_down,
    simd_shuffle_xor as simd_shuffle_xor,
    simd_broadcast as simd_broadcast,
    metal_cast as metal_cast,
    as_type as as_type,
    barrier as barrier,
    simd_barrier as simd_barrier,
    threadgroup_alloc as threadgroup_alloc,
    atomic_load as atomic_load,
    atomic_store as atomic_store,
    atomic_exchange as atomic_exchange,
    atomic_fetch_add as atomic_fetch_add,
    atomic_fetch_sub as atomic_fetch_sub,
    atomic_fetch_min as atomic_fetch_min,
    atomic_fetch_max as atomic_fetch_max,
    atomic_fetch_and as atomic_fetch_and,
    atomic_fetch_or as atomic_fetch_or,
    atomic_fetch_xor as atomic_fetch_xor,
    atomic_compare_exchange_weak as atomic_compare_exchange_weak,
    quad_sum as quad_sum,
    quad_product as quad_product,
    quad_min as quad_min,
    quad_max as quad_max,
    quad_and as quad_and,
    quad_or as quad_or,
    quad_xor as quad_xor,
    quad_prefix_exclusive_sum as quad_prefix_exclusive_sum,
    quad_prefix_inclusive_sum as quad_prefix_inclusive_sum,
    quad_shuffle as quad_shuffle,
    quad_shuffle_up as quad_shuffle_up,
    quad_shuffle_down as quad_shuffle_down,
    quad_shuffle_xor as quad_shuffle_xor,
    quad_broadcast as quad_broadcast,
    make_vec as make_vec,
    make_float2 as make_float2,
    make_float3 as make_float3,
    make_float4 as make_float4,
    vec_extract as vec_extract,
    pack_float_to_snorm4x8 as pack_float_to_snorm4x8,
    pack_float_to_unorm4x8 as pack_float_to_unorm4x8,
    pack_float_to_snorm2x16 as pack_float_to_snorm2x16,
    pack_float_to_unorm2x16 as pack_float_to_unorm2x16,
    pack_float_to_srgb_unorm4x8 as pack_float_to_srgb_unorm4x8,
    pack_float_to_unorm10a2 as pack_float_to_unorm10a2,
    unpack_snorm4x8_to_float as unpack_snorm4x8_to_float,
    unpack_unorm4x8_to_float as unpack_unorm4x8_to_float,
    unpack_snorm2x16_to_float as unpack_snorm2x16_to_float,
    unpack_unorm2x16_to_float as unpack_unorm2x16_to_float,
    unpack_srgb_unorm4x8_to_float as unpack_srgb_unorm4x8_to_float,
    unpack_unorm10a2_to_float as unpack_unorm10a2_to_float,
    dot as dot,
    length as length,
    distance as distance,
    cross as cross,
    normalize as normalize,
    reflect as reflect,
    refract as refract,
    faceforward as faceforward,
    # --- Comparisons ---
    cmp_eq as cmp_eq,
    cmp_ne as cmp_ne,
    cmp_lt as cmp_lt,
    cmp_le as cmp_le,
    cmp_gt as cmp_gt,
    cmp_ge as cmp_ge,
    cmp_ult as cmp_ult,
    cmp_ule as cmp_ule,
    cmp_ugt as cmp_ugt,
    cmp_uge as cmp_uge,
    # --- Grid / thread queries (x/y/z variants) ---
    thread_position_in_grid_xyz as thread_position_in_grid_xyz,
    thread_position_in_threadgroup as thread_position_in_threadgroup,
    threadgroup_position_in_grid as threadgroup_position_in_grid,
    threads_per_threadgroup as threads_per_threadgroup,
    threads_per_grid as threads_per_grid,
    threadgroups_per_grid as threadgroups_per_grid,
    grid_size as grid_size,
    thread_index_in_threadgroup as thread_index_in_threadgroup,
    thread_index_in_simdgroup as thread_index_in_simdgroup,
    simdgroup_index_in_threadgroup as simdgroup_index_in_threadgroup,
    threads_per_simdgroup as threads_per_simdgroup,
    simdgroups_per_threadgroup as simdgroups_per_threadgroup,
    # --- Function constants ---
    function_constant as function_constant,
    # --- Control flow ---
    for_range as for_range,
    if_ as if_,
    while_ as while_,
    Carry as Carry,
    # --- Tiled copy / register tensors / predication / pipeline / async ---
    copy as copy,
    register_tensor as register_tensor,
    RegisterTensor as RegisterTensor,
    load_if as load_if,
    store_if as store_if,
    pipeline as pipeline,
    Pipeline as Pipeline,
    async_copy_to_threadgroup as async_copy_to_threadgroup,
    async_copy_commit as async_copy_commit,
    async_copy_wait as async_copy_wait,
    # --- Simdgroup matrix ops ---
    simdgroup_matrix_load as simdgroup_matrix_load,
    simdgroup_matrix_store as simdgroup_matrix_store,
    simdgroup_multiply_accumulate as simdgroup_multiply_accumulate,
    make_filled_simdgroup_matrix as make_filled_simdgroup_matrix,
    # --- Regular matrix ops ---
    matmul as matmul,
    transpose as transpose,
    determinant as determinant,
    # --- Errors ---
    EnigmaError as EnigmaError,
    # --- AST preprocessor support ---
    _EnigmaRange as _EnigmaRange,
    _EnigmaRangeConstexpr as _EnigmaRangeConstexpr,
)

range = _EnigmaRange()
range_constexpr = _EnigmaRangeConstexpr()

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
