# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Tile-level ``gemm`` op for Enigma.

This is the Enigma equivalent of tilelang's ``T.gemm`` (PRs #1869 and
#2118). Two lowering paths are provided:

  * **simdgroup MMA** — for an exact 8x8x8 tile with float/half elements,
    lower to ``simdgroup_load -> simdgroup_multiply_accumulate ->
    simdgroup_store`` (PR #1869). Tile chaining (M/N/K > 8) is *not*
    supported yet because the dialect's ``simdgroup_matrix_load`` op
    takes no element offset, so multi-tile chains would all read the
    same 8x8 block.

  * **scalar fallback** — otherwise emit a triple ``for_range`` scalar
    accumulator into a ``RegisterTensor``. Correctness-first, mirroring
    PR #2118: any kernel written today still runs on Metal even when
    MMA isn't applicable.
"""

from typing import Optional

from .._tracing import (
    EnigmaError,
    RegisterTensor,
    _ensure_ir,
    _require_builder,
    barrier,
    for_range,
    make_filled_simdgroup_matrix,
    metal_cast,
    simdgroup_matrix_load,
    simdgroup_matrix_store,
    simdgroup_multiply_accumulate,
)
from ..tensor import Tensor

__all__ = ["gemm"]


_SIMD_TILE = 8


def _is_simdgroup_compatible(M: int, N: int, K: int, elem: str) -> bool:
    if elem not in ("float", "half", "f32", "f16"):
        return False
    return M == _SIMD_TILE and N == _SIMD_TILE and K == _SIMD_TILE


def gemm(
    A_s,
    B_s,
    C,
    *,
    M: int,
    N: int,
    K: int,
    transpose_A: bool = False,
    transpose_B: bool = False,
    accum_dtype: str = "float",
    use_simdgroup: Optional[bool] = None,
) -> None:
    """Compute ``C += A @ B`` over an MxN tile (with reduction K).

    Parameters
    ----------
    A_s, B_s : Tensor
        Threadgroup-resident (or device) tile-sized inputs.
    C : RegisterTensor or Tensor
        Output tile. For the simdgroup path, a ``Tensor`` (MMA result is
        stored back into shared/threadgroup memory). For the scalar
        fallback, a ``RegisterTensor`` of shape ``(M, N)``.
    M, N, K : int
        Tile shapes in elements.
    transpose_A, transpose_B : bool
        Treat the corresponding input as transposed. Honoured by the
        scalar path only; the simdgroup path rejects either being set.
    accum_dtype : str
        Accumulator element type. Defaults to ``"float"``.
    use_simdgroup : bool, optional
        Force a path. ``None`` (default) auto-picks: simdgroup MMA for
        the exact 8x8x8 case, scalar fallback otherwise. Setting
        ``True`` raises if the simdgroup path can't be used.
    """
    _require_builder("gemm")

    if M <= 0 or N <= 0 or K <= 0:
        raise EnigmaError(f"gemm: M/N/K must be positive, got M={M}, N={N}, K={K}")

    elem = accum_dtype
    simd_ok = _is_simdgroup_compatible(M, N, K, elem) and not (transpose_A or transpose_B)

    if use_simdgroup is True and not simd_ok:
        raise EnigmaError(
            f"gemm: use_simdgroup=True but shape ({M}x{N}x{K}, elem={elem}) "
            f"is not 8x8x8 (or transpose flags set). The simdgroup path "
            f"currently supports a single 8x8x8 tile only."
        )

    pick_simd = simd_ok if use_simdgroup is None else bool(use_simdgroup)

    if pick_simd:
        if not isinstance(C, Tensor):
            raise EnigmaError(
                "gemm: simdgroup path requires C to be a Tensor (threadgroup or "
                "device buffer); got RegisterTensor. Pass use_simdgroup=False to "
                "use the scalar accumulator path."
            )
        _gemm_simdgroup(A_s, B_s, C, elem=elem)
    else:
        if isinstance(C, Tensor):
            raise EnigmaError(
                "gemm: scalar fallback path requires C to be a RegisterTensor of "
                "shape (M, N). Pass `enigma.register_tensor((M, N), ...)` for C."
            )
        _gemm_scalar(
            A_s, B_s, C, M=M, N=N, K=K, elem=elem,
            transpose_A=transpose_A, transpose_B=transpose_B,
        )


def _gemm_simdgroup(A_s: Tensor, B_s: Tensor, C: Tensor, *, elem: str) -> None:
    """Single 8x8 simdgroup MMA. Assumes row-major contiguous tiles."""
    barrier("mem_threadgroup")
    acc = make_filled_simdgroup_matrix(
        metal_cast(0, elem), elem=elem,
        rows=_SIMD_TILE, cols=_SIMD_TILE,
    )
    a_tile = simdgroup_matrix_load(A_s, _SIMD_TILE, elem=elem)
    b_tile = simdgroup_matrix_load(B_s, _SIMD_TILE, elem=elem)
    acc = simdgroup_multiply_accumulate(a_tile, b_tile, acc)
    simdgroup_matrix_store(acc, C, _SIMD_TILE)


def _gemm_scalar(
    A_s, B_s, C: RegisterTensor, *,
    M: int, N: int, K: int, elem: str,
    transpose_A: bool, transpose_B: bool,
) -> None:
    """Triple-loop scalar accumulator into a RegisterTensor.

    ``C[m, n] += A[m, k] * B[k, n]``, with optional A/B transpose.
    Correctness-first path from tilelang PR #2118.
    """
    if C.shape != (M, N):
        raise EnigmaError(
            f"gemm: register tensor shape {C.shape} does not match (M={M}, N={N})"
        )

    K_ir = _ensure_ir(int(K))

    for m in range(M):
        m_v = metal_cast(m, "uint")
        for n in range(N):
            n_v = metal_cast(n, "uint")
            with for_range(0, K_ir, init=[C[m, n]]) as (k, carry):
                k_v = metal_cast(k, "uint")
                a_idx = k_v * int(M) + m_v if transpose_A else m_v * int(K) + k_v
                b_idx = n_v * int(K) + k_v if transpose_B else k_v * int(N) + n_v
                a_val, b_val = A_s[a_idx], B_s[b_idx]
                if a_val.dtype != elem:
                    a_val = metal_cast(a_val, elem)
                if b_val.dtype != elem:
                    b_val = metal_cast(b_val, elem)
                carry[0] = carry[0] + a_val * b_val
            C[m, n] = carry[0]
