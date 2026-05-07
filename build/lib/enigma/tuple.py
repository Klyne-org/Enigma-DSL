# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Hierarchical tuple utilities for CuTe-style layout algebra.

Pure Python math — no GPU or tracing dependencies.
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Sequence, Tuple, Union

Int = int
Shape = Union[int, Tuple["Shape", ...]]
Stride = Union[int, Tuple["Stride", ...]]
Coord = Union[int, Tuple["Coord", ...]]


def is_int(x: Any) -> bool:
    return isinstance(x, int)


def is_tuple(x: Any) -> bool:
    return isinstance(x, tuple)


def rank(x: Shape) -> int:
    """Number of top-level modes."""
    return len(x) if is_tuple(x) else 1


def depth(x: Shape) -> int:
    """Maximum nesting depth."""
    if is_int(x):
        return 0
    return 1 + max((depth(e) for e in x), default=0)


def flatten(x: Shape) -> Tuple[int, ...]:
    """Flatten hierarchical tuple to 1-D."""
    if is_int(x):
        return (x,)
    result: Tuple[int, ...] = ()
    for e in x:
        result += flatten(e)
    return result


def product(x: Shape) -> int:
    """Product of all leaf elements."""
    return reduce(mul, flatten(x), 1)


def inner_product(a: Shape, b: Stride) -> int:
    """Dot product of flattened tuples."""
    fa, fb = flatten(a), flatten(b)
    assert len(fa) == len(fb)
    return sum(x * y for x, y in zip(fa, fb))


def crd2idx(crd: Coord, shape: Shape, stride: Stride) -> int:
    """Hierarchical coordinate -> linear offset."""
    if is_int(crd) and is_int(shape) and is_int(stride):
        return crd * stride
    if is_int(crd) and is_tuple(shape):
        result = 0
        for s, d in zip(shape, stride):
            sz = product(s)
            result += crd2idx(crd % sz, s, d)
            crd //= sz
        return result
    if is_tuple(crd):
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    return crd * stride


def idx2crd(idx: int, shape: Shape) -> Coord:
    """Linear index -> hierarchical coordinate (colexicographic)."""
    if is_int(shape):
        return idx % shape
    result = []
    for s in shape:
        sz = product(s)
        result.append(idx2crd(idx % sz, s))
        idx //= sz
    return tuple(result)


def compact_col_major(shape: Shape, current: int = 1) -> Stride:
    """Column-major (mode-0-fastest) strides."""
    if is_int(shape):
        return current if shape > 0 else 0
    result = []
    for s in shape:
        result.append(compact_col_major(s, current))
        current *= product(s)
    return tuple(result)


def compact_order(shape: Shape, order: Tuple[int, ...]) -> Stride:
    """Strides with custom dimension ordering. order[i] = priority (0 = innermost)."""
    n = len(shape) if is_tuple(shape) else 1
    if n == 1:
        return compact_col_major(shape)
    dims = sorted(range(n), key=lambda i: order[i])
    strides = [0] * n
    current = 1
    for dim in dims:
        s = shape[dim] if is_tuple(shape) else shape
        strides[dim] = current
        current *= product(s) if is_tuple(s) else s
    return tuple(strides)


def prefix_product(shape: Shape, init: int = 1) -> Tuple[int, ...]:
    """Running product of a flat shape."""
    flat = flatten(shape)
    result, current = [], init
    for s in flat:
        result.append(current)
        current *= s
    return tuple(result)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def shape_div(a: int, b: int) -> int:
    """CuTe integer shape division."""
    if b == 1:
        return a
    return max(1, a // b) if a >= b else 1


def elem_scale(a: Shape, b: Shape) -> Shape:
    """Element-wise multiplication with matching structure."""
    if is_int(a) and is_int(b):
        return a * b
    if is_int(a):
        return tuple(elem_scale(a, bi) for bi in b)
    if is_int(b):
        return tuple(elem_scale(ai, b) for ai in a)
    return tuple(elem_scale(ai, bi) for ai, bi in zip(a, b))


def select(x: Shape, mode: Union[int, Sequence[int]]) -> Shape:
    """Select mode(s) from a tuple."""
    if isinstance(mode, (list, tuple)):
        return tuple(x[m] for m in mode)
    return x[mode]


def repeat_like(val: Any, ref: Shape) -> Shape:
    """Broadcast val to match structure of ref."""
    if is_int(ref) or not is_tuple(ref):
        return val
    return tuple(repeat_like(val, r) for r in ref)


def is_congruent(a: Shape, b: Shape) -> bool:
    """Same hierarchical structure."""
    if is_int(a) and is_int(b):
        return a == b
    if is_tuple(a) and is_tuple(b) and len(a) == len(b):
        return all(is_congruent(ai, bi) for ai, bi in zip(a, b))
    return False


def is_compatible(a: Shape, b: Shape) -> bool:
    """Same total size, structure may differ."""
    return product(a) == product(b)
