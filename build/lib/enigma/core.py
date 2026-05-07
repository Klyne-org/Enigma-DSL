# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Layout algebra engine for CuTe-style tiling on Apple Metal.

A Layout is (Shape, Stride) mapping coordinates to memory offsets:
    L(coord) = crd2idx(coord, shape, stride)
"""

from __future__ import annotations

from typing import Tuple

from .tuple import (
    Coord,
    Shape,
    Stride,
    ceil_div,
    compact_col_major,
    compact_order,
    crd2idx,
    depth,
    flatten,
    gcd,
    is_int,
    is_tuple,
    product,
)
from .tuple import (
    rank as _tuple_rank,
)


class Swizzle:
    """CuTe-style Swizzle<B, M, S> for bank-conflict avoidance.

    Remaps shared memory offsets via XOR to eliminate bank conflicts when
    accessing columns of a row-major tile.  Apple Silicon threadgroup memory
    has 32 banks with 4-byte granularity — the same conflict patterns as
    CUDA shared memory.

    Parameters
    ----------
    bits : int
        Number of XOR bits (B).  Controls the period of the swizzle pattern.
        Typical values: 1-3.
    base : int
        Bit position of the lowest XOR target bit (M).  Usually the
        log2 of the number of banks divided by the element size.
    shift : int
        Bit position of the lowest source bit (S).  The source bits are
        shifted right by S before XORing into [M, M+B).

    The transformation on a linear byte offset *s* is::

        mask = ((1 << bits) - 1) << base
        s ^= ((s >> shift) & mask)      # when shift >= base  (right-shift variant)
        s ^= ((s << (base - shift)) & mask)  # when shift < base

    CuTe uses the convention ``Swizzle<B, M, S>`` where the XOR is always::

        s ^= (s >> S) & (((1 << B) - 1) << M)

    which is the ``shift >= base`` path.

    Example — 128-byte-wide tile of float (32 columns × 4 bytes = stride 128):

    >>> sw = Swizzle(bits=3, base=3, shift=6)   # 8 banks, XOR bits [3,5] from bits [6,8]
    >>> sw(0)    # row 0, col 0 → offset 0
    0
    >>> sw(128)  # row 1, col 0 → XOR remaps to avoid bank 0 collision
    192
    """

    __slots__ = ("bits", "base", "shift")

    def __init__(self, bits: int, base: int, shift: int):
        if bits < 0:
            raise ValueError(f"Swizzle bits must be non-negative, got {bits}")
        if base < 0:
            raise ValueError(f"Swizzle base must be non-negative, got {base}")
        if shift < 0:
            raise ValueError(f"Swizzle shift must be non-negative, got {shift}")
        self.bits = bits
        self.base = base
        self.shift = shift

    def __call__(self, offset: int) -> int:
        """Apply swizzle to a linear offset."""
        if self.bits == 0:
            return offset
        mask = ((1 << self.bits) - 1) << self.base
        return offset ^ ((offset >> self.shift) & mask)

    def __repr__(self):
        return f"Swizzle(bits={self.bits}, base={self.base}, shift={self.shift})"

    def __eq__(self, other):
        if not isinstance(other, Swizzle):
            return NotImplemented
        return self.bits == other.bits and self.base == other.base and self.shift == other.shift


class SwizzledLayout:
    """A Layout composed with a Swizzle: offset = swizzle(layout(coord)).

    Behaves like a Layout but applies an XOR-based address remapping after
    the normal (Shape, Stride) coordinate-to-offset computation.  Use this
    for threadgroup memory tiles to eliminate bank conflicts.

    Example::

        base_layout = Layout((16, 32), (32, 1))     # 16×32 row-major tile
        sw = Swizzle(bits=3, base=2, shift=5)
        swizzled = SwizzledLayout(base_layout, sw)
        offset = swizzled((row, col))  # bank-conflict-free offset
    """

    __slots__ = ("layout", "swizzle")

    def __init__(self, layout: "Layout", swizzle: Swizzle):
        self.layout = layout
        self.swizzle = swizzle

    def __call__(self, coord) -> int:
        return self.swizzle(self.layout(coord))

    @property
    def shape(self):
        return self.layout.shape

    @property
    def stride(self):
        return self.layout.stride

    def size(self, mode=None) -> int:
        return self.layout.size(mode)

    def rank(self) -> int:
        return self.layout.rank()

    def depth(self) -> int:
        return self.layout.depth()

    def cosize(self) -> int:
        return self.layout.cosize()

    def __repr__(self):
        return f"SwizzledLayout({self.layout!r}, {self.swizzle!r})"

    def __eq__(self, other):
        if not isinstance(other, SwizzledLayout):
            return NotImplemented
        return self.layout == other.layout and self.swizzle == other.swizzle


def swizzle(layout: "Layout", bits: int, base: int, shift: int) -> SwizzledLayout:
    """Compose a Layout with an XOR swizzle for bank-conflict avoidance.

    Parameters
    ----------
    layout : Layout
        The base shared-memory layout (e.g., row-major tile).
    bits : int
        Number of XOR bits (B).
    base : int
        Bit position of the lowest target bit (M).
    shift : int
        Bit position of the lowest source bit (S).

    Returns
    -------
    SwizzledLayout
        A layout whose ``__call__`` applies the swizzle after the normal
        coordinate-to-offset mapping.

    Example — 16×32 float tile in threadgroup memory::

        tile = Layout((16, 32), (32, 1))
        swizzled = swizzle(tile, bits=3, base=2, shift=5)
        # Now swizzled((r, c)) avoids bank conflicts on column access
    """
    return SwizzledLayout(layout, Swizzle(bits, base, shift))


class Layout:
    """CuTe-style layout: (Shape, Stride) -> coordinate-to-offset mapping."""

    __slots__ = ("shape", "stride")

    def __init__(self, shape: Shape, stride: Stride = None):
        if stride is None:
            stride = compact_col_major(shape)
        self.shape = shape
        self.stride = stride

    def __call__(self, coord: Coord) -> int:
        return crd2idx(coord, self.shape, self.stride)

    def size(self, mode=None) -> int:
        s = self.shape
        if mode is not None:
            if isinstance(mode, (list, tuple)):
                for m in mode:
                    s = s[m] if is_tuple(s) else s
            else:
                s = s[mode] if is_tuple(s) else s
        return product(s)

    def rank(self) -> int:
        return _tuple_rank(self.shape)

    def depth(self) -> int:
        return depth(self.shape)

    def cosize(self) -> int:
        """Maximum offset + 1."""
        if self.size() == 0:
            return 0
        max_offset = 0
        for s, d in zip(flatten(self.shape), flatten(self.stride)):
            if s > 0 and d > 0:
                max_offset += (s - 1) * d
        return max_offset + 1

    def __repr__(self):
        return f"{_fmt(self.shape)}:{_fmt(self.stride)}"

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return NotImplemented
        return self.shape == other.shape and self.stride == other.stride


def _fmt(x):
    if is_int(x):
        return str(x)
    return "(" + ",".join(_fmt(e) for e in x) + ")"


# --- Constructors ---


def make_layout(shape: Shape, stride: Stride = None) -> Layout:
    return Layout(shape, stride)


def make_ordered_layout(shape: Shape, order: Tuple[int, ...]) -> Layout:
    """Layout with custom dim ordering. order[i] = priority (0 = innermost)."""
    return Layout(shape, compact_order(shape, order))


def make_identity_layout(shape: Shape) -> Layout:
    return Layout(shape, compact_col_major(shape))


# --- Queries ---


def size(x, mode=None) -> int:
    """Size of a layout, tensor, or shape."""
    if isinstance(x, Layout):
        return x.size(mode)
    if hasattr(x, "layout"):
        return x.size(mode)
    s = x
    if mode is not None:
        if isinstance(mode, (list, tuple)):
            for m in mode:
                s = s[m] if is_tuple(s) else s
        else:
            s = s[mode] if is_tuple(s) else s
    return product(s)


def cosize(x: Layout) -> int:
    return x.cosize()


# --- Coalesce ---


def coalesce(layout: Layout) -> Layout:
    """Flatten and merge adjacent modes with compatible strides."""
    flat_s = list(flatten(layout.shape))
    flat_d = list(flatten(layout.stride))

    pairs = [(s, d) for s, d in zip(flat_s, flat_d) if s != 1]
    if not pairs:
        return Layout(1, 0)

    # Sort by stride for canonical form (required by complement/composition)
    pairs.sort(key=lambda p: (p[1], p[0]))
    new_s, new_d = [pairs[0][0]], [pairs[0][1]]

    for i in range(1, len(pairs)):
        s_i, d_i = pairs[i]
        if d_i == new_d[-1] * new_s[-1]:
            new_s[-1] *= s_i
        else:
            new_s.append(s_i)
            new_d.append(d_i)

    if len(new_s) == 1:
        return Layout(new_s[0], new_d[0])
    return Layout(tuple(new_s), tuple(new_d))


# --- Complement ---


def complement(layout: Layout, cosize_val: int = None) -> Layout:
    """Complementary layout covering elements not in layout's image."""
    coal = coalesce(layout)
    if cosize_val is None:
        cosize_val = coal.cosize()

    sorted_pairs = sorted(
        ((d, s) for d, s in zip(flatten(coal.stride), flatten(coal.shape)) if d != 0),
        key=lambda p: p[0],
    )

    result_s, result_d = [], []
    current = 1

    for d, s in sorted_pairs:
        if d > current:
            result_s.append(d // current)
            result_d.append(current)
        current = d * s

    if current < cosize_val:
        result_s.append(ceil_div(cosize_val, current))
        result_d.append(current)

    if not result_s:
        return Layout(1, cosize_val)
    if len(result_s) == 1:
        return Layout(result_s[0], result_d[0])
    return Layout(tuple(result_s), tuple(result_d))


# --- Composition ---


def composition(a, b) -> Layout:
    """Compose layouts: (a . b)(c) = a(b(c))."""
    if isinstance(a, Layout) and isinstance(b, Layout):
        return _compose_layout_layout(a, b)
    raise TypeError(f"composition not implemented for ({type(a).__name__}, {type(b).__name__})")


def _compose_layout_layout(a: Layout, b: Layout) -> Layout:
    a_coal = coalesce(a)
    if is_tuple(b.shape) and is_tuple(b.stride):
        shapes, strides = [], []
        for sb, db in zip(b.shape, b.stride):
            s, d = _compose_impl(a_coal, sb, db)
            shapes.append(s)
            strides.append(d)
        return Layout(tuple(shapes), tuple(strides))
    s, d = _compose_impl(a_coal, b.shape, b.stride)
    return Layout(s, d)


def _compose_impl(a: Layout, b_shape: Shape, b_stride: Stride):
    if is_tuple(b_shape):
        ss, ds = [], []
        for sb, db in zip(b_shape, b_stride):
            s, d = _compose_impl(a, sb, db)
            ss.append(s)
            ds.append(d)
        return tuple(ss), tuple(ds)

    if b_stride == 0:
        return b_shape, 0

    a_flat_s = list(flatten(a.shape))
    a_flat_d = list(flatten(a.stride))
    result_s, result_d = [], []
    remaining = b_shape

    for sa, da in zip(a_flat_s, a_flat_d):
        if remaining <= 1:
            break
        if da == 0:
            continue
        if b_stride % da == 0:
            scale = b_stride // da
            available = ceil_div(sa, scale) if scale > 0 else sa
            take = min(remaining, available)
            if take > 1:
                result_s.append(take)
                result_d.append(da * scale)
            remaining = ceil_div(remaining, max(take, 1))
        elif da % b_stride == 0:
            take = min(remaining, da // b_stride)
            if take > 1:
                result_s.append(take)
                result_d.append(b_stride)
            remaining = ceil_div(remaining, max(take, 1))
        else:
            g = gcd(da, b_stride)
            take = min(remaining, da // g)
            if take > 1:
                result_s.append(take)
                result_d.append(g)
            remaining = ceil_div(remaining, max(take, 1))

    if not result_s:
        return 1, 0
    if len(result_s) == 1:
        return result_s[0], result_d[0]
    return tuple(result_s), tuple(result_d)


# --- Inverses ---


def right_inverse(layout: Layout) -> Layout:
    """Right inverse: offsets -> coordinates."""
    coal = coalesce(layout)
    flat_s, flat_d = list(flatten(coal.shape)), list(flatten(coal.stride))
    idx = sorted(range(len(flat_d)), key=lambda i: flat_d[i])
    inv_s, inv_d = [0] * len(flat_s), [0] * len(flat_d)
    current = 1
    for i in idx:
        inv_s[i] = flat_s[i]
        inv_d[i] = current
        current *= flat_s[i]
    if len(inv_s) == 1:
        return Layout(inv_s[0], inv_d[0])
    return Layout(tuple(inv_s), tuple(inv_d))


def left_inverse(layout: Layout) -> Layout:
    return right_inverse(make_layout(layout.shape, layout.stride))


# --- Divide ---


def logical_divide(layout: Layout, tiler) -> Layout:
    """Split layout into (tile, rest)."""
    if isinstance(tiler, (int, tuple)) and not isinstance(tiler, Layout):
        tiler = Layout(tiler)
    comp = complement(tiler, size(layout))
    combined = Layout((tiler.shape, comp.shape), (tiler.stride, comp.stride))
    return composition(layout, combined)


def zipped_divide(layout: Layout, tiler) -> Layout:
    """Per-mode divide: result = ((tile_modes), (rest_modes))."""
    if isinstance(tiler, (int, tuple)) and not isinstance(tiler, Layout):
        tiler_shape = tiler if is_tuple(tiler) else (tiler,)
    elif isinstance(tiler, Layout):
        tiler_shape = tiler.shape if is_tuple(tiler.shape) else (tiler.shape,)
    else:
        tiler_shape = (tiler,)

    layout_shape = layout.shape if is_tuple(layout.shape) else (layout.shape,)
    layout_stride = layout.stride if is_tuple(layout.stride) else (layout.stride,)

    tile_s, tile_d, rest_s, rest_d = [], [], [], []
    for i in range(len(layout_shape)):
        s_i, d_i = layout_shape[i], layout_stride[i]
        t_i = tiler_shape[i] if i < len(tiler_shape) else 1
        tile_s.append(t_i)
        tile_d.append(d_i)
        rest_s.append(product(s_i) // product(t_i))
        rest_d.append(d_i * product(t_i))

    if len(tile_s) == 1:
        return Layout((tile_s[0], rest_s[0]), (tile_d[0], rest_d[0]))
    return Layout((tuple(tile_s), tuple(rest_s)), (tuple(tile_d), tuple(rest_d)))


# --- Product ---


def blocked_product(a: Layout, b: Layout) -> Layout:
    """Blocked product: each element of b gets a full copy of a."""
    a_s, a_d = flatten(a.shape), flatten(a.stride)
    b_s, b_d = flatten(b.shape), flatten(b.stride)
    new_s = a_s + b_s
    new_d = a_d + tuple(d * a.cosize() for d in b_d)
    if len(new_s) == 1:
        return Layout(new_s[0], new_d[0])
    return Layout(tuple(new_s), tuple(new_d))


# --- Recast ---


def recast_layout(new_bits: int, old_bits: int, layout: Layout) -> Layout:
    """Rescale layout for different element bit-width."""
    if new_bits == old_bits:
        return layout
    ratio = new_bits // old_bits
    flat_s, flat_d = list(flatten(layout.shape)), list(flatten(layout.stride))
    for i in range(len(flat_d)):
        if flat_d[i] == 1 and flat_s[i] > 1:
            flat_s[i] //= ratio
            for j in range(len(flat_d)):
                if flat_d[j] > 1:
                    flat_d[j] //= ratio
            break
    return _rebuild_hierarchy(layout.shape, flat_s, flat_d)


def _rebuild_hierarchy(ref_shape: Shape, flat_s: list, flat_d: list) -> Layout:
    idx = [0]

    def _build_s(ref):
        if is_int(ref):
            v = flat_s[idx[0]]
            idx[0] += 1
            return v
        return tuple(_build_s(r) for r in ref)

    def _build_d(ref):
        if is_int(ref):
            v = flat_d[idx2[0]]
            idx2[0] += 1
            return v
        return tuple(_build_d(r) for r in ref)

    new_shape = _build_s(ref_shape)
    idx2 = [0]
    new_stride = _build_d(ref_shape)
    return Layout(new_shape, new_stride)


# --- TV Layout ---


def make_layout_tv(thr_layout: Layout, val_layout: Layout) -> Tuple[Shape, Layout]:
    """Construct Thread-Value layout from thread and value layouts.

    Returns (tiler_mn, tv_layout) where tv_layout maps (tid, vid) to
    a col-major linear offset within the tile.
    """
    thr_s = thr_layout.shape if is_tuple(thr_layout.shape) else (thr_layout.shape,)
    thr_d = thr_layout.stride if is_tuple(thr_layout.stride) else (thr_layout.stride,)
    val_s = val_layout.shape if is_tuple(val_layout.shape) else (val_layout.shape,)
    val_d = val_layout.stride if is_tuple(val_layout.stride) else (val_layout.stride,)

    n_modes = len(thr_s)
    assert len(val_s) == n_modes

    tiler = tuple(product(thr_s[i]) * product(val_s[i]) for i in range(n_modes))

    tile_mode_strides = []
    acc = 1
    for i in range(n_modes):
        tile_mode_strides.append(acc)
        acc *= tiler[i]

    # Sort modes by stride (ascending) to match colexicographic decomposition
    def _inv_order(strides):
        flat = [s if is_int(s) else min(flatten(s)) for s in strides]
        return sorted(range(n_modes), key=lambda i: flat[i])

    thr_inv = _inv_order(thr_d)
    val_inv = _inv_order(val_d)

    tv_thr_shape = tuple(thr_s[thr_inv[k]] for k in range(n_modes))
    tv_thr_stride = tuple(
        product(val_s[thr_inv[k]]) * tile_mode_strides[thr_inv[k]] for k in range(n_modes)
    )
    tv_val_shape = tuple(val_s[val_inv[k]] for k in range(n_modes))
    tv_val_stride = tuple(tile_mode_strides[val_inv[k]] for k in range(n_modes))

    tv = Layout((tv_thr_shape, tv_val_shape), (tv_thr_stride, tv_val_stride))
    return tiler, tv
