"""Tensor = pointer + Layout. Supports both JIT-time symbolic ops and kernel-time IR tracing."""

from __future__ import annotations

from typing import Any

from .core import Layout
from .core import zipped_divide as _layout_zipped_divide
from .tuple import flatten, idx2crd, is_int, is_tuple, product


class Tensor:
    """A tensor: buffer name + layout + base offset (int or IRValue)."""

    def __init__(
        self, name: str, buffer_index: int, metal_dtype: str, layout: Layout, base_offset: Any = 0
    ):
        self.name = name
        self.buffer_index = buffer_index
        self.metal_dtype = metal_dtype
        self.layout = layout
        self.base_offset = base_offset

    @property
    def shape(self):
        return self.layout.shape

    @property
    def stride(self):
        return self.layout.stride

    @property
    def element_type(self):
        return self.metal_dtype

    def size(self, mode=None):
        return self.layout.size(mode)

    def __getitem__(self, coord):
        """Hierarchical slicing: None=keep, int=static fix, IRValue=runtime fix."""
        from ._tracing import IRValue

        if not isinstance(coord, tuple):
            coord = (coord,)

        shape = self.layout.shape if is_tuple(self.layout.shape) else (self.layout.shape,)
        stride = self.layout.stride if is_tuple(self.layout.stride) else (self.layout.stride,)

        offset = self.base_offset
        new_shapes, new_strides = [], []

        for i, c in enumerate(coord):
            s_i, d_i = shape[i], stride[i]

            if c is None:
                new_shapes.append(s_i)
                new_strides.append(d_i)
            elif isinstance(c, IRValue):
                offset = _add_ir_offset(offset, c, s_i, d_i)
            elif is_int(c):
                offset = _add_static_offset(offset, c, s_i, d_i)
            elif is_tuple(c) and is_tuple(s_i):
                sub_shapes, sub_strides = [], []
                for j, cc in enumerate(c):
                    ss = s_i[j]
                    dd = d_i[j] if is_tuple(d_i) else d_i
                    if cc is None:
                        sub_shapes.append(ss)
                        sub_strides.append(dd)
                    elif isinstance(cc, IRValue):
                        offset = _add_ir_offset(offset, cc, ss, dd)
                    elif is_int(cc):
                        offset = _add_static_offset(offset, cc, ss, dd)
                if sub_shapes:
                    new_shapes.append(sub_shapes[0] if len(sub_shapes) == 1 else tuple(sub_shapes))
                    new_strides.append(
                        sub_strides[0] if len(sub_strides) == 1 else tuple(sub_strides)
                    )

        if not new_shapes:
            return offset

        new_shape = new_shapes[0] if len(new_shapes) == 1 else tuple(new_shapes)
        new_stride = new_strides[0] if len(new_strides) == 1 else tuple(new_strides)
        return Tensor(
            self.name,
            self.buffer_index,
            self.metal_dtype,
            Layout(new_shape, new_stride),
            base_offset=offset,
        )

    def __setitem__(self, coord, value):
        """Simple 1D store for naive kernels."""
        from ._tracing import IROp, IRValue, get_builder

        if not isinstance(coord, tuple):
            coord = (coord,)
        if len(coord) == 1 and isinstance(coord[0], IRValue):
            builder = get_builder()
            assert builder is not None
            builder.record(
                IROp(
                    "store",
                    None,
                    [coord[0], value],
                    attrs={"buffer": self.name, "buffer_index": self.buffer_index},
                )
            )
            return
        raise TypeError("Use .store() for TV-layout kernels")

    def load(self):
        """Vectorized load of all elements in this tensor view."""
        from ._tracing import IROp, get_builder

        builder = get_builder()
        assert builder is not None

        flat_s, flat_d = flatten(self.layout.shape), flatten(self.layout.stride)
        n_elem = product(self.layout.shape)
        groups = _group_contiguous(_compute_value_offsets(flat_s, flat_d, n_elem))

        result = builder.new_value(self.metal_dtype)
        result._tv_groups = groups
        builder.record(
            IROp(
                "tv_load",
                result,
                [],
                attrs={
                    "buffer": self.name,
                    "buffer_index": self.buffer_index,
                    "base_offset": self.base_offset,
                    "groups": groups,
                    "dtype": self.metal_dtype,
                    "num_elements": n_elem,
                },
            )
        )
        return result

    def store(self, value):
        """Vectorized store of all elements."""
        from ._tracing import IROp, get_builder

        builder = get_builder()
        assert builder is not None

        flat_s, flat_d = flatten(self.layout.shape), flatten(self.layout.stride)
        n_elem = product(self.layout.shape)
        groups = _group_contiguous(_compute_value_offsets(flat_s, flat_d, n_elem))

        builder.record(
            IROp(
                "tv_store",
                None,
                [value],
                attrs={
                    "buffer": self.name,
                    "buffer_index": self.buffer_index,
                    "base_offset": self.base_offset,
                    "groups": groups,
                    "dtype": self.metal_dtype,
                    "num_elements": n_elem,
                },
            )
        )

    def __repr__(self):
        return f"Tensor({self.name}, layout={self.layout}, offset={self.base_offset})"


def tensor_composition(tensor: Tensor, tv_layout: Layout, tiler) -> Tensor:
    """Compose tensor's tile layout with a TV layout.

    Converts TV layout strides from col-major tile indices to actual memory offsets.
    """
    tile_stride = (
        tensor.layout.stride if is_tuple(tensor.layout.stride) else (tensor.layout.stride,)
    )
    tiler_t = tiler if is_tuple(tiler) else (tiler,)

    def _convert_stride(tile_linear_stride):
        if is_int(tile_linear_stride):
            coord = idx2crd(tile_linear_stride, tiler_t)
            if is_tuple(coord):
                return sum(c * d for c, d in zip(coord, tile_stride))
            return coord * tile_stride[0]
        return tuple(_convert_stride(s) for s in tile_linear_stride)

    new_stride = (_convert_stride(tv_layout.stride[0]), _convert_stride(tv_layout.stride[1]))
    return Tensor(
        tensor.name,
        tensor.buffer_index,
        tensor.metal_dtype,
        Layout(tv_layout.shape, new_stride),
        base_offset=tensor.base_offset,
    )


def tensor_zipped_divide(tensor: Tensor, tiler) -> Tensor:
    """Tile a tensor using zipped_divide."""
    new_layout = _layout_zipped_divide(tensor.layout, tiler)
    return Tensor(
        tensor.name,
        tensor.buffer_index,
        tensor.metal_dtype,
        new_layout,
        base_offset=tensor.base_offset,
    )


def make_identity_tensor(shape) -> Tensor:
    return Tensor("__identity__", -1, "uint", Layout(shape))


# --- Internal helpers ---


def _add_static_offset(base, idx: int, shape, stride):
    if is_tuple(shape):
        coord = idx2crd(idx, shape)
        flat_c = flatten((coord,) if is_int(coord) else coord)
        contribution = sum(c * d for c, d in zip(flat_c, flatten(stride)))
    else:
        contribution = idx * stride
    return base + contribution


def _add_ir_offset(base, ir_idx, shape, stride):
    """Generate IR arithmetic for runtime index decomposition."""
    flat_s = flatten(shape) if is_tuple(shape) else (shape,)
    flat_d = flatten(stride) if is_tuple(stride) else (stride,)

    contribution, remaining = None, ir_idx
    for s, d in zip(flat_s, flat_d):
        component = remaining % s if len(flat_s) > 1 else remaining
        if len(flat_s) > 1:
            remaining = remaining // s
        term = component * d
        contribution = term if contribution is None else contribution + term

    if contribution is None:
        return base
    if isinstance(base, int) and base == 0:
        return contribution
    return base + contribution


def _compute_value_offsets(flat_shape, flat_stride, n_elem):
    offsets = []
    for vid in range(n_elem):
        off, idx = 0, vid
        for s, d in zip(flat_shape, flat_stride):
            off += (idx % s) * d
            idx //= s
        offsets.append(off)
    return offsets


def _group_contiguous(offsets):
    """Group offsets into contiguous runs for vectorization."""
    if not offsets:
        return []
    groups, start, count = [], offsets[0], 1
    for i in range(1, len(offsets)):
        if offsets[i] == offsets[i - 1] + 1:
            count += 1
        else:
            groups.append((start, count))
            start, count = offsets[i], 1
    groups.append((start, count))
    return groups
