import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from enigma.core import (
    Layout,
    coalesce,
    complement,
    make_layout_tv,
    make_ordered_layout,
    recast_layout,
    zipped_divide,
)
from enigma.tuple import (
    compact_col_major,
    compact_order,
    crd2idx,
    depth,
    elem_scale,
    flatten,
    idx2crd,
    is_compatible,
    is_congruent,
    product,
    rank,
    select,
)


class TestTuple(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(rank(8), 1)
        self.assertEqual(rank((4, 8)), 2)
        self.assertEqual(rank((4, (2, 3))), 2)

    def test_depth(self):
        self.assertEqual(depth(5), 0)
        self.assertEqual(depth((3, 4)), 1)
        self.assertEqual(depth((3, (4, 5))), 2)

    def test_flatten(self):
        self.assertEqual(flatten(5), (5,))
        self.assertEqual(flatten((4, (2, 3))), (4, 2, 3))

    def test_product(self):
        self.assertEqual(product(8), 8)
        self.assertEqual(product((4, 8)), 32)
        self.assertEqual(product((4, (2, 3))), 24)

    def test_crd2idx_idx2crd_roundtrip(self):
        for shape in [(8,), (4, 8), (2, 3, 4)]:
            stride = compact_col_major(shape if len(shape) > 1 else shape[0])
            n = product(shape if len(shape) > 1 else shape[0])
            for i in range(n):
                s = shape if len(shape) > 1 else shape[0]
                c = idx2crd(i, s)
                self.assertEqual(i, crd2idx(c, s, stride))

    def test_compact_order(self):
        self.assertEqual(compact_order((4, 64), (1, 0)), (64, 1))
        self.assertEqual(compact_order((4, 64), (0, 1)), (1, 4))

    def test_select(self):
        self.assertEqual(select((10, 20, 30), mode=1), 20)
        self.assertEqual(select((10, 20, 30), mode=[2, 0]), (30, 10))

    def test_elem_scale(self):
        self.assertEqual(elem_scale((2, 3), (10, 100)), (20, 300))

    def test_predicates(self):
        self.assertTrue(is_congruent((4, 8), (4, 8)))
        self.assertFalse(is_congruent((4, 8), (8, 4)))
        self.assertTrue(is_compatible((4, 8), (32,)))


class TestLayout(unittest.TestCase):
    def test_basic(self):
        L = Layout((4, 8), (1, 4))
        self.assertEqual(L((0, 0)), 0)
        self.assertEqual(L((2, 1)), 6)
        self.assertEqual(L.size(), 32)
        self.assertEqual(L.cosize(), 32)

    def test_make_ordered_layout(self):
        L = make_ordered_layout((4, 64), order=(1, 0))
        self.assertEqual(L.stride, (64, 1))

    def test_coalesce(self):
        self.assertEqual(coalesce(Layout((2, 6), (1, 2))).shape, 12)
        self.assertEqual(coalesce(Layout((4, 1, 8), (1, 999, 4))).shape, 32)

    def test_complement(self):
        self.assertEqual(complement(Layout(4, 1)).shape, 1)
        self.assertEqual(complement(Layout(4, 2)).shape, 2)
        self.assertEqual(complement(Layout(4, 2)).stride, 1)

    def test_recast_layout(self):
        R = recast_layout(16, 8, Layout((16, 16), (16, 1)))
        self.assertEqual(flatten(R.shape), (16, 8))
        self.assertEqual(flatten(R.stride), (8, 1))

    def test_zipped_divide_preserves_size(self):
        L = Layout((64, 512), (512, 1))
        self.assertEqual(zipped_divide(L, (16, 8)).size(), 64 * 512)

    def test_zipped_divide_unique_offsets(self):
        L = Layout((64, 512), (512, 1))
        R = zipped_divide(L, (16, 8))
        offsets = {R(i) for i in range(R.size())}
        self.assertEqual(len(offsets), R.size())


class TestTVLayout(unittest.TestCase):
    def test_make_layout_tv(self):
        thr = make_ordered_layout((4, 64), order=(1, 0))
        val = Layout((16, 8), (8, 1))
        tiler, tv = make_layout_tv(thr, val)
        self.assertEqual(tiler, (64, 512))
        self.assertEqual(tv.size(mode=0), 256)
        self.assertEqual(tv.size(mode=1), 128)

    def test_tv_unique_offsets(self):
        thr = make_ordered_layout((4, 64), order=(1, 0))
        val = Layout((16, 8), (8, 1))
        _, tv = make_layout_tv(thr, val)
        offsets = {tv((tid, vid)) for tid in range(256) for vid in range(128)}
        self.assertEqual(len(offsets), 32768)
        self.assertEqual(min(offsets), 0)
        self.assertEqual(max(offsets), 32767)

    def test_tv_coalesced_access(self):
        thr = make_ordered_layout((4, 64), order=(1, 0))
        val = Layout((16, 8), (8, 1))
        _, tv = make_layout_tv(thr, val)
        off0, off1 = tv((0, 0)), tv((1, 0))
        self.assertEqual(off0 % 64, off1 % 64)
        self.assertEqual(off1 // 64 - off0 // 64, 8)

    def test_tv_f32_layout(self):
        thr = make_ordered_layout((4, 64), order=(1, 0))
        val = make_ordered_layout((4, 4), order=(1, 0))
        tiler, tv = make_layout_tv(thr, val)
        self.assertEqual(tiler, (16, 256))
        self.assertEqual(tv.size(mode=0), 256)
        self.assertEqual(tv.size(mode=1), 16)
        offsets = {tv((tid, vid)) for tid in range(256) for vid in range(16)}
        self.assertEqual(len(offsets), 4096)


if __name__ == "__main__":
    unittest.main()
