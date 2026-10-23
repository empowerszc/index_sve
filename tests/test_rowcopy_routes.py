import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C


def test_small_row_raw_route():
    x = torch.randn(20000, 8)
    idx = torch.randint(0, 20000, (8192,), dtype=torch.int32)
    ref = torch.index_select(x, 0, idx)
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = index_sve.index(x, [idx])
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before + 1, "A=8 multi-thread did not skip linearize"
    finally:
        torch.set_num_threads(old)


def test_multi_index_small_route():
    x = torch.randn(20000, 8, 64)
    i = torch.randint(0, 20000, (8192,))
    j = torch.randint(0, 8, (8192,))
    ref = torch.index_select(x.reshape(-1, 64), 0, i * x.size(1) + j)
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, ref)
    assert C.small_multi_count() == before + 1, "small multi-index did not take the direct path"


def test_multi_index_raw_route():
    x = torch.randn(20000, 8, 64)
    i = torch.randint(0, 20000, (131072,))
    j = torch.randint(0, 8, (131072,))
    ref = torch.index_select(x.reshape(-1, 64), 0, i * x.size(1) + j)
    old = torch.get_num_threads()
    try:
        for t in (16, 36):
            torch.set_num_threads(t)
            before = C.rowcopy_raw_count()
            y = index_sve.index(x, [i, j])
            torch.testing.assert_close(y, ref)
            assert C.rowcopy_raw_count() == before + 1, f"multi-index A=64 t={t} did not skip linearize"
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        index_sve.index(x, [i, j])
        assert C.rowcopy_raw_count() == before, "multi-index t=8 should keep linearize"
    finally:
        torch.set_num_threads(old)


def test_grain_floor():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(32)
        assert C.debug_grain(100) == 64
        assert C.debug_grain(1_000_000) == 1_000_000 // (32 * 8)
    finally:
        torch.set_num_threads(old)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
