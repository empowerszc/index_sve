import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C


def check(x, idx, threads=(1, 8)):
    ref = torch.index_select(x, 0, idx)
    old = torch.get_num_threads()
    try:
        for t in threads:
            torch.set_num_threads(t)
            y = index_sve.index(x, [idx])
            assert torch.equal(y, ref), (x.shape, x.stride(), idx.shape, t)
    finally:
        torch.set_num_threads(old)


def test_strided_run10():
    x = torch.randn(64, 200, 10).permute(1, 0, 2)
    idx = torch.randint(0, 200, (4096,))
    check(x, idx)


def test_strided_run16():
    x = torch.randn(64, 200, 16).permute(1, 0, 2)
    idx = torch.randint(0, 200, (8192,))
    check(x, idx)


def test_strided_run1():
    x = torch.randn(64, 20000).t()
    idx = torch.randint(0, 20000, (4096,))
    check(x, idx)


def test_strided_int32_index():
    x = torch.randn(64, 20000).t()
    idx = torch.randint(0, 20000, (8192,), dtype=torch.int32)
    check(x, idx)


def test_strided_route_used():
    x = torch.randn(64, 200, 10).permute(1, 0, 2)
    idx = torch.randint(0, 200, (4096,))
    before = C.strided_rows_count()
    index_sve.index(x, [idx])
    assert C.strided_rows_count() == before + 1, "run>=4 tail did not use strided-rows"


def test_strided_run1_not_used():
    x = torch.randn(64, 20000).t()
    idx = torch.randint(0, 20000, (4096,))
    before = C.strided_rows_count()
    index_sve.index(x, [idx])
    assert C.strided_rows_count() == before, "run=1 tail must keep the scalar/fused path"


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
