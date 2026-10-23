import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C


def _ref_prefix(x, *indices):
    nidx = len(indices)
    flat = x.reshape(-1, *x.shape[nidx:])
    strides = [0] * nidx
    stride = 1
    for d in range(nidx - 1, -1, -1):
        strides[d] = stride
        stride *= x.shape[d]
    def norm(v, size):
        v = v.reshape(-1).to(torch.int64)
        return torch.where(v < 0, v + size, v)
    off = torch.zeros(indices[0].numel(), dtype=torch.int64)
    for j in range(nidx):
        off = off + norm(indices[j], x.shape[j]) * strides[j]
    out_shape = tuple(indices[0].shape) + tuple(x.shape[nidx:])
    return flat.index_select(0, off).reshape(out_shape)


def test_multi_a1():
    x = torch.randn(64, 32, 8)
    i = torch.randint(0, 64, (8, 64))
    j = torch.randint(0, 32, (8, 64))
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))
    assert C.small_multi_count() == before + 1


def test_multi_a4():
    x = torch.randn(64, 32, 4)
    i = torch.randint(0, 64, (8, 64))
    j = torch.randint(0, 32, (8, 64))
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))


def test_multi_a64_int32():
    x = torch.randn(100, 50, 64)
    i = torch.randint(0, 100, (8, 64), dtype=torch.int32)
    j = torch.randint(0, 50, (8, 64), dtype=torch.int32)
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))


def test_multi_dom1():
    x = torch.randn(64, 32, 4)
    i = torch.full((8, 64), 7, dtype=torch.int64)
    j = torch.full((8, 64), 3, dtype=torch.int64)
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))


def test_multi_negative():
    x = torch.randn(64, 32, 4)
    i = torch.full((8, 64), -1, dtype=torch.int64)
    j = torch.full((8, 64), -2, dtype=torch.int64)
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))


def test_multi_broadcast_not_routed():
    x = torch.randn(64, 32, 4)
    i = torch.randint(0, 64, (8, 64))
    j = torch.randint(0, 32, (1, 1))
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i.expand(8, 64), j.expand(8, 64)))
    assert C.small_multi_count() == before, "broadcasted indices must keep the normal path"


def test_multi_oob():
    x = torch.randn(8, 4, 2)
    i = torch.full((2, 3), 8, dtype=torch.int64)
    j = torch.zeros(2, 3, dtype=torch.int64)
    try:
        index_sve.index(x, [i, j])
    except IndexError as e:
        assert "out of bounds" in str(e)
    else:
        raise AssertionError("out-of-bounds multi-index must fail")


def test_multi_empty():
    x = torch.randn(8, 4, 2)
    i = torch.empty(0, 3, dtype=torch.int64)
    j = torch.empty(0, 3, dtype=torch.int64)
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    assert y.shape == (0, 3, 2)
    assert C.small_multi_count() == before + 1


def test_multi_zero_tail():
    x = torch.randn(8, 4, 0)
    i = torch.randint(0, 8, (8, 64))
    j = torch.randint(0, 4, (8, 64))
    y = index_sve.index(x, [i, j])
    assert y.shape == (8, 64, 0)
    assert y.numel() == 0


def test_multi_noncontig_index_not_routed():
    x = torch.randn(64, 32, 4)
    i = torch.randint(0, 64, (8, 128))[:, ::2]
    j = torch.randint(0, 32, (8, 128))[:, ::2]
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, _ref_prefix(x, i, j))
    assert C.small_multi_count() == before, "non-contiguous index must keep the normal path"


def test_multi_noncontig_not_routed():
    x = torch.randn(64, 32, 4).permute(1, 0, 2)
    i = torch.randint(0, 32, (8, 64))
    j = torch.randint(0, 64, (8, 64))
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    ref = _ref_prefix(x.contiguous(), i, j)
    assert torch.equal(y, ref)
    assert C.small_multi_count() == before, "non-contiguous self must keep the normal path"


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
