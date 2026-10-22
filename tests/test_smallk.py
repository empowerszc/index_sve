import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C


def ref(x, idx):
    a = 1
    for s in x.shape[1:]:
        a *= s
    flat = idx.reshape(-1)
    flat = torch.where(flat < 0, flat + x.shape[0], flat)
    r = x.reshape(-1, a).index_select(0, flat)
    return r.reshape(tuple(idx.shape) + tuple(x.shape[1:]))


def check(x, idx):
    y = x[idx]
    r = ref(x, idx)
    assert tuple(y.shape) == tuple(r.shape), (y.shape, r.shape)
    assert torch.equal(y, r), "value mismatch"


def test_small_k_1d():
    nt = torch.get_num_threads()
    try:
        for t in (1, 8):
            torch.set_num_threads(t)
            for dt in (torch.float32, torch.float64, torch.int64):
                x = torch.randint(0, 1000, (4096,)).to(dt)
                for K in (0, 1, 16, 127, 512, 1023, 1024):
                    idx = torch.randint(-4096, 4096, (K,))
                    check(x, idx)
    finally:
        torch.set_num_threads(nt)


def test_small_k_trailing_dims():
    nt = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        x = torch.randn(4096, 5, 7)
        for K in (1, 128, 1023):
            check(x, torch.randint(0, 4096, (K,)))
    finally:
        torch.set_num_threads(nt)


def test_small_k_nd_index():
    nt = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        x = torch.randn(4096, 16)
        for shape in ((4, 5), (2, 3, 5), (1, 7)):
            idx = torch.randint(0, 4096, shape)
            check(x, idx)
    finally:
        torch.set_num_threads(nt)


def test_small_k_scalar_index():
    x = torch.randn(64, 3, 4)
    y = x[torch.tensor(5)]
    assert tuple(y.shape) == (3, 4)
    torch.testing.assert_close(y, x[5])


def test_small_k_fallbacks():
    nt = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        # 非连续 self
        x = torch.randn(16, 9).t()
        check(x, torch.randint(0, 9, (500,)))
        # 非连续 idx
        check(torch.randn(1024, 4), torch.arange(0, 1024, 2)[:500])
        # 多索引不走该路径，但语义须一致
        z = torch.randn(32, 8)
        i, j = torch.randint(0, 32, (5,)), torch.randint(0, 8, (5,))
        torch.testing.assert_close(z[i, j], torch.stack([z[a, b] for a, b in zip(i, j)]))
    finally:
        torch.set_num_threads(nt)


def test_small_k_oob():
    x = torch.arange(10, dtype=torch.float32)
    for v, msg in ((10, "index 10 is out of bounds for dimension 0 with size 10"),
                   (-11, "index -11 is out of bounds for dimension 0 with size 10")):
        try:
            x[torch.tensor([v])]
        except IndexError as e:
            assert str(e) == msg, str(e)
        else:
            raise AssertionError("expected IndexError")


def test_small_k_route_used():
    before = C.small_single_count()
    x = torch.randn(8192, 16)
    x[torch.randint(0, 8192, (1023,))]
    assert C.small_single_count() == before + 1
    x[torch.randint(0, 8192, (4096,))]
    assert C.small_single_count() == before + 2
    x[torch.randint(0, 8192, (4097,))]
    assert C.small_single_count() == before + 2
    # 多索引 / mask 不应命中
    x[torch.randint(0, 8192, (100,)), torch.randint(0, 16, (100,))]
    assert C.small_single_count() == before + 2
    x[torch.rand(8192) < 0.5]
    assert C.small_single_count() == before + 2


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
