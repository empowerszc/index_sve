import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C


def mask_ref(x, m):
    # 真参考：nonzero + index_select（index_select 不受覆盖影响）
    flat = m.reshape(-1).nonzero().squeeze(1)
    a = 1
    for s in x.shape[m.dim():]:
        a *= s
    r = x.reshape(-1, a).index_select(0, flat)
    return r.reshape((flat.numel(),) + tuple(x.shape[m.dim():]))


def check(x, m):
    y = x[m]
    r = mask_ref(x, m)
    assert tuple(y.shape) == tuple(r.shape), (y.shape, r.shape)
    assert torch.equal(y, r), "value mismatch"


def test_mask_1d_a1():
    x = torch.randn(50_000)
    m = torch.rand(50_000) < 0.3
    check(x, m)


def test_mask_1d_a16():
    for dt in (torch.float32, torch.float64, torch.int64):
        x = torch.randint(0, 1000, (20_000, 16)).to(dt)
        m = torch.rand(20_000) < 0.3
        check(x, m)


def test_mask_full_dims():
    x = torch.randn(64, 32)
    m = torch.rand(64, 32) < 0.5
    check(x, m)


def test_mask_partial_dims():
    x = torch.randn(64, 32, 8)
    m = torch.rand(64, 32) < 0.5
    check(x, m)
    m1 = torch.rand(64) < 0.5
    check(x, m1)


def test_mask_empty():
    x = torch.randn(20_000, 16)
    check(x, torch.zeros(20_000, dtype=torch.bool))
    x0 = torch.randn(0, 5)
    check(x0, torch.zeros(0, dtype=torch.bool))


def test_mask_byte():
    x = torch.randn(10_000, 8)
    m = (torch.rand(10_000) < 0.3).to(torch.uint8)
    check(x, m)


def test_mask_multithread():
    nt = torch.get_num_threads()
    try:
        x = torch.randn(200_000, 16)
        m = torch.rand(200_000) < 0.3
        for t in (1, 2, 4, 8):
            torch.set_num_threads(t)
            check(x, m)
    finally:
        torch.set_num_threads(nt)


def test_mask_noncontig_inputs():
    nt = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        # 非连续 self（转置）→ 回退路径，仍须正确
        x = torch.randn(16, 9).t()
        check(x, torch.rand(9) < 0.4)
        # 非连续 mask（步进）→ 回退路径
        x2 = torch.randn(9, 4)
        check(x2, (torch.rand(18) < 0.4)[::2])
    finally:
        torch.set_num_threads(nt)


def test_mask_fused_route_used():
    # 该用例在实现前失败（没有 mask_fused_count），确保融合路径真的被走到
    before = C.mask_fused_count()
    x = torch.randn(100_000, 16)
    m = torch.rand(100_000) < 0.3
    x[m]
    assert C.mask_fused_count() == before + 1, "x[mask] did not take the fused path"


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
