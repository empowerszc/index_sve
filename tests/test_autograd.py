"""autograd 专项回归：覆盖激活态下 `x[...]` 的前向记录与 backward 梯度。

背景（override-mechanism.md §3、semantics.md §6.4）：本算子只覆盖
aten::index.Tensor 的 CPU dispatch key，dispatcher 链 AutogradCPU（记录）→
CPU（我们）不受影响；backward 由 ATen 的 IndexBackward 公式（index_put_，
未被覆盖）计算。此前梯度仅有两进程对拍记录，套件无专项回归——本文件补齐。

参考构造原则（同 test_mask_fused.py 规避"对拍陷阱"）：覆盖激活后 `x[...]`
自洽，参考必须用未被覆盖的算子——前向用 index_select，反向用 zeros +
index_add_（负索引自行归一），均不经 aten::index。
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # noqa: F401
from index_sve import _hpc_index as C


def norm_idx(i, n):
    i = i.to(torch.long)
    return torch.where(i < 0, i + n, i)


def test_forward_metadata():
    x = torch.randn(20, 6, requires_grad=True)
    i = torch.tensor([0, 3, 19, 3])
    y = x[i]
    assert y.requires_grad and y.grad_fn is not None and not y.is_leaf
    assert tuple(y.shape) == (4, 6)
    torch.testing.assert_close(y, x.detach().index_select(0, i))
    with torch.no_grad():
        y2 = x[i]
    assert y2.grad_fn is None and not y2.requires_grad


def test_backward_1d_duplicates():
    x = torch.randn(100, requires_grad=True)
    i = torch.randint(0, 100, (37,))
    g = torch.randn(37)
    y = x[i]
    y.backward(g, retain_graph=True)
    ref = torch.zeros(100)
    ref.index_add_(0, i, g)
    torch.testing.assert_close(x.grad, ref)
    y.backward(g)                       # 二次 backward：梯度累加
    torch.testing.assert_close(x.grad, 2 * ref)


def test_backward_negative_indices():
    x = torch.randn(6, requires_grad=True)
    i = torch.tensor([1, -1, 1, -5])    # 归一后 [1, 5, 1, 1]
    g = torch.randn(4)
    y = x[i]
    y.backward(g)
    in_ = norm_idx(i, 6)
    ref = torch.zeros(6)
    ref.index_add_(0, in_, g)
    torch.testing.assert_close(x.grad, ref)
    torch.testing.assert_close(y, x.detach().index_select(0, in_))


def test_backward_column_gather():
    x = torch.randn(8, 5, requires_grad=True)
    i = torch.tensor([0, 3, 1, 3])
    y = x[:, i]                          # B=8、K=4 → gather_direct
    g = torch.randn(4, 8).t()            # 非连续 incoming gradient
    y.backward(g)
    ref = torch.zeros(8, 5)
    ref.index_add_(1, i, g)
    torch.testing.assert_close(x.grad, ref)
    torch.testing.assert_close(y, x.detach().index_select(1, i))


def test_backward_multi_index_broadcast():
    x = torch.randn(3, 4, 5, requires_grad=True)
    i = torch.tensor([[0], [2]])
    j = torch.tensor([[1, 3, 1]])
    y = x[i, j]                          # 广播 (2,3) → 输出 (2,3,5)
    g = torch.randn(2, 3, 5)
    y.backward(g)
    ib = i.expand(2, 3).reshape(-1)
    jb = j.expand(2, 3).reshape(-1)
    pos = ib * 4 + jb
    ref = torch.zeros(3, 4, 5)
    ref.view(12, 5).index_add_(0, pos, g.reshape(6, 5))
    torch.testing.assert_close(x.grad, ref)
    yref = x.detach().reshape(12, 5).index_select(0, pos).reshape(2, 3, 5)
    torch.testing.assert_close(y, yref)


def test_backward_mask_1d():
    x = torch.randn(1000, requires_grad=True)
    m = torch.rand(1000) < 0.3
    y = x[m]                              # mask 快路径（gather_mask）
    g = torch.randn(y.shape[0])
    y.backward(g)
    rows = m.nonzero().squeeze(1)
    ref = torch.zeros(1000)
    ref.index_add_(0, rows, g)
    torch.testing.assert_close(x.grad, ref)


def test_backward_mask_rows():
    x = torch.randn(500, 16, requires_grad=True)
    m = torch.rand(500) < 0.4
    y = x[m]
    g = torch.randn(y.shape[0], 16)
    y.backward(g)
    rows = m.nonzero().squeeze(1)
    ref = torch.zeros(500, 16)
    ref.index_add_(0, rows, g)
    torch.testing.assert_close(x.grad, ref)
    torch.testing.assert_close(y, x.detach().index_select(0, rows))


def test_backward_int32_index():
    x = torch.randn(64, requires_grad=True)
    i = torch.tensor([3, 7, 3], dtype=torch.int32)
    g = torch.randn(3)
    y = x[i]
    y.backward(g)
    ref = torch.zeros(64)
    ref.index_add_(0, i.to(torch.long), g)
    torch.testing.assert_close(x.grad, ref)


def test_backward_noncontig_self():
    xt = torch.randn(6, 8).t()            # (8,6) 视图，strides (1,8)
    xt.requires_grad_(True)
    i = torch.tensor([0, 5, 2])
    y = xt[i]                             # after 非连续 → gather_fused 路由
    g = torch.randn(3, 6)
    y.backward(g)
    ref = torch.zeros(8, 6)
    ref.index_add_(0, i, g)
    torch.testing.assert_close(xt.grad, ref)
    torch.testing.assert_close(y, xt.detach().index_select(0, i))


def test_chained_nonleaf():
    base = torch.randn(64, requires_grad=True)
    x = base * 2
    i = torch.randint(0, 64, (128,))
    (x[i] * 3).sum().backward()
    ref = torch.zeros(64)
    ref.index_add_(0, i, torch.full((128,), 6.0))    # dy/dbase = 2*3 per hit
    torch.testing.assert_close(base.grad, ref)


def test_gradcheck_1d():
    x = torch.randn(8, dtype=torch.double, requires_grad=True)
    i = torch.tensor([0, 2, 5, 2])       # 含重复下标
    assert torch.autograd.gradcheck(lambda t: t[i], (x,))


def test_gradcheck_column():
    x = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
    i = torch.tensor([0, 3, 1, 3])
    assert torch.autograd.gradcheck(lambda t: t[:, i], (x,))


def test_backward_half_bf16():
    for dt in (torch.float16, torch.bfloat16):
        x = torch.randn(64, dtype=dt, requires_grad=True)
        i = torch.tensor([3, 7, 3, 21])
        g = torch.randn(4, dtype=dt)
        y = x[i]
        y.backward(g)
        ref = torch.zeros(64, dtype=dt)
        ref.index_add_(0, i, g)
        torch.testing.assert_close(x.grad, ref)


def test_route_counters_under_autograd():
    # 带 grad 的前向真的穿过本算子内核（AutogradCPU 记录后 redispatch 到 CPU 键）
    x = torch.randn(2000, 64, requires_grad=True)
    i = torch.randint(0, 2000, (8192,))  # M1 raw：A=64 且 B*K=8192≤65536
    before = C.rowcopy_raw_count()
    y = x[i]
    assert C.rowcopy_raw_count() == before + 1, "grad 前向未走 raw rowcopy"
    y.sum().backward()                    # backward 走 index_put_，与本计数无关
    xm = torch.randn(100_000, 16, requires_grad=True)
    m = torch.rand(100_000) < 0.3
    before = C.mask_fused_count()
    xm[m]
    assert C.mask_fused_count() == before + 1, "grad 前向未走 mask 融合路径"


def test_multithread_backward():
    nt = torch.get_num_threads()
    try:
        x = torch.randn(20_000, 64, dtype=torch.double, requires_grad=True)
        i = torch.randint(0, 20_000, (8192,))
        g = torch.randn(8192, 64, dtype=torch.double)
        ref = torch.zeros(20_000, 64, dtype=torch.double)
        ref.index_add_(0, i, g)
        for t in (1, 2, 8):
            torch.set_num_threads(t)
            x.grad = None
            y = x[i]
            y.backward(g)
            torch.testing.assert_close(x.grad, ref)
    finally:
        torch.set_num_threads(nt)


def test_ops_entry_no_autograd():
    # torch.ops.hpc_index.* 只注册了 CPU 键、无 autograd 公式：直调时 fallback
    #（warn 模式）建一个断链的 CppFunction——下游 backward 不报错，但梯度
    # 被静默丢弃、不回传 x。训练场景必须用 x[...]（aten::index 的
    # AutogradCPU 链完整）。
    x = torch.randn(16, requires_grad=True)
    i = torch.tensor([0, 3])
    y = torch.ops.hpc_index.index(x, [i])
    torch.testing.assert_close(y, x.detach().index_select(0, i))
    assert y.requires_grad               # 图被建（断链节点）
    (y * 2).sum().backward()
    assert x.grad is None                # 但梯度不回传——直调不可用于训练


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
