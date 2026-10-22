import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # noqa: F401


def check_out(x, indices, ref):
    out = torch.empty(0)          # deliberately wrong shape: dispatcher resizes it
    r = torch.ops.aten.index.Tensor_out(x, indices, out=out)
    assert r is out, "Tensor_out must return the provided out tensor"
    assert tuple(out.shape) == tuple(ref.shape), (out.shape, ref.shape)
    torch.testing.assert_close(out, ref)


def test_index_out_pure():
    x = torch.randn(5_000_000)
    i = torch.randint(0, 5_000_000, (2_000_000,))
    check_out(x, [i], x[i])


def test_index_out_rowcopy():
    x = torch.randn(200, 64)
    i = torch.randint(0, 200, (20_000,))
    check_out(x, [i], x[i])


def test_index_out_mask():
    x = torch.randn(64, 32)
    m = torch.rand(64, 32) > 0.5
    check_out(x, [m], x[m])


def test_index_out_multi_index():
    x = torch.randn(500, 500)
    i = torch.randint(0, 500, (100_000,))
    j = torch.randint(0, 500, (100_000,))
    check_out(x, [i, j], x[i, j])


def test_index_out_noncontiguous_out():
    x = torch.randn(200, 64)
    i = torch.randint(0, 200, (2000,))
    ref = x[i]
    storage = torch.empty(2000, 128)
    out = storage[:, ::2]          # 非连续 out，stride(128, 2)
    r = torch.ops.aten.index.Tensor_out(x, [i], out=out)
    assert r is out
    torch.testing.assert_close(out, ref)


def test_index_out_uses_hpc_kernel():
    # out= 变体由 C++ TORCH_LIBRARY_IMPL(aten, CPU) 直接覆盖：profiler 只应看到
    # aten::index，不应再有 Python 覆盖层的 hpc_index::index_out 嵌套事件。
    x = torch.randn(2_000_000)
    i = torch.randint(0, 2_000_000, (2_000_000,))
    out = torch.empty(2_000_000)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        torch.ops.aten.index.Tensor_out(x, [i], out=out)
    keys = [e.key for e in p.key_averages()]
    assert any(k.startswith("aten::index") for k in keys), keys
    assert not any("hpc_index::index_out" in k for k in keys), keys
    # 自定义错误信息只在本次算子实现中出现（原生 aten 报 cast 错误）
    try:
        torch.ops.aten.index.Tensor_out(
            x, [i], out=torch.empty(1, dtype=torch.float64))
    except RuntimeError as e:
        assert "index_out: self" in str(e), str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_index_out_dtype_mismatch():
    x = torch.randn(10)
    out = torch.empty(2, dtype=torch.float64)
    try:
        torch.ops.aten.index.Tensor_out(x, [torch.tensor([0, 1])], out=out)
    except RuntimeError as e:
        assert "same scalar type" in str(e), str(e)
    else:
        raise AssertionError("expected RuntimeError for dtype mismatch")


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
