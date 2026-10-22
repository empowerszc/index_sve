import os, sys, subprocess, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index


def check_oob(v, size, dim):
    x = torch.arange(size, dtype=torch.float32)
    try:
        index(x, [torch.tensor([v])])
    except IndexError as e:
        assert str(e) == f"index {v} is out of bounds for dimension {dim} with size {size}", str(e)
    else:
        raise AssertionError("expected IndexError")

def test_oob_positive():
    check_oob(10, 10, 0)
    check_oob(4, 4, 0)

def test_oob_negative():
    check_oob(-11, 10, 0)
    check_oob(-5, 4, 0)

def test_oob_small_domain_bucket():
    # T4 raw 桶化路径（B==1 A==1 单索引、值域<=2048）：越界必须同样抛错
    # 小张量 + 大越界值使 dom 仍 <=2048，桶化启用时不能吞掉越界
    for v in (200, -11, 10, -10):
        size = 10
        if v == -10:
            # v=-size 是合法负索引（归一为 0），不抛错
            x = torch.arange(size, dtype=torch.int64)
            i = torch.randint(0, size, (2048,))
            i[0] = v
            torch.testing.assert_close(index(x, [i]), x[i])
            continue
        x = torch.arange(size, dtype=torch.int64)
        i = torch.randint(0, size, (2048,))
        i[0] = v
        try:
            index(x, [i])
        except IndexError as e:
            assert str(e) == f"index {v} is out of bounds for dimension 0 with size {size}", str(e)
        else:
            raise AssertionError(f"expected IndexError for v={v}, got none")

def test_empty_indices():
    x = torch.randn(3)
    try:
        index(x, [])
    except IndexError as e:
        assert "at least one index" in str(e)
    else:
        raise AssertionError("expected IndexError")

def test_too_many_indices():
    x = torch.randn(3)
    try:
        index(x, [torch.tensor([0]), torch.tensor([1])])
    except IndexError as e:
        assert "too many indices" in str(e)
    else:
        raise AssertionError("expected IndexError")

def test_meta_index_does_not_crash():
    # 索引张量不在 CPU 上时必须干净报错，而不是解引用空 data_ptr 段错误。
    # T10 起 hpc_index::index 只注册 CPU kernel；meta 输入由 PT2 报
    # "no Meta kernel"（未开 HPC_INDEX_COMPILE 时），同样是无崩溃的干净错误。
    # 子进程隔离：修复前是 SIGSEGV，不能让整个测试进程挂掉。
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import torch, index_sve\n"
        "x = torch.randn(5)\n"
        "i = torch.tensor([0, 1], device='meta')\n"
        "try:\n"
        "    torch.ops.hpc_index.index(x, [i])\n"
        "except (RuntimeError, NotImplementedError) as e:\n"
        "    msg = str(e)\n"
        "    assert ('indices should be either on cpu' in msg) or ('Meta tensors' in msg), msg\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(3)\n"
    )
    env = dict(os.environ, PYTHONPATH=root)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
    assert p.returncode == 0, (
        f"meta index crashed (rc={p.returncode}); stderr tail:\n"
        f"{p.stderr.decode()[-600:]}")

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
