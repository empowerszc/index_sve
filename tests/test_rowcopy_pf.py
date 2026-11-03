"""行拷贝软件预取（H1）的正确性守护。

预取只改循环结构，输出必须与关掉预取时**逐位一致**；`HPC_PF_DIST` 是运行时旋钮
（每次调用读一次），所以这里可以在同一进程内把 D 扫一遍，包括 D > K、D 越过
下标数组尾巴、D=0 这些边界。性能对照见 RESULTS §2.18。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # noqa: F401

PF_ENV = "HPC_PF_DIST"


def _set_d(d):
    if d is None:
        os.environ.pop(PF_ENV, None)
    else:
        os.environ[PF_ENV] = str(d)


def _check_all_d(x, idx, ref, threads=1, ds=(None, 0, 1, 3, 16, 64, 4096)):
    old = torch.get_num_threads()
    base = None
    try:
        torch.set_num_threads(threads)
        for d in ds:
            _set_d(d)
            y = index_sve.index(x, [idx])
            assert y.shape == ref.shape, (d, y.shape, ref.shape)
            if base is None:
                assert torch.equal(y, ref), f"D={d} differs from aten"
                base = y
            else:
                assert torch.equal(y, base), f"D={d} changed the output"
    finally:
        torch.set_num_threads(old)
        _set_d(None)


def test_pf_invariance_short_row():
    # A=3 → raw 行拷贝（A<16 直接放行）
    torch.manual_seed(0)
    x = torch.randn(4096, 3)
    idx = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    _check_all_d(x, idx, torch.index_select(x, 0, idx))


def test_pf_invariance_one_line_row():
    # A=16、B*K<=65536 → raw；行宽恰好一条 line，预取收益最大的一档
    torch.manual_seed(1)
    x = torch.randn(4096, 16)
    idx = torch.randint(0, 4096, (8192,), dtype=torch.int64)
    ref = torch.index_select(x, 0, idx)
    _check_all_d(x, idx, ref)
    _check_all_d(x, idx, ref, threads=8)


def test_pf_invariance_wide_row():
    # A=64（256B/行，4 条 line）→ raw
    torch.manual_seed(2)
    x = torch.randn(2048, 64)
    idx = torch.randint(0, 2048, (4096,), dtype=torch.int64)
    _check_all_d(x, idx, torch.index_select(x, 0, idx))


def test_pf_invariance_linearize_path():
    # A=16、B*K>65536 → 不走 raw，走 linearize + rowcopy_sve（预取的第二个落点）
    torch.manual_seed(3)
    x = torch.randn(262144, 16)
    idx = torch.randint(0, 262144, (131072,), dtype=torch.int64)
    _check_all_d(x, idx, torch.index_select(x, 0, idx), threads=8,
                 ds=(None, 0, 16, 4096))


def test_pf_invariance_broadcast():
    # B>1：x 是 3D，下标只作用于第 0 维 → 每个选中行块重复 B 次（B=4、A=16）
    torch.manual_seed(4)
    x = torch.randn(600, 4, 16)
    idx = torch.randint(0, 600, (600,), dtype=torch.int64)
    _check_all_d(x, idx, torch.index_select(x, 0, idx), ds=(None, 0, 16, 64))


def test_pf_negative_index():
    torch.manual_seed(5)
    x = torch.randn(2048, 16)
    idx = torch.tensor([-1, -2, 0, -2048, 7, -8] * 400, dtype=torch.int64)
    _check_all_d(x, idx, torch.index_select(x, 0, idx % 2048), ds=(None, 0, 16))


def test_pf_oob_reports_first_bad_index():
    # 预取会提前读第 k+D 个下标，但越界判定/报错仍必须按原顺序发生在第 1 个坏下标上
    torch.manual_seed(6)
    x = torch.randn(1500, 16)
    idx = torch.randint(0, 1500, (1500,), dtype=torch.int64)
    idx[100] = 200000
    idx[1300] = 300000
    for d in (None, 0, 16, 64):
        _set_d(d)
        try:
            index_sve.index(x, [idx])
        except IndexError as e:
            msg = str(e)
            assert "200000" in msg and "300000" not in msg, f"D={d}: {msg}"
        else:
            raise AssertionError(f"D={d} did not raise")
        finally:
            _set_d(None)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
