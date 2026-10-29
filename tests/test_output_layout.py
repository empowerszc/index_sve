"""输出内存布局与 aten 一致性（TensorIterator 置换恢复语义）。

aten::index 的输出不是恒 NCHW 连续：输出由 TensorIterator 分配——按
operands（restrided self + reshape/broadcast 后的 index 张量）的 stride 做
should_swap 插入排序得 perm，再按 perm 序 dense 分配
（aten/src/ATen/TensorIterator.cpp NOTE [Computing output strides]）。

E2E 影响：channels_last（CNN/AMP 常态）self 上 x[idx] 若返回 NCHW 连续，
每次索引都静默打断格式传播链；而连续 self 恒为连续输出（零偏差）。
本文件用进程内元数据断言（stride/内存格式）钉住真值（各期望值均为
HPC_INDEX_NO_OVERRIDE=1 无覆盖进程对 torch 2.10.0+cpu 的实测）；值级+stride
的两进程真 aten 对拍见 tests/verify_e2e.py（g5 组）。
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # noqa: F401  激活 aten::index.Tensor 覆盖（被测对象）


def _cl4d():
    return torch.randn(4, 32, 14, 14).to(memory_format=torch.channels_last)


def test_cl4d_row_index_keeps_channels_last():
    y = _cl4d()[torch.arange(4)]
    assert y.is_contiguous(memory_format=torch.channels_last), y.stride()
    assert y.stride() == (6272, 1, 448, 32)


def test_cl4d_row_index_neg_dup_keeps_channels_last():
    y = _cl4d()[torch.tensor([3, -1, 3, 0])]
    assert y.stride() == (6272, 1, 448, 32), y.stride()


def test_cl4d_noncontig_row_index_keeps_channels_last():
    y = _cl4d()[torch.arange(20)[::2] % 4]
    assert y.stride() == (6272, 1, 448, 32), y.stride()


def test_cl4d_mask_keeps_channels_last():
    m = torch.tensor([True, False, True, False])
    y = _cl4d()[m]
    assert y.stride() == (6272, 1, 448, 32), y.stride()


def test_cl4d_2d_index_keeps_cl():
    y = _cl4d()[torch.arange(6).reshape(2, 3) % 4]
    assert y.stride() == (18816, 6272, 1, 448, 32), y.stride()


def test_cl4d_row_then_slice_keeps_cl_stride():
    y = _cl4d()[torch.arange(2), 1:3]
    assert y.stride() == (392, 1, 28, 2), y.stride()


def test_cl4d_newaxis_keeps_cl():
    y = _cl4d()[torch.arange(2), None]
    assert y.stride() == (6272, 32, 1, 448, 32), y.stride()


def test_cl4d_col_index_is_contiguous():
    y = _cl4d()[:, torch.arange(5)]
    assert y.is_contiguous(), y.stride()


def test_cl4d_multi2_is_contiguous():
    y = _cl4d()[torch.arange(2).unsqueeze(1), torch.arange(3).unsqueeze(0)]
    assert y.is_contiguous(), y.stride()


def test_cl4d_broadcast_multi_is_contiguous():
    y = _cl4d()[torch.arange(3).unsqueeze(1) % 4, torch.arange(2).unsqueeze(0)]
    assert y.is_contiguous(), y.stride()


def test_cl4d_ellipsis_t_permuted_stride():
    y = _cl4d()[..., torch.arange(3)]
    assert y.stride() == (1344, 3, 96, 1), y.stride()


def test_cl5d_row_index_keeps_cl3d():
    x = torch.randn(2, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)
    y = x[torch.arange(2)]
    assert y.stride() == (360, 1, 90, 18, 3), y.stride()


def test_cl5d_col_index_is_contiguous():
    x = torch.randn(2, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)
    y = x[:, torch.arange(2)]
    assert y.is_contiguous(), y.stride()


def test_3d_inner_transpose_row_index_keeps_strides():
    x = torch.randn(4, 5, 6).transpose(1, 2)
    y = x[torch.arange(2)]
    assert y.stride() == (30, 1, 6), y.stride()


def test_3d_inner_transpose_2d_index_keeps_strides():
    x = torch.randn(4, 5, 6).transpose(1, 2)
    y = x[torch.arange(4).reshape(2, 2)]
    assert y.stride() == (60, 30, 1, 6), y.stride()


def test_contig_self_row_index_contiguous_out():
    x = torch.randn(8, 32, 14, 14)
    y = x[torch.arange(5)]
    assert y.is_contiguous(), y.stride()


def test_contig_self_col_index_contiguous_out():
    x = torch.randn(4, 32, 14, 14)
    y = x[:, torch.arange(5)]
    assert y.is_contiguous(), y.stride()


def test_contig_self_multi_contiguous_out():
    x = torch.randn(4, 32, 14, 14)
    y = x[torch.arange(2).unsqueeze(1), torch.arange(3).unsqueeze(0)]
    assert y.is_contiguous(), y.stride()


def test_contig2d_transposed_2d_index_noncontig_out():
    x = torch.randn(5, 6)
    t = torch.arange(6).reshape(2, 3).t() % 5
    y = x[t]
    assert y.stride() == (6, 18, 1), y.stride()


def test_contig3d_transposed_2d_index_noncontig_out():
    x = torch.randn(5, 6, 7)
    t = torch.arange(6).reshape(2, 3).t() % 5
    y = x[t]
    assert y.stride() == (42, 126, 7, 1), y.stride()


def test_2d_transpose_self_contiguous_out():
    x = torch.randn(5, 6).t()
    y = x[torch.arange(3)]
    assert y.is_contiguous(), y.stride()


def test_strided_self_contiguous_out():
    x = torch.randn(8, 32, 14, 14)[::2]
    y = x[torch.arange(3)]
    assert y.is_contiguous(), y.stride()


def test_expanded_self_contiguous_out():
    x = torch.randn(1, 5, 6).expand(4, -1, -1)
    y = x[torch.arange(3)]
    assert y.is_contiguous(), y.stride()


def test_values_unchanged_cl4d_row():
    x = _cl4d()
    rows = torch.tensor([3, -1, 3, 0])
    y = x[rows]
    ref = x.contiguous()[rows].contiguous()
    assert torch.equal(y, ref)


def test_values_unchanged_cl5d_row():
    x = torch.randn(2, 3, 4, 5, 6).to(memory_format=torch.channels_last_3d)
    rows = torch.tensor([1, 0])
    y = x[rows]
    ref = x.contiguous()[rows].contiguous()
    assert torch.equal(y, ref)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
