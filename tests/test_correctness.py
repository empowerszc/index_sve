import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index
from index_sve import _hpc_index as C


def check_case(self, indices, ref):
    y = index(self, indices)
    torch.testing.assert_close(y, ref)


def test_1d_gather():
    x = torch.arange(10, dtype=torch.float32)
    i = torch.tensor([1, 3, 0, 3, 9])
    check_case(x, [i], x[i])

def test_negative_indices():
    x = torch.arange(10, dtype=torch.float32)
    i = torch.tensor([-1, -10, 0])        # -10 == size
    check_case(x, [i], x[i])

def test_row_gather():
    x = torch.randn(8, 5)
    i = torch.tensor([1, 2, 4])
    check_case(x, [None, i], x[:, i])

def test_dims_after():
    x = torch.randn(8, 5)
    i = torch.tensor([0, 3, 7])
    check_case(x, [i], x[i])

def test_multi_index():
    x = torch.randn(3, 4, 5)
    i = torch.tensor([0, 2])
    j = torch.tensor([1])
    check_case(x, [i, j], x[i, j])        # broadcast

def test_scalar_0dim_index():
    x = torch.randn(4, 5)
    i = torch.tensor(2)
    check_case(x, [i], x[i])

def test_bool_mask():
    x = torch.randn(4, 5)
    m = torch.tensor([True, False, True, False])
    check_case(x, [m], x[m])

def test_2d_mask():
    x = torch.randn(3, 4)
    m = torch.randn(3, 4) > 0
    check_case(x, [m], x[m])

def test_transpose_view():
    x = torch.randn(4, 5).t()             # 非连续
    i = torch.tensor([1, 3])
    check_case(x, [i], x[i])

def test_strided_view():
    x = torch.arange(20, dtype=torch.float32)[::2]     # stride 2
    i = torch.tensor([0, 4])
    check_case(x, [i], x[i])

def test_int32_index():
    x = torch.randn(100)
    i = torch.tensor([10, 20, 30], dtype=torch.int32)
    check_case(x, [i], x[i])

def test_duplicate_and_random():
    x = torch.randn(5000)
    i = torch.randint(0, 5000, (10000,))   # 重复+乱序，K>=1024 走 linearize+gather_index
    check_case(x, [i], x[i])

def test_2d_float64():
    x = torch.randn(64, 32, dtype=torch.float64)
    i = torch.tensor([0, 5, 31])
    check_case(x, [None, i], x[:, i])

def test_all_dtypes_smoke():
    for dt in (torch.float32, torch.float64, torch.int64, torch.int32, torch.int16,
               torch.int8, torch.uint8, torch.bool, torch.float16, torch.bfloat16):
        x = torch.arange(24, dtype=torch.float32).to(dt)
        i = torch.tensor([0, 5, 23])
        check_case(x, [i], x[i])
    xc = torch.randn(4, 3, dtype=torch.complex64)
    i = torch.tensor([0, 2])
    check_case(xc, [i], xc[i])

def test_nd_integer_index():
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    i = torch.tensor([[0, 1], [2, 3]])           # 2-D 下标 -> replacement_shape (2,2), K=4
    check_case(x, [i], x[i])
    j = torch.tensor([[[1], [0]], [[2], [3]]])   # 3-D 下标
    check_case(x, [j], x[j])

def test_nd_integer_index_with_row():
    x = torch.randn(6, 4, 5)
    i = torch.tensor([[0, 1], [2, 3]])
    check_case(x, [None, i], x[:, i])

def test_main_path_int64_idx():
    x = torch.randn(4096, dtype=torch.float64)     # f64 out -> use_i32=false -> idx int64
    i = torch.randint(0, 4096, (4096,))
    check_case(x, [i], x[i])

def test_rowcopy_noncontig_after():
    x = torch.randn(16, 64)
    xt = x.t()                                      # after 维 stride 16 -> 非连续，须走标量
    i = torch.randint(0, 64, (1024,), dtype=torch.int32)
    check_case(xt, [i], xt[i])

def test_linearize_values():
    x = torch.arange(12).reshape(3, 4).float()          # sizes[3,4] strides[4,1]
    i = torch.tensor([2, 0]); j = torch.tensor([3, 1])
    idx = C.linearize_spec(x, [i, j])                    # 元素偏移
    assert idx.dtype == torch.int64
    assert idx.tolist() == [2 * 4 + 3 * 1, 0 * 4 + 1 * 1]  # [11, 1]

def test_complex128():
    x = torch.randn(8, 4, dtype=torch.complex128)       # 16B dtype -> 标量（A8，无 LD1Q）
    i = torch.tensor([1, 3])
    check_case(x, [i], x[i])

def test_half_bf16_int16():
    for dt in (torch.float16, torch.bfloat16, torch.int16):
        x = torch.arange(12, dtype=torch.float32).to(dt)
        i = torch.tensor([0, 11, 5])
        check_case(x, [i], x[i])

def test_bool_uint8_int8():
    for dt in (torch.bool, torch.uint8, torch.int8):
        x = (torch.arange(12) % 2).to(dt)
        i = torch.tensor([0, 11, 5])
        check_case(x, [i], x[i])

def test_f32_overflow_fallback():
    # self.numel() >= 2^31 时 use_i32=False（idx 用 int64、f32 走标量回退）。
    # 用 debug helper 直接验证判定逻辑，不构造 2^31 元素的大张量。
    assert C.idx_dtype_choice(5, "Float") == 1            # < 2^31, f32 -> int32
    assert C.idx_dtype_choice(1 << 31, "Float") == 0      # >= 2^31, f32 -> int64
    assert C.idx_dtype_choice(5, "Double") == 0           # 8B dtype -> int64
    assert C.idx_dtype_choice(5, "Half") == 0             # 2B dtype -> int64

def test_half_large_k_scalar():
    # K>=1024 主路径：half 无 SVE 分支，强制走 linearize + dispatch_scalar
    x = torch.randn(5000).to(torch.float16)
    i = torch.randint(0, 5000, (5000,))
    check_case(x, [i], x[i])

def test_large_k_scalar_dtypes():
    # K>=1024 主路径：2B/1B/bool/complex 均无 SVE 分支，强制走 linearize + dispatch_scalar
    i = torch.randint(0, 5000, (5000,))
    for dt in (torch.float16, torch.bfloat16, torch.int16):
        x = torch.randint(0, 5000, (5000,)).to(dt)
        check_case(x, [i], x[i])
    for dt in (torch.int8, torch.uint8):
        x = torch.randint(0, 100, (5000,)).to(dt)
        check_case(x, [i], x[i])
    x = (torch.arange(5000) % 2).to(torch.bool)
    check_case(x, [i], x[i])
    for dt in (torch.complex64, torch.complex128):
        x = torch.randn(5000).to(dt)
        check_case(x, [i], x[i])

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
