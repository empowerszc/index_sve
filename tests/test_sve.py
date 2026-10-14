import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index

def test_f64_pure_gather():
    for K in (1, 7, 8, 9, 16, 100):                    # 尾数谓词边界
        x = torch.randn(K)
        i = torch.randint(0, K, (K * 3,))
        y = index(x, [i])
        torch.testing.assert_close(y, x[i])

def test_f64_row_gather():
    x = torch.randn(64, 8)
    i = torch.tensor([0, 3, 7])
    torch.testing.assert_close(index(x, [None, i]), x[:, i])

def test_i64():
    x = torch.arange(16, dtype=torch.int64)
    i = torch.tensor([0, 15, 5, 5])
    torch.testing.assert_close(index(x, [i]), x[i])

def test_sve_f64_tail():
    for K in (1025, 1031, 1032, 1033, 1040, 1124):     # >=1024 走 SVE；尾数 1/7/0/1/0/4
        x = torch.randn(K, dtype=torch.float64)
        i = torch.randint(0, K, (K,))
        torch.testing.assert_close(index(x, [i]), x[i])

def test_sve_i64():
    for K in (1025, 2000):
        x = torch.randint(-1000, 1000, (K,), dtype=torch.int64)
        i = torch.randint(0, K, (K,))
        torch.testing.assert_close(index(x, [i]), x[i])

def test_f32_pure_gather():
    for K in (1, 15, 16, 17, 100):
        x = torch.randn(K, dtype=torch.float32)
        i = torch.randint(0, K, (K * 3,))
        torch.testing.assert_close(index(x, [i]), x[i])

def test_i32_index_gather():
    x = torch.randn(5000)
    i = torch.tensor([0, 100, 4999, 2500], dtype=torch.int32)
    torch.testing.assert_close(index(x, [i]), x[i])

def test_sve_f32_tail():
    for K in (1025, 1031, 1032, 1033, 1040, 1124):     # >=1024 走 SVE；尾数 1/7/8/9/0/4
        x = torch.randn(K, dtype=torch.float32)
        i = torch.randint(0, K, (K,))
        torch.testing.assert_close(index(x, [i]), x[i])

def test_sve_i32():
    for K in (1025, 2000):
        x = torch.randint(-1000, 1000, (K,), dtype=torch.int32)
        i = torch.randint(0, K, (K,))
        torch.testing.assert_close(index(x, [i]), x[i])

def test_rowcopy_f64():
    x = torch.randn(8, 4, dtype=torch.float64)
    i = torch.tensor([1, 3, 7])
    torch.testing.assert_close(index(x, [i]), x[i])

def test_rowcopy_f32_large_after():
    x = torch.randn(50, 256)
    i = torch.randint(0, 50, (4096,))
    torch.testing.assert_close(index(x, [i]), x[i])

def test_rowcopy_small_after():
    x = torch.randn(50, 3)                        # pos[src] 场景 A=3
    i = torch.randint(0, 50, (8192,))
    torch.testing.assert_close(index(x, [i]), x[i])

def test_rowcopy_with_before():
    x = torch.randn(4, 8, 6, 3)                   # 双 before 行(4,8) + A=3
    i = torch.tensor([1, 5])
    torch.testing.assert_close(index(x, [None, None, i]), x[:, :, i, :])

def test_rowcopy_f64_large_after():
    x = torch.randn(8, 68, dtype=torch.float64)   # A=68: 8B 主循环 8×8 + 尾 4
    i = torch.randint(0, 8, (2000,))
    torch.testing.assert_close(index(x, [i]), x[i])

def test_rowcopy_f32_tail_after():
    x = torch.randn(8, 100)                       # A=100: 4B 主循环 6×16 + 尾 4
    i = torch.randint(0, 8, (2000,))
    torch.testing.assert_close(index(x, [i]), x[i])

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
