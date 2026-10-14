import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import _hpc_index as C


def spec(self, indices):
    r = C.debug_spec(self, indices)
    return {"self": r[0], "dims": r[1], "sizes": r[2], "strides": r[3],
            "B": r[4], "K": r[5], "A": r[6], "out": r[7],
            "row_off": r[8], "col_off": r[9], "after_contiguous": r[10]}

def test_basic_1d():
    x = torch.randn(5)
    s = spec(x, [torch.tensor([1, 3])])
    assert s["B"] == 1 and s["K"] == 2 and s["A"] == 1
    assert s["sizes"] == [5] and s["strides"] == [1]
    assert s["out"] == [2]

def test_row_gather_dims_before():
    x = torch.randn(4, 5)                      # x[:, idx]
    s = spec(x, [None, torch.tensor([1, 2])])
    assert s["B"] == 4 and s["K"] == 2 and s["A"] == 1
    assert s["sizes"] == [5] and s["strides"] == [1]
    assert s["out"] == [4, 2]
    assert s["row_off"] == [0, 5, 10, 15]      # 每行元素偏移

def test_dims_after():
    x = torch.randn(4, 5)                      # x[idx, :]
    s = spec(x, [torch.tensor([1, 3])])
    assert s["B"] == 1 and s["K"] == 2 and s["A"] == 5
    assert s["after_contiguous"] is True
    assert s["out"] == [2, 5]

def test_multi_index_broadcast():
    x = torch.randn(3, 4, 5)
    s = spec(x, [torch.tensor([0, 2]), torch.tensor([1])])   # [2] broadcast [1] -> [2]
    assert s["B"] == 1 and s["K"] == 2 and s["A"] == 5
    assert s["sizes"] == [3, 4]

def test_transpose_to_front():
    x = torch.randn(3, 4, 5)                   # x[i, :, j]: 下标不邻接
    s = spec(x, [torch.tensor([0, 2]), None, torch.tensor([1])])
    assert s["B"] == 1 and s["K"] == 2 and s["A"] == 4
    assert s["out"] == [2, 4]                  # broadcast ++ dims_after(中间维 4)

def test_bool_mask():
    x = torch.randn(4, 5)
    mask = torch.tensor([True, False, True, False])
    s = spec(x, [mask])
    assert s["K"] == 2 and s["sizes"] == [4]

def test_int32_index_kept():
    x = torch.randn(5)
    i = torch.tensor([1, 3], dtype=torch.int32)
    s = spec(x, [i])
    assert s["K"] == 2
    inds = C.debug_spec(x, [i])[11]
    assert len(inds) == 1 and inds[0].dtype == torch.int32

def test_int64_index_kept():
    x = torch.randn(5)
    i = torch.tensor([1, 3], dtype=torch.int64)
    inds = C.debug_spec(x, [i])[11]
    assert len(inds) == 1 and inds[0].dtype == torch.int64

def test_mask_shape_mismatch():
    x = torch.randn(4, 5)
    try:
        spec(x, [torch.tensor([True, False, True])])   # 3 != 4
    except IndexError as e:
        assert "does not match" in str(e)
    else:
        raise AssertionError("expected IndexError")

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
