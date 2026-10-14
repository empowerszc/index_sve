import os, sys, torch

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

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
