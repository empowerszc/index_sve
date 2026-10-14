import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index

def test_registered():
    x = torch.arange(12).reshape(3, 4).float()
    y = index(x, [torch.tensor([2, 0])])
    torch.testing.assert_close(y, x[torch.tensor([2, 0])])

def test_with_none():
    x = torch.randn(4, 5)
    i = torch.tensor([1, 3])
    torch.testing.assert_close(index(x, [None, i]), x[:, i])

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
