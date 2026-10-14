import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index

def test_s1_spconv_row_gather():
    N, C, R = 50_000, 64, 400_000
    features = torch.randn(N, C)
    inp_idx = torch.randint(0, N, (R,), dtype=torch.int32)
    torch.testing.assert_close(index(features, [inp_idx]), features[inp_idx])

def test_s1_bucket_pattern():
    N, C, nb, nk = 50_000, 32, 27, 2000
    features = torch.randn(N, C)
    rulebook = torch.randint(0, N, (nb, nk), dtype=torch.int32)
    flat = rulebook.reshape(-1)
    y = index(features, [flat]).view(nb, nk, C)
    ref = features[flat].view(nb, nk, C)
    torch.testing.assert_close(y, ref)

def test_s2_padding_sentinel():
    N, C = 10_000, 16
    features = torch.randn(N, C)
    idx = torch.tensor([-1, 100, 5, -1], dtype=torch.int64)
    torch.testing.assert_close(index(features, [idx]), features[idx])

def test_s3_col_perm():
    x = torch.randn(128, 512)
    perm = torch.randperm(512)
    torch.testing.assert_close(index(x, [None, perm]), x[:, perm])

def test_s4_random_big():
    x = torch.randn(1_000_000)
    perm = torch.randperm(1_000_000)
    torch.testing.assert_close(index(x, [perm]), x[perm])

def test_p1_h_src():
    N, C, E = 200_000, 128, 2_000_000
    h = torch.randn(N, C)
    src = torch.randint(0, N, (E,), dtype=torch.int64)
    torch.testing.assert_close(index(h, [src]), h[src])

def test_p2_pos_src():
    N, E = 200_000, 2_000_000
    pos = torch.randn(N, 3)
    src = torch.randint(0, N, (E,), dtype=torch.int64)
    torch.testing.assert_close(index(pos, [src]), pos[src])

def test_p3_mask_select():
    N, C = 100_000, 32
    x = torch.randn(N, C)
    m = torch.rand(N) < 0.3
    torch.testing.assert_close(index(x, [m]), x[m])

def test_p4_embedding():
    vocab, C = 500_000, 64
    emb = torch.randn(vocab, C)
    sp = torch.randint(0, vocab, (2_000_000,), dtype=torch.int32)
    torch.testing.assert_close(index(emb, [sp]), emb[sp])

if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
