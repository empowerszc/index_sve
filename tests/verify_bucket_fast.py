"""两进程对拍：专门覆盖桶化"源 ≤2048 免 pass 1"与"跨度 ≤64KB 直查源"两条快路径
（K>4096 才走得到）。

  HPC_INDEX_NO_OVERRIDE=1 python tests/verify_bucket_fast.py gen
  python tests/verify_bucket_fast.py check
"""
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "agent_space", "bucket_fast_ref.pkl")

DTYPES = [torch.float32, torch.float64, torch.int64, torch.int32,
          torch.float16, torch.bfloat16, torch.int8, torch.bool, torch.complex64]


def cases():
    g = torch.Generator().manual_seed(1234)
    for size in (2048, 1000, 7):
        for n in (4097, 20000):
            # 正负混合、重复、首尾边界
            i = torch.randint(-size, size, (n,), generator=g)
            i[0] = -size
            i[1] = size - 1
            i[2] = i[3]
            for dt in DTYPES:
                if dt in (torch.float32, torch.float64, torch.complex64):
                    x = torch.randn(size, generator=g).to(dt)
                elif dt in (torch.float16, torch.bfloat16):
                    x = torch.randn(size, generator=g).to(dt)
                elif dt is torch.bool:
                    x = (torch.arange(size) % 2).bool()
                else:
                    x = torch.randint(-100, 100, (size,), generator=g).to(dt)
                yield f"size{size}_n{n}_{str(dt).split('.')[-1]}", x, i
    # 跨度 > 64KB：表仍在（stride0=64 的切片源，跨度 512KB）
    x = torch.randn(2048 * 64, generator=g)[::64]
    i = torch.randint(-2048, 2048, (20000,), generator=g)
    i[0] = -2048
    i[1] = 2047
    yield "sliced_s64_n20000", x, i
    # size > 2048：回退"扫值域定界"。注意值域必须收在 2048 宽的窗口内，否则桶化
    # 落选、走到 SVE/单遍，就测不到这条路了。正窗与负窗各一例（负窗的 mn=2048）。
    x = torch.randn(4096, generator=g)
    i = torch.randint(0, 2048, (20000,), generator=g)
    i[0] = -4096          # 归一化到 0
    i[1] = 2047
    yield "size4096_winpos_n20000", x, i
    i = torch.randint(-4096, -2048, (20000,), generator=g)
    i[0] = -4096          # 归一化到 0
    i[1] = -2049          # 归一化到 2047
    yield "size4096_winneg_n20000", x, i


def main():
    mode = sys.argv[1]
    if mode == "gen":
        out = {}
        for name, x, i in cases():
            out[name] = (x, i, x[i])
        pickle.dump(out, open(REF, "wb"))
        print(f"gen {len(out)} cases")
    else:
        ref = pickle.load(open(REF, "rb"))
        bad = 0
        for name, x, i in cases():
            y = x[i]
            xr, ir, want = ref[name]
            if not torch.equal(x, xr) or not torch.equal(i, ir):
                print(f"MISMATCH-INPUT {name}")
                bad += 1
                continue
            if not torch.equal(y, want):
                print(f"MISMATCH {name}")
                bad += 1
        print(f"== {len(ref) - bad}/{len(ref)} identical ==")
        sys.exit(1 if bad else 0)


main()
