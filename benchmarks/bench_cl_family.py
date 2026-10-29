"""cl/置换布局行索引族前后对照：aten/hpc 两进程交替（RESULTS §2.9.2 复现脚本）。

Usage:
  # 覆盖是进程级开关（index_sve 导入时读取），必须两进程背靠背交替
  python benchmarks/bench_cl_family.py aten
  python benchmarks/bench_cl_family.py hpc

覆盖形态：cl4d 行索引 / cl4d 掩码（K=2048）、cl3d 置换行索引（K=2048）、
contig4d 行索引（**箱态对照**，与 cl 家族同形但走连续路径——它随箱态漂移
的幅度即可标定当轮噪声；R10 复测轮曾见其与 cl 家族同倍劣化 3 倍）。

口径：每进程内 3 rep 取中位数；比值 = aten ÷ hpc。同轮交替为准，
跨轮只比形态。环境变量：HPC_NT（线程数，默认 8）、HPC_REPS（轮数，默认 3）。
"""
import os
import statistics
import sys
import time

modes = ("aten", "hpc")
mode = sys.argv[1] if len(sys.argv) > 1 else "hpc"
if mode not in modes:
    sys.exit(f"usage: {sys.argv[0]} [{'|'.join(modes)}]")
if mode == "aten":
    os.environ["HPC_INDEX_NO_OVERRIDE"] = "1"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def med(f, warmup, iters):
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def main():
    import torch
    import index_sve  # noqa: F401  激活覆盖（aten 模式已由 NO_OVERRIDE 关闭）

    torch.set_num_threads(int(os.environ.get("HPC_NT", "8")))
    cl4 = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last)
    cl3 = torch.randn(4096, 96, 32).permute(0, 2, 1)  # (4096,32,96) strides (3072,1,32)
    c4 = cl4.contiguous()
    rows = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    m = torch.zeros(4096, dtype=torch.bool)
    m[torch.randperm(4096)[:2048]] = True

    cases = [
        ("cl4d-row K2048", lambda: cl4[rows], 3, 50),
        ("cl4d-mask K2048", lambda: cl4[m], 3, 50),
        ("cl3d-row K2048", lambda: cl3[rows], 3, 50),
        ("contig4d K2048(ctl)", lambda: c4[rows], 3, 50),
    ]
    for rep in range(int(os.environ.get("HPC_REPS", "3"))):
        outs = [f"{name}={med(fn, wu, it):9.1f}" for name, fn, wu, it in cases]
        print(f"[{mode} rep{rep}] " + "  ".join(outs), flush=True)


if __name__ == "__main__":
    main()
