"""真实内核的绝对带宽 → 64B 行流量，用来和 memprobe 的硬件上限对照。

degradation-analysis.md §6.4 的"是否撞到带宽墙"结论依赖这张表：1D 随机
gather 每取 1 个 4B/8B 元素都要搬一条 64B line，所以"有用 GB/s"很低，
真正和节点读带宽可比的是 line GB/s。

用法（node 5）：
  numactl --membind=5 --physcpubind=190-227 python benchmarks/bench_hw_traffic.py
"""
import statistics
import time

import torch
import index_sve  # noqa: F401  (installs the aten::index CPU override)

K = 2_000_000


def bench(dtype, nt, reps=15):
    torch.set_num_threads(nt)
    src = torch.randn(K, dtype=dtype)
    idx = torch.randint(0, K, (K,), dtype=torch.int64)
    for _ in range(3):
        torch.ops.aten.index.Tensor(src, [idx])
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.ops.aten.index.Tensor(src, [idx])
        ts.append(time.perf_counter() - t0)
    ms = statistics.median(ts) * 1e3
    useful = K * src.element_size() / (ms * 1e-3) / 1e9
    lines = K * 64 / (ms * 1e-3) / 1e9
    print(f"{str(dtype):18s} nt={nt:2d}  {ms:7.2f} ms  useful={useful:5.2f} GB/s  "
          f"line={lines:5.1f} GB/s")


if __name__ == "__main__":
    for nt in (1, 36):
        bench(torch.float32, nt)
    for nt in (1, 36):
        bench(torch.float64, nt)
