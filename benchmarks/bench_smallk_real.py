"""小 K 真实路径对拍：`x[idx]`（覆盖开关由 HPC_INDEX_NO_OVERRIDE 控制）。

小 K 的单次调用只有几微秒，`bench_threads.py` 的 pybind `hpc_index()` 包装
（Python 调用 + list 转换）会污染比值；这里用真实 `x[idx]`，并由外层驱动脚本
交替启动 aten/hpc 两个进程（见 `bench_smallk_driver.py`），取每个进程的中位数。

用法（单进程，输出 JSON）：
  HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_smallk_real.py > aten.json
  python benchmarks/bench_smallk_real.py > hpc.json
"""
import ctypes
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # 覆盖是否安装由 HPC_INDEX_NO_OVERRIDE 决定

_LIBOMP = None
for _line in open("/proc/self/maps"):
    if "libomp.so" in _line:
        _LIBOMP = ctypes.CDLL(_line.split()[-1])
        break
if _LIBOMP is not None:
    _LIBOMP.omp_set_num_threads.argtypes = [ctypes.c_int]


def set_threads(t):
    torch.set_num_threads(t)
    if _LIBOMP is not None:
        _LIBOMP.omp_set_num_threads(t)

_affinity = os.environ.get("HPC_AFFINITY", "190-227")
_cpus = set()
for _part in _affinity.replace(" ", "").split(","):
    if "-" in _part:
        _lo, _hi = _part.split("-")
        _cpus.update(range(int(_lo), int(_hi) + 1))
    else:
        _cpus.add(int(_part))
os.sched_setaffinity(0, _cpus)

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,2,4,8,16,32,36,38").split(",")]
ITERS = int(os.environ.get("HPC_ITERS", "200"))
REPS = int(os.environ.get("HPC_REPS", "3"))


def med(fn, n=ITERS):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    g = torch.Generator().manual_seed(7)
    x1 = torch.randn(4096, generator=g)
    x2 = torch.randn(4096, 16, generator=g)
    mode = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
    rows = []
    for t in THREADS:
        set_threads(t)
        for K in (0, 1, 64, 512, 1023, 1024, 4096, 8192, 65536):
            i1 = torch.randint(0, 4096, (K,), generator=g)
            s1 = med(lambda: x1[i1]) * 1e6
            rows.append({"mode": mode, "threads": t, "case": "1D A=1",
                         "K": K, "us": s1})
        for K in (128, 1023, 1024, 4096, 8192, 65536):
            i2 = torch.randint(0, 4096, (K,), generator=g)
            s2 = med(lambda: x2[i2]) * 1e6
            rows.append({"mode": mode, "threads": t, "case": "2D A=16",
                         "K": K, "us": s2})
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
