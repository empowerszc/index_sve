"""单 case 的 perf 驱动（供 `perf stat/record` 使用）。

用法：
  MODE=hpc  python benchmarks/perf_one.py col 1 5
  MODE=aten HPC_INDEX_NO_OVERRIDE=1 python benchmarks/perf_one.py col 1 5

第二个参数是线程数，第三个是秒数。注意 MODE=aten 时必须同时设
HPC_INDEX_NO_OVERRIDE=1，否则 `x[...]` 仍走本算子。
"""
import ctypes
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve

_affinity = os.environ.get("HPC_AFFINITY", "190-227")
_cpus = set()
for _part in _affinity.replace(" ", "").split(","):
    if "-" in _part:
        _lo, _hi = _part.split("-")
        _cpus.update(range(int(_lo), int(_hi) + 1))
    else:
        _cpus.add(int(_part))
os.sched_setaffinity(0, _cpus)

_LIBOMP = None
for _line in open("/proc/self/maps"):
    if "libomp.so" in _line:
        _LIBOMP = ctypes.CDLL(_line.split()[-1])
        break


def set_threads(t):
    torch.set_num_threads(t)
    if _LIBOMP is not None:
        _LIBOMP.omp_set_num_threads(t)


case, t, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
mode = os.environ.get("MODE", "hpc")
set_threads(t)
g = torch.Generator().manual_seed(7)

# 行拷贝形态：K = rows//2，路由由 K 与 A 决定（见 op.cpp 的 M1 raw 守卫）
ROWCASE = {
    "r3": (1400000, 3),        # A=3 A<16 → raw
    "r16raw": (131072, 16),    # A=16 且 B*K=65536 → raw
    "r16": (262144, 16),       # A=16 且 B*K=131072 → linearize + rowcopy_sve
    "r64raw": (65536, 64),     # A=64 且 B*K=32768 → raw
    "r64": (262144, 64),       # A=64 且 B*K=131072 → linearize + rowcopy_sve
    "r256": (65536, 256),      # 宽行（生产 A=256 两遍模型）→ linearize + rowcopy_sve
}

if case == "none":
    time.sleep(secs)
    sys.exit(0)

if case == "col":
    x = torch.randn(200000, 64, generator=g)
    i = torch.randint(0, 64, (64,), generator=g)
    f = (lambda: x[:, i]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [None, i]))
elif case == "s1":
    x = torch.randn(50000, 64, generator=g)
    i = torch.randint(0, 50000, (1000000,), dtype=torch.int32, generator=g)
    f = (lambda: x[i]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [i]))
elif case == "bf16":
    x = torch.randn(2000000, generator=g).to(torch.bfloat16)
    i = torch.randint(0, 2000000, (2000000,), generator=g)
    f = (lambda: x[i]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [i]))
elif case == "mask_noncontig":
    x = torch.randn(1000000, 16, generator=g)
    m = (torch.rand(2000000, generator=g) < 0.3)[::2]
    f = (lambda: x[m]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [m]))
elif case == "s4":
    x = torch.randn(200000, dtype=torch.float64, generator=g)
    i = torch.randperm(200000, generator=g)
    f = (lambda: x[i]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [i]))
elif case in ROWCASE:
    rows, A = ROWCASE[case]
    x = torch.randn(rows, A, generator=g)
    i = torch.randint(0, rows, (rows // 2,), generator=g)
    f = (lambda: x[i]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [i]))
elif case == "multi":
    x = torch.randn(5000, 5000, generator=g)
    i = torch.randint(0, 5000, (100000,), generator=g)
    j = torch.randint(0, 5000, (100000,), generator=g)
    f = (lambda: x[i, j]) if mode == "aten" else (lambda: torch.ops.hpc_index.index(x, [i, j]))
else:
    raise SystemExit("unknown case")

for _ in range(3):
    f()
t0 = time.time()
n = 0
while time.time() - t0 < secs:
    f()
    n += 1
print(case, mode, t, n)
