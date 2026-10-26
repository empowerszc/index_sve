"""Real-path workload A/B: `x[idx]` in two processes (aten vs hpc).

This benchmark exists because almost every other script calls
``hpc_index(x, [idx])`` directly.  That bypasses the dispatcher and makes
short calls look better than they are in a real model.  Here both sides go
through the real ``x[idx]`` path; the driver starts one process with
``HPC_INDEX_NO_OVERRIDE=1`` (native aten) and one without (index_sve).

Cases are chosen from the real workload review:
  - M5 scalar loop (FPS / ball_query style, K=1 repeated many times)
  - shortq A in {2,3,4}, K in {64,256,512,1023}, i32/i64
  - A1 single-element high-frequency gather (i64 fallback candidate)
  - M1 realistic weight blocks (<=16KB per row)
  - M2 representative spconv row gathers

Note: the M5/A1 absolute per-call times include the Python loop / tensor-iteration
overhead of the harness (identical on both sides), so quote the aten/hpc ratio,
not the absolute us, in results.

Usage (single process, JSON):
  HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_realpath.py > aten.json
  python benchmarks/bench_realpath.py > hpc.json

Use benchmarks/bench_realpath_driver.py to run both modes and print a table.
"""
import ctypes
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve  # override controlled by HPC_INDEX_NO_OVERRIDE

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

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,8,16,38").split(",")]
ITERS = int(os.environ.get("HPC_ITERS", "100"))
LOOP_ITERS = int(os.environ.get("HPC_LOOP_ITERS", "5"))
BIG_ITERS = int(os.environ.get("HPC_BIG_ITERS", "20"))
CALLS = int(os.environ.get("HPC_CALLS", "10000"))


def med(fn, n=ITERS):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def add(rows, mode, threads, case, us):
    rows.append({"mode": mode, "threads": threads, "case": case, "us": us})


def scalar_loop_us(x, idx, n=CALLS, iters=LOOP_ITERS):
    def run():
        for i in idx:
            x[i]
    return med(run, n=iters) / n * 1e6


def main():
    g = torch.Generator().manual_seed(11)
    mode = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
    rows = []

    for t in THREADS:
        set_threads(t)

        # --- M5 scalar loop: 1D and 2D A=3, one element per call ---
        x = torch.randn(4096, generator=g)
        idx = torch.randint(0, 4096, (CALLS, 1), generator=g)
        add(rows, mode, t, f"M5 scalar 1D K=1 calls={CALLS}",
            scalar_loop_us(x, idx))

        x = torch.randn(4096, 3, generator=g)
        idx = torch.randint(0, 4096, (CALLS, 1), generator=g)
        add(rows, mode, t, f"M5 scalar 2D A=3 K=1 calls={CALLS}",
            scalar_loop_us(x, idx))

        # --- A1 single-element high-frequency gather (i64 fallback candidate) ---
        N = 200_000
        x = torch.randn(N, generator=g)
        for dt, name in ((torch.int64, "i64"), (torch.int32, "i32")):
            idx = torch.randint(0, N, (CALLS, 1), dtype=dt, generator=g)
            add(rows, mode, t, f"A1 {name} K=1 calls={CALLS}",
                scalar_loop_us(x, idx))

        # --- shortq: A in {2,3,4}, small K, i32/i64 ---
        N = 20_000
        for A in (2, 3, 4):
            x = torch.randn(N, A, generator=g)
            for K in (64, 256, 512, 1023):
                for dt, name in ((torch.int64, "i64"), (torch.int32, "i32")):
                    idx = torch.randint(0, N, (K,), dtype=dt, generator=g)
                    us = med(lambda x=x, idx=idx: x[idx]) * 1e6
                    add(rows, mode, t, f"shortq A={A} K={K} {name}", us)

        # --- M1 realistic weight blocks (<=16KB per row) ---
        E = 20_000
        for shape in ((27, 32, 32), (27, 256, 8), (27, 64, 64)):
            w = torch.randn(*shape, generator=g)
            for dt, name in ((torch.int64, "i64"), (torch.int32, "i32")):
                idx = torch.randint(0, shape[0], (E,), dtype=dt, generator=g)
                us = med(lambda w=w, idx=idx: w[idx], n=BIG_ITERS) * 1e6
                add(rows, mode, t,
                    f"M1 real {shape} E={E} {name}", us)

        # --- M2 representative spconv row gathers ---
        E = 100_000
        for C in (64, 8, 256):
            x = torch.randn(20_000, C, generator=g)
            idx = torch.randint(0, 20_000, (E,), dtype=torch.int32, generator=g)
            us = med(lambda x=x, idx=idx: x[idx], n=BIG_ITERS) * 1e6
            add(rows, mode, t, f"M2 (20000,{C}) E={E} i32", us)

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
