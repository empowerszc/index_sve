"""Degradation probes: absolute bandwidth, small-workload scaling, mask breakdown.

Run on an idle NUMA node (default node 5), from a directory that does not shadow
the local torch sources:

    HPC_AFFINITY=190-227 python benchmarks/degrade_probe.py

The numbers printed here back the tables in docs/degradation-analysis.md
(sections 6.3/6.4 and 5.5).
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HPC_INDEX_NO_OVERRIDE", "1")

import torch  # noqa: E402
import index_sve  # noqa: E402
import index_sve._hpc_index as ext  # noqa: E402


def bind():
    spec = os.environ.get("HPC_AFFINITY", "190-227")
    lo, _, hi = spec.partition("-")
    cpus = list(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, cpus)


def median_ms(fn, iters=15, warmup=3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def absolute():
    print("## absolute (hpc vs aten)")
    print("| case | threads | hpc ms | aten ms | ratio | useful GB/s | line GB/s |")
    print("|---|---|---|---|---|---|---|")
    cases = [
        ("1d-f32-2M", torch.float32, 2_000_000),
        ("1d-f32-10M", torch.float32, 10_000_000),
        ("1d-f64-2M", torch.float64, 2_000_000),
        ("1d-f64-10M", torch.float64, 10_000_000),
    ]
    for name, dtype, k in cases:
        x = torch.randn(k, dtype=dtype)
        idx = torch.randperm(k)
        for threads in (1, 8, 36):
            torch.set_num_threads(threads)
            hpc = median_ms(lambda: index_sve.index(x, [idx]))
            aten = median_ms(lambda: x[idx])
            useful = k * x.element_size() / 1e9 / (hpc / 1e3)
            lines = k * 64 / 1e9 / (hpc / 1e3)
            print(f"| {name} | {threads} | {hpc:.3f} | {aten:.3f} | {aten/hpc:.2f}x | "
                  f"{useful:.2f} | {lines:.2f} |")

    for a, e in ((16, 200_000), (64, 1_000_000)):
        x = torch.randn(e, a)
        idx = torch.randint(0, e, (e,), dtype=torch.int64)
        for threads in (1, 8, 36):
            torch.set_num_threads(threads)
            hpc = median_ms(lambda: index_sve.index(x, [idx]))
            aten = median_ms(lambda: x[idx])
            useful = e * a * 4 / 1e9 / (hpc / 1e3)
            print(f"| rowcopy A={a} K={e} | {threads} | {hpc:.3f} | {aten:.3f} | "
                  f"{aten/hpc:.2f}x | {useful:.2f} | {useful:.2f} |")


def threshold():
    print("\n## small-workload scaling (interleaved 1t vs t, median ratio)")
    print("| case | total | 4t | 8t | 16t | 36t |")
    print("|---|---|---|---|---|---|")
    cases = [
        ("1D f32 K=8192", lambda: (torch.randn(8192), torch.randperm(8192)), 8192 * 4),
        ("1D f32 K=65536", lambda: (torch.randn(65536), torch.randperm(65536)), 65536 * 4),
        ("2D A=16 K=8192", lambda: (torch.randn(8192, 16),
                                   torch.randint(0, 8192, (8192,), dtype=torch.int64)), 8192 * 64),
        ("2D A=16 K=65536", lambda: (torch.randn(65536, 16),
                                    torch.randint(0, 65536, (65536,), dtype=torch.int64)), 65536 * 64),
    ]
    for label, make, out_bytes in cases:
        x, idx = make()
        cells = []
        for threads in (4, 8, 16, 36):
            ratios = []
            for _ in range(7):
                torch.set_num_threads(1)
                t1 = median_ms(lambda: index_sve.index(x, [idx]), iters=31)
                torch.set_num_threads(threads)
                tn = median_ms(lambda: index_sve.index(x, [idx]), iters=31)
                ratios.append(t1 / tn)
            r = statistics.median(ratios)
            cells.append(f"{r:.2f}x ({100*r/threads:.0f}%)")
        print(f"| {label} | {out_bytes//1024}KB | " + " | ".join(cells) + " |")


def masks():
    print("\n## non-contiguous mask breakdown (1D self 2M, nnz~600k)")
    x = torch.randn(2_000_000)
    for label, mask in (("contig", torch.rand(2_000_000) < 0.3),
                        ("stride2", (torch.rand(4_000_000) < 0.3)[::2])):
        spec = ext.debug_spec(x, [mask])
        print(f"| {label} nnz={int(mask.sum())} stride={mask.stride()} B={spec[4]} K={spec[5]} A={spec[6]} |")
        for threads in (1, 8, 16, 36):
            torch.set_num_threads(threads)
            hpc = median_ms(lambda: index_sve.index(x, [mask]), iters=9)
            aten = median_ms(lambda: x[mask], iters=9)
            nz = median_ms(lambda: mask.nonzero(), iters=9)
            lin = median_ms(lambda: ext.linearize_spec(x, [mask]), iters=9)
            print(f"| t={threads} | hpc {hpc:.3f} | aten {aten:.3f} | {aten/hpc:.2f}x | "
                  f"nonzero {nz:.3f} | norm+lin {lin:.3f} | gather {hpc-lin:.3f} |")


if __name__ == "__main__":
    bind()
    absolute()
    threshold()
    masks()
