"""多线程扫描：核心场景 × 1/2/4/8/16/32/36/38 线程（interleaved median）。

替代 bench_gate.py 的顺序法：aten/hpc 交错 40 iters，取中位数（共享箱抗噪）。
场景覆盖模型形态（S1/P1/P2/P3/S4）+ 1D pure（f64/f32）+ T5 修复形态。

用法：
  HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
    python benchmarks/bench_threads.py
"""
import os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,2,4,8,16,32,36,38").split(",")]
ITERS = int(os.environ.get("HPC_ITERS", "40"))

def med_pair(fa, fh, n=ITERS):
    for _ in range(5):
        fa(); fh()
    tas, ths = [], []
    for _ in range(n):
        t0 = time.perf_counter(); fa(); tas.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fh(); ths.append(time.perf_counter() - t0)
    return statistics.median(tas), statistics.median(ths)

def gbps(nbytes, sec):
    return nbytes / sec / 1e9

def sweep(name, x, idx, aten_fn, bytes_):
    for t in THREADS:
        torch.set_num_threads(t)
        ta, th = med_pair(aten_fn, lambda: hpc_index(x, idx))
        print(f"| {name} | {x.dtype} | {t} | {gbps(bytes_, ta):.2f} | {gbps(bytes_, th):.2f} | {ta/th:.2f}x |",
              flush=True)

def main():
    affinity = os.environ["HPC_AFFINITY"]
    spec = "".join(affinity.split())
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-"); cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    os.sched_setaffinity(0, cpus)
    print(f"# bench_threads.py  affinity={affinity}  threads={THREADS}  iters={ITERS}")
    print()
    rng = torch.Generator().manual_seed(2024)

    N, C, E = 50_000, 64, 1_000_000
    features = torch.randn(N, C, generator=rng)
    inp_idx = torch.randint(0, N, (E,), dtype=torch.int32, generator=rng)
    sweep("S1 features[inp_idx] C=64 E=1M", features, [inp_idx], lambda: features[inp_idx], E * 4 + E * C * 4)

    N, C, E = 200_000, 64, 1_000_000
    h = torch.randn(N, C, generator=rng)
    src = torch.randint(0, N, (E,), dtype=torch.int64, generator=rng)
    sweep("P1 h[src] i64 C=64 E=1M", h, [src], lambda: h[src], E * 8 + E * C * 4)

    N, E = 200_000, 2_000_000
    pos = torch.randn(N, 3, generator=rng)
    src = torch.randint(0, N, (E,), dtype=torch.int32, generator=rng)
    sweep("P2 pos[src] A=3 E=2M", pos, [src], lambda: pos[src], E * 4 + E * 3 * 4)

    N, C = 200_000, 16
    x = torch.randn(N, C, generator=rng)
    m = torch.rand(N, generator=rng) < 0.3
    nnz = int(m.sum().item())
    sweep(f"P3 x[mask] nnz={nnz} C=16", x, [m], lambda: x[m], N * 1 + nnz * C * 4)

    N = 200_000
    x = torch.randn(N, dtype=torch.float64, generator=rng)
    perm = torch.randperm(N, generator=rng)
    sweep("S4 x[perm] f64 N=200k", x, [perm], lambda: x[perm], N * 16)

    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float64, generator=rng)
        i = torch.randint(0, K, (K,), generator=rng)
        sweep(f"1D f64 K={K}", x, [i], lambda: x[i], K * 16)

    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float32, generator=rng)
        i = torch.randint(0, K, (K,), dtype=torch.int32, generator=rng)
        sweep(f"1D f32 K={K}", x, [i], lambda: x[i], K * 8)

    # T5 修复形态（1 线程曾是 0.6-0.9x）
    x = torch.randn(5000, 5000, generator=rng)
    i = torch.randint(0, 5000, (100_000,), generator=rng)
    j = torch.randint(0, 5000, (100_000,), generator=rng)
    sweep("multi x[i,j] f32 100k", x, [i, j], lambda: x[i, j], 2 * 100_000 * 8 + 100_000 * 4)

    x = torch.randn(20000, 4, generator=rng)
    i = torch.randint(0, 20000, (200_000,), dtype=torch.int32, generator=rng)
    sweep("rowcopy A=4 f32 K=200k", x, [i], lambda: x[i], 200_000 * 4 + 200_000 * 4 * 4)

    x = torch.randn(5_000_000, generator=rng)
    i = torch.randint(0, 5_000_000, (2_000_000,), generator=rng)
    sweep("1D pure f32 K=2M (T5)", x, [i], lambda: x[i], 2 * 2_000_000 * 8 + 2_000_000 * 4)

if __name__ == "__main__":
    main()
