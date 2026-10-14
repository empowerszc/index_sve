import os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

NT = os.environ.get("HPC_INDEX_NT", "0")
THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,38").split(",")]

def med(f, warmup=3, iters=20):
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def gbps(nbytes, sec):
    return nbytes / sec / 1e9

def gate(name, x, idx, aten_fn, bytes_, threads):
    torch.set_num_threads(threads)
    ta = med(aten_fn)
    th = med(lambda: hpc_index(x, idx))
    print(f"| {name} | {x.dtype} | {threads} | {gbps(bytes_, ta):.1f} | {gbps(bytes_, th):.1f} | {ta/th:.2f}x |")

def gates(name, x, idx, aten_fn, bytes_):
    for t in THREADS:
        gate(name, x, idx, aten_fn, bytes_, t)

def main():
    affinity = os.environ["HPC_AFFINITY"]
    spec = "".join(affinity.split())
    cpus = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-"); cpus.update(range(int(lo), int(hi)+1))
        else:
            cpus.add(int(part))
    os.sched_setaffinity(0, cpus)
    print(f"# bench_gate.py  HPC_INDEX_NT={NT}  affinity={affinity}  threads={THREADS}")
    print()

    N, C, E = 50_000, 64, 1_000_000
    features = torch.randn(N, C)
    inp_idx = torch.randint(0, N, (E,), dtype=torch.int32)
    gates("S1 features[inp_idx] C=64 E=1M", features, [inp_idx], lambda: features[inp_idx], E*4 + E*C*4)

    N, C, E = 200_000, 64, 1_000_000
    h = torch.randn(N, C)
    src = torch.randint(0, N, (E,), dtype=torch.int64)
    gates("P1 h[src] i64 C=64 E=1M", h, [src], lambda: h[src], E*8 + E*C*4)

    N, E = 200_000, 2_000_000
    pos = torch.randn(N, 3)
    src = torch.randint(0, N, (E,), dtype=torch.int32)
    gates("P2 pos[src] A=3 E=2M", pos, [src], lambda: pos[src], E*4 + E*3*4)

    N, C = 200_000, 16
    x = torch.randn(N, C)
    m = torch.rand(N) < 0.3
    nnz = int(m.sum().item())
    gates(f"P3 x[mask] nnz={nnz} C=16", x, [m], lambda: x[m], N*1 + nnz*C*4)

    N = 200_000
    x = torch.randn(N, dtype=torch.float64)
    perm = torch.randperm(N)
    gates("S4 x[perm] f64 N=200k", x, [perm], lambda: x[perm], N*16)

    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float64)
        i = torch.randint(0, K, (K,))
        gates(f"1D f64 K={K}", x, [i], lambda: x[i], K*16)

    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float32)
        i = torch.randint(0, K, (K,), dtype=torch.int32)
        gates(f"1D f32 K={K}", x, [i], lambda: x[i], K*8)

if __name__ == "__main__":
    main()
