import os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,2,4,8,16,32,36").split(",")]

def med(f, warmup=3, iters=20):
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def gbps(nbytes, sec):
    return nbytes / sec / 1e9

def main():
    print("# bench_t1.py  A-sweep short-row (M2)  thread sweep")
    print("| A | threads | aten GB/s | hpc GB/s | speedup |")
    print("|---|---|---|---|---|")
    N, E = 200_000, 100_000
    for A in (2, 3, 4, 5, 6, 8, 12, 16, 32, 64):
        x = torch.randn(N, A)
        idx = torch.randint(0, N, (E,), dtype=torch.int32)
        for t in THREADS:
            torch.set_num_threads(t)
            ta = med(lambda: x[idx])
            th = med(lambda: hpc_index(x, [idx]))
            nbytes = E * 4 + E * A * 4
            print(f"| {A} | {t} | {gbps(nbytes, ta):.1f} | {gbps(nbytes, th):.1f} | {ta/th:.2f}x |")

if __name__ == "__main__":
    main()
