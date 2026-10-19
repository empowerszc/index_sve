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

def main():
    print("# bench_t2.py  M1 large-block  thread sweep")
    print("| shape | idx | threads | aten ms | hpc ms | speedup |")
    print("|---|---|---|---|---|---|")
    cases = [
        ("m1_small (27,32,32) A=1024", (27, 32, 32), 2000),
        ("m1_mid   (27,64,64) A=4096", (27, 64, 64), 20000),
        ("m1_large (27,256,256) A=65536", (27, 256, 256), 20000),
    ]
    for name, shape, E in cases:
        w = torch.randn(*shape)
        row_elems = shape[1] * shape[2]
        for itype in ("i64", "i32"):
            if itype == "i64":
                idx = torch.randint(0, shape[0], (E,))
            else:
                idx = torch.randint(0, shape[0], (E,), dtype=torch.int32)
            for t in THREADS:
                torch.set_num_threads(t)
                ta = med(lambda: w[idx])
                th = med(lambda: hpc_index(w, [idx]))
                print(f"| {name} | {itype} E={E} | {t} | {ta*1e3:.3f} | {th*1e3:.3f} | {ta/th:.2f}x |", flush=True)

if __name__ == "__main__":
    main()
