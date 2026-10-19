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

def run_case(name, x, idx, nbytes, thread):
    torch.set_num_threads(thread)
    ta = med(lambda: x[idx])
    th = med(lambda: hpc_index(x, [idx]))
    print(f"| {name} | {x.dtype} | {thread} | {gbps(nbytes, ta):.1f} | {gbps(nbytes, th):.1f} | {ta/th:.2f}x |", flush=True)

def main():
    print("# bench_ai4s.py  workload-shaped cases (ai4s doc §3.1)  thread sweep")
    print("| case | dtype | threads | aten GB/s | hpc GB/s | speedup |")
    print("|---|---|---|---|---|---|")

    # M1 weight.view(K,C,o)[ki] — hottest
    for name, shape, E, itype in [
        ("m1_small (27,32,32) E=2k", (27, 32, 32), 2000, "i64"),
        ("m1_mid   (27,64,64) E=20k", (27, 64, 64), 20000, "i64"),
        ("m1_large (27,256,256) E=20k", (27, 256, 256), 20000, "i64"),
        ("m1_mid_i32 (27,64,64) E=20k", (27, 64, 64), 20000, "i32"),
    ]:
        w = torch.randn(*shape)
        idx = (torch.randint(0, shape[0], (E,), dtype=torch.int32)
               if itype == "i32" else torch.randint(0, shape[0], (E,)))
        nbytes = E * idx.element_size() + E * shape[1] * shape[2] * w.element_size()
        for t in THREADS:
            run_case(f"{name} {itype}", w, idx, nbytes, t)

    # M2 feat[r] — spconv
    for name, shape, E, itype in [
        ("m2_spconv (20000,64) E=100k", (20000, 64), 100000, "i64"),
        ("m2_smallC (20000,8) E=100k", (20000, 8), 100000, "i64"),
        ("m2_largeC (20000,256) E=100k", (20000, 256), 100000, "i64"),
        ("m2_int32 (20000,64) E=100k", (20000, 64), 100000, "i32"),
    ]:
        x = torch.randn(*shape)
        idx = (torch.randint(0, shape[0], (E,), dtype=torch.int32)
               if itype == "i32" else torch.randint(0, shape[0], (E,)))
        nbytes = E * idx.element_size() + E * shape[1] * x.element_size()
        for t in THREADS:
            run_case(f"{name} {itype}", x, idx, nbytes, t)

    # M3 keys[order] pure gather, small domain
    for name, N, E, itype in [
        ("m3_lookup int64[2048] E=54k", 2048, 54000, "i64"),
        ("m3_int32 int64[2048] E=54k", 2048, 54000, "i32"),
    ]:
        x = torch.randint(0, 1000, (N,), dtype=torch.int64)
        idx = (torch.randint(0, N, (E,), dtype=torch.int32)
               if itype == "i32" else torch.randint(0, N, (E,)))
        nbytes = E * idx.element_size() + E * 8
        for t in THREADS:
            run_case(f"{name} {itype}", x, idx, nbytes, t)

    # M4 feat[inverse] high-duplicate
    x = torch.randn(200, 64)
    idx = torch.randint(0, 200, (20000,))
    nbytes = 20000 * 8 + 20000 * 64 * 4
    for t in THREADS:
        run_case("m4_inverse (200,64) E=20k", x, idx, nbytes, t)

    # M6 2D index (pf[vi])
    x = torch.randn(20000, 64)
    idx = torch.randint(0, 20000, (2048, 3), dtype=torch.int64)
    nbytes = 2048 * 3 * 8 + 2048 * 3 * 64 * 4
    for t in THREADS:
        run_case("m6_2d_idx (2048,3)->flat", x, idx, nbytes, t)

if __name__ == "__main__":
    main()
