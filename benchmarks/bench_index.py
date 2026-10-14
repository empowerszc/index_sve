import os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

NT = os.environ.get("HPC_INDEX_NT", "0")

def med(f, warmup=3, iters=10):
    for _ in range(warmup):
        f()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

def gbps(nbytes, sec):
    return nbytes / sec / 1e9

def run_case(name, x, idx, aten_fn, bytes_):
    rows = []
    for threads in (1, 608):
        torch.set_num_threads(threads)
        ta = med(aten_fn)
        th = med(lambda: hpc_index(x, idx))
        rows.append((name, x.dtype, threads, gbps(bytes_, ta), gbps(bytes_, th), ta / th))
    return rows

def main():
    rows = []

    # 1D pure gather (B==1) -- the critical gate case
    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float64)
        i = torch.randint(0, K, (K,))
        rows += run_case(f"1D f64 K={K}", x, [i], lambda: x[i], K * 16)

    for K in (2_000_000, 10_000_000):
        x = torch.randn(K, dtype=torch.float32)
        i = torch.randint(0, K, (K,), dtype=torch.int32)
        rows += run_case(f"1D f32 K={K}", x, [i], lambda: x[i], K * 8)

    K = 2_000_000
    x = torch.randint(-1000, 1000, (K,), dtype=torch.int64)
    i = torch.randint(0, K, (K,))
    rows += run_case("1D i64 K=2M", x, [i], lambda: x[i], K * 16)

    x = torch.randint(-1000, 1000, (K,), dtype=torch.int32)
    i = torch.randint(0, K, (K,), dtype=torch.int32)
    rows += run_case("1D i32 K=2M", x, [i], lambda: x[i], K * 8)

    # x[:, idx] (S3): pure gather over B rows sharing idx
    B, N, KK = 256, 50_000, 100_000
    x = torch.randn(B, N)
    i = torch.randint(0, N, (KK,), dtype=torch.int32)
    rows += run_case(f"x[:,idx] f32 B={B} K={KK}", x, [None, i], lambda: x[:, i], KK * 4 + B * KK * 4)

    # x[i, j]
    M, I, J = 4096, 2048, 2048
    x = torch.randn(M, M)
    i = torch.randint(0, M, (I,), dtype=torch.int32)
    j = torch.randint(0, M, (J,), dtype=torch.int32)
    rows += run_case(f"x[i,j] f32 I={I} J={J}", x, [i, j], lambda: x[i, j], I * 4 + J * 4 + I * J * 4)

    # x[idx, :] (RowCopy A=C)
    N, C, E = 200_000, 32, 1_000_000
    x = torch.randn(N, C)
    i = torch.randint(0, N, (E,), dtype=torch.int32)
    rows += run_case(f"x[idx,:] f32 C={C} E={E}", x, [i], lambda: x[i], E * 4 + E * C * 4)

    # S1 spconv row gather
    N, C, E = 50_000, 64, 1_000_000
    features = torch.randn(N, C)
    inp_idx = torch.randint(0, N, (E,), dtype=torch.int32)
    rows += run_case(f"S1 features[inp_idx] C={C} E={E}", features, [inp_idx], lambda: features[inp_idx], E * 4 + E * C * 4)

    # P1 h[src] int64 idx
    N, C, E = 200_000, 64, 1_000_000
    h = torch.randn(N, C)
    src = torch.randint(0, N, (E,), dtype=torch.int64)
    rows += run_case(f"P1 h[src] i64 C={C} E={E}", h, [src], lambda: h[src], E * 8 + E * C * 4)

    # P2 pos[src] A=3 scalar path
    N, E = 200_000, 2_000_000
    pos = torch.randn(N, 3)
    src = torch.randint(0, N, (E,), dtype=torch.int32)
    rows += run_case(f"P2 pos[src] A=3 E={E}", pos, [src], lambda: pos[src], E * 4 + E * 3 * 4)

    # P3 mask select
    N, C = 200_000, 16
    x = torch.randn(N, C)
    m = torch.rand(N) < 0.3
    nnz = int(m.sum().item())
    rows += run_case(f"P3 x[mask] nnz={nnz} C={C}", x, [m], lambda: x[m], N * 1 + nnz * C * 4)

    # S4 random permutation big gather
    N = 200_000
    x = torch.randn(N, dtype=torch.float64)
    perm = torch.randperm(N)
    rows += run_case(f"S4 x[perm] f64 N={N}", x, [perm], lambda: x[perm], N * 16)

    print(f"# bench_index.py  HPC_INDEX_NT={NT}")
    print()
    print("| shape | dtype | threads | aten GB/s | hpc GB/s | speedup |")
    print("|---|---|---|---|---|---|")
    for name, dtype, threads, ga, gh, sp in rows:
        print(f"| {name} | {dtype} | {threads} | {ga:.1f} | {gh:.1f} | {sp:.2f}x |")

if __name__ == "__main__":
    main()
