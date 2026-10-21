import os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,8,32").split(",")]

def med_pair(fa, fh, warmup=5, iters=40):
    for _ in range(warmup):
        fa(); fh()
    tas, ths = [], []
    for _ in range(iters):
        t0 = time.perf_counter(); fa(); tas.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fh(); ths.append(time.perf_counter() - t0)
    return statistics.median(tas), statistics.median(ths)

def run(name, x, idx, bytes_, threads):
    torch.set_num_threads(threads)
    ta, th = med_pair(lambda: x[idx], lambda: hpc_index(x, [idx]))
    flag = "REGRESS" if ta / th < 0.90 else ""
    print(f"| {name} | {x.dtype} | {threads} | {bytes_/ta/1e9:.1f} | {bytes_/th/1e9:.1f} | {ta/th:.2f}x | {flag}", flush=True)

def sweep(name, x, idx, bytes_):
    for t in THREADS:
        run(name, x, idx, bytes_, t)

def main():
    if os.environ.get("HPC_AFFINITY"):
        affinity = os.environ["HPC_AFFINITY"]
        spec = "".join(affinity.split())
        cpus = set()
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-"); cpus.update(range(int(lo), int(hi)+1))
            else:
                cpus.add(int(part))
        os.sched_setaffinity(0, cpus)
    print(f"# regression scan threads={THREADS}")
    print("| case | dtype | threads | aten GB/s | hpc GB/s | speedup | flag |")
    print("|---|---|---|---|---|---|---|")

    # A: 1D pure gather 大 K（SVE 4B/8B 路径）
    for dt, dtname, esz in ((torch.float32, "f32", 4), (torch.float64, "f64", 8), (torch.int64, "i64", 8), (torch.int32, "i32", 4)):
        K = 5_000_000
        x = torch.randn(K).to(dt)
        i = torch.randint(0, K, (K,))
        sweep(f"1D pure K=5M {dtname} i64idx", x, i, K*8 + K*esz)

    # B: 1D pure 大 K 小值域（T4 raw 桶化）
    for dt, dtname in ((torch.float32, "f32"), (torch.float64, "f64")):
        x = torch.randn(2048).to(dt)
        i = torch.randint(0, 2048, (2_000_000,), dtype=torch.int32)
        sweep(f"1D small-domain K=2M {dtname}", x, i, 2_000_000*4 + 2_000_000*4)

    # C: rowcopy 短/中/长行
    for A in (4, 8, 32, 256, 4096):
        x = torch.randn(20000, A)
        i = torch.randint(0, 20000, (200_000,), dtype=torch.int32)
        sweep(f"rowcopy A={A} f32", x, i, 200_000*4 + 200_000*A*4)

    # D: 2D pure (x[:, idx]) B>1 A==1
    x = torch.randn(64, 2048)
    i = torch.randint(0, 2048, (100_000,), dtype=torch.int32)
    torch.set_num_threads(THREADS[0])
    ta, th = med_pair(lambda: x[:, i], lambda: hpc_index(x, [None, i]))
    print(f"| 2D pure x[:,i] B=64 A==1 | f32 | {THREADS[0]} | {100_000*4+100_000*4/ta/1e9:.1f} | {100_000*4+100_000*4/th/1e9:.1f} | {ta/th:.2f}x |", flush=True)

    # E: scalar 小块 A=2/3（T3 fused）
    for A in (2, 3):
        x = torch.randn(200000, A)
        i = torch.randint(0, 200000, (2_000_000,), dtype=torch.int64)
        sweep(f"scalar A={A} E=2M", x, i, 2_000_000*8 + 2_000_000*A*4)

    # F: mask
    x = torch.randn(200000, 32)
    m = torch.rand(200000) < 0.3
    nnz = int(m.sum().item())
    sweep(f"mask nnz={nnz} C=32", x, m, 200000*1 + nnz*32*4)

    # G: 2D index x[i,j]（多下标 broadcast）
    x = torch.randn(5000, 5000)
    i = torch.randint(0, 5000, (100_000,))
    j = torch.randint(0, 5000, (100_000,))
    torch.set_num_threads(THREADS[0])
    ta, th = med_pair(lambda: x[i, j], lambda: hpc_index(x, [i, j]))
    print(f"| x[i,j] 2D multidx | f32 | {THREADS[0]} | {100_000*16/ta/1e9:.1f} | {100_000*16/th/1e9:.1f} | {ta/th:.2f}x |", flush=True)

    # H: N-D 下标
    x = torch.randn(64, 64, 64)
    i = torch.randint(0, 64, (20000, 2))
    torch.set_num_threads(THREADS[0])
    ta, th = med_pair(lambda: x[i], lambda: hpc_index(x, [i]))
    print(f"| N-D 2x20000 idx | f32 | {THREADS[0]} | {20000*2*8/ta/1e9:.1f} | {20000*2*8/th/1e9:.1f} | {ta/th:.2f}x |", flush=True)

    # I: 非连续 after（transposed）
    x = torch.randn(256, 20000).t()
    i = torch.randint(0, 20000, (100_000,), dtype=torch.int32)
    sweep("noncontig-after x.t()", x, i, 100_000*4 + 100_000*256*4)

    # J: 多下标 3D x[i, :, j]
    x = torch.randn(100, 50, 40)
    i = torch.randint(0, 100, (50_000,), dtype=torch.int32)
    j = torch.randint(0, 40, (50_000,), dtype=torch.int32)
    torch.set_num_threads(THREADS[0])
    ta, th = med_pair(lambda: x[i, :, j], lambda: hpc_index(x, [i, None, j]))
    print(f"| x[i,:,j] 2-idx+None | f32 | {THREADS[0]} | {50_000*8+50_000*50*4/ta/1e9:.1f} | {50_000*8+50_000*50*4/th/1e9:.1f} | {ta/th:.2f}x |", flush=True)

    # K: M1 实际 workload（27,64,64）i64/i32
    for itype in ("i64", "i32"):
        x = torch.randn(27, 64, 64)
        i = torch.randint(0, 27, (20000,), dtype=torch.int64 if itype == "i64" else torch.int32)
        sweep(f"M1 (27,64,64) E=20k {itype}", x, i, 20000*8 + 20000*4096*4)

    # L: M2 feat[r] (20000,64)
    x = torch.randn(20000, 64)
    i = torch.randint(0, 20000, (100_000,), dtype=torch.int32)
    sweep("M2 (20000,64) E=100k", x, i, 100_000*4 + 100_000*64*4)

    # M: M3 small-domain int64 (keys[order])
    x = torch.randint(0, 1000, (2048,), dtype=torch.int64)
    i = torch.randint(0, 2048, (54000,))
    sweep("M3 keys[order] int64[2048] E=54k", x, i, 54000*8 + 54000*8)

if __name__ == "__main__":
    main()
