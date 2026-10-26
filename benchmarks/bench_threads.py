"""多线程扫描：核心场景 × 1/2/4/8/16/32/36/38 线程（interleaved median）。

替代 bench_gate.py 的顺序法：aten/hpc 交错 iters，取中位数（共享箱抗噪）。
覆盖模型形态（S1/P1/P2/P3/S4）、1D pure 4B/8B、mask（1D/2D/空）、小值域桶化、
rowcopy A=2/3/4、多索引、列索引、非连续、2B/1B/complex（M4）、以及 T8/T9 形态。
本脚本 hpc 侧为 hpc_index() pybind 直呼（绕过 dispatcher），适合大 K / 全线程
正式矩阵；短调用 / 真实 `x[idx]` 路径结论以 bench_realpath_driver.py 为准，
小 K（<16384 元素）另见 bench_smallk_real.py 两进程法。

用法：
  HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
    python benchmarks/bench_threads.py
"""
import ctypes, os, sys, time, torch, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from index_sve import index as hpc_index

# 扩展用 BiSheng clang 编译，ATen 的 invoke_parallel 是 header inline，实际团队
# 大小由扩展链接的 libomp 决定；torch.set_num_threads 只改 libgomp 的 ICV，
# 对扩展里的 #pragma omp parallel 无效（实测 2t 仍用 16 个 worker）。必须同时
# 调 libomp 的 omp_set_num_threads，线程标签才可信。
_LIBOMP = None
for _line in open("/proc/self/maps"):
    if "libomp.so" in _line and ".so" in _line:
        _path = _line.split()[-1]
        _LIBOMP = ctypes.CDLL(_path)
        break
if _LIBOMP is not None:
    _LIBOMP.omp_set_num_threads.argtypes = [ctypes.c_int]
    _LIBOMP.omp_get_max_threads.restype = ctypes.c_int


def set_threads(t):
    torch.set_num_threads(t)
    if _LIBOMP is not None:
        _LIBOMP.omp_set_num_threads(t)

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
        set_threads(t)
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

    # --- 4B/8B 下标组合（同 out dtype，验证 linearize/gather 分支） ---
    x = torch.randn(10_000_000, generator=rng)
    i = torch.randint(0, 10_000_000, (10_000_000,), generator=rng)
    sweep("1D f32 K=10M idx64", x, [i], lambda: x[i], 10_000_000 * (8 + 4))
    x = torch.randn(10_000_000, dtype=torch.float64, generator=rng)
    i = torch.randint(0, 10_000_000, (10_000_000,), dtype=torch.int32, generator=rng)
    sweep("1D f64 K=10M idx32", x, [i], lambda: x[i], 10_000_000 * (4 + 8))

    # --- 2B/1B/complex：无 SVE 内核，dispatch_scalar（M4） ---
    for dt, nbytes in ((torch.float16, 2), (torch.bfloat16, 2)):
        x = torch.randn(2_000_000, generator=rng).to(dt)
        i = torch.randint(0, 2_000_000, (2_000_000,), generator=rng)
        sweep(f"1D {str(dt).replace('torch.','')} K=2M", x, [i],
              lambda: x[i], 2_000_000 * (8 + nbytes))
    # 1B 整数用 randint 直接生成全值域随机字节；randn().to() 向零截断后只有
    # 少数近零值（0 ≈70%，±1 ≈27%），数据不具代表性。
    for dt, name, lo, hi in ((torch.int8, "int8", -128, 128),
                             (torch.uint8, "uint8", 0, 256)):
        x = torch.randint(lo, hi, (2_000_000,), dtype=dt, generator=rng)
        i = torch.randint(0, 2_000_000, (2_000_000,), generator=rng)
        sweep(f"1D {name} K=2M", x, [i], lambda: x[i], 2_000_000 * (8 + 1))
    x = torch.randn(2_000_000, dtype=torch.complex64, generator=rng)
    i = torch.randint(0, 2_000_000, (2_000_000,), generator=rng)
    sweep("1D complex64 K=2M", x, [i], lambda: x[i], 2_000_000 * (8 + 8))
    for dt, name in ((torch.int32, "int32"), (torch.int64, "int64")):
        x = torch.randint(0, 1000, (2_000_000,), dtype=dt, generator=rng)
        i = torch.randint(0, 2_000_000, (2_000_000,), generator=rng)
        sweep(f"1D {name} K=2M", x, [i], lambda: x[i],
              2_000_000 * (8 + x.element_size()))

    # --- rowcopy 短行 A=2/3（M3）与大行 A=16 ---
    for A in (2, 3, 16):
        x = torch.randn(200_000, A, generator=rng)
        i = torch.randint(0, 200_000, (200_000,), dtype=torch.int32, generator=rng)
        sweep(f"rowcopy A={A} f32 K=200k", x, [i], lambda: x[i],
              200_000 * (4 + A * 4))

    # --- 非连续 self / idx（回退到 fused/scalar） ---
    base = torch.randn(200_000, 32, generator=rng)
    x = base[:, ::2]
    i = torch.randint(0, 200_000, (200_000,), dtype=torch.int32, generator=rng)
    sweep("noncontig self A=16 stride2", x, [i], lambda: x[i],
          200_000 * (4 + 16 * 4))
    x = torch.randn(2_000_000, generator=rng)
    i = torch.randint(0, 2_000_000, (2_000_000,), generator=rng)[::2]
    sweep("noncontig idx K=1M", x, [i], lambda: x[i], 1_000_000 * (8 + 4))

    # --- mask：1D / 空 mask / 非连续 mask ---
    x = torch.randn(4_000_000, generator=rng)
    m = torch.rand(4_000_000, generator=rng) < 0.3
    sweep(f"mask 1D nnz={int(m.sum())}", x, [m], lambda: x[m],
          4_000_000 + int(m.sum()) * 4)
    m = torch.zeros(1_000_000, dtype=torch.bool)
    x = torch.randn(1_000_000, generator=rng)
    sweep("mask empty N=1M", x, [m], lambda: x[m], 1_000_000)
    x = torch.randn(2_000_000, generator=rng)
    m = (torch.rand(4_000_000, generator=rng) < 0.3)[::2]
    sweep(f"mask noncontig nnz={int(m.sum())}", x, [m], lambda: x[m],
          2_000_000 + int(m.sum()) * 4)

    # --- 列索引 x[:, i]（B>1，A=0 形态） ---
    x = torch.randn(200_000, 64, generator=rng)
    i = torch.randint(0, 64, (64,), generator=rng)
    sweep("col x[:,i] C=64 K=64", x, [None, i], lambda: x[:, i],
          200_000 * 64 * 4 * 2)

    # --- 小值域桶化（T4）：值域 2048 内 ---
    x = torch.randn(2_000_000, generator=rng)
    i = torch.randint(0, 2048, (2_000_000,), generator=rng)
    sweep("1D f32 K=2M range2048", x, [i], lambda: x[i], 2_000_000 * (8 + 4))

if __name__ == "__main__":
    main()
