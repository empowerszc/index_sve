"""生产形态基准：小 K + 生产线程数 + 真实 `x[idx]`（走 dispatcher）。

`bench_index.py` 测的是"理想大形态"（K=2M/10M、1/608 线程、直呼 `hpc_index()`），
它的倍率不能代表模型里的收益。本脚本按真实 trace 的形态分布构造用例：

- K 在生产区间 512-55296（大 K 只留 256000 一档作对照）；
- 线程 1/6/12/19，对齐生产 4-19 的并集，不用 608；
- 每次调用都是 `x[idx]`，包含 normalize / 分配 / dispatcher 的固定开销；
- bucket（K 小、值域 ≤2048）、small_single（A=1 无桶）、rowcopy（A=3/64）
  三类形态都在库里，不再只测 bucket 之外的大 K。

单进程只输出 JSON，aten/hpc 的交替对拍由 `bench_production_driver.py` 负责
（小 K 单次只有几微秒，两个进程交替跑再取中位数才稳）。

用法：
  # 测量（由 driver 调用）
  HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_production_shapes.py
  # 只打印用例库
  python benchmarks/bench_production_shapes.py --emit-cases
  # 从 trace CSV 提取热点形态（不需要 torch）
  python benchmarks/bench_production_shapes.py --gen-cases trace.csv > cases.json
  # 按调用次数加权合成模型级 index 占比（不需要 torch）
  python benchmarks/bench_production_shapes.py --weighted \
      benchmarks/prod_weights.example.json --times prod_times.json
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,6,12,19").split(",")]
AFFINITY = os.environ.get("HPC_AFFINITY", "190-227")
IDX_STRIDE = int(os.environ.get("HPC_IDX_STRIDED", "1"))

# ---------------------------------------------------------------------------
# 生产形态库
#
# "path" 是本仓库路由的形态标签，不是 trace 里的路径编号：
#   bucket       A==1 且值域 <=2048 -> gather_bucket_pure_raw（两趟 parallel_for）
#   small_single A==1 值域大        -> 1t 标量单遍 / 多线程 SVE 单遍
#   rowcopy      A>=2               -> 行/块拷贝
# 名字里的 K/A/dom 与 trace 一一对应，K 取值来自反馈里的热点（SGI/PPT/Sonata）。
# ---------------------------------------------------------------------------
K_SWEEP = [
    # 固定 A=1、dom=2048（bucket 生效），只扫 K：用来拟合 T = 固定开销 + 斜率*K
    dict(name=f"ksweep_K{K}_A1_dom2k", group="k_sweep", K=K, A=1, dom=2048)
    for K in (512, 1024, 2048, 4096, 8192, 16384, 24576)
]

PROD_CASES = [
    # bucket：生产最高频，K 小、值域小（embedding / vocab / 小词表）
    dict(name="bucket_K512_dom1k", group="prod", K=512, A=1, dom=1024),
    dict(name="bucket_K528_dom1k", group="prod", K=528, A=1, dom=1024),
    dict(name="bucket_K2048_dom2k", group="prod", K=2048, A=1, dom=2048),
    dict(name="bucket_K2112_dom2k", group="prod", K=2112, A=1, dom=2048),
    dict(name="bucket_K55296_dom2k", group="prod", K=55296, A=1, dom=2048),
    # 纯 gather：值域超出桶上限，走 SVE / 标量单遍
    dict(name="gather_K13824_dom1m", group="prod", K=13824, A=1, dom=1 << 20),
    dict(name="gather_K55296_dom1m", group="prod", K=55296, A=1, dom=1 << 20),
    # small_single：A 小、无桶（trace 里 A-br 低位的形态）
    dict(name="rowcopy_K512_A3", group="prod", K=512, A=3, dom=1024),
    dict(name="rowcopy_K2048_A3", group="prod", K=2048, A=3, dom=2048),
    # rowcopy：短行 / 整行（A=3 与 A=64）
    dict(name="rowcopy_K13824_A3", group="prod", K=13824, A=3, dom=1 << 20),
    dict(name="rowcopy_K8192_A64", group="prod", K=8192, A=64, dom=1 << 20),
    dict(name="rowcopy_K55296_A64", group="prod", K=55296, A=64, dom=1 << 20),
    dict(name="rowcopy_K256000_A64", group="prod", K=256000, A=64, dom=1 << 20),
]

CASES = PROD_CASES + K_SWEEP
for _c in CASES:
    # 路由标签按 op.cpp 的实际顺序：K<1024 先落 gather_direct（不建桶），
    # K>=1024 才可能走桶；A>1 走行/块拷贝。
    _c["path"] = ("direct" if _c["A"] == 1 and _c["K"] < 1024
                  else "bucket" if _c["A"] == 1 and _c["dom"] <= 2048
                  else "single" if _c["A"] == 1 else "rowcopy")

# HPC_SUBSET=bucket_K512,gather_K 只保留名字含任一子串的用例（细扫线程/单点复测用）
_SUBSET = [x for x in os.environ.get("HPC_SUBSET", "").split(",") if x]
if _SUBSET:
    CASES = [c for c in CASES if any(x in c["name"] for x in _SUBSET)]

# 诊断模式：固定 K 只扫值域，用来看桶化路径的"建表/查表"占多少固定成本
# （dom<=2048 走桶，>2048 回退 SVE/标量单遍，两侧就是同一形态的两种内核）。
DOM_SWEEP_K = int(os.environ.get("HPC_DOM_SWEEP", "0"))
if DOM_SWEEP_K:
    CASES = [dict(name=f"dom{d}", group="dom", K=DOM_SWEEP_K, A=1, dom=d,
                  path="bucket" if d <= 2048 else "single")
             for d in (8, 64, 512, 2048, 4096, 1 << 20)]

# trace 里的 A-br 是分桶码，不是真实 A。反馈里 path11/path2 的 A-br=3 对应真实
# A∈[16,128)（典型 64）；这里显式列出来，映射不到就直接报错而不是瞎猜。
ABR_HINTS = {1: 1, 2: 4, 3: 64, 4: 64, 5: 64}
PATH_ALIASES = {
    "path3": "bucket", "bucket": "bucket",
    "path11": "small_single", "small_single": "small_single",
    "path2": "rowcopy", "rowcopy": "rowcopy",
}


def _affinity():
    cpus = set()
    for part in AFFINITY.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _set_threads(torch, omp, t):
    torch.set_num_threads(t)
    if omp is not None:  # 扩展用 BiSheng libomp，torch.set_num_threads 只改 ATen
        omp.omp_set_num_threads(t)


def _omp_handle():
    import ctypes
    for line in open("/proc/self/maps"):
        if "libomp.so" in line:
            lib = ctypes.CDLL(line.split()[-1])
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            return lib
    return None


def _median_us(fn, target_s=0.15):
    """先标定单次耗时，再按目标总时长定迭代数：大 K 不跑几百次，微秒级也够样本。"""
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    per = (time.perf_counter() - t0) / 5
    iters = int(max(15, min(300, target_s / max(per, 1e-7))))
    for _ in range(min(iters, 30)):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6, iters


def measure(cases, threads):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import index_sve  # noqa: F401  (是否安装覆盖由 HPC_INDEX_NO_OVERRIDE 决定)

    os.sched_setaffinity(0, _affinity())
    omp = _omp_handle()
    mode = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
    g = torch.Generator().manual_seed(7)

    # 源按最大 A 分配一次；A=1 用 4MB（L3 内），A>1 用 A*4MB
    x1 = torch.randn(1 << 20, generator=g)
    x_multi = {A: torch.randn(1 << 20, A, generator=g).contiguous()
               for A in sorted({c["A"] for c in cases if c["A"] > 1})}

    rows = []
    for t in threads:
        _set_threads(torch, omp, t)
        for c in cases:
            K, A, dom = c["K"], c["A"], c["dom"]
            src = x1 if A == 1 else x_multi[A]
            # HPC_IDX_STRIDED=2：把下标换成 stride=2 的非连续视图（生产里很常见，
            # 如列切片）；=0：换成 stride=0 的广播视图（`idx.expand(K)`）。
            # 两者都会离开 fast_1d 快路径，见 degradation-analysis.md §6.5.7。
            step = IDX_STRIDE
            if step == 0:
                idx = torch.randint(0, dom, (1,), generator=g).expand(K)
            elif step > 1:
                base = torch.randint(0, dom, (K * step,), generator=g)
                idx = base[::step]
            else:
                idx = torch.randint(0, dom, (K,), generator=g)
            us, iters = _median_us(lambda: src[idx])
            row = dict(c, mode=mode, threads=t, us=round(us, 2), iters=iters,
                       idx_stride=step)
            rows.append(row)
            print(f"{mode:5s} {c['name']:26s} t={t:2d} {us:8.2f} us",
                  file=sys.stderr, flush=True)
    return rows


def gen_cases(path):
    """从 trace CSV 提取热点形态：列 = path,K,A,dom,calls[,threads]。"""
    agg = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path_name = PATH_ALIASES.get(r["path"].strip())
            if path_name is None:
                continue
            K, abr = int(r["K"]), int(r["A"])
            dom = int(r["dom"]) if r.get("dom") not in (None, "", "0") else 2048
            A = 1 if path_name == "bucket" else ABR_HINTS.get(abr)
            if A is None:
                raise SystemExit(f"A-br={abr} 没有映射规则，先加到 ABR_HINTS 再跑")
            calls = int(r["calls"])
            name = f"gen_{path_name}_K{K}_A{A}_dom{dom}"
            cur = agg.setdefault(name, dict(name=name, group="gen", K=K, A=A,
                                            dom=dom, path=path_name, calls=0))
            cur["calls"] += calls
    return sorted(agg.values(), key=lambda c: -c["calls"])


def weighted(weights_path, times_path):
    """按 trace 调用次数加权：ratio = Σ(w*aten) / Σ(w*hpc)，>1 表示本算子更快。"""
    spec = json.load(open(weights_path, encoding="utf-8"))
    times = json.load(open(times_path, encoding="utf-8"))
    table = {}
    for r in times:
        table[(r["name"], r["mode"], r["threads"])] = r["us"]

    print(f"# 权重来源：{spec.get('note', weights_path)}")
    print("| 模型 | TH | 覆盖调用 | 未覆盖 | Σaten µs | Σhpc µs | aten/hpc |")
    print("|---|---|---|---|---|---|---|")
    for m in spec["models"]:
        th = m["threads"]
        covered = miss = 0
        sa = sh = 0.0
        for name, w in m["calls"].items():
            a = table.get((name, "aten", th))
            h = table.get((name, "hpc", th))
            if a is None or h is None:
                miss += w
                continue
            covered += w
            sa += w * a
            sh += w * h
        total = m.get("total_calls", covered + miss)
        ratio = f"{sa / sh:.3f}x" if sh else "n/a"
        print(f"| {m['name']} | {th} | {covered}/{total} "
              f"({100 * covered / total:.0f}%) | {miss} | {sa:.0f} | {sh:.0f} | {ratio} |")
    print("\n> 只覆盖库内形态；未覆盖长尾按 0 计入，方向看覆盖部分的比值。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-cases", action="store_true", help="打印用例库 JSON")
    ap.add_argument("--gen-cases", metavar="CSV", help="从 trace CSV 提取用例")
    ap.add_argument("--weighted", metavar="WEIGHTS", help="按调用次数加权")
    ap.add_argument("--times", metavar="JSON", help="--weighted 用的测量结果")
    args = ap.parse_args()

    if args.gen_cases:
        print(json.dumps(gen_cases(args.gen_cases), ensure_ascii=False, indent=1))
        return
    if args.emit_cases:
        print(json.dumps(CASES, ensure_ascii=False, indent=1))
        return
    if args.weighted:
        if not args.times:
            raise SystemExit("--weighted 需要 --times（driver 的 --out 结果）")
        weighted(args.weighted, args.times)
        return

    rows = measure(CASES, THREADS)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
