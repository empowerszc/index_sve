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
  # 从 trace CSV 提取热点形态（不需要 torch），再按生成的用例测一遍
  python benchmarks/bench_production_shapes.py --gen-cases trace.csv > cases.json
  HPC_CASES=cases.json python benchmarks/bench_production_driver.py --out t.json
  # 按调用次数加权合成（不需要 torch；cases.json 直接当权重文件）
  # 默认窄版（一模型一块，<=70 列）；--md 出文档表；--detail 加按耗时的排名
  python benchmarks/bench_production_shapes.py --weighted cases.json --times t.json --detail
  python benchmarks/bench_production_shapes.py --weighted cases.json --times t.json --md

两条口径一起打印，避免把方向读反：
  aten/hpc       >1 = 本算子更快（本仓库文档的默认口径）
  index_sve/原生  <1 = 本算子更快（反馈 §11.3.3 表格用的口径，互为倒数）

"全程①/③" 是给未覆盖长尾的两种补全假设（持平 / 本算子慢 2x），括号外的
覆盖内比值只说明被覆盖的形态；①和③夹住 1.0 就说明覆盖率不足以下方向结论。
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

# 反馈 §11.2.5 给出的 SGI 热点形态（按调用次数）。A 由 A-br 桶中值推测（见 ABR_HINTS），
# 不是 trace 真值；rows 只影响源张量大小，必须 >= dom。
SGI_CASES = [
    dict(name="sgi_bucket_K512_dom1k", group="sgi", K=512, A=1, dom=1024, calls=2322),
    dict(name="sgi_bucket_K13824_dom2k", group="sgi", K=13824, A=1, dom=2048, calls=1620),
    dict(name="sgi_bucket_K528_dom1k", group="sgi", K=528, A=1, dom=1024, calls=1064),
    dict(name="sgi_rowcopy_K512_A64", group="sgi", K=512, A=64, dom=1024,
         rows=8192, calls=1748),
    dict(name="sgi_rowcopy_K528_A256", group="sgi", K=528, A=256, dom=1024,
         rows=8192, calls=896),
    dict(name="sgi_rowcopy_K2_A256", group="sgi", K=2, A=256, dom=1024,
         rows=8192, calls=768),
]

CASES = PROD_CASES + K_SWEEP + SGI_CASES
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

# HPC_CASES=cases.json 用 --gen-cases 的产物直接测（告别人工抄形态）
_CASES_FILE = os.environ.get("HPC_CASES")
if _CASES_FILE:
    CASES = json.load(open(_CASES_FILE, encoding="utf-8"))
    for _c in CASES:
        _c.setdefault("group", "gen")

# 诊断模式：固定 K 只扫值域，用来看桶化路径的"建表/查表"占多少固定成本
# （dom<=2048 走桶，>2048 回退 SVE/标量单遍，两侧就是同一形态的两种内核）。
DOM_SWEEP_K = int(os.environ.get("HPC_DOM_SWEEP", "0"))
if DOM_SWEEP_K:
    CASES = [dict(name=f"dom{d}", group="dom", K=DOM_SWEEP_K, A=1, dom=d,
                  path="bucket" if d <= 2048 else "single")
             for d in (8, 64, 512, 2048, 4096, 1 << 20)]

# trace 里的 A-br 是分桶码，不是真实 A。只有 A-br=3 被反馈锚定（path11 的
# A∈[16,128)，典型 64）；4/5 按几何中点猜 256/2048，可用 --abr-map 覆盖。
# 映射不到就报错，不静默猜。
ABR_HINTS = {1: 1, 2: 4, 3: 64, 4: 256, 5: 2048}


def _parse_abr_map(spec):
    if not spec:
        return dict(ABR_HINTS)
    out = dict(ABR_HINTS)
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        k, v = part.split("=")
        out[int(k)] = int(v)
    return out
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

    # 源按 (A, rows) 分配一次；A=1 用 4MB（L3 内），A>1 用 rows*A*4B。
    # rows 默认 1<<20，显式给 rows 的用例（如 A=256）用小源，避免 1GB 级分配。
    x1 = torch.randn(1 << 20, generator=g)
    keys = sorted({(c["A"], int(c.get("rows", 1 << 20))) for c in cases if c["A"] > 1})
    x_multi = {k: torch.randn(k[1], k[0], generator=g).contiguous() for k in keys}

    rows = []
    for t in threads:
        _set_threads(torch, omp, t)
        for c in cases:
            K, A, dom = c["K"], c["A"], c["dom"]
            src = x1 if A == 1 else x_multi[(A, int(c.get("rows", 1 << 20)))]
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


def _parse_totals(spec):
    """`--total 'SGI=20800,PPT=4896'`：CSV 只含热点时，用它给出真实总调用数。"""
    out = {}
    for part in (spec or "").replace(" ", "").split(","):
        if part:
            k, v = part.split("=")
            out[k] = int(v)
    return out


def gen_cases(path, abr_map=None, per_model=False, default_th=12, totals=None):
    """从 trace CSV 提取热点形态：列 = model,path,K,A,dom,calls[,threads]。

    默认输出扁平用例列表（可直接给 `HPC_CASES`）；`per_model=True` 输出
    `{"models": [...]}`，每个模型带自己的线程数和调用次数，给 `--weighted` 用。
    """
    abr_map = _parse_abr_map(None) if abr_map is None else abr_map
    agg, models = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        lines = (ln for ln in f if not ln.lstrip().startswith("#"))
        for r in csv.DictReader(lines):
            path_name = PATH_ALIASES.get(r["path"].strip())
            if path_name is None:
                continue
            K, abr = int(r["K"]), int(r["A"])
            dom = int(r["dom"]) if r.get("dom") not in (None, "", "0") else 2048
            # A==1 的形态不分桶化/单遍，一律按 (dom, A) 重新打标签：
            # dom<=2048 走桶化，否则单遍；A>1 才是行拷贝。trace 的路径号只作溯源。
            A = 1 if path_name == "bucket" else abr_map.get(abr)
            if A is None:
                raise SystemExit(f"A-br={abr} 没有映射规则，用 --abr-map '4=256' 指定")
            shape = ("bucket" if dom <= 2048 else "gather") if A == 1 else "rowcopy"
            calls = int(r["calls"])
            name = f"gen_{shape}_K{K}_A{A}_dom{dom}"
            cur = agg.setdefault(name, dict(name=name, group="gen", K=K, A=A,
                                            dom=dom, path=shape, calls=0))
            cur["calls"] += calls
            mname = (r.get("model") or os.path.basename(path)).strip()
            th = int(r["threads"]) if r.get("threads") else default_th
            m = models.setdefault(mname, dict(name=mname, threads=th,
                                              total_calls=0, calls={}))
            m["calls"][name] = m["calls"].get(name, 0) + calls
            m["total_calls"] += calls
    print(f"# A-br 映射：{abr_map}（只有 3->64 有反馈锚定，其余是几何中点假设）",
          file=sys.stderr)
    if per_model:
        for m in models.values():
            # CSV 只有热点行时，覆盖率会假性 100%；--total 给真实总调用数。
            m["total_calls"] = (totals or {}).get(m["name"], m["total_calls"])
        return {"note": f"{os.path.basename(path)}：按 trace 调用次数加权；"
                        "形态由 A-br 桶中值映射，非真实 A",
                "models": sorted(models.values(), key=lambda m: m["name"])}
    return sorted(agg.values(), key=lambda c: -c["calls"])


def _load_weights(path, threads):
    """兼容两种输入：{models:[...]} 规格，或 --gen-cases 输出的用例列表。"""
    spec = json.load(open(path, encoding="utf-8"))
    if isinstance(spec, dict):
        return spec
    calls = {c["name"]: int(c.get("calls", 0)) for c in spec if c.get("calls")}
    return {"note": f"{path}（用例列表，调用次数即权重；覆盖率 = 库内形态占比）",
            "models": [{"name": os.path.basename(path).replace(".json", ""),
                        "threads": threads, "total_calls": sum(calls.values()),
                        "calls": calls}]}


def _detail_lines(rows, top=6):
    """按 hpc 耗时（次数×单次）排序，不用调用次数排——两者经常不是一回事。"""
    tot_h = sum(r[0] for r in rows) or 1.0
    out = [f"  按耗时 Top {min(top, len(rows))}"
           f"（形态  calls  hpcµs  占比  aten/hpc）"]
    for wh, w, name, a, h in sorted(rows, reverse=True)[:top]:
        out.append(f"    {name:34s} {w:6d} {h:7.1f} {100 * wh / tot_h:4.0f}% "
                   f"{a / h:5.2f}x")
    return out


def weighted(weights_path, times_path, threads=12, detail=False, md=False):
    """按调用次数加权，两个口径一起给，并把"未覆盖长尾"的不确定性显式化。

    默认输出是窄版（一模型一块，宽度 <=70 列，方便直接看终端）；`md=True`
    输出 markdown 表，给文档用。覆盖内比值只反映被覆盖的形态，所以额外给
    两个"全程"估计：未覆盖调用按与本算子持平（①）/ 按本算子慢 2x（③）补全。
    中间那种"按覆盖内同比例"补全恒等于覆盖内比值，不单列。
    """
    spec = _load_weights(weights_path, threads)
    times = json.load(open(times_path, encoding="utf-8"))
    table = {}
    for r in times:
        table[(r["name"], r["mode"], r["threads"])] = r["us"]

    print(f"# 权重来源：{spec.get('note', weights_path)}")
    print("# aten/hpc >1 = 本算子更快；index_sve/原生 <1 = 本算子更快（互为倒数）")
    if md:
        # ②（未覆盖与覆盖内同比例）恒等于覆盖内比值，单列没信息量，省掉。
        print("| 模型 | TH | 覆盖 | aten/hpc | sve/原生 | 全程① | 全程③ |")
        print("|---|---|---|---|---|---|---|")

    for m in spec["models"]:
        th = m.get("threads", threads)
        covered = miss = 0
        sa = sh = 0.0
        rows = []
        for name, w in m["calls"].items():
            a = table.get((name, "aten", th))
            h = table.get((name, "hpc", th))
            if a is None or h is None:
                miss += w
                continue
            covered += w
            sa += w * a
            sh += w * h
            rows.append((w * h, w, name, a, h))
        total = m.get("total_calls", covered + miss)
        # total_calls 可能大于权重表里的热调用之和：差额是根本没列进 trace 的长尾。
        miss = max(miss, total - covered)
        if not sh or not covered:
            head = f"| {m['name']} | {th} | 0/{total} |"
            print(head + " n/a | n/a | n/a | n/a |" if md
                  else f"{m['name']}  TH={th}  覆盖 0/{total}，无可用测量")
            continue
        a_mean = sa / covered
        cov = sa / sh
        # 未覆盖调用的单次 aten 取覆盖内均值；hpc 侧按 ①持平 / ③慢2x 两种假设。
        est0 = (sa + miss * a_mean) / (sh + miss * a_mean)
        est2 = (sa + miss * a_mean) / (sh + miss * a_mean * 2.0)
        if md:
            print(f"| {m['name']} | {th} | {100 * covered / total:.0f}% "
                  f"({covered}/{total}) | {cov:.3f}x | {1 / cov:.3f}x | "
                  f"{est0:.3f}x | {est2:.3f}x |")
            if detail:
                # 每行再缩进 4 格：markdown 里是代码块，等宽对齐才不会被折掉。
                print("\n" + "\n".join("    " + ln for ln in _detail_lines(rows)))
                print()
            continue
        print(f"\n{m['name']}  TH={th}  覆盖 {covered}/{total} "
              f"({100 * covered / total:.0f}%)")
        print(f"  覆盖内    aten/hpc {cov:.3f}x   index_sve/原生 {1 / cov:.3f}x")
        print(f"  全程      ①持平 {est0:.3f}x   ③慢2x {est2:.3f}x"
              f"   （②同覆盖 = 上面 {cov:.3f}x）")
        if detail:
            print("\n".join(_detail_lines(rows)))

    print("\n① 未覆盖调用按 aten/hpc=1（与本算子持平）补全"
          "\n② 按覆盖内同比值补全（恒等于覆盖内比值，故不单列）"
          "\n③ 按本算子慢 2x（aten/hpc=0.5）补全"
          "\n   ①和③夹住 1.0 就说明覆盖率不足以下方向结论。"
          "\n> 覆盖内比值只说明「被覆盖的形态」；按调用次数覆盖率不等于按耗时"
          "覆盖率——\n  长尾大形态少数派可以主导总时间（RESULTS §2.11.5）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-cases", action="store_true", help="打印用例库 JSON")
    ap.add_argument("--gen-cases", metavar="CSV", help="从 trace CSV 提取用例")
    ap.add_argument("--weighted", metavar="WEIGHTS", help="按调用次数加权")
    ap.add_argument("--times", metavar="JSON", help="--weighted 用的测量结果")
    ap.add_argument("--th", type=int, default=12,
                    help="--weighted 用哪个线程档（用例列表输入时）")
    ap.add_argument("--detail", action="store_true", help="打印按耗时的形态排名")
    ap.add_argument("--md", action="store_true",
                    help="--weighted 输出 markdown 表（默认是窄版，给终端看）")
    ap.add_argument("--abr-map", metavar="SPEC", default="",
                    help="覆盖 A-br 映射，如 '3=64,4=256'")
    ap.add_argument("--per-model", action="store_true",
                    help="--gen-cases 输出按模型分组的权重（含各自线程数）")
    ap.add_argument("--total", metavar="SPEC", default="",
                    help="真实总调用数，如 'SGI=20800,PPT=4896'（CSV 只含热点时用）")
    ap.add_argument("--cases", metavar="JSON",
                    help="用生成的用例列表测量（等价于 HPC_CASES）")
    args = ap.parse_args()

    if args.cases:
        global CASES
        CASES = json.load(open(args.cases, encoding="utf-8"))
        for _c in CASES:
            _c.setdefault("group", "gen")
    if args.gen_cases:
        print(json.dumps(gen_cases(args.gen_cases, _parse_abr_map(args.abr_map),
                                   args.per_model, args.th,
                                   _parse_totals(args.total)),
                         ensure_ascii=False, indent=1))
        return
    if args.emit_cases:
        print(json.dumps(CASES, ensure_ascii=False, indent=1))
        return
    if args.weighted:
        if not args.times:
            raise SystemExit("--weighted 需要 --times（driver 的 --out 结果）")
        weighted(args.weighted, args.times, args.th, args.detail, args.md)
        return

    rows = measure(CASES, THREADS)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
