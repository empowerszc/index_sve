"""按**路径**（而不是按场景）组织的收益矩阵。

回答的问题：`index_into` 里挂的那些路，各自到底值不值得存在。做法有两条口径：

1. **路径 vs ATen**：落在该路径上的形态相对 aten 的比值（这是产品口径）。
2. **路径 vs 兄弟路径**：同一份工作量、只差一个谓词，从而落到另一条路上。
   这条是"这条路径存在的理由"——没有它，只能看到"这个形态快"，说不清是"这条路
   快"还是"这个形态本身就快"。

为什么必须有第 2 条：本仓没有 per-route kill switch，无法把某条路关掉再测；能做的
是用谓词边界造出"同工作量的两条路"。每个 case 的 `sibling` 字段就是它的对照。

每个 case 在 hpc 模式下会读出路由计数器，和 `expect` 对照打印 `EXPECT MISMATCH`。
**不要凭条件猜路由**（本仓前两版探针就栽在 randn 默认 f32、arange 是连续张量上）。

`--ab` 默认跑 `HPC_ROUNDS=3` 轮，每轮都重新起 aten/hpc 两个子进程（覆盖开关是
进程级的，同进程内切不了）；表中取每格的中位数，并给三轮极差作噪声。1t 的比值
稳定到两位有效数字，16t 只能按"同带"读——单次调用只有几十 µs 时，固定开销与
线程池自旋态抖动占比大（RESULTS §2.11.1、§2.14.4）。

用法：
  # 两进程交替（aten / hpc 各一个子进程）× 3 轮，打印单表
  HPC_AFFINITY=190-227 python benchmarks/bench_path_matrix.py --ab
  # 只看每个 case 期望落在哪条路（不测时间，改路由后先跑这个）
  HPC_AFFINITY=190-227 python benchmarks/bench_path_matrix.py --routes
  # 单进程原始 JSON（调试用）
  HPC_AFFINITY=190-227 python benchmarks/bench_path_matrix.py --mode hpc --json
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

AFFINITY = os.environ.get("HPC_AFFINITY", "190-227")
THREAD_LIST = [int(t) for t in os.environ.get("HPC_THREADS", "1,8,16").split(",")]
ROUNDS = int(os.environ.get("HPC_ROUNDS", "3"))

COUNTERS = ["mask_fused_count", "small_single_count", "small_multi_count",
            "strided_rows_count", "rowcopy_raw_count", "single_1d_count",
            "single_1d_sve_count"]


def _affinity():
    cpus = set()
    for part in AFFINITY.replace(" ", "").split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


def _omp():
    import ctypes
    for line in open("/proc/self/maps"):
        if "libomp.so" in line:
            lib = ctypes.CDLL(line.split()[-1])
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            return lib
    return None


def med_us(fn, target_s=0.12):
    for _ in range(20):
        fn()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    per = (time.perf_counter() - t0) / 5
    iters = int(max(20, min(2000, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6



def _expect_at(expect, t):
    """expect 可以是 str / tuple / {线程数: tag}（dict 里缺的线程号表示不校验）。"""
    if isinstance(expect, dict):
        return expect.get(str(t))
    return expect


def build(torch):
    """返回 [(case dict)]；张量在这里一次性建好，`sibling` 指向对照 case 的 key。"""
    g = torch.Generator().manual_seed(7)
    x1 = torch.randn(1 << 20, generator=g)                       # 4MB，A=1 用
    x_mask64 = torch.randn(1 << 16, generator=g)
    x_mask = torch.randn(1 << 18, generator=g)                   # mask 用（O(numel) 太慢）
    x_mask1m = torch.randn(1 << 20, generator=g)
    small_src = torch.randn(4096, generator=g)                   # 16KB，单遍标量门用
    x_a16 = torch.randn(1 << 18, 16, generator=g)                # 16MB
    x_a64 = torch.randn(8192, 64, generator=g)                   # 2MB
    x_a256 = torch.randn(8192, 256, generator=g)                 # 8MB
    x_a3 = torch.randn(1 << 18, 3, generator=g)
    x_a3i = torch.randint(0, 1 << 20, (1 << 18, 3), generator=g).to(torch.int32)
    x_perm = torch.randn(64, 200, 10, generator=g).permute(1, 0, 2)   # A=640，run=10
    x_t = torch.randn(64, 20000, generator=g).t()                     # A=64，run=1
    x_cl = torch.randn(4, 32, 14, 14, generator=g).to(memory_format=torch.channels_last)
    x_multi = torch.randn(512, 512, 64, generator=g)                   # 多索引 + A=64

    def i64(K, dom):
        return torch.randint(0, dom, (K,), generator=g)

    def i32(K, dom):
        return torch.randint(0, dom, (K,), generator=g, dtype=torch.int32)

    def i64_strided(K, dom):
        # 非连续视图；注意 torch.arange(0, 8, 2) 是连续张量，不能拿来当"非连续"
        base = torch.randint(0, dom, (2 * K,), generator=g)
        return base[::2]

    if1 = torch.randint(0, 512, (1024,), generator=g)
    if2 = torch.randint(0, 512, (1024,), generator=g)
    if1b = torch.randint(0, 512, (16384,), generator=g)
    if2b = torch.randint(0, 512, (16384,), generator=g)
    # run=2 的兄弟形态：A 与 run=10 相同（640），只有"连续 run >= 4"这一个谓词不同。
    # 必须保持非连续（permute 视图），否则小 K 时会被 small_single 在入口吃掉。
    x_run2 = torch.randn(320, 200, 2, generator=g).permute(1, 0, 2)

    C = []

    def add(key, path, desc, self_, idx, expect, sibling=None, threads=None,
            scale=""):
        C.append(dict(key=key, path=path, desc=desc, scale=scale,
                      **{"self": self_}, idx=idx, expect=expect,
                      sibling=sibling, threads=threads or THREAD_LIST))

    T3 = [1, 8, 16]

    # ---- 入口快路径：mask -------------------------------------------------
    for K, src in ((1 << 16, x_mask64), (1 << 18, x_mask), (1 << 20, x_mask1m)):
        add(f"e1_mask_k{K >> 10}k", "E1 mask", f"1D mask K={K >> 10}k / 50%",
            src, torch.rand(K, generator=g) < 0.5, "mask_fused_count",
            threads=[1, 8], scale=f"K={K >> 10}k")

    # ---- 入口快路径：small_single（K<=4096 且下标连续） --------------------
    for K in (512, 2048, 4096):
        add(f"e2_ss_k{K}", "E2 small_single", f"x(256k,16) K={K} 连续",
            x_a16, i64(K, 1 << 18), "small_single_count", threads=T3,
            scale=f"K={K}", sibling=None if K != 4096 else "e2_ss_nc_k4096")
    add("e2_ss_nc_k4096", "E2 small_single", "同形但下标非连续 → 掉出快路径",
        x_a16, i64_strided(4096, 1 << 18), "<no counter>", threads=T3,
        scale="K=4096", sibling="e2_ss_k4096")
    add("e2_ss_k8192", "R2 rowcopy_raw", "x(256k,16) K=8192（越过 4096 → raw 行拷贝）",
        x_a16, i64(8192, 1 << 18), "rowcopy_raw_count", threads=T3, scale="K=8192")

    # ---- 入口快路径：small_multi（多索引同形、连续、n<=65536） --------------
    add("e3_sm_k1024", "E3 small_multi", "x(512,512,64) 2 索引 K=1024 连续",
        x_multi, (if1, if2), "small_multi_count", threads=T3, scale="K=1024")

    # ---- R0 rowcopy_direct（通道后置布局直写） -----------------------------
    add("r0_direct", "R0 rowcopy_direct", "cl4d K=1024 连续下标（直写）",
        x_cl, i64(1024, 4), "rowcopy_raw_count", threads=T3, scale="K=1024")
    add("r0_bypass", "R0 rowcopy_direct", "同形但下标非连续 → 失去直写资格",
        x_cl, i64_strided(1024, 4),
        {"1": "<no counter>", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=1024", sibling="r0_direct")
    add("r0_k512", "R1 gather_direct", "cl4d K=512（<1024 → 通用直取）",
        x_cl, i64(512, 4), "<no counter>", threads=T3, scale="K=512")

    # ---- R1 gather_direct（K<1024） ---------------------------------------
    add("r1_direct", "R1 gather_direct", "x(1M) K=512 下标非连续",
        x1, i64_strided(512, 1 << 20), "<no counter>", threads=T3, scale="K=512")

    # ---- R2 rowcopy_raw vs R9 两遍：B*K<=65536 的门 ------------------------
    for K in (4097, 8192, 65536):
        add(f"r2_raw_k{K}", "R2 rowcopy_raw", f"x(8192,64) K={K}（越过 4096、B*K<=65536）",
            x_a64, i64(K, 8192), "rowcopy_raw_count", threads=T3, scale=f"K={K}")
    add("r9_k131072", "R9 两遍模型", "x(8192,64) K=131072（B*K 越门）", x_a64,
        i64(131072, 8192),
        {"1": "rowcopy_raw_count", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=131072", sibling="r2_raw_k65536")
    add("r9_a256", "R9 两遍模型", "x(8192,256) K=8192（A>64 → 两遍）", x_a256,
        i64(8192, 8192),
        {"1": "rowcopy_raw_count", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=8192")

    # ---- R3 strided_rows vs R10 gather_fused：连续 run>=4 的门 -------------
    add("r3_run10", "R3 strided_rows", "permute(1,0,2) A=640 run=10", x_perm,
        i64(1024, 200), "strided_rows_count", threads=T3, scale="K=1024 run=10")
    add("r3_run10_big", "R3 strided_rows", "同上 K=8192", x_perm,
        i64(8192, 200), "strided_rows_count", threads=T3, scale="K=8192 run=10")
    add("r10_run2", "R10 gather_fused", "同 A=640 但 run=2（run<4 → 不接管）",
        x_run2, i64(8192, 200),
        {"1": "<no counter>", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=8192 run=2", sibling="r3_run10_big")
    add("r10_transpose", "R10 gather_fused", "x(20000,64).t() A=64 run=1",
        x_t, i64(1024, 20000), "<no counter>", threads=T3, scale="K=1024 run=1")

    # ---- R4 多索引 fused：线程门 16 --------------------------------------
    add("r4_multi_t8", "R4 rowcopy_fused", "2 索引非连续 A=64 K=8192 × 8t（未过线程门）",
        x_multi, (if1b[::2], if2b[::2]), "<no counter>", threads=[8],
        scale="K=8192", sibling="r4_multi_t16")
    add("r4_multi_t16", "R4 rowcopy_fused", "同上 × 16t（线程门放行）", x_multi,
        (if1b[::2], if2b[::2]), "rowcopy_raw_count", threads=[16],
        scale="K=8192", sibling="r4_multi_t8")

    # ---- R5 桶化 vs R9 两遍：只差值域（MAX_DOMAIN=2048） -------------------
    for K in (8192, 65536):
        add(f"r5_bucket_k{K}", "R5 桶化", f"x(1M) K={K} dom=2048", x1,
            i32(K, 2048), "<no counter>", threads=T3, scale=f"K={K}")
    add("r9_dom2049", "R9 两遍模型", "同上 K=65536 但 dom=2049（门 +1）", x1,
        i32(65536, 2049),
        {"1": "single_1d_count", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=65536", sibling="r5_bucket_k65536")

    # ---- R6 大源单遍 SVE vs R7 标量单遍：源大小 / 下标宽度的 2×2 -----------
    for K in (8192, 65536):
        add(f"r6_sve_k{K}", "R6 single_1d_sve", f"源 4MB + i64 下标 K={K}", x1,
            i64(K, 1 << 20),
            {"1": "single_1d_count", "8": "single_1d_sve_count",
             "16": "single_1d_sve_count"},
            threads=T3, scale=f"K={K}")
        add(f"r7_scalar_k{K}", "R7 single_1d", f"源 16KB + i64 下标 K={K}", small_src,
            i64(K, 4096), "single_1d_count", threads=T3, scale=f"K={K}")
    add("r9_small_i32", "R9 两遍模型", "源 16KB + i32 下标（源太小 → 不走专用 SVE）",
        small_src, i32(65536, 4096),
        {"1": "single_1d_count", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=65536", sibling="r7_scalar_k65536")
    add("r9_big_i32", "R9 两遍模型", "源 4MB + i32 下标（i64 专用门不认）", x1,
        i32(65536, 1 << 20),
        {"1": "single_1d_count", "8": "<no counter>", "16": "<no counter>"},
        threads=T3, scale="K=65536", sibling="r6_sve_k65536")

    # ---- R10 fused vs R2 raw：只差数据 dtype（A=3 无 raw 内核） ------------
    add("r10_i32_a3", "R10 gather_fused", "x(256k,3) int32 K=8192",
        x_a3i, i64(8192, 1 << 18), "<no counter>", threads=T3, scale="K=8192")
    add("r2_f32_a3", "R2 rowcopy_raw", "同上但 f32（有 raw 行拷贝内核）", x_a3,
        i64(8192, 1 << 18), "rowcopy_raw_count", threads=T3, scale="K=8192",
        sibling="r10_i32_a3")
    return C


def measure(mode):
    import torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    import index_sve
    from index_sve import _hpc_index as C

    os.sched_setaffinity(0, _affinity())
    omp = _omp()
    cases = build(torch)
    rows = []
    for t in sorted({t for c in cases for t in c["threads"]}):
        torch.set_num_threads(t)
        if omp is not None:
            omp.omp_set_num_threads(t)
        for c in cases:
            if t not in c["threads"]:
                continue
            self_, idx = c["self"], c["idx"]
            args = list(idx) if isinstance(idx, tuple) else [idx]
            before = ({k: getattr(C, k)() for k in COUNTERS} if mode == "hpc" else {})
            fn = (lambda: self_[args[0]]) if len(args) == 1 else \
                 (lambda: self_[args[0], args[1]])
            us = med_us(fn)
            route = ""
            if mode == "hpc":
                hit = {k: getattr(C, k)() - v for k, v in before.items()
                       if getattr(C, k)() - v}
                route = ",".join(sorted(hit)) if hit else "<no counter>"
            rows.append({"mode": mode, "key": c["key"], "path": c["path"],
                         "desc": c["desc"], "scale": c["scale"],
                         "sibling": c["sibling"], "threads": t,
                         "us": round(us, 3), "route": route,
                         "expect": _expect_at(c["expect"], t)})
            print(f"{mode:5s} t={t:2d} {c['key']:14s} {us:9.3f} us  {route}",
                  file=sys.stderr, flush=True)
    return rows


def _one_round():
    out = {}
    for mode in ("aten", "hpc"):
        env = dict(os.environ)
        env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", mode,
                            "--json"], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"{mode} 子进程失败：\n{p.stderr[-2000:]}")
        for r in json.loads(p.stdout):
            out[(r["mode"], r["key"], r["threads"])] = r
    return out


def run_ab():
    rounds = [_one_round() for _ in range(ROUNDS)]
    rows = []
    for (mode, key, t) in sorted(rounds[0]):
        if mode != "hpc":
            continue
        ratios, aten_us, us = [], [], []
        for rr in rounds:
            a, h = rr[("aten", key, t)], rr[(mode, key, t)]
            aten_us.append(a["us"])
            us.append(h["us"])
            ratios.append(a["us"] / h["us"])
        last = rounds[-1][(mode, key, t)]
        rows.append({**last,
                     "aten_us": round(statistics.median(aten_us), 3),
                     "us": round(statistics.median(us), 3),
                     "ratio": round(statistics.median(ratios), 3),
                     "ratio_lo": round(min(ratios), 3),
                     "ratio_hi": round(max(ratios), 3),
                     "rounds": len(rounds)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aten", "hpc"], default="hpc")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--routes", action="store_true", help="只打印每个 case 的路由")
    args = ap.parse_args()

    if args.ab:
        rows = run_ab()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False))
            return
        by_key = {}
        for r in rows:
            by_key.setdefault(r["key"], []).append(r)
        print(f"{'case':16s} {'path':20s} {'t':>3s} {'aten':>9s} {'hpc':>9s} "
              f"{'ratio':>6s} {'noise':>6s}  route")
        bad = 0
        for key, rs in by_key.items():
            for r in rs:
                flag = "" if (r["expect"] is None or r["route"] == r["expect"]) \
                    else "  <-- EXPECT MISMATCH"
                bad += bool(flag)
                noise = (r["ratio_hi"] - r["ratio_lo"]) / max(r["ratio"], 1e-9)
                print(f"{r['key']:16s} {r['path']:20s} {r['threads']:3d} "
                      f"{r['aten_us']:9.2f} {r['us']:9.2f} {r['ratio']:6.3f} "
                      f"{noise * 100:5.0f}%  {r['route']}{flag}")
        print(f"\n{bad} 处 EXPECT MISMATCH")
        return

    if args.routes:
        for c in build(__import__("torch")):
            print(f"{c['key']:14s} {c['path']:22s} expect={c['expect']:20s} "
                  f"sibling={c['sibling']}")
        return

    rows = measure(args.mode)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False))
        return
    for r in rows:
        flag = "" if (r["expect"] is None or r["route"] == r["expect"]) \
                    else "  <-- EXPECT MISMATCH"
        print(f"{r['key']:14s} t={r['threads']:2d} {r['us']:9.3f} us  "
              f"{r['route']}{flag}")


if __name__ == "__main__":
    main()
