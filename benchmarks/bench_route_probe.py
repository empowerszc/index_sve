"""路由成本分层探针：把"路径条数"和"每次调用的固定开销"分开测。

动机：`index_into` 里挂了很多条路由，直觉上"路越多越慢"。本脚本量三件事，
用来分辨哪种开销真的与路数有关：

  1. `chain`  —— 短调用到底有没有进入路由链（用计数器确认走了哪条路，而不是
     靠条件推断；本脚本的前两次尝试就是栽在"用例其实还在快路径上"）；
  2. `gate`   —— 会扫数据的资格判定（值域探测）失败一次的代价。做法是把门的
     两侧做成**同局部性**：值域恰好等于阈值 vs 阈值+1，源都只摸前几 KB；
  3. `scan`   —— 一遍 O(K) 扫描的裸价（`aminmax` 近似），给 `gate` 的差值找参照；
  4. `census` —— 语料里 26 个形态实际落到哪些路（决定"哪条路可以删"）。

结果写入 `benchmarks/RESULTS.md` §2.13，其他文档只引结论。

用法：
  HPC_AFFINITY=190-227 HPC_THREADS=8 python benchmarks/bench_route_probe.py
  HPC_THREADS=8 python benchmarks/bench_route_probe.py --json > probe.json
"""
import argparse
import json
import os
import sys
import time

AFFINITY = os.environ.get("HPC_AFFINITY", "190-227")
THREADS = int(os.environ.get("HPC_THREADS", "8"))
ROUNDS = int(os.environ.get("HPC_ROUNDS", "5"))
GATE_K = int(os.environ.get("HPC_GATE_K", "65536"))
GATE_DOMS = [int(d) for d in os.environ.get("HPC_GATE_DOMS", "512,2048,2049,4096").split(",")]

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


def med_us(fn, target_s=0.15):
    import statistics
    for _ in range(30):
        fn()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    per = (time.perf_counter() - t0) / 5
    iters = int(max(20, min(3000, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def measure():
    import torch
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "benchmarks"))
    import index_sve
    from index_sve import _hpc_index as C

    if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1":
        raise SystemExit("本脚本需要覆盖生效（不要设 HPC_INDEX_NO_OVERRIDE=1）")
    os.sched_setaffinity(0, _affinity())
    torch.set_num_threads(THREADS)
    g = torch.Generator().manual_seed(7)
    out = {"threads": THREADS, "rounds": ROUNDS}

    def route(fn):
        before = {k: getattr(C, k)() for k in COUNTERS}
        fn()
        hit = {k: getattr(C, k)() - v for k, v in before.items()
               if getattr(C, k)() - v}
        return ",".join(sorted(hit)) if hit else "<no counter>"

    # ---- 1) chain：短调用有没有进链 -------------------------------------
    x1 = torch.randn(8, generator=g)
    c4 = torch.arange(4)
    nc4 = torch.arange(8)[::2]          # 非连续视图（arange(0,8,2) 是连续的）
    x2 = torch.randn(4, 8, generator=g)
    i2 = torch.tensor([1])
    chain_cases = [
        ("k4_contig", "1D K=4 下标连续", lambda: x1[c4]),
        ("k4_strided", "1D K=4 下标非连续", lambda: x1[nc4]),
        ("k1_2d_a8", "2D A=8 K=1 下标连续", lambda: x2[i2]),
        ("k1_pyint", "1D K=1 Python int（参照）", lambda: x1[3]),
    ]
    out["chain"] = []
    for key, desc, fn in chain_cases:
        out["chain"].append({"case": key, "desc": desc, "route": route(fn),
                             "us": round(med_us(fn), 3)})

    # ---- 2) gate：同局部性下探测落选的代价 -------------------------------
    K = GATE_K
    src = torch.randn(1 << 20, generator=g)          # 4MB，A=1 用
    idx_by_dom = {d: torch.randint(0, d, (K,), generator=g, dtype=torch.int32)
                  for d in GATE_DOMS}
    # 交替轮次：每轮把各 dom 都测一遍，取每个 dom 的中位数，抵消箱态漂移
    per_dom = {d: [] for d in GATE_DOMS}
    for _ in range(ROUNDS):
        for d in GATE_DOMS:
            idx = idx_by_dom[d]
            per_dom[d].append(med_us(lambda: src[idx], target_s=0.05))
    import statistics
    out["gate"] = []
    for d in GATE_DOMS:
        idx = idx_by_dom[d]
        out["gate"].append({"dom": d, "K": K, "route": route(lambda: src[idx]),
                            "us": round(statistics.median(per_dom[d]), 3),
                            "runs": [round(v, 3) for v in per_dom[d]]})

    # ---- 3) scan：一遍 O(K) 扫描的裸价 -----------------------------------
    out["scan"] = []
    for dt, tag in ((torch.int32, "i32"), (torch.int64, "i64")):
        idx = torch.randint(0, 1 << 20, (K,), generator=g, dtype=dt)
        out["scan"].append({"item": "aminmax_" + tag, "K": K,
                            "us": round(med_us(lambda: torch.aminmax(idx)), 3)})
    idx32 = idx_by_dom[GATE_DOMS[0]].to(torch.int32)
    out["scan"].append({"item": "linearize_spec", "K": K,
                        "us": round(med_us(lambda: C.linearize_spec(src, [idx32])), 3)})
    out["scan"].append({"item": "debug_spec", "K": K,
                        "us": round(med_us(lambda: C.debug_spec(src, [idx32])), 3)})

    # ---- 4) census：语料形态落到哪些路 -----------------------------------
    import bench_production_shapes as BPS
    A1_src = torch.randn(1 << 20, generator=g)
    multi = {}
    for c in BPS.CASES:
        if c["A"] > 1:
            rows = int(c.get("rows", 1 << 20))
            multi[(c["A"], rows)] = torch.randn(rows, c["A"], generator=g)
    tally = {}
    out["census"] = []
    for c in BPS.CASES:
        A = c["A"]
        src_c = A1_src if A == 1 else multi[(A, int(c.get("rows", 1 << 20)))]
        idx = torch.randint(0, c["dom"], (c["K"],), generator=g)
        tag = route(lambda: src_c[idx])
        out["census"].append({"name": c["name"], "K": c["K"], "A": A,
                              "dom": c["dom"], "route": tag})
        tally[tag] = tally.get(tag, 0) + 1
    out["census_tally"] = tally
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = measure()
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    print(f"threads={res['threads']} rounds={res['rounds']}")
    print("\n[chain] 短调用是否进链")
    for r in res["chain"]:
        print(f"  {r['desc']:22s} us={r['us']:7.3f}  route={r['route']}")
    print("\n[gate] 同局部性下值域门两侧（K=%d）" % res["gate"][0]["K"])
    base = next((r["us"] for r in res["gate"] if r["dom"] == 2048), None)
    for r in res["gate"]:
        ratio = f"  x{r['us'] / base:5.2f}" if base else ""
        print(f"  dom={r['dom']:8d} us={r['us']:8.3f}{ratio}  route={r['route']}")
    print("\n[scan] 一遍 O(K) 扫描的裸价")
    for r in res["scan"]:
        print(f"  {r['item']:16s} us={r['us']:8.3f}")
    print("\n[census] 语料路由分布（%d 形态）" % len(res["census"]))
    for tag, n in sorted(res["census_tally"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {tag}")
    print("\n[census 明细]")
    for r in res["census"]:
        print(f"  {r['name']:26s} K={r['K']:6d} A={r['A']:4d} dom={r['dom']:8d}"
              f"  {r['route']}")


if __name__ == "__main__":
    main()
