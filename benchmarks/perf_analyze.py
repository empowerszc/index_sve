"""用 `perf stat` 量"每行花了多少周期、缺了几条 line"，反推等效在飞条数。

这是 `docs/headroom.md` §3 里"aten 行路径 ≈ 1 条在飞"那句话的实测依据：探针只能
说明"硬件能给 16-20 条"，要说某条实现只用了 1 条，必须同时拿到**每行周期数**和
**每行缺页行数**，再用

    等效在飞条数 = (L1 refill / 行) × 274 周期 ÷ (周期 / 行)

用 `l1d_cache_refill`（L1 数据缺行 = 真的去 L2 要了一条 line）当缺行数：它对
A=16（64B/行）正好是 1.00/行，与"每行读一条 line、写整行不需要 RFO"完全对得上，
是本表里唯一可以直接当"line 数"用的计数器。`l2d/l3d_cache_refill` 在同表里
明显大于 L1 的缺行数（aten 8.24 vs 1.04），量纲/口径对不上（疑似按扇区或含
预取），只作相对参考，不用来推在飞条数。

其中 274 = §2.16.3 实测的一条依赖链延迟（137ns @ 2.0GHz，探针口径 cyc = 2×ns）。

**为什么要两段相减**：进程启动（`import torch` 本机 7.2s + 建张量 3s）比一次
3 秒的负载还长，直接拿 `perf stat` 的总数除行数会把启动的周期也算进去（实测放大
4 倍）。所以每个格子跑两次（`--secs/3` 与 `--secs`），把计数器与调用次数都相减，
得到只有循环那一段的增量。这也让墙钟列变成两段的差 —— 比单次计时更干净。

口径提醒：三个 refill 计数器同时包含输出写与硬件预取的流量，绝对值只作同表内
相对比较；推导只信 `L1rf/行`。

用法（node 5）：
  numactl --physcpubind=190-227 --membind=5 \
    python benchmarks/perf_analyze.py --cases r16,r16raw --threads 1,8 --secs 9
  python benchmarks/perf_analyze.py --cases r16 --threads 8 --record   # 热点符号
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVENTS = ["cycles", "instructions", "l1d_cache_refill", "l2d_cache_refill",
          "l3d_cache_refill"]
LAT_CYC = 274.0   # 一条依赖链的周期数（RESULTS §2.16.3）
K_OF = {"r3": 700000, "r16raw": 65536, "r16": 131072, "r64raw": 32768,
        "r64": 131072, "r256": 32768, "s1": 1000000, "col": 200000,
        "s4": 200000}


def run_counters(case, mode, threads, secs, events):
    env = dict(os.environ)
    env["MODE"] = mode
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    cmd = ["perf", "stat", "-x,", "-e", ",".join(events), "--",
           sys.executable, os.path.join(HERE, "perf_one.py"), case,
           str(threads), str(secs)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"perf 失败：\n{p.stderr[-3000:]}")
    counters = {}
    for line in p.stderr.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[0].replace(",", "").isdigit():
            if parts[2] in events:
                counters[parts[2]] = int(parts[0].replace(",", ""))
    calls = None
    for line in p.stdout.splitlines():
        toks = line.split()
        if len(toks) == 4 and toks[0] == case:
            calls = int(toks[3])
    if calls is None or "cycles" not in counters:
        raise SystemExit(f"没解析到计数器/调用次数：\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    return counters, calls


def one(case, mode, threads, secs):
    lo, n_lo = run_counters(case, mode, threads, secs / 3.0, EVENTS)
    hi, n_hi = run_counters(case, mode, threads, secs, EVENTS)
    dc = {k: hi[k] - lo[k] for k in hi}
    calls = n_hi - n_lo
    rows = calls * K_OF[case]
    if rows <= 0:
        raise SystemExit(f"{case} {mode} {threads}t 两段调用次数相同，加大 --secs")
    cyc_row = dc["cycles"] / rows
    row = dict(case=case, mode=mode, t=threads, calls=calls, rows=rows,
               ns_row=(secs * 2.0 / 3.0) * 1e9 / rows,
               cyc_row=cyc_row,
               ins_row=dc["instructions"] / rows,
               ipc=dc["instructions"] / max(dc["cycles"], 1),
               l1=dc["l1d_cache_refill"] / rows,
               l2=dc["l2d_cache_refill"] / rows,
               l3=dc["l3d_cache_refill"] / rows)
    row["inflight"] = row["l1"] * LAT_CYC / max(cyc_row, 1e-9)
    return row


def record(case, mode, threads, secs):
    env = dict(os.environ)
    env["MODE"] = mode
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    out = f"/tmp/opencode/perf-{case}-{mode}-{threads}.data"
    subprocess.run(["perf", "record", "-q", "-o", out, "-g", "--",
                    sys.executable, os.path.join(HERE, "perf_one.py"), case,
                    str(threads), str(secs)], env=env, check=True)
    p = subprocess.run(["perf", "report", "--no-children", "--percent-limit",
                        "1.0", "-i", out, "--stdio"], capture_output=True,
                       text=True)
    print(f"\n== perf record {case} {mode} {threads}t (data: {out})")
    print("\n".join(p.stdout.splitlines()[:22]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="r16,r16raw")
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--secs", type=float, default=9.0,
                    help="长段的秒数；短段自动取它的 1/3 并相减")
    ap.add_argument("--modes", default="aten,hpc")
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if c]
    threads = [int(t) for t in a.threads.split(",") if t]
    modes = [m for m in a.modes.split(",") if m]

    rows = []
    for case in cases:
        for t in threads:
            for mode in modes:
                r = one(case, mode, t, a.secs)
                rows.append(r)
                print(f"{case} {mode} {t}t ns/行={r['ns_row']:.1f} "
                      f"cyc/行={r['cyc_row']:.0f}", file=sys.stderr, flush=True)
                if a.record:
                    record(case, mode, t, a.secs / 3.0)

    print(f"\n{'case':>7s} {'mode':>4s} {'t':>3s} {'ns/行':>7s} {'cyc/行':>7s} "
          f"{'ins/行':>7s} {'IPC':>5s} {'L1rf/行':>7s} {'L2rf/行':>7s} "
          f"{'L3rf/行':>7s} {'在飞':>5s}")
    for r in rows:
        print(f"{r['case']:>7s} {r['mode']:>4s} {r['t']:3d} {r['ns_row']:7.1f} "
              f"{r['cyc_row']:7.0f} {r['ins_row']:7.1f} {r['ipc']:5.2f} "
              f"{r['l1']:7.2f} {r['l2']:7.2f} {r['l3']:7.2f} {r['inflight']:5.1f}")


if __name__ == "__main__":
    main()
