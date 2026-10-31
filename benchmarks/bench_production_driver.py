"""生产形态对拍驱动：交替启动 aten/hpc 两个进程，汇总 `bench_production_shapes.py`。

用法：
  HPC_AFFINITY=190-227 HPC_REPS=3 python benchmarks/bench_production_driver.py \
      --out results.json [--fit]

两模式交替跑 HPC_REPS 轮，每个 (case, threads) 取轮间中位数再算比值。小 K 的
单次调用只有几微秒，同进程内 aten/hpc 会互相影响（覆盖开关是进程级），所以必须
两进程；共享箱上绝对时间会漂移，**比值只认同一轮交替的数据**。

`--fit` 对 k_sweep 组做 T = 固定开销 + 斜率*K 的最小二乘拟合，用来把"每元素
成本"和"不随 K 缩小的固定开销"分开——生产形态下后者才是 <1x 的主因。
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "benchmarks", "bench_production_shapes.py")
REPS = int(os.environ.get("HPC_REPS", "3"))
THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,6,12,19").split(",")]


def run(mode):
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.check_output([sys.executable, SCRIPT], env=env, cwd=ROOT,
                                  stderr=subprocess.DEVNULL)
    rows = json.loads(out)
    for r in rows:
        r.pop("iters", None)
    return rows


def fit(rows, name_prefix="ksweep"):
    """T(K) = a + b*K（µs / µs per element），逐 (mode, threads) 拟合。

    斜率就是"每元素墙钟成本"，它随线程数的下降速度就是内核自身的强扩展效率
    （与总时间不同：总时间被固定开销摊平，看不出并行段的真实扩展）。
    """
    slope = {}
    for mode in ("aten", "hpc"):
        for t in THREADS:
            pts = [(r["K"], r["us"]) for r in rows
                   if r["mode"] == mode and r["threads"] == t
                   and r["name"].startswith(name_prefix)]
            if len(pts) < 3:
                continue
            n = len(pts)
            sk = sum(k for k, _ in pts)
            su = sum(u for _, u in pts)
            skk = sum(k * k for k, _ in pts)
            sku = sum(k * u for k, u in pts)
            b = (n * sku - sk * su) / (n * skk - sk * sk)
            a = (su - b * sk) / n
            slope[(mode, t)] = b * 1e3
            print(f"{mode:5s} t={t:2d}  固定开销 {a:7.2f} us   斜率 {b * 1e3:5.2f} ns/元素")

    tmax = max(THREADS)
    for mode in ("aten", "hpc"):
        if (mode, 1) in slope and (mode, tmax) in slope:
            s1, sn = slope[(mode, 1)], slope[(mode, tmax)]
            sp = s1 / sn
            # 两段模型 slope(t) = 串行 a + 并行 b/t：a 是不随线程缩小的每元素成本
            a = (s1 - sn * tmax) / (1 - tmax)
            print(f"{mode:5s} 每元素 {s1:.2f} -> {sn:.2f} ns（{sp:.1f}x / {tmax}t = "
                  f"{100 * sp / tmax:.0f}% 效率）；串行段约 {a:.2f} ns/元素"
                  f"（{100 * a / s1:.0f}%）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="把合并后的测量行写成 JSON")
    ap.add_argument("--fit", action="store_true", help="拟合固定开销/每元素成本")
    ap.add_argument("--fit-from", default="", metavar="JSON",
                    help="只对已有的测量结果做拟合（不重跑）")
    args = ap.parse_args()

    if args.fit_from:
        print("# 拟合（k_sweep 组：A=1, dom=2048）")
        fit(json.load(open(args.fit_from, encoding="utf-8")))
        return

    runs = {"aten": [], "hpc": []}
    for _ in range(REPS):
        for mode in ("aten", "hpc"):
            runs[mode].append(run(mode))

    acc = {}
    for mode, reps in runs.items():
        for rows in reps:
            for r in rows:
                key = (r["name"], r["K"], r["A"], r["dom"], r["threads"])
                acc.setdefault(key, {"aten": [], "hpc": [], "meta": r})[mode].append(r["us"])

    merged = []
    for key, d in sorted(acc.items()):
        name = key[0]
        a = statistics.median(d["aten"])
        h = statistics.median(d["hpc"])
        merged.append(dict(d["meta"], mode="aten", us=round(a, 2)))
        merged.append(dict(d["meta"], mode="hpc", us=round(h, 2)))

    if args.out:
        json.dump(merged, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    print("| case | K | A | dom | t | aten µs | hpc µs | aten/hpc |")
    print("|---|---|---|---|---|---|---|---|")
    for key, d in sorted(acc.items()):
        name, K, A, dom, t = key
        a = statistics.median(d["aten"])
        h = statistics.median(d["hpc"])
        flag = "**" if a / h < 1 else ""
        print(f"| {name} | {K} | {A} | {dom} | {t} | {a:.2f} | {h:.2f} | "
              f"{flag}{a / h:.2f}x{flag} |")

    below = [(k[0], k[4], statistics.median(d["aten"]) / statistics.median(d["hpc"]))
             for k, d in acc.items()]
    bad = [b for b in below if b[2] < 1.0]
    print(f"\n低于 1x：{len(bad)}/{len(below)} 格", file=sys.stderr)
    if args.fit:
        print("\n# 拟合（k_sweep 组：A=1, dom=2048）")
        fit(merged)


if __name__ == "__main__":
    main()
