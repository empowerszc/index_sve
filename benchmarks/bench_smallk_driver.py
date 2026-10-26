"""交替启动 aten/hpc 两个进程，汇总 `bench_smallk_real.py` 的中位数。

用法：
  HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
    python benchmarks/bench_smallk_driver.py

每个 (case, K, threads) 在每个进程里取中位数；两模式交替跑 HPC_REPS 轮，
再对轮间中位数取中位数，最后打印 aten/hpc 比值。共享箱上绝对时间会漂移，
比值以同一轮交替数据为准。
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "benchmarks", "bench_smallk_real.py")
REPS = int(os.environ.get("HPC_REPS", "3"))


def run(mode):
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.check_output([sys.executable, SCRIPT], env=env, cwd=ROOT)
    return json.loads(out)


def main():
    runs = {"aten": [], "hpc": []}
    for _ in range(REPS):
        for mode in ("aten", "hpc"):
            runs[mode].append(run(mode))

    acc = {}
    for mode, reps in runs.items():
        for rows in reps:
            for r in rows:
                key = (r["case"], r["K"], r["threads"])
                acc.setdefault(key, {"aten": [], "hpc": []})[mode].append(r["us"])

    print("| case | K | threads | aten us | hpc us | speedup |")
    print("|---|---|---|---|---|---|")
    for (case, K, t), d in sorted(acc.items()):
        a = statistics.median(d["aten"])
        h = statistics.median(d["hpc"])
        print(f"| {case} | {K} | {t} | {a:.2f} | {h:.2f} | {a / h:.2f}x |")


if __name__ == "__main__":
    main()
