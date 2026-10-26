"""Alternate aten/hpc processes for benchmarks/bench_realpath.py.

Usage:
  HPC_AFFINITY=190-227 HPC_THREADS=1,8,16,38 \
    python benchmarks/bench_realpath_driver.py

Each (case, threads) is measured as the median inside every process; the two
modes are run alternately for HPC_REPS rounds, then the median of the
per-round medians is compared.  This keeps the shared-box drift out of the
aten/hpc ratio.
"""
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "benchmarks", "bench_realpath.py")
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
                key = (r["case"], r["threads"])
                acc.setdefault(key, {"aten": [], "hpc": []})[mode].append(r["us"])

    print("| case | threads | aten us | hpc us | speedup |")
    print("|---|---|---:|---:|---:|")
    for (case, t), d in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        a = statistics.median(d["aten"])
        h = statistics.median(d["hpc"])
        print(f"| {case} | {t} | {a:.2f} | {h:.2f} | {a / h:.2f}x |")


if __name__ == "__main__":
    main()
