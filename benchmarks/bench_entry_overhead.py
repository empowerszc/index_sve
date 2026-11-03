"""入口固定开销拆解：`x[idx]` 全链路里，dispatcher / `__getitem__` / normalize /
输出分配 / 内核各占多少。

动机：`bench_index.py` 直呼 `hpc_index()`，看不到 normalize 与 dispatcher；生产
`x[idx]` 走全链路。本脚本把这条链路拆开，并检查"形态越复杂、normalize 越贵"。

方法：
- `K=0` 的空下标把内核工作降到 0，测到的就是纯入口（dispatch + normalize 设置
  + 空输出分配）；
- 同一 K 换下标表示（连续 i64 / i32 / stride=2 / 2D (1,K) / 双索引）只改
  normalize 的解码复杂度；
- `torch.empty(K)` 单独测输出分配；`hpc_index()` 直呼测"绕过 dispatcher"能省多少
  （注意它多了 Python list + pybind 转换）。

用法：
  # 两进程交替（覆盖开关是进程级），打印拆解表
  HPC_AFFINITY=190-227 HPC_THREADS=1,19 python benchmarks/bench_entry_overhead.py --ab
  # 单进程原始 JSON
  HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_entry_overhead.py
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

AFFINITY = os.environ.get("HPC_AFFINITY", "190-227")
THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,19").split(",")]
KS = [int(k) for k in os.environ.get("HPC_KS", "0,512,2048").split(",")]


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
    iters = int(max(50, min(2000, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def measure():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import index_sve  # noqa: F401  (是否安装覆盖由 HPC_INDEX_NO_OVERRIDE 决定)
    from index_sve import index as hpc_index

    os.sched_setaffinity(0, _affinity())
    omp = _omp()
    mode = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1 << 20, generator=g)
    x2 = torch.randn(1 << 20, 4, generator=g)

    rows = []
    for t in THREADS:
        torch.set_num_threads(t)
        if omp is not None:
            omp.omp_set_num_threads(t)
        for K in KS:
            base64 = torch.randint(0, 1 << 20, (max(2 * K, 2),), generator=g, dtype=torch.long)
            i26 = base64[:K]                                   # 连续 1D i64
            i32 = i26.to(torch.int32)                          # 连续 1D i32
            strided = base64[::2][:K]                          # 非连续（stride 2）
            idx2d = i26.view(1, -1) if K else i26.view(1, 0)   # 2D (1,K)
            j = torch.randint(0, 4, (max(K, 1),), generator=g)[:K]
            idx_list = [i26]

            cases = {
                "getitem_1d_i64": lambda: x[i26],
                "getitem_1d_i32": lambda: x[i32],
                "getitem_1d_strided": lambda: x[strided],
                "getitem_2d_1xK": lambda: x[idx2d],
                "getitem_multi2": lambda: x2[i26, j],
                "opcall_aten": lambda: torch.ops.aten.index.Tensor(x, idx_list),
                "opcall_hpc": lambda: torch.ops.hpc_index.index(x, idx_list),
                "hpc_pywrap": lambda: hpc_index(x, idx_list),
                "empty_alloc": lambda: torch.empty(K),
            }
            for name, fn in cases.items():
                us = med_us(fn)
                rows.append({"mode": mode, "threads": t, "K": K, "case": name,
                             "us": round(us, 3)})
                print(f"{mode:5s} t={t:2d} K={K:5d} {name:20s} {us:8.3f} us",
                      file=sys.stderr, flush=True)
    return rows


def run(mode):
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    out = subprocess.check_output([sys.executable, os.path.abspath(__file__)],
                                  env=env, cwd=os.path.dirname(os.path.dirname(
                                      os.path.abspath(__file__))),
                                  stderr=subprocess.DEVNULL)
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="两进程交替并打印拆解表")
    ap.add_argument("--json", default="", help="把原始行写成 JSON")
    args = ap.parse_args()

    if not args.ab:
        print(json.dumps(measure(), ensure_ascii=False))
        return

    rows = run("aten") + run("hpc")
    if args.json:
        json.dump(rows, open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    table = {(r["mode"], r["threads"], r["K"], r["case"]): r["us"] for r in rows}
    cases = ["empty_alloc", "getitem_1d_i64", "getitem_1d_i32",
             "getitem_1d_strided", "getitem_2d_1xK", "getitem_multi2",
             "opcall_aten", "opcall_hpc", "hpc_pywrap"]
    print("| mode | t | K | " + " | ".join(cases) + " |")
    print("|---|---|---|" + "---|" * len(cases))
    for mode in ("aten", "hpc"):
        for t in THREADS:
            for K in KS:
                cells = " | ".join(f"{table[(mode, t, K, c)]:.2f}" for c in cases)
                print(f"| {mode} | {t} | {K} | {cells} |")

    print("\n# 拆解（µs，hpc 进程内）")
    for t in THREADS:
        for K in KS:
            total = table[("hpc", t, K, "getitem_1d_i64")]
            entry = table[("hpc", t, 0, "getitem_1d_i64")]
            direct = table[("hpc", t, K, "hpc_pywrap")]
            alloc = table[("hpc", t, K, "empty_alloc")]
            two = table[("hpc", t, K, "getitem_2d_1xK")]
            strided = table[("hpc", t, K, "getitem_1d_strided")]
            multi = table[("hpc", t, K, "getitem_multi2")]
            print(f"t={t:2d} K={K:5d}  全链路 {total:7.2f} = K=0 入口 {entry:5.2f} "
                  f"+ 内核/数据 {total - entry:5.2f}  |  Python 包装直呼 "
                  f"{direct:6.2f}（差 {direct - total:+5.2f}）  |  分配 {alloc:5.2f}  "
                  f"|  2D 下标 {two:6.2f}（+{two - total:5.2f}）  "
                  f"stride2 {strided:6.2f}（+{strided - total:5.2f}）  "
                  f"双索引 {multi:6.2f}（+{multi - total:5.2f}）")


if __name__ == "__main__":
    main()
