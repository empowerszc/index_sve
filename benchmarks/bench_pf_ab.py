"""H1（行拷贝软件预取）/ H6（预取记账批量化）的定向 A/B：只变旋钮，其余全不动。

`HPC_PF_DIST` / `HPC_PF_BATCH` 都是运行时旋钮（每次调用读一次），所以 hpc 侧可以
在**同一个进程**里把 D 和批大小扫完，不用为每档重启、也不用换 .so；aten 侧仍需
子进程（覆盖开关是进程级的）。

用法（node 5）：
  numactl --physcpubind=190-227 --membind=5 \
    python benchmarks/bench_pf_ab.py --a 3,16,64 --threads 1,8 --dist 0,4,8,16,24
  HPC_TOTAL_MB=64 python benchmarks/bench_pf_ab.py --a 16 --threads 8 --dist 0,8,16
  python benchmarks/bench_pf_ab.py --a 3,16,64 --threads 1,8 --dist -1   # -1 = 内核默认
  python benchmarks/bench_pf_ab.py --a 3,16 --threads 1,8 --dist 0,-1 --batch 0,4  # H6

输出列：`ns/行`（= 墙钟 / K，含入口开销，所以与 RESULTS §2.16.4 的探针口径
可以差一点，但同一进程内的 D/B 之间完全可比）、`vs D0`、`vs B0`、`aten/hpc`。
`vs B0` = 同 D 下逐行旧写法（batch 0）的 ns/行 ÷ 当前——H6 的 A/B 列。
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

TOTAL_MB = int(os.environ.get("HPC_TOTAL_MB", "8"))
AFFINITY = os.environ.get("HPC_AFFINITY", "190-227")
BATCH_ENV = "HPC_PF_BATCH"
DIST_ENV = "HPC_PF_DIST"


def med_us(fn, target_s=0.2):
    for _ in range(8):
        fn()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    per = (time.perf_counter() - t0) / 5
    iters = int(max(10, min(400, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def make_cases(a_list, thread_list):
    import torch
    g = torch.Generator().manual_seed(11)
    out = []
    total = TOTAL_MB << 20
    for A in a_list:
        K = max(4096, total // (A * 4))
        x = torch.randn(K * 2, A, generator=g)
        idx = torch.randint(0, K * 2, (K,), generator=g)
        for nt in thread_list:
            out.append(dict(A=A, K=K, nt=nt, x=x, idx=idx))
    return out


def _set_knob(env, v):
    # v < 0 表示"不设环境变量"，用内核默认值
    if v < 0:
        os.environ.pop(env, None)
    else:
        os.environ[env] = str(v)


def measure_hpc(a_list, thread_list, dists, batchs):
    import torch
    import index_sve  # noqa: F401
    rows = []
    for c in make_cases(a_list, thread_list):
        torch.set_num_threads(c["nt"])
        x, idx = c["x"], c["idx"]
        for b in batchs:
            _set_knob(BATCH_ENV, b)
            for d in dists:
                _set_knob(DIST_ENV, d)
                t = med_us(lambda: index_sve.index(x, [idx]))
                rows.append(dict(A=c["A"], K=c["K"], nt=c["nt"], dist=d,
                                 batch=b, us=round(t, 3)))
                print(f"hpc A={c['A']:3d} t={c['nt']:2d} D={d:3d} B={b:3d} {t:9.1f} us",
                      file=sys.stderr, flush=True)
    _set_knob(DIST_ENV, -1)
    _set_knob(BATCH_ENV, -1)
    return rows


def measure_aten(a_list, thread_list):
    import torch
    rows = []
    for c in make_cases(a_list, thread_list):
        torch.set_num_threads(c["nt"])
        x, idx = c["x"], c["idx"]
        t = med_us(lambda: x[idx])
        rows.append(dict(A=c["A"], K=c["K"], nt=c["nt"], us=round(t, 3)))
        print(f"aten A={c['A']:3d} t={c['nt']:2d} {t:9.1f} us",
              file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="3,16,64")
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--dist", default="0,4,8,16,24")
    ap.add_argument("--batch", default="0,4")
    ap.add_argument("--mode", choices=["hpc", "aten"])
    a = ap.parse_args()
    lo, hi = (AFFINITY.split("-") + [None])[:2]
    cpus = set(range(int(lo), int(hi) + 1)) if hi else {int(lo)}
    os.sched_setaffinity(0, cpus)
    a_list = [int(v) for v in a.a.split(",") if v]
    thread_list = [int(v) for v in a.threads.split(",") if v]
    dists = [int(v) for v in a.dist.split(",") if v]
    batchs = [int(v) for v in a.batch.split(",") if v]

    if a.mode:
        import json
        fn = measure_hpc if a.mode == "hpc" else measure_aten
        args = ((a_list, thread_list, dists, batchs) if a.mode == "hpc"
                else (a_list, thread_list))
        print(json.dumps(fn(*args)))
        return

    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1"
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", "aten",
                        "--a", a.a, "--threads", a.threads], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"aten 子进程失败：\n{p.stderr[-2000:]}")
    import json
    aten = {(r["A"], r["nt"]): r["us"] for r in json.loads(p.stdout)}
    hpc = measure_hpc(a_list, thread_list, dists, batchs)

    print(f"\n{TOTAL_MB}MB 输出（源 = 2×）、ns/行 = µs*1000/K、D/B=-1 表示内核默认；"
          f"vs D0 = 同 batch 下 D0 ÷ 当前、vs B0 = 同 D 下 batch0 ÷ 当前；aten/hpc 用 aten 列")
    print(f"{'A':>4s} {'行B':>5s} {'t':>3s} {'K':>7s} {'D':>4s} {'B':>3s} {'ns/行':>8s} "
          f"{'vs D0':>6s} {'vs B0':>6s} {'aten/hpc':>9s} {'有用GB/s':>9s}")
    base_d0, base_b0 = {}, {}
    for r in sorted(hpc, key=lambda r: (r["A"], r["nt"], r["dist"], r["batch"])):
        key = (r["A"], r["nt"])
        npr = r["us"] * 1000.0 / r["K"]
        if r["dist"] == 0:
            base_d0[key + (r["batch"],)] = npr
        if r["batch"] == 0:
            base_b0[key + (r["dist"],)] = npr
        d0 = base_d0.get(key + (r["batch"],), npr)
        b0 = base_b0.get(key + (r["dist"],), npr)
        at = aten[key]
        gb = (2 * r["K"] * r["A"] * 4) / (r["us"] * 1e-6) / 1e9
        print(f"{r['A']:4d} {r['A'] * 4:5d} {r['nt']:3d} {r['K']:7d} {r['dist']:4d} "
              f"{r['batch']:3d} {npr:8.2f} {d0 / npr:6.2f} {b0 / npr:6.2f} "
              f"{at / r['us']:9.2f} {gb:9.2f}")


if __name__ == "__main__":
    main()
