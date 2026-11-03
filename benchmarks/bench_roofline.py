"""把每个形态的实测时间换算成"离硬件上限还有多远"。

三列对照，同一份 shape、同一份数据：

- `aten`：`x[idx]` 走 aten 原生实现（子进程里 `HPC_INDEX_NO_OVERRIDE=1`）。
- `sel`：`torch.index_select(x, 0, idx)`——aten 自己的"无 gather 机制"行拷贝。
  它和 `aten` 的差 = aten 索引机制本身的成本；它同时是一个**同库内可达的参考**：
  如果 `hpc` 比 `sel` 还慢，说明慢在我们这边，不是"内存墙"。
- `hpc`：`x[idx]` 走本扩展。

覆盖开关是进程级的（同进程内切不了），所以 aten/hpc 必须分两个子进程跑；
`sel` 是纯 aten op，两个进程里都一样，评分取 aten 进程那份。

字节口径（写进表头，别混）：
  有用字节 = K*A*esize 读 + 同量写 + K*idx_bytes
  line 读字节 = K*ceil(A*esize/64)*64 —— 随机行之间不共享 line 的保守假设；
  A*esize >= 64 时两者相等（含 64B 内尾块浪费）。

用法（node 5）：
  numactl --physcpubind=190-227 --membind=5 python benchmarks/bench_roofline.py --sweep-a
  numactl --physcpubind=190-227 --membind=5 python benchmarks/bench_roofline.py --stream
  HPC_THREADS=1,8,16,36 HPC_A_LIST=1,3,16,64 python benchmarks/bench_roofline.py --sweep-a
  python benchmarks/bench_roofline.py --sweep-a --mode aten --json   # 单进程原始数据（调试用）
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

# node 5 实测流式上限（RESULTS §2.16）：读 ~168GB/s（16t 起平台）、拷贝 ~145GB/s。
# 表里只打印"占这些上限的百分比"，上限本身的测法见 RESULTS §2.16。
READ_BW = {1: 19.4, 2: 37.8, 4: 71.4, 8: 127.4, 16: 169.2, 32: 168.5, 38: 168.0}
COPY_BW = {1: 28.9, 2: 56.5, 4: 104.4, 8: 132.3, 16: 145.0, 32: 141.7, 38: 116.3}

THREAD_LIST = [int(t) for t in os.environ.get("HPC_THREADS", "1,8,16").split(",")]
TOTAL_MB = int(os.environ.get("HPC_TOTAL_MB", "8"))
A_LIST = [int(a) for a in os.environ.get("HPC_A_LIST",
                                        "1,2,3,4,8,16,32,64,128,256").split(",")]


def med_us(fn, target_s=0.15):
    for _ in range(10):
        fn()
    t0 = time.perf_counter()
    for _ in range(5):
        fn()
    per = (time.perf_counter() - t0) / 5
    iters = int(max(10, min(300, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def _nearest(table, nt):
    return table[min(table, key=lambda k: abs(k - nt))]


def cases():
    import torch
    g = torch.Generator().manual_seed(11)
    out = []
    total = TOTAL_MB << 20
    for A in A_LIST:
        for nt in THREAD_LIST:
            K = max(1024, total // (A * 4))
            if A == 1:
                x = torch.randn(K * 2, generator=g)
            else:
                x = torch.randn(K * 2, A, generator=g)
            idx = torch.randint(0, K * 2, (K,), generator=g)
            out.append(dict(A=A, nt=nt, K=K, x=x, idx=idx))
    return out


def measure(mode):
    import torch
    import index_sve  # noqa: F401
    rows = []
    for c in cases():
        torch.set_num_threads(c["nt"])
        x, idx = c["x"], c["idx"]
        if mode == "hpc":
            t = med_us(lambda: index_sve.index(x, [idx]))
        else:
            t = med_us(lambda: x[idx])
        t_sel = med_us(lambda: torch.index_select(x, 0, idx))
        rows.append({k: c[k] for k in ("A", "nt", "K")} |
                    {"mode": mode, "us": round(t, 3), "sel_us": round(t_sel, 3)})
        print(f"{mode:4s} A={c['A']:3d} t={c['nt']:2d} {t:9.1f} us", file=sys.stderr,
              flush=True)
    return rows


def one_round():
    out = {}
    for mode in ("aten", "hpc"):
        env = dict(os.environ)
        env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--mode", mode, "--json"], env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"{mode} 子进程失败：\n{p.stderr[-2000:]}")
        for r in json.loads(p.stdout):
            out[(r["mode"], r["A"], r["nt"])] = r
    return out


def run_ab():
    r = one_round()
    for (mode, A, nt), row in sorted(r.items()):
        if mode != "hpc":
            continue
        a = r[("aten", A, nt)]
        row["aten_us"] = a["us"]
        row["sel_us"] = a["sel_us"]
    return [v for k, v in sorted(r.items()) if k[0] == "hpc"]


def report(rows):
    print(f"\n{'A':>4s} {'t':>3s} {'K':>8s} {'aten':>9s} {'sel':>9s} {'hpc':>9s} "
          f"{'aten/hpc':>8s} {'sel/hpc':>8s} {'有用GB/s':>9s} {'lineGB/s':>9s} "
          f"{'占拷贝':>7s}")
    for r in rows:
        A, K, nt = r["A"], r["K"], r["nt"]
        rowb = A * 4
        useful = 2 * K * rowb + K * 8
        lines = K * ((rowb + 63) // 64) * 64
        bw = useful / (r["us"] * 1e-6) / 1e9
        lbw = lines / (r["us"] * 1e-6) / 1e9
        cap = _nearest(COPY_BW, nt)
        print(f"{A:4d} {nt:3d} {K:8d} {r['aten_us']:9.1f} {r['sel_us']:9.1f} "
              f"{r['us']:9.1f} {r['aten_us'] / r['us']:8.2f} "
              f"{r['sel_us'] / r['us']:8.2f} {bw:9.2f} {lbw:9.1f} "
              f"{100 * bw / cap:6.0f}%")


def stream(nt_list):
    import torch
    import index_sve  # noqa: F401
    print(f"\n== 连续拷贝对照（{TOTAL_MB}MB，无索引）")
    for nt in nt_list:
        torch.set_num_threads(nt)
        x = torch.randn(TOTAL_MB << 18)
        t = med_us(lambda: x.clone())
        bw = 2 * x.numel() * 4 / (t * 1e-6) / 1e9
        print(f"t={nt:3d} clone {t:9.1f} us  读+写 {bw:6.1f} GB/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["aten", "hpc"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sweep-a", action="store_true")
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    os.sched_setaffinity(0, set(range(190, 228)))
    if a.mode:
        print(json.dumps(measure(a.mode), ensure_ascii=False))
        return
    if not (a.sweep_a or a.stream):
        a.sweep_a = a.stream = True
    if a.sweep_a:
        report(run_ab())
    if a.stream:
        stream(THREAD_LIST)


if __name__ == "__main__":
    main()
