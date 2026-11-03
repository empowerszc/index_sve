"""上下文开销拆解：把"模型内单次 index 比热循环基准慢"拆成可控实验。

`bench_entry_overhead.py` 量的是**热循环**里的入口链（K=0 外推 ~6.4µs）。但模型里
每次调用之间隔着别的算子，线程池会睡下去、源数据也离开 cache。本脚本按上下文
分类，量化这些差异到底有多大：

- `hot`：背靠背重复调用（基准口径，最乐观）；
- `spaced`：每次调用前睡 200µs，让 ATen/libgomp 线程池退出自旋（模型里的真实
  状态），测 fork-join 的真实代价；
- `cold_src`：在 192 个互不相邻的源张量间轮转，每次调用的随机行都是 DRAM 冷访问；
- `autograd`：`requires_grad=True` 的源，每次调用建一个 autograd 节点。

`chain` 组固定 K=1（输出只有一个元素），把"每次调用必须付的固定成本"单独拎出来，
再对比几种"K 很小但固定成本可能很大"的形态：非连续源/下标、2D 下标、双索引、
按 self 元素数计成本的一维/二维 mask（mask 路径在 normalize 里做 `nonzero()`）。

两个进程各测一遍（覆盖开关是进程级），比值仅供参考——本脚本主要看**绝对 µs**。

用法：
  HPC_AFFINITY=190-227 HPC_THREADS=1,6,12,19 \
    python benchmarks/bench_context_overhead.py --ab
  # 单进程原始 JSON（HPC_INDEX_NO_OVERRIDE=1 得到 aten 侧）
  python benchmarks/bench_context_overhead.py
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
THREADS = [int(t) for t in os.environ.get("HPC_THREADS", "1,6,12,19").split(",")]
CTX_KS = [int(k) for k in os.environ.get("HPC_CTX_KS", "512,2048,8192").split(",")]
SPACED_S = float(os.environ.get("HPC_SPACED_S", "0.0002"))
NSRC = 192          # cold_src 的源个数（192 x 4MB = 768MB，远大于单 node 的 L3）


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


def med_us(fn, gap=None, iters=0, target_s=0.10):
    for _ in range(20):
        fn()
    if iters <= 0:
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        per = (time.perf_counter() - t0) / 5
        iters = int(max(30, min(400, target_s / max(per, 1e-8))))
    ts = []
    for _ in range(iters):
        if gap is not None:
            gap()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def measure():
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import index_sve  # noqa: F401  (是否安装覆盖由 HPC_INDEX_NO_OVERRIDE 决定)

    os.sched_setaffinity(0, _affinity())
    omp = _omp()
    mode = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
    g = torch.Generator().manual_seed(7)
    rows = []

    def rec(group, name, threads, us, extra=None):
        row = {"mode": mode, "group": group, "case": name, "threads": threads,
               "us": round(us, 3)}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f"{mode:5s} t={threads:2d} {group:5s} {name:34s} {us:9.2f} us",
              file=sys.stderr, flush=True)

    def set_threads(t):
        torch.set_num_threads(t)
        if omp is not None:
            omp.omp_set_num_threads(t)

    for t in THREADS:
        set_threads(t)

        # ---------------- 上下文组：同一形态、不同调用环境 ----------------
        for K in CTX_KS:
            idx = torch.randint(0, 1 << 20, (K,), generator=g)
            src = torch.randn(1 << 20, generator=g)
            src_g = src.clone().requires_grad_(True)

            # 顺序要紧：hot/autograd 必须在分配 192 个源（768MB）之前测，否则
            # 大堆残留会让 aten 的多线程热循环异常退化（实测 12t 572µs vs 32µs）。
            rec("ctx", f"hot_K{K}", t, med_us(lambda: src[idx]), {"K": K})
            rec("ctx", f"autograd_K{K}", t, med_us(lambda: src_g[idx]), {"K": K})

            srcs = [torch.randn(1 << 20, generator=g) for _ in range(NSRC)]
            cyc = [0]

            def cold_src():
                cyc[0] = (cyc[0] + 37) % NSRC
                return srcs[cyc[0]][idx]

            rec("ctx", f"cold_src_K{K}", t, med_us(cold_src), {"K": K})
            del srcs, src_g, cyc
            rec("ctx", f"spaced_K{K}", t,
                med_us(lambda: src[idx], gap=lambda: time.sleep(SPACED_S)), {"K": K})

        # ---------------- 固定成本组：输出只有 1 个元素 ----------------
        x1 = torch.randn(1 << 20, generator=g)
        x1nc = x1[::2]                                   # 非连续源
        x2 = torch.randn(1 << 20, 4, generator=g)
        i1 = torch.randint(0, 1 << 20, (1,), generator=g)
        i512 = torch.randint(0, 1 << 20, (512,), generator=g)
        j512 = torch.randint(0, 4, (512,), generator=g)
        i512nc = torch.randint(0, 1 << 20, (1024,), generator=g)[::2]
        i512_2d = i512.view(1, 512)
        i512_half = torch.randint(0, 1 << 19, (512,), generator=g)

        mask_sparse = torch.zeros(1 << 20, dtype=torch.bool)
        mask_sparse[torch.randperm(1 << 20, generator=g)[:64]] = True
        mask_dense = torch.rand(1 << 20, generator=g) < 0.5
        mask2d = torch.zeros(1 << 20, 4, dtype=torch.bool)
        mask2d[:, 0] = mask_dense

        chain = {
            "chain_K1": lambda: x1[i1],
            "tensor_1d_K512": lambda: x1[i512],
            "idx_noncontig_K512": lambda: x1[i512nc],
            "idx_2d_1xK512": lambda: x1[i512_2d],
            "self_noncontig_K512": lambda: x1nc[i512_half],
            "double_idx_K512": lambda: x2[i512, j512],
            "mask1d_sparse64": lambda: x1[mask_sparse],
            "mask1d_dense_1m": lambda: x1[mask_dense],
            "mask2d_dense": lambda: x2[mask2d],
        }
        for name, fn in chain.items():
            rec("chain", name, t, med_us(fn))

        # ---------------- 形态组：A 敏感性 / 纯固定开销小形态 / mask 的 normalize 代理 ----------------
        # 行数取 200k，A=256 时源 205MB，避免把内存压力混进来。
        N = 200_000
        for A in (3, 16, 64, 256):
            xa = torch.randn(N, A, generator=g).contiguous()
            ia = torch.randint(0, N, (512,), generator=g)
            rec("forms", f"K512_A{A}", t, med_us(lambda: xa[ia]), {"A": A})
            if A in (64, 256):
                rec("forms", f"K2_A{A}", t, med_us(lambda: xa[ia[:2]]), {"A": A})
            del xa
        # mask：normalize 里的 nonzero 是唯一"与 K 无关但按 self.numel 计费"的阶段，
        # 单独测 nonzero 与整条 mask 路径，差额即压实+分配。
        m_sparse = torch.zeros(1 << 20, dtype=torch.bool)
        m_sparse[torch.randperm(1 << 20, generator=g)[:64]] = True
        m_dense = torch.rand(1 << 20, generator=g) < 0.5
        xm = torch.randn(1 << 20, generator=g)
        rec("forms", "nonzero_1m_sparse64", t, med_us(lambda: m_sparse.nonzero()),
            {"nnz": 64})
        rec("forms", "nonzero_1m_dense", t, med_us(lambda: m_dense.nonzero()),
            {"nnz": 1 << 19})
        rec("forms", "mask_index_1m_sparse64", t, med_us(lambda: xm[m_sparse]),
            {"nnz": 64})
        del m_sparse, m_dense, xm

        # ---------------- 混合组：单调用平均值能不能被"形态分布"解释 ----------------
        # 1000 次调用里 88% 是小 K、8% 中 K、3%+1% 是 rowcopy。逐形态热耗时是该序列
        # 的预测值；实测 = 整段总时间 / 调用数。两者接近 → 平均值由形态分布决定，
        # 不需要"每次调用 170µs 的隐藏固定开销"。
        mix_spec = [("bucket_K512", 880, dict(K=512, A=1, dom=1024)),
                    ("gather_K13824_A1", 80, dict(K=13824, A=1, dom=1 << 20)),
                    ("rowcopy_K8192_A64", 30, dict(K=8192, A=64, dom=1 << 20)),
                    ("rowcopy_K55296_A64", 10, dict(K=55296, A=64, dom=1 << 20))]
        calls, weights, pred, per_shape = [], [], 0.0, {}
        for name, w, sp in mix_spec:
            if sp["A"] == 1:
                src = torch.randn(1 << 20, generator=g)
            else:
                src = torch.randn(1 << 20, sp["A"], generator=g).contiguous()
            idx = torch.randint(0, sp["dom"], (sp["K"],), generator=g)
            us = med_us(lambda: src[idx])
            per_shape[name] = us
            pred += w * us
            weights.append(w)
            calls.extend([lambda s=src, i=idx: s[i]] * w)
            rec("mix", f"shape_{name}", t, us, {"weight": w})
        pred /= sum(weights)
        # 打散顺序，避免同类调用连成块（那会变成另一种"热循环"）
        step = 617 % len(calls)
        order = [calls[(i * step) % len(calls)] for i in range(len(calls))]
        order = order[:len(calls)]
        seq = []
        for _ in range(12):
            t0 = time.perf_counter()
            for fn in order:
                fn()
            seq.append(time.perf_counter() - t0)
        avg = statistics.median(seq) / len(order) * 1e6
        rec("mix", "sequence_avg", t, avg, {"predicted": round(pred, 2)})
        print(f"{mode:5s} t={t:2d} mix   预测 {pred:8.2f} us  实测 {avg:8.2f} us  "
              f"比值 {avg / pred:.2f}", file=sys.stderr, flush=True)

    return rows


def run(mode):
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1" if mode == "aten" else "0"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.check_output([sys.executable, os.path.abspath(__file__)],
                                  env=env, cwd=root, stderr=subprocess.DEVNULL)
    return json.loads(out)


def report(rows, json_path=""):
    if json_path:
        json.dump(rows, open(json_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    d = {(r["mode"], r["group"], r["case"], r["threads"]): r["us"] for r in rows}
    cases = []
    for r in rows:
        if (r["group"], r["case"]) not in cases:
            cases.append((r["group"], r["case"]))

    for group in ("ctx", "chain", "forms", "mix"):
        print(f"\n### {group}（µs，绝对值；比值 = aten ÷ hpc）\n")
        print("| case | t | aten µs | hpc µs | aten/hpc |")
        print("|---|---|---|---|---|")
        for grp, case in cases:
            if grp != group:
                continue
            for t in THREADS:
                a = d.get(("aten", grp, case, t))
                h = d.get(("hpc", grp, case, t))
                if a is None:
                    continue
                print(f"| {case} | {t} | {a:.2f} | {h:.2f} | "
                      f"{(a / h if h else 0):.2f}x |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", action="store_true", help="两进程各测一遍并出表")
    ap.add_argument("--json", default="", help="原始行写成 JSON")
    args = ap.parse_args()

    if not args.ab:
        print(json.dumps(measure(), ensure_ascii=False))
        return

    rows = run("aten") + run("hpc")
    report(rows, args.json)


if __name__ == "__main__":
    main()
