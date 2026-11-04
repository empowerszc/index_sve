#!/usr/bin/env python3
"""量 CPU 封装内内存（无 CPU 的 NUMA node）的带宽结构，不需要 root。

用法: python3 tools/probe-onpkg-bw.py [--quick]
      --quick 只跑 7/8/9（DDRC 反证、四堵墙、UC 归属），约 1 分钟；
      完整跑还包含 1-6（距离阶梯、14 窗口扫描、聚合、集中对照、线程、缓冲对照）。

口径：memwin 固定时间窗口（[start, start+sec) 内反复读自己的缓冲区），
每进程绑自己的 32 核；1GB 缓冲是为了跑满窗口内部的地址交织。
"""
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = "/sys/devices/system/node"
DEV = "/sys/bus/event_source/devices"


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def memfree(node):
    try:
        for line in open(f"{NODE}/node{node}/meminfo"):
            if line.startswith("Node") and "MemFree" in line:
                return int(line.split()[3]) // 1024
    except OSError:
        pass
    return 0


def distance(a, b):
    try:
        return int(open(f"{NODE}/node{a}/distance").read().split()[b])
    except OSError:
        return None


def topology():
    cpus, sizes = {}, {}
    for line in sh(["numactl", "-H"]).splitlines():
        m = re.match(r"node (\d+) cpus: (.*)", line)
        if m:
            v = [int(x) for x in m.group(2).split()]
            if v:
                cpus[int(m.group(1))] = v
        m = re.match(r"node (\d+) size: (\d+) MB", line)
        if m:
            sizes[int(m.group(1))] = int(m.group(2))
    return cpus, sizes


class Probe:
    def __init__(self, tmp):
        self.bin = os.path.join(tmp, "bin")
        os.makedirs(self.bin, exist_ok=True)
        for src, name in (("benchmarks/memwin.c", "memwin"),
                          ("benchmarks/memprobe.c", "memprobe")):
            subprocess.run(["gcc", "-O2", "-fopenmp", "-o", os.path.join(self.bin, name),
                            os.path.join(REPO, src)], check=True)
        self.memwin = os.path.join(self.bin, "memwin")

    def run(self, node, cpus, mb, sec, nt, reps=1):
        vals = []
        for _ in range(reps):
            start = time.time() + 1.0
            out = subprocess.run(
                ["numactl", f"--physcpubind={cpus}", f"--membind={node}", self.memwin,
                 str(mb), str(sec), f"{start:.3f}", str(nt)],
                capture_output=True, text=True)
            if out.returncode != 0:
                return None
            vals.append(int(out.stdout.strip()))
        return [v / sec / 1e9 for v in vals]

    def agg(self, specs, mb, sec, nt):
        start = time.time() + 3.0
        procs = [subprocess.Popen(
            ["numactl", f"--physcpubind={c}", f"--membind={n}", self.memwin,
             str(mb), str(sec), f"{start:.3f}", str(nt)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for n, c in specs]
        tot, ok = 0.0, True
        for p in procs:
            out, _ = p.communicate()
            if p.returncode != 0 or not out.strip():
                ok = False
                continue
            tot += int(out.strip())
        return tot / sec / 1e9 if ok else None

    def perf(self, events, sec, specs, mb=1024):
        """在 perf 窗口里跑一段负载，返回 {event: 计数}。"""
        cmd = ["perf", "stat", "-a"]
        for e in events:
            cmd += ["-e", e]
        cmd += ["--", "sleep", str(sec + 2.0)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.5)
        bw = self.agg(specs, mb, sec, 32)
        txt = p.communicate()[0]
        counts = {}
        for line in txt.splitlines():
            m = re.match(r"\s*([\d,]+)\s+(\S+)/(\w+)/", line)
            if m:
                counts[(m.group(2), m.group(3))] = int(m.group(1).replace(",", ""))
        return counts, bw


def core_slice(cpus, n=32):
    return f"{cpus[0]}-{cpus[0] + n - 1}"


def main():
    quick = "--quick" in sys.argv
    cpus, sizes = topology()
    cpu_nodes = sorted(cpus)
    mem_nodes = [n for n in sorted(sizes) if n not in cpus]
    if not mem_nodes:
        print("本机没有无 CPU 的 node，退出")
        return 1

    pair = {}
    for w in mem_nodes:
        cand = [(distance(c, w), c) for c in cpu_nodes if distance(c, w) is not None]
        pair[w] = min(cand)[1]

    usable = [n for n in mem_nodes if memfree(n) >= 1500]
    print(f"# CPU node {len(cpu_nodes)} 个、无 CPU node {len(mem_nodes)} 个，"
          f"其中可用（空闲 ≥1.5GB）{len(usable)} 个；每个无 CPU node 的最近 CPU node：")
    for w in mem_nodes:
        print(f"  node{w:<3} → node{pair[w]:<3} 距离 {distance(pair[w], w)}  "
              f"核 {cpus[pair[w]][0]}-{cpus[pair[w]][-1]}")

    tmp = tempfile.mkdtemp(prefix="onpkg-", dir="/tmp/opencode")
    p = Probe(tmp)
    ws = usable

    if not quick:
        print("\n## 1. 距离阶梯：同一个窗口，换不同距离的核去打")
        w = usable[-1]
        by_dist = {}
        for c in cpu_nodes:
            by_dist.setdefault(distance(c, w), c)
        for d in sorted(by_dist):
            c = by_dist[d]
            v = p.run(w, core_slice(cpus[c]), 1024, 2, 32, 2)
            print(f"  node{c:<3}（距离 {d:<2}）→ node{w}: "
                  + ("分配失败" if v is None else f"{min(v):.1f}-{max(v):.1f} GB/s"))

        print("\n## 2. 每个窗口单测（配对核、1GB、32 线程、两次）")
        singles = {}
        for w2 in mem_nodes:
            if memfree(w2) < 1500:
                print(f"  node{w2:<3} 空闲 {memfree(w2)}MB，跳过")
                continue
            v = p.run(w2, core_slice(cpus[pair[w2]]), 1024, 2, 32, 2)
            if v is None:
                print(f"  node{w2:<3} 分配失败，跳过")
                continue
            singles[w2] = v
            print(f"  node{w2:<3}（node{pair[w2]:<3} 的核）{min(v):6.1f}-{max(v):6.1f} GB/s")
        if not singles:
            return 1
        ws = sorted(singles)

        print("\n## 3. 聚合：多个窗口同时压（每个用自己的配对核）")
        cpu0 = [x for x in ws if x < 24]
        cpu1 = [x for x in ws if x >= 24]
        groups = [("1 个", ws[:1]), ("2 个", ws[:2]), ("4 个", ws[:4]),
                  ("socket0 七窗口", cpu0[:7]), ("socket1 七窗口", cpu1[:7]),
                  ("全部", ws)]
        for label, pick in groups:
            v = p.agg([(x, core_slice(cpus[pair[x]])) for x in pick], 1024, 2, 32)
            print(f"  {label:<16}{pick if len(pick) <= 8 else '(共 %d)' % len(pick)} → "
                  + ("失败" if v is None else f"{v:.1f} GB/s"))

        print("\n## 4. 集中 vs 摊开（同样 7×32 线程、每进程 512MB）")
        ws7 = [x for x in ws if x >= 24][:7]
        spread = p.agg([(x, core_slice(cpus[pair[x]])) for x in ws7], 512, 2, 32)
        if spread:
            lo = cpus[pair[ws7[0]]][0]
            conc = p.agg([(ws7[0], f"{lo + 32 * i}-{lo + 32 * i + 31}") for i in range(7)],
                         512, 2, 32)
            print(f"  摊到 {ws7} → {spread:.1f} GB/s")
            print(f"  全压 node{ws7[0]}（同样 7×32 线程）→ "
                  + ("失败（窗口只有 4GB，7×512MB 可能放不下）" if conc is None
                     else f"{conc:.1f} GB/s，摊开是它的 {spread / conc:.1f} 倍"))

        print("\n## 5. 线程扫描（单窗口，配对核整段）")
        w = ws[-1]
        for nt in (16, 24, 32, 38, 48, 76):
            v = p.run(w, f"{cpus[pair[w]][0]}-{cpus[pair[w]][-1]}", 1024, 2, nt, 1)
            print(f"  node{w} {nt:2d} 线程 → " + ("失败" if v is None else f"{v[0]:.1f} GB/s"))

        print("\n## 6. 方法对照：同一窗口，128MB vs 1GB 缓冲（各 4 次）")
        w = ws[0]
        for mb in (128, 1024):
            v = p.run(w, core_slice(cpus[pair[w]]), mb, 2, 32, 4)
            if v is None:
                continue
            print(f"  {mb:4d}MB: " + " ".join(f"{x:6.1f}" for x in v)
                  + f"   max/min={max(v) / min(v):.2f}")

    print("\n## 7. DDRC 反证：压封装内内存时，主内存通道计数器看到了什么")
    w = ws[-1]
    devs = [d for d in os.listdir(DEV) if "_ddrc" in d]
    counts, bw = p.perf([f"{d}/flux_rd/" for d in devs], 3,
                        [(w, core_slice(cpus[pair[w]]))])
    if bw:
        tot = sum(counts.values())
        print(f"  负载：node{w} {bw:.1f} GB/s；{len(devs)} 个 DDRC 的 flux_rd 合计 {tot:,}")
        print(f"  按 32 字节/事务折算 ≈ {tot * 32 / 3 / 1e9:.2f} GB/s —— "
              f"相比 {bw:.0f} GB/s 可忽略 → 封装内内存不经过 DDR5 通道")

    print('\n## 8. 四堵墙：换"核站在哪"，看多路能不能叠加')
    w = ws[-1]
    grp = pair[w] - pair[w] % 4                       # 与该窗口同 SCCL 的 4 个 CPU node
    others = [c for c in range(grp, grp + 4) if c != pair[w]]
    same_socket = [c for c in cpu_nodes if (c >= 8) == (pair[w] >= 8)]
    other_socket = [c for c in cpu_nodes if (c >= 8) != (pair[w] >= 8)]
    other_sccl = [c for c in same_socket if c // 4 != grp // 4]
    cases = [
        ("配对（61）", core_slice(cpus[pair[w]])),
        ("同 SCCL 非配对（71）", core_slice(cpus[others[0]])),
        ("跨 SCCL（81）", core_slice(cpus[other_sccl[0]])),
        ("跨 socket（91）", core_slice(cpus[other_socket[0]])),
    ]
    for label, cores in cases:
        v = p.agg([(w, cores)], 1024, 2, 32)
        print(f"  {label:<20}单路 → " + ("失败" if v is None else f"{v:6.1f} GB/s"))
    windows = [x for x in ws if grp <= x - 16 < grp + 4]
    combos = [
        ("配对 ×%d 窗口" % len(windows),
         [(x, core_slice(cpus[pair[x]])) for x in windows]),
        ("同 SCCL 非配对 ×%d 路" % len(windows),
         [(x, core_slice(cpus[others[(i + 1) % len(others)]]))
          for i, x in enumerate(windows)]),
        ("跨 SCCL ×%d 路" % len(windows),
         [(x, core_slice(cpus[other_sccl[i % len(other_sccl)]]))
          for i, x in enumerate(windows)]),
        ("跨 socket ×%d 路" % len(windows),
         [(x, core_slice(cpus[other_socket[i % len(other_socket)]]))
          for i, x in enumerate(windows)]),
    ]
    for label, specs in combos:
        v = p.agg(specs, 1024, 2, 32)
        print(f"  {label:<20}同时 → " + ("失败" if v is None else f"{v:6.1f} GB/s"))
    label = "同一个窗口 ×%d 个源" % len(others)
    v = p.agg([(w, core_slice(cpus[others[i]])) for i in range(len(others))], 1024, 2, 32)
    print(f"  {label:<20}同时 → "
          + ("失败" if v is None else f"{v:6.1f} GB/s（窗口端口上限）"))

    print("\n## 9. 哪个 UC 在服务这个窗口（perf 事件 cpu_rd）")
    first = cpus[pair[w]][0]
    ucs = [d for d in os.listdir(DEV) if "_uc" in d and d.endswith("_0")]
    ucs = [d for d in ucs if open(f"{DEV}/{d}/cpumask").read().strip() != ""]
    mine = [d for d in ucs if (int(open(f"{DEV}/{d}/cpumask").read().split()[0]) < 304)
            == (first < 304)]
    counts, bw = p.perf([f"{d}/cpu_rd/" for d in mine], 2,
                        [(w, core_slice(cpus[pair[w]]))])
    if bw:
        print(f"  负载：node{w} {bw:.1f} GB/s（压的是 node{pair[w]} 的配对窗口）")
        for d in mine:
            c = counts.get((d, "cpu_rd"), 0)
            print(f"    {d:<24}{c:>16,}" + ("   ← 就是它" if c > 1e8 else ""))
        print("  只有配对的那个 UC 大幅计数；同 SCCL 的另外三个、另一个 SCCL 的都只有背景")
    return 0


if __name__ == "__main__":
    sys.exit(main())
