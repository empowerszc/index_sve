"""机器健康金丝雀（RESULTS §2.22）：纯 torch，不 import 本仓。

背景：2026-11-03 的机器级慢态对本仓代码、libtorch、CPython 一视同仁地罚 2-3x，
但 bucketprobe/chase/idxgather 等 C 探针完全看不见。唯一敏感的金丝雀是
「Python 进程里的 torch 调用」——本脚本只做这件事。

判快态（node 5 / 12t，健康基线见 §1.1、§2.12.1）：
    empty512          ≈ 4.3 µs
    aten x[idx] K512  ≈ 15.5 µs
    aten x[idx] K2048 ≈ 21.8 µs
任一项超过基线 1.5x 即判慢态，当天基准全部作废，等机器恢复再测。
add512 无健康基线，仅看趋势。
"""

import time

import torch


def bench(fn, warm=30, reps=40):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1024, generator=g)
    x2 = torch.randn(2048, generator=g)
    i512 = torch.randint(0, 1024, (512,), generator=g)
    i2048 = torch.randint(0, 2048, (2048,), generator=g)
    a = torch.randn(512, generator=g)
    b = torch.randn(512, generator=g)
    torch.set_num_threads(12)
    print("canary mode=aten (本仓未 import，x[idx] 走原生 aten)")
    print("empty512        %6.2f us (健康 ≈4.3)" % bench(lambda: torch.empty(512)))
    print("x[idx] K512     %6.2f us (健康 ≈15.5)" % bench(lambda: x[i512]))
    print("x[idx] K2048    %6.2f us (健康 ≈21.8)" % bench(lambda: x2[i2048]))
    print("add512          %6.2f us (无基线，看趋势)" % bench(lambda: a + b))


if __name__ == "__main__":
    main()
