"""宿主多线程并发调用 x[idx] 的超订实验（integration.md §6.1）。

问题：宿主程序已有 k 个线程并发调本算子时，每次调用仍按 torch.set_num_threads(nt)
拉满 OpenMP team → k*nt 线程压核 → 超订劣化。本脚本固定总工作量，对比
"单线程顺序"与"k 线程并发"的墙钟，扫 NT 展示守则"宿主线程数 × nt ≈ 核数"。

用法（单进程，环境变量参数；建议 taskset 定点单节点，如 -c 190-227）：
  NT=36 K=4 CALLS=24 python bench_embed_threads.py            # 超订形态
  NT=9  K=4 CALLS=24 python bench_embed_threads.py            # 公平形态
  HPC_INDEX_NO_OVERRIDE=1 NT=36 K=4 python bench_embed_threads.py  # aten 对照
"""
import torch
import time
import threading
import os

NT = int(os.environ.get("NT", "36"))            # torch 线程数（内核内并行）
K = int(os.environ.get("K", "4"))               # 宿主并发线程数
CALLS = int(os.environ.get("CALLS", "24"))      # 总调用数（两模式工作量相同）
torch.set_num_threads(NT)

x = torch.randn(50000, 64)                       # f32 行取数（raw rowcopy 路径）
idx = torch.randint(0, 50000, (200000,), dtype=torch.int64)


def work(n):
    y = x
    for _ in range(n):
        y = x[idx]
    return y


for _ in range(3):
    work(2)  # warmup

t0 = time.perf_counter()
work(CALLS)
t1 = time.perf_counter()
single = t1 - t0

per = CALLS // K
ths = [threading.Thread(target=work, args=(per,)) for _ in range(K)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
t1 = time.perf_counter()
multi = t1 - t0

side = "aten" if os.environ.get("HPC_INDEX_NO_OVERRIDE") == "1" else "hpc"
print(f"[{side}] nt={NT:2d} host_threads={K}: single={single * 1000:7.1f}ms "
      f"multi{K}={multi * 1000:7.1f}ms multi/single={multi / single:5.2f}x")
