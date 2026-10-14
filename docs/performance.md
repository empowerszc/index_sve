# 性能基准

完整报告见 `benchmarks/RESULTS.md`（§1 原始矩阵、§2 全矩阵 6 轮中位数、§3 NT
diff、§4 聚焦门禁、§5 结论、§6 备注、§7 集成、**§8 定点复测**）。本文档是
方法论与结论摘要。

## 1. 环境

- 机器：HiSilicon 920F，608 核 / 32 NUMA（每节点 38 核），SVE2 VL=512-bit，
  **共享箱（5+ 用户，噪声大）**。
- 工具链：BiSheng clang 19.1.7，
  `-march=armv8-a+sve2 -msve-vector-bits=512 -O3 -std=c++17 -fopenmp`。
- torch 2.10.0+cpu。基准用 `HPC_INDEX_NO_OVERRIDE=1`（见 integration.md）。

## 2. 方法论

- 脚本：`benchmarks/bench_index.py`（原始全矩阵）、`bench_pinned.py` /
  `bench_gate.py`（定点核复测）。
- 计量：3 warmup + 10-20 iters，取 **median**；`bench_index.py` 每格 6 轮中位数；
  `bench_gate.py` 3 轮中位数（min/med/max）。
- 带宽口径：bytes = idx 读 + 数据进/出（见 RESULTS.md 表头）。
- **共享箱噪声处理**：608 线程数字受其他用户支配，单次 run 与中位数差可达
  ~3x。**可靠数字 = 1 线程 + 定点核复测（§8）**。

## 3. 模型场景结论（§8 定点，3 轮中位数）

| 场景 | 1 线程 | 38 线程 |
|---|---|---|
| S1 spconv features[inp_idx] C=64 | **3.96x** | 1.72x |
| P1 h[src] i64 C=64 | **3.86x** | 1.94x |
| P2 pos[src] A=3 | **1.76x** | 0.86x |
| P3 x[mask] C=16 | 1.11x | 0.35x |
| S4 x[perm] f64 N=200k | 0.94x | 0.53x |
| 1D f64 K=2M | 0.80x | 0.77x |
| 1D f64 K=10M | 0.74x | 0.80x |
| 1D f32 K=2M | 0.88x | 0.71x |
| 1D f32 K=10M | 0.69x | **0.18x** |

RowCopy 附加数据（node 4 全矩阵定点）：`x[:,idx]` B=256 K=100k **3.48x**、
`x[idx,:]` C=32 E=1M **3.50x**（1 线程）。608 全核定点（node 4）S4 可达 5.6x
（原始 §1），说明 8B 大随机 gather 在 608 线程有真实收益，但共享箱上波动大。

## 4. 收益来源分解

- **RowCopy（主要收益）**：S1/P1/x[idx,:] 1 线程 ~3.5-4.0x；P2 A=3 是"短行"行
  拷贝 ~1.8x。SVE `ld1/st1` 连续拷贝 + 单次 idx 读取，正中 spconv 规则表 / 图
  消息路径。
- **8B PureGather（S4）**：608 线程 5.6x；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复，Task 8 前 0.02-0.3x）。
- **不收益**：1D f32/i32 大 K 随机 gather（4B SVE gather 硬件 MLP 限制，见
  kernels.md §2）；P3 mask 多线程（nonzero→scalar 并行性差，0.35x）；x[i,j]
  随机二维 gather（0.72-0.85x）。

## 5. 与设计文档 §8 的对照

设计优先级 A-linearize → L-SVE512 gather → C-prefetch / M-NT 全部确认：
PF=16 为 {0,16,32,64} 最优；NT 在噪声带内，默认关。设计 §13 里程碑
"PureGather 1.5-3×" 在 f64 小/中 K 达成（1.6-1.8x），**但 4B 大 K 随机 gather
未达成（0.18x，内存受限）**——设计文档已如实标注为诚实偏差。算子是模型的
**定向**加速器（行/8B 路径），非通用 1D 索引加速。

## 6. 如何复测

```bash
# 1) 全矩阵（共享箱，仅作参考）
HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_index.py

# 2) 定点复测（推荐，可靠性高）
# 先扫空闲节点：确认某节点全核 <5% busy
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=<quiet_node_cpuset> HPC_THREADS=38 \
  python benchmarks/bench_pinned.py
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=<quiet_node_cpuset> \
  python benchmarks/bench_gate.py   # 模型场景聚焦

# 3) 单核探针（验证 4B SVE gather 硬件限制，可选）
# agent_space/sdd-index-sve/ 下有 gather_probe 源码可复跑
```

> 复测注意：共享箱上先 `python3` 采样 `/proc/stat` 2s 确认目标节点空闲；
> 每轮间留间隔；1 线程数字跨轮最稳定，38/608 线程以中位数为准。
