# 性能基准

完整报告见 `benchmarks/RESULTS.md`（§1 原始矩阵、§2 全矩阵 6 轮中位数、§3 NT
diff、§4 聚焦门禁、§5 结论、§6 备注、§7 集成、**§8 定点复测**、**§9 AI4S
workload 优化轮 T0-T4**、**§10 T0-T5 全线程扫描 1-38t**）。本文档是方法论与
结论摘要；T5（1 线程回归修复）见 `optimizations.md` §9。

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

### 3.1 线程数扫描（2/4/8/16/32/36 线程，node 5，3 轮中位数）

`bench_gate.py` 支持 `HPC_THREADS` 环境变量（逗号分隔列表，默认 "1,38"）。

| 场景 | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | **33.7x** | 19.4x | 5.6x | 4.3x | 2.3x | 2.2x |
| P1 h[src] i64 C=64 E=1M | **30.5x** | 17.5x | 8.7x | 4.4x | 2.3x | 2.1x |
| P2 pos[src] i32 A=3 E=2M | **19.7x** | 6.3x | 5.8x | 1.9x | 3.4x | 1.1x |
| P3 x[mask] nnz≈60k C=16 | 1.2x | **1.5x** | 1.4x | 1.2x | 1.1x | 1.1x |
| S4 x[perm] f64 N=200k | **20.7x** | 13.0x | 8.0x | 4.1x | 8.6x | 11.6x |
| 1D f64 K=2M | **6.8x** | 3.0x | 1.8x | 1.9x | 2.6x | 1.9x |
| 1D f64 K=10M | **10.5x** | 5.2x | 2.7x | 1.4x | 0.9x | 0.8x |
| 1D f32 K=2M | **10.6x** | 2.8x | 1.5x | 4.1x | 1.8x | 2.6x |
| 1D f32 K=10M | 1.7x | 1.2x | 0.7x | 0.3x | 0.2x | 0.2x |

**关键结论：低线程数收益最强。**
- RowCopy（S1/P1/P2）在 **2-4 线程就饱和单 NUMA 节点的内存带宽**（~60 GB/s，
  S1 2t 达 33.7x）：2 个线程足够跑满带宽，而 aten 标量 gather 受延迟限制，
  需要 ~30+ 线程才接近同一带宽。这是**扩展性优势**：算子用少量核心即达峰值。
- 8B PureGather（S4/1D f64）同样在低线程闪耀（S4 2t 20.7x）——SVE gather +
  预取把延迟隐藏，2 线程标量无法做到。
- f32 K=10M 在扫描中再次确认 4B 硬件天花板：2t 1.7x 是唯一收益窗口，8t 起
  <1x，16-36t 塌到 0.2x（kernels.md §2）。
- P3 mask 全程平 ~1.1-1.5x，无线程依赖，与 §3 中 0.35x（38t）一致——mask 路径
  并行性差是独立于内核的。
- 32/36t 单元（P2 3.4x、S4 8.6-11.6x）run-to-run 散布更大（中位数对应 0.8-3.4x
  的轮间摆动），2-16t 单元最可靠；36t 是 38 核节点近满载，受 SMT 邻居影响。

## 4. 收益来源分解

- **RowCopy（主要收益）**：S1/P1/x[idx,:] 1 线程 ~3.5-4.0x；P2 A=3 是"短行"行
  拷贝 ~1.8x。SVE `ld1/st1` 连续拷贝 + 单次 idx 读取，正中 spconv 规则表 / 图
  消息路径。
- **8B PureGather（S4）**：608 线程 5.6x；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复，Task 8 前 0.02-0.3x）。
- **不收益**：1D f32/i32 大 K 随机 gather（4B SVE gather 硬件 MLP 限制，见
  kernels.md §2）；P3 mask 多线程（nonzero→scalar 并行性差，0.35x）；x[i,j]
  随机二维 gather（0.72-0.85x）。

### 4.1 AI4S workload 优化轮（T0-T4，见 RESULTS §9）

真实 AI4S 模型端到端无提升甚至劣化后，按优化任务文档实现 T0-T4（自适应粒度、
rowcopy 阈值 A>=4、M1 skip-linearize、T3 融合单遍、T4 小值域桶化），并用
`benchmarks/bench_ai4s.py`（M1-M6 workload 形态，§3.1）定点复测。净效果：

- **M1（最热模式，14 模型累计 27270ms）**：i64 E=2k-20k 1 线程 **8-10x**、
  8 线程 1.3-2.7x；i32 变体 8 线程 2.7x。此前是 §4 的"无收益"形态。
- **M2 feat[r]（spconv）**：1 线程 ~3.9-4.0x、8 线程 2.6-4.0x；C=8 短行 ~0.93x。
- **M3 keys[order] 小值域查表**：T4 raw 桶化把 0.2x（优化前）→ 1.4-2.6x（1t）
  和 8-12x（32t）。
- **M4 feat[inverse] 高重复**：1t 3.6x、8-32t 3.8-4.8x。
- **4B f32 K=10M 仍是硬件天花板**（§8.2）：T4 桶化值域需 ≤2048，大值域随机
  gather 仍走 SVE gather，高线程 0.2x。

### 4.2 T5 1 线程回归修复（bd21485，见 optimizations.md §9）

回归扫描（`benchmarks/bench_regression.py`）发现 1 线程多形态 0.6-0.9x（根因：
linearize 两遍模型 + SVE gather 单线程低 MLP + 内层循环 at::Tensor 方法开销）。
T5 用单遍标量（`gather_single_1d` / 1 线程 fused）+ fused/linearize 的 nb==1
全长连续快路径修复，全部 ≥1.0x；2+ 线程本就全面占优。mask 1 线程保持 ~1.0x
噪声级，不做 fallback。完整数据见 `optimizations.md` §9 实测表。

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
# 模型场景聚焦（默认线程 1,38；HPC_THREADS 可自定义如 2,4,8,16,32,36）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=<quiet_node_cpuset> \
  python benchmarks/bench_gate.py

# 3) 单核探针（验证 4B SVE gather 硬件限制，可选）
# agent_space/sdd-index-sve/ 下有 gather_probe 源码可复跑
```

> 复测注意：共享箱上先 `python3` 采样 `/proc/stat` 2s 确认目标节点空闲；
> 每轮间留间隔；1 线程数字跨轮最稳定，38/608 线程以中位数为准。
