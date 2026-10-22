# 性能基准

完整报告见 `benchmarks/RESULTS.md`（§1 原始矩阵、§2 全矩阵 6 轮中位数、§3 NT
diff、§4 聚焦门禁、§5 结论、§6 备注、§7 集成、**§8 定点复测**、**§9 AI4S
workload 优化轮 T0-T4**、**§10 T0-T5 全线程扫描 1-38t（2-36t 已作废）**、**§15 T11 修正矩阵**）。本文档是方法论与
结论摘要；T5（1 线程回归修复）见 `optimizations.md` §9。

## 1. 环境

- 机器：鲲鹏 920F（Kunpeng 920F），608 核 / 32 NUMA（每节点 38 核），SVE2 VL=512-bit，
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

## 3. 模型场景结论（T11 矩阵，node 5，25 iters 中位数）

| 场景 | 1 线程 | 36 线程 |
|---|---|---|
| S1 spconv features[inp_idx] C=64 | **4.46x** | 8.37x |
| P1 h[src] i64 C=64 | **4.42x** | 5.96x |
| P2 pos[src] A=3 | **2.05x** | 8.73x |
| P3 x[mask] C=16 | **2.23x** | 4.20x |
| S4 x[perm] f64 N=200k | 1.26x | 1.01x |
| 1D f64 K=2M | 1.26x | 1.51x |
| 1D f64 K=10M | 1.17x | 1.95x |
| 1D f32 K=2M | 1.07x | 2.17x |
| 1D f32 K=10M | 1.06x | 3.92x |

T11 起 `torch.set_num_threads` 对扩展生效；旧 §8.3/§10 的 2-36t 数据因
libomp/libgomp 双 ICV 作废。小 K 单下标 `x[idx]`（K≤1023）真实路径 1.34-2.08x，
K=4096 约 0.9x，K≥8192 高线程 0.6-0.9x（RESULTS §15.4）。

RowCopy 附加数据（node 4 全矩阵定点）：`x[:,idx]` B=256 K=100k **3.48x**、
`x[idx,:]` C=32 E=1M **3.50x**（1 线程）。608 全核定点（node 4）S4 可达 5.6x
（原始 §1），说明 8B 大随机 gather 在 608 线程有真实收益，但共享箱上波动大。

### 3.1 线程数扫描（1/2/4/8/16/32/36，node 5，T11 修正）

> **旧表作废**：本节与 RESULTS §10 的 2-36t 数据曾受 libomp/libgomp 双 ICV 影响
> （扩展实际用 38 个 worker），相对值被高估。下表是 T11 修正后的矩阵（30 场景
> 完整版见 RESULTS §15，逐 case 分析见 degradation-analysis.md）。

| 场景 | 1t | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | 4.46x | 4.62x | 3.95x | 3.55x | 2.97x | 6.69x | 8.37x |
| P1 h[src] i64 C=64 E=1M | 4.42x | 4.43x | 3.83x | 4.02x | 3.52x | 6.97x | 5.96x |
| P2 pos[src] A=3 E=2M | 2.05x | 2.12x | 2.06x | 2.90x | 1.78x | 6.64x | 8.73x |
| P3 x[mask] nnz≈60k C=16 | 2.23x | 1.82x | 1.53x | 1.34x | 1.87x | 3.76x | 4.20x |
| S4 x[perm] f64 N=200k | 1.26x | 1.36x | 1.09x | 0.87x | 0.76x | 1.35x | 1.01x |
| 1D f64 K=2M | 1.26x | 1.18x | 1.15x | 1.11x | 1.03x | 1.08x | 1.51x |
| 1D f64 K=10M | 1.17x | 1.06x | 1.03x | 1.06x | 1.03x | 2.01x | 1.95x |
| 1D f32 K=2M | 1.07x | 1.41x | 1.40x | 1.35x | 1.24x | 1.48x | 2.17x |
| 1D f32 K=10M | 1.06x | 1.11x | 1.11x | 1.14x | 1.16x | 2.28x | 3.92x |

**结论**：
- **RowCopy 仍是最强项**：S1/P1 1-16t 为 2.9-4.6x；32/36t 的 6-8x 有一部分来自
  aten 在带宽饱和后变慢（见 degradation-analysis.md §6.2）。
- **8B pure gather** 1.0-1.5x，高线程 1.5-2.2x；**4B pure gather** 1-16t 约
  1.1x、32/36t 2.3-3.9x。
- **P3 mask** 1.3-4.2x；**S4** 8/16t 0.87/0.76x 是小规模强扩展问题。
- 绝对带宽/线程扩展效率见 `degradation-analysis.md` §6。

## 4. 收益来源分解

- **RowCopy（主要收益）**：S1/P1 1 线程 **4.4-4.5x**；P2 A=3 是"短行"行
  拷贝 **2.05x**（T11 矩阵）。SVE `ld1/st1` 连续拷贝 + 单次 idx 读取，正中 spconv 规则表 / 图
  消息路径。
- **8B PureGather（S4）**：608 线程 5.6x；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复，Task 8 前 0.02-0.3x）。
- **不收益**：1D f32/i32 大 K 随机 gather（64B 行放大 + DRAM 延迟；SVE 4B 与
  标量持平，见 kernels.md §2 / RESULTS §8.4）；x[i,j] 随机二维 gather
  （0.72-0.85x）。P3 mask 已由 T8 单遍融合修复（2-8x）。

### 4.1 AI4S workload 优化轮（T0-T4，见 RESULTS §9）

真实 AI4S 模型端到端无提升甚至劣化后，按优化任务文档实现 T0-T4（自适应粒度、
rowcopy 阈值 A>=4、M1 skip-linearize、T3 融合单遍、T4 小值域桶化），并用
`benchmarks/bench_ai4s.py`（M1-M6 workload 形态，§3.1）定点复测。净效果
（注意：M1-M4 的 8-32t 列来自 T11 前，受双 ICV 影响不可信；1t 数字有效）：

- **M1（最热模式，14 模型累计 27270ms）**：i64 E=2k-20k 1 线程 **8-10x**、
  8 线程 1.3-2.7x；i32 变体 8 线程 2.7x。此前是 §4 的"无收益"形态。
- **M2 feat[r]（spconv）**：1 线程 ~3.9-4.0x、8 线程 2.6-4.0x；C=8 短行 ~0.93x。
- **M3 keys[order] 小值域查表**：T4 raw 桶化把 0.2x（优化前）→ 1.4-2.6x（1t）
  和 8-12x（32t）。
- **M4 feat[inverse] 高重复**：1t 3.6x、8-32t 3.8-4.8x。
- **4B f32 K=10M 仍受行放大限制**（§8.4）：T4 桶化值域需 ≤2048，大值域随机
  gather 仍走 SVE gather；T11 修正矩阵 1-16t 1.06-1.16x、32/36t 2.3-3.9x，
  绝对带宽上限仍由 64B 行放大 + DRAM 延迟决定。

### 4.2 T5 1 线程回归修复（bd21485，见 optimizations.md §9）

回归扫描（`benchmarks/bench_regression.py`）发现 1 线程多形态 0.6-0.9x（根因：
linearize 两遍模型 + 单线程随机访存延迟 + 内层循环 at::Tensor 方法开销）。
T5 用单遍标量（`gather_single_1d` / 1 线程短行 fused）+ fused/linearize 的 nb==1
全长连续快路径修复，全部 ≥1.0x；2+ 线程本就全面占优。mask 1 线程保持 ~1.0x
噪声级，不做 fallback。完整数据见 `optimizations.md` §9 实测表。

## 5. 与设计文档 §8 的对照

设计优先级 A-linearize → L-SVE512 gather → C-prefetch / M-NT 全部确认：
PF=16 为 {0,16,32,64} 最优；NT 在噪声带内，默认关。设计 §13 里程碑
"PureGather 1.5-3×" 在 f64 小/中 K 达成（1.5-1.8x），**但 4B 大 K 随机 gather
未达成（T11 修正后 1-16t 1.06-1.16x、32/36t 2.3-3.9x；绝对带宽受 64B 行放大 +
DRAM 延迟限制）**——旧 "0.18x" 是双 ICV 下的失真数据。算子是模型的
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
