# 性能基准

完整报告见 `benchmarks/RESULTS.md`：**§2.4 T13 全量矩阵**（31 场景 × 7 线程；
T15 未改这些路由，数据继续有效）
+ **§2.3 小 K 两进程**，§2.1 为 T11 基线；§3 历史轮次（§3.1-§3.5 早期 1/608
线程与定点复测、§3.7 4B 归因复核、§3.8 AI4S T0-T4、§3.9 T0-T5 全线程（已作废）、
§3.10-§3.15 compile/mask/小K/opt-in/T12/T13）。本文档是方法论与结论摘要；
T5（1 线程回归修复）见 `optimizations.md` §9。

> **先读结论（T15）**：RowCopy（整行/整块取数）仍是主要收益，1t 约 3.0x
> （run 间 3.0-4.5x）；1D 8B 纯 gather 1.0-1.5x；4B 1D 纯 gather 1-16t 约
> 1.1x、高线程靠 aten 退化才显得更高；小张量/小 K 不要开满线程。T15 修复
> 2D A=16 K=8192/65536 高线程空档（32-38t 1.26-2.17x）与 f32+i64 大源
> SVE 路由（K=10M idx64 8t 2.22x）。当前残余主要是 S4/multi 8-16t、1B 16t、
> complex 1t、非连续 mask、1D K=65536 小源 32-38t（0.90-0.92x）。完整
> 矩阵见 RESULTS §2.4。
>
> **术语**：`1t/8t/36t` = 线程数；`ratio = aten 时间 / hpc 时间`，>1 为更快；
> GB/s 是“索引 + 读 + 写”的有效字节带宽；S1/P1... 是模型形态代号（见 §3）。

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
- **短调用 / 真实路径**：`bench_realpath_driver.py` 交替启动 aten/hpc 两个子进程，
  两边都走真实 `x[idx]`（dispatcher 全链路），按 (case, threads) 取轮间中位数，
  避免 pybind 直呼绕过 dispatcher 导致短调用比值偏乐观；覆盖 M5 标量循环、
  shortq A=2-4、A1 单元素高频、M1 真实权重块（≤16KB/行）、M2 代表形态
  （脚本清单与真实 workload 预测力矩阵见 `benchmarks.md` §3）。
- 带宽口径：bytes = idx 读 + 数据进/出（见 RESULTS.md 表头）。
- **共享箱噪声处理**：608 线程数字受其他用户支配，单次 run 与中位数差可达
  ~3x。**可靠数字 = node 5 定点交替矩阵（RESULTS §2）+ 1 线程定点**；旧
  §8.3/§10 的 2-36t 标签因双 OpenMP ICV 失效，勿再引用。

## 3. 模型场景结论（T11 基线，node 5，25 iters 中位数）

> 场景/负载代号（S1/P1/P2/P3/S4、M1-M6、A1/shortq…）与形状记号
> （A/B/K/C/E）的白话含义见 `glossary.md`（术语与代号速查）。

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
libomp/libgomp 双 ICV 作废。T15 小 K 复测（RESULTS §2.3）：单下标 `x[idx]`
K≤1024 为 1.83-2.36x；2D A=16 K=8192/65536 32-38t **1.26-2.17x**（T15 raw
放行，原 0.73/0.76x，hpc 绝对时间砍半）；K=1024 与 K=1023 全线程拉平
（并行阈值线程缩放）；1D K=8192/65536 高线程 0.90-0.95x（小源保持标量，
残留为前置 bucket 扫描 ~15µs）。多索引
小 K（T13）在 K=512、A=1/4/64 为 1.3-3.4x（1t A=64 3.4x），见 RESULTS §3.15。

**真实 workload 对拍（T14，RESULTS §2.5，1/8/16/38t）**：M1 权重块
（≤16KB/行）1t 13.0-13.6x、8t 5.1-5.8x；M5 标量循环 / A1 单元素高频
2.6-3.8x；shortq A=2-4 K=64-1023 为 2.6-4.3x（T14 修复 A=3 短行 0.84→2.6x）；
M2 (20000,8/64/256) 全形态 ≥1.11x（38t 带宽饱和）。这是短调用 / 真实路径
收益结论的首选基准。

RowCopy 附加数据（node 4 全矩阵定点）：`x[:,idx]` B=256 K=100k **3.48x**、
`x[idx,:]` C=32 E=1M **3.50x**（1 线程）。608 全核定点（node 4）S4 可达 5.6x
（原始 §1），说明 8B 大随机 gather 在 608 线程有真实收益，但共享箱上波动大。

### 3.1 线程数扫描（1/2/4/8/16/32/36，node 5，T11 修正）

> **旧表作废**：本节与 RESULTS §3.9 的 2-36t 数据曾受 libomp/libgomp 双 ICV 影响
> （扩展实际用 38 个 worker），相对值被高估。下表是 T11 修正后的矩阵（31 场景
> 完整版见 RESULTS §2，逐 case 分析见 degradation-analysis.md）。

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
- **P3 mask** 1.3-4.2x；**S4** 8/16t 0.98/0.91x（focused 0.94/0.86x）是小规模
  强扩展问题。
- 绝对带宽/线程扩展效率见 `degradation-analysis.md` §6。T12 grain 64 + T13
  入口快路径后，1D K=8192 在 36t 已回到 1.04x（不再低于 1t）；T15 又修复
  2D A=16 K=8192/65536 的 32-38t 空档（1.26-2.17x）与 K=1024 并行边界；
  1D K=65536 小源 32-38t 仍 0.90-0.92x（标量本就不慢，残留为前置扫描）。
  每线程 ≥64KB 时效率约 40-90%（§6.3），所以小张量不要开满线程。

## 4. 收益来源分解

- **RowCopy（主要收益）**：S1/P1 1 线程 **4.4-4.5x**；P2 A=3 是"短行"行
  拷贝 **2.05x**（T11 矩阵）。SVE `ld1/st1` 连续拷贝 + 单次 idx 读取，正中 spconv 规则表 / 图
  消息路径。
- **8B PureGather（S4）**：608 线程 5.6x；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复，Task 8 前 0.02-0.3x）。
- **不收益**：1D f32/i32 大 K 随机 gather（64B 行放大 + DRAM 延迟；SVE 4B 与
  标量持平，见 kernels.md §2 / RESULTS §3.7）；x[i,j] 随机二维 gather
  （0.72-0.85x）。P3 mask 已由 T8 单遍融合修复（2-8x）。

### 4.1 AI4S workload 优化轮（T0-T4，见 RESULTS §3.8）

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
全长连续快路径修复，表中形态回到 ≥1.07x；T11 矩阵中这些形态 2t 也 ≥1.09x
（旧 2-36t 标签不可信）。T13 矩阵：连续 P3 1 线程 1.78x、非连续 1D mask
0.86-0.98x（T8 只融合连续 mask），不做 fallback。完整数据见 `optimizations.md`
§9 实测表与 RESULTS §2.4。

### 4.3 胜出场景的硬件根因（T15 视角：为什么比 aten 快）

劣化侧的对称分析见 `degradation-analysis.md` §5.9；本节解释**胜出从哪里
来**。锚点均取 RESULTS §2/§3 与 realpath（§2.5）的 1-16t 可靠数据，高线程
单点倍率含 aten 退化的部分单列（见本节末"诚实边界"）。

| 家族 | 代表场景（锚点） | 实测 | 主根因（层面） |
|---|---|---|---|
| 整行拷贝 | S1/P1（C=64）、rowcopy A=16 K=200k、2D A=16 K=8192/65536（T15） | 1t 3.0-4.6x（S1 4.46、A=16 3.58；A=2/3 短行 1.5-2.1x）；8t 2.6-4.0x（M2）；32-38t 官方 1.26-2.17x（聚焦 2.12-2.82x） | 微架构（指令吞吐/依赖链）+ 系统结构（行有效性） |
| gather MLP | 1D f64 K=10M、f32+i64 K=10M（T15-C 直连） | 内核段 ~1.9x（8t −40%）；1t 1.17x；36t 1.95/2.22x（部分为 aten 退化） | 微架构（在飞负载数） |
| 遍数/物化消除 | mask T8 单遍（mask 1D 8-16t 1.30x、P3 1t 2.23x）、bf16 fused | 1.2-2.4x | 系统结构（中间流量）+ 并行（遍数×fork-join） |
| 固定开销消除 | 小 K x[idx] K<=1024（T9/T13）、shortq（T14） | 全线程 1.83-2.6x | 软件层（绕 normalize/迭代器构建） |
| 值域局部性 | M3 桶化（range2048，值域≤2048） | 1t 1.4-2.6x（优化前 0.2x） | 系统结构（随机读 → 顺序源读+散射） |
| 串行边界免疫 | K≈1024 高线程（T15 阈值）、shortq K=512-1023 | 2.83-4.57x | 并行编程（小工作量退串行） |

#### 整行拷贝为什么赢得最狠（1t 3-4.6x）

1. **形态决定上限：随机性从"每元素"升到"每行"**。被索引的是 A×4B 连续块，
   A>=16（f32）即每个 64B line 的有用字节 100%——与 1B/4B 纯 gather（1/64、
   1/16 行）正相反。rowcopy 赢的是"形态红利"，内核只决定逼近上限的程度。
2. **每轮 256B 的 SVE 拷贝**：`copy_contig_sve` 每迭代 4 条全宽 load/store
   （VL=512）。perf（degradation §7）S1 1t：hpc IPC 0.40 甚至低于 aten
   0.44，仍快 4.4x——**bytes/cycle 才是关键，不是 IPC**；`x[:,i]` 1t 同理，
   hpc 每元素指令数 ~2x 少。
3. **预计算偏移 + 单依赖短链**：行循环 = 读 8B 偏移 → 加基址 → 拷贝，乱序
   窗口同时挂几十行。aten 的 TensorIterator 为任意形状/dtype/设备统一服务，
   热循环保留运行期步长乘法与类型擦除（char*）拷贝——**通用性的代价**。
4. **raw 不等于无条件赢**（T15 教训）：大 K 冷源下 raw 的逐行地址链压低
   MLP，慢 12-32%；小 K 省 fork-join 才赢。最终路由
   `A>=4096||A<16||(A<=64&&B*K<=65536)`（degradation §5.9.4）。

#### gather 的 8/16 路 MLP（大源 1D，内核段 ~1.9x）

aten 标量循环每元素一条 load：源在 L3/DRAM 时成本 ≈ 延迟÷有效 MLP（实测
~19ns/元素）；SVE gather 一条指令 8（f64）/16（f32）路独立 lane 负载，延迟
被摊薄（~10ns/元素）——内核段 ~1.9x、用户可见 8t −40%。1t 也赢 ~1.17x
（一条 gather 替代"load+CSEL+load+store"链，每元素指令更少）。**源在 L1
时反转**（gather 微码/发射下限 ~18ns vs 标量 ~7.1ns），所以 T15-C 门取源
>512KB，完整推导见 degradation §5.9.1。

#### 遍数与物化流量（mask 1.30x / fused 1.2-2.4x / raw 省一遍）

aten 处理 bool mask = `nonzero`（物化 int64 下标，串行段 + 并行扫描）+
linearize + gather，**三遍**全张量流量；T8 单遍压实 = 读 mask 边写输出，
**一遍**。8t 分解实测：nonzero 1.88 + linearize 0.86 + gather 0.24 ≈
2.97ms vs T8 1.95ms——多出的 ~1ms 全是中间物化流量（8B×nnz 的 int64 下标
+ 两遍 mask 读）。同一逻辑解释 bf16 fused 与 raw rowcopy（各省一遍
linearize）；并行侧每省一遍同时省一次 fork-join（degradation §5.9.2）。

#### 固定开销（小 K 1.83-2.6x：软件层的胜利）

aten 每调用付 dispatcher → make_info → TensorIterator 构建（维度重排/步长
计算/类型提升，µs 级）；对 K<=1024、总时长 <50µs 的调用占 30-50%。T9/T13
用编译期知识（f32/i64/连续/末维）绕过 normalize 直接紧凑循环，固定成本压到
~1-2µs。这不是硬件胜利，是**分层抽象的代价被专门化消除**——也是对 aten
最"稳"的胜利（不依赖箱态与线程数）。

#### 值域局部性（M3 桶化 0.2x → 1.4-2.6x）

值域 <=2048 时把"按输出顺序随机读源"反转为"按源顺序读 + 直方图散射"：
源（<=2048 元素）常驻 L1 且读序化（预取友好），随机性转移到写侧（写不读
回、无行放大惩罚）。访问结构的改变绕开了延迟暴露，属于系统结构层的收益。

#### 串行边界免疫（2.83-4.57x）

小工作量（`n*A < max(16384, 512×nt)`）主动退串行，免疫 fork-join；aten 对
K=1023×A>=3 在 16/38t 有 539-615µs 的 fork-join 病理（realpath 表 22-26x
的格子即来源于此）。这一条是**并行编程层面的胜利**：不是内核更快，是更
知道什么时候不该并行。

#### 诚实边界：哪些"胜出"不是我们的功劳

32/36t 的 5-10x 有一部分是 aten 自身带宽饱和后变慢（degradation §6.2/
§5.8-F）；realpath 的 22-26x 格子是 aten fork-join 病理。判读标准：**1-16t
中位趋势 + hpc 绝对时间/强扩展效率**才是真胜出；高线程单点倍率不作收益
宣传。胜出的本质浓缩为三句话：**让每条 cache line 满载（rowcopy/桶化），
让每条指令携带更多在飞负载（SVE gather / 256B 拷贝），少付抽象与同步的
固定税（fast path / 单遍 / 串行边界）**。

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
