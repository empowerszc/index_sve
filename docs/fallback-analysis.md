# 慢场景与回退路径深度分析

本文档回答一个问题：**为什么 index_sve 在某些形态下性能不佳（甚至慢于
aten::index），受限于什么**。这是"回退到 aten"类问题（integration.md §7）背后的
完整技术分析——每个慢形态的触发条件、慢的根本机制、硬件/算法限制、实测数据、
以及为什么最终选择"不 fallback"。

> 读者：想理解本项目性能边界的人 / agent。结论摘要可直接看 §1 与 §4；§3 是
> 逐场景深度分析。

> **TL;DR**：慢场景分四类——(M1) 64B 行放大 + DRAM 延迟、(M2) linearize 两遍
> 固定开销、(M3) 串行/低并行阶段、(M4) 没有 SVE 内核只能标量。**都不适合用
> “回退 aten”修**：aten 同样受 M1 限制，而 Python 包装层每次调用还要 ~45µs，
> 小算子会被这笔固定开销吃掉（§4）。正确做法是针对性内核优化（T5/T9/T11）
> 或接受硬件上限。
>
> **名词**：**fallback（回退）** = 放弃本算子、转调 aten；**cache line** = CPU
> 搬数据的最小单位（64B）；**MLP** = memory-level parallelism，能同时在途的
> 访存数；**wrapper** = Python 层调用包装；**dispatch_scalar** = 本算子内部
> 的标量兜底内核；**linearize** = 下标→一维偏移的换算/检查过程。

## 1. 总览：四个根本机制

所有慢形态都可归因于以下四个机制之一（或叠加）：

| 机制 | 本质 | 影响的形态 | 能否软件修复 |
|---|---|---|---|
| **M1 缓存行放大 + DRAM 延迟** | 随机 gather 每个输出元素要拉一条 64B cache line，4B 元素只用 4B（16x 放大）；工作集超过 LLC 后受 DRAM 延迟/行流量限制。同构建 A/B 实测 SVE 4B 与标量持平（1.85 vs 1.86 ms），**不是 SVE 指令 MLP 缺陷** | 4B（f32/i32）大 K 大值域 pure gather；高线程数下尤甚 | 部分（小值域桶化、预取、改行 gather）；随机访问的行放大不可消除 |
| **M2 两遍模型固定开销** | linearize 先物化偏移数组、gather 再读它 = 两次遍历 + 一次分配；小 K 时固定开销占比高；1 线程无并行加速 | 1 线程所有需 linearize 的形态（T5 已大部分修复） | 已修（单遍路径） |
| **M3 串行/低并行阶段** | 某些形态内部有无法并行化的阶段（如 scalar_gather 的 col_off 解引用） | A<4 短行 rowcopy | 结构性问题，难修 |
| **M4 语义强制的标量回退** | 某些 dtype/形状**根本没有 SVE 内核可走**，只能落 `dispatch_scalar` | 2B/1B、complex、非连续 after、f32 超大 numel | 需新增内核（未做） |

此外有一个**全局性开销**：Python wrapper（若做 fallback）会给每次调用加 ~45µs
（见 §4.2），远大于小 op 本身耗时。

## 2. 路由与回退路径全图

`index_impl`（op.cpp）与 `gather_index`（gather.cpp）的决策树，标出每个**回退
（非 SVE 主路径）**与触发条件：

```
index_impl(self, indices)
  normalize → IndexSpec（B=行前, K=被索引, A=行后）
  │
  ├─ 单 mask && 覆盖前 m 维 && 连续 && CPU → gather_mask（T8 单遍压实）
  ├─ 单整数下标 && K<=4096 && 连续 && CPU → small_single_index（T9）
  ├─ K < 1024                    → gather_direct（快速路径，无 linearize）
  ├─ 单索引 && A>=4096 && 连续 && f32/f64 → M1 raw rowcopy（skip-linearize）★主路径
  ├─ 单索引 && B==1 && A==1      → T4 raw 桶化（值域≤2048 查表）
  │     └─ 值域>2048 && 1 线程   → gather_single_1d（T5 单遍标量）
  ├─ 1 线程 && 单索引 && A>1     → A>=4 连续 ? raw rowcopy : gather_fused（T11）
  │
  └─ linearize + gather_index
        │  （A==1 或 [连续 && A>=4 && f32/f64]）
        │
        ▼ gather_index 内部分支（按优先级）
  pure(A==1):
     1. 值域≤2048 ? bucket_gather_pure（查表）★主路径
     2. !idx32 && f64/i64        → 8B SVE gather ★主路径（高效）
     3. idx32  && f32/i32        → 4B SVE gather ⚠（M1 硬件限制）
     4. 其他 dtype/idx 组合      → dispatch_scalar ←【回退】
  rowcopy(A>1):
     1. 连续 && A>=4 && f32/f64  → rowcopy_sve ★主路径
     2. 其他（A<4/非连续/非f32f64）→ dispatch_scalar ←【回退】
  │
  └─ 其余（A<4 或非连续小块）    → gather_fused（T3/T5 单遍）
```

**回退落点汇总**（`dispatch_scalar` 或单遍标量，皆非 SVE）：

| 触发条件 | 回退内核 | 慢机制 |
|---|---|---|
| dtype ∈ {half, bf16, i16, i8, u8, bool} 纯 gather | dispatch_scalar | M4（无 4B/8B 覆盖）|
| complex 纯 gather | dispatch_scalar | M4 |
| f32 且 numel ≥ 2^31（防 i32 越界） | dispatch_scalar | M4（idx 被迫 int64，8B 分支又只认 f64/i64 out）|
| rowcopy A∈[2,3] 短行 | dispatch_scalar | M3（col_off 逐个解引用）|
| rowcopy 非连续 after 维 | dispatch_scalar | M3（无连续块可向量化）|
| 4B 大 K 大值域 pure gather（idx32+f32/i32） | 4B SVE gather（与标量持平，cache-line 放大） | M1 |
| mask（bool） | T8 gather_mask 单遍压实（不可融合时仍 nonzero + gather） | 旧路径 M3+M2；T8 已修 |
| 1 线程单索引 A==1 大值域 | gather_single_1d（T5，已追平） | M2（已修）|

## 3. 逐场景深度分析

### 3.1 4B（f32/i32）大 K 大值域 pure gather —— cache-line 放大 + DRAM 延迟

**触发**：`pure && idx32 && sve_512_ok() && out∈{f32,i32}`，即 int32 索引的
1D 纯 gather，且值域 >2048（否则 T4 桶化接管）。典型：`x[i32]`，x 为 10M 以上
f32 张量、i 为 int32 随机下标。

**为什么慢**：这是**纯随机访存**形态：每个输出元素 `dst[p] = src[idx[p]]`，
src 地址随机。内存系统按 64B cache line 搬运，一个 4B 元素只用其中 4B（**16x
行放大**；8B 元素是 8x）。工作集超过 LLC 后，时间由"要拉多少条 line"和 DRAM
延迟决定，而不是 SVE 指令能发多少在途请求：

- **同构建 A/B（真实内核，同一随机模式）**：SVE 4B gather vs 标量循环，
  8/38 线程下 src=8MB 为 **1.85 vs 1.86 ms**，src=256MB 为
  **7.46 vs 7.45 ms**——两者持平。换指令或回退标量都不能改善。
- **早期独立探针的 0.75 vs 3.4 GB/s 不能端到端复现**（工作集/预取状态与真实
  内核不同），已撤回作为归因依据；RESULTS §3.7 记录了复核。
- 8B 路径（f64/i64）更快，是因为同样条数的 cache line 携带 2x 有用字节
  （行放大减半），并非 8B gather 的 MLP 更高。
- **硬件实测**（degradation §6.4）：本地 DRAM 随机延迟 115-192ns，单节点读
  带宽 73GB/s；同模式 C 探针的 dTLB miss ≈1.1-1.2 次/访问，IPC 0.34-0.48。
  真实内核 4B K=2M 在 1t/36t 的行流量为 2.8/73-76GB/s——**1t 是延迟受限，
  36t 是行带宽受限**，中间线程则是两者叠加。

**限制性质**：随机元素 gather 的行放大是访存模式问题，不是 920F 4B 指令缺陷。
软件只能减少行流量或隐藏延迟：
- 值域 ≤2048 → T4 桶化：把随机地址变成 L1/L2 常驻小表，行放大消失 ✅ 已实现；
- 值域大 → 软件预取/软件流水、按 cache line 分桶排序，或把"按元素 gather"
  改成"按行 gather"（A≥4 连续块，一次 line 拉多个有用字节）。回退 aten 或
  内核内标量均无收益（A/B 已持平），还受 §4.2 全局开销制约。

**实测**（T11 修正矩阵，node 5，interleaved median；RESULTS §2）：

| 线程 | 1t | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|---|
| 1D f32 K=10M 相对 aten | 1.06x | 1.11x | 1.11x | 1.14x | 1.16x | 2.28x | 3.92x |

1-16t 基本持平（行放大决定的上限），32/36t 的高比值有一部分来自 aten 在带宽
饱和后变慢（绝对带宽见 degradation-analysis.md §6）。**随机行放大本身仍在**，
它决定"绝对带宽上限"，不是"相对 aten 的比值"。旧文档的 2-36t 数字（如 2t
13.9x）因扩展实际使用 libomp 默认 38 worker 而失真，已作废。

### 3.2 2B/1B（half/bf16/i16/i8/u8/bool）与 complex 纯 gather

**触发**：`pure && A==1` 但 out dtype 不在 {f64,i64,f32,i32}。SVE 内核只写了
8B/4B 两种元素宽度的 gather/rowcopy，2B/1B 与 complex 没有向量实现。

**为什么慢**：纯 `dispatch_scalar` 标量逐元素。没有内核覆盖（M4）。但要注意：
- 这不是"比 aten 慢"，只是"没吃到 SVE 加速"。aten 也是标量 TensorIterator。
- T11 修复后 half/bf16 K=2M：1.22-2.40x（1-36t），1-16t 也全部 ≥1.22x。
- complex64 K=2M：0.95-2.76x（1-16t 0.95-1.00x，32/36t 1.78/2.76x）—— complex 是 8B 元素但**走标量**（没有 complex
  的 SVE gather 内核），收益来自单遍路径而非向量化；基本与 aten 持平。

**限制性质**：实现空白（可补），非硬件。未来加 2B/1B SVE gather（或用
`svld1_gather_u64index` 按字节偏移 trick）即可覆盖。当前不值得——这些 dtype
在 AI4S 模型 gather 路径罕见。

### 3.3 rowcopy 短行 A∈[2,3]

**触发**：`A>1 && A<4`（或 A>=4 但非连续/dtype 不符）。M2 spconv 的
`pos[src]` 是 A=3 短行——**这是真实模型形态**。

**为什么慢**：`dispatch_scalar` 的 `scalar_gather` 每元素要
`d[c] = s[col_off[c]]`——A=3 时每行 3 次解引用（col_off 数组查），且行间随机
跳转。SVE rowcopy 门槛 A>=4 是有意为之（A=2,3 时 SVE 向量化收益被加载开销
吃掉，T1 微基准已测）。A=3 无连续块向量化。

**但实测并不差**：T11 矩阵里 rowcopy A=3 K=200k 为 1.34-3.34x（1-36t），
P2 A=3 E=2M 为 1.78-8.73x。**这形态是"慢于 SVE 主路径"而非"慢于 aten"**——
相对 aten 是赢的，只是幅度不如长行；旧文档的 8t 7.58x 来自失真的线程标签。

**限制性质**：结构性（短行无法向量化拷贝）。不可修，但也不是回归。

### 3.4 rowcopy 非连续 after 维

**触发**：`A>1` 但 after 维 stride 不连续（如 `x.t()` 后的 `x[i]`、或
`x[:, i, :]` 中 i 后还有非连续段）。`rowcopy_sve` 要求 `after_contiguous`
（`col_off[c]==c`），否则整行不是一段连续内存。

**为什么慢**：必须逐元素解引用 col_off，SVE 连续 `ld1/st1` 用不上（M3/M4）。
同 §3.3——相对 aten 通常仍有收益，只是吃不到向量化。

**限制性质**：结构性。无法在现有模型下规避（除非先做一次内存重排，成本更高）。

### 3.5 mask（bool 索引）

**现状（T8 后已修复）**：旧路径 bool 掩码 → `normalize` 先 `index.nonzero()`
展开成位置下标，再走 linearize + gather。两段式 + nonzero 输出长度数据相关、
并行度差，高线程比值退化（P3 38t 0.35x）。

T8 在入口识别"单 mask + 覆盖前 m 维 + mask/self 连续 + CPU"走 `gather_mask`：
一遍计数 + 前缀和，第二遍直接压实并整行拷贝，跳过 nonzero 与第二次 gather。
P3（N=200k C=16 nnz≈60k，T11 修正矩阵）：1t 4.86→2.05ms（2.23x vs aten）、
8t 2.66→0.85ms（1.34x；旧 "8.0x" 是 38 worker 标签失真）、36t 0.94ms
（4.20x，32t 0.72ms / 5.60 GB/s 为最好点）。非连续 mask/self、混合其它
下标仍走旧路径（正确性不变）。

### 3.6 多索引广播形态（x[i,j]）

**触发**：多索引且形状需广播（如 `x[i, j]`，i/j 不同长或含标量）。走
linearize（混合进制 decode）+ gather。

**为什么慢（历史）**：1 线程时 linearize 两遍模型固定开销 + 每次访问
at::Tensor 方法（引用计数/虚调用）→ 0.66x。**T5 已修复**（nb==1 全长连续
快路径 + 属性预提取）：linearize 4.84→1.10ms，x[i,j] 1t 0.66x→**1.21x**（T11
重新测量 1.19x）。
T11 修正矩阵为 1.05-1.25x（1/2/4t 1.05-1.19x；8/16t 0.96/0.89x 是 400KB
小输出的任务切分噪声；32/36t 回 1.24/1.25x）。

**剩余限制**：非全长/非连续索引（广播标量等）仍走 generic decode，慢于快路径，
但正确性不受影响；相对 aten 仍 ≥1x。

### 3.7 f32 且 numel ≥ 2^31（防 int32 越界）

**触发**：`self.numel() >= 2^31` 时 `use_i32=false`，f32 也强制 int64 索引——
因为元素偏移必须 ≥2^31 时 int32 装不下。而 8B SVE gather 分支只认
`out∈{f64,i64}`，f32+int64idx 落标量（M4）。

**为什么慢**：语义安全优先于性能——这是**正确的**选择（2^31 元素越界 int32）。
超大张量本身罕见；标量路径相对 aten 通常仍有收益（idx 变 8B 使 linearize
写放大，但 gather 段二者都标量）。

**限制性质**：语义强制的实现空白。可补 f32+i64idx 的 8B 偏移 SVE gather（off
按 8B 存、按 4B 取），未做。

### 3.8 1 线程（T5 已修）

历史 1 线程多形态 0.6-0.9x（M2：linearize 两遍固定开销；SVE 单线程在此形态
并无 MLP 优势）。T5
（bd21485）用**单遍标量**（`gather_single_1d`、1 线程短行 fused）+ nb==1 快路径
修复，表中形态回到 ≥1.07x（T11 复核）：

| 形态 | 修复前 | 修复后 |
|---|---|---|
| 1D pure f32/f64 K=5M | 0.75x / 0.71x | 1.07x / 1.24x |
| x[i,j] 1t | 0.66x | 1.21x |
| rowcopy A=4 1t | 0.87x | 1.39x（T11 2.41x） |
| mask 1t | ~1.0x | **2.1x**（T8）|

详细见 `optimizations.md` §9。

## 4. 为什么不做 fallback 到 aten::index

integration.md §7 有完整调查；此处给出机制层面的简明理由：

### 4.1 慢形态不是"能 fallback 修好"的形态

- **4B 大 K 大值域（最严重）**：同构建 A/B 显示 SVE 4B 与标量持平
  （1.85 vs 1.86 ms），内核内回退标量不会改善；要更快只能减少 cache line
  流量（桶化/预取/行 gather）。而 Python fallback 到 aten 会给**所有**形态
  引入全局开销（见下），不划算。
- **A<4 / 非连续 / 2B-1B**：这些形态**相对 aten 通常是 ≥1x**（§3.2-§3.5
  实测），只是吃不到 SVE 主路径的加速。mask 经 T8 后 1.2-3.8x。fallback 反而可能
  更慢（多了分支）。
- **1 线程**：T5 已修，无需 fallback。

### 4.2 全局开销不可接受（Python wrapper ~45µs/调用）

若在 Python 层做 fallback（`torch.library.get_kernel` + `with_keyset` +
`call_boxed` 调回 true aten）：实测每次 `x[...]` 增加 **~45µs**（小 op
13.7µs→63µs，S1 4ms 场景 +1%）。对本项目——收益主体是 RowCopy 小/中张量
（单次调用本就 <100µs）——45µs 是**全局性回归**。一行"慢就 fallback"的分支
逻辑，会让所有快场景买单。

### 4.3 C++ 内核内回退也追不平

在 C++ 里对慢形态直接调 true aten（绕过覆盖）是可行的（无递归），但：
- 需要在 dispatcher 里保留 true aten 的 CPU kernel 引用，绕开本覆盖；
- 4B 场景即使调 true aten，也只是把历史 0.18x 拉回 ~1x（`x[:,i]` 在 T11 前
  0.4-0.7x，A==1 紧凑循环后已 1.1-2.0x），而该判断本身每次要执行；
- 收益仅限极少数形态，复杂度和维护成本不成比例。

**结论**：**不实现 fallback**。慢形态如实记录为硬件/算法边界；可修复的
（T5）已修；不可修的（4B 随机行放大）用桶化/单遍在可行处绕开，其余接受。

## 5. 慢形态判定速查表（给别人/agent 用）

遇到慢查询时按此表定位原因：

| 现象 | 先查 | 最可能原因 | 处置 |
|---|---|---|---|
| f32/i32 1D 大 K，值域大，高线程 <1x | idx 是否 int32、值域 >2048 | M1 64B 行放大 + DRAM 延迟（SVE≈标量） | 访存模式，接受；小值域走桶化/预取 |
| 高线程 mask 0.35x | 掩码路径 | M3 nonzero 串行 | T8 已修复（§3.5） |
| 1 线程 <1x | 是否命中 T5 单遍路径 | M2 两遍模型 | 已修；确认构建含 bd21485 |
| int8/uint8 1-16t 0.9-1.0x | dtype/值域 | M4 无 SVE 内核 + 64x 行放大 | 接受；高线程 1.9-2.0x |
| complex 1t 0.95x | dtype | M4 走标量 | 接受；32/36t 1.8-2.8x |
| rowcopy A∈[2,3] <1x | A 值 | M3 短行 | 相对 aten 通常 ≥1x |
| 超大 f32（≥2^31 元素） | numel | M4 防溢出强制 i64 | 语义正确优先 |

## 6. 复现命令

```bash
# 4B 大 K 大值域（SVE/标量 A/B 在 RESULTS §3.7；相对比值受共享箱噪声影响）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=<quiet_node> python benchmarks/bench_gate.py

# 回归扫描（1/8/32 线程，pin node 5 静默节点）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py

# true-aten 值级对拍（正确性，非性能）
python tests/verify_against_aten.py --mode both
```

> 共享箱噪声：相对速度跨轮波动可达 3x（performance.md §2）。1 线程数字最稳；
> 高线程以多轮中位数为准；硬件限制以同构建 A/B 为准（RESULTS §3.7；
> 旧独立探针 §8.2 已撤回归因）。
