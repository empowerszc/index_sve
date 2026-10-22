# 慢场景与回退路径深度分析

本文档回答一个问题：**为什么 index_sve 在某些形态下性能不佳（甚至慢于
aten::index），受限于什么**。这是"回退到 aten"类问题（integration.md §7）背后的
完整技术分析——每个慢形态的触发条件、慢的根本机制、硬件/算法限制、实测数据、
以及为什么最终选择"不 fallback"。

> 读者：想理解本项目性能边界的人 / agent。结论摘要可直接看 §1 与 §4；§3 是
> 逐场景深度分析。

## 1. 总览：三个根本机制

所有慢形态都可归因于以下三个机制之一（或叠加）：

| 机制 | 本质 | 影响的形态 | 能否软件修复 |
|---|---|---|---|
| **M1 内存级并行度（MLP）瓶颈** | 随机 gather 是延迟敏感操作；920F 的 SVE gather 指令能发的**在途内存请求（outstanding loads）太少**，向量宽度弥补不了随机访问的延迟 | 4B（f32/i32）大 K 大值域 pure gather；高线程数下尤甚 | 部分（软件预取/多路标量），硬件上限不可消除 |
| **M2 两遍模型固定开销** | linearize 先物化偏移数组、gather 再读它 = 两次遍历 + 一次分配；小 K 时固定开销占比高；1 线程无并行加速 | 1 线程所有需 linearize 的形态（T5 已大部分修复） | 已修（单遍路径） |
| **M3 串行/低并行阶段** | 某些形态内部有无法并行化的阶段（如 mask 的 nonzero 定位、scalar_gather 的 col_off 解引用） | mask；A<4 短行 rowcopy | 结构性问题，难修 |
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
  ├─ K < 1024                    → gather_direct（快速路径，无 linearize）
  ├─ 单索引 && A>=4096 && 连续 && f32/f64 → M1 raw rowcopy（skip-linearize）★主路径
  ├─ 单索引 && B==1 && A==1      → T4 raw 桶化（值域≤2048 查表）
  │     └─ 值域>2048 && 1 线程   → gather_single_1d（T5 单遍标量）
  ├─ 1 线程 && 单索引 && A>1     → gather_fused（T5 单遍）
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
| 4B 大 K 大值域 pure gather（idx32+f32/i32） | 4B SVE gather（本身低效） | M1 |
| mask（bool） | linearize(nonzero) + gather | M3（nonzero 定位串行）+ M2 |
| 1 线程单索引 A==1 大值域 | gather_single_1d（T5，已追平） | M2（已修）|

## 3. 逐场景深度分析

### 3.1 4B（f32/i32）大 K 大值域 pure gather —— 硬件 MLP 上限（最严重）

**触发**：`pure && idx32 && sve_512_ok() && out∈{f32,i32}`，即 int32 索引的
1D 纯 gather，且值域 >2048（否则 T4 桶化接管）。典型：`x[i32]`，x 为 10M 以上
f32 张量、i 为 int32 随机下标。

**为什么慢**：这是**纯内存延迟敏感**的随机 gather：每个输出元素
`dst[p] = src[idx[p]]`，src 地址完全随机。性能由"能同时有多少个在途内存
加载"决定——这就是 **MLP（Memory-Level Parallelism）**。

- **标量循环**：编译器/CPU 可软件流水展开，多个 load 同时飞行 → MLP 高。
  实测 38 线程标量 gather ~3.4 GB/s。
- **920F 的 SVE 4B gather**（`svld1_gather_u32index_f32`）：一条指令取一个
  16 元素向量，但**指令的硬件实现限制了在途请求数量**（独立探针：38 线程仅
  ~0.75 GB/s）。向量宽度（16 元素/次）救不了随机访问——带宽没到瓶颈，延迟
  隐藏能力才是瓶颈。
- 对照：8B gather（f64/i64，`svld1_gather_u64index_*`）在 920F 上高效
  （~5.0 GB/s），因为 8B 路径硬件实现不同。**4B 的弱是 920F 具体指令特性**，
  不是通用 SVE 缺点。

**限制性质**：硬件指令设计，非实现缺陷。不可通过改代码消除；只能绕：
- 值域 ≤2048 → T4 桶化（L1 常驻表，标量随机读命中缓存）✅ 已实现；
- 值域大 → 只剩 SVE 4B 或标量。**实测标量（dispatch_scalar）与 SVE 4B 相当或
  略好**（integration.md §7：clang `-O3` 会把标量循环自动向量化成同样低 MLP 的
  SVE gather，`#pragma clang loop vectorize(disable)` 后 ~0.99 GB/s，仍 < ATen 2.9
  GB/s）——所以"回退到标量"也救不了，回退到 ATen 又受 §4 开销制约。

**实测**（本机 node 5，interleaved median，30 iters；RESULTS §10 完整多线程扫描）：

| 线程 | 1t | 2t | 4t | 8t | 16t | 32t | 38t |
|---|---|---|---|---|---|---|---|
| 1D f32 K=10M 相对 aten | 1.07x | 13.9x | 16.6x | 6.0x | 2.6x | 2.6x | 2.1x |

当前构建的相对比值已无 <1x 格（T0-T5 后），但**硬件 MLP 上限本身仍在**——真正
证明它的是独立探针（§3.1 的 0.75 vs 3.4 GB/s），那个数字不随构建变化。历史
§1/§2 的 0.18-0.24x（608 共享箱）是共享箱噪声 + 608 全核模式的产物；相对 ATen
的比值受共享箱 aten 绝对速度影响大，跨轮波动可达 3x（performance.md §2 方法论）。
**结论以探针为准：4B 大值域 gather 仍是本项目最确定的慢形态，只是当前构建下
不再表现为 <1x。**

### 3.2 2B/1B（half/bf16/i16/i8/u8/bool）与 complex 纯 gather

**触发**：`pure && A==1` 但 out dtype 不在 {f64,i64,f32,i32}。SVE 内核只写了
8B/4B 两种元素宽度的 gather/rowcopy，2B/1B 与 complex 没有向量实现。

**为什么慢**：纯 `dispatch_scalar` 标量逐元素。没有内核覆盖（M4）。但要注意：
- 这不是"比 aten 慢"，只是"没吃到 SVE 加速"。aten 也是标量 TensorIterator。
- 实测 half K=10M：1t 1.43x、8t 5.14x、32t 0.68x。低线程仍有收益（adaptive_grain
  单遍 + 少一次 aten 的调度）；高线程被 aten 标量摊平反超。
- complex64 K=5M：8t 4.76x、32t 1.83x —— complex 是 8B 元素但**走标量**（没有
  complex 的 SVE gather 内核），收益来自 T0/T4 的 linearize 优化而非向量化。

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

**但实测并不差**：本机 A=3 K=10M：1t 1.93x、8t 7.58x、32t 2.57x。低线程
significant 收益（单遍 + 自适应粒度），高线程回落但仍 >1x。**这形态是"慢于 SVE
主路径"而非"慢于 aten"**——相对 aten 其实是赢的，只是赢的幅度不如长行。

**限制性质**：结构性（短行无法向量化拷贝）。不可修，但也不是回归。

### 3.4 rowcopy 非连续 after 维

**触发**：`A>1` 但 after 维 stride 不连续（如 `x.t()` 后的 `x[i]`、或
`x[:, i, :]` 中 i 后还有非连续段）。`rowcopy_sve` 要求 `after_contiguous`
（`col_off[c]==c`），否则整行不是一段连续内存。

**为什么慢**：必须逐元素解引用 col_off，SVE 连续 `ld1/st1` 用不上（M3/M4）。
同 §3.3——相对 aten 通常仍有收益，只是吃不到向量化。

**限制性质**：结构性。无法在现有模型下规避（除非先做一次内存重排，成本更高）。

### 3.5 mask（bool 索引）

**触发**：bool 掩码 → normalize 先 `index.nonzero()` 展开成位置下标，再走
linearize + gather。

**为什么慢**：**两段式 + 一个串行环节（M3+M2）**：
1. `nonzero()` 本身是一次遍历，且输出长度取决于掩码密度（数据相关，难以
   高效并行/预分配）；
2. 之后才是 linearize + gather，等价于"定位（nonzero）+ 取值（gather）"两次
   完整的内存遍历；
3. aten 的 mask 路径同样有 nonzero 成本，但本项目在**高线程**下比值退化
   （P3 mask 38t 0.35x）——因为 nonzero 阶段并行度差，把整体拖住。

**实测**（本机）：mask nnz≈0.3M：1t 0.99x、8t 1.11x、32t 1.06x。全程平 ~1x，
无线程收益。历史 P3（nnz≈60k C=16）：1t 1.11x、38t 0.35x。

**限制性质**：nonzero 的串行性 + 两段式。理论上可做"直接行式 gather 并行"
（跳过 nonzero 单独成段），列为未来优化（status.md §7）。当前模型用 mask 少、
低线程接近 1x，可接受。

### 3.6 多索引广播形态（x[i,j]）

**触发**：多索引且形状需广播（如 `x[i, j]`，i/j 不同长或含标量）。走
linearize（混合进制 decode）+ gather。

**为什么慢（历史）**：1 线程时 linearize 两遍模型固定开销 + 每次访问
at::Tensor 方法（引用计数/虚调用）→ 0.66x。**T5 已修复**（nb==1 全长连续
快路径 + 属性预提取）：linearize 4.84→1.10ms，x[i,j] 1t 0.66x→**1.21x**。
多线程本就占优（2t 5.98x、8t 3.02x、32t 1.13x）。

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

历史 1 线程多形态 0.6-0.9x（M2：linearize 两遍 + SVE 单线程 MLP 低）。T5
（bd21485）用**单遍标量**（`gather_single_1d`、1 线程 fused）+ nb==1 快路径
修复，全部 ≥1.0x：

| 形态 | 修复前 | 修复后 |
|---|---|---|
| 1D pure f32/f64 K=5M | 0.75x / 0.71x | 1.07x / 1.24x |
| x[i,j] 1t | 0.66x | 1.21x |
| rowcopy A=4 1t | 0.87x | 1.39x |
| mask 1t | ~1.0x | ~1.0x（噪声级）|

详细见 `optimizations.md` §9。

## 4. 为什么不做 fallback 到 aten::index

integration.md §7 有完整调查；此处给出机制层面的简明理由：

### 4.1 慢形态不是"能 fallback 修好"的形态

- **4B 大 K 大值域（最严重）**：fallback 到 aten 会让该形态变好（aten 2.9
  GB/s 标量），但**代价是给所有形态引入全局开销**（见下）。且本项目自己的
  标量回退（0.32-0.55 GB/s，auto-vec 后 0.85-0.99 GB/s）也远低于 aten——
  单纯"内核内回退标量"追不平，必须真调 aten。
- **mask / A<4 / 非连续 / 2B-1B**：这些形态**相对 aten 通常是 ≥1x**（§3.2-§3.5
  实测），只是吃不到 SVE 主路径的加速。fallback 反而可能更慢（多了分支）。
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
- 4B 场景即使调 true aten，也就是从 0.18x 变 ~1x，而该判断本身每次要执行；
- 收益仅限极少数形态，复杂度和维护成本不成比例。

**结论**：**不实现 fallback**。慢形态如实记录为硬件/算法边界；可修复的
（T5）已修；不可修的（4B MLP）用桶化/单遍在可行处绕开，其余接受。

## 5. 慢形态判定速查表（给别人/agent 用）

遇到慢查询时按此表定位原因：

| 现象 | 先查 | 最可能原因 | 处置 |
|---|---|---|---|
| f32/i32 1D 大 K，值域大，高线程 <1x | idx 是否 int32、值域 >2048 | M1 4B SVE MLP 上限 | 硬件，接受；小值域走桶化 |
| 高线程 mask 0.35x | 掩码路径 | M3 nonzero 串行 | 结构，未来优化 |
| 1 线程 <1x | 是否命中 T5 单遍路径 | M2 两遍模型 | 已修；确认构建含 bd21485 |
| half/bf16/complex <1x 高线程 | dtype | M4 无内核 | 接受；罕见 dtype |
| rowcopy A∈[2,3] <1x | A 值 | M3 短行 | 相对 aten 通常 ≥1x |
| 超大 f32（≥2^31 元素） | numel | M4 防溢出强制 i64 | 语义正确优先 |

## 6. 复现命令

```bash
# 4B 大 K 大值域（探针数据在 RESULTS §8.2；相对比值受共享箱噪声影响）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=<quiet_node> python benchmarks/bench_gate.py

# 回归扫描（1/8/32 线程，pin node 5 静默节点）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py

# true-aten 值级对拍（正确性，非性能）
python tests/verify_against_aten.py --mode both
```

> 共享箱噪声：相对速度跨轮波动可达 3x（performance.md §2）。1 线程数字最稳；
> 高线程以多轮中位数为准；硬件限制以独立探针为准（RESULTS §8.2）。
