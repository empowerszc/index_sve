# SVE512 内核详解

本文档走读 `gather.cpp` 中各内核的实现、选择门槛与性能特性。所有 SVE 指令按
`-msve-vector-bits=512` 编译（VL=512-bit，`svcntb()==64`），运行时用
`sve_512_ok()` 门控。

> **一句话**：内核选择只由三个量决定——**A**（每次访问连续元素数）、**K**
> （索引元素数）、**B**（索引广播出的前导行数）。A 大走 RowCopy，A==1 走
> PureGather，K 小走 `gather_direct`/`small_single_index`。
>
> **名词**：**SVE** = ARM 可伸缩向量扩展；**VL** = 向量长度（512-bit）；
> **gather** = 按地址向量取元素（随机）；**rowcopy** = 用 `ld1/st1` 搬连续
> 一段；**prefetch（预取）** = 提前把 cache line 拉进缓存掩盖延迟；**NT**
> = non-temporal，绕过缓存直接写内存（适合只写一次的大块）；**bucket（桶化）**
> = 值域小（≤2048）时先建“值→行”查表，把随机 gather 变成顺序查表；
> **linearize** = 多维下标换一维偏移。

### 内核选择速查

| 条件 | 走哪个 | 为什么 |
|---|---|---|
| 多索引同 shape/连续、n<=65536 | `small_multi_index`（T13） | 省掉 normalize + 广播解码 |
| K<1024 且单整数下标/简单 1D | `gather_direct`（A==1 紧凑循环） | 省掉 linearize 的分配与一遍遍历 |
| A>=4 且 after 维连续、单索引 | `gather_rowcopy_raw` | 连续块摊薄 64B 行放大 |
| A>=4 且 after 维连续、多索引 ≥16t | `gather_rowcopy_fused` | 跳过 linearize，高线程避免串行写/读 |
| after 维非连续、存在 ≥4 连续 run | `strided_rows_raw` | 按 run 块拷贝，减少逐元素 load/store |
| A<16 单索引 | raw rowcopy（多线程也跳过 linearize） | 小行宽 linearize 占比高 |
| 2<=A<4 | `gather_fused`（T11 1t）/ 标量 | 块太短，SVE 收益被开销吃掉 |
| A==1、8B 元素 | SVE `svld1_gather_u64index` | 8 个元素一条指令，带宽受限 |
| A==1、4B 元素 | SVE `svld1_gather_u32index` | 4B 只用 1/16 行，受行放大限制；T15-C 起多线程 f32/i32 数据 + i64 下标且源 >512KB（L2 实测 768KB，取其下保守界）时走专用单遍内核 `pure_gather_sve_4b_i64idx`（向量检查/归一 + uzp1 收窄后复用 u32index）；小源标量更快（每元素标量 L1 命中 ~7.1ns vs 4B gather 微码/发射下限 ~18ns，大源反转为 ~19 vs ~10ns，见 degradation-analysis.md §5.9） |
| 值域 ≤2048 | `bucket_gather_pure(_raw)` | 查表替代随机地址 |
| 2B/1B/complex/非连续 after | 标量单遍（T11 起） | 无 SVE 内核，避免两遍 |
| bool mask | `gather_mask` 单遍压实 | 跳过 `nonzero()` 物化 |

## 1. 8B 元素 gather（f64/i64）

```cpp
pure_gather_sve_8b_f64 / pure_gather_sve_8b_s64
```

- 核心指令 `svld1_gather_u64index_f64(pg, base, off)`：`off` 为 **u64 元素偏移**
  （硬件按 esize=8 自动缩放）。`svcntd()==8`（每向量 8 个 8B 元素）。
- 主循环 `pf>0` 时双发：先 `svprfd_gather_u64index(pg, src, off[i+pf], SV_PLDL1KEEP)`
  预取前 16 个向量位置，再 gather 当前向量 —— 隐藏随机访问延迟。
- 尾数用 `svwhilelt_b64(i, n)` 半掩码向量。
- NT 变体：`svldnt1_gather_u64index_f64` + `svstnt1_f64`（HPC_INDEX_NT=1）。

**性能**：8B gather 是 1D/大随机 gather 的主要收益来源。T11 定点复测（node 5，
1D f64 K=2M）hpc 绝对带宽 1/2/4/8/16/32t = 0.61/1.08/2.09/3.91/6.87/6.80 GB/s，
相对 aten 1.03-1.26x；K=10M 时 32/36t 为 9.1-9.2 GB/s（aten 4.5-4.8 GB/s）。旧文档"4-38 线程稳定 5.0 GB/s"的线程标签不可信，已由
RESULTS §2 的修正矩阵取代（`torch.set_num_threads` 对扩展生效问题见 §7）。

## 2. 4B 元素 gather（f32/i32）

```cpp
pure_gather_sve_4b_f32 / pure_gather_sve_4b_s32 / pure_gather_sve_4b_i64idx（T15-C，§2.1）
```

- `svld1_gather_u32index_f32(pg, base, off)`：`off` 为 **u32 元素偏移**（esize=4）。
  `svcntw()==16`（每向量 16 个 4B 元素）。
- 预取 `svprfw_gather_u32index`；NT 时 store 换 `svstnt1_f32/s32`（**无 4B NT
  gather 指令**，gather 本身始终常规 load）。
- 门控：`pure_gather_sve_4b_f32/s32` 仅 `idx32`（线性化后 int32 下标）且
  dtype∈{f32,i32}；原始 i64 下标的 4B gather 走 §2.1（多线程 + 源 >512KB
  才启用）。

**⚠ 已知访存限制（重要，T11 复核）**：4B 随机 gather 的瓶颈是 **64B cache line
放大 + DRAM 延迟**，不是 SVE 指令的 MLP。同一个真实内核、同一随机模式做 A/B：

| 内核 | src 8MB（8t） | src 256MB（38t） |
|---|---|---|
| SVE 4B gather | 1.85 ms | 7.46 ms |
| 标量循环 | 1.86 ms | 7.45 ms |

两者持平：换标量、改指令都无效。早期独立探针（38t 0.75 vs 3.4 GB/s）没有在
端到端复现，已撤回归因（RESULTS §3.7）。8B 路径更快是因为同样条数的 line 携带
2x 有用字节（8B 行放大 8x vs 4B 16x），而非指令 MLP 差异。因此 4B 大 K 大值域
的可行方向是减少行流量（T4 小值域桶化、预取、行 gather），而不是换指令；
f64 8B 路径不受影响。

### 2.1 T15-C 大源直连 SVE `pure_gather_sve_4b_i64idx`（gather.cpp）

上述"换标量无效"的结论对**中等源**成立；T15-C 补上了**大源**（L3/DRAM 驻留）
入口：标量循环的乱序重叠有限，摊不掉大源延迟；SVE gather 每迭代 8 路在飞
负载，实测大源每元素 ~10ns vs 标量 ~19ns。此前缺入口的原因是 ACLE 没有
32 位数据的 64 位下标 gather。

**内核**（每迭代 `svcntd()`=8 个元素，主循环 + `svwhilelt_b64` 尾部）：

- `svld1_s64` 向量读 8 个原始 i64 下标；
- 向量检查 `v>=size || v<-size`（命中才回退逐元素 `TORCH_CHECK_INDEX`），
  负索引掩码加 `+size` 归一；
- `uzp1` 抽各 i64 低 32 位 → u32 元素偏移（调用方保证 `size<2^32` 且
  `stride==1`，低 32 位无损失），复用 `svld1_gather_u32index_f32/s32` 与
  `svprfw_gather_u32index` 预取（PREFETCH_DIST）；store 固定常规（无 NT）。

**入口 `gather_single_1d_sve`**（op.cpp 在 `gather_bucket_pure_raw` 分支内
短路，不经 linearize，也省掉 bucket 值域扫描的两遍 fork-join）：

- 门控：1D 单索引 A==1、原始 i64 下标、dtype∈{f32,i32}、`stride0==1`、
  `size<2^32`、SVE512、**线程数>1**、**源 >512KB**（L2 实测 768KB，取其下
  保守界）；只显式实例化 float/int32；
- `at::parallel_for(0, K, adaptive_grain(K))` 分块调内核；计数器
  `g_single_1d_sve`；
- **小源与 1 线程不走**：小源标量更快（L1 命中 ~7.1ns/元素 vs SVE 微码/
  发射下限 ~18ns，双内核复现）；1 线程仍走 `gather_single_1d` 标量单遍。
  交叉点 ≈L2，机理见 degradation §5.9；
- 实测：1D f32 K=10M idx64 8t **2.22x**（hpc 绝对时间 -40%），8-36t
  1.44-2.22x（optimizations §19、degradation §5.9）。

## 3. 行拷贝 RowCopy（A>1 连续块）

```cpp
rowcopy_sve<T, IdxT, NT> → copy_contig_sve<T, NT>
```

- 条件：`A>1 && after_contiguous && sve_512_ok() && A>=4 && dtype∈{f32,f64}`。
  `after_contiguous` 保证 `col_off[c]==c`（dims_after 连续），元素
  `out[p*A + c] = src[row_off[r] + idx[k] + c]` 变成**一段连续 A 拷贝**。
- `copy_contig_sve`：8B 用 `svld1_f64/svst1_f64`，4B 用 `svld1_f32/svst1_f32`；
  **4×VL 多寄存器展开**维持内存级并行，之后 VL 步进 + `whilelt` 尾数；NT 时
  store 换 `svstnt1_*`。
- **≤64B 小行/短 run 走 `memcpy`**：4-16 个 f32 或 4-8 个 f64 正好 1-2 条
  cache line，masked SVE 的谓词构造/循环开销大于一次 `memcpy`；非连续尾维的
  40B/64B run 也走这里（实测 run10 由 0.72-0.89x 提升到 0.98-1.42x）。
- 行拷贝只读一次 idx（`src + row_off[r] + idx[k]`），A 元素连续向量化 ——
  这是 S1 spconv / P1 图消息路径 ~4x（1 线程）/ ~1.9x（38 线程）的收益来源。

**门槛**：T1 起从 A>=16 放宽到 **A>=4**（微基准 A-sweep 显示 A=4-16 短行在
1 线程即有 1.1-17x 收益，A>=16 长行路径不变）。A∈[2,3] 落标量 `scalar_gather`。

### 3.1 M1 raw rowcopy（T2，跳过 linearize）

```cpp
rowcopy_raw_single<T, IdxT, NT>（入口：gather_rowcopy_raw，op.cpp 短路）
```

- 条件：**单索引 && after_contiguous && A>1 && (A>=4096 || A<16 || (A<=64 && B*K<=65536)) && f32/f64**。
  A<16 由 T12 补齐：小行宽时 linearize 的固定开销占比高，多线程也直接 raw。
  T15 放行 A 17..64 的小 K（B*K≤65536）形态（2D A=16 K=8192 曾落 linearize 两遍，
  32/36t 0.73/0.76x）；大 K 不放行——linearize 的预校验偏移+无检查拷贝在冷源
  行拷贝下更快（raw 慢 12-32%）。
- 与 `rowcopy_sve` 唯一区别：不消费 linearize 的 idx 数组，而是直接遍历
  `spec.indices[0]` 原始下标（int32/int64），内联 `v>=size || v<-size` 检查 +
  负索引归一，随后 `copy_contig_sve` 拷贝 A 连续块。
- 省掉 linearize 的一次 broadcast 解码 + 一次 K×N idx 写入；对单索引大块
  （A 大）收益显著：M1 weight.view(K,C,o)[ki] 1 线程 2-13x。
- `B==1` 单独分支：`k=p`，去掉每行 `p/K`、`p%K`；A=8/15 多线程再提升
  1.2-1.4x（见 §3.2）。

### 3.2 非连续尾维 strided-rows（T12）

```cpp
strided_rows_raw<T, IdxT, NT>（入口：gather_strided_rows_raw，op.cpp 短路）
```

- 触发：单索引、`!after_contiguous`、A>1、索引连续、f32/f64，且 `col_off` 中
  存在长度 ≥4 的连续 run。
- `make_col_runs(col_off, A)` 把相邻 `col_off[c+1]==col_off[c]+1` 合并成
  `{dst_col, src_off, len}`；每行只算一次基址 `src + row_off[r] + idx[k]*stride`，
  每个 run 一次 `copy_contig_sve`（≤64B 自动 `memcpy`）。
- 例：尾维 `[64,200,16] -> permute(1,0,2)` 的 `col_off` 是 64 段×16 连续；
  A=1024 时从 1024 次标量拷贝变成 64 次 64B 拷贝。实测 1t 2.23x、8t 1.71x。
- **run=1 不启用**：纯转置（tail 64、stride 20000）没有连续后缀，块拷贝只能退化
  为逐元素；保留 `gather_fused`，原因见
  `index_sve_vs_fastindex_gap_20261023.md` §19.1。

### 3.3 多索引 raw rowcopy（T12）

```cpp
rowcopy_fused_sve<T, NT>（入口：gather_rowcopy_fused，op.cpp 短路）
```

- 触发：多索引、`after_contiguous`、A>=4、f32/f64，且 **线程数 ≥16**。
  n≤65536 且索引同 shape/连续时已被 T13 `small_multi_index` 提前接走，本节主要
  覆盖 n>65536 或广播/非连续索引。
- 复用 `gather_fused` 的预提取 + broadcast 解码，但内层改为
  `copy_contig_sve` 整块拷贝，不物化 idx。
- 门槛来自实测：1-8t 时 linearize 的写+读可摊销，raw 反而慢 9-23%；16t 后
  linearize 的串行段占比上升，raw 提升 1.12-1.48x。**低线程仍走旧路径。**

### 3.4 多索引小 K 入口快路径（T13）

```cpp
small_multi_index（op.cpp 入口，normalize 之前）
```

- 条件：索引数 ≥2、全是 int32/int64、**同 shape 且连续**、self 连续、索引覆盖
  self 前 nidx 维、`n=idx.numel()<=65536`。
- 不经过 `normalize_index`：直接 `out_shape = idx.shape + self.shape[nidx:]`；
  行内偏移一次预校验/预计算 `off[k] = Σ v_j[k]*stride_j`。
- 拷贝：扫描 offset 检测 `dom==1`（全同），A==1 用 `std::fill`，A>1 循环
  `memcpy` 同一行；非全同时 A==1 标量、A≥2 逐行 `memcpy`。
- 并行：`n*A >= 2048` 才 `at::parallel_for`；更小的短调用串行，避免线程池
  起停成本。
- 不覆盖广播、非连续索引、非连续 self、n>65536；这些回退通用路径。
- 实测 K=512：A=1 1t/8t/36t 为 1.59/1.99/1.67x，A=4 为 1.90/2.31/1.64x，
  A=64 1t 3.41x（相对 aten）。

## 4. 标量回退 `dispatch_scalar`

```cpp
scalar_gather(src, idx, dst, row_off, col_off, B, K, A)
```

- `at::parallel_for(0, B*K, adaptive_grain(B*K))`；每位置 `dst[p*A+c] = src[row_off[r]+idx[k]+col_off[c]]`。
- 覆盖所有 SVE 分支不满足的输入：2B/1B dtype（half/bf16/i16/i8/u8/bool）、
  complex、f32 且 numel>=2^31（idx 自动为 int64）、A∈[2,3]、非连续 after-col、
  非 512-bit VL 运行时、`idx32` 与 dtype 不匹配等。
- **正确性关键**：此路径对所有 dtype/形状都正确（逐元素地址计算），是"任何
  门控外的输入类"的安全落点。

### 4.1 T3 融合单遍 `gather_fused`

```cpp
gather_fused<T>（入口：gather.cpp 包装，op.cpp 最后分支）
```

- 条件：**A<4 或非连续小块**（A==1 纯 gather 与 A>=4 连续 rowcopy 之外的其余）。
- 与 `gather_direct` 同源：并行循环内先算偏移（内联边界检查/负索引规整），再
  就地标量拷贝；**不物化 idx 数组**（省一次 K×N 分配 + 一次遍历）。
- 与 T2 互斥（T2 只处理单索引大块）。A/B 实测：A>=4 连续块 linearize+rowcopy
  仍快于 fused（单线程 C=8 1.05x vs 0.69x；C=64 3.85x vs 1.50x），故 fused 只
  用于 scalar 小块路径。

### 4.2 T4 小值域桶化 `bucket_gather_pure`

```cpp
bucket_gather_pure<T, IdxT>（gather_index 内，先于 SVE 分支）
```

- 条件：`pure`（A==1）+ **偏移值域 ≤2048**。先并行扫一遍 linearize 后的 idx
  求 min/max（每 chunk 局部 min/max，串行合并，无锁），值域超限则返回 false
  落到 SVE/scalar。
- 值域 ≤2048 时构建 L1/L2 常驻查找表（4B 表 8KB / 8B 表 16KB），标量循环
  `dst[p] = table[row_off[r] + idx[k] - mn]`。
- **动机**：小值域时源数据块常驻 L1/L2，标量查表命中缓存；4B SVE gather 仍受
  64B 行放大限制（§2）。T11 复测 1D f32 K=2M、值域 2048：1-36t 相对 aten
  0.96-1.34x（1t 1.23x，16t 0.96x 为并行开销/噪声）。

### 4.3 T4 raw 桶化 `bucket_gather_pure_raw`

```cpp
bucket_gather_pure_raw<T, IdxT>（op.cpp 短路，单索引 B==1 A==1）
```

- 条件：**单索引 && B==1 && A==1**（1D 纯 gather，M3 keys[order]、M4 inverse
  等）。完全跳过 linearize，直接对 `spec.indices[0]` 原始下标求值域并查表
  （B==1 时偏移 == idx[k]，无 broadcast 解码）。
- 实测：M3 keys[order] int64[2048] E=54k 从 0.2x → 1.4-2.6x（1t）与 8-12x
  （32t）。

### 4.4 T5 1 线程单遍路径 `gather_single_1d` / fused nb==1 快路径（bd21485）

**背景**：回归扫描发现 1 线程多形态劣化（0.6-0.9x），根因是 linearize 两遍模型
固定开销 + SVE gather 单线程在此形态不占优（§2），且 fused/linearize 内层循环
每次访问 at::Tensor 方法（引用计数 + 虚调用）。2+ 线程多数占优。

```cpp
gather_single_1d<T, IdxT>（gather.cpp 新增，op.cpp 短路：1 线程单索引 B==1 A==1 大值域）
gather_fused nb==1 快路径（all_full_contig：所有索引 numel==K 且连续时直读 idx_j[k]）
linearize 同样预提取 + nb==1 快路径
```

- **T5-1** `gather_single_1d`：单索引 1D 纯 gather，1 线程跳过 linearize 直接
  标量单遍（read idx → 边界检查 → 负索引归一 → `dst[k]=src[v]`）。下标张量属性
  循环外一次提取。1D f32 K=5M 0.75x → 1.07x，f64 0.71x → 1.24x。
- **T5-2** 1 线程单索引短行（A<4）改走 `gather_fused` 单遍：rowcopy A=4
  K=200k 0.87x → 1.39x。T11 起 A>=4 连续块改回 raw rowcopy（SVE 整块拷贝）：
  A=4 1t 2.41x、A=16 1t 3.58x、S1（A=64）1t 4.46x（T13 复测 1.87x/
  2.14x/3.04x，run 间差异主要来自共享箱）。
- **T5-3** fused/linearize 预提取索引张量属性 + nb==1 全长连续快路径（跳过
  bfactor 混合进制广播解码，直读 `idx_j[k]`）：x[i,j] 双全长索引 0.66x → 1.21x
  （T11 修正为 1.19x）
  （linearize 4.84 → 1.10ms）。
- **路由边界**：多索引 A==1（x[i,j]）在 1 线程**保持 linearize+SVE**；
  blanket "1 线程全 fused" 实测更差（x[i,j] 0.49x、x[:,i] 0.28x），因 fused 解码
  开销在广播/多索引形态超过两遍模型的收益。单 mask 由 T8 `gather_mask` 单遍
  压实处理（见 §9）。

## 5. 快速路径 `gather_direct`（K<1024）

- 跳过线性化分配，直接遍历原始下标张量（`spec.indices[j]` 逐个元素读
  int64/int32），内联 `v>=size || v<-size` 检查 + 负数归一，累加
  `v * indexed_strides[j]`。
- N-D 下标按 `broadcast_shape` 混合进制解码（bfactor），**非** `k*stride(0)`
  （A7 修复 —— 若回归成简单 stride 会破坏 N-D 下标语义）。
- 等价 ATen `Indexer` 的逐元素遍历，故无需线性化临时缓冲。
- **T11 A==1 快路径**（`gather_direct` 入口，`K<1024`）：`A==1 && 单索引 &&
  全长连续 && after_contiguous` 时先把
  K 个下标校验/归一化一次到小缓冲，再跑无检查的紧凑循环。`x[:,i]`（B 大、K 小）
  从 0.18x 提升到 1.1-2.0x（1t 2.0x）；剩余差距见
  degradation-analysis.md。

## 6. 预取与 NT 决策（Task 8 结论）

- **PREFETCH_DIST=16**：微基准扫描 {0,16,32,64} 后定默认（16 最优）。
- **HPC_INDEX_NT=1 默认关**：NT 开关在 608 线程共享箱上的增量在噪声带内
  （§3 RESULTS.md），无可解析收益；保留为 opt-in。

## 7. 线程并行

- 所有大路径 `at::parallel_for(0, B*K, adaptive_grain(B*K))`（T0，
  `grain.h`：`max(1, min(n, 65536, n/(nt*8)))`）；`-fopenmp` 必须（否则
  `parallel_for` 串行化，历史根因见 architecture.md §7）。
- 线程数：T11 起内核入口调用 `omp_set_num_threads(at::get_num_threads())`，
  `torch.set_num_threads(t)` 对扩展生效。扩展链接 BiSheng libomp、ATen 链接
  libgomp，二者 ICV 独立；此前只设 torch 的线程数时，扩展实际用 libomp 默认
  team（本机 affinity 38），导致 2-36t 的旧数据标签失效（RESULTS §3.6.1/§3.9）。
- 鲲鹏 920F：608 核 / 32 NUMA，单节点 38 核。共享箱上 608 线程数字噪声大，
  **以定点核复测为准**（performance.md §8）。

## 8. 修改内核时的门槛清单（防回归）

改 `gather_index` 分支条件时，逐一核对：
1. 新分支条件是否覆盖其内核的全部不变量（idx 位宽、dtype、A、连续度、VL）。
2. 不满足的输入是否落到正确的 `dispatch_scalar`。
3. int32 下标与 f32 numel>=2^31 的交互是否仍然安全（use_i32 决定 idx dtype）。
4. N-D 下标在 `gather_direct` 的 bfactor 解码不可丢。
5. 跑全量测试 + 两进程 true-aten 对拍（semantics.md §6）。

## 9. T8 mask 单遍压实 `gather_mask`（gather.cpp）

```cpp
gather_mask(self, mask)  // op.cpp mask_fast_path 调用
```

- 条件：`indices` 仅一个 bool/byte mask、覆盖 self 前 `md` 维且形状匹配、
  mask 与 self 均连续、均在 CPU。否则回退 `normalize_index` 的 nonzero 路径。
- 第 1 遍按 chunk 统计 true 数（`n<1MB` 串行，避免并行任务开销），前缀和求
  `nnz` 与每 chunk 写偏移；`at::empty({nnz, *trailing})`。
- 第 2 遍按 chunk 并行：顺序扫描掩码，命中则把 `self[p*A : p*A+A]` 整行拷到
  `out[w*A]`；f32/f64 且 `A>=4` 用 `copy_contig_sve`，其余 dtype/短行标量拷贝。
  chunk 内 `w` 局部递增，无原子操作。
- `nnz==0 || A==0` 直接返回空输出；空掩码 8t ~0.15ms。

## 10. T9 小 K 单整数下标精简路径（op.cpp）

单整数下标、末维 `K<=4096`、self/idx 连续、CPU 时，`small_single_index` 直接在
`index_impl` 入口按 idx 平坦顺序取行：`out[p] = self[idx[p]]`，A>=2 用 `memcpy`
整行拷贝（T14：A=3 的 3 次迭代标量循环每行耗时 ~2x 于 memcpy（shortq 实测
35µs vs 16µs，A=2 仅 ~1.1x）），`n*A < max(16384, 512×线程数)` 时串行（T15 起按
线程数缩放：38t 下 16384 元素的 fork-join 净亏 ~10µs；并行任务开销大于收益；只看 n
会让 2D A=16 K=4096 误判为串行，8-36t 掉到 0.55-0.66x，T11 已修）。绕过
`normalize_index` 的 vector 分配/transpose 分析与 `gather_direct` 的内层
`at::Tensor` 访问。`K>4096` 仍走 SVE/桶化主路径。
