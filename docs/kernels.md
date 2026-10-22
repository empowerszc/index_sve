# SVE512 内核详解

本文档走读 `gather.cpp` 中各内核的实现、选择门槛与性能特性。所有 SVE 指令按
`-msve-vector-bits=512` 编译（VL=512-bit，`svcntb()==64`），运行时用
`sve_512_ok()` 门控。

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

**性能**：920F 上 8B gather 是高效的（实测 4-38 线程稳定 ~5.0 GB/s，而标量
~3.6 GB/s，见 performance.md）。这是 S4 大随机 gather 与 1D f64 的收益来源。

## 2. 4B 元素 gather（f32/i32）

```cpp
pure_gather_sve_4b_f32 / pure_gather_sve_4b_s32
```

- `svld1_gather_u32index_f32(pg, base, off)`：`off` 为 **u32 元素偏移**（esize=4）。
  `svcntw()==16`（每向量 16 个 4B 元素）。
- 预取 `svprfw_gather_u32index`；NT 时 store 换 `svstnt1_f32/s32`（**无 4B NT
  gather 指令**，gather 本身始终常规 load）。
- 门控：仅 `idx32`（线性化后 int32 下标）且 dtype∈{f32,i32}。

**⚠ 已知硬件限制（重要）**：920F 的 `svld1_gather_u32index_*` **内存级并行度
（MLP）极低**。独立探针（OpenMP，38 线程，node 5）：

| 内核 | 38 线程带宽 |
|---|---|
| SVE 4B gather（`svld1_gather_u32index_f32`） | ~0.75 GB/s |
| 标量 gather 循环 | ~3.4 GB/s |
| SVE 8B gather（对照） | ~5.0 GB/s |

因此 **1D f32/i32 大 K（≥10M）随机 pure gather 显著慢于 aten 标量**
（38 线程实测 hpc ~1.2 GB/s vs aten 6.7 GB/s，0.18x）。这不是实现缺陷，而是
920F SVE gather 指令的硬件特性：向量宽度无法弥补随机 4B 访问的 MLP 瓶颈。
f64 8B 路径不受影响。若未来要在 4B 大 K 场景加速，可考虑预取到 L2/软件流水
深展开，或改走标量+显式多路预取（见 porting.md）。

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
- 行拷贝只读一次 idx（`src + row_off[r] + idx[k]`），A 元素连续向量化 ——
  这是 S1 spconv / P1 图消息路径 ~4x（1 线程）/ ~1.9x（38 线程）的收益来源。

**门槛**：T1 起从 A>=16 放宽到 **A>=4**（微基准 A-sweep 显示 A=4-16 短行在
1 线程即有 1.1-17x 收益，A>=16 长行路径不变）。A∈[2,3] 落标量 `scalar_gather`。

### 3.1 M1 raw rowcopy（T2，跳过 linearize）

```cpp
rowcopy_raw_single<T, IdxT, NT>（入口：gather_rowcopy_raw，op.cpp 短路）
```

- 条件：**单索引 && after_contiguous && A>=4096 && f32/f64**（M1 最热形态）。
- 与 `rowcopy_sve` 唯一区别：不消费 linearize 的 idx 数组，而是直接遍历
  `spec.indices[0]` 原始下标（int32/int64），内联 `v>=size || v<-size` 检查 +
  负索引归一，随后 `copy_contig_sve` 拷贝 A 连续块。
- 省掉 linearize 的一次 broadcast 解码 + 一次 K×N idx 写入；对单索引大块
  （A 大）收益显著：M1 weight.view(K,C,o)[ki] 1 线程 2-13x。

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
- **动机**：920F 的 SVE 4B gather MLP 低（§2），小值域时表查的标量随机读命中
  缓存，实测优于 SVE gather。1D f32/f64 K=1M 小值域在 8-32 线程 1.2-3.1x
  （SVE 0.2-0.9x）。

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
固定开销 + 920F SVE gather 单线程低 MLP（§2），且 fused/linearize 内层循环每次
访问 at::Tensor 方法（引用计数 + 虚调用）。2+ 线程全部占优。

```cpp
gather_single_1d<T, IdxT>（gather.cpp 新增，op.cpp 短路：1 线程单索引 B==1 A==1 大值域）
gather_fused nb==1 快路径（all_full_contig：所有索引 numel==K 且连续时直读 idx_j[k]）
linearize 同样预提取 + nb==1 快路径
```

- **T5-1** `gather_single_1d`：单索引 1D 纯 gather，1 线程跳过 linearize 直接
  标量单遍（read idx → 边界检查 → 负索引归一 → `dst[k]=src[v]`）。下标张量属性
  循环外一次提取。1D f32 K=5M 0.75x → 1.07x，f64 0.71x → 1.24x。
- **T5-2** 1 线程单索引 A>1（rowcopy/scalar 短行）改走 `gather_fused` 单遍：
  rowcopy A=4 K=200k 0.87x → 1.39x（linearize 原占 80% 开销，跳过即追平）。
- **T5-3** fused/linearize 预提取索引张量属性 + nb==1 全长连续快路径（跳过
  bfactor 混合进制广播解码，直读 `idx_j[k]`）：x[i,j] 双全长索引 0.66x → 1.21x
  （linearize 4.84 → 1.10ms）。
- **路由边界**：多索引 A==1（x[i,j]）在 1 线程**保持 linearize+SVE**；
  blanket "1 线程全 fused" 实测更差（x[i,j] 0.49x、x[:,i] 0.28x），因 fused 解码
  开销在广播/多索引形态超过两遍模型的收益。单 mask 由 T8 `gather_mask` 单遍
  压实处理（见 §6）。

## 5. 快速路径 `gather_direct`（K<1024）

- 跳过线性化分配，直接遍历原始下标张量（`spec.indices[j]` 逐个元素读
  int64/int32），内联 `v>=size || v<-size` 检查 + 负数归一，累加
  `v * indexed_strides[j]`。
- N-D 下标按 `broadcast_shape` 混合进制解码（bfactor），**非** `k*stride(0)`
  （A7 修复 —— 若回归成简单 stride 会破坏 N-D 下标语义）。
- 等价 ATen `Indexer` 的逐元素遍历，故无需线性化临时缓冲。

## 6. 预取与 NT 决策（Task 8 结论）

- **PREFETCH_DIST=16**：微基准扫描 {0,16,32,64} 后定默认（16 最优）。
- **HPC_INDEX_NT=1 默认关**：NT 开关在 608 线程共享箱上的增量在噪声带内
  （§3 RESULTS.md），无可解析收益；保留为 opt-in。

## 7. 线程并行

- 所有大路径 `at::parallel_for(0, B*K, adaptive_grain(B*K))`（T0，
  `grain.h`：`max(1, min(n, 65536, n/(nt*8)))`）；`-fopenmp` 必须（否则
  `parallel_for` 串行化，历史根因见 architecture.md §7）。
- 线程数由 `torch.set_num_threads()` 控制；920F 为 608 核 / 32 NUMA，单节点
  38 核。共享箱上 608 线程数字噪声大，**以定点核复测为准**（performance.md §8）。

## 8. 修改内核时的门槛清单（防回归）

改 `gather_index` 分支条件时，逐一核对：
1. 新分支条件是否覆盖其内核的全部不变量（idx 位宽、dtype、A、连续度、VL）。
2. 不满足的输入是否落到正确的 `dispatch_scalar`。
3. int32 下标与 f32 numel>=2^31 的交互是否仍然安全（use_i32 决定 idx dtype）。
4. N-D 下标在 `gather_direct` 的 bfactor 解码不可丢。
5. 跑全量测试 + 两进程 true-aten 对拍（semantics.md §6）。

## 6. T8 mask 单遍压实 `gather_mask`（gather.cpp）

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

## 7. T9 小 K 单整数下标精简路径（op.cpp）

单整数下标、末维 `K<=4096`、self/idx 连续、CPU 时，`small_single_index` 直接在
`index_impl` 入口按 idx 平坦顺序取行：`out[p] = self[idx[p]]`，A>=4 用 `memcpy`
整行拷贝，n<16384 串行（并行任务开销大于收益）。绕过 `normalize_index` 的
vector 分配/transpose 分析与 `gather_direct` 的内层 `at::Tensor` 访问。
`K>4096` 仍走 SVE/桶化主路径。
