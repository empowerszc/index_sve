# 架构

`index_sve` 实现 `aten::index.Tensor` 语义的高性能 CPU 算子，三段流水线：
**归一化 → 线性化 → gather**。本文档走读控制流与数据布局。

> **一句话**：先把用户下标“翻译”成统一描述（normalize），再决定能不能省掉
> 线性化，最后按 A/K 选 RowCopy/PureGather/标量把数据搬出去。
>
> **名词**：**B** = 索引前面还有多少行（广播出的行数）；**K** = 索引张量的
> 元素个数；**A** = 每次索引对应 self 上连续元素的个数（`after` 维的积）；
> **row_off/col_off** = 行偏移与行内列偏移；**after_contiguous** = 被索引维
> 之后的所有维是否连续；**mask** = bool 索引；**linearize** = 多维下标 →
> 一维元素偏移。
>
> **为什么有两遍**：linearize 要先把所有偏移算成数组，gather 才能并行地按
> 偏移取数；小 K 时这两遍的固定成本大于收益，所以有 `gather_direct`（K<1024）
> 和 `small_single_index`（单整数下标）直接跳过。T5/T11 的 1 线程优化本质是
> 把“两遍”合成“一遍”。

## 1. 入口 `index_impl`（op.cpp）

```
index_impl(self, indices)
  1. TORCH_CHECK_INDEX(!vec.empty(), "at least one index must be provided")
  2. TORCH_CHECK_INDEX(vec.size() <= self.dim(),
       "too many indices for tensor of dimension {d} (got {n})")
  3. sync_omp_threads()                     # libomp team = at::get_num_threads()（T11）
  4. mask_fast_path(self, vec, out)         # 单 mask 连续：gather_mask 单遍压实（T8）
  5. small_single_index(self, vec, out)     # 单整数下标 && K<=4096：绕过 normalize（T9）
  5b. small_multi_index(self, vec, out)     # 多索引同 shape/连续 && n<=65536：绕过 normalize（T13）
  6. spec = normalize_index(self, vec)      # → IndexSpec
  7. out = at::empty(spec.out_shape, self.options())
  8. index_into(self, spec, out)：
       a. K < 1024 → gather_direct
            └ A==1 单索引全长连续 → 预校验紧凑循环（T11）
       b. M1: 单索引 && 连续 idx && after_contiguous && A>1
              && (A>=4096 || A<16 || (A<=64 && B*K<=65536)) && f32/f64
            → gather_rowcopy_raw（跳过 linearize，T2/T12；T15 放行小 K 中等行宽）
       b2. 单索引 && !after_contiguous && A>1 && 连续 idx && f32/f64
              && col_off 存在 ≥4 连续 run
            → gather_strided_rows_raw（run 块拷贝，T12）
       b3. 多索引 && after_contiguous && A>=4 && f32/f64 && 线程数≥16
            → gather_rowcopy_fused（跳过 linearize，T12；n>65536 或广播/非连续）
       c. T4 raw: 单索引 && B==1 && A==1 && 连续 idx
            → gather_bucket_pure_raw（小值域查表）
            └ f32/i32+i64 下标 && 源>512KB && >1 线程 → gather_single_1d_sve
              （T15-C 单遍 SVE：uzp1 收窄后 u32index gather）
            └ 其余无 SVE 内核 dtype 或 1 线程 → gather_single_1d（单遍标量，T5/T11）
       d. 1 线程 && 单索引 && A>1
            → A>=4 连续 ? gather_rowcopy_raw : gather_fused（T5/T11）
       e. A==1 || (after_contiguous && A>=4 && f32/f64)
            → linearize + gather_index（SVE/桶化/scalar）
       f. 其余（A<4 或非连续小块，含 run=1 纯转置）→ gather_fused（T3）
```

`index_out_impl` 不经过 mask/small 快路径（需支持任意 `out` 布局），校验后同样
调用 `index_into`；非连续 `out` 先算到连续临时张量再 `copy_`。

静态开关 `g_nt`：`HPC_INDEX_NT` 环境变量（非 "0" 即开），default off，进
`gather_index`/`gather_rowcopy_raw` 的 `non_temporal` 参数。

优化任务（T0-T11）引入的优先级：**先试不物化 idx 的短路路径**（gather_direct /
M1 raw rowcopy / T4 raw 桶化），需要 idx 时再 linearize；T3 fused 只为 A<4 或
非连续小块服务；**T5 让 1 线程走单遍**（单索引 A==1 大值域
`gather_single_1d`、A<4 短行 `gather_fused`），避开 linearize 两遍模型；T11
又把 1 线程 A>=4 连续块改回 raw rowcopy（SVE 整块拷贝）。T12 补齐非连续尾维
run 块拷贝、A<=64 且 B*K<=65536 的多线程 raw（T15，原仅 A<16）、多索引 A>=4（仅 ≥16t）raw；run=1 纯转置与多索引
A==1 **保留旧路径**（无连续 run / fused 解码更差）；单 mask 在入口走 `mask_fast_path` →
`gather_mask` 单遍压实（T8）。T13 把“多索引同 shape/连续、n≤65536”的短调用
提前到 normalize 之前，直接预校验偏移并逐行 memcpy。

## 2. `IndexSpec` 数据布局（normalize.h）

归一化输出的结构体，所有 gather 内核共享：

```cpp
struct IndexSpec {
  at::Tensor transposed_self;    // 已 transpose_to_front 的 self（视图，非拷贝）
  std::vector<int64_t> indexed_dims;      // 被索引维（原始维度号）
  std::vector<int64_t> indexed_sizes;     // 各被索引维大小
  std::vector<int64_t> indexed_strides;   // transposed_self 上各维 stride
  int64_t B;                    // dims_before 元素数（未索引维前缀乘积）
  int64_t K;                    // 下标张量 broadcast 形状的元素数（replacement_shape 积）
  int64_t A;                    // dims_after 元素数（后缀乘积）
  std::vector<int64_t> out_shape;
  std::vector<int64_t> row_off;  // 行首偏移（元素单位，transposed 坐标）
  std::vector<int64_t> col_off;  // A 维列偏移（元素单位）
  bool after_contiguous;         // dims_after 全连续（行拷贝前提）
  std::vector<at::Tensor> indices;  // 归一化后的下标张量（int32 保持）
  std::vector<int64_t> broadcast_shape; // N-D 下标 broadcast 形状（A7）
};
```

输出张量布局：`out[r*K + k, c]` 对应元素 `src[row_off[r] + idx[k] + col_off[c]]`，
其中 `r ∈ [0,B)`，`k ∈ [0,K)`，`c ∈ [0,A)`。行首 `row_off[r]` 按 transposed 坐标
逐维累计 `indexed_strides[j]`，由 K==0 的极端情况单独置 0。

## 3. 归一化 `normalize_index`（normalize.cpp）

镜像 ATen `TensorAdvancedIndexingUtils`：

1. `check_index_types` — 下标 dtype 白名单 {int64, int32, uint8, bool}（
   ATen `checkIndexTensorTypes(allow_int=true)`）。越界 dtype（u32/f16/f8 等）报
   "tensors used as indices must be long, int, byte or bool tensors"。
2. **mask 处理** — 可融合的单 mask 已在 `op.cpp` 入口短路（T8，见 §1）；其余
   bool/byte 下标在此先 `nonzero()` 再 `select(1, j)` 成 1 列，同时
   维护 `srcIdx`（mask 展开后结果维度）。mask 的维度不对报 "The shape of the
   mask {..} at index {j} does not match the shape of the indexed tensor {..} at
   index {0}"。byte mask 保留 ATen 的 deprecation 警告。
3. **transpose_to_front** — 有连续子空间时把被索引维搬到前面（
   `has_contiguous_subspace`），保证 `indexed_dims` 前面的 dims_before 维与后面的
   dims_after 维各自连续。若无法转置（含 None 穿插打散连续子空间），`dims_before
   == 0`，整体按 K×A 二维展开。
4. **dims_before/dims_after 计算 + size-0 检查** — 镜像 `AdvancedIndex` 构造：
   dims_before 是被索引维之前全部维的乘积，dims_after 是其后全部维的乘积。
   size-0 的被索引维报 "index is out of bounds for dimension with size 0"。
5. **`spec.indices` 保存归一化后的下标**；`broadcast_shape` 记录 N-D 下标
   broadcast 后的逐维大小（gather_direct 用，A7）。

### 有意差异（normalize，见 semantics.md）

- ATen 在 `make_info` 把 int32 下标升成 int64；本算子**保持 int32**，线性化与
  4B SVE 分支直接消费（等值，省一次转换）。这是设计决策，不是语义偏差。

## 4. 线性化 `linearize`（linearize.cpp）

并行（`at::parallel_for`）对 `K` 个输出位置各算一次目标元素偏移：

- 逐下标张量 `j` 展开：元素坐标 → 线性偏移 `+ idx_j[k'] * indexed_strides[j]`。
- **负下标归一**：`v < 0 → v += size[j]`。
- **越界检查**：`v >= size[j] || v < -size[j]` 时抛
  `"index {v} is out of bounds for dimension {j} with size {s}"`（`j` 为下标序号，
  与 ATen `Indexer` 的报错一致）。
- N-D 下标按 `broadcast_shape` 混合进制（bfactor）解码成逐维坐标（A7），非简单
  `k*stride(0)`。
- int32 保持：`use_i32` 时**直接写 int32**（不再经 K×8B int64 临时缓冲，T4
  后的改进），8B 直接并行填充 int64。

## 5. Gather 派发（gather.cpp `gather_index`）

按优先级（`idx32` = idx dtype 是否 int32；`pure` = A==1；行 = A 维元素数）：

| 分支 | 条件 | 内核 |
|---|---|---|
| 小值域桶化 | `pure`（先于 SVE 检查，值域 ≤2048 才启用） | `bucket_gather_pure<T,IdxT>` |
| 8B SVE | `pure && !idx32 && sve_512_ok() && dtype∈{f64,i64}` | `pure_gather_sve_flat<T,int64>` |
| 4B SVE | `pure && idx32 && sve_512_ok() && dtype∈{f32,i32}` | `pure_gather_sve_flat<T,int32>` |
| RowCopy | `A>1 && after_contiguous && sve_512_ok() && A>=4 && dtype∈{f32,f64}` | `rowcopy_sve<T,IdxT>` |
| Scalar | 其余全部 | `dispatch_scalar` |

**注意：`bucket_gather_pure` 在 SVE 分支之前**，因为小值域（值域 ≤2048）时
L1/L2 常驻查找表的标量随机读避免了 64B 行放大，远优于大值域随机 gather
（kernels.md §2；SVE 4B 与标量在大值域持平）。
值域 >2048 时返回 false，继续走 SVE/scalar。

`gather_index` 之外还有两个**入口级短路**（在 op.cpp，不经过 linearize）：

| 短路 | 条件 | 内核 |
|---|---|---|
| M1 raw rowcopy | 单索引 && after_contiguous && A>1 && (A>=4096 \|\| A<16 \|\| (A<=64 && B*K<=65536)) && f32/f64 | `rowcopy_raw_single<T,IdxT,NT>` |
| strided rows | 单索引 && !after_contiguous && col_off 有 ≥4 run | `strided_rows_raw<T,IdxT,NT>` |
| 多索引 raw | 多索引 && after_contiguous && A>=4 && ≥16t | `rowcopy_fused_sve<T,NT>` |
| T4 raw 桶化 | 单索引 && B==1 && A==1（1D 纯 gather） | `bucket_gather_pure_raw<T,IdxT>` |

这两个路径直接消费 `spec.indices[0]` 原始下标，内联边界检查 + 负索引规整，
跳过 linearize 的 broadcast 解码。T4 raw 桶化是 M3 keys[order]/M4 等 1D
小值域形态的专用加速（值域 ≤2048）。

`sve_512_ok()`：运行时 `svcntb()==64` 才启用 SVE（编译按
`-msve-vector-bits=512`，运行时 VL 不符会 UB，必须门控）。

**为什么 f32 且 numel>=2^31 会落标量**：`use_i32=false` → idx 为 int64 →
4B SVE 分支要求 `idx32` 不满足 → 落入 scalar。这是有意的防 int32 越界回退。

每个 SVE 分支的条件都满足其内核的全部不变量（idx 位宽、dtype、连续度），
**不存在"被门控掉但给错误值"的输入类** —— 要么走对内核，要么安全落标量。

## 6. 快速路径 `gather_direct`（K < 1024）

跳过线性化分配，直接读原始下标 + 内联逐维检查（等价 ATen `Indexer` 遍历）。
用 `row_off`/`col_off` 相同布局写 `dst[p*A + c] = src[off + col_off[c]]`。
N-D 整数下标用 bfactor 按 `broadcast_shape` 解码元素坐标（A7 修复，勿回归成
`k*stride(0)`）。

## 7. 并行模型

- 除 `gather_direct` 外的所有内核走 `at::parallel_for(0, B*K, adaptive_grain(B*K))`，
  grain 由 `grain.h` 的自适应公式给出：`max(1, min(n, 65536, n/(nt*8)))`（T0）。
  小 K 也能占满所有线程，大 K 不因 grain 过大退化。
- **`-fopenmp` 是硬依赖**：`at::parallel_for` 在扩展里是头文件内联模板，无
  `_OPENMP` 时循环实际串行（历史根因：曾导致 B==1 大 K 从预期加速变成
  0.02-0.3x）。setup.py 已加 `-fopenmp`。
- 线程数由调用方控制：`torch.set_num_threads()`。但扩展的 `parallel_for` 由
  BiSheng **libomp** 执行，ATen 用 **libgomp**；T11 起 `index_impl` /
  `index_out_impl` 入口调用 `sync_omp_threads()` 把 libomp team 同步为
  `at::get_num_threads()`，否则用户设的线程数对扩展无效（旧线程数据失真的
  根因，见 `degradation-analysis.md` §3.1）。

## 8. 关键调用图

```
op.cpp:index_impl
├── normalize.cpp:normalize_index
│   ├── IndexingUtils 语义：mask.nonzero() / select / transpose_to_front
│   └── AdvancedIndex 语义：dims_before/dims_after / size-0 检查
├── gather.cpp:gather_direct               # K<1024 快速路径
├── gather.cpp:gather_rowcopy_raw          # M1 单索引大块 skip-linearize（T2）
│   └── rowcopy_raw_single<T,IdxT,NT> → copy_contig_sve<T,NT>
├── gather.cpp:gather_bucket_pure_raw      # T4 raw 1D 小值域查表（单索引 B==1 A==1）
│   └── bucket_gather_pure_raw<T,IdxT>
├── gather.cpp:gather_fused                # T3 融合单遍（A<4 或非连续小块，不物化 idx）
├── linearize.cpp:linearize                # parallel_for
└── gather.cpp:gather_index                # dispatch_scalar / SVE / 桶化分支
    ├── bucket_gather_pure<T,IdxT>         # T4 小值域（A==1，先于 SVE）
    ├── pure_gather_sve_flat<T,IdxT,NT>
    │   └── pure_gather_sve_{8b_f64|8b_s64|4b_f32|4b_s32}
    ├── rowcopy_sve<T,IdxT,NT> → copy_contig_sve<T,NT>
    └── dispatch_scalar → scalar_gather
```
