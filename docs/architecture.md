# 架构

`index_sve` 实现 `aten::index.Tensor` 语义的高性能 CPU 算子，三段流水线：
**归一化 → 线性化 → gather**。本文档走读控制流与数据布局。

## 1. 入口 `index_impl`（op.cpp）

```
index_impl(self, indices)
  1. TORCH_CHECK_INDEX(!vec.empty(), "at least one index must be provided")
  2. TORCH_CHECK_INDEX(vec.size() <= self.dim(),
       "too many indices for tensor of dimension {d} (got {n})")
  3. spec = normalize_index(self, vec)        # → IndexSpec
  4. out = at::empty(spec.out_shape, self.options())
  5. if spec.K < 1024: gather_direct(spec, out); return
  6. use_i32 = self.numel() < 2^31 && dtype ∈ {f32, i32}
  7. idx = at::empty({K}, dtype = use_i32 ? int32 : int64)
  8. linearize(spec, idx)                     # 并行，一次遍历
  9. gather_index(spec, idx, out, g_nt)       # 派发 SVE/scalar
```

静态开关 `g_nt`：`HPC_INDEX_NT` 环境变量（非 "0" 即开），default off，进
`gather_index` 的 `non_temporal` 参数。

## 2. `IndexSpec` 数据布局（normalize.h）

归一化输出的结构体，所有 gather 内核共享：

```cpp
struct IndexSpec {
  at::Tensor transposed_self;    // 已 transpose_to_front 的 self（视图，非拷贝）
  std::vector<int64_t> indexed_dims;      // 被索引维（原始维度号）
  std::vector<int64_t> indexed_sizes;     // 各被索引维大小
  std::vector<int64_t> indexed_strides;   // transposed_self 上各维 stride
  int64_t B;                    // dims_before 元素数（未索引维前缀乘积）
  int64_t K;                    // 被索引维大小的乘积（indexed_sizes 积）
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
2. **mask 处理** — bool/byte 下标先 `nonzero()` 再 `select(1, j)` 成 1 列，同时
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
- int32 保持：`use_i32` 时先写 int64 临时再转换（K×8B 临时缓冲，已知可优化点，
  见 performance.md）；8B 直接并行填充 int64。

## 5. Gather 派发（gather.cpp `gather_index`）

按优先级（`idx32` = idx dtype 是否 int32；`pure` = A==1）：

| 分支 | 条件 | 内核 |
|---|---|---|
| 8B SVE | `pure && !idx32 && sve_512_ok() && dtype∈{f64,i64}` | `pure_gather_sve_flat<T,int64>` |
| 4B SVE | `pure && idx32 && sve_512_ok() && dtype∈{f32,i32}` | `pure_gather_sve_flat<T,int32>` |
| RowCopy | `A>1 && after_contiguous && sve_512_ok() && A>=16 && dtype∈{f32,f64}` | `rowcopy_sve<T,IdxT>` |
| Scalar | 其余全部 | `dispatch_scalar` |

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

- 除 `gather_direct` 外的所有内核走 `at::parallel_for(0, B*K, GRAIN=4096)`。
- **`-fopenmp` 是硬依赖**：`at::parallel_for` 底层是 `at::parallel_for_impl`，
  无 `_OPENMP` 时 `p-7 for loop in different threads` 实际是串行（历史根因：
  曾导致 B==1 大 K 从预期加速变成 0.02-0.3x）。setup.py 已加 `-fopenmp`。
- 线程数由调用方控制：`torch.set_num_threads()`。

## 8. 关键调用图

```
op.cpp:index_impl
├── normalize.cpp:normalize_index
│   ├── IndexingUtils 语义：mask.nonzero() / select / transpose_to_front
│   └── AdvancedIndex 语义：dims_before/dims_after / size-0 检查
├── linearize.cpp:linearize            # parallel_for
├── gather.cpp:gather_index            # dispatch_scalar / SVE 分支
│   ├── pure_gather_sve_flat<T,IdxT,NT>
│   │   └── pure_gather_sve_{8b_f64|8b_s64|4b_f32|4b_s32}
│   ├── rowcopy_sve<T,IdxT,NT> → copy_contig_sve<T,NT>
│   └── dispatch_scalar → scalar_gather
└── gather.cpp:gather_direct           # K<1024 快速路径
```
