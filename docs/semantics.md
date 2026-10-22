# 语义与正确性

本算子承诺**语义等价于 torch 2.10.0 的 `aten::index.Tensor`（CPU 参考实现）**，
逐字节对齐。本文档列出对齐点与**有意的差异**，以及正确性验证方法。

## 1. 入口行为

| 输入条件 | 行为 | ATen 参考 |
|---|---|---|
| 空下标列表 | 抛 `at least one index must be provided` | 相同 |
| 下标数 > self.dim() | 抛 `too many indices for tensor of dimension {d} (got {n})` | 相同（`d` 为维度数，`n` 为下标数） |
| 下标 dtype 白名单 | {int64, int32, uint8, bool} | ATen `checkIndexTensorTypes(allow_int=true)` |
| 白名单外 dtype | 抛 `tensors used as indices must be long, int, byte or bool tensors` | 相同 |
| byte mask | 同 bool 处理 + deprecation 警告 | 相同 |
| mask 形状不匹配 | 抛 `The shape of the mask {..} at index {j} does not match ...` | 相同 |
| size-0 被索引维 | 抛 `index is out of bounds for dimension with size 0` | 相同 |
| 索引不在 CPU（meta/CUDA） | 抛 `indices should be either on cpu or on the same device as the indexed tensor (cpu)` | 相同（T6 前会 SIGSEGV） |
| `out=` 变体（`index.Tensor_out`） | 写入调用方 out；形状不符 resize；非连续 out 计算到临时张量再 copy | 相同 |

## 2. 归一化等价性

`normalize_index` 镜像 ATen `TensorAdvancedIndexingUtils` 的：
- `make_info` 语义（dims_before/dims_after、连续子空间转置、None 展开）；
- `has_contiguous_subspace` / `transpose_to_front`（行列各自连续）；
- mask → `nonzero()` + `select(1, j)` 并维护 `srcIdx`；
- `AdvancedIndex` 构造的 size-0 检查与维度展开顺序。

支持的全部输入类：
- 1-D / 2-D / 3-D 整数下标张量（含 **N-D 下标**，A7）；
- 负数下标（`v<0 → v+=size`）；
- bool / byte mask（含 2-D mask）；
- `None` 条目（unsqueeze，不参与索引）；
- dims_before / dims_after 非平凡（`x[None, None, i]`、`x[:, i]` 等）；
- 转置到前的非连续下标布局（`has_contiguous_subspace`）；
- 0 维标量下标（`x[tensor(0-dim)]`）；
- broadcast 多下标（`x[i, j]` 形状广播）；
- 空 / 过多下标列表；
- size-0 维度。

## 3. 越界错误逐字对齐

```
index {v} is out of bounds for dimension {j} with size {s}
```

其中 `j` 是**下标序号**（归一化/转置后的 defined-index 顺序，与 ATen
`Indexer` 相同），`s` 为该维大小。负数越界同样抛此信息（`v < -size`）。测试
`tests/test_errors.py` 断言逐字格式。

## 4. dtype 覆盖

与 ATen CPU index 内核的 dtype 集合**完全一致**：
`AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool)`。
即：f32/f64/i32/i64/i16/i8/u8/bool/half/bf16/complex64/complex128。
u32/u64/f8 等在任何实现下都报错（白名单外）。**无 dtype 回归**。

## 5. 有意差异（设计决策，非 bug）

| # | 差异 | 影响 | 理由 |
|---|---|---|---|
| D1 | int32 下标**不升 int64**，保持 int32 贯穿 | 值与 aten 完全一致；性能上允许 4B SVE 分支直接消费 | ATen 在 `make_info` 升成 int64（值不变）；本算子选择保持以省转换 |
| D2 | 越界错误消息不追加非广播形状列表（`normalize.cpp` 一处） | 仅报文本少一段形状说明，**首句**逐字相同 | minor；已在测试中标注 |
| D3 | 覆盖的是 CPU dispatch key（默认开启） | 全局 `x[...]` 行为被改 | 设计目标（A6）——零改动接入 |

## 6. 正确性验证方法

1. **测试套件**（102/102 通过，覆盖激活态；T6 后 test_index_out 7、T8 后 test_mask_fused 9、T9 后 test_smallk 7）：
   - `test_smoke.py` — 基本注册/调用；
   - `test_normalize.py` — spec 字段结构断言（dims_before/dims_after、transpose、
     mask 展开、after_contiguous）；
   - `test_linearize_values` — 手算元素偏移对拍；
   - `test_correctness.py` — 各类下标 × dtype 矩阵 vs `x[...]`（含 K<1024 与
     K>=1024 两路径、N-D、0 维、broadcast、int32/64 下标、f32 溢出回退、
     complex/2B/1B 在 K>=1024 的 `test_large_k_scalar_dtypes` 循环）；
   - `test_errors.py` — 逐字错误格式；
   - `test_sve.py` — SVE 分支门控、值正确性、NT 路径；
   - `test_model.py` — 9 个模型场景正确性（`x[:, perm]` 注意是 `:` 而非 `None`，
     torch 2.10 下 `x[None, perm]` 会 OOB）。
2. **覆盖下的对拍陷阱**：覆盖激活后 `x[...]` 也走本算子，测试变自洽。因此：
   - 错误/形状/崩溃类仍有效；
   - 值级真对拍必须用**两进程法**：无覆盖进程（`HPC_INDEX_NO_OVERRIDE=1`）生成
     true-aten 参考 `.pt`，覆盖进程比对。现成工具：`tests/verify_against_aten.py`
     （`--mode both` 自动跑两进程，small/large/all 三套 shape 矩阵，含 K>=1024、
     桶化、M1 raw、T5 单遍、多索引、mask、广播、非连续 stride 等 44 用例，参考写到 git-ignored
     `agent_space/`）。
   - **`None` 占位语义**：传入 op 的索引列表里 `None` 是"跳过维"占位
     （`aten::index(x,[None,i])` 语义 == `x[:, i]`，shape (8,3)），**不是** Python
     的 newaxis。用户级 `x[None, i]`（新轴，shape (1,3,5)）由 TensorIndexing 在
     到达 op 前处理，从不进本算子。对拍脚本据此用 `slice(None)` 构造含 None 用例
     的参考。
3. **递归安全**：`index_impl` 只调用 `at::expand_outplace`、`index.nonzero()`、
   `permute`、`at::empty`、linearize、gather，**从不调用** `at::index` /
   `self.index` / `aten::index`，无覆盖自递归。
4. **autograd**：backward 由 `IndexBackward` 公式提供（独立于 CPU kernel 注册），
   梯度值/形状/dtype 已与 true-aten 对拍验证。

## 7. 语义红线（回归测试不可破坏）

- 错误消息逐字等于 ATen（`index {v} is out of bounds for dimension {j} with size {s}`）。
- int32 下标保留不转 int64。
- f32 且 `self.numel() >= 2^31` 必须走标量（防 int32 越界）。
- `x[None]` 不进本算子（`TensorIndexing.h` 走 unsqueeze）；`x[tensor(0-dim int)]`
  走 `applySelect`（不进 op）——两者行为与覆盖前一致，测试套件已覆盖。
