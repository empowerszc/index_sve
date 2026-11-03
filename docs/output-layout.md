# 输出内存布局：与 aten 逐维一致（2026-10-29 E2E 正确性修复）

> **读者**：持有 index_sve 早期版本、需要排查/修复同一 bug 的 agent 或人。
> **一句话**：非连续/置换布局（channels_last 等）self 上的 `x[idx]`，值全对，
> 但输出被恒分配为 NCHW 连续；aten 的输出由 TensorIterator **置换恢复**分配，
> 会保持 self 的布局。E2E channels_last 模型（CNN/AMP）每次 `x[idx]` 都会
> 静默打断格式传播链。本修复在 `index_impl` 主路径复刻 aten 的分配算法
> （§5），并同日落地 R10 直写消除 tmp+copy_ 的带宽代价（§8）。

## 1. 现象与实测对照

torch 2.10.0+cpu 无覆盖进程（true aten）实测，`x[idx]` 输出 stride：

| self 布局 | 索引形态 | aten 输出 stride | 修复前（本算子） |
|---|---|---|---|
| cl4d (4,32,14,14) strides (6272,1,448,32) | 行索引 `x[rows]` | **(6272,1,448,32)** = channels_last | (6272,196,14,1) 连续 ✗ |
| 同上 | `x[mask]`（dim0 掩码） | **(6272,1,448,32)** | 连续 ✗ |
| 同上 | 2D 索引 `x[t2d]` | **(18816,6272,1,448,32)** | 连续 ✗ |
| 同上 | `x[t, 1:3]`（切片先行） | **(392,1,28,2)** | 连续 ✗ |
| 同上 | `x[t, None]` | **(6272,32,1,448,32)** | 连续 ✗ |
| 同上 | `x[..., t]` | **(1344,3,96,1)** | 连续 ✗ |
| cl5d (2,3,4,5,6) cl3d | 行索引 | **(360,1,90,18,3)** | 连续 ✗ |
| 3D t(1,2) (4,6,5) strides (30,1,6) | 行索引 | **(30,1,6)** | 连续 ✗ |
| 同上 | 2D 索引 | **(60,30,1,6)** | 连续 ✗ |
| **连续** 2D (5,6) + 转置 2D 索引 (3,2)/(1,3) | `x[t.t()]` | **(6,18,1)** 非连续！ | 连续 ✗ |
| **连续** 4D (4,32,14,14) + 转置 2D 索引 | `x[t.t()]` | **(6272,18816,196,14,1)** | 连续 ✗ |
| cl4d | 列索引 `x[:, t]` | (980,196,14,1) 连续 | 连续 ✓ |
| cl4d | multi2 `x[a,b]` | (588,196,14,1) 连续 | 连续 ✓ |
| cl4d | 广播 multi `x[(3,1),(1,2)]` | (392,196,14,1) 连续 | 连续 ✓ |
| 任意连续 self | 任意 1D/常规形态 | 连续 | 连续 ✓ |
| strided `[::2]` / expand / 2D 转置 self | 行索引 | 连续 | 连续 ✓ |

要点：
- **值全部正确**，错的只是 stride/内存格式元数据（`is_contiguous()`、
  `is_contiguous(memory_format=channels_last)`、`suggest_memory_format`）。
- 分歧只出现在 **normalize 主路径**（非连续 self、None 混合、大 K、mask
  展开后的 nonzero 列）。`small_single`/`small_multi`/`mask_fast_path` 有
  `self.is_contiguous()` 守卫，天然免疫。
- 连续 self 上也可能非连续（转置 2D 索引行）——**不能**用
  "`suggest_memory_format` + `memory_format=`" 捷径：cl4d 列索引时 aten 返回
  连续而非 cl，捷径会引入新偏差；3D 置换 (30,1,6) 也不可经
  `MemoryFormat::ChannelsLast` 分配（`c10/core/TensorImpl.cpp:943` 硬校验
  rank==4）。

## 2. 根因

- 本算子 `index_impl`（op.cpp）：`out = at::empty(spec.out_shape, self.options())`
  恒 NCHW 连续；且所有内核按行主序线性写 out。
- aten 的输出**不是**在 `TensorAdvancedIndexing.cpp` 里分配的：`index.Tensor`
  是结构化 op，输出由 `build_index_op` 构造的 **TensorIterator** 分配
  （`TensorAdvancedIndexing.cpp:468` `build_index_op` → `TensorIterator.cpp:574`
  `allocate_or_resize_outputs`）。TensorIterator 会按输入的 stride 做**置换恢复**：
  输出与"主导输入"（restrided self）使用同一维序的 dense 布局。

## 3. aten 机制（复刻依据，均已在源码与实测双确认）

链路：`index.Tensor` → `make_info`（广播索引、`restride_src` 置零被索引维、
`reshape_indexer` 前后补 1 维）→ `build_index_op`（operands = [undefined 输出,
restrided self, 各 index]）→ `TensorIterator::build`：

1. **compute_strides**（TensorIterator.cpp:1277）：每个 operand 逐维取
   `stride × element_size`（字节）；operand 该维尺寸 1 而 out 该维 >1 视作
   广播 → 0；维数不足时前补 0。
2. **reorder_dimensions**（TensorIterator.cpp:232，NOTE [Computing output strides]）：对维做
   `should_swap` 插入排序（perm 内维在前，初始为逆序恒等）：
   - 某一 operand 上两维 stride 任一为 0 → 跳到下一 operand 再比；
   - 严格小于/大于 → 定序；相等 → 平局按 out 该维 shape，大者靠外，仍平则
     跳下一 operand；全部无定论 → 保持。
3. **allocate_or_resize_outputs**（TensorIterator.cpp:574）：perm 恰为逆序恒等 → 连续快路径；
   否则 `compatible_stride` 按 perm 序做 dense 分配，`invert_perm` 映回原维
   序，`set_output_raw_strided` → `empty_strided`。

由此可推出（并与上表全部吻合）：**连续 self + 连续常规索引 → 恒连续**；
**置换 dense self（cl4d/cl3d/3D 转置）+ 行索引族（含 mask、2D 索引、前置
切片）→ 保持原置换**；**列索引/multi（被替换维 stride=0 参与排序）→ 通常连续**；
**非 dense self（strided/expand/overlap）→ 连续**。

## 4. 一分钟自检（其他 agent 检测自己持有的版本）

```bash
python -c "
import torch, index_sve   # 你的版本；覆盖激活
x = torch.randn(4, 32, 14, 14).to(memory_format=torch.channels_last)
y = x[torch.arange(4)]
assert y.is_contiguous(memory_format=torch.channels_last), f'BUG: {y.stride()}'
print('output-layout OK')
"
```

更完整：拷本仓库 `tests/test_output_layout.py`（25 例钉 stride 真值，仅依赖
torch + 包 import）与 `tests/verify_e2e.py`（两进程差分，值+shape+stride 全
比对）到你的版本运行。

## 5. 修复移植指南

唯一改动点：`index_impl` 的 out 分配（op.cpp）。三步：

1. 移植 `aten_out_strides(const IndexSpec&)`（见 op.cpp，~90 行）：按 §3 构建
   operands 的 stride_bytes（restrided self：before 维保 stride、replacement 维
   置 0、after 维保 stride；各 index 经 `reshape` 前后补 1 维后同
   compute_strides 规则），`should_swap` 插入排序，逆序恒等走连续，否则按
   perm 序 dense 分配。注意 `IndexSpec` 已含全部所需信息
   （`transposed_self`/`indexed_dims`/`indices`（已广播）/`broadcast_shape`/
   `out_shape`）。
2. `out = at::empty_strided(spec.out_shape, aten_out_strides(spec), self.options())`。
3. 写入：`out.is_contiguous()` → 原内核直写（行为与性能零变化）；否则先算入
   连续 tmp 再 `out.copy_(tmp)`（与 `index_out_impl` 的非连续 out 防线同模式；
   内核保持行主序写不变）。这是**可移植的最小正确修复**；本仓库同日已进一步
   实现直写优化（§8），移植时按需取舍。

守卫自查：若你的版本 `small_single_index`/`small_multi_index`/
`mask_fast_path` 没有 `self.is_contiguous()`（或等价）守卫，它们也接非连续
self，需要同样处理；本仓库三者均有守卫（op.cpp:200 / op.cpp:230 / op.cpp:348），且连续
self 时 aten 恒连续输出，无需改动。

`index.Tensor_out`（`out=` 变体）不需要改：aten 写调用方提供的 out（不重
分配），本算子的 resize+非连续 tmp+copy_ 防线已覆盖任意布局 out。

## 6. 验证（修复后实测，2026-10-29）

- `tests/test_output_layout.py`：**25/25**（修复前 12/25——13 项失败即本 bug 面；
  25 个期望 stride 值先在无覆盖进程对 true aten 逐一核实后钉入）。
- `tests/verify_e2e.py --mode both --threads`：**124/124** 两进程差分全绿
  （修复轮 120/120；R10 轮追加 4 例 g5big 大 K 直写用例。修复前 104/120；
  值+shape+dtype+stride 全比对），线程一致性矩阵 1/4/38t **46 例 ALL OK**。
  g5 布局×形态专项 30 例覆盖 §1 全部行，g5big 4 例穿过 §8 直写路径。
- 15 套件 **174/174**（新增 test_output_layout 后原 143/143 无回归；R10 轮
  test_rowcopy_routes 8→14 后全绿）。
- `verify_against_aten.py --shape-set all`：**44/44** 无回归。

## 7. 性能影响（布局修复轮实测，2026-10-29 上午）

同轮交替实测（8t，该轮机器噪声 ±30%+——未改动的 contig4d（走
small_single 快路径）rep 间亦波动 3 倍，可作噪声标定）：

| 形态 | aten | 本算子（tmp+copy_） | 比值 |
|---|---|---|---|
| col-index (1000,32) t128 | 95-118µs | 85-90µs | 1.1-1.4x（领先不变） |
| M1 (27,64,64) K20k | 12.8-16.0ms | 2.0-2.8ms | 4.6-6.3x |
| cl4d 行索引 K2048 | 3.0-3.4ms | 8.1-10.0ms | **0.30-0.42x** |

- 连续 self 热路径：分配仍为连续、原内核直写，仅新增 `aten_out_strides`
  簿记（1-2 个 reshape 视图 + O(ndim²) 排序，µs 级），在本轮噪声带内
  不可分辨；small/mask 快路径（K<=4096 单索引等）完全不经主路径，零影响。
  簿记成分微基准（R10 轮复测）：`empty_strided` 比 `empty` **快 9-10µs**
  （后者多走 memory-format 建议路径），reshape 视图 ~7.4µs/索引——净中性
  偏正。
- 非连续输出（cl4d/cl3d/3D 置换行索引族）：tmp+copy_ 多一趟输出带宽
  （上表 cl4d 行）。**同日已由 R10 直写消除（§8）**。
- 对 E2E 的净效应：channels_last 模型的格式传播链恢复（下游 conv/AMP 不再
  因 `x[idx]` 静默转 NCHW），通常远大于本算子内一趟 copy_ 的代价。

## 8. R10 直写优化（同日落地，消除 tmp+copy_）

**原理**：cl 尾块 (C,H,W)/(1,WC,C) 是 **dense 置换**——源行（self 的一个
n 块）与 cl 输出行（out 的一个 k 块）都是整块 dense 内存，且逻辑→内存的
置换相同；现有 rowcopy 内核是**纯块拷贝**（`copy_contig_sve(src+off,
dst+p*A, A)`，不经 col_off），对"源/出同置换"天然正确。直写即消除 tmp+
copy_ 的一趟输出带宽（aten 走 TensorIterator 通用散写，无快路径）。

**资格判定** `rowcopy_direct_ok`（op.cpp，O(ndim)，任一不满足即回退
tmp+copy_，外部行为不变）：单索引 + 索引连续 + A>=4 + K>=1024 + f32/f64 +
SVE512 + self 尾块 strides 全正且构成 dense 置换（sorted==前缀积链，自动
排除切片/expand/flip）+ out 尾块与 self 尾块逐维相等 + out 前导块
（before+replacement 维）行主序（保证内核 `dst[p*A]` 恰为正确偏移）。
置位后 `index_into` 顶部自包含守卫只允许块拷贝内核（raw 形态走
`gather_rowcopy_raw`，否则 linearize+`gather_index` 的 `rowcopy_sve`，
后者守卫同样承认 `rowcopy_direct`）；**绝不能落入 col_off 类路径**（对
置换 out 是块内错位写入）。多索引/`out=` 变体/非 f32-f64/A<4/K<1024
均不参与（v1 范围）。

**实测（2026-10-29 R10 轮，8t，两进程背靠背 3 rep × 3 轮交替）**：

| 形态 | aten | 本算子（R10 直写） | 比值 |
|---|---|---|---|
| cl4d 行索引 K2048 | 3.05-3.30ms | 1.30-2.11ms | **1.45-2.54x** |
| cl4d mask K2048 | 3.21-3.30ms | 1.54-2.33ms | **1.40-2.08x** |
| cl3d 置换行索引 K2048 | 1.76-2.33ms | 0.85-1.41ms | **1.35-2.63x** |
| contig4d 对照（未经 R10 改动） | 3.06-3.46ms | 1.21-1.95ms | 1.57-2.73x |

- 对比 §7：cl4d 行索引本算子时间 8.1-10.0ms → **1.3-2.1ms**（降 4-6 倍），
  比值 0.30-0.42x → **1.45-2.54x**（家族含 cl3d 置换 1.35-2.63x）。
- cl 行索引与连续行索引基本拉平（1.30 vs 1.21ms、1.98 vs 1.84ms，
  差 ~5-8%）——"dense 置换行块 memcpy 与连续行等价"得到实证。
- 第 2 轮交替曾见 hpc 侧整体劣化 3 倍：**未经 R10 改动的 contig4d 对照组
  同倍劣化**（1.21→4.45ms），判共享箱噪声，非代码问题；跨轮只比形态。

**验证**：6 个新路由用例（tests/test_rowcopy_routes.py：直写 2 例——cl4d 行索引
含 1t/8t 双档、cl4d mask；排除 3 例计数器断言；cl3d linearize 臂值/stride 1 例）
+ 既有 25 布局例（小 K，覆盖回退路径）+ E2E 124 例差分（新增 g5big 大 K 4 例：
行索引 raw/linearize 两臂、mask、cl5d 穿过直写路径，值/shape/dtype/stride 对
true aten）+ 44 对拍 + 线程矩阵 46 例全绿（R10 轮 174/174）。
