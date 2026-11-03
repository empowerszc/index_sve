# `aten::index`（高级索引）在 aarch64 CPU 上的原理、实现与优化指南

> 本文基于 PyTorch 源码 `pytorch-v2.12.1`，梳理 `aten::index`（即 `tensor[indices]` 的核心算子）
> 从 Python 调用到 aarch64 CPU kernel 的完整链路，并系统讨论它的性能优化——包括
> SVE512 / SME 机器上的专属机会。
>
> **写作目标**：让不同基础的读者都能各取所需——
> 不熟悉 C++ / 微架构的读者能看懂"它是什么、为什么慢、优化在干嘛"；
> PyTorch 开发者能顺着行号读懂每一步实现；
> 性能工程师能拿到可落地的优化清单和优先级建议。
> 所有 `文件:行号` 引用均以本仓库代码为准。

---

## 目录

- [0. 阅读指南](#0-阅读指南)
- [1. 速览：一页看懂全文](#1-速览一页看懂全文)
- [2. 预备知识（按需选读）](#2-预备知识按需选读)
- [3. `aten::index` 的语义：它到底做什么](#3-atenindex-的语义它到底做什么)
- [4. PyTorch CPU 实现全链路](#4-pytorch-cpu-实现全链路)
- [5. 性能画像：时间都花在哪](#5-性能画像时间都花在哪)
- [6. 通用优化思路（不挑硬件）](#6-通用优化思路不挑硬件)
- [7. SIMD 优化：SVE256 / SVE512 / SME 机器](#7-simd-优化sve256--sve512--sme-机器)
- [8. 怎么选：场景决策表与优先级](#8-怎么选场景决策表与优先级)
- [9. 实战案例：spconv SubMConv3d 中的 index 优化](#9-实战案例spconv-submconv3d-中的-index-优化)
- [10. 实战案例：AI4S 点云模型中的 index 优化](#10-实战案例ai4s-点云模型中的-index-优化)
- [附录 A：相关文件与关键行号索引](#附录-a相关文件与关键行号索引)
- [附录 B：术语表](#附录-b术语表)

---

## 0. 阅读指南

不同基础的读者建议按下面的路径选读（全文约 40 分钟通读，选读 10~15 分钟）：

> **不熟 PyTorch 内部机制？** 建议先读《`aten::index` 通俗导读》
> （`aten_index_primer.md`）建立整体图景，再回来读本文。

| 你是谁 | 建议路径 | 能获得什么 |
|---|---|---|
| 只会用 PyTorch，不读 C++ | §1 → §2 → §3 → §5.3 → §8 | 明白 `x[idx]` 背后发生了什么、为什么慢、模型里怎么避坑 |
| 熟悉 PyTorch，想读懂源码 | §1 → §3 → §4 全部 → §5 → 附录 A | 一条调用链 + 行号索引，可对照仓库逐行精读 |
| 性能/系统工程师 | §1 → §2.3/2.4 → §5 → §6 → §7 → §8 | 优化思路全集、SVE512/SME 机会、优先级与决策表 |
| 评审优化方案 / 做技术决策 | §1 → §5 → §8 | 快速把握瓶颈本质和投入产出 |

**全文约定**：
- `文件:行号` 指本仓库源码位置，如 `IndexKernelUtils.h:52`；
- "下标（index）"指用户传入的索引 tensor；"self"指被索引的 tensor；
- 所有 stride 若无说明均为**字节**单位。

---

## 1. 速览：一页看懂全文

**一句话**：`aten::index` 就是"按清单取货"——拿着下标 tensor 这张清单，去 self tensor
这个仓库里把元素一个个取出来，按清单顺序放进新分配的结果 tensor。

```python
x = torch.randn(4, 5)
idx = torch.tensor([1, 3, 0])
y = x[idx]        # y[0]=x[1], y[1]=x[3], y[2]=x[0] —— 本质是逐元素"查表取数"（gather）
```

**CPU 上的实现本质**（去掉一切包装后，核心循环就这几行，
`IndexKernelUtils.h:52`）：

```cpp
for (每个输出元素 i) {
    offset = 0;
    for (每个被索引的维 j) {
        v = index_j[i];              // ① 读下标（8 字节 int64）
        检查 v 是否越界;              // ② 每个元素都查（运行时检查，关不掉）
        if (v < 0) v += size_j;      // ③ 负下标归一
        offset += v * stride_j;      // ④ 算出取货地址
    }
    dst[i] = src[offset];            // ⑤ 取货 + 放货
}
```

**逐行拆解**（建议细读——全文说的"慢"和"优化"，根源都在这 5 行）：

先把每个符号讲清楚（正式定义见 §4）：

- `i`：**输出元素**的序号，把结果拉平后从 0 数起。循环每转一圈，产出结果的一个元素；
- `j`：**被 tensor 下标"占据"的那个维**。`x[idx]` 只有 1 个（dim 0）；
  `x[i_idx, j_idx]` 有 2 个（dim 0 和 dim 1）。注意 `x[:, idx]` 里被索引的是 dim 1、
  dim 0 没有 tensor 下标——**没有下标的维不进这个内层循环**，由外层框架
  （TensorIterator，§4.6）负责逐行推进；
- `index_j[i]`：**第 j 个下标 tensor 的第 i 个值**。为什么是 8 字节：`make_info`
  会把所有下标统一转成 int64（§4.4 第 ⑤ 步），kernel 就按 8 字节读；
- `size_j`：self 在第 j 个被索引维上的长度（用来判越界、归一负数）；
- `stride_j`：self 在第 j 个被索引维上的**字节**步长（下标值每 +1，地址跨多少字节）；
- `offset`：一圈内层循环累加出的**字节偏移**——"从 self 开头走多少字节，
  能取到当前这个元素"。

配一个能整条跑通的数值例子：

```python
x = torch.arange(12).reshape(3, 4).float()   # 值 0..11；连续布局，字节 stride = [16, 4]
i_idx = torch.tensor([2, 0])                 # 下标1，占 dim0（int64）
j_idx = torch.tensor([3, 1])                 # 下标2，占 dim1（int64）
y = x[i_idx, j_idx]                          # 期望 y = [x[2,3], x[0,1]] = [11.0, 1.0]
```

此时 `size = [3, 4]`（两维的长度），`stride = [16, 4]`（字节）。逐元素推演：

```
输出元素 i = 0（要用 i_idx[0]=2、j_idx[0]=3 去 x 里取数）：
  j=0: v = i_idx[0] = 2 → 检查 -3 ≤ 2 < 3 通过 → offset += 2 × 16 = 32
  j=1: v = j_idx[0] = 3 → 检查 -4 ≤ 3 < 4 通过 → offset += 3 × 4  = 12
  ⑤ dst[0] = src[32 + 12 = 44 字节] = 第 11 个元素 = x[2][3] = 11.0 ✓

输出元素 i = 1（用 i_idx[1]=0、j_idx[1]=1）：
  j=0: v = 0 → offset += 0 × 16 = 0
  j=1: v = 1 → offset += 1 × 4  = 4
  ⑤ dst[1] = src[4 字节] = 第 1 个元素 = x[0][1] = 1.0 ✓
```

若下标含负值（如 `i_idx = [-1, 0]`），③ 负归一把 -1 变成 2（`-1 + size_0 = 2`，
即 dim0 的最后一行）——Python 里 `x[-1]` 取最后一行就是这条分支在起作用。

两个简化说明（真实代码在 §4.7）：本例两个维都被索引、没有剩余维，所以
`dst[i] = src[offset]` 是精确的；若还有未索引维（如 `x[:, idx]` 的 dim0），
"走到第几行"由 TensorIterator 推进基址指针负责，`offset` 只补被索引维的部分。
另外真实代码里 `i` 是"当前一维切片内"的偏移而非全局编号（§4.7 有解释），道理相同。

**为什么慢**（详见 §5）：
1. 纯标量循环——NEON/SVE 一点没用上，纯靠编译器；
2. 每个元素都做边界检查和负下标分支（②③），这些是"管理成本"而非"搬运成本"；
3. ⑤ 的取货地址依赖 ①④ 的结果，随机访存无法提前发起，DRAM 延迟直接暴露。

**优化主线**（详见 §6/§7）：
1. **少算**——把 ①~④ 提前批量做完（下标线性化），主循环只剩 `dst[i] = src[off[i]]`；
2. **少等**——软件预取 + 更宽的向量 gather，把 DRAM 延迟藏起来；
3. **少搬**——非临时访存、int32 下标、整 cache line 写，降低内存系统压力。

**SVE512 / SME 机器的额外机会**（详见 §7）：
- **非流式 SVE 就是 512-bit**（本文目标机如此），一条 gather 指令一次取 8 个元素，
  直接可用，无需任何模式切换；
- SVE2 的**非临时访存**让"取一次就不再用"的数据不污染缓存；
- SME 的流式模式与 ZA 是追加能力。注意这不是所有机器的常态——不少 SME 服务器
  非流式只有 128/256-bit，那种机器的 512-bit 要进流式模式才拿得到（§7.1 有对照表）。

---

## 2. 预备知识（按需选读）

### 2.1 两种索引：基本索引 vs 高级索引

`x[...]` 方括号里的东西分两大类，**它们的代价天差地别**：

| | 基本索引 | 高级索引（advanced indexing） |
|---|---|---|
| 下标类型 | `int`、`slice`（`1:3`）、`None`、`...` | **Tensor**（如 `tensor([1,3])`）、bool mask |
| 返回 | 原数据的**视图**（view），零拷贝 | **新分配**的 tensor，逐元素搬运数据 |
| 实现 | `select`/`slice`/`unsqueeze`，只改形状和步长 | `aten::index`，走本文的完整链路 |
| 例子 | `x[2]`、`x[:, 1:3]` | `x[[1,3]]`、`x[:, idx]`、`x[mask]` |

> **避坑提示**：`x[:, 1:3]` 是免费的（视图），`x[:, [1, 2]]` 却要搬数据。
> 模型里如果把切片写成 list/tensor 下标，等于凭空引入一次 gather。

### 2.2 gather：`index` 的本质，配一个比喻

**gather（按索引收集）**：给定一张"清单"（下标 tensor），按清单顺序从源数组取数。
与之相对的是 **scatter（按索引分发）**（`index_put`：把值按清单写到目标位置）。

```
self（仓库）:      [ A, B, C, D, E ]        下标（清单）: [3, 0, 3, 1]
                      ↓  ↓  ↓  ↓
result（传送带）:  [ D, A, D, B ]           ← 按清单顺序取货、顺序放货
```

记住这幅图，后文所有优化都能映射到它：
- **清单读起来顺不顺**（下标是否连续、是否重复读）；
- **取货地址算得快不快**（每件都要心算库位 = 偏移计算）；
- **货架跑不跑冤枉路**（随机 vs 顺序访存、缓存命中）。

### 2.3 Arm 服务器上的 SIMD 家族：NEON / SVE / SVE2 / SVE512 / SME

只列对本文有用的部分，一个演进故事讲完：

| 名称 | 一句话 | 向量宽度 | 与本文的关系 |
|---|---|---|---|
| **NEON** | Arm 的"传统 SIMD"，对标 x86 SSE/AVX | 固定 128-bit | PyTorch aarch64 的默认向量后端；**没有 gather 指令** |
| **SVE** | 可伸缩向量扩展：宽度由硬件定，软件运行时查询 | 128~2048-bit | 提供 `gather` 指令（`LD1G` 系）；PyTorch 目前只按 **256-bit 固定宽度**编译 |
| **SVE2** | SVE 的指令增强版 | 同上 | 新增**非临时访存**（`LDNT1`/`STNT1`）、128-bit 元素 gather（`LD1Q`）等 |
| **SVE512** | 向量宽度为 512-bit 的 SVE 实现 | 512-bit | 一次 gather 8×int64 或 8×f32；**本文目标机即此配置** |
| **SME** | Armv9 矩阵扩展：① 流式 SVE 模式（SSVE）② ZA 矩阵寄存器 | 流式宽度独立于非流式宽度 | 对 `index` 的价值：流式模式（部分机器比非流式更宽）与 ZA（研究性）。**非流式宽度由实现决定——本文目标机非流式即 512-bit，不进流式也能用满** |

**一个容易踩的坑（§7 会展开）**：非流式 SVE 宽度与 SME 流式模式宽度是**两个独立的
实现参数**，发布前务必分别确认。典型的 SME 服务器（如 Neoverse V2）非流式只有
128/256-bit，想用 512-bit 必须 `SMSTART` 进流式模式、`SMSTOP` 退出（切换有几百
周期代价，只适合长循环）。**而本文的目标机器非流式 SVE 就是 512-bit**——不需要进
流式模式，直接按 512-bit 用；SME 在这台机器上的增量价值是流式模式（若更宽）与 ZA。

### 2.4 内存与微架构术语速查表

| 术语 | 人话解释 |
|---|---|
| cache line（缓存行） | 内存与缓存之间搬运的最小单位，通常 64 字节；取 1 个 float 也会把整行 64B 拉进缓存 |
| DRAM 延迟 | 内存访问约 100~200 个 CPU 周期，是 L1 命中的几十倍——随机 gather 慢就慢在这 |
| MLP（内存级并行） | 一个核同时"在路上"的访存请求数；请求越多、延迟隐藏越好 |
| LSU | load/store 单元，CPU 里真正执行访存指令的部件；gather 每周期吞吐有限 |
| NUMA | 双路服务器每个 CPU 插槽有自己的内存；访问"对面插槽的内存"更慢 |
| first-touch | 内存页归属第一次写它的那个 NUMA 节点；分配后先让本节点线程摸一遍 |
| 非临时访存（non-temporal） | 告诉硬件"这数据用完就扔，别进缓存"，避免污染缓存 |
| 软件预取（prefetch） | 用 `prfm` 指令提前把数据拉进缓存，藏住 DRAM 延迟 |
| grain size | 并行切分的最小块大小；小于它的活不派给其他线程 |
| 边界检查外提（hoisting） | 把"每个元素都要做的检查"改成"循环前一次性检查" |
| VL（vector length） | SVE 向量寄存器的位宽；`svcntd()` 等内建可运行时查询 |

---

## 3. `aten::index` 的语义：它到底做什么

先把规则说清楚（写优化代码时每条规则都会影响实现）：

```python
x = torch.randn(4, 5)          # self
idx = torch.tensor([1, 3, 0])  # 下标
```

1. **下标放在哪一维，就取哪一维的元素**：`x[idx]` 取第 0 维 → 结果形状 `(3, 5)`；
   `x[:, idx]` 取第 1 维 → 结果形状 `(4, 3)`。
2. **多个 Tensor 下标会 broadcast 到同一形状**（NumPy 规则）：
   `x[i, j]` 中 `i`、`j` 形状必须可广播，结果的这部分形状 = 广播后的形状。
3. **bool mask 等价于坐标索引**：`x[mask]`（mask 形状须匹配对应维度）=
   先 `mask.nonzero()` 得到坐标，再按坐标取。所以 mask 索引隐含一次额外的
   全张量扫描（见 §4.4 第 ① 步）。
4. **负下标合法**：`-1` 表示倒数第一个（`v < 0` 则 `v += size`）。
5. **越界直接抛 IndexError**——注意这是**逐元素运行时检查**，不是编译期可消除的（§5）。
6. **结果总是新分配的连续 tensor**（不存在视图）。

**它与"亲戚算子"的分工**（读源码时别走错门）：

| 算子 | 语义 | CPU 实现位置 |
|---|---|---|
| `aten::index` | 任意个 Tensor 下标做 gather（本文主角） | `native/cpu/IndexKernel.cpp` |
| `aten::index_select` | 单维下标 gather，**有更快特化** | `native/TensorAdvancedIndexing.cpp` |
| `aten::gather` | 逐元素按另一个 tensor 的坐标取 | `native/cpu/ScatterGatherKernel.cpp` |
| `aten::take` | self 拉平成一维后按线性索引取 | `native/cpu/IndexKernel.cpp` |
| `aten::index_put` | `x[indices] = values`（scatter 写） | `native/cpu/IndexKernel.cpp` |
| `aten::nonzero` | bool mask → 坐标矩阵 | `native/TensorAdvancedIndexing.cpp` |

`aten::index` 在 `native_functions.yaml` 中的声明（行 3093/3105）——注意它是
**structured 算子**，且带 **precomputed** 参数（meta 阶段算好的 `sizes/strides`
直接传给 kernel，这是个不错的模式）：

```yaml
- func: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  structured_delegate: index.Tensor_out
  dispatch:
    QuantizedCPU: quantized_index          # 量化走 dq→index→q 的朴素路径

- func: index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  precomputed:
  - indices -> DimVector sizes, DimVector strides   # = indexed_sizes / indexed_strides
  dispatch:
    CPU, CUDA, MPS: index_out
```

---

## 4. PyTorch CPU 实现全链路

### 4.1 全景图：五站流水线

`y = x[idx]` 从 Python 到内存搬运，要过五站。**前三站全是"准备工作"（零拷贝），
真正的数据搬运只发生在第四站**：

```
┌─────────────────────────────────────────────────────────────────────┐
│ ① Python 入口          y = x[idx]                                    │
│    torch/csrc/autograd/python_variable_indexing.cpp: THPVariable_getitem │
├─────────────────────────────────────────────────────────────────────┤
│ ② 索引翻译层          整数/切片/.../None → select/slice 视图（零拷贝）  │
│    TensorIndexing.h: get_item / applySlicing / dispatch_index        │
│    Tensor 下标 → 收集起来，交给下一站                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ③ 归一化（make_info）  bool mask → nonzero 坐标；多下标 broadcast；    │
│    int32→int64；下标挪到前面；算出 indexed_sizes/strides              │
│    TensorAdvancedIndexingUtils.h:72 / TensorAdvancedIndexing.cpp:628 │
├─────────────────────────────────────────────────────────────────────┤
│ ④ 搭台（TensorIterator）把 output/restrided self/各下标放进统一的     │
│    多维遍历框架，重排维序使输出最内层连续                              │
│    TensorAdvancedIndexing.cpp: build_index_op(468)                   │
├─────────────────────────────────────────────────────────────────────┤
│ ⑤ 唱戏（cpu_index_kernel）逐元素标量 gather：                         │
│    dst[i] = src[ offset(idx[i]) ]                                    │
│    IndexKernelUtils.h:52 + IndexKernel.cpp:24                        │
│    ↓ 多线程                                                          │
│ ⑥ 并行（at::parallel_for，grain=3000）多线程分块执行 ⑤                │
└─────────────────────────────────────────────────────────────────────┘
```

下面逐站展开。

### 4.2 第一站：Python 入口（`THPVariable_getitem`）

文件 `torch/csrc/autograd/python_variable_indexing.cpp`，入口在行 390。

流程（简化）：

1. 单对象下标（`x[3]`、`x[1:2]`、`x[None]`、`x[...]`、`x[True]`）→ 直接转成 C++ 的
   `at::indexing::TensorIndex`，调 `at::indexing::get_item`（行 400-428）；
2. 多对象下标（`x[i, 1:3]`）→
   - `count_specified_dimensions`（行 91）：统计下标"消耗"了多少维
     （普通 tensor 下标消耗 1 维，bool/uint8 mask 消耗 `mask.dim()` 维）；
   - `applySlicing`（行 213）：把整数/切片/`None`/`...` 翻译成 `select`/`slice`/`unsqueeze`
     （全是视图操作），同时把 Tensor 下标收集进 `variableIndices`；
   - `variableIndices` 非空 → `at::indexing::dispatch_index`（行 460）→ `Tensor::index`。

**本站要点**：所有"便宜"的索引在这一站就被翻译成视图，根本不会走到 `aten::index`；
只有真正的 Tensor 下标（含 mask 展开）才会继续往下走。

### 4.3 第二站：索引翻译层（`TensorIndexing.h`）

文件 `aten/src/ATen/TensorIndexing.h`。核心是 `get_item`（行 636）和
`applySlicing`（行 539）里的逐维处理 `handleDimInMultiDimIndexing`（行 438）：

| 下标类型 | 动作 | 对 dim 游标的影响 |
|---|---|---|
| 整数 | `select`（视图） | 维度被"消费"，不变 |
| 切片 | `slice`（视图） | +1 |
| `...` | 跳过若干维 | +ellipsis_ndims |
| `None` | `unsqueeze` | +1 |
| 布尔标量 | `unsqueeze` + 记录一个等价 tensor 下标 | +1 |
| **Tensor** | `recordTensorIndex`（行 314）收集 | 普通 tensor +1；mask +mask.dim() |

收集完的 Tensor 下标经 `dispatch_index`（行 580）——去掉末尾空位、转成
`List<optional<Tensor>>`——调用 `Tensor::index`（`TensorIndexing.cpp:67`），
进入 ATen 的 `at::index`。

**本站要点**：到这里为止，一切切片/select 都是零拷贝视图；数据搬运一行都还没发生。

### 4.4 第三站：归一化（`make_info`，五步流水）

文件 `aten/src/ATen/native/TensorAdvancedIndexingUtils.h:72`。这一站把五花八门的
下标**归一化成统一格式**：

```cpp
inline AdvancedIndex make_info(Tensor self, IOptTensorListRef orig) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);   // ⓪ 只允许 long/int/byte/bool
  auto indices = expandTensors(self, orig, true);     // ① bool mask → nonzero 坐标
  indices = expand_outplace(indices);                 // ② 所有下标 broadcast 到同形状
  while (indices.size() < self.dim())                 // ③ 补空位到 self.dim()
    indices.emplace_back();
  if (!hasContiguousSubspace(indices))                // ④ 下标不连续邻接 → 一起 transpose 到前面
    std::tie(self, indices) = transposeToFront(self, indices);
  for (auto& indice : indices)                        // ⑤ int32 → int64
    if (indice.defined() && indice.dtype() == at::kInt) indice = indice.to(at::kLong);
  return AdvancedIndex(self, indices);
}
```

五步的人话版：

- **① mask → 坐标**（`IndexingUtils.h:23`）：`mask.nonzero()` 得到 `[nnz, mask.dim()]`
  的坐标矩阵，然后 `select(1, j)` 逐列取出。**注意取出的列是非连续的**
  （行 stride），后面 kernel 读它们会更费劲——mask 索引天然比整型索引多付一点成本；
- **② broadcast**：生成 0-stride 的 expanded 视图（不拷贝数据）；
- **④ transposeToFront**（`IndexingUtils.h:126`）：如果非空下标不相邻（如 `x[a, :, b]`），
  把 self 和下标一起 permute，让被索引维**挪到最前面连续排布**——这保证了后面
  `AdvancedIndex` 的 `dims_before / dims_indexed / dims_after` 三段式结构；
- **⑤ int32→int64**：统一按 int64 读。**这是一个可优化的开销点**（§6.4）。

然后 `AdvancedIndex` 构造函数（`TensorAdvancedIndexing.cpp:628`）把信息装订成册：

- `dims_before / dims_after`：下标前面/后面还剩几维没被索引
  （`x[:, idx]` 的 dims_before=1；`x[idx, :]` 的 dims_after=1）；
- `indexed_sizes`：每个被索引维的**原始尺寸**（给边界检查用）。
  例：`x` 形状 `(3,4)` 被两个下标分别索引 dim0/dim1 → `[3, 4]`；
- `indexed_strides`：每个被索引维的**字节 stride**（给偏移计算用）：
  `indexed_strides[j] = src.stride(dim_j) * element_size_bytes`。
  上例（float32，元素 4 字节）→ `[4×4B, 1×4B] = [16, 4]`。
  这两个数组经 precompute 传进 kernel，就是 §1 核心循环里的 `size_j / stride_j`；
- `info.src`：self 被"改造"后的视图（见下一小节）；
- `info.indices`：每个下标被 reshape 成可广播的形状。

### 4.5 核心机关：`restride_src`（把"取哪"和"放到哪"解耦）

这是整个实现里最巧（也最绕）的一步，值得单独讲清楚（`TensorAdvancedIndexing.cpp:595`）。

**问题**：TensorIterator 天生只会"按形状和步长做同步的多维遍历"，但 `index` 里
输出和输入的对应关系是**由下标的值**决定的，形状对不上，怎么放进同一个迭代器？

**解法**：把 self "改造"成一个形状与输出完全一致、但被索引维 **stride 置 0** 的视图：

```
例：x = randn(4,5)，y = x[:, [1,2]]

改造前 self:   shape [4,5], strides [5,1]        （float，元素 4 字节）
改造后 info.src: shape [4,2], strides [5,0]      ← dim1（被索引维）stride=0！
下标 reshape:   shape [1,2] → 广播成 [4,2], strides [0,8]
输出:           shape [4,2], strides [2,1]       （新分配，连续）
```

于是 TensorIterator 看到的三个操作数**形状完全相同**，可以正常遍历：
- 输出、下标按各自 stride 正常前进；
- `info.src` 沿被索引维"原地踏步"（stride 0），沿着未索引维正常前进——
  **"取第几行"由迭代器管，"取行内第几个"由 kernel 拿下标值算**。

Kernel 里算的那部分就是：`offset = 下标值 × indexed_strides[j]`（字节偏移）。
一个机构，把"遍历"与"寻址"干净地分了工。

### 4.6 第四站：TensorIterator（搭台）

`build_index_op`（`TensorAdvancedIndexing.cpp:468`）把三个角色放进 TensorIterator：

```cpp
config.set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_output(result)                 // 输出
      .add_owned_const_input(info.src);   // 输入0：restride 过的 self
for (auto& index : info.indices)
  config.add_owned_const_input(index);    // 输入1..k：各下标 tensor
```

meta 函数（`TORCH_PRECOMPUTE_META_FUNC2(index, Tensor)`，行 507）在此完成检查
（下标数量/设备/dtype/overlap），并把 `indexed_sizes`/`indexed_strides` 作为
**precomputed** 返回；实现函数 `TORCH_IMPL_FUNC(index_out)`（行 715）一行转发：

```cpp
index_stub(device_type(), *this, sizes, strides);   // → cpu/IndexKernel.cpp
```

TensorIterator 搭台时还做两件影响性能的事（`TensorIterator.cpp`）：

- **维序重排** `reorder_dimensions`（行 232）：把所有维按 stride **升序**排，
  使最内层迭代维是输出上最连续的维——保证写输出尽量顺序。
  直觉：让"每步走 4 字节"的维当内层、"每步跨一整行"的维当外层，
  CPU 预取器和缓存最爱这种内层连续的循环（§4.9 例 2 有一次完整演示）；
- **维合并** `coalesce_dimensions`（行 638）：相邻维若满足
  `shape[n] × stride[n] == stride[n+1]`（对所有操作数成立）就合并成一个。
  直觉：形状 `[4,2]`、步长 `[2,1]` 的连续输出，两个维可并成一个长度 8 的
  一维循环——少一层循环开销。

### 4.7 第五站：`cpu_index_kernel`（唱戏）

文件 `aten/src/ATen/native/cpu/IndexKernelUtils.h:52`（kernel 主体）和
`IndexKernel.cpp:24`（dtype 分发入口）。这就是 §1 速览里那段核心循环的真身：

```cpp
void index_kernel(...) {                              // IndexKernel.cpp:24
  AT_DISPATCH_V2(iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride,
      [](char* dst, char* src, int64_t offset) {      // ← gather 动作本身
        *(scalar_t*)dst = c10::load((scalar_t*)(src + offset));
      });
  }, ...);
}
```

```cpp
template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIteratorBase& iter, IntArrayRef index_size,
                      IntArrayRef index_stride, const func_t& f, ...) {
  const int index_parallel_grain_size = 3000;         // 并行最小块（为 index_put 调出来的）
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];                              // 输出
    char* src = data[1];                              // restrided self
    if (is_constant_index(ntensor, strides)) {        // 特化：所有下标 stride 全 0
      int64_t offset = indexer.get(0);                //   → 只算一次偏移
      for (i in 0..n) f(dst + strides[0]*i, src + strides[1]*i, offset);
    } else {
      for (i in 0..n) {
        int64_t offset = indexer.get(i);              // ← 每元素：读下标+检查+归一+乘加
        f(dst + strides[0]*i, src + strides[1]*i, offset);
      }
    }
  };
  iter.for_each(loop, index_parallel_grain_size);     // 并行！
}
```

先解释 `loop` 每次收到的参数（对照代码里的 `data` / `strides` 两个数组）：

- `data[0]` = 输出指针，`data[1]` = restrided self 的指针，`data[2..k+2]` = 各下标
  tensor 的指针——**都已被 TensorIterator 推进到"当前这段一维切片"的起点**
  （比如 `x[:, idx]` 的某一行开头）；
- `strides[...]` 与之一一对应，是各 tensor 沿**内层迭代维**的字节步长
  （下标连续时是 8；下标在该维上是广播的则为 0——此时内层循环反复读同一个下标值）；
- `n` = 该切片的长度。所以 `Indexer::get(i)` 里的 `i` 是**切片内**偏移（0..n-1），
  不是全局元素编号。§1 的简化写法把 `i` 当全局编号，在"没有未索引维"的例子里
  （如 `x[i_idx, j_idx]`）两者恰好一致；有未索引维时（如 `x[:, idx]`）每个外层
  行都会重新从 `i=0` 数起——这正是 §4.9 例 2 "每行重读清单"的根源。

其中 `Indexer::get`（同文件行 18）就是每个被索引维都要跑一遍的"管理成本"：

```cpp
int64_t get(int64_t idx) {
  int64_t offset = 0;
  for (j in 0..num_indexers) {
    int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]]; // 读下标（可跨步）
    TORCH_CHECK_INDEX(value >= -size && value < size, ...);            // 边界检查（关不掉！）
    if (value < 0) value += size;                                      // 负下标归一
    offset += value * original_strides[j];                             // 乘字节 stride 累加
  }
    return offset;
}
```

逐字段对应（拿 §1 的数值例子 `x[i_idx, j_idx]` 对号入座）：

- `indexers[j]`：第 j 个下标 tensor 在当前切片的基址。j=0 时指向 `i_idx` 的数据，
  `indexer_strides[0] = 8`（连续 int64，沿内层维每步跨 8 字节）；
- `original_sizes / original_strides`：就是 §4.4 装订的 `indexed_sizes /
  indexed_strides`，本例为 `[3,4]` / `[16,4]`（字节）；
- 于是 `i=0, j=0` 这一次调用：`value = 2` → 检查 `-3 ≤ 2 < 3` 通过 →
  `offset += 2 × 16 = 32`——与 §1 的推演逐字一致。

两个值得记住的细节：

- **`TORCH_CHECK_INDEX` 是始终生效的运行时检查**（`c10/util/Exception.h:551`——
  `TORCH_CHECK` 不受 `NDEBUG` 影响），所以边界检查没法靠 Release 编译消除，
  只能靠"外提"（§6.1）；
- **`is_constant_index` 特化**（行 7）：当所有下标在内层维 stride 全为 0
  （典型：0 维标量下标 `x[tensor(2)]`），偏移只算一次，内层退化成近似 memcpy——
  这是当前实现里唯一的"捷径"。

**`index_put` 是同一个舞台的镜像**（`IndexKernel.cpp:165`）：iterator 换成
"output=info.src（self 的 restrided 视图）+ value + 下标"，gather 改 scatter。
两个额外分支：单标量 value + 纯 mask 场景短路成 `masked_fill_`
（`canDispatchToMaskedFill`，`TensorAdvancedIndexingUtils.h:26`）；
`accumulate=True` 时非确定性要求下强制串行、float 走原子加（行 171-193）。
CUDA/XPU 才有的"按索引排序"路径（`index_put_with_sort_stub`）CPU 不走。

### 4.8 并行机制：怎么把循环分给多核

`iter.for_each(loop, 3000)` 一路向下（`TensorIterator.cpp:763`）：

```
TensorIteratorBase::for_each(loop, grain=3000)
  ├─ numel < 3000 或单线程 → 串行执行
  └─ 否则 at::parallel_for(0, numel, 3000, ...)        Parallel-inl.h:10
       └─ invoke_parallel：把 [0,numel) 切成连续区间    ParallelNative.cpp:144
            chunk = max(3000, numel / num_threads)
            每个线程领一段，各自 serial_for_each
                 └─ DimCounter 把线性区间翻译回多维坐标  TensorIterator.cpp:1712
                      → 以 2D 切片（内层 size0 × 外层 size1）回调 loop
```

什么是"把线性区间翻译回多维坐标"？线程领到的任务是"处理第 begin 到第 end 个
输出元素"这种**线性编号**，但数据是按形状摆成多维的。`DimCounter` 就是个进制
换算器——比如形状 `[4, 8]` 时，第 19 个元素（从 0 数起）就是"第 2 行第 3 列"
（19 = 2×8 + 3，除法取余各得一个坐标），它把线性编号这样拆回各维坐标，
再据此把 `data[]` 里每个指针推到正确位置，然后按"内层一段 × 外层逐行"的方式
回调 `loop`（内层长度就是 §4.7 说的 `n`）。

几个要点：

- `3000` 这个 grain size 注释里写明是**为 `index_put` 的 benchmark 调的**，
  比默认 `GRAIN_SIZE=32768`（`TensorIterator.h:78`）激进得多——目的是多核负载更均衡；
- 原生线程池按 `task_id` **连续切分**（非 work-stealing），2 路 NUMA 下可能出现
  访存倾斜（§6.5）；
- 并行后端：默认 OpenMP（`USE_OPENMP=ON`，`CMakeLists.txt:310`），
  不可用则回退 native 线程池；两者都定义 `INTRA_OP_PARALLEL` 使 `parallel_for` 生效。

### 4.9 六个例子走读（形状/步长推演）

每个例子给出"数据怎么流"，读完能对号入座自己的使用场景。
（设 float32、元素 4 字节；`indexed_strides` 均为**字节**。）

**例 1：一维 gather——`x = randn(4); y = x[idx]`，`idx = [1,3,0]`**
- `indexed_sizes=[4]`（边界 `[-4,4)`），`indexed_strides=[4]`（dim0 字节 stride）；
- 下标连续（stride 8B），输出连续；
- 每元素：读 `idx[i]` → 检查/归一 → `offset = idx[i]*4` → `dst[i]=src[offset]`；
- **人话**：最标准的"按清单取货"。输出顺序写、self 随机读（idx 递增则近似顺序读）。

**例 2：带前置维——`x = randn(4,5); y = x[:, [1,2]]`（最典型的真实场景）**
- `info.src` shape `[4,2]` strides `[20,0]`；下标广播成 `[4,2]` strides `[0,8]`；
- 维序重排后内层维 = 下标维（size 2），外层 = 4 行；每行调用一次 `loop(n=2)`：
  `dst[row*2+i] = src[row*20 + idx[i]*4]`；
- **人话**：同一张 2 元素清单，**每行都重新读一遍**——4 行共读 8 次，实际只需 2 个值。
  行数和 idx 都大时（如 `x[:, perm]`），这是显著浪费（§6.1 线性化的头号目标场景）。

**例 3：多下标——`x = randn(3,4,5); y = x[i, j]`，`i`、`j` 同为 shape `[100]`**
- `indexed_sizes=[3,4]`，`indexed_strides=[80,20]`；
- 每元素：`offset = i[k]*80 + j[k]*20`——**2 次下标读 + 2 次检查 + 1 次随机 gather**；
- **人话**：管理成本随下标个数线性放大，多下标场景最吃亏。

**例 4：bool mask——`x = randn(4,5); y = x[mask]`，mask 形状 `[4,5]`**
- 先 `mask.nonzero()` → `[nnz,2]` 坐标，取两列作下标（**非连续**，行 stride 16B）；
- 之后同例 3：`indexed_sizes=[4,5]`，`indexed_strides=[20,4]`；
- **人话**：mask 索引 = 一次额外全扫描（nonzero）+ 非连续下标读取 + 多下标开销，
  是所有形式里最贵的；能用整型下标就别用 mask。

**例 5：0 维标量下标——`x = randn(4,5); y = x[tensor(2)]`**
- 下标广播后所有 stride 为 0 → 命中 `is_constant_index` 特化；
- `offset = 2*20` 只算一次，内层是连续 `dst[i] = src[base+offset]`，近似 memcpy；
- **人话**：当前实现里唯一的"免费午餐"。但注意 `x[2]`（int）根本不走这条路
  （早在第二站就 select 了），只有 `x[tensor(2)]` 这种罕见写法才会到这。

**例 6：对照 `index_put`——`x[idx] = v`（accumulate=False）**
- 同构的循环，方向反过来：`*(dst + offset) = load(src)`，dst 是 self 的 restrided 视图；
- `accumulate=True` 时：非确定性 + float → `cpu_atomic_add_float` 原子加；
  确定性模式 → **强制串行**（`IndexKernel.cpp:171-193`）；
- **人话**：写散（scatter）比读聚（gather）多一个"重复下标怎么处理"的语义包袱，
  确定性要求下并行直接被关掉。

---

## 5. 性能画像：时间都花在哪

### 5.1 每个输出元素的成本拆解

把 §4.7 的内层循环逐条标价。对照 §1 的数值推演来读：j=0 时"读下标值"就是读出 2、
"偏移乘加"就是 `2×16=32`——下面每一行都是**真实执行的指令**，不是抽象描述：

```
每个输出元素 = Σ(每个被索引维 j) [
    读下标值       8B 内存读（依赖链的起点，可能非连续）
    边界检查       2 次比较 + 分支（恒定开销，Release 也关不掉）
    负下标归一     1 个分支
    偏移乘加       1 乘 1 加
] + self gather 读（地址依赖前面所有步骤 → DRAM 延迟完全暴露）
  + 结果 store    （连续写，便宜）
```

### 5.2 六个定性结论

1. **内存系统受限**：每元素至少搬 `8B(下标) + 2×elem_size(数据进出)`，
   多下标时下标流量线性放大；1 字节 dtype（bool/uint8）时管理成本远超搬运本身；
2. **随机 gather 的 DRAM 延迟是最大 stall 源**——尤其 self 远大于 LLC、下标乱序时；
3. **恒定开销占比高**：边界检查/负归一/`Indexer` 间接寻址，数据全在 L1 也省不掉；
4. **零 SIMD、零预取、零下标预处理**：`index_stub` 的 kernel（`IndexKernel.cpp:29`
   的 lambda）纯标量。通用路径被逐元素分支挡住向量化；`is_constant_index`
   特化的行拷贝环虽无分支，但 dst/src 步长是**运行期值**（编译器不做
   stride versioning），实际同样向量化不了——SVE256 重编译（§5.4）对它没有
   增益。注意边界：同文件的 `flip_cpu` 用了 `Vectorized`，邻居 `index_select`
   的行拷贝走 `memcpy`/向量化的 `copy_stub`（§6.3）——它们在 aarch64 上
   吃得到 NEON/SVE，唯独 `index` 本体吃不到；
5. **并行粒度 3000**：几十万元素以上能打满多核；小 tensor 退化为串行，
   dispatch/meta 固定开销反成大头；
6. **缺少单维特化**：对照 `index_select_out_cpu_dim1_`（`TensorAdvancedIndexing.cpp:1538`）
   ——它**预先一次性检查下标范围**（`check_indexarray_range`）+ 块 memcpy + 顺序读下标，
   是"单维、大块"场景的正确姿势；通用 `index` 没有这条快速路径。

### 5.3 什么时候 `index` 会成为瓶颈（给模型开发者）

- 大 embedding 查表（`weight[idx]`，每 step 数百万元素）；
- 大 batch 的按行/按列取样（`x[:, perm]`、负采样）；
- mask 索引大 tensor（`x[mask]`，mask 命中率高时输出巨大）；
- 以上操作出现在**训练热路径**且被 profiler 标红时。

**不改代码的缓解**：优先用 `index_select`/`gather`（有特化）、避免 mask 改用
`nonzero` 结果复用、或交给 `torch.compile`（§6.7）。

### 5.4 aarch64 构建现状（与优化可行性直接相关）

- 顶层 CMake 识别 `CPU_AARCH64`（`CMakeLists.txt:189-190`）；
- 向量后端：`vec128.h` 在 `__aarch64__` 下选择 **NEON**（`vec128.h:6-13`）；
- **SVE256 变体**：`FindARM.cmake` 探测通过后，`Codegen.cmake:425-429` 以
  `-march=armv8-a+sve+bf16 -msve-vector-bits=256` 追加 `SVE256` capability——
  **注意是硬编码 256-bit**；
- **没有 SVE512 / SME 支持**：`FindARM.cmake` 只探测 SVE256；`arm_sme.h`、
  `-march=armv9-a+sme` 均未接入。要用上它们需要新的编译路径（§7.6）；
- `native/cpu/*.cpp` 按 capability 重复编译（`Codegen.cmake:363-449`），但
  `cpu_index_kernel` 里没有任何向量类型——对 `index` 而言重复编译没有增益
  （同文件的 `flip_cpu` 用了 `Vectorized`，倒是受益者；对 index 这反而说明
  改造空间干净）；
- aarch64 Linux 链接 `-z,max-page-size=0x10000`（64KiB 页，`CMakeLists.txt:699-702`）
  利于大 tensor 的 TLB；所有构建强制 `-fno-omit-frame-pointer`（行 1196-1201）。

---

## 6. 通用优化思路（不挑硬件）

### 6.0 先给一个思考框架：少算、少等、少搬

`index` 的每元素工作可以三分，所有优化都落在这三类里：

| 类别 | 含义 | 对应手段 |
|---|---|---|
| **少算** | 减少每元素的指令数（寻址/检查/分支） | A 线性化、D int32、J 微优化 |
| **少等** | 藏住内存延迟、提高访存并行度 | C 预取、B/L 向量 gather、E 块拷贝 |
| **少搬** | 降低内存系统的总流量与污染 | M 非临时、O 整行写、H 排序分块、F NUMA |

看任何一个优化提案，先问它属于哪类、瓶颈是否匹配（延迟受限选"少等"，
带宽受限选"少搬"，小 dtype/多下标选"少算"）。

以下每条优化统一按 **问题 → 直觉 → 做法 → 为什么有效 → 代价/风险** 展开。

### 6.1 A. 下标线性化 + 检查外提（收益极高，本文第一优先级）

**问题**：主循环里每个元素都要跑一遍 `Indexer::get`——读下标、边界检查、负归一、
乘加。这些是"每件货都心算一遍库位"。

**直觉**：清单是提前知道的！出发前把整张清单**一次性**换算成"库位编号"
（字节偏移），顺便把非法取件码全部核掉——之后取货时照单直取，不用再动脑。

**做法**：
1. 预处理扫描所有下标（可向量化）：检查 `-size <= v < size`（**一次**，失败才抛错）；
2. 负值归一：`v < 0 → v += size`（写回或写进临时缓冲）；
3. 多维折算成一维：`off[i] = Σ_j v_j[i] * indexed_strides[j]`；
4. 主循环变成纯粹的 1D gather：`for i: dst[i] = load(src_base + off[i])`。

拿 §1 的数值例子说：线性化就是把推演里的 ①~④ **提前算完**，得到
`off = [44, 4]`；主循环只剩 `dst[0]=src[44]; dst[1]=src[4]`——两个 load、
两个 store，没有检查、没有分支、没有乘加。对 `x[:, idx]`（例 2）这种"前置维
重复同一组下标"的场景，偏移数组只算一次（尺寸 = 广播后下标形状），每行直接
复用——逐行重读彻底消失。

**为什么有效**：主循环指令数减 50~70%（多下标更多）；分支归零（无检查、无负归一）；
下标读取从"跨步/重复"变成"顺序读偏移数组"，预取器友好；且为 §7 的向量 gather
铺平了道路（向量 gather 需要连续的偏移输入）。

**代价/风险**：需要一块 `int64` 偏移缓冲（尺寸 = 结果 numel，或例 2 场景下只是
下标 numel）；预处理本身有一次全量读下标的开销——对极小的调用反而变慢，
需要阈值（如 `numel < 1024` 走原路径）。`_unsafe_index` 已证明"跳过部分检查"
在语义上可行（`TensorAdvancedIndexing.cpp:736`），线性化只是把检查挪了位置、
并未取消，风险可控。

### 6.2 C. 软件预取（收益中高，几乎白送）

**问题**：gather 的 load 地址依赖下标计算，DRAM 延迟（100~200 周期）直接暴露成 stall。

**直觉**：清单是顺序可读的——让助手提前 K 件把货搬到手边，等取到那件时货已就位。

**做法**：主循环里对未来的地址发预取提示（aarch64 即 `prfm`，编译器内建
`__builtin_prefetch`）：

```c
for (i...) {
    __builtin_prefetch(&src_base[off[i + K]], 0, 3);   // K ≈ 16~64，按延迟×吞吐估
    dst[i] = load(src_base + off[i]);
}
```

**为什么有效**：预取指令不阻塞、不占寄存器结果，只让内存系统提前干活；
随机 gather 场景下常有两倍级提升。

**代价/风险**：K 太小藏不住延迟、太大被 L1 挤出；下标极乱时预取命中率下降。
前提是偏移可顺序预读（即 6.1 已做，或下标本来连续）。

### 6.3 E. 单维 + 前置维的特化：对标 `index_select`（收益中高）

**问题**：`x[:, idx]` 是最高频的真实场景，但通用 kernel 为它付出全套管理成本，
而它明明是"每行按同一张清单取一串连续块"。

**直觉**：当被索引维是行内维、下标连续时，每次取货搬的不是 4 字节而是
"一整块"（后缀维的连续内存）——从"逐件取"升级成"整箱搬"。

**做法**：检测 `dims_indexed == 1` 且被索引维可合并到内层、下标连续：
`for each row: for i: memcpy(dst+..., src+row_base+idx[i]*block, block_bytes)`，
前置维用 `at::parallel_for` 并行。参考实现就是
`index_select_out_cpu_dim1_`（`TensorAdvancedIndexing.cpp:1538-1595`）——
它还演示了**边界检查前置**（`check_indexarray_range` 先查一遍）。
顺带一个实测含义：它的行拷贝是字面意义的 `memcpy`（`TensorAdvancedIndexing.cpp:1589`，
通用路径 contiguous 分支同样，行 1766）——libc 的 memcpy 在 aarch64 上由
NEON/SVE 实现，所以 `index_select` 的搬运本身已吃到 SIMD；非 contiguous 时走
`copy_stub`，其 CPU 实现（`CopyKernel.cpp:231` 的 `cpu_kernel_vec`）也是向量化的。

**为什么有效**：块 memcpy 让单位管理成本摊到整块；self 的访问从"逐元素随机"
变成"块级随机"（块内连续），缓存与带宽效率大幅提升。

**代价/风险**：只覆盖单下标场景；多下标/广播/0 维下标需回退通用路径。

### 6.4 D. 下标用 int32（收益中，顺手）

**问题**：`make_info` 第 ⑤ 步把所有下标统一成 int64（`TensorAdvancedIndexingUtils.h:97-99`），
kernel 也按 8 字节读——下标流量翻倍。

**直觉**：绝大多数真实下标值远小于 2³¹，4 字节足够。

**做法**：kernel 按下标 dtype 分派（int32/int64 两个模板实例）；
或预处理时转换并保持 int32 缓冲（`_unsafe_index` 已允许 int 下标，语义无碍）。

**为什么有效**：下标内存流量减半；NEON/SVE 向量化时一次处理的下标数翻倍
（128-bit 装 4 个 int32 vs 2 个 int64）。

**代价/风险**：多一条模板路径；超大 tensor（>2³¹ 元素）不适用。

### 6.5 F. NUMA 感知的并行与分配（收益中，服务器必看）

**问题**：native 后端按 `task_id` **连续切分**区间（`ParallelNative.cpp:163-167`），
2 路服务器上任务与内存页的归属可能错位——线程在 A 节点、数据页在 B 节点。

**直觉**：让每个工人优先在自己仓库那排货架取货。

**做法**：
- 结果与偏移缓冲**first-touch**：分配后由将来写它的线程先摸一遍；
- `OMP_PROC_BIND=true` / `OMP_PLACES=cores` 绑核（OpenMP 后端）；
- self 的随机读跨节点无法避免，但可按"结果块对应 self 大致区域"做区块调度，
  让每个线程的 gather 集中在较近的 NUMA 页。

**为什么有效**：跨 NUMA 访存延迟约为本地的 1.5~2 倍；gather 本就是延迟敏感型。

**代价/风险**：区块调度实现复杂；绑核与容器/其他进程争抢需运维配合。

### 6.6 H. 排序 + 分块：把乱序 gather 变近似顺序（收益看数据，难度高）

**问题**：下标完全乱序时，self 的访问是纯随机——缓存与 DRAM 页都浪费。

**直觉**：把清单**按库位排好序**再取，走的路就顺了；取完再按原顺序摆回传送带。

**做法**：对偏移数组排序 → 按序 gather 到临时区 → 按原序 scatter 回结果。
等价于 CUDA `index_put_with_sort_stub` 的思路（`TensorAdvancedIndexing.cpp:562`）。

**为什么有效**：self 与结果都远大于 LLC、下标近乎均匀随机时，收益可达 2~3×。

**代价/风险**：排序 + 两次搬运的额外成本；下标本已有序时是纯负优化——
需要运行时判定（如相邻偏移差的方差）再选路径。

### 6.7 I. 借力 `torch.compile` / Inductor（不改 ATen 就能拿收益）

Inductor 把高级索引降成 `_unsafe_index`（`native_functions.yaml:3116`）：不允许
bool 索引（避免动态形状）、只接受 long/int 下标、直接调 `at::index`——
配合生成的内联 C++ 循环和算子融合（`index → add` 等），省掉 tensor 物化与
部分 dispatch。`_unsafe_masked_index`（带 fill 的掩码 gather）还能避免 `nonzero`
物化。**这是不改一行 ATen 源码就能生效的手段，SVE 机器上同样适用。**

### 6.8 G / J. 小优化（顺手做）

- **grain size 再调参**：`3000` 是为 `index_put` 调的；纯读的 `index` 可重测
  （4096/8192 候选）；
- **消除 `Indexer` 间接性**：把 `indexer_strides`/`original_sizes` 等拷进局部变量、
  指针加 `C10_RESTRICT`，帮助编译器展开 `get`（`IndexKernelUtils.h` 目前没有）；
- **`is_constant_index` 推广**：下标内层 stride 全 0 但外层有 stride 时
  （如 `x[[2], :]`），把 offset 计算提升到外层，内层退化为连续拷贝；
- **对齐**：偏移缓冲/输出按 128B 对齐，利于向量化与写合并；
- **极端小 case 短路**（G）：`numel` 很小时 dispatch/meta 固定开销占大头，
  可仿照 `canDispatchToMaskedFill` 的短路思路给"单 0 维下标"等加快速路径。

---

## 7. SIMD 优化：SVE256 / SVE512 / SME 机器

> 本章所有优化都**以 §6.1（下标线性化）为前提**——向量 gather 需要连续可读的
> 偏移数组。需要支持 SVE2/SME ACLE 内建的编译器与相应 `-march`/`-msve-vector-bits`
> 编译选项。

### 7.1 先弄清楚：硬件到底给了什么（避免踩坑）

SVE 的向量长度"可扩展"指的是**由硬件实现决定**（128~2048-bit），软件只能查询、
不能指定；并且**非流式 SVE 宽度与 SME 流式模式宽度是两个独立的数**。动手前第一
件事是确认目标机器这两个宽度各是多少（Linux 可用 `PR_SVE_GET_VL` prctl /
PyTorch 启动日志 / `cpuinfo` 探测）。

各档硬件对照（**注意本文目标机是倒数第一行**）：

| 配置 | 非流式宽度 | 流式（SME）宽度 | 512-bit 怎么拿 |
|---|---|---|---|
| 仅 NEON（老款处理器） | 无 SVE | — | 拿不到；走标量优化（§6.1/§6.2） |
| SVE256（常见） | 256-bit | — | 拿不到；按 256-bit 用 |
| 典型 SME 服务器（如 Neoverse V2） | 128/256-bit | 512-bit | **必须 `SMSTART` 进流式模式** |
| **本文目标机：SVE512 + SME** | **512-bit** | ≥512-bit + ZA | **非流式直接可用，无需 SMSTART** |

针对本文目标机的关键结论：

- 非流式即 512-bit → 直接用 `-msve-vector-bits=512` 构建（或 VL-agnostic，§7.6），
  **没有进出流式模式的切换代价**——§7.2 的向量 gather 主路径就是它；
- SME 对本机的**增量**价值变成：① 流式模式（若流式 VL 更宽，可作为长循环的
  备选路径）② ZA 矩阵寄存器（研究性，§7.7）；
- 一般性提醒：**别的**机器（如 Neoverse V2）非流式可能只有 128/256-bit，同一份
  代码拿过去宽度会变——这正是 §7.6 推荐 VL-agnostic 写法（运行时查宽度）的原因。
  部署前先按本节表格对号入座。

### 7.2 B / L. 向量化 gather：512-bit 主路径与流式备选（SIMD 优化的核心）

**问题**：主循环是纯标量，每元素一套 load/检查/乘加，MLP 只能靠乱序执行硬撑。

**直觉**：手臂变宽，一次抓 8 个（512-bit；256-bit 机器则是 4 个）；一条 gather
指令同时发起多个独立 load，内存系统自然并行。

**做法**（承接 §6.1 的偏移数组）：

```c
// 主路径——本文目标机：非流式 SVE 即 512-bit。
// 构建加 -msve-vector-bits=512 即可，无需 SMSTART，没有进出流式模式的切换代价：
loop_over_chunk {
  svuint64_t off  = svld1(off_ptr + i);               // 一次 8 个 int64 偏移（64B）
  svfloat32_t v   = svld1_gather_u64base_f32(src, off); // 一条指令 8 个独立 load
  svst1(dst_ptr + i, v);                              // 连续写 64B（恰好一整缓存行，见 §7.5）
}

// 备选路径——若同一份代码部署到"非流式只有 128/256-bit"的机器，
// 用 SME 流式模式拿 512-bit：在并行任务入口进出一次，
// 几百周期的切换代价由长循环摊销（阈值判定见 §7.8）：
svstartsv();                                          // SMSTART：每线程一次，循环外
loop_over_chunk {
  svuint64_t off  = svld1(off_ptr + i);               // 一次 8 个偏移
  svfloat32_t v   = svld1_gather_u64base_f32(src, off);
  svst1(dst_ptr + i, v);
}
svstopsv();                                           // SMSTOP
// 不进流式的 256-bit 机器：同样的循环体，每条指令处理 4 个元素（宽度减半，写法不变）
```

**为什么有效**：指令数骤降；一条 gather 发起 N 个独立访存，MLP 显著提升；
`svst1` 一次写整条输出段。对多下标场景（例 3）收益叠加——原来每个下标一套标量
开销，现在偏移合成已在 §6.1 向量化完成。

**代价/风险**：
- **gather 吞吐不随 VL 线性增长**——LSU 端口才是上限，宽 VL 的主要收益是
  指令数和 MLP，需 microbenchmark 实测再定论；
- 备选路径（流式模式）的进出代价要求"长循环才划算"：在任务入口判断
  `chunk 元素数 > 阈值` 再 `svstartsv`（见 §7.8）；**本文目标机走主路径，无此开销**；
- 流式模式是**每 PE/每线程**的状态，线程池里各线程独立进入即可。

### 7.3 M. 非临时访存：别让"一次性数据"污染缓存（SVE2）

**问题**：self 的数据通常取一次就不再用，结果写一次就不再读——但硬件默认把它们
都拉进缓存，把真正该留下的（偏移数组、输出行）挤出去。

**直觉**：告诉搬运工"这批货出库后不再用，别占工作台"。

**做法**（SVE2 指令）：
- gather 侧：`svldnt1_gather_u64base_f32(...)`（`LDNT1G` 系）；
- store 侧：`svstnt1_f32(...)`（`STNT1` 系）非临时写，绕过写分配。

**为什么有效**：`index` 恰好是"复用度极低"的典型场景，非临时访存的缓存让渡
收益最大；与 §6.2 的预取组合（读侧预取、写侧非临时）。

**代价/风险**：数据若真会被复用（如 self 刚被写过还在缓存里）则是负优化；
非临时写需要后续显式同步语义配合（一般 `stnt1` 后接 `stnt1` 收尾即可）。

### 7.4 N. 128-bit 元素 gather：`complex128` / 16 字节数据（SVE2）

**问题**：complex128 等大元素按 64-bit gather 要拆两次。

**做法**：用 `svld1q_gather_u64base_*`（`LD1Q` 系）一次 gather 一个 128-bit lane；
VL=512 时一条指令取 4 个 16B 元素，配宽 store 直接搬 64B。

**为什么有效**：指令数减半、LSU 占用更低。

**代价/风险**：只影响 16B dtype，覆盖面窄，优先级低。

### 7.5 O. 整 cache line 写：VL=512 的天然红利

**问题**：部分行写（写不满 64B）会引发硬件 read-modify-write。

**直觉**：VL=512（64B）的 `svst1` 恰好一整行——写满即走，无回读。

**为什么有效**：连续输出场景下免除 RMW，配合写预取（`prfm` 写提示）效果更好；
128B 缓存线的 Neoverse 变体上需按实际线宽重新对齐（两条 64B store）。

**代价/风险**：几乎无；注意尾部不满一行的元素用普通 store 收尾。

### 7.6 P. VL-agnostic 构建：突破 `-msve-vector-bits=256` 的硬编码（工程必做）

**问题**：PyTorch 的 SVE256 变体把 256-bit **编译期写死**，同一份二进制既上不了
512-bit 机器，也没法为每种 VL 单独发版。

**做法**（三选一或组合）：
1. 新增 `SVE512` capability：`-msve-vector-bits=512`（**本文目标机正是"非流式即有
   512"的硬件，首选**）；
2. **VL-agnostic 循环**：不固定 vector bits，运行时 `svcntd()`/`svcntw()` 查询 VL、
   循环步长随 VL 自适应——同一二进制自适应 256/512/2048；
3. 流式路径：固定 `svstartsv()` 后，宽度即流式 VL（备选，见 §7.2）。

启动时用 `at::cpuinfo`/能力位探测（SVE/SVE2/SME、流式 VL）选择路径。

**为什么有效**：一份构建覆盖多代硬件；未来 VL 再翻倍也不用改代码。

**代价/风险**：VL-agnostic 代码不能假设向量元素个数（循环写法要小心）；
探测逻辑要进 dispatch 层。

### 7.7 Q. ZA tile：研究性方向（收益待验证，别押注）

SME 的 ZA 是 VL×VL 字节矩阵寄存器（512-bit 时 64×64B=4KB），以"整行 64B"为单位
读写（`ld1h_z_az`/`st1h` 系）。理论上可用于 §6.6 排序分块的 scatter-back/转置缓冲，
或 `index_put` accumulate 的局部累加器。**但 ATen 目前没有任何 ZA 用法**，
每指令整行搬运的模式对随机 gather 是否有净收益纯属未知——很可能是
"理论可行、工程收益存疑"。建议先 microbenchmark 再决定投入。

### 7.8 R. 流式模式的工程注意事项（上生产前必读；仅备选路径需要）

- 本节适用于"用流式模式拿 512-bit"的**备选路径**（本文目标机不需要，见最后一条）；
- `SMSTART`/`SMSTOP` 是每线程状态，几百周期的切换代价要求**长循环摊销**——
  在并行任务入口设阈值（chunk 元素数 > 阈值才进流式模式）；
- 流式模式可能改变功耗/降频特征，多核满载需实测（性能数字别只看单核）；
- 与 §6.5 结合：SME 机器上 `torch.set_num_threads`、`OMP_PROC_BIND` 同样影响
  NUMA 下的 gather 带宽；
- 本文目标机（非流式 512-bit + SME）：主路径直接用非流式 512-bit，**无需
  `SMSTART`**，上面第一条的阈值判定也不需要；流式模式仅在需要更宽 VL 或 ZA 时
  作为备选。若同一份代码要部署到"非流式只有 128/256-bit"的机器，再启用流式
  模式方案，并按本节阈值控制进出。

---

## 8. 怎么选：场景决策表与优先级

### 8.1 按场景选优化

| 你的场景 | 首选 | 次选 | 说明 |
|---|---|---|---|
| 任意场景的通用提速 | A 线性化 | C 预取 | 一切的地基，先做它 |
| 本文目标机（SVE512+SME）大 gather | A + §7.2 非流式 512-bit gather | M 非临时 | 目标机无需流式模式；跨机部署按 §7.1 判定 |
| `x[:, idx]` 型批量取样 | E 块拷贝特化 | A | 单维场景收益最大 |
| bool mask 索引 | 换整型下标 | `_unsafe_masked_index` | mask 本身就有额外全扫描 |
| 乱序大 gather（负采样等） | H 排序分块 | C 预取 | 先运行时判定下标乱不乱 |
| 不想改 ATen 源码 | I torch.compile | — | 立即可用 |
| 双路服务器带宽上不去 | F NUMA 亲和 | M 非临时 | 检查 `numactl` 与绑定 |
| 小 tensor 频繁调用 | G 小 case 短路 | — | dispatch 开销占比高 |
| spconv SubMConv3d（gather+bmm 分解） | index→index_select + rulebook 缓存 | gather-GEMM 融合 | 完整案例见 §9 |
| AI4S 点云 GNN（消息传递/kNN） | 拓扑缓存 + 不变量只取一次 | 聚合转 SpMM | 完整案例见 §10 |

### 8.2 优先级总表（含 SVE512/SME）

```
P0（先做、收益大、风险低）：
  A. 下标线性化 + 边界/负归一外提（无分支主循环）
  C. 软件预取（在 A 之上顺带实现）
  I. 配合 torch.compile 的 _unsafe_index 路径

P1（收益大、实现成本中等）：
  B. SVE256 gather 向量化（仅 SVE 主机）
  L. 512-bit 向量 gather（本文目标机非流式直用；流式模式仅为其他机器备选）
  P. VL-agnostic 构建与运行时能力探测
  E. 单维+前置维 → index_select 式块拷贝特化
  F. NUMA 亲和 / 线程绑定 / first-touch

P2（按需）：
  M. 非临时 gather/store（SVE2 主机）
  N/O. 128-bit 元素 gather、整行 store（按 dtype/场景）
  D. int32 下标
  H. 排序+分块（仅对乱序大 gather）
  G/J. dispatch 与微优化
  Q. ZA tile（研究性，收益待验证）
```

---

## 9. 实战案例：spconv `SubMConv3d` 中的 index 优化

**背景**：在 aarch64 CPU 上跑稀疏卷积网络（3D 检测/分割）时，profile 常出现 `bmm`
和 `index` 两个算子霸榜。本节把 §1–§8 的知识拧成一股绳，用这个真实负载回答：
`index` 在这里到底慢在哪、按什么顺序救。

> 声明：spconv 源码不在本仓库。本节按 `ConvAlgo.Native` 及各类纯 PyTorch 复现的
> **典型分解**来描述，不同版本细节略有出入，但骨架一致。

### 9.1 场景与分解：rulebook + gather + bmm + scatter

`SubMConv3d`（submanifold 稀疏卷积）的特点是**输出活跃位点 = 输入活跃位点**：
卷积核不扩张坐标，只在原有位点间传递信息。3×3×3 卷积核有 KV=27 个偏移，
每个偏移对应一本"花名册"（rulebook），记录该偏移下哪些输入/输出位点之间有连接。

典型计算分解：

```python
# features: [N, C_in]（N = 活跃体素数，每个位点一行）
# rulebook: 每个 kernel 偏移 kv 一桶，含 inp_idx [N_k]、out_idx [N_k]
out = torch.zeros(N, C_out)
for kv in range(27):                      # 3x3x3 → KV = 27
    A = features[inp_idx]                 # ← aten::index：按清单整行取货
    B = torch.bmm(A, W[kv])               #   [N_k, C_in] @ [C_in, C_out]
    out.index_add_(0, out_idx, B)         # ← scatter：搬回原位累加
```

（也有"大 bmm"变体：把 27 桶 pad 到统一长度 `[27, N_max]`，一次 gather、一次 bmm、
一次 scatter。两者的取舍见 9.4 L3。）

这个场景对 `index` 有四个**天生丽质**的结构特征——后面所有优化都从它们出发：

| # | 结构特征 | 内容 | 意味着 |
|---|---|---|---|
| 1 | 单维整行 gather | features `[N, C]` 只按 dim0 索引；每条规则搬 C 个连续 float（C=64 → 256B，恰好 4 条 cache line） | §6.3 E 的理想场景：按"箱"搬，不是按"件"搬 |
| 2 | 下标天然合法 | rulebook 由坐标匹配生成，`0 ≤ idx < N` 恒成立 | `Indexer::get` 每行一次的边界检查是纯浪费 |
| 3 | 下标可缓存复用 | rulebook 只依赖几何（坐标+核形状），与特征数据无关；同核连续 SubM 层之间、相邻推理步之间完全相同 | int32→int64 转换这类"每调用成本"可以摊薄到零 |
| 4 | 逐桶局部性好 | 桶内下标按坐标序生成，近似单调 | §6.2 C 的预取命中率天然高；代价是 27 桶合计把 features 扫 ~27 遍 |

外加第五个特征：**每桶的 `out_idx` 互不重复**（一个输出位点在每个偏移下至多被
一条规则命中）——scatter 侧可以免原子加法（9.4 L3）。

### 9.2 现状盘点：`features[inp_idx]` 在 ATen 里怎么跑

把 §4 的全链路套到具体形状上（features `[N, 64]` float32，`inp_idx` `[27, N_k]` int32）：

1. **入口**（§4.2–4.3）：`features[inp_idx]` → ATen `index`；
2. **归一化**（§4.4 `make_info`）：第 ⑤ 步把整个 rulebook **int32 → int64 全量转换**——
   每次调用都转一遍（`TensorAdvancedIndexingUtils.h:96-100`，无条件执行），
   每元素读 4B 写 8B；
3. **restride**（§4.5）：下标视作 `[27, N_k, 1]`，self 广播成 `[27, N_k, 64]`、
   strides `[0, 0, 4]`（0-stride 视图，不搬数据）；
4. **迭代器**（§4.6）：reorder 把 C 维放最内层 → 内层的下标 stride 为 0 →
   `is_constant_index` 命中（`IndexKernelUtils.h:7`）；
5. **kernel**（§4.7）：每行 = 读 1 个 int64 下标 + 1 次边界检查 + 64 个连续 float 拷贝。

对照 9.1 的结构特征，三笔**"白付"的开销**：

| 白付的开销 | 量级 | 本可免费的原因 |
|---|---|---|
| 下标 int64 全量转换（每次调用） | R×12B 读写（R=规则总数；R≈2.7M 时约 32MB，比 features 本体 25.6MB 还大） | 特征 3：rulebook 可缓存，一次都不用转 |
| 每行一次边界检查 + 负数归一 | 2 次比较 + 2 个分支/条规则（§5.1 定价） | 特征 2：下标天然合法 |
| padding 哨兵的无效 gather | 见下 | — |

**padding 哨兵陷阱**（大 bmm 变体）：各桶 pad 到统一长度时的哨兵值若为 -1，
`Indexer::get` 会把它当作**合法负下标**（`value >= -size` 检查通过，
`IndexKernelUtils.h:41-44`），归一化成 `N-1` 后**静默 gather 最后一行**——
不报错、结果被后续 mask 掩盖，白白多搬整行还查不出来。哨兵若是 0 同理
（gather 第 0 行）。这个陷阱决定了 9.4 L2 的"去掉边界检查"必须先消灭 padding。

### 9.3 一笔内存账：为什么 profile 里 index + bmm 霸榜

设 N=100k 活跃体素、C_in=C_out=64、KV=27、float32。SubM 每个位点平均发出
≈KV 条规则（按邻居占据率打折，这里保守按 KV×N 上界计）：

| 张量 | 形状 | 大小 |
|---|---|---|
| features | N×C_in | 25.6 MB |
| **A（gather 中间量）** | KV×N×C_in | **≈691 MB** |
| **B（bmm 输出）** | KV×N×C_out | **≈691 MB** |

大 bmm 变体下 A、B 各要**写一遍 + 读一遍**：约 **2.8 GB/层**的纯中转流量。
按双路 aarch64 典型有效带宽 150–250 GB/s 估算，仅搬运就 **11–19 ms/层**；
10 层 backbone 就是 0.1–0.2 秒。这就是 `index`+`bmm` 霸榜的定量解释——
瓶颈不是循环写得差，而是**中间量太胖**。

**结论先行**：§6/§7 的循环级优化（预取、SVE512 行拷贝）能把 gather 本身提速 ~2×，
但 691 MB 的中间量一 byte 都不会少。**结构性赢法是让 A、B 根本不落地**（9.4 L3）。

### 9.4 优化清单（按投入产出分层）

**L1 — 一行改动（当天可做）**

- **`index` → `index_select`**：`features.index_select(0, inp_idx.reshape(-1)).view(27, N_k, C)`。
  `index_select` 用 `AT_DISPATCH_INDEX_TYPES` 按下标 dtype 分发
  （`TensorAdvancedIndexing.cpp:1689`），**int32 原生支持**，省掉 make_info ⑤
  的全量转换；链路也更短（无 restride/broadcast 机关），且行拷贝经
  `memcpy`/向量化的 `copy_stub` 实际吃到 NEON/SVE（§6.3）——搬运本身也快。
  两个注意点：(a) dim≠1 走通用路径——逐元素检查 + 行拷贝
  （`TensorAdvancedIndexing.cpp:1668-1708`），没有 dim1 那条
  "预检查 + 块 memcpy"快车道（`TensorAdvancedIndexing.cpp:1538`），
  那正是 L2 要补的；(b) 它的检查是 `>= 0`（`TensorAdvancedIndexing.cpp:1704-1706`），
  **含 -1 哨兵的下标会直接报错**——这其实把 9.2 的陷阱显式化了，是好事，
  但意味着大 bmm 变体要先清哨兵；
- **查隐藏转换**：torch profiler 算子表里搜 `aten::to` / `to_copy`——
  下标转换常在这里现形；
- **rulebook 缓存**：几何不变则 rulebook 不变（特征 3）。同核连续 SubM 层共享一份；
  推理场景/固定点云整个网络只算一次。（训练时每步几何变化，缓存无效。）

**L2 — kernel 级（映射 §6/§7，1–2 周量级）**

- **整行块拷贝特化**（§6.3 E 落地）：每条规则一次 256B `memcpy`；C=64 恰好
  4×64B cache line，目标机非流式 SVE512 下就是 4 组 `ld1/st1`（§7.2）；
- **软件预取**（§6.2 C）：桶内下标近似单调（特征 4），提前几行 `prfm` 拉取，
  命中率天然高；
- **去掉边界检查**：下标天然合法（特征 2）→ `_unsafe_index`
  （`TensorAdvancedIndexing.cpp:736`）或自定义 kernel。**前提**：先消灭
  padding 哨兵（9.2 陷阱），否则 -1 变成巨大偏移直接越界崩掉；
- **NUMA / first-touch**（§6.5 F）：gather 是纯带宽负载，features 按线程亲和分布，
  别让单 socket 带宽顶天花板。

**L3 — 算法级（收益上限最大，1–2 月量级）**

- **消灭 padding**：各桶压缩掉哨兵位再 gather/bmm，中间量从 `27×N_max×C`
  缩到 `ΣN_k×C`，哨兵陷阱一并消失；
- **逐桶 vs 大 bmm 的权衡**：

| | 逐桶循环 | 大 bmm（pad + stack） |
|---|---|---|
| 中间量驻留 | 单桶 N_k×C（几十 MB 量级） | ≈691 MB（含 padding） |
| GEMM 效率 | 深层桶小 → BLAS 吃不满 | 大矩阵 → BLAS 峰值 |
| scatter | 27 次 `index_add_`；每桶 out_idx 唯一（特征 5）→ 可免原子 | 1 次；但 out_idx 跨桶重复 → 需原子 |
| padding | 可完全避免 | 必须 pad 到 N_max |

  真实答案常是"逐桶 + 桶内分块"或下面的融合——上表是决策起点，不是结论；

- **gather-GEMM 融合（implicit GEMM 的 CPU 版）**：直接计算
  `out[o, :] += features[inp_idx[r], :] @ W[kv]`，A、B 都不落地，
  2.8 GB/层的中转流量直接归零。这是**总收益上限最大**的一档，代价是写分块
  kernel（行分组 + 小 GEMM 微核——§7.2 的 SVE512 微核正好用上）。GPU 上
  spconv 的 ImplicitGemm 干的就是这件事，CPU 目前没有现成等价物；
- **scatter 侧**：每桶 out_idx 唯一且按坐标序 → `index_add_` 可特化为无原子的
  顺序累加；下标同样建议 int32（§6.4）。

**L4 — torch.compile（§6.7）**

Inductor 能把 gather 与相邻逐元素算子融合，但**不会融合进 `bmm`**——
对"index → bmm → index_add_"这个骨架，L4 只能修剪边角，动不了主干。

### 9.5 小结

- profile 里 **index+bmm 霸榜 = gather 中间量落地的典型症状**；药方主要在
  算法层（L3），不在循环层；
- 通用 `index` 的三大固定成本（下标转换、逐行边界检查、逐元素偏移计算）在这个
  负载里**全部可免**——因为四个结构特征（整行 / 合法 / 可缓存 / 单调）把它们逐一
  抵消了。把 §8 的框架套到具体负载时，**先找结构特征，再对号入座**；
- 优先级建议：**L1 立刻做 → L3 决定上限 → L2 填细节**；L4 在此骨架上收益有限；
- （GPU 用户一句话：直接选 spconv 的 ImplicitGemm 模式，本节问题在 GPU 上
  已有现成解。）

---

## 10. 实战案例：AI4S 点云模型中的 index 优化

**背景**：AI4S（AI for Science）点云模型——分子/材料的势能与力场 GNN
（SchNet、NequIP、MACE 一族）、气象网格预报（GraphCast 类 mesh GNN）、
流体粒子模拟（GNS/MeshGraphNets 类）、天文点云——profile 里 `index`
同样霸榜。这类模型与 §9 的稀疏卷积**同源但更杂**：本质都是
"邻居图 + gather/scatter"，但拓扑来源、payload 大小、生命周期各不相同，
**药方要先分档再开**。

> 声明：本节按这类模型的典型实现描述（PyG 风格的消息传递骨架），
> 具体模型细节有出入，但索引模式一致。

### 10.1 共性：邻居图 + gather/scatter

不管哪个领域，计算骨架几乎都长这样：

```python
# 节点特征 h: [N, C]，坐标 pos: [N, 3]
# 邻居图 edge_index: [2, E]（src/dst），E ≈ N×K，K = 邻居数（常见 16~128）
msg = f(h[src], h[dst], pos[src] - pos[dst])    # ← gather×3 + 逐元素
out = out.index_add_(0, dst, msg)               # ← scatter 回节点
```

与 §9 对照，结构特征大部分继承，但有两点关键差异：

| | spconv SubMConv3d（§9） | AI4S 点云 GNN |
|---|---|---|
| 拓扑来源 | rulebook（坐标匹配） | kNN / 半径搜索 / 网格连接 |
| 拓扑生命周期 | 同核层间、相邻步间可复用 | **更强**：晶体/网格完全静态；MD 邻居表可隔 n 步重建（Verlet 式） |
| gather payload | 固定 C=64+ 的特征行 | **两档分化**：pos 只有 12B/边，特征 64–256B/边 |
| gather 之后 | bmm（GEMM） | 逐元素 MLP 或小 GEMM |

### 10.2 六类典型场景与优化方向

| # | 场景 | 典型代码 | 优化方向 |
|---|---|---|---|
| ① | 边消息传递 gather | `h[src]`、`pos[src]-pos[dst]` | 最高频，专门拆解 → 10.3 |
| ② | 邻居搜索产物 | knn/radius/topk 的下标（通常 int64） | 转一次 int32（§6.4）；邻居表按 Verlet 间隔复用；搜索本身（常是 cdist+topk）单独 profile |
| ③ | 网格↔粒子插值 | 手算 flat 下标 + 权重（PIC/气象） | 网格固定 → 预构建稀疏矩阵转 SpMM；scatter 侧原子加是真正痛点 → 粒子排序（§6.6） |
| ④ | bool mask 选点 | `x[mask]`（变长 batch） | 换整型下标（§8.1）；`nonzero` 移出热路径 |
| ⑤ | scatter 回写 | `index_add_` / `scatter_reduce`，与 ① 成对出现 | 按 dst 排序一次（可缓存）→ `torch.segment_reduce` 免原子（现状：重复下标靠原子兜底 `IndexKernel.cpp:180`，确定性模式退串行 行 192） |
| ⑥ | 元素/类型查表 | `emb_table[species]` | 换 `F.embedding`：内部就是 `weight.index_select(0, indices)`（`Embedding.cpp:45`），且原生接受 int32（`Embedding.cpp:41`）——不收 §4.4 ⑤ 的转换费 |

> 动手前先切一刀：profile 里 gather（`aten::index`）与
> scatter（`aten::index_put_`/`index_add_`）各占多少——两者病因不同，药方也不同。

### 10.3 重点拆解：边消息传递（①）

**1. 拓扑半静态是 AI4S 最大红利。** 晶体/网格的 edge_index 完全不变；
MD 的邻居表可隔 n 步重建（Verlet 列表思想）。由此派生一串
"一次计算、处处复用"的缓存物：edge_index 本体（转 int32 后）、
按 dst 排序的排列、CSR/稀疏矩阵结构、下标线性化后的偏移表（§6.1）。
这是 §9 结构特征 3 的强化版——spconv 只能"同核层间"复用，
AI4S 常常"整个数据集"复用。

**2. sum 聚合层转 SpMM。** 消息 M `[E, C]` 算完后的
`out.index_add_(0, dst, M)` 数学上就是一个 0/1 稀疏矩阵乘：
`out = S @ M`，`S[dst[e], e] = 1`。拓扑缓存成 CSR 后一条 `torch.sparse.mm`
搞定，冲突/原子彻底消失。适用边界：只有"线性消息 + sum 聚合"的层能整体转
SpMM；非线性消息（先 MLP 再聚合）只能转聚合那半步，gather 那半步退而求
gather→GEMM 融合（§9 L3 同款）。

**3. 不变量只 gather 一次。** pos、边长、边特征在层间不变（推理时 pos 甚至
全程不变）——审计每层是否在重复 gather 相同的量。零风险纯赚。

**4. 小 payload 是另一档战场。** `pos[src]` 每边只搬 12B，而 int64 下标读 8B +
边界检查与 payload 同量级（§5.2 结论 1 的临界区）。这档的主菜是
**省管理开销**：下标线性化预计算偏移（§6.1）+ int32（§6.4）+ 与后续算术
融合；SIMD 行拷贝无用武之地。C≥64 的特征 gather 才轮到 §6.3 行块 +
§7.2 SVE512。

### 10.4 横向决策轴：payload 大小

```
每条边的 payload（字节）
 ├─ 小档 ≤16B（pos 12B、边长 4B）
 │    瓶颈：管理开销（下标读 8B + 检查 + 偏移计算 ≈ payload 本身）
 │    主菜：线性化(§6.1) + int32(§6.4) + 融合；SIMD 无用武之地
 └─ 大档 ≥64B（C=64+ 特征行）
      瓶颈：带宽 + 中间量落地
      主菜：行块(§6.3) + SVE512(§7.2) + SpMM/融合(§9 L3)
```

（坐标 gather 常年混进 profile 前列，却容易被当成"特征搬运"误诊——
先分档，再开方子。）

### 10.5 小结

- 与 §9 同病同方：**E×C 的 gather 中间量落地是结构性病根**，药方是
  融合/SpMM/复用，不是调循环；
- AI4S 的独有红利是**拓扑半静态**（晶体/网格全静态、MD 可 Verlet 复用）——
  凡是从拓扑派生的量，都值得问一句"能不能缓存"；
- 动手顺序：先切 gather/scatter（10.2 注），再按 payload 分档（10.4）→
  审计不变量重复 gather（纯赚）→ ⑥/②/④ 零成本替换 → SpMM/排序缓存 →
  gather-GEMM 融合。

---

## 附录 A：相关文件与关键行号索引

| 文件 | 关键位置 |
|---|---|
| `torch/csrc/autograd/python_variable_indexing.cpp` | `THPVariable_getitem` 行 390；`applySlicing` 行 213 |
| `aten/src/ATen/TensorIndexing.cpp` | `Tensor::index` 行 67 |
| `aten/src/ATen/TensorIndexing.h` | `get_item` 行 636；`applySlicing` 行 539；`handleDimInMultiDimIndexing` 行 438；`dispatch_index` 行 580；`recordTensorIndex` 行 314 |
| `aten/src/ATen/native/TensorAdvancedIndexing.cpp` | `index` meta 行 507；`build_index_op` 行 468；`index_out` 行 715；`AdvancedIndex` 构造 行 628；`restride_src` 行 595；`reshape_indexer` 行 616；`_index_put_impl_` 行 953；`index_select_out_cpu_dim1_` 行 1538；`index_select` 通用路径 行 1668-1708（int32 原生分发 行 1689，逐元素检查 行 1704-1706）；`quantized_index` 行 720；`_unsafe_index` 行 736 |
| `aten/src/ATen/native/TensorAdvancedIndexingUtils.h` | `make_info` 行 72；`canDispatchToMaskedFill` 行 26 |
| `aten/src/ATen/native/IndexingUtils.h` | `expandTensors` 行 23；`transposeToFront` 行 126；`AdvancedIndex` 结构 行 172 |
| `aten/src/ATen/native/cpu/IndexKernel.cpp` | `index_kernel` 行 24；`index_put_kernel` 行 165；`cpu_take_put_kernel` 行 63；`IndexToOffset` 行 45；`REGISTER_DISPATCH` 行 806 |
| `aten/src/ATen/native/cpu/IndexKernelUtils.h` | `is_constant_index` 行 7；`Indexer` 行 18；`cpu_index_kernel` 行 52 |
| `aten/src/ATen/native/IndexKernel.h` | dispatch stub 声明 行 18-39 |
| `aten/src/ATen/native/Embedding.cpp` | `embedding_symint` 行 37；内部 `index_select` 行 45/53；原生接受 int32 下标 行 41 |
| `aten/src/ATen/native/SegmentReduce.cpp` | `_segment_reduce_lengths/offsets_stub` 分派 行 23-26 |
| `aten/src/ATen/TensorIterator.cpp` | `for_each` 行 763；`serial_for_each` 行 784；`reorder_dimensions` 行 232；`coalesce_dimensions` 行 638；`DimCounter` 行 1712 |
| `aten/src/ATen/TensorIteratorInternal.h` | `serial_for_each` 行 39；`get_data_ptrs` 行 23 |
| `aten/src/ATen/TensorIterator.h` | `GRAIN_SIZE` 行 78；`for_each` 行 435/439 |
| `aten/src/ATen/Parallel-inl.h` | `parallel_for` 行 10 |
| `aten/src/ATen/ParallelNative.cpp` | `calc_num_tasks_and_chunk_size` 行 131；`invoke_parallel` 行 144 |
| `c10/util/Exception.h` | `TORCH_CHECK` 行 551；`TORCH_CHECK_INDEX` 行 635 |
| `aten/src/ATen/native/native_functions.yaml` | `index.Tensor` 行 3093；`index.Tensor_out` 行 3105 |
| `aten/src/ATen/cpu/vec/vec128/vec128.h` | aarch64 NEON 选择 行 6 |
| `aten/src/ATen/cpu/vec/sve/` | SVE256 向量（`vec_common_sve.h` 等） |
| `cmake/Modules/FindARM.cmake` | SVE256 探测 |
| `cmake/Codegen.cmake` | CPU_CAPABILITY 编译 行 363-449（SVE256 行 425-429） |
| `CMakeLists.txt` | `CPU_AARCH64` 行 189-190；aarch64 帧指针 行 1196-1201；大页 行 699-702 |

---

## 附录 B：术语表

| 术语 | 解释 |
|---|---|
| 高级索引（advanced indexing） | 用 Tensor/mask 作下标的索引，产生数据搬运；区别于产生视图的基本索引 |
| gather / scatter | 按清单收集 / 按清单分发；`index` 是 gather，`index_put` 是 scatter |
| restride | 修改 tensor 的形状+步长得到新视图（`as_strided`），不动数据 |
| `indexed_sizes` / `indexed_strides` | 被索引维的原始尺寸（边界检查用）/ 字节步长（偏移计算用） |
| TensorIterator | PyTorch CPU 的"多维同步遍历"通用框架，负责维序重排、合并与并行切分 |
| dispatch stub | ATen 的算子分派机制（`index_stub`），按设备/能力选择具体 kernel |
| structured 算子 / precomputed | 算子声明模式：meta 阶段算好的派生参数直接传给实现 |
| grain size | 并行切分的最小块；`index` 用 3000（默认 32768） |
| NEON | Arm 传统 SIMD，固定 128-bit，无 gather 指令 |
| SVE / SVE2 | 可伸缩向量扩展 / 其增强版；SVE 提供 gather，SVE2 增加非临时访存、`LD1Q` 等 |
| VL（vector length） | SVE 向量寄存器位宽，硬件实现决定（128~2048-bit） |
| `-msve-vector-bits=N` | 编译选项：把 VL 编译期固定为 N；PyTorch 目前固定 256 |
| SME / 流式模式（SSVE） | Armv9 矩阵扩展；流式模式宽度与非流式宽度是两个独立参数（切换需 `SMSTART`/`SMSTOP`）。本文目标机非流式即 512-bit |
| ZA | SME 的矩阵累加寄存器，VL×VL 字节，按整行访问 |
| `svld1_gather_u64base_*` | SVE gather 内建：按向量里的字节偏移一次取多个元素 |
| `LDNT1G` / `STNT1` | SVE2 非临时 gather / 非临时 store |
| MLP | 内存级并行：一个核同时在途的访存请求数 |
| LSU | load/store 单元 |
| cache line | 缓存行（通常 64B），内存搬运最小单位 |
| NUMA / first-touch | 非统一内存访问 / 内存页归属"第一次触碰"的节点 |
| 非临时访存（non-temporal） | 提示硬件"数据用完即弃"，不进缓存 |
| 软件预取（prefetch / `prfm`） | 提前发起取数，隐藏 DRAM 延迟 |
| 边界检查外提（hoisting） | 把逐元素检查改为循环前一次性检查 |
