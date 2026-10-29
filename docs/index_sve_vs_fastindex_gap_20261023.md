# fastindex 能力差异清单与 index_sve 当前仓库审计

> **日期**：2026-10-23
> **快照注意**：本文审计基于 `a91d566`（T13 时点），文中"#2/#8 已落地"等
> 表述是当时状态；#1 strided-rows 等 T12 项亦已在其后落地。**当前进度以
> [status.md](status.md) §2 为准**，本文的价值在 §19.1（纯转置 run=1 慢因）
> 等仍被引用的深度分析与 fastindex 能力对照。
> **当前仓库**：`index_sve` @ `a91d566`
> **外部输入**：另一 agent 提供的《fastindex 能力差异清单》（Part I，原文保留）
> **fastindex 源码**：清单中的
> `/home/z00889957/100Work/1012_index_op_opt/fastindex/fastindex/` 在本工作区
> 不存在，因此本次审计以清单描述 + 当前 `index_sve` 源码/实测为准，没有直接读
> fastindex 实现。
> **结论先行**：当前仓库不是清单所说的“融合前基线”。M1/T3/T4/T5/T8/T11 已经
> 落地；本轮补齐了 **#1 strided-rows**、**#3 grain 下限**、**#8 小行宽/多线程
> skip-linearize**，并把 **#2 多索引 A>=4** 限定在 ≥16t 使用。**#4 `_offset`
> 微基准无收益、#5 mask int64 是 `nonzero` 语义、#6 在本架构不适用**，均已关闭；
> 纯转置（run=1）无连续后缀，块拷贝无收益，保留原路径并记录原因。
> 后续 T13 又补了“多索引小 K”入口快路径（§21），K=512/A=1 从 0.74x 提升到
> 1.6-2.0x。
> **T15 更新（2026-10-26）**：raw 条件放宽为
> `A>=4096||A<16||(A<=64&&B*K<=65536)`；新增 f32/i32+i64 大源直连 SVE
> `pure_gather_sve_4b_i64idx`（源>512KB 门）；小 K 并行阈值线程缩放
> `max(16384, 512×nt)`。正文标注“当前”的路由描述如有出入，以
> `architecture.md` §1 与 `optimizations.md` §19 为准；`copy_contig_sve`
> 的 memcpy 阈值已放宽到 A>=2（T14）。

---

# Part I：外部差异清单（原文）

> 以下为外部输入的完整原文，仅去掉开头终端控制字符。文中的 **“index_sve” 指
> 尚未融合 fastindex 能力的旧基线**，不代表本仓库 @ `a91d566` 的状态；状态与
> 路径请以 Part II 为准。

## fastindex 能力差异清单（给 index_sve 优化 agent 的补齐指引）

> **日期**：2026-10-23 ｜ **基线说明（重要）**：本文中的 **"index_sve" 指
> 尚未把 fastindex 能力融合进来的版本**（即生成 `index_sve-main_1021` 优化轮的
> agent 所面对的基线，也即 `index_sve-main/` 那代代码）。2026-10-22 已完成的
> 「融合」（memcpy 精确拷贝 + skip-linearize + B==1 快路径）是在本文列出的差异
> 点之上做的第一轮补齐，**每一条差异点都标注了「已融合关闭 / 仍开放」**，以便
> 后续 agent 知道哪些已做、哪些还值得做。
> **目标读者**：只做 index_sve 优化、不接触 fastindex 代码的 agent。读完本文后，
> 应能按差异点逐条把 fastindex 的差异化能力补进 index_sve，以获得更好的性能。
> **配套文档**：`docs/index_sve_vs_fastindex_structural_20261022.md`（结构差异
> 通俗版，含融合记录 §6.1）；`docs/CHANGES_20261022.md`（已落地变更总览）。
> fastindex 源码参考：`/home/z00889957/100Work/1012_index_op_opt/fastindex/fastindex/`
> （`__init__.py`、`csrc/index_frontend.cpp`、`csrc/index_optimized.cpp`、
> `csrc/fastindex.cpp`、`csrc/index_info.h`）。

### 0. 术语表（与 index_sve 文档同一套）

通用术语与场景/负载代号（S1/P1/M1…）见 [`glossary.md`](glossary.md)；
下表只列本对比特有的读法约定：

| 术语 | 含义 |
|---|---|
| **ratio** | 原生耗时 / 实现耗时，>1 = 实现快，<1 = 退化 |
| **B / K / A** | 前导批维 / 索引个数 / 尾随连续元素数 |
| **linearize** | 物化 idx 数组（写一遍 + 读一遍，两遍内存流量） |
| **after_contiguous** | 尾维是否连续（行拷贝前提；否则要按步长取） |
| **suffix_packed** | fastindex 判定尾维是否可整行 memcpy 的函数 |
| **strided-rows** | fastindex 对「尾维非连续」形态的分块 memcpy 内核 |
| **make_index_info** | fastindex 前端规范化（复刻 ATen make_info，~15μs 固定开销） |
| **GRAIN_MIN** | fastindex 并行分块粒度下限（=64），防任务调度风暴 |

### 1. 差异点总览（按收益确定性排序）

| # | 差异点 | fastindex 能力 | 融合前 index_sve | 状态（2026-10-23） | 预期收益 |
|---|---|---|---|---|---|
| **1** | **strided-rows 非连续尾维内核** | 尾维非连续时按「最长连续后缀块」分块 memcpy，每列算一次偏移 | 无此内核，非连续形态落逐元素标量拷贝（`d[c]=s[col_off[c]]`） | **仍开放** | 转置/非连续形态 1.3-2x |
| **2** | **offset 融合单遍** | 偏移计算融合进 gather 循环，一次遍历，不物化 idx | 主路径 `linearize+gather_index` 两遍；仅 M1 raw/T3 fused/T4 bucket 已跳过 | **部分已做（M1/T3/T4）；多索引 + A==1 大 K 4B 仍 linearize** | 按形态 1.1-1.5x |
| **3** | **grain 下限 64** | `GRAIN_MIN=64` 防高线程任务风暴 | `adaptive_grain` 下限=1，小任务高线程 fork-join 固定开销大 | **仍开放** | 小任务高线程固定开销下降 |
| **4** | **SVE gather `_offset` 变体** | 用 `svld1_gather_u32offset_f32`（字节偏移语义） | 用 `svld1_gather_u32index_f32`（index 语义，按 esize 缩放） | **未评估（需微基准）** | 未知，可能省一层缩放 |
| **5** | **int32 下标保持** | 前端统一转 int64（**劣势**） | normalize 保留 int32，linearize/桶化直读 i32（**已优于 fastindex**） | **index_sve 已赢，无需改**（确认 mask+i32 组合也保持） | 防御性 |
| **6** | **slice 提前回退** | monkey-patch 在入口检测 slice/None/Ellipsis 天然回退原生 | dispatcher 全局覆盖，P0 fallback 只在 `K<1024` 判据处回退 | **P0 已做（延迟到 K<1024）；未提前到 normalize 前** | 混合 slice 形态省白做 normalize |
| **7** | **语义对齐（mask/广播/负索引/越界）** | 复刻 ATen make_info | 复刻 ATen make_info，且 int32 保持更优 | **已对齐，无需改** | — |
| **8** | **极小行宽精确拷贝 + skip-linearize** | `std::memcpy` 精确拷贝 + 单索引 offset1 无除法 | SVE 满 VL 加载浪费 + linearize 两遍 | **已融合关闭**（`copy_contig_sve` memcpy + `rowcopy_raw A<16` + B==1 快路径，C=4 0.86x→1.84x） | — |

### 2. 差异点 1（最高价值）：strided-rows 非连续尾维内核

#### fastindex 怎么做

`run_strided_rows`（`index_optimized.cpp:176-281`），触发条件 `dims_before == 0`
（无前导批维）且尾维非连续（`suffix_packed == false`）：

1. 尾维非连续时整行 memcpy 不可行，ATen 会退化成逐元素读 idx + 越界检查 + 标量拷贝。
2. fastindex 的做法：`dims_before == 0` 时广播索引的列偏移**每列恒定**，只算一次；
3. 把尾维按行优先拆成「**最长的连续后缀块** + 若干外层跨步维」，每行用**少量大块
   memcpy**：例如尾维 `[200,10]` 元素步长 `[3000,1]` → 200 次 40B memcpy，而不是
   2000 次 4B memcpy。

#### 融合前 index_sve 缺什么

`after_contiguous == false` 时全部落 `gather_fused`（op.cpp:118）或
`dispatch_scalar`（gather.cpp:273），是**逐元素标量拷贝**
（`d[c] = s[col_off[c]]`），没有 strided-rows 的分块 memcpy 优化。测试里
「非连续/转置」用例是**正确性通过但性能未优化**的形态。

#### 补齐动作

在 `gather_index` / `gather_fused` 里，对 `after_contiguous == false && B == 1`
（dims_before==0）的形态，移植 fastindex 的 `run_strided_rows` 算法：

- 先算每列恒定偏移（循环外）；
- 按「最长连续后缀块」做分块 memcpy（尾维非连续部分的步长走增量 walk）。

#### 验证

构造 `x.t()` 转置 + 单索引（如 `x=(64,20000)` 转置后 `x.t()[idx]`），对拍正确性
+ 隔离测 ratio（pin node5 + min-of-30）。

### 3. 差异点 2（高价值）：offset 融合单遍 vs linearize 两遍

#### fastindex 怎么做

`run_block` / `run_pure_gather` / `run_strided_rows` 全部把偏移计算**融合进并行
循环**（`IndexView::offsetAt` 循环内直读 idx + 越界检查 + 负索引归一 + 乘字节步长），
**一次遍历**，不物化 idx 数组，对齐 ATen `cpu_index_kernel`。

#### 融合前 index_sve 缺什么

主路径（`linearize + gather_index`，op.cpp:114）**两遍**：linearize 物化 idx
（写一遍）+ gather 读 idx（再读一遍）。融合后 M1 raw（单索引大块）、T3 fused
（A<4/非连续）、T4 bucket（小值域 1D）已跳过 linearize，但**多索引 / mask 展开后 /
A==1 大 K 的 f32/i32 4B 路径仍在 linearize**（op.cpp:112-116）。

#### 补齐动作

把「多索引 / mask 展开后 / A==1 大 K」这些仍走 linearize 的形态，评估改造成
`gather_fused` 风格的单遍（融合解码 + 拷贝）。**注意**：gather_fused 对 A==1 多索引
实测「解码开销更大更差」（op.cpp 注释），所以**必须按形态实测**，只对确认
linearize 净亏的形态切换，不能一刀切。

#### 验证

`bench_case_modes.py` 的 m6_2d_idx（多索引 2048×3）、mask dense 等 case 隔离对拍。

### 4. 差异点 3（中价值）：grain 下限对齐

#### fastindex 怎么做

`grain_for` 有 `GRAIN_MIN = 64`（`index_optimized.cpp:98-106`），注释明确写着
防「高线程数下切出单元素 chunk → 任务调度风暴」，并点名 index_sve 的 1.86x→0.51x
回归。

#### 融合前 index_sve 缺什么

`adaptive_grain`（grain.h:8）下限是 `1`（`std::max(1, ...)`），无 64 下限。
小任务高线程（如 m3 桶化 T≥16）时 grain 极小 → fork-join 固定开销放大。

#### 补齐动作

把 `adaptive_grain` 下限从 1 提到 64（或对桶化/小任务路径单独加下限），回归大 K
形态确认不劣化。

#### 验证

`/tmp/m3_threads_ab.py` 高线程（T=16/32）复测 + 26-case merge 确认大 K 无回归。

### 5. 差异点 4（低价值）：SVE gather `_offset` vs `_index` 变体

#### fastindex 怎么做

`svld1_gather_u32offset_f32` / `svld1_gather_u64offset_f64`（**offset 变体**，偏移即
元素字节偏移），注释明确实测 `_base` 绝对地址变体会 trap、`*_offset` 可用。

#### 融合前 index_sve 缺什么

用 `svld1_gather_u32index_f32` / `svld1_gather_u64index_f64`（**index 变体**，按
esize 缩放）。两者都是合法 SVE gather，fastindex 的 offset 语义可省一层缩放，
**性能差异未知**。

#### 补齐动作

在 `/tmp/gather_micro.c` 指令级微基准上 A/B 对比 `_offset` vs `_index` 变体的指令
吞吐，选快的。

#### 验证

`/tmp/gather_micro.c`（bisheng clang 编译，taskset -c 190 跑）。

### 6. 差异点 5（低价值，防御性）：int32 下标保持

#### fastindex 怎么做

前端统一 `int32 -> int64`（`index_frontend.cpp:108-110` `indice.to(kLong)`），内核
只读 int64——这是它的**劣势**（m3 i32 时付转换费）。

#### 融合前 index_sve 缺什么

**不缺，反而已赢**：normalize 保留 int32 下标，linearize 直接写 int32，桶化直接读
i32——这是 index_sve 已优于 fastindex 的点（m3 i32 1.9-2.1x）。

#### 补齐动作（防御性）

fastindex 的「缺点」恰好证明 index_sve 的 int32 保持策略正确，**不需要改**。但要
**确认**：`x[mask]` 且 idx 为 i32 的组合，normalize 在 nonzero 展开后是否仍保持
i32（如果会转 i64 就补上保持逻辑）。

#### 验证

`x[mask]` 且 idx 为 i32 的形态对拍。

### 7. 差异点 6（低价值）：slice 提前回退

#### fastindex 怎么做

monkey-patch 在 Python 层**入口检测** slice/None/Ellipsis 并回退原生
（`__init__.py:69-80`），slice 形态永远 1.00x 不劣化。

#### 融合前 index_sve 缺什么

dispatcher 全局覆盖，slice 形态也走自己内核——曾 0.21-0.29x 严重劣化，**已由 P0
fallback（`B>1 && K<1024` 回退原生，op.cpp:44）修复**，但判据在 normalize 之后才
评估，`x[:, 0:10]`、`x[0:10, idx]` 等混合 slice 形态仍会先做 normalize。

#### 补齐动作

把 slice 检测**提前到 normalize 之前**（在 index_impl 入口对原始 indices 检测
slice），命中即回退原生，避免 normalize/linearize 白做。

#### 验证

`x[:, 0:10]`、`x[slice, idx]` 等混合 slice 形态对拍。

### 8. 差异点 7（已对齐，无需改）：语义层

fastindex 的 `make_index_info` 与 index_sve 的 `normalize_index` 都复刻 ATen
`make_info` 同一套逻辑（checkIndexTensorTypes → expandTensors → expand_outplace
→ transposeToFront → restride/reshape）。mask、广播、负索引、越界检查语义已对齐，
**不需要动**；index_sve 的 int32 保持还优于 fastindex。

### 9. 已融合关闭的点（2026-10-22 融合已完成，新 agent 不要重复做）

**差异点 8：极小行宽精确拷贝 + skip-linearize**（原 index_sve 0.86x 弱项 → 融合后
1.84x）：

1. `gather.cpp copy_contig_sve`：`n < VL`（f32<16 / f64<8，行 <64B）改
   `std::memcpy` 精确拷贝，消除 SVE 满 VL 加载浪费；
2. `op.cpp` rowcopy_raw 分支：`A >= 4096` 放宽为 `(A >= 4096 || A < 16)` 让极小行宽
   skip-linearize（注意排除 A==1，见 CHANGES §3）；
3. `gather.cpp rowcopy_raw_single`：加 `B==1` 快速路径（跳过 p/K 除法，对齐 fastindex
   单索引 offset1）。

### 10. 优先级建议

> 注：本表 P0-P5 是**本文档内部**的差异点排序，与模型场景代号 P1/P2/P3
> （h[src] 等，见 `glossary.md`）及演进路线 R1-R9
> （`status.md` §7）均无关。

| 优先级 | 差异点 | 预期收益 | 改动量 |
|---|---|---|---|
| **P0** | #1 strided-rows 非连续尾维内核 | 转置/非连续形态从逐元素标量 → 分块 memcpy，预计 1.3-2x | 中（新增一个内核 + 路由） |
| **P1** | #3 grain 下限 64 | 小任务高线程 fork-join 固定开销下降 | 极小（grain.h 一行） |
| **P2** | #6 slice 提前回退 | 混合 slice 形态省掉白做的 normalize | 小 |
| **P3** | #2 linearize→单遍按形态评估 | A==1 大 K / 多索引形态 | 中（需逐形态实测） |
| **P4** | #4 `_offset` vs `_index` 变体 | 未知（微基准定） | 极小 |
| **P5** | #5 int32 保持（确认 mask 组合） | 防御性 | 极小 |
| — | #7 语义对齐 / #8 已融合 | 无需做 | — |

### 11. 复现命令

```bash
# 环境（2.10 或 2.5 均可；2.10 是更优运行环境）
source /data1/miniconda3/bin/activate torch2100_py12
# 或 source /data1/miniconda3/bin/activate torch2.5_envs
export LD_LIBRARY_PATH=$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8
cd /home/z00889957/100Work/1012_index_op_opt/index_sve-main_1021
# strided-rows（差异点 1）形态构造与对拍：
#   x = torch.randn(64, 20000).t()  →  x.t()[idx]（尾维非连续）
# 隔离交错 A/B：参考 /tmp/fusion_ab.py 的 ab() 写法（pin node5 + min-of-30）
# grain 下限（差异点 3）高线程复测：
OMP_NUM_THREADS=32 python /tmp/m3_threads_ab.py lookup
# 26-case merge（确认无回归）：
python -u benchmarks/bench_case_modes.py --measure at > /tmp/at.txt 2>&1
python -u benchmarks/bench_case_modes.py --measure opt > /tmp/opt.txt 2>&1
# SVE gather 变体微基准（差异点 4）：
#   /home/home/HPCKit/26.0.RC1/compiler/bisheng/bin/clang -O2 -march=armv8-a+sve2 \
#     -o /tmp/gather_micro /tmp/gather_micro.c && taskset -c 190 /tmp/gather_micro
```

---

# Part II：当前仓库（@ `a91d566`）逐条审计

## 12. 审计方法

- 读当前 `index_sve/op.cpp`、`normalize.cpp`、`gather.cpp`、`grain.h`。
- 用 `C.debug_spec(x, idx)` 打印 `B/K/A/after_contiguous/row_off/col_off`，确认
  实际路由形态。
- 用 `torch.profiler` 列出 `x[:, 0:10]`、`x[0:10, idx]`、`x[:, idx]`、
  `x[None, idx]` 真正调用的 aten op。
- pin node 5（CPU 190-227），`HPC_INDEX_NO_OVERRIDE=1`，同进程交替测
  `aten` 与 `index_sve.index`（避免覆盖干扰）。

## 13. 状态总表（以本仓库为准）

| # | 清单状态 | 当前仓库状态 | 结论 |
|---|---|---|---|
| 1 | 仍开放 | **已实现**：`col_off` 的相邻 run 合并，单索引 raw + run 块拷贝；仅 run≥4 接管 | **完成（run16 1.7-2.2x）** |
| 2 | 部分已做 | **部分实现**：多索引 A>=4 raw 仅 ≥16t 启用；A==1 多索引保持 linearize | **完成（按实测阈值）** |
| 3 | 仍开放 | **已实现**：`adaptive_grain` 下限 64 | **完成** |
| 4 | 未评估 | **已评估**：`_index` 与预计算 `_offset` 差 0-2.3%，且真实改造要存字节偏移/多一次缩放 | **放弃改代码** |
| 5 | 已赢/确认 | 整数索引保持 int32；mask 的 `nonzero()` 固定 int64 | **确认，不转换** |
| 6 | 已做但未提前 | 本架构不适用：slice 在 `aten::slice` 已消费 | **关闭** |
| 7 | 已对齐 | 已对齐 | 无需改 |
| 8 | 已融合关闭 | **已实现**：≤64B run/行用 memcpy；A<16 多线程 raw；B==1 去除法 | **完成（A8/A15 1.2-1.4x）** |

## 14. 逐条差异与当前代码证据

### 14.1 差异点 1：strided-rows

**形态实测**（`C.debug_spec`）：

- `x = torch.randn(64,20000).t()` → shape `(20000,64)`、stride `(1,20000)`；
  `x[idx]`：`B=1, K=4096, A=64, after_contiguous=false`，`col_off` 步长 20000
  （run 长度 1）。
- `x = torch.randn(64,200,10).permute(1,0,2)` → shape `(200,64,10)`、stride
  `(10,2000,1)`；`x[idx]`：`B=1, K=4096, A=640, after_contiguous=false`，
  `col_off` 为 64 段、每段 10 个连续元素（run 长度 10）。
- `x = torch.randn(64,200,16).permute(1,0,2)`：A=1024，64 段 × 16 连续。

**当前路由**：`A>1 && !after_contiguous` 不满足 M1 raw、不满足 rowcopy、不满足
`gather_fused` 条件之外的 SVE 分支，最终 `gather_fused` 的
`for c in 0..A: d[c]=s[col_off[c]]` 逐元素拷贝。

**基线**（node5，interleaved，单位 ms，ratio=aten/hpc）：

| 形态 | 线程 | aten | hpc | ratio |
|---|---|---|---|---|
| run10 A640 | 1 | 9.94 | 4.90 | 1.91x |
| run10 A640 | 8 | 1.50 | 1.05 | 1.38x |
| run10 A640 | 36 | 1.16 | 0.69 | 1.65x |
| run16 A1024 | 1 | 14.67 | 9.23 | 1.61x |
| run16 A1024 | 8 | 2.09 | 1.59 | 1.31x |
| transpose run1 A64 | 1 | 3.24 | 3.10 | 1.05x |
| transpose run1 A64 | 8 | 0.76 | 0.91 | **0.84x** |
| transpose run1 A64 | 36 | 0.83 | 0.37 | 2.27x |

结论：run≥10 已因 aten 更慢而有比值优势，但 hpc 自身仍是逐元素；块拷贝可继续
降低绝对时间。run=1 的纯转置没有连续块，strided-rows 不能改善，0.84x 需按噪声/
调度复核，不能用块拷贝掩盖。

**实现方案**：从 `col_off` 预生成 `(dst_col, src_off, len)` run 列表（相邻且
`col_off[c+1]==col_off[c]+1` 合并）；仅对 `max_run >= 4` 启用；单索引先走 raw
（不物化 idx），多索引走 fused 解码；每行每 run 调 `copy_contig_sve`/`memcpy`。
run=1 仍走现有 fused。

### 14.2 差异点 2：offset 融合单遍

当前已跳过 linearize 的路径：

- `gather_rowcopy_raw`：单索引、`after_contiguous`、`A>=4096||A<16||(A<=64&&B*K<=65536)`（T15 起放宽；1t 另 A>=4）、f32/f64。
- `gather_fused`：A<4 或非连续小块。
- `bucket_gather_pure_raw`：单索引、B==1、A==1。
- `gather_single_1d`：1t 单索引 A==1。
- `small_single_index`：单索引 K<=4096（绕 normalize）。

仍 linearize：

- **多索引 + A>=4 + after_contiguous + f32/f64**（如 `x[i,j]` 取行）；
- **多索引 + A==1 大 K**（4B/8B）；
- **单索引 + A∈[16,4095] 多线程**（本轮可评估放宽 raw）。

方案：新增多索引 raw rowcopy（复用 `gather_fused` 的预提取 + broadcast 解码，
内层改 `copy_contig_sve`），路由在 linearize 之前；A==1 多索引仍按清单警告保持
linearize，先做 A/B 再决定，不盲切。

### 14.3 差异点 3：grain 下限

`adaptive_grain` 当前：

```cpp
int64_t grain = std::max<int64_t>(1, n / (nt * 8));
```

改为 `max(64, n/(nt*8))`，并保留 `min(grain, MAX_GRAIN)`。影响仅在
`n/(nt*8)<64` 的小任务；大 K 不变。需回归 `gather_mask`（它用
`grain` 切 count chunk）、桶化 min/max、`gather_direct`。

### 14.4 差异点 4：`_offset` vs `_index`

当前所有 SVE gather 用 `svld1_gather_u32index_*` / `u64index_*`；`_offset` 变体
接收字节偏移。计划写独立 C 微基准：同一随机 idx，分别构造
`idx*esize` 字节偏移，比较 `_index` 与 `_offset` 的 1t/8t 吞吐。若差异 <3%，
保持现状（`_index` 语义更安全，避免 32-bit 字节偏移溢出）；若稳定更快再改。

### 14.5 差异点 5：int32 与 mask

- 普通整数索引：`test_normalize.py:53 test_int32_index_kept` 已保证保持 int32。
- mask 路径：`normalize.cpp expand_tensors()` 调 `index.nonzero()`，PyTorch 的
  `nonzero` 输出固定 int64；`debug_spec`/`linearize_spec` 对非连续 mask 实测
  `idx_dtype=torch.int64`。
- 连续 mask 走 `mask_fast_path`→`gather_mask`，不生成 idx，因此不受影响。
- 结论：非连续 mask 的 int64 是 `nonzero` 语义，不是 normalize 丢 int32；额外
  `.to(kInt)` 会多一遍转换，净收益不明，**不做**。若将来扩展 T8 支持 strided
  mask，应直接在 mask 上按 stride 压实，而不是转 idx dtype。

### 14.6 差异点 6：slice 提前回退

Profiler 证据：

| 表达式 | 实际 aten op |
|---|---|
| `x[:,0:10]` | `aten::slice` |
| `x[0:10, idx]` | `aten::slice` → `aten::index` |
| `x[:,idx]` | `aten::index` |
| `x[None,idx]` | `aten::unsqueeze` → `aten::index` |

`aten::index` 的 schema 只接收 `Tensor?[]`，slice 在到达它之前已被 view 化；
`index_impl` 看不到原始 slice。清单所说的“入口检测”只对 Python 层
`Tensor.__getitem__` monkey-patch 有效，与本仓库“C++ dispatcher 覆盖”架构冲突。
因此 **#6 关闭**；混合 slice 的成本已由 `aten::slice` + 子张量 index 承担，通常
子张量更小，不会出现旧版 0.21x 的整张量劣化。

### 14.7 差异点 7：语义

无需改。现有 `tests/test_normalize.py`、`test_errors.py`、`verify_against_aten.py`
覆盖 mask/广播/负索引/越界；T11 修复非连续 stride 正确性。

### 14.8 差异点 8：小行宽 + skip-linearize

当前：

- `copy_contig_sve` 尾部用 `svwhilelt`，不会越界读整 VL；但 `n<VL` 时仍走
  masked SVE load/store，不是 `memcpy`。
- 多线程单索引 A∈[17,64] 且大 K（`B*K>65536`）仍 `linearize+gather_index`
  （M1 条件 T15 起为 `A>=4096||A<16||(A<=64&&B*K<=65536)`；A<16 与小总
  工作量已放行。大 K 保 linearize 是实测分界：raw 逐行地址链在冷源下慢
  12-32%，见 `degradation-analysis.md` §5.9.4）。
- `rowcopy_raw_single` 每行 `p/K`、`p%K`，B==1 时是常量除法，可消。

方案：
1. `copy_contig_sve` 对 `n*sizeof(T) <= 16` 走 `std::memcpy`（A=4 f32 / A=2 f64），
   其余保持 SVE；
2. 单索引 raw rowcopy 门槛按 A/B 实测放宽到 A<16（排除 A==1）；
3. `rowcopy_raw_single` 加 `B==1` 分支，`k=p`；
4. 用 A-sweep（A=4/8/16/64）验证不劣化。

## 15. 本轮实施顺序与验收

| 阶段 | 内容 | 验收 | 状态 |
|---|---|---|---|
| 0 | 本文档 + 索引 | 链接/表格检查 | 完成 |
| 1 | grain 下限 64；B==1 除法快路径；小行宽 memcpy；A<16 路由 | 全量测试；A-sweep 1/8/36t 不劣化 | 完成 |
| 2 | strided-rows run Kernel（单索引 raw） | 新增 run10/run16/transpose 对拍；A/B 表 | 完成 |
| 3 | 多索引 raw rowcopy A>=4 | 多索引 case 对拍 + 1/8/36t | 完成（≥16t） |
| 4 | `_offset` 微基准；A==1 多索引 A/B | 微基准数据；若不快则不改 | 完成（均不改） |
| 5 | 文档同步（kernels/optimizations/status/degradation） | 交叉引用与数字一致 | 完成 |

测试环境与现有约定一致：

```bash
source ~/.bashrc && conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=/home/z00889957/miniconda3/envs/torch210_env/lib/python3.12/site-packages/torch/lib:/home/envs/HPCKit_26.1.RC1/HPCKit/26.1.RC1/compiler/bisheng/lib:$LD_LIBRARY_PATH
cd index_sve
python setup.py build_ext --inplace
```

## 16. 风险与边界

- **strided-rows 收益取决于 run 长度**：run=1 的纯转置没有连续后缀，块拷贝只能
  退化为逐元素，不能承诺 1.3-2x；文档中的收益只适用于 run≥4 的尾维布局。
- **A<16 放宽 raw** 可能让 linearize 的 amortized 成本消失后暴露行拷贝开销；
  必须 A-sweep 后按阈值放行，不能一刀切。
- **多索引 raw** 的 broadcast 解码比线性化后读 idx 更贵；只对 A>=4 连续块尝试。
- **grain=64** 对小 K 高线程是收益，对 `gather_mask` 的 chunk 数有影响；需验证
  小 mask（n<1M 串行计数）与大 mask 两条路。
- 所有性能结论必须 pin node 5、同进程交替、报告绝对 ms + ratio；共享 608 线程
  箱的 32/36t 比值可能被 aten 退化放大，不能单独作为收益证据。

---

# Part III：实施结果（2026-10-23）

## 17. 代码变更

| 文件 | 变更 |
|---|---|
| `index_sve/grain.h` | `adaptive_grain` 下限 1→64；大 K 公式不变 |
| `index_sve/gather.cpp` | `ColRun`/`make_col_runs` + `strided_rows_raw`；`copy_contig_sve` 对 ≤64B 走 `memcpy`；`rowcopy_raw_single` 加 `B==1` 分支；`rowcopy_fused_sve` 多索引 raw；`strided_rows_count`/`rowcopy_raw_count` 计数 |
| `index_sve/op.cpp` | 路由：run≥4 非连续尾维、单索引 A<16、多索引 A≥4 且 ≥16t；调试绑定 `debug_grain` |
| `tests/test_strided_rows.py` | run10/run16/run1/int32 正确性 + 路由计数（6） |
| `tests/test_rowcopy_routes.py` | A=8 raw、多索引 A=64 路由、grain 下限（3） |

> `HPC_INDEX_LEGACY` 只用于同进程 A/B，验证后已删除；提交代码不含该开关。

## 18. 性能结果（node 5，同进程交替，单位 ms）

**strided-rows（新 vs 旧路由）**：

| 形态 | 线程 | 旧 | 新 | 提升 |
|---|---|---|---|---|
| run16 A1024 | 1 | 10.56 | 4.74 | **2.23x** |
| run16 A1024 | 4 | 2.75 | 1.51 | 1.81x |
| run16 A1024 | 8 | 1.59 | 0.934 | 1.71x |
| run16 A1024 | 16 | 1.06 | 0.578 | 1.82x |
| run16 A1024 | 36 | 0.716 | 0.360 | 1.99x |
| run10 A640 | 1 | 4.71 | 3.50 | 1.36x |
| run10 A640 | 8 | 0.886 | 0.858 | 1.03x |
| run10 A640 | 16 | 0.588 | 0.599 | 0.98x |
| run10 A640 | 36 | 0.426 | 0.300 | 1.42x |
| run1 A64（纯转置） | 1/8/36 | 3.25/0.895/0.285 | 3.25/0.895/0.285 | 1.00x（未启用） |

**小行宽 raw（单索引 A=8/15，旧 linearize vs 新 raw）**：

| 形态 | 1t | 4t | 8t | 16t | 36t |
|---|---|---|---|---|---|
| A=8 | 1.03x | 1.21x | 1.24x | 1.27x | 1.37x |
| A=15 | 1.04x | 1.15x | 1.22x | 1.29x | 1.37x |

**多索引 A≥4（仅在 ≥16t 启用）**：

| 形态 | 1t | 4t | 8t | 16t | 36t |
|---|---|---|---|---|---|
| A=64 | 1.00x | 1.00x | 1.02x | 1.12x | 1.16x |
| A=8 | 1.00x | 1.00x | 1.01x | 1.19x | 1.48x |

**grain 下限 64（相对 1t 的强扩展，非 aten 比值）**：

| 形态 | 4t | 8t | 16t | 36t |
|---|---|---|---|---|
| 1D K=8192 旧 | 1.20 | 1.18 | 1.05 | **0.76** |
| 1D K=8192 新 | 1.18 | 1.24 | 1.17 | **0.98** |
| 2D A=16 K=8192 旧 | 4.10 | 6.57 | 6.43 | 4.66 |
| 2D A=16 K=8192 新 | 4.95 | 8.32 | 8.94 | **10.18** |

2D K=8192 的 36t 绝对时间从 0.097ms 降到 0.055ms（1.76x）；共享箱上比值会受
aten 退化放大，**以绝对时间/同进程 A/B 为准**。

## 19. 放弃/收敛的项与系统原因

### 19.1 纯转置 run=1：放弃块拷贝方案

`x = randn(64,20000).t(); x[idx]` 的 `col_off` 是 `0,20000,40000,...`，所有 run
长度都是 1。按定义不存在“连续后缀”，`run_strided_rows` 只能退化成逐元素；实测
1t 1.04x、8t **0.84x**、36t 2.7x（8t 下 hpc 0.895ms vs aten 0.751ms）。

- **系统原因**：tail 64 个元素分布在 64 条不同 cache line，且每行 stride=20000，
  没有可合并的连续块；`memcpy`/块拷贝无对象可拷。低线程下瓶颈是 64 次独立
  load 的延迟，aten 的标量 Indexer 与我们的 fused 循环同类，0.84x 是调度/解码
  固定开销，不是块拷贝能修的问题。
- **是否还有调优空间**：可以另写“固定 `col_off` 的 SVE gather”内核（每行用
  `svld1_gather_u32index/offset` 一次取 16 列），但这已不是 strided-rows，且
  收益未证明；本轮不引入。**保留 fused 路径，放弃该形态的块拷贝优化。**

### 19.2 多索引 A≥4 在 1-8t：回退 linearize

同进程 A/B：A=64 在 1/4/8t 新/旧为 0.77/0.87/0.91，A=8 为 0.76/1.01/1.14。
低线程下 `linearize` 的一次写 + rowcopy 一次读是可摊销的，而 raw 每行重复
broadcast 解码 + 边界检查更贵；16t 后 linearize 的串行写/读带宽占比上升，raw
才转正。**因此按 `nt>=16` 启用，低线程保留旧路径**，不是全量切换。

### 19.3 多索引 A==1：不启用 fused

当前 `linearize+gather` 在 `x[i,j]`（f32/i64，K=8192）实测 1t 1.08x、8t 1.24x，
36t 8.9x（aten 高线程退化）。清单也警告 fused 对 A==1 多索引“解码更慢”；
**保持现状，不做一刀切**。

### 19.4 `_offset` vs `_index`：不改进

预计算字节偏移的微基准（4M 随机访问，64MB 源，单线程）：

| 轮次 | `_index` ns/访问 | `_offset` ns/访问 | index/offset |
|---|---|---|---|
| 1 | 25.87 | 25.29 | 1.023 |
| 2 | 25.48 | 25.30 | 1.007 |
| 3 | 25.58 | 25.58 | 1.000 |

差 0-2.3%，在噪声内；真实改造还要存字节偏移（改变 linearize 的共享 idx 语义）
或每向量乘 4。**保持 `_index`**。

### 19.5 mask 的 int32：不转换

`nonzero()` 的返回类型固定 int64；`debug_spec`/`linearize_spec` 对非连续 mask
实测 `torch.int64`。连续 mask 已走 `gather_mask`（不生成 idx）。额外 `.to(kInt)`
会多一遍转换，且 T8 的扩展方向是“按 stride 直接压实 mask”，不是转 idx dtype；
**不做**。

### 19.6 slice 提前回退：架构上不可做

`aten::index` 的 indices 只含 `Tensor?`；`x[:,0:10]` 只有 `aten::slice`，
`x[0:10,idx]` 是 slice 后再 index。进入 `index_impl` 时原始 slice 已丢失，除非
改成 Python `Tensor.__getitem__` monkey-patch，与本仓库 C++ dispatcher 覆盖
架构冲突。**关闭该项。**

## 20. 验收

- 全量：**111/111**（原 102 + 新增 9）。
- 对拍：`tests/verify_against_aten.py --mode both` **44/44**。
- 关键 A/B 见 §18；纯转置 run=1 的 8t 0.84x 是既有问题，未用块拷贝掩盖，已在
  §19.1 记录。
- 共享箱注意：32/36t 的 aten 比值可能被带宽饱和放大，结论以绝对 ms + 同进程
  A/B 为准；多索引 16t 阈值是 920F/node5 的实测值，换机需复测。


## 21. 后续：多索引小 K 入口快路径（T13）

### 问题

T12 后实测 `x[i,j]`、K=512 的多索引小调用仍慢于 aten：1t A=1 为 0.67x、
8t 为 0.74x。原因不在单索引 `small_single_index`（它已串行化且 1.5-2.4x），
而在多索引仍走 `normalize + gather_direct`：

- `normalize_index` 每调用做 `expand_outplace` + 转置分析 + vector 分配；
- `gather_direct` 的 `all_full_contig=(nb==1)` 对 2D 索引恒 false，逐元素做
  `rr / bfactor[d] % bshape[d]` 广播解码；
- 内层 `d[c]=s[col_off[c]]` 标量拷贝。

### 变更

`op.cpp` 新增入口 `small_multi_index`（normalize 之前）：

- 索引 ≥2、全 int32/int64、同 shape 且连续、self 连续、覆盖前 nidx 维、
  `n≤65536`；
- 直接构造 out shape，一次预校验并算 `off[k]`；
- 扫描 offset 检测 dom==1：A==1 `fill`，A>1 循环 memcpy 同一行；
- 非全同时 A==1 标量、A≥2 逐行 memcpy；`n*A≥2048` 才并行。

不覆盖广播、非连续索引/self、n>65536，回退原路径。

### 实测（真实 `x[i,j]`，K=512，3000 次，µs/调用）

| 形态 | 线程 | aten | T13 | 比值 |
|---|---|---:|---:|---:|
| B=8，A=1 | 1 / 8 / 36 | 23.0 / 29.9 / 31.4 | 14.5 / 15.1 / 18.8 | 1.59 / 1.99 / 1.67x |
| B=8，A=4 | 1 / 8 / 36 | 36.0 / 52.9 / 48.8 | 18.9 / 22.9 / 29.9 | 1.90 / 2.31 / 1.64x |
| B=8，A=64 | 1 / 8 / 36 | 184.3 / 39.1 / 45.4 | 54.0 / 29.9 / 33.9 | 3.41 / 1.31 / 1.34x |

### 仍开放

- 广播索引、非连续索引/self 的小 K 仍走 `gather_direct`；可做 P1：
  修正 `all_full_contig`（全长连续即可线性读，不必 `nb==1`）+ 复用偏移预校验。
- ~~`small_single_index` 的 A==2 仍走标量循环，可统一为 memcpy~~（T14 已修，
  放宽到 A>=2；实测收益主要在 A=3：标量循环每行 ~2x 于 memcpy，shortq A=3
  K=1023 i64 从 0.84-0.99x 修到 2.6x+）。
- C++ 原生 fallback（`at::native::index_out`）只作为未覆盖形状的兜底，重跑
  `make_info` 固定成本高，不作为主路径。
