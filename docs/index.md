# index_sve 文档系列

> **这篇讲什么**：`index_sve` 文档的入口地图——按你的目标选一条阅读路径，再按分组查具体文档。
> **读完你能回答**：我该从哪篇开始？哪篇是数字的唯一出处？术语和代号去哪查？文档之间是什么关系？
> **本篇代号**：S1/P1/M1/shortq（场景与负载）、A/B/K（索引几何）、R10（当前轮次）、
> RESULTS（`benchmarks/RESULTS.md`）。完整表见 [glossary.md](glossary.md)。

## 按目标选路径

| 我想… | 先读 | 再读 |
|---|---|---|
| 先跑起来 | 仓库根 `README.md` 的“快速开始” | [semantics.md](semantics.md) §1-§4（行为边界） |
| 系统了解（或讲给同事） | [summary.md](summary.md)（10-15 分钟） | [performance.md](performance.md) §3 |
| 评估性能 / 做选型 | [performance.md](performance.md) §3 | `benchmarks/RESULTS.md` §2 → [degradation-analysis.md](degradation-analysis.md) §0/§5 |
| 读源码 | [code-walkthrough.md](code-walkthrough.md) | [architecture.md](architecture.md)（设计动机） |
| 改内核 / 调优 | [kernels.md](kernels.md) | [architecture.md](architecture.md) → [optimizations.md](optimizations.md) §4 → [degradation-analysis.md](degradation-analysis.md) §7 |
| 集成 / 排障 / 与 compile 共存 | [integration.md](integration.md) | [override-mechanism.md](override-mechanism.md) → [torch-compile.md](torch-compile.md) |
| 移植到别的 CPU / torch | [porting.md](porting.md) | [kernels.md](kernels.md) §1-§3 |

## 文档地图

### 入门与总览

| 文档 | 内容 |
|---|---|
| [summary.md](summary.md) | **项目汇总（给同事的系统介绍）**：背景（AI4S 热点 27270ms）、aten::index 机制与慢因、六类改进手段、使用方法、基准方法与正确性、性能结果、劣化根因、为何不做性能级回退、边界清单、训练适用性、FAQ |
| [glossary.md](glossary.md) | **术语与代号速查（唯一名词入口）**：技术术语、场景/负载代号、形状记号、同名冲突警示 |
| [status.md](status.md) | **当前进展快照**：完成的工作、路由现状、性能结论、已知限制、演进方向（§7 优先级表，接续开发从这看） |

### 架构与内核

| 文档 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 三段流水线整体架构、控制流、dispatch 树、数据布局 |
| [code-walkthrough.md](code-walkthrough.md) | **源码走读（拿着代码读）**：文件顺序、一次 `x[idx]` 的完整旅程（真实函数名+行号）、IndexSpec 逐字段、op/normalize/linearize/gather 精读、路由表、并行约定、十个不变量、行号速查、调试出口 |
| [kernels.md](kernels.md) | SVE512 各内核（8B/4B gather、rowcopy、scalar）的实现、选择门槛与性能特性；含 T12-T15 新路径与预取/NT 开关 |

### 性能与劣化

| 文档 | 内容 |
|---|---|
| [performance.md](performance.md) | 基准方法论、当前轮（R10）结果表、胜出场景的硬件根因（§4.3）、已知硬件限制、如何复测 |
| [benchmarks.md](benchmarks.md) | **benchmarks/ 目录指南**：逐脚本用途、环境变量、典型命令、覆盖矩阵、与功能测试的区别 |
| [degradation-analysis.md](degradation-analysis.md) | **性能退化与线程扩展分析**：线程矩阵、逐 case 微架构定位、perf 方法、§5.9 残余劣化的硬件根因、优化优先级；矩阵为 T15 时点，§0 有当前对照 |
| [fallback-analysis.md](fallback-analysis.md) | **慢场景/回退路径深度分析**：为什么某些形态慢（64B 行放大、两遍模型、串行阶段、无内核 dtype）、为何不做性能级 fallback、判定速查 |

### 语义、集成与移植

| 文档 | 内容 |
|---|---|
| [semantics.md](semantics.md) | 与 aten `aten::index.Tensor` 的语义对齐点与**有意差异**、错误信息、dtype 覆盖 |
| [output-layout.md](output-layout.md) | **输出内存布局修复 + R10 直写**：channels_last/置换 self 上输出布局背离 aten 的 E2E bug、TensorIterator 置换恢复机制、自检探针、移植指南（§5）、R10 行块直写（§8） |
| [integration.md](integration.md) | 默认覆盖 `aten::index.Tensor` 的机制、运行时环境契约、逃逸开关、慢场景 fallback 决策 |
| [override-mechanism.md](override-mechanism.md) | **Dispatcher 劫持机制详解**：如何覆盖 aten::index、autograd 为何不受影响、递归/唯一性边界、验证方法 |
| [torch-compile.md](torch-compile.md) | **与 `torch.compile` 的关系**：Inductor 内联导致覆盖失效、三份实现对比、强制 fallback、缓存/私有 API 坑、`HPC_INDEX_COMPILE=1` opt-in |
| [porting.md](porting.md) | 移植到其他 aarch64/SVE 变体、非 512-bit VL、其他 torch 版本 |

### 编年史

| 文档 | 内容 |
|---|---|
| [optimizations.md](optimizations.md) | **T0-T15 优化/修复轮记录（编年史，非状态文档）**：每轮的问题、改动、实测、验证；变更清单、路由、复测 |

### 归档（archive/）

被后续轮次整体取代的快照文档。保留可追溯性，**不代表当前状态**；归档规则见
[archive/README.md](archive/README.md)。

| 文档 | 内容 | 归档日 / 取代者 |
|---|---|---|
| [index_sve_vs_fastindex_gap_20261023.md](archive/index_sve_vs_fastindex_gap_20261023.md) | fastindex 能力差异清单 + T13 时点仓库审计：strided-rows、offset 融合、grain 下限、`_offset` 变体、int32/mask、slice、小行宽 | 2026-10-29 / `status.md` §2、RESULTS §2.9；§19.1 深度分析仍被引用 |

### 背景参考（reference/）

| 文档 | 内容 | 读者 |
|---|---|---|
| [reference/aten_index_primer.md](reference/aten_index_primer.md) | **通俗导读**：`x[idx]` 从 Python 到 C++ 的完整思维模型（两层架构、Tensor 数据模型、视图/拷贝、六站旅程、慢的三个根因） | 懂 C++ 但不熟 PyTorch 内部机制的读者 |
| [reference/aten_index_aarch64_analysis.md](reference/aten_index_aarch64_analysis.md) | **深度分析**：基于 PyTorch 源码行号的 `aten::index` 全链路、性能画像、通用/SVE512/SME 优化清单、spconv 与 AI4S 实战案例、术语表 | 需要查证源码细节、规划移植/优化的读者 |

## 数字口径（一条规则）

**当前值的唯一矩阵出处 = `benchmarks/RESULTS.md` §2.9**（31 场景 ×
1/2/4/8/16/32/36/38t + realpath + cl 家族专项）。§2 顶部的轮次索引表标明每小节
是哪个轮次、是否当前；其它文档只写结论 + 指针，不复制全矩阵。

`README.md`、`summary.md`、`performance.md` 是允许保留代表值的“摘要位”；
`output-layout.md` §8 保留 R10 的前后对照作为该专项的实测细节。其余正文只写指针。
§2 之前任何标着 2-36t 的旧数据还受“libomp/libgomp 双 ICV”影响
（degradation-analysis.md §3.1），只作历史记录。

## 速览

下面是一张路由速览；完整架构与分支条件见 [architecture.md](architecture.md)。

```
import index_sve
        │
        ▼  (TORCH_LIBRARY_IMPL(aten, CPU) 覆盖 CPU 键)
aten::index.Tensor ──► index_impl(self, indices)
        │                 │
        │                 ├─ 单 mask && 覆盖前 m 维 && 连续 ? gather_mask()  (T8 单遍压实)
        │                 ├─ 单整数下标 && K<=4096 && 连续 ? small_single_index()  (T9)
        │                 ├─ 多索引同 shape/连续 && n<=65536 ? small_multi_index()  (T13)
        │                 ├─ normalize_index()     对齐 aten make_info
        │                 ├─ rowcopy_direct ? raw 块拷贝 / linearize+gather_index  (R10：cl/置换 out 直写，禁止 col_off 路径)
        │                 ├─ K < 1024 ? gather_direct()      (快速路径，无线性化；A==1 预校验紧凑循环)
        │                 ├─ 机制 M1 ? gather_rowcopy_raw()  (单索引 A>1 && (≥4096||<16||(≤64&&B·K≤65536)))
        │                 ├─ 非连续尾维有 ≥4 run ? strided_rows_raw()  (T12)
        │                 ├─ 多索引 A>=4 && ≥16t ? rowcopy_fused_sve() (T12)
        │                 ├─ T4 ? gather_bucket_pure_raw()   (单索引 B==1 A==1 小值域查表)
        │                 │        ├─ 大值域 && f32/i32+i64 && 源>512KB && >1t ? gather_single_1d_sve()  (T15-C 单遍 SVE)
        │                 │        └─ 其余大值域 ? gather_single_1d()  (单遍标量)
        │                 ├─ 1 线程 && 单索引 && A>1 ? A>=4 连续 ? raw rowcopy : fused
        │                 └─ A==1 || (连续 && A>=4 && f32/f64) ? linearize() + gather_index()
        │                    (否则 fused)
        │                                              │
        │                 ┌───────┬───────┬─────┬──────┴──────────┐
        │                 ▼       ▼       ▼     ▼               ▼
        │            小值域桶化 8B SVE  4B SVE  RowCopy SVE  dispatch_scalar
        │            (A==1     gather  gather  (A>1, A>=4,    (回退全部
        │             值域     (f64/i64)(f32/i32) f32/f64 连续) 其他)
        │             ≤2048)
        │         └─ 另：A<4/非连续小块 → gather_fused()（T3/T5 单遍）
        ▼
    out
```

## 维护约定

**数据时效（最重要）**

1. **当前值唯一出处 = `benchmarks/RESULTS.md` §2.9**。任何文档写“当前”数字都必须
   与它同源；改数值先改 RESULTS，其他地方只放摘要 + 指针。
2. **任何实测表格前必须有“数据轮次”行**：`> 数据轮次：<轮次>（<commit/日期>）｜当前`
   或 `｜历史（当前见 §x.y）`。没有轮次标注的数字视为不可引用。
3. **已修复项固定前缀 `【历史·已修】`**，且只能标在确实不再出现的形态上；仍 <1x 的
   形态不许标已修。判据是 RESULTS §2.9.1 的逐格清单。
4. **整篇被取代的快照进 `docs/archive/`**，文首写归档日/取代者/保留理由，规则见
   `archive/README.md`。局部历史数据不搬，留在原文档标 `【历史·<轮次>】`。

**其它**

- `status.md` 每轮实质变更后更新（§2 变更日志、§5.1 当前性能、§3 路由）。
- 归因类结论（尤其 4B gather）以**同构建 A/B** 为准；旧的独立探针数字只作历史
  背景，不得再作为因果依据（见 `fallback-analysis.md` §3.1、RESULTS §3.7）。
- 行号锚点：改完源码跑 `python3 tools/check_doc_anchors.py`（逐条校验 `file:NNN`
  是否还落在含对应符号的行上，漂移即报错）。
- 链接锚点：改完文档跑 `python3 tools/check_doc_links.py`（相对路径 + `#anchor`）。
- 新增或大改文档时：首屏加统一导读块；超过 300 行加 `## 目录`。
- `reference/` 为背景资料，不承诺与当前实现逐行同步；实现细节以其他文档/源码为准。
