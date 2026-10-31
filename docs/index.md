# index_sve 文档系列

本目录是 `index_sve` 算子的详细技术文档，面向要理解、扩展或移植该算子的开发者。

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [summary.md](summary.md) | **项目汇总（给同事的系统介绍）**：背景（AI4S 热点 27270ms）、aten::index 机制与慢因、六类改进手段、使用方法（含 torch.compile 重载）、基准方法与正确性、性能测试结果（通用 + AI4S 专项）、劣化根因、为何不做性能级回退、绕过/边界清单、训练适用性、FAQ | 第一次接触项目、要与他人系统讨论的读者 |
| [glossary.md](glossary.md) | **术语与代号速查（唯一名词入口）**：技术术语表、场景/负载代号（S1/P1/M1-M6/A1/shortq…）、形状记号、代号冲突警示 | 读任何文档遇到看不懂的名词/代号时查 |
| [architecture.md](architecture.md) | 三段流水线整体架构、控制流、dispatch 树、数据布局 | 架构理解、代码走读 |
| [code-walkthrough.md](code-walkthrough.md) | **源码走读（拿着代码读）**：文件清单与阅读顺序、一次 `x[idx]` 的完整旅程（真实函数名+行号）、IndexSpec 逐字段与 B-K-A 算例、op/normalize/linearize/gather 逐段精读、index_into 路由表、并行约定、十个不变量、函数→行号速查、pybind 调试出口 | 要读懂/修改源码的读者 |
| [semantics.md](semantics.md) | 与 ATen `aten::index.Tensor` 的语义对齐点与**有意差异**、错误信息、dtype 覆盖 | 正确性审查、回归测试 |
| [output-layout.md](output-layout.md) | **输出内存布局修复 + R10 直写（2026-10-29）**：channels_last/置换 self 上输出布局背离 aten 的 E2E bug——现象实测表、TensorIterator 置换恢复机制（源码行号）、一分钟自检探针、其他版本移植指南（§5）；R10 行块直写（§8：dense 置换等价性、资格判定、0.30-0.42x→1.45-2.54x 实测） | 排查同类 bug、持有旧版本要修复的 agent、做布局相关优化 |
| [kernels.md](kernels.md) | SVE512 各内核（8B/4B gather、rowcopy、scalar）实现细节 + 优化轮新路径（M1 raw rowcopy、T3 fused、T4 桶化、T12 strided/multi raw、T13 small_multi、T15-C 大源 4B 直连 SVE）、预取、NT 开关 | 性能优化、内核修改 |
| [performance.md](performance.md) | 基准方法论、**当前轮（R10）模型场景结果表**、**胜出场景的硬件根因（§4.3：rowcopy/gather MLP/遍数消除/固定开销/桶化/串行边界）**、已知硬件限制、如何复测 | 性能评估、验收、想知道"为什么快"的读者 |
| [benchmarks.md](benchmarks.md) | **benchmarks/ 目录指南**：逐脚本用途、公共环境变量、典型命令、真实 workload 覆盖与预测力矩阵、与功能测试的区别 | 想知道"该跑哪个基准、结果能不能外推"的读者 |
| [degradation-analysis.md](degradation-analysis.md) | **性能退化与线程扩展分析**（矩阵为 T15 时点历史；当前值见该文文首 R10 表）：1/2/4/8/16/32/36 线程矩阵、逐 case 微架构定位（x[:,i]、2B/1B、非连续 mask、小规模并行）、perf 方法、**§5.9 残余劣化的硬件根因**、**§6.5 生产形态（小 K + 6-19t）的劣化家族**（含 §6.5.7 对"固定开销占 93-95%"的逐项裁决）、优化优先级 | 想知道"哪个线程数慢、为什么慢"的读者 |
| [optimizations.md](optimizations.md) | **T0-T15 优化/修复轮记录（编年史，非状态文档）**：自适应粒度、rowcopy A>=4、M1 skip-linearize、融合单遍、小值域桶化、1 线程单遍、跨设备校验、mask 单遍压实、小 K 精简路径、compile opt-in、T12-T15（strided run 块/multi raw/raw 放宽/4B 直连 SVE）；变更、路由、实测数据、复测 | 想了解"做了哪些优化"的读者 |
| [fallback-analysis.md](fallback-analysis.md) | **慢场景/回退路径深度分析**（文首有当前 <1x 速查；已修项标【历史·已修】）：为什么某些形态慢（4B 随机 gather 的 64B 行放大 + DRAM 延迟、两遍模型、串行阶段、无内核 dtype）、受限于什么、为何不做 fallback、判定速查表 | 想搞清"为什么慢、能不能修"的读者 |
| [integration.md](integration.md) | 默认覆盖 `aten::index.Tensor` 机制、运行时环境契约、逃逸开关、慢场景 fallback 决策（不改） | 集成部署、排障 |
| [torch-compile.md](torch-compile.md) | **与 `torch.compile` 的关系**：Inductor 内联导致覆盖失效、三份实现对比、强制 fallback 让 compile 用回本算子（实测 2.6x→1.07-1.3x）、缓存/私有 API/FakeTensor 坑、`HPC_INDEX_COMPILE=1` opt-in 落地 | 需要在 compile/export 场景部署的读者 |
| [override-mechanism.md](override-mechanism.md) | **Dispatcher 劫持机制详解**：如何覆盖 aten::index、autograd 为何不受影响、递归/唯一性边界、验证方法 | 想搞清"怎么做到的"的读者 |
| [porting.md](porting.md) | 移植到其他 aarch64/SVE 变体、非 512-bit VL、其他 torch 版本 | 移植维护 |
| [status.md](status.md) | **当前进展快照**：完成的工作、路由现状、性能结论、已知限制、**演进方向（§7 优先级表，接续开发从这看）** | 新读者/agent 快速上手 |

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

## 推荐阅读路径

- **第一次接触，只想跑起来**：仓库 `README.md` 的“术语与结果口径”+“快速开始”
  → [semantics.md](semantics.md) §1-§4（行为边界）→ [status.md](status.md) §3-§4。
- **评估性能/做选型**：[performance.md](performance.md) §3 → `benchmarks/RESULTS.md`
  §2 → [degradation-analysis.md](degradation-analysis.md)（先看文首 R10 表与 §5 逐案）；
  短调用 / 真实 workload 收益另跑 `bench_realpath_driver.py`（见
  [benchmarks.md](benchmarks.md) §3 覆盖矩阵）。
- **要读源码**：[code-walkthrough.md](code-walkthrough.md)（按文件与行号走读全部核心
  源码，含路由表与调试出口）→ 遇到"为什么这样设计"回 [architecture.md](architecture.md)。
- **要改内核/调优**：[kernels.md](kernels.md) → [architecture.md](architecture.md)
  → [optimizations.md](optimizations.md) §4 路由 → [degradation-analysis.md](degradation-analysis.md)
  §7 微架构证据 → [fallback-analysis.md](fallback-analysis.md)
  → [archive/index_sve_vs_fastindex_gap_20261023.md](archive/index_sve_vs_fastindex_gap_20261023.md)
  （fastindex 能力差异与待补项，历史快照）。
- **集成 / 排障 / 与 compile 共存**：[integration.md](integration.md)
  → [override-mechanism.md](override-mechanism.md) → [torch-compile.md](torch-compile.md)。
- **移植到别的 CPU/torch**：[porting.md](porting.md) → [kernels.md](kernels.md) §1-§3。

> **数字口径（一条规则）**：**当前值唯一出处 = `benchmarks/RESULTS.md` §2.9**
> （31 场景 × 1/2/4/8/16/32/36/38t + realpath + cl 家族专项）、**§2.10**
> （生产形态轴：K 512-55296 × 1/6/12/19t，`x[idx]` 走 dispatch）与 **§2.11**
> （上下文开销：hot/池冷/源冷/autograd + 固定链 + 混合序列）。§2 顶部的轮次
> 索引表标明每小节是哪个轮次、是否当前；其它文档只写结论 + 指针，不复制全矩阵。
> §2 之前任何标着 2-36t 的旧数据还受"libomp/libgomp 双 ICV"影响（degradation §3.1），
> 只作历史记录。

## 速览

```
import index_sve
        │
        ▼  (TORCH_LIBRARY_IMPL(aten, CPU) 覆盖 CPU 键)
aten::index.Tensor ──► index_impl(self, indices)
        │                 │
        │                 ├─ 单 mask && 覆盖前 m 维 && 连续 ? gather_mask()  (T8 单遍压实)
        │                 ├─ 单整数下标 && K<=4096 && 连续 ? small_single_index()  (T9)
        │                 ├─ 多索引同 shape/连续 && n<=65536 ? small_multi_index()  (T13)
        │                 ├─ normalize_index()     对齐 ATen make_info
        │                 ├─ rowcopy_direct ? raw 块拷贝 / linearize+gather_index  (R10：cl/置换 out 直写，禁止 col_off 路径)
        │                 ├─ K < 1024 ? gather_direct()      (快速路径，无线性化；A==1 预校验紧凑循环)
        │                 ├─ M1 ? gather_rowcopy_raw()       (单索引 A>1 && (≥4096||<16||(≤64&&B·K≤65536)))
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

完整架构见 [architecture.md](architecture.md)。快速开始/构建/基准摘要见仓库根
`README.md`。

## 维护约定

**数据时效（最重要）**

1. **当前值唯一出处 = `benchmarks/RESULTS.md` §2.9（全量矩阵）+ §2.10（生产形态轴）
   + §2.11（上下文/固定开销）**。
   任何文档写"当前"数字都必须与它同源；改数值先改 RESULTS，其他地方只放摘要 + 指针。
2. **任何实测表格前必须有"数据轮次"行**：`> 数据轮次：<轮次>（<commit/日期>）｜当前`
   或 `｜历史（当前见 §x.y）`。没有轮次标注的数字视为不可引用。
3. **已修复项固定前缀 `【历史·已修】`**，且只能标在确实不再出现的形态上；仍 <1x 的
   形态不许标已修。判据是 RESULTS §2.9.1 的逐格清单。
4. **整篇被取代的快照进 `docs/archive/`**，文首写归档日/取代者/保留理由，规则见
   `archive/README.md`。局部历史数据不搬，留在原文档标 `【历史·<轮次>】`。

**其它**

- `status.md` 每轮实质变更后更新（§2 变更日志、§5.1 当前性能、§3 路由）。
- 归因类结论（尤其 4B gather）以**同构建 A/B** 为准；旧的独立探针数字只作历史
  背景，不得再作为因果依据（见 `fallback-analysis.md` §3.1、RESULTS §3.7）。
- 行号锚点：改完源码跑 `python tools/check_doc_anchors.py`。它校验 `file:NNN`、
  `symbol:NNN` 与裸 `:NNN` 三种写法：行号范围、区间两端（含"终点是不是越界到
  下一个函数的注释/模板行"）、注释与残句；写成 `符号 :NNN` 的，还要求行号落在
  该符号的函数体内。输出按覆盖强度分三桶报数——**函数体已校验 / 单引用弱校验
  （±3 行窗口里出现文档点名的符号）/ 仅结构校验**，看最后一桶就知道有多少条只挡
  得住大位移。**一张表里同时引用多个文件时，行号必须写成 `file:NNN`**；关键锚点
  尽量用 `file:NNN`/`symbol:NNN` 或"符号 :NNN"。`docs/archive/`（冻结快照）与
  `docs/reference/`（aten 导读）豁免。
- `reference/` 为背景资料，不承诺与当前实现逐行同步；实现细节以其他文档/源码为准。
