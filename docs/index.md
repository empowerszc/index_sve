# index_sve 文档系列

本目录是 `index_sve` 算子的详细技术文档，面向要理解、扩展或移植该算子的开发者。

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [architecture.md](architecture.md) | 三段流水线整体架构、控制流、dispatch 树、数据布局 | 架构理解、代码走读 |
| [semantics.md](semantics.md) | 与 ATen `aten::index.Tensor` 的语义对齐点与**有意差异**、错误信息、dtype 覆盖 | 正确性审查、回归测试 |
| [kernels.md](kernels.md) | SVE512 各内核（8B/4B gather、rowcopy、scalar）实现细节 + 优化轮新路径（M1 raw rowcopy、T3 fused、T4 桶化）、预取、NT 开关 | 性能优化、内核修改 |
| [performance.md](performance.md) | 基准方法论、模型场景结果表、已知硬件限制、如何复测 | 性能评估、验收 |
| [degradation-analysis.md](degradation-analysis.md) | **性能退化与线程扩展分析**：术语表、修正后的 1/2/4/8/16/32/36 线程矩阵、逐 case 微架构定位（x[:,i]、2B/1B、非连续 mask、小规模并行）、perf 方法、优化优先级 | 想知道"哪个线程数慢、为什么慢"的读者 |
| [optimizations.md](optimizations.md) | **T0-T11 优化/修复轮记录**：自适应粒度、rowcopy A>=4、M1 skip-linearize、融合单遍、小值域桶化、1 线程单遍、跨设备校验、mask 单遍压实、小 K 精简路径、compile opt-in；变更、路由、实测数据、复测 | 想了解"做了哪些优化"的读者 |
| [fallback-analysis.md](fallback-analysis.md) | **慢场景/回退路径深度分析**：为什么某些形态慢（4B 随机 gather 的 64B 行放大 + DRAM 延迟、两遍模型、串行阶段、无内核 dtype）、受限于什么、为何不做 fallback、判定速查表 | 想搞清"为什么慢、能不能修"的读者 |
| [integration.md](integration.md) | 默认覆盖 `aten::index.Tensor` 机制、运行时环境契约、逃逸开关、慢场景 fallback 决策（不改） | 集成部署、排障 |
| [torch-compile.md](torch-compile.md) | **与 `torch.compile` 的关系**：Inductor 内联导致覆盖失效、三份实现对比、强制 fallback 让 compile 用回本算子（实测 2.6x→1.07-1.3x）、缓存/私有 API/FakeTensor 坑、`HPC_INDEX_COMPILE=1` opt-in 落地 | 需要在 compile/export 场景部署的读者 |
| [override-mechanism.md](override-mechanism.md) | **Dispatcher 劫持机制详解**：如何覆盖 aten::index、autograd 为何不受影响、递归/唯一性边界、验证方法 | 想搞清"怎么做到的"的读者 |
| [porting.md](porting.md) | 移植到其他 aarch64/SVE 变体、非 512-bit VL、其他 torch 版本 | 移植维护 |
| [status.md](status.md) | **当前进展快照**：完成的工作、路由现状、性能结论、已知限制、下一步 | 新读者/agent 快速上手 |

### 背景参考（reference/）

| 文档 | 内容 | 读者 |
|---|---|---|
| [reference/aten_index_primer.md](reference/aten_index_primer.md) | **通俗导读**：`x[idx]` 从 Python 到 C++ 的完整思维模型（两层架构、Tensor 数据模型、视图/拷贝、六站旅程、慢的三个根因） | 懂 C++ 但不熟 PyTorch 内部机制的读者 |
| [reference/aten_index_aarch64_analysis.md](reference/aten_index_aarch64_analysis.md) | **深度分析**：基于 PyTorch 源码行号的 `aten::index` 全链路、性能画像、通用/SVE512/SME 优化清单、spconv 与 AI4S 实战案例、术语表 | 需要查证源码细节、规划移植/优化的读者 |

## 速览

```
import index_sve
        │
        ▼  (TORCH_LIBRARY_IMPL(aten, CPU) 覆盖 CPU 键)
aten::index.Tensor ──► index_impl(self, indices)
        │                 │
        │                 ├─ 单 mask && 覆盖前 m 维 && 连续 ? gather_mask()  (T8 单遍压实)
        │                 ├─ 单整数下标 && K<=4096 && 连续 ? small_single_index()  (T9)
        │                 ├─ normalize_index()     对齐 ATen make_info
        │                 ├─ K < 1024 ? gather_direct()      (快速路径，无线性化；A==1 预校验紧凑循环)
        │                 ├─ M1 ? gather_rowcopy_raw()       (单索引 A>=4096 skip-linearize)
        │                 ├─ T4 ? gather_bucket_pure_raw()   (单索引 B==1 A==1 小值域查表)
        │                 │        └─ 无 SVE 内核 dtype 或 1 线程 ? gather_single_1d()  (单遍标量)
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

- `status.md` 每轮实质变更后更新（测试数、commit、性能结论、路由）。
- 归因类结论（尤其 4B gather）以**同构建 A/B** 为准；旧的独立探针数字只作历史
  背景，不得再作为因果依据（见 `fallback-analysis.md` §3.1、RESULTS §8.4）。
- `reference/` 为背景资料，不承诺与当前实现逐行同步；实现细节以其他文档/源码为准。
