# index_sve 文档系列

本目录是 `index_sve` 算子的详细技术文档，面向要理解、扩展或移植该算子的开发者。

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [architecture.md](architecture.md) | 三段流水线整体架构、控制流、dispatch 树、数据布局 | 架构理解、代码走读 |
| [semantics.md](semantics.md) | 与 ATen `aten::index.Tensor` 的语义对齐点与**有意差异**、错误信息、dtype 覆盖 | 正确性审查、回归测试 |
| [kernels.md](kernels.md) | SVE512 各内核（8B/4B gather、rowcopy、scalar）实现细节 + 优化轮新路径（M1 raw rowcopy、T3 fused、T4 桶化）、预取、NT 开关 | 性能优化、内核修改 |
| [performance.md](performance.md) | 基准方法论、模型场景结果表、已知硬件限制、如何复测 | 性能评估、验收 |
| [optimizations.md](optimizations.md) | **T0-T5 优化轮记录**：针对 AI4S 真实负载的逐项优化（自适应粒度、rowcopy A>=4、M1 skip-linearize、融合单遍、小值域桶化、1 线程单遍修复），变更、路由、实测数据、复测 | 想了解"做了哪些优化"的读者 |
| [integration.md](integration.md) | 默认覆盖 `aten::index.Tensor` 机制、运行时环境契约、逃逸开关、慢场景 fallback 决策（不改） | 集成部署、排障 |
| [override-mechanism.md](override-mechanism.md) | **Dispatcher 劫持机制详解**：如何覆盖 aten::index、autograd 为何不受影响、递归/唯一性边界、验证方法 | 想搞清"怎么做到的"的读者 |
| [porting.md](porting.md) | 移植到其他 aarch64/SVE 变体、非 512-bit VL、其他 torch 版本 | 移植维护 |
| [status.md](status.md) | **当前进展快照**：完成的工作、路由现状、性能结论、已知限制、下一步 | 新读者/agent 快速上手 |

## 速览

```
import index_sve
        │
        ▼  (torch.library.impl 覆盖 CPU 键)
aten::index.Tensor ──► index_impl(self, indices)
        │                 │
        │                 ├─ normalize_index()     对齐 ATen make_info
        │                 ├─ K < 1024 ? gather_direct()      (快速路径，无线性化)
        │                 ├─ M1 ? gather_rowcopy_raw()       (单索引 A>=4096 skip-linearize)
        │                 ├─ T4 ? gather_bucket_pure_raw()   (单索引 B==1 A==1 小值域查表)
        │                 │        └─ 大值域 && 1 线程 ? gather_single_1d()  (单遍标量)
        │                 ├─ 1 线程 && 单索引 && A>1 ? gather_fused()  (跳过 linearize)
        │                 └─ linearize() + gather_index()
        │                                              │
        │                 ┌───────┬───────┬─────┬──────┴──────────┐
        │                 ▼       ▼       ▼     ▼               ▼
        │            小值域桶化 8B SVE  4B SVE  RowCopy SVE  dispatch_scalar
        │            (A==1     gather  gather  (A>1, A>=4,    (回退全部
        │             值域     (f64/i64)(f32/i32) f32/f64 连续) 其他)
        │             ≤2048)
        │         └─ 另：A<4/非连续小块、1 线程单索引 A>1 → gather_fused()（T3/T5 单遍）
        ▼
    out
```

完整架构见 [architecture.md](architecture.md)。快速开始/构建/基准摘要见仓库根
`README.md`。
