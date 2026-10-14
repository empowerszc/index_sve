# index_sve 文档系列

本目录是 `index_sve` 算子的详细技术文档，面向要理解、扩展或移植该算子的开发者。

## 文档索引

| 文档 | 内容 | 读者 |
|---|---|---|
| [architecture.md](architecture.md) | 三段流水线整体架构、控制流、dispatch 树、数据布局 | 架构理解、代码走读 |
| [semantics.md](semantics.md) | 与 ATen `aten::index.Tensor` 的语义对齐点与**有意差异**、错误信息、dtype 覆盖 | 正确性审查、回归测试 |
| [kernels.md](kernels.md) | SVE512 各内核（8B/4B gather、rowcopy、scalar）实现细节、预取、NT 开关 | 性能优化、内核修改 |
| [performance.md](performance.md) | 基准方法论、模型场景结果表、已知硬件限制、如何复测 | 性能评估、验收 |
| [integration.md](integration.md) | 默认覆盖 `aten::index.Tensor` 机制、运行时环境契约、逃逸开关 | 集成部署、排障 |
| [porting.md](porting.md) | 移植到其他 aarch64/SVE 变体、非 512-bit VL、其他 torch 版本 | 移植维护 |

## 速览

```
import index_sve
        │
        ▼  (torch.library.impl 覆盖 CPU 键)
aten::index.Tensor ──► index_impl(self, indices)
        │                 │
        │                 ├─ normalize_index()     对齐 ATen make_info
        │                 ├─ K < 1024 ? gather_direct()   (快速路径，无线性化)
        │                 └─ linearize() + gather_index()
        │                                              │
        │                 ┌─────────┬───────────┬─────┴──────────┐
        │                 ▼         ▼           ▼                ▼
        │             8B SVE   4B SVE    RowCopy SVE        dispatch_scalar
        │             gather    gather    (A>1, A>=16,        (回退全部
        │             (f64/i64) (f32/i32)  f32/f64 连续)        其他)
        ▼
    out
```

完整架构见 [architecture.md](architecture.md)。快速开始/构建/基准摘要见仓库根
`README.md`。
