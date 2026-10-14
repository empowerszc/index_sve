# 如何覆盖 / 劫持 aten::index（Dispatcher 机制详解）

本算子最核心的设计决策是**零改动接入**：`import index_sve` 后，进程中所有
CPU 张量的高级索引调用（`x[idx]`、`x[:, i]`、`x[i, j]`、bool mask、N-D 整数
下标）都自动改走本算子，无需修改任何调用点。本文档讲清楚这背后的 PyTorch
**Dispatcher（调度器）**机制、边界与验证方法。

## 1. 一句话原理

`aten::index.Tensor` 在 dispatcher 里有一个按 **dispatch key**（CPU、CUDA、
Autograd 等）组织的 kernel 插槽表。我们做的不是改源码，而是**把 `CPU` 这个
key 指向的实现换成我们自己的**——这通过官方 API `torch.library.impl` 完成。

## 2. 三步注册过程

### Step 1 — 注册自己的算子（op.cpp:38-43）

```cpp
TORCH_LIBRARY(hpc_index, m) {
  m.def("index(Tensor self, Tensor?[] indices) -> Tensor", &index_impl);
}
TORCH_LIBRARY_IMPL(hpc_index, CPU, m) {
  m.impl("index", &index_impl);
}
```

把 `torch.ops.hpc_index.index`（C++ 入口 `index_impl`）挂进 dispatcher，schema
与 `aten::index.Tensor` 完全相同（self + 可选下标列表 → Tensor）。

### Step 2 — import 时把 aten 的 CPU 插槽指到我们（register_aten.py:9-10）

```python
torch.library.impl("aten::index.Tensor", "CPU", torch.ops.hpc_index.index)
```

`torch.library.impl` 的意思是：向 dispatcher 中 **`aten::index.Tensor` 算子** 的
**`CPU` key** 注册一个 kernel，kernel 内容是 `torch.ops.hpc_index.index`。
这就在运行时**覆盖**了 PyTorch 默认挂在 CPU key 上的 C++ 实现
（`RegisterCPU_1.cpp` 中的 `index_cpu`）。

### Step 3 — `x[idx]` 的调用路径

1. `x[idx]` → `Tensor.__getitem__` → 底层 schema `aten::index.Tensor`；
2. dispatcher 按优先级解析 key：对纯 CPU tensor 是 `AutogradCPU → CPU`；
3. **`CPU` key 查到的是我们刚注册的函数**，于是路由进 `index_impl` →
   normalize → linearize → SVE gather。

全程无调用点改动。这就是"劫持"的本质：**不是替换 Python 层的 `__getitem__`，
而是替换 dispatcher 表中该算子 CPU key 的 kernel**。

## 3. 为什么 autograd backward 不受影响

`x[i]` 的求导公式（`IndexBackward`）挂在 **`Autograd` key** 上，是独立的
kernel，不依赖 CPU key 的实现。dispatcher 链为：

```
AutogradCPU → CPU（我们的实现）→ 底层
```

forward 走到我们；backward 的梯度散射仍是 ATen 自己的公式。所以测试中
autograd 完全正常（梯度值/形状/dtype 已与 true-aten 对拍验证，见 semantics.md
§6）。

## 4. 三个边界（为什么这方案成立、约束在哪）

| 边界 | 说明 | 处理 |
|---|---|---|
| 一次性 UserWarning | `torch.library.impl` 发现 CPU key 已有 kernel，警告 "Overriding a previously registered kernel..." | 预期、无害，故意不压制（见 integration.md §4） |
| **递归风险** | 我们的 `index_impl` 内部**绝不能**再调 `at::index` / `self.index` / `aten::index`，否则 dispatcher 又回到我们自己 → 无限递归 | 全部手写 normalize/linearize/gather；代码审查确认无任何此类调用（semantics.md §6.3） |
| 同进程唯一 | torch 2.10 不允许同进程第二个 Python 覆盖注册到同一 op 同一 key（`RuntimeError: already a kernel registered from python`） | 默认覆盖设计的固有约束；需要第二个实现请改用 `HPC_INDEX_NO_OVERRIDE` 场景 |

## 5. 验证方法

1. **dispatcher 表**：`torch._C._dispatch_dump_table()` 打印 dispatch 表，可见
   `aten::index.Tensor` 的 CPU key 从 `RegisterCPU_1.cpp` 变为 Python override
   （Task 10 用此法确认）。
2. **两进程 true-aten 对拍**：无覆盖进程（`HPC_INDEX_NO_OVERRIDE=1`）生成参考，
   覆盖进程比较——覆盖激活后 `x[i]` 已走本算子，**同一进程内无法再对拍**
   （semantics.md §6.2 的"覆盖下对拍陷阱"）。
3. **全量测试**：64/64 在覆盖激活态通过（含 backward、错误消息逐字、
   K<1024/K>=1024、N-D 下标等）。

## 6. 为什么是 `aten::index.Tensor` 而不是别的

- `x[i]`（1 个整数张量下标）→ `aten::index.Tensor`；
- `x[:, i]` / `x[i, j]` / bool mask / N-D → 都归并到 `aten::index.Tensor`；
- `x[None]` → `TensorIndexing.h` 走 `unsqueeze`（**不进本算子**）；
- `x[tensor(0-dim int)]` → `applySelect`（**不进本算子**）。

这两类"非本算子"路径行为与覆盖前完全一致，测试套件已覆盖
（semantics.md §7）。

## 7. 相关环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HPC_INDEX_NO_OVERRIDE` | 0 | `=1` 时 import **跳过**覆盖（基准/对拍用，见 integration.md §2） |
| `HPC_INDEX_NT` | 0 | 非临时 gather/store 开关（opt-in，默认关） |
