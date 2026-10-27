# 如何覆盖 / 劫持 aten::index（Dispatcher 机制详解）

本算子最核心的设计决策是**零改动接入**：`import index_sve` 后，进程中所有
CPU 张量的高级索引调用（`x[idx]`、`x[:, i]`、`x[i, j]`、bool mask、N-D 整数
下标）都自动改走本算子，无需修改任何调用点。本文档讲清楚这背后的 PyTorch
**Dispatcher（调度器）**机制、边界与验证方法。

> **读前须知**：本文假设你知道“算子（operator）”和“内核（kernel）”的区别。
> PyTorch 里一个算子可以有多份内核，Dispatcher 根据 **dispatch key**（CPU、
> CUDA、Autograd、Conjugate...）选一份执行。**schema** 是算子的参数/返回值
> 声明；**boxed** 调用是把参数打包成 `IValue` 的通用入口（便于跨语言/回退）；
> **autograd key** 在普通 CPU key 之前，负责记录反向图。理解这四点就能读懂
> 全文。

## 1. 一句话原理

`aten::index.Tensor` 在 dispatcher 里有一个按 **dispatch key**（CPU、CUDA、
Autograd 等）组织的 kernel 插槽表。我们做的不是改源码，而是**把 `CPU` 这个
key 指向的实现换成我们自己的**——在扩展的 C++ 初始化里用
`TORCH_LIBRARY_IMPL(aten, CPU, m)` 完成（T9 起；早期用 Python 的
`torch.library.impl`，每调用多一层解释器往返）。

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

### Step 2 — 扩展加载时把 aten 的 CPU 插槽指到我们（op.cpp）

```cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  const char* e = getenv("HPC_INDEX_NO_OVERRIDE");
  if (e == nullptr || std::string(e) != "1") {
    m.impl("index.Tensor", &index_impl);
    m.impl("index.Tensor_out", &index_out_aten);
  }
}
```

`TORCH_LIBRARY_IMPL(aten, CPU, ...)` 直接向 dispatcher 的
**`aten::index.Tensor` 算子 CPU key** 注册 kernel，**不经过 Python**。早期用
`torch.library.impl` 时，Python 侧 kernel 每次调用要往返解释器一次（实测小 K
多 ~10-20us）；C++ 注册后该开销消失。注册要求 C++ 签名与 schema 完全一致：
`index.Tensor` 的 indices 必须是 `const c10::List<...>&`，
`index.Tensor_out` 必须返回 `at::Tensor&`（故用 `index_out_aten` 薄包装）。
`register_aten.install()` 现为空实现，仅为兼容保留。

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
| 一次性告警 | 扩展加载时 `TORCH_LIBRARY_IMPL(aten, CPU)` 覆盖已有 kernel，dispatcher 打印一次 "Overriding a previously registered kernel..."（`previous kernel: BatchRulesScatterOps.cpp`） | 预期、无害，故意不压制（见 integration.md §4） |
| **递归风险** | 我们的 `index_impl` 内部**绝不能**再调 `at::index` / `self.index` / `aten::index`，否则 dispatcher 又回到我们自己 → 无限递归 | 全部手写 normalize/linearize/gather；代码审查确认无任何此类调用（semantics.md §6.3） |
| 重复注册 | C++ 同 key 重复注册会替换旧 kernel 并再打印一次告警；`importlib.reload(index_sve)` 不会破坏覆盖 | 需要关闭覆盖用 `HPC_INDEX_NO_OVERRIDE=1`（须在 import/dlopen 前设置） |

## 5. 验证方法

1. **dispatcher 表**：`torch._C._dispatch_dump_table()` 打印 dispatch 表，可见
   `aten::index.Tensor` 的 CPU key 从 `RegisterCPU_1.cpp` 变为扩展注册的 C++
   kernel（T9 前为 Python override；Task 10 曾用此法确认）。
2. **两进程 true-aten 对拍**：无覆盖进程（`HPC_INDEX_NO_OVERRIDE=1`）生成参考，
   覆盖进程比较——覆盖激活后 `x[i]` 已走本算子，**同一进程内无法再对拍**
   （semantics.md §6.2 的"覆盖下对拍陷阱"）。
3. **全量测试**：127/127 在覆盖激活态通过（错误消息逐字、K<=4096 小 K 路径、
   N-D 下标、mask 融合路径等；梯度对拍记录见 semantics.md §6.4，套件内暂无
   autograd 专项回归用例）。

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
