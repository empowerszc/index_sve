# 集成 / 运行时契约

本文档说明本算子如何接入 torch 运行时、环境要求，以及集成时要注意的全局
副作用。

## 1. 默认覆盖 aten::index.Tensor（A6）

`index_sve/index_sve/__init__.py`：

```python
from . import _hpc_index            # dlopen 时注册 torch.ops.hpc_index
from .register_aten import install as _install_aten
import torch, os
if os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") != "1":
    _install_aten()                 # 直接覆盖 aten::index.Tensor（CPU）
```

`register_aten.install()`：

```python
torch.library.impl("aten::index.Tensor", "CPU", torch.ops.hpc_index.index)
torch.library.impl("aten::index.Tensor_out", "CPU", torch.ops.hpc_index.index_out)
```

效果：**`import index_sve` 之后，进程中所有 CPU 张量的高级索引
（`x[idx]`、`x[:, i]`、`x[i, j]`、bool mask、N-D 下标）都改走本算子**，无需改
调用点。已验证：
- dispatcher dump 证实 CPU key 从 `RegisterCPU_1.cpp` 移至 Python 覆盖；
- 覆盖后 autograd backward 正常（`IndexBackward` 独立注册，不受 CPU kernel 影响）。

## 2. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HPC_INDEX_NT` | 0 | 非临时 gather/store 开关（`!= "0"` 即开；opt-in，默认关） |
| `HPC_INDEX_NO_OVERRIDE` | 0 | `=1` 时 import **跳过** aten::index 覆盖（基准/对拍用） |

`HPC_INDEX_NO_OVERRIDE` 的用途：
- 只想用 `torch.ops.hpc_index.index` 本体、不改全局 `x[...]` 的部署；
- 基准/测试需要 `x[i]` 引用仍指向 true aten（覆盖激活后 `x[i]` 会自洽化，
  无法对拍——见 semantics.md §6 的"覆盖下对拍陷阱"）。

## 3. 运行时环境契约

扩展链接了 BiSheng 工具链的 `libomp.so`（`-fopenmp`），且依赖 torch 的 lib：

```
LD_LIBRARY_PATH=<torch_lib>:<bisheng_lib>:$LD_LIBRARY_PATH
```

- `<torch_lib>`：torch 包内 `lib`（本机
  `/home/z00889957/miniconda3/envs/torch210_env/lib/python3.12/site-packages/torch/lib`）；
- `<bisheng_lib>`：BiSheng 编译器 lib 目录（本机
  `/home/envs/HPCKit_26.1.RC1/HPCKit/26.1.RC1/compiler/bisheng/lib`）。

缺 `<bisheng_lib>` 时：`import index_sve` 报
`libomp.so: cannot open shared object file`（dlopen 失败）。

## 4. 一次性 UserWarning（正常，勿压制）

首次 import 会从 `torch.library.impl` 触发：

```
UserWarning: Overriding a previously registered kernel for the same operator
and the same dispatch key
  operator: aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  dispatch key: CPU
```

**预期且无害**，故意不压制（用户应知晓全局覆盖已生效）。`_hpc_index` 内的
`torch.ops.hpc_index` 注册没有此类警告。

## 5. 全局副作用与限制

- **覆盖是进程级全局的**：任何 import index_sve 的进程，其 `x[...]` 全部改道。
  这是设计目标（零改动接入），但部署时需明确告知所有调用方。
- **单覆盖限制**：torch 2.10 不允许同进程第二个 Python CPU kernel 注册到
  `aten::index.Tensor`（`RuntimeError: already a kernel registered from python`）。
  这是默认覆盖设计的固有约束；需要第二个实现请改注册名或用
  `HPC_INDEX_NO_OVERRIDE` 场景。
- `importlib.reload(index_sve)` 会再次调用 `install()` 触发同样错误（固有，
  正常流程不应 reload）。
- `aten::index.Tensor_out` 也已覆盖（T6）：`out=` 调用同样走本算子；
  QuantizedCPU 等其它 dispatch key 不受影响。

## 6. 部署建议

- 模型/服务入口最先 import index_sve（保证所有索引早于计算改道）；
- 用 `HPC_INDEX_NT=1` 试验大输出场景（默认关，需实测收益）；
- 提供环境的 `setvars.sh` + LD_LIBRARY_PATH 到部署脚本；
- 用 `torch._C._dispatch_dump_table()` 验证 CPU key 已到覆盖（调试用）。

## 7. 慢场景 fallback 决策（已评估，结论：不改）

针对"本算子性能差于 aten::index 时回退到原生 aten::index"的诉求，做了一轮完整
调查，结论是**不实现 fallback**。记录理由供日后复读：

- **慢场景**（RESULTS §8/performance §3）：1D f32/i32 大 K（K=10M）纯 gather 在
  ≥8 线程（0.18-0.7x）、P2 608t（0.38x）。P3 mask 高线程问题已由 T8 单遍融合
  修复（38t 4.6x）。低线程的
  RowCopy/PureGather 8B 是本算子的可靠收益（2-4t 20-34x），不受影响。
- **Python-wrapper fallback 不可取**：实测 `torch.library.get_kernel` +
  `with_keyset` + `call_boxed` 能正确回退且 autograd 安全（无递归），但任何
  Python 层包装会给**每次** `x[...]` 调用增加 ~45µs（小 op 13.7µs→63µs，
  S1 4ms 场景仅 +1%）。对以 RowCopy 小/中张量为收益主体的算子，这是全局性
  回归，不符合"常规标准流程"。
- **C++ 内核内 scalar 回退也追不平**：4B 纯 gather 落 `dispatch_scalar` 后实测
  0.32-0.55 GB/s（ATen 2.9 GB/s，1D f32 K=10M 32t）——clang `-O3` 会把标量循环
  自动向量化为同样低 MLP 的 SVE gather 指令（独立探针：auto-vec 0.85 GB/s，
  `#pragma clang loop vectorize(disable)` 后 0.99 GB/s，仍 < ATen 2.9）。
  920F 的 `svld1_gather_u32index` 内存级并行度低是硬件上限（kernels.md §2）。
- **维持现状**：override 仅注册 C++ op（每次调用 ~13µs 额外开销，S1 4ms 场景
  <0.5%），慢场景如实记录为硬件/算法边界，不做特殊回退。

## 8. 与 torch.compile 的关系

`torch.compile` 默认会把 `aten::index` 内联成 Inductor 自己生成的 kernel，运行时
不再有 dispatcher 调用，**CPU 覆盖不生效**（实测慢 2.6x）。已验证强制 fallback
可让 compile 重新走回本算子（1.07-1.3x）；机制、缓存/私有 API/FakeTensor 坑与
opt-in 落地计划见 [torch-compile.md](torch-compile.md)。部署时若模型使用
`torch.compile`，需注意这一点。
