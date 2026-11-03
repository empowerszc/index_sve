# 集成 / 运行时契约

本文档说明本算子如何接入 torch 运行时、环境要求，以及集成时要注意的全局
副作用。

> **TL;DR**：`import index_sve` 即默认接管所有 CPU `aten::index.Tensor`；
> 不想被接管就设 `HPC_INDEX_NO_OVERRIDE=1`（必须在 import 前）。`torch.compile`
> 默认绕过所有运行时内核，要 opt-in 设 `HPC_INDEX_COMPILE=1`。两个变量互斥。
>
> **名词**：**dispatch key** = 调度器选内核的键（CPU/CUDA/Autograd...）；
> **dlopen / 扩展加载** = `import` C++ 扩展时执行静态注册；**boxed fallback**
> = 用通用 `IValue` 参数包转调内核的慢路径；**ICV** = OpenMP 线程数配置，
> libomp 与 libgomp 各一份（T11 已同步）。

## 0. 30 秒检查清单

1. 只需 `import index_sve`；确认日志里出现一次预期告警（见 §4）。
2. 多线程程序不要只调 `torch.set_num_threads` 就假设扩展已跟随——T11 起
   `index_impl` 会主动同步 `at::get_num_threads()`；直接用 `torch.ops.hpc_index.*`
   时也会同步。但直调 `torch.ops` 无 autograd 公式：requires_grad 输入的梯度
   会被 fallback 静默丢弃（下游 backward 不报错但梯度不回传）——训练场景
   务必用 `x[...]`（aten::index 的 AutogradCPU 链完整；行为由
   `tests/test_autograd.py` 钉住）。
3. 用 `torch.compile` 时先决定：要 compile 的图优化，还是要本算子覆盖
   （二选一，见 §8 与 `torch-compile.md`）。

## 1. 默认覆盖 aten::index.Tensor（A6）

`index_sve/index_sve/__init__.py`：

```python
from . import _hpc_index            # dlopen 时注册 torch.ops.hpc_index（+ CPU 覆盖）
from .register_aten import install as _install_aten
import torch, os
OVERRIDE_ACTIVE = os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") != "1"  # 供测试判断
if OVERRIDE_ACTIVE:
    _install_aten()                 # 兼容旧调用；实际覆盖在 C++ dlopen 时完成
if os.environ.get("HPC_INDEX_COMPILE", "0") == "1":
    from .compile_support import enable as _enable_compile
    _enable_compile()               # opt-in：fake + aten fallback + 独立 cache（T10）
```

覆盖实际由 C++ 扩展在 dlopen 时完成（`op.cpp`）：

```cpp
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  const char* e = getenv("HPC_INDEX_NO_OVERRIDE");
  if (e == nullptr || std::string(e) != "1") {
    m.impl("index.Tensor", &index_impl);
    m.impl("index.Tensor_out", &index_out_aten);
  }
}
```

`register_aten.install()` 现为空实现（兼容旧调用）。直接 C++ 注册避免了 Python
覆盖层每调用的解释器往返（小 K 实测 ~10-20us）。效果：**`import index_sve`
之后，进程中所有 CPU 张量的高级索引（`x[idx]`、`x[:, i]`、`x[i, j]`、bool
mask、N-D 下标）都改走本算子**，无需改调用点。已验证：
- dispatcher dump 证实 CPU key 指向扩展注册的 C++ kernel；
- 覆盖后 autograd backward 正常（`IndexBackward` 独立注册，不受 CPU kernel 影响）。

## 2. 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `HPC_INDEX_NT` | 0 | 非临时 gather/store 开关（`!= "0"` 即开；opt-in，默认关） |
| `HPC_INDEX_NO_OVERRIDE` | 0 | `=1` 时 import **跳过** aten::index 覆盖（基准/对拍用） |
| `HPC_INDEX_COMPILE` | 0 | `=1` 时启用 `torch.compile` 支持（fake + aten fallback + 独立 cache 目录，T10；须在 import 前设置） |

`HPC_INDEX_NO_OVERRIDE` 的用途：
- 只想用 `torch.ops.hpc_index.index` 本体、不改全局 `x[...]` 的部署（该形态
  无 autograd 公式，梯度静默不回传，仅推理/对拍适用——见 §0 条目 2）；
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

## 4. 一次性告警（正常，勿压制）

首次 import（dlopen `_hpc_index`）时 dispatcher 打印一次：

```
Warning: Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
  dispatch key: CPU
  previous kernel: registered at .../BatchRulesScatterOps.cpp
       new kernel: registered at index_sve/op.cpp
```

**预期且无害**，故意不压制（用户应知晓全局覆盖已生效）。

## 5. 全局副作用与限制

- **覆盖是进程级全局的**：任何 import index_sve 的进程，其 `x[...]` 全部改道。
  这是设计目标（零改动接入），但部署时需明确告知所有调用方。
- **重复注册**：C++ 同 key 重复注册会替换旧 kernel 并再打印一次告警；
  `importlib.reload(index_sve)` 不会破坏覆盖。
- `HPC_INDEX_NO_OVERRIDE=1` 在 dlopen 前设置时，C++ 注册整体跳过。
- `aten::index.Tensor_out` 也已覆盖（T6）：`out=` 调用同样走本算子；
  QuantizedCPU 等其它 dispatch key 不受影响。

## 6. 部署建议

- 模型/服务入口最先 import index_sve（保证所有索引早于计算改道）；
- 用 `HPC_INDEX_NT=1` 试验大输出场景（默认关，需实测收益）；
- 提供环境的 `setvars.sh` + LD_LIBRARY_PATH 到部署脚本；
- 用 `torch._C._dispatch_dump_table()` 验证 CPU key 已到覆盖（调试用）。

### 6.1 宿主程序已是多进程/多线程（实测 2026-10-27，节点 5 定点）

**结论：没有算子层面的冲突。** 覆盖、路由、OpenMP team、grain 全部是进程内
状态，多进程天然隔离——本项目所有正式数字本就在 5+ 用户共享箱（事实上的
多进程环境）上测得。T11 起每次内核入口把 torch 线程数同步给扩展的 libomp
（`sync_omp_threads`），所以 `torch.set_num_threads` / `OMP_NUM_THREADS` 的
线程预算对扩展真实生效。需要管理的是**超订**（PyTorch 通用问题，非本算子
特有，aten 同样中招）：

| 宿主形态 | 结论 |
|---|---|
| 多进程（DataLoader / torchrun / 服务多 worker） | 无共享状态，仅硬件竞争；守则：进程数 × 每进程线程数 ≤ 核数 |
| 进程内 k 线程并发调 `x[idx]` | 有超订代价（下表）；把真实预算告诉 torch——`set_num_threads(≈ 核数/k)`，扩展会跟随 |
| **宿主 C++ OpenMP region 内调 torch** | **唯一应避免的形态**：ATen 的嵌套守卫按 runtime 生效——宿主 region 在 libgomp、本算子在 libomp，嵌套关系跨 runtime 不可见 → 静默超订（宿主线程数 × nt）；同 runtime 则被守卫强制串行。必须在 region 内调用时先 `set_num_threads(1)` |
| fork 的 DataLoader worker | 继承注册；OpenMP pool 经 atfork 处理，常规用法安全（spawn 需 worker 侧 import，summary.md §4.6） |

实测（`x[idx]` 200k 行 × C=64 f32，共 24 次调用，`bench_embed_threads.py`）：

| 配置 | 结果 | 读法 |
|---|---|---|
| nt=36，单线程顺序 | 1.7 ms/调用 | 基线 |
| nt=36，4 宿主线程（4×36=144 线程压 38 核） | 14.6 ms/调用，**劣化 8.6x** | 超订代价；aten 同配置同现象（总时间劣化 2.60x vs 本算子 2.14x，单轮共享箱数字看趋势） |
| nt=9，4 宿主线程（4×9≈核数） | 总时间为同 nt 顺序的 0.59x | 预算说真话后，宿主并行正常工作 |
| nt=1，4 宿主线程 | 聚合加速 3.72x | 小调用/高并发宿主的正确姿势（GIL 在 op 内释放，串行内核干净叠加） |

另注：`adaptive_grain` 自适应的是**工作量 × torch 线程数**（chunk 防任务
风暴，下限 64），不是系统负载——它感知不到其他进程/线程的占用。调用大
（K·A 大、并行摊得开）优先内部并行（高 nt、宿主侧串行批处理）；调用小
（fork-join 占比高）优先宿主并行 + 低 nt（`small_*` 路径的串行阈值
`max(16384, 512×nt)` 正是为此设计的）。

## 7. 慢场景 fallback 决策（已评估，结论：不改）

针对"本算子性能差于 aten::index 时回退到原生 aten::index"的诉求，做了一轮完整
调查，结论是**不实现 fallback**。记录理由供日后复读：

- **慢场景（当前）**：只有六个家族 <1x——空 mask 4-38t 0.42-0.66x（aten 并行
  早退，绝对差 1.5-2ms）、1B 0.91-0.99x、complex 0.92-0.95x、非连续 mask
  0.79-0.98x、S4 0.77-0.92x、噪声单格；逐格清单见 RESULTS §2.9.1。31 场景 ×
  8 档共 248 格，其余 217 格 ≥1.00x。4B 大 K 随机 gather 的绝对带宽仍受 64B 行放大 +
  DRAM 延迟限制（§8.4），这是物理上限而非实现差距。
  RowCopy/8B PureGather 是可靠收益（1t S1/P1 2.89-2.98x、8t 3.0-3.3x）；
  `x[:,i]` 当前 1t 2.00x（T11 前 0.4-0.7x，A==1 紧凑循环修复）。
- **Python-wrapper fallback 不可取**：实测 `torch.library.get_kernel` +
  `with_keyset` + `call_boxed` 能正确回退且 autograd 安全（无递归），但任何
  Python 层包装会给**每次** `x[...]` 调用增加 ~45µs（小 op 13.7µs→63µs，
  S1 4ms 场景仅 +1%）。对以 RowCopy 小/中张量为收益主体的算子，这是全局性
  回归，不符合"常规标准流程"。
- **C++ 内核内 scalar 回退也追不平**：同构建 A/B（真实内核、同一随机模式）
  SVE 4B gather 与标量循环持平（src 8MB 1.85 vs 1.86 ms、256MB 7.46 vs 7.45 ms，
  8/38t）。瓶颈是 64B cache line 放大（4B 元素 16x 行放大）+ DRAM 延迟，不是
  SVE 指令 MLP；换标量或换指令都无效（kernels.md §2、RESULTS §3.7）。可行方向
  是减少行流量（T4 桶化/预取/行 gather）。
- **维持现状**：override 仅注册 C++ op（每次调用 ~13µs 额外开销，S1 4ms 场景
  <0.5%），慢场景如实记录为硬件/算法边界，不做特殊回退。

## 8. 与 torch.compile 的关系

`torch.compile` 默认会把 `aten::index` 内联成 Inductor 自己生成的 kernel，运行时
不再有 dispatcher 调用，**CPU 覆盖不生效**（该轮实测慢 2.6x）。已验证强制 fallback
可让 compile 重新走回本算子（1.07-1.3x）；已实现为 opt-in：import 前设
`HPC_INDEX_COMPILE=1`（与 `HPC_INDEX_NO_OVERRIDE=1` 冲突会报错）。机制、缓存/
私有 API/FakeTensor 坑见 [torch-compile.md](torch-compile.md) §6。
