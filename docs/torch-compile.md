# torch.compile 与本算子的关系（能力与 opt-in 落地）

> **一句话结论**：`torch.compile` 默认会把 `aten::index` **内联成 Inductor 自己生成的
> kernel**，根本不在运行时调用任何 index 内核，所以本算子的 CPU 覆盖**不生效**
> （实测慢 2.6x）。把该 op "钉"回成运行时调用（Inductor fallback）后，覆盖重新
> 生效，差距收敛到 1.07-1.3x。方案已验证，**尚未落地**为默认能力。

本文记录 2026-10 的 spike 结论（torch 2.10.0+cpu，pin node 5，8 线程）。

## 1. 到底有几份实现

`x[i]` 在不同执行方式下，实际跑的代码是**三份不同的实现**：

| 执行方式 | 实际执行索引的代码 | 2M 元素 f32 |
|---|---|---|
| eager，未 import `index_sve` | ATen 原生 CPU 内核（`index_stub` / TensorIterator） | ≈8.9 ms* |
| eager，import `index_sve` | **本算子 SVE 内核**（dispatcher CPU 覆盖） | **2.01 ms** |
| `torch.compile`（默认） | **Inductor 生成的 C++ kernel**（`cpp_fused_index_0`） | 5.30 ms |
| `torch.compile` + 强制 fallback | 运行时回到 dispatcher → **本算子 SVE 内核** | **2.57-2.67 ms** |

\* 真 aten 参考需在**另一个进程**用 `HPC_INDEX_NO_OVERRIDE=1` 跑 `aten::index.Tensor_out`
测得（T6 后 `Tensor_out` 也被覆盖）；functional 变体数值可能略有差异。

规模放大后 compile 的固定开销被摊薄（compiled-fallback / eager）：

| N | eager | compiled+fallback | 比值 |
|---|---|---|---|
| 1k | 0.067 ms | 0.282 ms | 4.2x |
| 100k | 0.172 ms | 0.552 ms | 3.2x |
| 2M | 2.01 ms | 2.57 ms | 1.28x |
| 8M | 8.00 ms | 8.56 ms | **1.07x** |

小 N 的差距主要不是内核，而是 `torch.compile` 的固定调用开销：同一台机器上
`torch.compile(lambda a: a.clone())` 相对 eager 就有约 **0.3ms** 的固定增量。

## 2. 为什么 compile 下覆盖不生效

关键区别是 **"运行时内核" vs "编译期代码生成"**：

```
eager:
  x[i] → Tensor.__getitem__ → dispatcher → aten::index.Tensor CPU 键
                                             └─ 已被覆盖 → 本算子内核

torch.compile:
  Dynamo 追踪 x[i] → FX 图节点 aten::index.Tensor
        │
        ├─ AOTAutograd：不分解（aten.index 不在 core_aten_decompositions）
        │
        └─ Inductor lowering：torch/_inductor/lowering.py:3878
               @register_lowering(aten.index)
               def index(...): return index_impl(...)
               └─ 生成 cpp_fused_index_0（自带 OOB 检查的循环）
```

也就是说，编译产物里**没有 `aten::index.Tensor` 这个调用**，只有 Inductor 生成的
循环。dispatcher 覆盖只在"真的执行 aten op"时才有机会介入，因此完全被绕开。

> 注意：compile 默认也不是"跑 aten 原生内核"，而是 Inductor 的**第三份实现**。
> 它比 true aten 快、但比本算子慢。

## 3. 让 compile 用回本算子：强制 fallback

Inductor 对单个 op 提供了 `force_fallback`（也是其单测使用的机制）：

```python
from torch._inductor.lowering import force_fallback

with force_fallback(torch.ops.aten.index.Tensor):
    compiled = torch.compile(fn)
    compiled(x, i)          # 首次调用必须在 with 内，编译在此发生
```

效果：Inductor 不再生成 `index_impl`，而是发出一个**运行时调用**
`aten::index.Tensor`。运行时链路变成：

```
compiled 产物 → aten::index.Tensor → dispatcher → CPU 键
                                                   ├─ 装了 index_sve → 本算子内核
                                                   └─ 没装         → true aten
```

因此它**天然兼容未覆盖场景**（没装覆盖时退回 true aten，正确性不变），也不会像
"自定义 op 分解"路线那样破坏 autograd——反向仍走 aten 自己的 `index_backward`。

## 4. 坑（落地前必须处理）

1. **Inductor 磁盘缓存会让补丁静默失效。**
   第一次实验时 `force_fallback` 看起来无效（仍 5.3ms），原因是图哈希命中了之前
   编译好的慢 artifact；lowering 注册**不参与**图哈希。`config.cache_key_tag`
   只影响 autotune cache，不影响 `FxGraphCache`。
   解决：覆盖生效时使用**独立的 inductor cache 目录**（或在文档中要求清缓存）。
2. **私有 API**：`torch._inductor.lowering.lowerings` / `force_fallback` 非公开，
   跨 torch 版本需要守卫（参见 `porting.md` 的 2.10 绑定）。
3. **FakeTensor/meta 缺失**：直接 `torch.compile(lambda a,i: torch.ops.hpc_index.index(a,[i]))`
   会报 `Dynamo failed to run FX node with fake tensors`。自定义 op 路线必须给
   `hpc_index::index` / `index_out` 注册 `torch.library.register_fake`（§6 已实现，
   仅 `HPC_INDEX_COMPILE=1` 时注册；整数下标支持，mask 因输出长度数据相关不覆盖）。
4. **小 N 固定开销**：compile 调用本身约 0.3ms；K≲100k 时即便修好也不如 eager，
   这类形态建议直接不 compile 或让模型整体编译摊薄。

## 5. 备选路线与否决原因

| 路线 | 结果 | 结论 |
|---|---|---|
| 默认 compile | Inductor 内联，覆盖失效，慢 2.6x | 不可用 |
| `force_fallback(aten.index.Tensor)` | 运行时回到覆盖，1.07-1.3x | **推荐** |
| 直接把 `x[i]` 编译成 `torch.ops.hpc_index.index` | 需先补 `register_fake`（§6 已提供）；且自定义 op 需 autograd 注册 | 未采用（fallback 更简单） |
| 注册 `aten.index.Tensor` decomposition → 自定义 op | AOT 阶段分解，训练需补 autograd，且 CPU-only 会误伤其他设备 | 不推荐 |

## 6. opt-in 落地（已实现，`HPC_INDEX_COMPILE=1`）

`index_sve/compile_support.py`，只在覆盖生效且 `HPC_INDEX_COMPILE=1` 时启用：

```bash
HPC_INDEX_COMPILE=1 python your_model.py
```

`import index_sve` 时依次执行：

1. `torch.library.register_fake("hpc_index::index"/"index_out")`：整数下标按
   broadcast 推形状，`index_out` 先 `resize_`；bool/byte mask 输出长度依赖
   `nonzero`，fake 直接抛 `NotImplementedError`（compile 的 mask 走 `aten::index`
   fallback，不用这个 fake）。
2. 版本守卫后 `register_lowering(aten.index.Tensor)(fallback_handler(...))`：
   永久替换 lowering，运行期回到 dispatcher，命中 CPU 覆盖。
3. 把 `TORCHINDUCTOR_CACHE_DIR` 指向 `<原 cache_dir>/hpc_index`：缓存 key 不含
   lowering 注册信息，必须与未打补丁的 artifact 隔离；重复调用幂等。

未开覆盖（`HPC_INDEX_NO_OVERRIDE=1`）时 `HPC_INDEX_COMPILE=1` 直接报错，避免
"以为启用了、其实 fallback 到原生 aten"的静默混用。默认关闭：monkeypatch 私有
API 有版本风险，且会改变用户的编译缓存行为，不适合作为隐式副作用。

验证：`tests/test_compile_optin.py`（3 例，子进程隔离）覆盖默认关闭、fake
形状、整数/掩码两条 compile 路径的数值等价、profiler 里 `aten::index` 命中、
cache 目录后缀、以及与 `HPC_INDEX_NO_OVERRIDE` 的冲突报错。

## 7. 复现

```bash
# 关闭 inductor 缓存，确保 force_fallback 生效
TORCHINDUCTOR_CACHE_DIR=/tmp/tind_hpc \
python - <<'PY'
import os, time, statistics, torch
os.sched_setaffinity(0, set(range(190, 228))); torch.set_num_threads(8)
import torch._inductor.config as c; c.force_disable_caches = True
import index_sve
from torch._inductor.lowering import force_fallback

def med(fn, n=20):
    for _ in range(5): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3

N = 2_000_000
x = torch.randn(N); i = torch.randint(0, N, (N,))
ref = x[i]
def f(a, idx): return a[idx]

torch._dynamo.reset()
c1 = torch.compile(f); assert torch.equal(c1(x, i), ref)
t1 = med(lambda: c1(x, i))

torch._dynamo.reset()          # 必须重置，避免复用上面缓存的编译结果
c2 = torch.compile(f)
with force_fallback(torch.ops.aten.index.Tensor):
    assert torch.equal(c2(x, i), ref)      # 首次调用在 with 内，编译发生在这里
t2 = med(lambda: c2(x, i))

print(f"eager={med(lambda: x[i]):.3f}ms default={t1:.3f}ms fallback={t2:.3f}ms")
PY
```

相关实现位置：`torch/_inductor/lowering.py:3878`（`aten.index` lowering）、
`torch/_inductor/lowering.py:2250`（`make_fallback`）、
`torch/_inductor/lowering.py:7667`（`force_fallback`）。
