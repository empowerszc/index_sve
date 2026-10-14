# index_sve — 高性能 CPU 高级索引算子（SVE512）

为昇腾 920F（HiSilicon 920F, aarch64, SVE2 VL=512-bit）实现的 `aten::index.Tensor`
语义替代算子。**`import index_sve` 后，进程中所有 CPU 张量的高级索引调用
（`x[idx]`、`x[:, i]`、`x[i, j]`、bool mask、N-D 整数下标等）自动改走本算子**，
无需修改任何调用点。语义与 torch 2.10 的 ATen 参考实现逐字节对齐，autograd
backward 不受影响。

## 设计要点

- 三段流水线：**归一化 normalize**（对齐 ATen `make_info`，保留 int32 下标）→
  **线性化 linearize**（一次遍历完成越界检查 + 负数归一 + 元素偏移）→
  **gather**（SVE512 向量 gather / 行拷贝 / 标量回退）。
- SVE512 路径：8B 元素（f64/i64）`svld1_gather_u64index_*`，4B 元素（f32/i32）
  `svld1_gather_u32index_*`，连续 A 元素块 `svld1/st1` 行拷贝，均带 `pf+16`
  预取（`svprfd/svprfw_gather_u32index`）。
- `K<1024` 快速路径 `gather_direct`：跳过线性化，直接读原始下标 + 内联检查
  （等价 ATen `Indexer`），支持 N-D 整数下标（broadcast_shape 混合进制解码）。
- 并行：`at::parallel_for`（GRAIN=4096）。**注意必须 `-fopenmp` 编译**，否则
  `parallel_for` 退化为串行（历史上曾导致 B==1 大 K 性能塌缩）。
- 条件回退标量 `dispatch_scalar`：2B/1B 元素（half/bf16/i16/i8/uint8/bool）、
  complex、f32 且 `numel>=2^31`（防 int32 越界）、rowcopy 的 A<16、非连续
  after-col 等所有 SVE 分支之外的输入类。

## 目录结构

```
index_sve/
├── setup.py              # BiSheng clang19, -march=armv8-a+sve2 -msve-vector-bits=512 -O3 -fopenmp
├── index_sve/
│   ├── __init__.py       # import 时安装 aten::index.Tensor CPU 覆盖（HPC_INDEX_NO_OVERRIDE=1 可跳过）
│   ├── register_aten.py  # torch.library.impl("aten::index.Tensor","CPU", torch.ops.hpc_index.index)
│   ├── op.cpp            # index_impl 入口 + pybind（debug_spec/linearize_spec/idx_dtype_choice）
│   ├── normalize.{h,cpp} # normalize_index：mask/broadcast/transpose_to_front，对齐 ATen make_info
│   ├── linearize.{h,cpp} # 线性化 + 越界检查，int32 保持
│   ├── gather.{h,cpp}    # SVE gather/rowcopy/scalar + K<1024 gather_direct
├── tests/                # 63+ 测试（smoke/normalize/correctness/errors/sve/model）
├── benchmarks/           # bench_index.py（原始矩阵）、bench_pinned.py/bench_gate.py（定点核复测）
│   └── RESULTS.md        # 完整基准报告（§1-§6 原测，§8 定点复测）
└── docs/                 # 本系列文档
```

## 快速开始

环境（本机 920F，BiSheng clang 19.1.7 + torch 2.10.0+cpu）：

```bash
source ~/.bashrc && conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=/home/z00889957/miniconda3/envs/torch210_env/lib/python3.12/site-packages/torch/lib:/home/envs/HPCKit_26.1.RC1/HPCKit/26.1.RC1/compiler/bisheng/lib:$LD_LIBRARY_PATH
```

> **运行时契约**：扩展链接了 BiSheng 的 `libomp.so`，`LD_LIBRARY_PATH` 必须含
> BiSheng lib 目录（上例最后一个路径），否则 `import` 报 `libomp.so: cannot open`。

构建（勿用 ninja）：

```bash
cd index_sve
python setup.py build_ext --inplace
```

使用：

```python
import torch
import index_sve          # 之后 x[idx] 等全部走本算子（首次 import 有一次性 UserWarning，正常）

x = torch.randn(50_000, 64)
i = torch.randint(0, 50_000, (1_000_000,), dtype=torch.int32)
y = x[i]                  # SVE 行拷贝路径，1 线程实测 ~4x
```

只想要算子本体、不覆盖 aten：

```bash
HPC_INDEX_NO_OVERRIDE=1 python your_script.py   # import 不再改 aten::index.Tensor
# 或直接调 torch.ops.hpc_index.index(x, [i])
```

## 基准结论（摘要）

定点核（quiet NUMA node，`bench_gate.py`，3 轮中位数）：

| 场景 | 1 线程 | 38 线程 |
|---|---|---|
| S1 spconv features[inp_idx] C=64 | **3.96x** | 1.72x |
| P1 h[src] i64 C=64 | **3.86x** | 1.94x |
| P2 pos[src] A=3 | **1.76x** | 0.86x |
| P3 x[mask] C=16 | 1.11x | 0.35x |
| S4 x[perm] f64 | 0.94x | 0.53x |

- 可靠收益在 **RowCopy**（行拷贝，模型 spconv/图消息主要形态）与 S4 8B 大
  随机 gather（608 线程可达 5.6x）；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复）。
- **低线程收益最强**（§8.3 线程扫描）：RowCopy 在 2-4 线程即饱和单节点内存
  带宽（S1 2t **33.7x**、P1 2t **30.5x**、P2 2t **19.7x**），S4 8B 2t 20.7x——
  算子用少量核心即达峰值，而 aten 标量 gather 需 ~30+ 线程。
- **已知硬件限制**：4B（f32/i32）大 K 随机 pure gather 在 920F 上不敌标量
  （SVE 4B gather 指令 MLP 极低，38 线程仅 ~0.75 GB/s，标量 3.4 GB/s）——因此
  1D f32/i32 大 K 会 <1x（仅 2-4 线程窗口有 1.2-1.7x 收益）。f64 8B 路径不受
  影响。设计定位是模型 gather 密集的行/8B 路径的定向加速器，非通用 1D 索引加速。
- 共享 608 线程箱噪声大：§1 的 608 列是噪声下限，**以 §8 定点数为准**。

## 已知限制 / 注意事项

- torch 2.10 限制：同进程内不允许第二个 Python CPU 覆盖 `aten::index.Tensor`
  （`RuntimeError: already a kernel registered from python`）。这是默认覆盖设计
  的固有约束。
- 覆盖是一次性的全局副作用：任何 import index_sve 的进程都会改掉全局
  `x[...]` 行为。如不希望全局生效，用 `HPC_INDEX_NO_OVERRIDE=1`。
- int32 线性化会临时分配 K×8B int64 缓冲（K=10M 时 80MB）；8B 线性化并行。
- 语义细节、内核细节、移植指南见 `docs/`（目录见 `docs/index.md`；
  "如何劫持 aten::index" 见 `docs/override-mechanism.md`）。

## 测试

```bash
cd index_sve
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py; do python $t; done
```

全部通过 = 64/64（覆盖激活态）。覆盖下的正确性对拍采用两进程法：无覆盖进程
生成 true-aten 参考，覆盖进程比较（`agent_space/sdd-index-sve/task10_ref_*`）。
