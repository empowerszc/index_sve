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
- **优化轮（T0-T5）**：自适应并行粒度 `grain.h`（按 K 与线程数缩放 chunk）；
  M1 单索引大块 `gather_rowcopy_raw` skip-linearize；T4 小值域桶化
  （`bucket_gather_pure` / `bucket_gather_pure_raw`，值域 ≤2048 查表）；T3
  `gather_fused` 融合单遍（A<4 或非连续小块，不物化 idx）；int32 linearize
  直接写 int32（不再经 int64 临时缓冲）；**T5 1 线程单遍修复**（`gather_single_1d`
  单索引大值域纯 gather、1 线程单索引 A>1 走 fused、fused/linearize 预提取索引
  属性 + nb==1 全长连续快路径）。
- 并行：`at::parallel_for`（`adaptive_grain`）。**注意必须 `-fopenmp` 编译**，
  否则 `parallel_for` 退化为串行（历史上曾导致 B==1 大 K 性能塌缩）。
- 条件回退标量 `dispatch_scalar`：2B/1B 元素（half/bf16/i16/i8/uint8/bool）、
  complex、f32 且 `numel>=2^31`（防 int32 越界）、rowcopy 的 A<4、非连续
  after-col 等所有 SVE 分支之外的输入类。

## 目录结构

```
index_sve/
├── setup.py              # BiSheng clang19, -march=armv8-a+sve2 -msve-vector-bits=512 -O3 -fopenmp
├── index_sve/
│   ├── __init__.py       # import 时安装 aten::index.Tensor / .Tensor_out CPU 覆盖（HPC_INDEX_NO_OVERRIDE=1 可跳过）
│   ├── register_aten.py  # 兼容层；实际覆盖由 op.cpp 的 TORCH_LIBRARY_IMPL(aten,CPU) 完成
│   ├── op.cpp            # index_impl 入口 + pybind（debug_spec/linearize_spec/idx_dtype_choice）
│   ├── normalize.{h,cpp} # normalize_index：mask/broadcast/transpose_to_front，对齐 ATen make_info
│   ├── linearize.{h,cpp} # 线性化 + 越界检查，int32 保持（直接写 int32）
│   ├── grain.h           # 自适应并行粒度（T0）
│   ├── gather.{h,cpp}    # SVE gather/rowcopy/scalar + gather_direct + M1 raw / T3 fused / T4 桶化
├── tests/                # 75 测试（smoke/normalize/correctness/errors/sve/model）
├── benchmarks/           # bench_index.py（原始矩阵）、bench_pinned.py/bench_gate.py（定点核复测）
│   │                     # bench_t1.py/bench_t2.py/bench_ai4s.py（优化轮专项）、bench_regression.py（回归扫描）
│   │                     # bench_threads.py（全线程扫描 1-38t）
│   └── RESULTS.md        # 完整基准报告（§1-§6 原测，§8 定点复测，§9 AI4S 优化轮，§10 全线程扫描）
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
| P3 x[mask] C=16 | **2.1x** | **4.6x** |
| S4 x[perm] f64 | 0.94x | 0.53x |

- 可靠收益在 **RowCopy**（行拷贝，模型 spconv/图消息主要形态）与 S4 8B 大
  随机 gather（608 线程可达 5.6x）；1D f64 小 K 1.6-1.8x（B==1 K 分块 +
  `-fopenmp` 修复）。
- **低线程收益最强**（§8.3 线程扫描）：RowCopy 在 2-4 线程即饱和单节点内存
  带宽（S1 2t **33.7x**、P1 2t **30.5x**、P2 2t **19.7x**），S4 8B 2t 20.7x——
  算子用少量核心即达峰值，而 aten 标量 gather 需 ~30+ 线程。
- **AI4S workload 优化轮（T0-T4，RESULTS §9，`bench_ai4s.py` M1-M6）**：M1
  weight.view[ki]（最热形态）1 线程 **8-10x** / 8 线程 1.3-2.7x；M2 feat[r]
  1-8 线程 2.6-4.0x；M3 keys[order] 小值域查表 1 线程 1.4-2.6x、32 线程 8-12x
  （优化前 0.2x）；M4 feat[inverse] 3.6-4.8x。
- **T5 1 线程回归修复（`bench_regression.py`，bd21485）**：修复 1 线程下
  0.6-0.9x 的劣化形态（linearize 两遍模型 + SVE 单线程低 MLP）：1D pure
  f32/f64 K=5M 0.75x/0.71x → **1.07x/1.24x**；x[i,j] 多索引 0.66x → **1.21x**；
  rowcopy A=4 短行 0.87x → **1.39x**。2+ 线程本就全面占优（1D pure 2t ~20-33x、
  x[i,j] 2t 5.98x、rowcopy 8t 8x）。mask 1 线程 ~1.0x 噪声级。
- **全线程扫描（RESULTS §10，`bench_threads.py`，1/2/4/8/16/32/36/38）**：当前
  构建核心场景**全部 8 档线程 ≥1.06x**（S1 2-4t 31-40x、P2 2t 66x、1D f64 2t
  19-23x、1D f32 K=10M 2-4t 14-17x），不再有历史 §1/§2 的 <1x 格——那些是共享箱
  608 全核模式的噪声产物。P3 mask 在 T8 前最弱（1-1.9x，nonzero 两段式）；
  T8 单遍融合后 1t 2.1x、2-32t 为新路径旧路径的 3.6-5.4x（RESULTS §12）。
- **已知硬件限制**：4B（f32/i32）大 K 随机 pure gather 的 SVE 4B gather 指令
  在 920F 上 MLP 极低（独立探针：38 线程仅 ~0.75 GB/s，标量 3.4 GB/s）——这是
  硬件指令特性（RESULTS §8.2，`docs/fallback-analysis.md` §3.1）。当前构建相对
  aten 已无 <1x 格（§10 全线程扫描 1D f32 K=10M 各线程 1.1-16.6x），但探针仍是
  硬件限制的确定性陈述；小值域由 T4 桶化规避。f64 8B 路径不受影响。设计定位是
  模型 gather 密集的行/8B/小值域路径的定向加速器，非通用 1D 索引加速。
- 共享 608 线程箱噪声大：§1 的 608 列是噪声下限，**以 §8/§10 定点与单节点数据
  为准**（全线程矩阵见 RESULTS §10、`bench_threads.py`）。

## 已知限制 / 注意事项

- torch 2.10 限制：同进程内不允许第二个 Python CPU 覆盖 `aten::index.Tensor`
  （`RuntimeError: already a kernel registered from python`）。这是默认覆盖设计
  的固有约束。
- 覆盖是一次性的全局副作用：任何 import index_sve 的进程都会改掉全局
  `x[...]` 行为。如不希望全局生效，用 `HPC_INDEX_NO_OVERRIDE=1`。
- 索引张量必须在 CPU（meta/CUDA 索引抛 aten 同文错误，不再段错误）；`out=`
  变体（`aten::index.Tensor_out`）也已覆盖。详见 semantics.md / optimizations.md §11。
- **`torch.compile` 下默认不生效**：Inductor 会把 `aten::index` 内联成自己的
  kernel，绕过 dispatcher 覆盖（实测慢 2.6x）。已验证强制 fallback 可恢复到
  1.07-1.3x；方案、坑与 opt-in 计划见 `docs/torch-compile.md`（尚未默认启用）。
- 语义细节、内核细节、移植指南见 `docs/`（目录见 `docs/index.md`；
  "如何劫持 aten::index" 见 `docs/override-mechanism.md`）。

## 测试

```bash
cd index_sve
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py; do python $t; done
```

全部通过 = 99/99（覆盖激活态；test_correctness 25、test_errors 6、test_model 9、
test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
test_smallk 7）。
覆盖下测试是自洽的（`x[i]` 也走
本算子），值级真对拍用两进程法（无覆盖进程生成 true-aten 参考，覆盖进程比对）：

```bash
python tests/verify_against_aten.py --mode both    # 40/40，参考写 agent_space/（git-ignored）
```
