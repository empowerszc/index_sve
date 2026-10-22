# index_sve — 高性能 CPU 高级索引算子（SVE512）

为鲲鹏 920F（Kunpeng 920F, aarch64, SVE2 VL=512-bit）实现的 `aten::index.Tensor`
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
- **优化轮（T0-T11）**：自适应并行粒度 `grain.h`；M1 单索引大块
  `gather_rowcopy_raw` skip-linearize；T4 小值域桶化（值域 ≤2048 查表）；
  T3 `gather_fused` 融合单遍；int32 linearize 直接写 int32；T5 1 线程单遍；
  T8 mask 单遍压实；T9 小 K（≤4096）精简路径；T10 `torch.compile` opt-in；
  T11 线程数同步 + `gather_direct` A==1 紧凑循环 + 2B/1B/complex 单遍 +
  非连续 stride 正确性修复 + 1 线程 A>=4 恢复 raw rowcopy。
- 并行：`at::parallel_for`（`adaptive_grain`）。**必须 `-fopenmp` 编译**。
  扩展链接 BiSheng libomp、ATen 链接 libgomp；T11 起入口
  `sync_omp_threads()` 使 `torch.set_num_threads(t)` 对扩展生效（此前只改
  libgomp，扩展实际用 libomp 默认 team，这也是旧线程数据失真的原因）。
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
├── tests/                # 102 测试（smoke/normalize/correctness/errors/sve/model/index_out/mask/smallk/compile）
├── benchmarks/           # bench_index.py（原始矩阵）、bench_pinned.py/bench_gate.py（定点核复测）
│   │                     # bench_t1.py/bench_t2.py/bench_ai4s.py（优化轮专项）、bench_regression.py（回归扫描）
│   │                     # bench_threads.py（全线程矩阵 1-36t）
│   │                     # bench_smallk_real.py / bench_smallk_driver.py（小 K 两进程对拍）
│   └── RESULTS.md        # 完整基准报告（§1-§14 历史；§15 T11 修正线程矩阵）
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

鲲鹏 920F，node 5（38 核），T11 构建，aten/hpc 同进程交替中位数。完整 31 场景
× 7 线程矩阵见 `benchmarks/RESULTS.md` §15，逐 case 分析见
`docs/degradation-analysis.md`。

| 场景 | 1 线程 | 36 线程 |
|---|---|---|
| S1 spconv features[inp_idx] C=64 | **4.46x** | 8.37x |
| P1 h[src] i64 C=64 | **4.42x** | 5.96x |
| P2 pos[src] A=3 | **2.05x** | 8.73x |
| P3 x[mask] C=16 | **2.23x** | 4.20x |
| S4 x[perm] f64 | **1.26x** | 1.01x |
| `x[:,i]` C=64 | **2.02x** | 1.87x |
| rowcopy A=4 / A=16 | **2.41x / 3.58x** | 2.10x / 5.80x |
| 1D f32/f64 K=2M | 1.07x / 1.26x | 2.17x / 1.51x |
| half / bf16 | 1.53x / 1.35x | 2.40x / 2.05x |

- **RowCopy 是主要收益**（1t 2.4-4.5x，2-16t 3-4.6x）：SVE `ld1/st1` 整块拷贝，
  正中 spconv 规则表 / 图消息路径。
- **小 K `x[idx]`**（K≤1024，两进程真实路径）**1.31-2.03x**；K=4096 约 0.9x；
  K≥8192 的高线程受"每线程字节数太小"影响 0.6-0.9x（绝对时间仅几十 µs）。
- **仍 <1x 的场景**：非连续 mask 0.84-0.98x（T8 只支持连续 mask）、
  int8/uint8/complex 0.9-1.0x（64B 行放大，aten 同为标量）、S4 8/16t
  0.87/0.76x 与小 K 高线程（小规模强扩展），均已在 degradation-analysis.md 定位。
- **4B 大 K 随机 pure gather 的硬限制**：64B cache line 放大 + DRAM 延迟；
  同构建 A/B 实测 SVE 4B 与标量持平（8MB 1.85 vs 1.86 ms、256MB 7.46 vs
  7.45 ms），**不是 SVE 指令 MLP 缺陷**（RESULTS §8.4；早期 0.75 vs 3.4 GB/s
  探针归因已撤回）。小值域由 T4 桶化规避行放大；f64 8B 路径不受影响。
- **线程数**：T11 起 `torch.set_num_threads(t)` 对扩展生效（入口同步 libomp
  ICV）。旧 RESULTS §8.3/§10 的 2-36t 数据因两套 OpenMP 运行时失效，已作废。
- 共享 608 线程箱噪声大：608 列只作参考，**以 node 5 定点矩阵为准**。

## 已知限制 / 注意事项

- torch 2.10 限制：同进程内不允许第二个 Python CPU 覆盖 `aten::index.Tensor`
  （`RuntimeError: already a kernel registered from python`）。这是默认覆盖设计
  的固有约束。
- 覆盖是一次性的全局副作用：任何 import index_sve 的进程都会改掉全局
  `x[...]` 行为。如不希望全局生效，用 `HPC_INDEX_NO_OVERRIDE=1`。
- 索引张量必须在 CPU（meta/CUDA 索引抛 aten 同文错误，不再段错误）；`out=`
  变体（`aten::index.Tensor_out`）也已覆盖。详见 semantics.md / optimizations.md §11。
- **`torch.compile` 下默认不生效**：Inductor 会把 `aten::index` 内联成自己的
  kernel，绕过 dispatcher 覆盖（实测慢 2.6x）。设 `HPC_INDEX_COMPILE=1`（在
  `import index_sve` 前）可 opt-in：注册 fake、把 lowering 永久设为 fallback、
  切换独立 inductor cache 子目录，运行期回到 dispatcher。默认关闭；细节与限制见
  `docs/torch-compile.md` §6。
- 语义细节、内核细节、移植指南见 `docs/`（目录见 `docs/index.md`；
  "如何劫持 aten::index" 见 `docs/override-mechanism.md`）。

## 测试

```bash
cd index_sve
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py \
         tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py \
         tests/test_compile_optin.py; do python $t; done
```

全部通过 = 102/102（覆盖激活态；test_correctness 25、test_errors 6、test_model 9、
test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
test_smallk 7、test_compile_optin 3）。
覆盖下测试是自洽的（`x[i]` 也走
本算子），值级真对拍用两进程法（无覆盖进程生成 true-aten 参考，覆盖进程比对）：

```bash
python tests/verify_against_aten.py --mode both    # 44/44，参考写 agent_space/（git-ignored）
```
