# index_sve — 高性能 CPU 高级索引算子（SVE512）

为鲲鹏 920F（Kunpeng 920F, aarch64, SVE2 VL=512-bit）实现的 `aten::index.Tensor`
语义替代算子。**`import index_sve` 后，进程中所有 CPU 张量的高级索引调用
（`x[idx]`、`x[:, i]`、`x[i, j]`、bool mask、N-D 整数下标等）自动改走本算子**，
无需修改任何调用点。语义与 torch 2.10 的 ATen 参考实现逐字节对齐，autograd
backward 不受影响。

## 先读：术语与结果口径（30 秒）

| 名词 | 一句话解释 |
|---|---|
| **高级索引** | 用张量当下标取值，如 `x[i]`、`x[:, i]`、`x[i, j]`、`x[mask]`；底层都是 `aten::index.Tensor` |
| **RowCopy** | 一次取一整行/一段连续内存（`A>=4`），用 SVE 的 `ld1/st1` 整块搬，最快 |
| **PureGather** | 一次只取一个元素（`A==1`），随机地址，受 64B cache line 放大限制 |
| **linearize** | 把“第几行、第几列”换算成一维元素偏移，并做越界/负数检查 |
| **SVE / VL** | ARM 可伸缩向量指令 / 向量长度；本机 VL=512-bit（16×f32 或 8×f64） |
| **cache line（缓存行）** | CPU 和内存搬数据的最小单位，64B；随机取 4B 也要搬 64B |
| **ICV** | OpenMP 的“线程数”配置；libomp 和 libgomp 各有一份，互不相通 |
| **ratio（加速比）** | 本文统一用 `aten 时间 / 本算子时间`；**>1 表示比 torch 快** |

**结果怎么读**：`1.26x` = 快 26%，`0.87x` = 慢 13%。高线程（32/36t）的比值有时
很高，一部分原因是 aten 在带宽饱和后自己变慢，所以文档同时给出绝对 GB/s；
判断真实收益要看“低/中线程比值 + 绝对带宽”，而不是只看最大比值。

**看不懂的代号**（S1、P1、M1、shortq、A/B/K…）和术语，查
[`docs/glossary.md`](docs/glossary.md)——全项目唯一的名词入口（技术术语表、
场景/负载代号、形状记号、同名冲突警示）。

**建议阅读路线**：
- 只想用：本节 → [快速开始](#快速开始) → [已知限制](#已知限制--注意事项)。
- 想看性能：`benchmarks/RESULTS.md` §2 → `docs/degradation-analysis.md`
  （劣化硬件根因 §5.9；胜出根因 `docs/performance.md` §4.3）。
- 想改内核：`docs/kernels.md` → `docs/architecture.md` → `docs/optimizations.md`
  → `docs/index_sve_vs_fastindex_gap_20261023.md`（fastindex 差异与待补项）。
- 想接续开发：`docs/status.md` §7（演进方向优先级表 R1-R9：R1 小源扫描
  合并、R2 mask stride、R3 端到端验证，含预期收益与根因交叉引用）。
- 想集成/排障：`docs/integration.md` → `docs/override-mechanism.md` → `docs/status.md`。

## 设计要点

- 三段流水线：**归一化 normalize**（对齐 ATen `make_info`，保留 int32 下标）→
  **线性化 linearize**（一次遍历完成越界检查 + 负数归一 + 元素偏移）→
  **gather**（SVE512 向量 gather / 行拷贝 / 标量回退）。
- SVE512 路径：8B 元素（f64/i64）`svld1_gather_u64index_*`，4B 元素（f32/i32）
  `svld1_gather_u32index_*`，连续 A 元素块 `svld1/st1` 行拷贝，均带 `pf+16`
  预取（`svprfd/svprfw_gather_u32index`）。
- `K<1024` 快速路径 `gather_direct`：跳过线性化，直接读原始下标 + 内联检查
  （等价 ATen `Indexer`），支持 N-D 整数下标（broadcast_shape 混合进制解码）。
- **优化轮（T0-T15）**：自适应并行粒度 `grain.h`；M1 单索引大块
  `gather_rowcopy_raw` skip-linearize；T4 小值域桶化（值域 ≤2048 查表）；
  T3 `gather_fused` 融合单遍；int32 linearize 直接写 int32；T5 1 线程单遍；
  T8 mask 单遍压实；T9 小 K（≤4096）精简路径；T10 `torch.compile` opt-in；
  T11 线程数同步 + `gather_direct` A==1 紧凑循环 + 2B/1B/complex 单遍 +
  非连续 stride 正确性修复 + 1 线程 A>=4 恢复 raw rowcopy；T12 非连续尾维
  strided-rows（run 块拷贝）、≤64B memcpy、A<16 raw、grain 下限 64、多索引
  A>=4（仅 ≥16t）raw；T13 多索引小 K 入口快路径（跳过 normalize、预校验、
  dom==1 fill、逐行 memcpy）；T14 短行 memcpy 放宽到 A>=2；T15 中等行宽
  raw 放行（`A<=64 && B*K<=65536`）+ f32/i32+i64 下标大源（>512KB）直连
  单遍 SVE（uzp1 收窄复用 u32index）+ `small_single` 并行阈值线程缩放。
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
├── tests/                # 123 测试（smoke/normalize/correctness/errors/sve/model/index_out/mask/smallk/compile/strided/routes/smallmulti）
├── benchmarks/           # bench_index.py（原始矩阵）、bench_pinned.py/bench_gate.py（定点核复测）
│   │                     # bench_t1.py/bench_t2.py/bench_ai4s.py（优化轮专项）、bench_regression.py（回归扫描）
│   │                     # bench_threads.py（全线程矩阵 1-38t）
│   │                     # bench_smallk_real.py / bench_smallk_driver.py（小 K 两进程对拍）
│   │                     # bench_realpath.py / bench_realpath_driver.py（真实 workload 两进程对拍：M5/shortq/A1/M1/M2）
│   │                     # degrade_probe.py（阈值/mask 分解）、memprobe.c（内存系统探针）
│   └── RESULTS.md        # 完整基准报告（§2.4 T13 全量矩阵，T15 未改路由仍有效；§2.1 T11 基线；§3 历史轮次）
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

鲲鹏 920F，node 5（38 核），aten/hpc 同进程交替中位数。当前 31 场景
× 7 线程矩阵见 `benchmarks/RESULTS.md` §2.4（T13 轮；T15 未改 31 场景路由，
逐格核对不变），逐 case 分析见
`docs/degradation-analysis.md` §5.8/§5.9。

| 场景 | 1 线程 | 36 线程 |
|---|---|---|
| S1 spconv features[inp_idx] C=64 | **3.04x** | 8.10x |
| P1 h[src] i64 C=64 | **2.97x** | 8.45x |
| P2 pos[src] A=3 | **1.81x** | 9.89x |
| P3 x[mask] C=16 | **1.78x** | 4.21x |
| S4 x[perm] f64 | **1.28x** | 1.17x |
| `x[:,i]` C=64 | **1.99x** | 2.19x |
| rowcopy A=4 / A=16 | **1.87x / 2.14x** | 2.68x / 3.09x |
| 1D f32/f64 K=2M | 1.07x / 1.25x | 3.98x / 1.56x |
| half / bf16 | 1.33x / 1.30x | 4.07x / 4.00x |

- **RowCopy 是主要收益**：S1/P1 2-8t 3.0-3.4x（run 间至 4.6x）；A=4/A=16
  块 1t 1.9x/2.1x（T11 曾测 2.4x/3.6x；高线程含 aten 退化放大）。SVE `ld1/st1`
  整块拷贝，正中 spconv 规则表 / 图消息路径。
- **小 K `x[idx]`**（两进程真实路径，T15 构建）：K≤1024 为 **1.83-2.36x**；
  2D A=16 K=8192/65536 32-38t **1.26-2.17x**（T15 raw 放行，原 0.73/0.76x，
  hpc 绝对时间砍半）；K=1024 与 K=1023 全线程拉平（并行阈值线程缩放）；
  1D K=8192/65536 高线程 0.90-0.95x（小源保持标量，残留为前置 bucket
  扫描 ~15µs）。
- **真实 workload 对拍**（两进程 `x[idx]` 全链路，RESULTS §2.5）：M1 权重块
  （≤16KB/行）1t **13.0-13.6x**、8t 5.1-5.8x；M5 标量循环 / A1 单元素高频
  2.6-3.8x；shortq A=2-4 **2.6-4.3x**（T14 修复 A=3 短行标量拷贝 0.84→2.6x）；
  M2 全形态 ≥1.11x（38t 带宽饱和）。
- **仍 <1x 的持续项**：非连续 1D mask 0.86-0.98x（T8 只支持连续 mask）、
  int8/uint8 8/16t 0.82-0.99x（64B 行只用 1B，aten 同为标量）、complex 1t
  0.95x（无 SVE，2t+ 持平）、S4 8/16t 0.98/0.91x、`x[i,j]` 8/16t 0.97/0.92x
  （小规模强扩展）、1D K=65536 小源 32-38t 0.90-0.92x（T15 判明小源标量
  更快，残留为前置 bucket 扫描；大源已修，K=10M idx64 8t 2.22x）。
  逐项原因见 `degradation-analysis.md` §5.8/§5.9。
- **4B 大 K 随机 pure gather 的硬限制**：64B cache line 放大 + DRAM 延迟；
  同构建 A/B 实测 SVE 4B 与标量持平（8MB 1.85 vs 1.86 ms、256MB 7.46 vs
  7.45 ms），**不是 SVE 指令 MLP 缺陷**（RESULTS §3.7；早期 0.75 vs 3.4 GB/s
  探针归因已撤回）。小值域由 T4 桶化规避行放大；f64 8B 路径不受影响。
- **线程数**：T11 起 `torch.set_num_threads(t)` 对扩展生效（入口同步 libomp
  ICV）。旧 RESULTS §3.6.1/§3.9 的 2-36t 数据因两套 OpenMP 运行时失效，已作废。
- 共享 608 线程箱噪声大：608 列只作参考，**以 node 5 定点矩阵为准**。

## 已知限制 / 注意事项

- torch 2.10 限制：同进程内不允许第二个 Python CPU 覆盖 `aten::index.Tensor`
  （`RuntimeError: already a kernel registered from python`）。这是默认覆盖设计
  的固有约束。
- 覆盖是一次性的全局副作用：任何 import index_sve 的进程都会改掉全局
  `x[...]` 行为。如不希望全局生效，用 `HPC_INDEX_NO_OVERRIDE=1`。
- 索引张量必须在 CPU（meta/CUDA 索引抛 aten 同文错误，不再段错误）；`out=`
  变体（`aten::index.Tensor_out`）也已覆盖。详见 semantics.md / optimizations.md §10。
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
         tests/test_compile_optin.py tests/test_strided_rows.py \
         tests/test_rowcopy_routes.py tests/test_small_multi.py; do python $t; done
```

全部通过 = 127/127（覆盖激活态；test_correctness 25、test_errors 6、test_model 9、
test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
test_smallk 7、test_compile_optin 3、test_strided_rows 6、test_rowcopy_routes 8
（T15 增至 8）、test_small_multi 11）。
覆盖下测试是自洽的（`x[i]` 也走
本算子），值级真对拍用两进程法（无覆盖进程生成 true-aten 参考，覆盖进程比对）：

```bash
python tests/verify_against_aten.py --mode both    # 44/44，参考写 agent_space/（git-ignored）
```
