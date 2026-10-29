# index_sve 项目汇总——一次讲清楚这件事

> 面向第一次接触本项目的同事：读完可以在评审与讨论中专业地讨论它。
> 每章开头一句话摘要；深挖按各章指针跳转。全文 10-15 分钟。
> 所有比值统一为 **aten 时间 ÷ 本算子时间**，>1 = 比 torch 原生快。
> 代号（S1/P1/M1/shortq…）与术语的白话速查见 [`glossary.md`](glossary.md)。

## 0. TL;DR：一页看懂

| 问题 | 答案 |
|---|---|
| **做了什么** | 在鲲鹏 920F（aarch64 SVE512）上，把 PyTorch CPU 的 `x[张量下标]`（`aten::index.Tensor`）换成等语义的自研实现；`import index_sve` 即全局生效，**用户代码零改动** |
| **为什么值得做** | AI4S 模型 profiler：14 个模型的 index 累计 **27270ms**，是热点算子；spconv 取特征、消息传递 `h[src]`、mask 筛选全走它 |
| **多快** | 整行/整块取数（spconv、消息传递主形态）1 线程 **1.6-4.6x**（跨轮，随箱态/形态漂移）；真实权重块 1t **13x**；短调用 **1.7-4.3x**（µs 级随箱态）；spconv 代表形态全档 ≥1.09x |
| **多可靠** | 14 套件 **143/143** 测试（含 autograd 专项回归）+ **44 用例**两进程 true-aten 对拍；语义与 aten 逐条对齐（含错误消息逐字一致） |
| **哪里不快** | 6 个家族、多数 ≤10-18%（1B/小工作量/非连续 mask 等），过半是 aten 同样撞的硬件墙，见 §7 |
| **怎么退回** | `HPC_INDEX_NO_OVERRIDE=1`（进程级逃生门）；**按性能自动回退刻意不做**，完整理由见 §8 |
| **边界** | CPU 张量 only；`torch.compile` 默认绕过（有 opt-in 开关）；换机器需重标定阈值，见 §9 |

三个代表数字：

1. `h[src]`（图/点云消息传递，(N, 64) 行取数）1 线程 **2.9-4.4x**（跨轮；2026-10-29 轮 2.93x）
2. 真实权重块 `w[ki]`（行 ≤16KB）1 线程 **12.9-13.6x**
3. FPS/ball_query 风格短调用（每次取 1 个、高频）**1.7-3.8x**（µs 级，随箱态漂移）

## 1. 背景：为什么做这件事

**一句话：AI4S 模型里最热的那个算子，就是 `x[idx]`。**

对 14 个 AI4S 模型做 profiler 的结论：`aten::index` 累计 **27270ms**，排名第一。
你平时写的这些代码，就是本项目的输入：

| 你写的代码 | 对应场景（基准代号） |
|---|---|
| `features[inp_idx]`（spconv 按行取特征） | S1 / M2 |
| `h[src]`、`pos[src]`（图/点云消息传递） | P1 / P2 |
| `x[mask]`（bool 掩码筛选） | P3 |
| `w[ki]`（按索引取权重块） | M1（最热） |
| FPS / ball_query 循环里 `x[idx]` 每次取 1 个 | M5 / A1 |

本项目不改任何一行模型代码——只把这些调用背后的 CPU 实现换掉。

**例 1（全文反复使用）**：`features` 形状 (50000, 64) f32（源 12.8MB），按
100 万个随机行号取行 → 输出 (1M, 64) ≈ 256MB。同一条调用，本算子 1 线程比
aten 快 **3.0-4.5x**（S1；2026-10-29 轮 3.14x，T11 轻载轮曾 4.46x——1t
比值随箱态轮间漂移，RESULTS §2.8）。为什么能快这么多——§2 讲原因，§3 讲手段。

## 2. `aten::index` 在 CPU 上是怎么工作的，为什么慢

**一句话：ATen 用"两遍 + 逐元素"的通用模型换正确性，慢在三个地方，其中
一个是物理墙。**

调用链 30 秒版：`x[i]` → Python `__getitem__` → **dispatcher**（按 key 查
内核表的调度层）→ `aten::index.Tensor` 的 CPU 内核。我们的覆盖就发生在最后
一步（§3）。

ATen 的实现是通用性优先的三段：

1. **normalize**：把 mask / 负数下标 / broadcast 归一成统一的整数下标表；
2. **linearize**：**先物化**一个 K 个元素的一维偏移数组（一遍写 + 越界检查）；
3. **gather**：再读这个数组、逐个元素取数（一遍读）。

→ 至少两遍内存流量 + 一次中间分配。慢的三个根因：

1. **行放大（物理墙，谁实现都躲不开）**：CPU 搬数据的最小单位是 64B cache
   line。随机取 1 个 4B 元素也要把整条 64B 搬进来——有效带宽 1/16。
   *算例（说明用，非实测）*：1D f32 随机取 100 万元 → 100 万条 line × 64B
   = 64MB 流量，有用的只有 4MB。而按**整行**取（一行 64×4B=256B，4 条 line
   全部有用）行放大 ≈ 1。**这就是本项目收益的物理来源：把"零取"变成"整行搬"。**
2. **两遍模型**：小调用（K≈1000）里 linearize 的固定开销占比高；1 线程没有
   并行来摊薄它。
3. **小 op 调度/迭代器开销**：如 `x[:, i]` 历史上只有 0.18x（T11 修复，§3）。

深挖：`reference/aten_index_primer.md`（通俗导读）、`architecture.md`。

## 3. 我们做了什么：核心改进

**一句话：不碰用户代码，把 dispatcher 里 `aten::index.Tensor` 的 CPU 键指向
自研 SVE512 实现，语义不变；内部按张量形态自动路由到六类内核。**

覆盖机制一句话：`import index_sve` 时扩展用 `TORCH_LIBRARY_IMPL(aten, CPU)`
完成注册；此后进程内所有 CPU 的 `x[...]` 直达本算子（没有 Python 层往返）。
dispatcher 链是 `AutogradCPU → CPU(我们)`，autograd 不受影响（§10）。

内部路由树（简化版，完整版见 `docs/index.md`）：

```
x[张量下标]（归一化后）
├─ 单 mask && 连续          → gather_mask      （T8 单遍压实）
├─ 小 K（K≤4096/1024）      → small_* / gather_direct（免 linearize）
├─ 单索引 A>1（行取数）     → gather_rowcopy_raw（raw rowcopy，跳 linearize）
├─ 非连续尾维有连续 run     → strided_rows_raw （T12 run 块拷贝）
├─ 下标值域 ≤2048           → bucket 桶化查表
│    └─ 4B 大源(>512KB)+多线程 → gather_single_1d_sve（T15-C 直连 SVE）
├─ 多索引 A≥4 且 ≥16 线程   → rowcopy_fused_sve（T12）
└─ 其余                     → linearize + SVE gather / dispatch_scalar（标量兜底）
```

六类手段（按收益排序）：

| # | 手段 | 主要内核（函数名） | 吃到的形态 | 效果来源 |
|---|---|---|---|---|
| 1 | **RowCopy 整行拷贝**（SVE `ld1/st1`） | `rowcopy_sve`（两遍后）/ `rowcopy_raw_single`（raw）/ `rowcopy_fused_sve`（多索引），底座 `copy_contig_sve` | A≥4 连续行：`h[src]`、`features[inp_idx]` | 一条 64B line 里有用字节接近 100%，行放大从 16x 降到 ≈1x → 1t 4.4x |
| 2 | **raw 单遍（跳过 linearize）** | 入口 `gather_rowcopy_raw` → `rowcopy_raw_single`；`gather_single_1d`；`gather_bucket_pure_raw` | 单索引行取数、权重块 M1 | 省一遍偏移写 + 一遍读 + 中间分配 → M1 1t 13x |
| 3 | **小值域桶化** | `bucket_gather_pure`（两遍后）/ `bucket_gather_pure_raw`（raw），入口 `gather_bucket_pure_raw` | 下标值域 ≤2048 的查表（如 M3 `keys[order]`） | 8MB 随机流量变成 KB 级查表 |
| 4 | **融合单遍 / 小 K 快路径** | `gather_mask`（T8 压实）、`gather_fused`（T3 融合）、`gather_direct` 与 `small_single_index` / `small_multi_index`（免 normalize） | mask 压实、K<1024 直接读原始下标 | 去掉两遍模型的固定开销 |
| 5 | **大源 4B 直连 SVE**（T15-C） | `gather_single_1d_sve` → `pure_gather_sve_4b_i64idx` | f32/i32 源 >512KB、多线程 | gather 每迭代 8 路在飞请求摊掉 DRAM 延迟（每元素 ~19ns→~10ns） |
| 6 | **并行粒度与线程同步** | `adaptive_grain`（grain.h）、`sync_omp_threads`（op.cpp） | 所有并行路径 | 自适应 grain；`torch.set_num_threads` 对扩展真正生效 |

T0-T15 = 16 轮"发现劣化 → 定位根因 → 修路由/内核"的迭代，每轮有记录
（`optimizations.md`）；历史劣化家族的修复台账见 §8。

深挖：`kernels.md`（各内核实现细节）、`override-mechanism.md`（劫持原理）。

## 4. 使用方法

### 4.1 环境与构建（本机 920F 示例，详见 README）

```bash
source ~/.bashrc && conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=<conda env>/lib/python3.12/site-packages/torch/lib:\
<HPCKit>/compiler/bisheng/lib:$LD_LIBRARY_PATH   # 必须含 BiSheng libomp 目录

cd index_sve && python setup.py build_ext --inplace   # 勿用 ninja
```

> 运行时契约：扩展链接 BiSheng 的 `libomp.so`，`LD_LIBRARY_PATH` 不含其
> 目录时 import 直接报错（这是部署最常见问题）。

### 4.2 生效与验证

```python
import torch, index_sve   # import 即全局生效

x = torch.randn(50_000, 64)
i = torch.randint(0, 50_000, (1_000_000,), dtype=torch.int32)
y = x[i]                  # 已走本算子，1 线程约 4x
```

确认已生效的三个办法：

1. **import 时的一次性 UserWarning**（"Overriding a previously registered
   kernel..."）——这是 dispatcher 对覆盖的提示，**预期、无害**，故意不压制；
2. `torch._C._dispatch_dump_table("aten::index.Tensor")` 可见 CPU 键已指向
   扩展注册的内核（`override-mechanism.md` §5）；
3. 内核命中计数器：`index_sve._hpc_index.rowcopy_raw_count()` /
   `single_1d_count()` / `single_1d_sve_count()`，跑一段后读数可确认路由。

### 4.3 环境变量开关

| 变量 | 作用 | 何时用 |
|---|---|---|
| `HPC_INDEX_NO_OVERRIDE=1` | **进程级逃生门**：不注册覆盖，全程原生 aten | 排障 / 基准对拍 / A-B 对比（须在 import 前设置） |
| `HPC_INDEX_COMPILE=1` | `torch.compile` opt-in（§4.5） | 编译态也要吃到收益时 |
| `HPC_INDEX_NT=1` | 大输出 NT store（绕缓存写） | 只写一次的大输出（默认关） |
| `PREFETCH_DIST` | gather 预取距离（构建期，默认 16） | 调优时 |

### 4.4 线程建议

- T11 起扩展内部同步 OpenMP 线程数：`torch.set_num_threads(t)` **对扩展
  真正生效**（此前存在双 OpenMP 运行时 ICV 不同步的坑，旧 2-36t 数据因此作废）。
- 本机单节点 38 核；共享箱上 608 全核数字噪声大，评估以**低/中线程（1-36t）
  + 绝对带宽**为准（高线程比值部分来自 aten 自身带宽饱和后变慢）。
- **宿主已多线程/多进程**：没有算子层冲突（状态全在进程内 + T11 同步）；
  要避免的是**超订**——进程内并发调用时守则"宿主线程数 × nt ≈ 核数"，用
  `set_num_threads(核数/并发数)` 把真实预算告诉 torch，本算子会跟随。实测
  4 线程 × nt=36 压 38 核时每调用劣化 8.6x（aten 同现象同量级）；
  见 `integration.md` §6.1（含实测表与复现脚本）。

### 4.5 与 torch.compile 共存（重载已做，opt-in）

**默认不生效**：Inductor 会把 `aten::index` 内联成自己的内核，绕过 dispatcher
覆盖（实测比覆盖态慢 2.6x）。设 `HPC_INDEX_COMPILE=1`（import 前）启用三件套：

1. 注册 fake/meta（整数下标按 broadcast 推形状；mask 输出长度依赖数据，
   compile 下仍走 aten fallback）；
2. **把 Inductor 的 `aten.index.Tensor` lowering 永久替换为 fallback**——
   运行期回到 dispatcher，命中 CPU 覆盖（实测 2.6x → 1.07-1.3x）；
3. 切换独立 inductor 缓存子目录（缓存 key 不含 lowering 注册信息，必须隔离）。

未开覆盖（`HPC_INDEX_NO_OVERRIDE=1`）时该开关直接报错，避免"以为启用了、
其实回落原生"的静默混用。深挖：`torch-compile.md`。

### 4.6 DataLoader 多进程

fork 模式的 worker 继承父进程的注册（无需重复 import）；spawn 模式需在
dataset 模块顶部 import `index_sve`。（Python/PyTorch 通用规则，未在仓库
基准中专项验证。）

## 5. 怎么证明的：基准方法与正确性

**一句话：共享机器上可信的测法是两进程交替 + 中位数；正确性靠 143 项测试
+ 44 用例 true-aten 两进程对拍。**

- **为什么不能裸跑分**：测试机是共享箱（5+ 用户），绝对时间漂移可达
  10-20%，单次数字不可信。
- **两进程交替中位数**：aten 进程（`HPC_INDEX_NO_OVERRIDE=1`）与本算子进程
  交替运行取中位数，对抗漂移——这是所有正式比值的标准口径。
- **覆盖矩阵**：31 场景 × 1/2/4/8/16/32/36/38 线程（RESULTS §2.8，2026-10-29
  轮）+ 真实 workload 对拍（RESULTS §2.5/§2.8/§3.8）。
- **正确性两层**：
  - 14 套件 143/143（覆盖激活态）：注册/normalize/各 dtype × 下标矩阵、
    错误消息逐字、SVE 分支门控、9 个模型场景、autograd 专项回归等；
  - **值级对拍必须两进程**——覆盖激活后 `x[i]` 也走本算子，同进程内无法
    自证（"对拍陷阱"）。`tests/verify_against_aten.py --mode both` 自动跑
    两进程：无覆盖进程生成 true-aten 参考，覆盖进程比对，44 用例全过
    （含 K≥1024、桶化、M1 raw、多索引、mask、广播、非连续 stride）。
- **autograd 已有专项回归**（`tests/test_autograd.py`，16 项：前向记录/grad_fn、
  梯度 vs `index_add_` 独立参考、负索引/重复/mask/多索引/非连续/gradcheck、路由
  计数器证明带 grad 前向穿透本算子）；两进程梯度对拍记录在 `semantics.md` §6.4
  （§10 训练场景有详细说明）。

## 6. 性能测试结果

**一句话：通用大面上"整行取数"家族收益最大；AI4S 专项全绿，权重块 13x；
纯随机 gather 受物理墙限制在 1x 附近。**

### 6.1 通用场景（31 场景矩阵的家族汇总）

| 形态族 | 代表场景 | 典型比值 |
|---|---|---|
| **RowCopy 整行/整块** | `x[:,idx]` B=256 K=100k **3.48x**、`x[idx,:]` C=32 E=1M **3.50x**（1t） | 1 线程 1.6-4.6x（跨轮；2026-10-29 轮 A=2-16 行 1.6-2.1x、S1/P1 2.9-3.1x），多线程至 8.7x |
| 纯 gather 8B（f64） | 1D f64 K=2M/10M | 1.0-1.5x（36t 至 1.95x） |
| 纯 gather 4B（f32/i32）大 K | 1-16t 1.06-1.16x；**i64 下标大源 8t 2.22x**（T15-C） | 行放大墙下的小幅正收益 |
| mask（连续） | P3 `x[mask]` | 1.2-3.8x（T8 单遍压实后） |
| 小 K（K≤1024） | `x[idx]` | **1.83-2.36x** |
| 2D A=16 中 K 高线程 | K=8192/65536，32-38t | 1.26-2.17x（T15 修复后，原 0.73/0.76x） |
| 1B / complex | int8/uint8/complex64 | ≈1x（标量单遍，无 SVE 收益也无回归） |
| 非连续 mask | stride 视图 | 0.86-0.98x（§7 劣化家族之一） |
| 608 全核定点 | S4 | 5.6x（可扩展性的存在性证明） |

### 6.2 AI4S workload 专项说明（真实形态对拍，全线程档）

2026-10-29 轮（当前构建，node5 空闲轻载）：两脚本同日各跑一轮、单轮自洽，
把 31 场景矩阵与真实 workload 全部补齐到 1/2/4/8/16/32/36/38 八档线程
（此前 AI4S 形态只有 1t 与 36/38t 两个端点）。完整 31 行矩阵与判读要点见
RESULTS §2.8。

**表 1：模型形态（31 场景矩阵中 AI4S 相关行，进程内交错中位数）**

| 你的代码 | 代号 | 1t | 2t | 4t | 8t | 16t | 32t | 36t | 38t |
|---|---|---|---|---|---|---|---|---|---|
| spconv `features[inp_idx]` C=64 | S1 | **3.14x** | 3.56x | 3.38x | 3.02x | 2.66x | 6.84x | 7.96x | 8.65x |
| 消息传递 `h[src]` i64 C=64 | P1 | **2.93x** | 3.62x | 3.45x | 3.36x | 3.01x | 6.25x | 7.01x | 5.44x |
| 位置取数 `pos[src]` A=3 | P2 | 1.76x | 1.84x | 1.72x | 1.67x | 1.71x | 3.88x | 3.16x | 6.99x |
| mask 筛选 `x[mask]` C=16 | P3 | 1.76x | 1.56x | 1.37x | 1.23x | 1.22x | 3.20x | 3.80x | 5.10x |
| 置换取数 `x[perm]` f64 | S4 | 1.27x | 1.33x | 1.10x | 0.91x | 0.80x | 1.16x | 1.24x | 1.06x |

**表 2：真实 workload（两进程真实 `x[idx]` 全链路，REPS=3）**

| 你的代码 | 代号 | 1t | 2t | 4t | 8t | 16t | 32t | 36t | 38t |
|---|---|---|---|---|---|---|---|---|---|
| **权重块 `w[ki]` 行 ≤16KB（3 形态 × i32/i64）** | M1 | **12.9-13.6x** | 12.4-12.9x | 10.1-11.1x† | 4.1-5.5x | 2.7-3.1x | 1.4-1.6x | 1.2-1.5x | 1.2-1.4x |
| spconv 代表 (20000,8) | M2 | 1.28x | 1.34x | 1.30x | 1.62x | 2.27x | 4.78x | 5.25x | 5.32x |
| spconv 代表 (20000,64) | M2 | 3.44x | 3.98x | 3.38x | 2.78x | 2.39x | 2.10x | 2.14x | 2.07x |
| spconv 代表 (20000,256) | M2 | 3.11x | 2.95x | 2.74x | 2.14x | 1.49x | 1.21x | 1.11x | **1.09x** |
| FPS/ball_query 标量循环 1D / 2D A=3 | M5 | 1.65/1.73x | 1.66/1.76x | 1.69/1.78x | 1.69/1.76x | 1.69/1.78x | 1.69/1.77x | 1.70/1.78x | 1.67/1.74x |
| 单元素高频 i32 / i64 | A1 | 2.49/1.66x | 2.50/1.68x | 2.51/1.68x | 2.51/1.67x | 2.51/1.68x | 2.49/1.67x | 2.55/1.67x | 2.57/1.67x |
| 短行查询 A=2-4 K=64-512 i32/i64 | shortq | 2.06-3.33x | 2.09-3.34x | 2.10-3.36x | 2.09-3.28x | 2.07-3.37x | 2.14-3.29x | 2.08-3.37x | 2.13-3.31x |
| 短行查询 K=1023 i32/i64（剔病理格\*） | shortq | 2.14-3.10x | 1.82-2.77x | 1.88-2.90x | 1.86-2.82x | 1.95-2.88x | 2.17-2.92x | 2.04-3.09x | 2.16-2.99x |

注记：

- M5/A1 每格 `a/b` = i32/i64（M5 为 1D/2D）；shortq 合并区间中 i32 靠上端
  （~2.9-3.4x）、i64 靠下端（~2.1-2.3x）。
- † M1 4t 的 i64 (27,32,32) 单格 6.35x 判噪声（aten 侧与 i32 全同、i32 同格
  11.04x、§2.6 同形状 10.0-10.2x），其余格 10.1-11.1x。
- \* K=1023×A≥3 部分高线程格触发 **aten 自身病理**（fork-join 风暴，aten
  539-615µs → 34-40x）——是 aten 退化不是本算子收益，判读剔除（§2.5 已
  记录，本轮散现 11 格）。
- **µs 级 case（M5/A1/shortq）比值随箱态漂移**：本轮轻载 1.65-3.4x，§2.5
  重载轮 2.6-4.3x；hpc 侧绝对时间稳定（M5 ~8.9µs、A1 ~8.7µs 全档持平），
  漂移主要来自 aten 侧。跨轮只比形态。
- mask empty（空掩码退化格）4-38t 0.51-0.73x 为本轮新观察项（绝对差
  ~0.3µs，aten 并行早退变快），待聚焦复测——详见 RESULTS §2.8。

读法说明：

- **S1/P1 为什么快**：行宽 64 的行取数，RowCopy + raw 单遍把行放大和两遍
  开销都消掉（§3 手段 1+2）。1t 2.9-3.1x、2-8t 3.0-3.6x（T11 轻载轮曾至
  4.4-4.6x，1t 比值轮间漂移 ~1.4x，引用标轮次）；32t 后双方带宽饱和、
  aten 掉得更多，比值走高但那部分属 aten 侧退化。
- **P2 为什么 1.7-1.8x 起步**：A=3 行 12B 覆盖不了一条 64B line（有用率
  3/16），多线程后延迟摊开到 7.0x（38t）。
- **S4 为什么 1 线程只有 1.27x**：f64 纯随机 gather，行放大 8x 是物理墙——
  不是实现差距，是"随机取 8B 也要搬 64B"的硬件事实（§7 有 aten 侧对照
  证据）；8/16t 0.91/0.80x 是小工作量下 fork-join 固定成本。
- **M1 为什么 13x**：权重块行 ≤16KB，一次大块拷贝 + 跳过 linearize，两遍
  模型在小 K 高重复形态下开销占比极高，raw 单遍几乎全省。1-4t 10.1-13.6x
  是集成收益主体，8t 起带宽饱和回落，38t 1.2-1.4x 非路由退化（§2.5/§2.8）。
- **M2 (20000,8) 为什么 1t 只有 1.28x**：32B 短行一次只吃半条 line；线程
  多后 aten 并行开销上升，38t 反超到 5.32x。M2 全形态 ≥1.09x。
- **短调用为什么快**：K 小，固定开销（linearize 物化、分配）占大头，
  精简路径直接砍掉。

### 6.3 数字怎么读（可信度分级）

| 可信级 | 口径 | 例子 |
|---|---|---|
| ★★★ | 两进程交替对拍（realpath / 对拍矩阵） | M1 13x、shortq 2.1-4.3x（跨轮） |
| ★★ | 定点核复测（单节点独占） | 31 场景矩阵主数字 |
| ★ | 共享箱单次跑 | 只看趋势，不引用 |

高线程比值部分来自 aten 带宽饱和后自身变慢——**判断真实收益看"低/中线程
比值 + 绝对 GB/s"**，不要只看最大比值。

## 7. 诚实的劣化清单：哪里比 aten 慢、为什么

**一句话：6 个家族多数 ≤10-18%，过半是 aten 同样撞的硬件墙；剩下两个
可修的走根因修复（R1/R2），不是回退。**

| 家族 | 数字 | 根因 | aten 侧对照 |
|---|---|---|---|
| 1B（int8/uint8）8-16t | 0.82-0.99x | 1B 行放大 **64x**（一条 line 只有 1/64 有用），物理墙 | 同为标量、同撞墙（双方有效带宽都 ~0.7GB/s） |
| 小工作量高线程（S4/多索引） | 0.91-0.98x | 每遍 5-15µs 的 fork-join 固定成本，与工作量无关 | aten 同样付，且 K=1023×A≥3 时有自己的 539-615µs 病理（我们串行 17µs 免疫） |
| 非连续 mask | 0.86-0.98x | T8 只融合连续 mask，非连续的按 ATen 语义先做 nonzero 物化 | **可修**（R2：按 stride 读 mask，根因修复中） |
| complex64 1t | 0.95x | 无 SVE 内核，标量单遍 | 差距 5% |
| 1D 小源高线程 | 0.90-0.95x | 前置值域扫描 ~15µs 占比 | **可修**（R1：扫描并入 gather 单遍） |
| 4B 大 K 大值域（历史最严重） | **已反超**：1-16t 1.06-1.16x，i64 大源 8t 2.22x | 曾 0.18-0.7x，T11/T15 修 | 见 §8 修复台账 |

**"aten 同样撞墙"的证据**（同构建、同随机模式的内核 A/B）：

| 内核 | 源 8MB（8t） | 源 256MB（38t） |
|---|---|---|
| SVE 4B gather | 1.85 ms | 7.46 ms |
| 标量循环 | 1.86 ms | 7.45 ms |

持平——换指令、换实现都无效，瓶颈是 64B 行放大 + DRAM 延迟本身。这决定了
4B 随机 gather 的可行方向是**减少行流量**（桶化/预取/改行取），而不是"换个
更快的实现"。

## 8. 为什么不做"慢了就回退 aten"

**一句话：回退分三层，前两层已经存在；被否决的只是第三层（按性能回退），
因为它用确定的成本买不确定且微小的收益。**

| 层级 | 机制 | 状态 |
|---|---|---|
| **机制级** | 算子内部每条路径本身就是回退链：桶化 → SVE gather → rowcopy → 标量兜底；dtype 没有 SVE 内核就落标量 | **已实现**（按 shape/dtype 静态属性选路，判断零成本） |
| **进程级** | `HPC_INDEX_NO_OVERRIDE=1` 整进程退回 aten | **已实现**（逃生门/对拍开关） |
| **性能级** | 运行时按"该形态是否比 aten 慢"改调 true aten | **否决**，理由如下 |

否决的四条理由（完整论证 `fallback-analysis.md` §4.1-4.7）：

1. **残余慢格多数是共同硬件墙**（§7）——回退买不到速度，SVE 4B vs 标量
   持平的 A/B 就是证据；
2. **固定成本会精准伤害收益主体**：Python 层回退实测每调用 +**45µs**
   （小 op 13.7µs→63µs，即 4.6x 回归）。本算子收益主体是小/中张量
   （单次 <100µs），一行"慢了就回退"的保险丝让所有快场景买单；
3. **"哪个格子慢"不可预知且不稳定**：同一形态随线程数反转（1D f32 K=10M
   1t 1.06x、36t 3.92x），随共享箱状态漂移 10-20%——静态回退表只能追噪声；
4. **根因修复被 15 轮验证有效**，fallback 会冻结它：

| 轮次 | 根因 | 数字 |
|---|---|---|
| T5 | 1 线程两遍模型 | 0.6-0.9x → ≥1.07x |
| T8 | mask 多遍物化 | → 1.2-3.8x |
| T11 | `x[:,i]` 迭代器开销 | 0.4-0.7x → 1.1-2.0x |
| T13 | 多索引小 K 走全量路径 | 0.74x → 1.6-2.0x |
| T14 | 短行标量循环 | 0.84x → 2.6x+ |
| T15 | 路由空档/大源缺 SVE/并行阈值 | 0.73x → 1.26-2.17x |

剩余两个可修家族（R1 小源扫描、R2 mask stride）同样走根因修复。

## 9. 适用边界：哪些调用方式会绕过

**一句话：CPU 张量 + 不用 compile + 不设逃生门——没有任何 `aten::index`
调用方式会绕过；三类例外和范围外情况列在下表。**

| 情况 | 行为 |
|---|---|
| CPU 张量上的 `x[i]` / `x[:, i]` / `x[i, j]` / `x[mask]` / N-D / `out=` 变体 | **全部走本算子，无绕过**（覆盖的是 dispatcher 的 CPU 键本身，覆盖注册后不存在旁路） |
| 非 CPU 张量（CUDA 等） | 走 aten 原生内核（覆盖只注册 CPU 键——范围外，非"回退"） |
| `torch.compile` 默认 | Inductor 内联绕过（慢 2.6x）；`HPC_INDEX_COMPILE=1` opt-in 后回到本算子（§4.5） |
| 直调 `torch.ops.hpc_index.index` / `index_sve.index`（不走 `x[...]`） | 走本算子本体，但**不经 aten::index 的 AutogradCPU 链**——无 autograd 公式，requires_grad 输入的梯度被静默丢弃；仅推理/对拍用，训练必须 `x[...]`（行为由 `tests/test_autograd.py` 钉住） |
| `HPC_INDEX_NO_OVERRIDE=1` | 逃生门：全程原生 aten（用户主动） |
| barebones dtype（u16/u32/u64、float8、complex32） | **不回退，抛 `NotImplementedError`**——dispatch 白名单未含（演进项 R8，覆盖缺口非性能缺口） |
| 索引张量在 meta/CUDA | 抛与 aten 同文的错误（行为对齐） |

**范围澄清（不是绕过）**：`index_put_`（写路径）、`index_select`、
`masked_select`、`take` 是**不同算子**，不在本项目覆盖范围；本项目只覆盖读
路径 `aten::index.Tensor`。

其他边界：

- **换机器**：阈值（512KB 源门、K<1024、并行阈值等）按 920F 实测标定，
  换 CPU/缓存配置需重标（`porting.md`）；
- **torch 2.10 约束**：同进程不允许第二个 Python CPU 覆盖 `aten::index.Tensor`；
- 覆盖是进程级全局副作用——这是设计（零改动收益），不是 bug。

## 10. 训练场景适用吗？

**一句话：适用——收益在训练前向的 CPU 取数（与推理同一条路径）；backward
不受影响也不加速；写路径不在范围。**

- **前向取数：直接受益**。训练的前向与推理的前向是同一条代码路径——
  spconv 的 `features[inp_idx]`、消息传递的 `h[src]`、数据增强里的 CPU
  gather，训练时同样走本算子、同样是 §6.2 的数字。
- **autograd 为什么不坏**：dispatcher 链是 `AutogradCPU（记录）→ CPU（我们）
  → 底层`。覆盖只替换 CPU 键；`x[i]` 的求导公式（`IndexBackward`）挂在独立的
  `Autograd` 键上，仍由 ATen 提供——前向记录正常、梯度数值/形状/dtype 有
  与 true-aten 的对拍记录（`semantics.md` §6.4），并有套件专项回归保险
  （`tests/test_autograd.py` 16 项：梯度 vs `index_add_` 独立参考、gradcheck、
  路由计数器证明带 grad 前向穿透本算子）。**直调注意**：`torch.ops.hpc_index.*`
  无 autograd 公式，梯度会被 fallback 静默丢弃——训练务必用 `x[...]`。
- **不会变快的**：backward 本身（aten 原生内核，不在本项目范围）；GPU 上的
  取数；写路径 `index_put_` / scatter（不同算子）。
- **DataLoader 多进程**：fork 继承注册；spawn 需在 worker 侧 import
  （§4.6）。

## 11. FAQ：讨论中一定会被问到的

**Q1：为什么不定 608 线程跑？**
共享箱（5+ 用户）上全核数字噪声大；本机 32 个 NUMA 节点，跨节点访存劣化。
正式口径是单节点 38 核定点复测；608 全核定点下 S4 也有 5.6x（可扩展性
存在），但日常引用以 1-36t 为准。

**Q2：端到端模型级收益验证了吗？**
算子级已验证（两进程对拍 + 真实 workload 形态对拍）；模型级传导是演进项
R3（决策项）——这也是用"真实形态对拍"做代理验证的原因。

**Q3：和 fastindex 什么关系？**
能力差异已逐条审计（`index_sve_vs_fastindex_gap_20261023.md`）：其核心项
strided-rows（P0）本仓库已在 T12 实现；剩余差异（`_offset` 变体等）按
收益排在其内部优先级表，未全部跟进。

**Q4：换一台机器能用吗？**
架构上需 aarch64 + SVE2；所有性能阈值按 920F 实测标定，换机器须重标
（`porting.md` 有完整移植清单与测试基线）。

**Q5：import 时的 UserWarning 是什么？**
dispatcher 对"覆盖已注册内核"的一次性提示，预期、无害，故意不压制
（压制会掩盖真正的重复注册问题）。

**Q6：训练里 spconv 前向会快吗？**
会——训练前向的 CPU 取数与推理同路径（§10）。

**Q7：mask 场景为什么"只有"2-4x，而行取数 4x+？**
连续 mask 已做单遍压实（T8）；非连续 mask 按 ATen 语义要先 nonzero 物化，
这是残余 0.86-0.98x 家族的来源，R2（按 stride 读 mask）是下一步根因修复。

**Q8：下一步最值得做什么？**
R1（1D 小源扫描合并，0.90-0.95x→≥1x）、R2（mask stride，→1.2-1.5x）、
R3（端到端验证，决定后续投入）；覆盖面缺口 R8（barebones dtype）。
完整优先级表见 `status.md` §7。

**Q9：我的程序已经是多进程/多线程，嵌入后会冲突吗？**
没有算子层冲突：覆盖/OpenMP 状态全在进程内，多进程天然隔离（正式数字本就
测自多用户共享箱）；T11 同步保证 torch 的线程预算对本算子真实生效。要管理
的是**超订**——并发调用时守则"宿主线程数 × nt ≈ 核数"（实测 4 线程 ×
nt=36 压 38 核，每调用劣化 8.6x，aten 同现象同量级）。唯一应避免的形态：
在宿主 C++ 自己的 OpenMP region 内直接调 torch（跨 runtime 嵌套守卫失效，
静默超订）。深挖 `integration.md` §6.1。

## 附录：深挖地图

| 想了解 | 去看 |
|---|---|
| 名词/代号白话速查 | `glossary.md` |
| x[idx] 从 Python 到 C++ 的完整机制 | `reference/aten_index_primer.md` |
| 覆盖/hijack 原理与边界 | `override-mechanism.md` |
| 各内核实现细节 | `kernels.md` |
| 逐场景性能数字 | `benchmarks/RESULTS.md` |
| 劣化的硬件根因推导 | `degradation-analysis.md`（§5.9） |
| 胜出场景的硬件根因 | `performance.md`（§4.3） |
| 不做 fallback 的完整论证 | `fallback-analysis.md`（§4） |
| 16 轮优化记录 | `optimizations.md` |
| 移植指南与重标定清单 | `porting.md` |
| 接续开发优先级 | `status.md`（§7，R1-R9） |
