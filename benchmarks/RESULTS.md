# 基准测试报告（RESULTS）

本文件是 `index_sve` 的正式性能报告：记录测试平台、测量口径、**T11 权威矩阵**，
以及 T0-T11 各优化轮的历史数据。所有比值统一为 **aten 时间 / 本算子时间**。

- 平台：鲲鹏 920F（aarch64，SVE2，VL=512-bit），608 核 / 32 NUMA，
  **共享机器（5+ 用户，噪声大）**。
- 构建：BiSheng clang 19.1.7，`-march=armv8-a+sve2 -msve-vector-bits=512 -O3 -fopenmp`。
- 固定配置：`PREFETCH_DIST=16`，`HPC_INDEX_NT` 默认关闭（opt-in），`-fopenmp` 开启。

> **先读结论**：整块/整行取数（RowCopy）收益最大，1 线程 2.4-4.5x；1D 8B
> 纯 gather 1.0-1.5x；4B 随机 gather 约 1.1x；1B/complex 基本持平；非连续
> mask 与“每线程工作量过小”的高线程单元仍低于 1x。**4B 大 K 随机 gather 的
> 上限是 64B cache line 放大 + DRAM 延迟，换 SVE/标量都无解**。
>
> **数据有效性**：当前有效结论只有 §2（T11：31 场景 × 7 线程 + 小 K 两进程）。
> §3 是历史记录，其中标着 2-36t 的数据都受“libomp/libgomp 双 ICV”影响
> （详见 §1.3），**只有 1t/38t 可信**。

---

## 0. 结论速览

| 形态 | 代表结果（T11） | 结论 |
|---|---|---|
| S1/P1 整行 RowCopy（C=64） | 1t **4.46x / 4.42x** | 最强项；spconv 规则表、图消息热路径 |
| rowcopy A=4 / A=16 | 1t **2.41x / 3.58x** | A>=4 连续块都值得走 SVE 整块拷贝 |
| P2 短行 A=3 | 1t 2.05x | 短行收益有限，但仍优于 aten |
| 1D f64/f32 K=2M | 1t 1.26x / 1.07x | 8B 明显好于 4B；中低线程 1.0-1.4x |
| 1D f32 K=10M | 1t 1.06x | 大 K 随机 4B，硬件上限；高线程倍率含 aten 退化 |
| half / bf16（2B） | 1t 1.53x / 1.35x | T11 单遍后稳定正收益 |
| int8 / uint8（1B） | 1-16t 0.90-1.06x | 基本持平；64B 行只用 1B |
| complex64 | 1-16t 0.95-1.00x | 无 SVE 内核，标量单遍，基本持平 |
| bool mask（连续 P3） | 1t **2.23x** | T8 单遍压实，跳过 `nonzero()` |
| bool mask（非连续） | 0.84-0.98x | T8 只融合连续 mask，回退旧路径 |
| 小 K `x[idx]`（K=0-1024） | 1.31-2.03x | 两进程真实路径；K>=8192 高线程 0.60-0.96x |
| `x[:,i]` | 1t 2.02x | T11 A==1 紧凑循环修复，旧 0.18x 已消失 |

**一句话**：本算子不是“通用 1D 索引加速器”，而是**以整行/整块取数为收益主体**
的定向加速器；随机单元素 gather 受内存系统限制，收益有限，4B 大 K 不要期待加速。

---

## 1. 测试环境、口径与可信度

### 1.1 平台与构建

- CPU：鲲鹏 920F，SVE2，VL=512-bit；608 个逻辑 CPU，1 thread/core（无 SMT），
  32 个 NUMA 节点。
- 主力测试节点：node 5（CPU 190-227，38 核，本地内存 64GB）。
- 缓存：L1d 32KB 8-way，L2 768KB 12-way，64B cache line；未暴露统一 LLC。
- 频率：1.2-2.0GHz（随负载/功耗浮动）。
- torch 2.10.0+cpu；基准统一用 `HPC_INDEX_NO_OVERRIDE=1`，让 `x[i]` 指向真 aten
  （否则同进程 `x[i]` 也会被覆盖，无法对照）。

### 1.2 结果怎么读

- **比值（speedup）** = `aten 时间 / hpc 时间`：`1.26x` 表示快 26%，`0.87x`
  表示慢 13%，`1.00x` 表示持平。
- **线程数** `1t/2t/.../36t` 是 OpenMP 工作线程数，不是 CPU 型号。
- **GB/s** =（索引读字节 + 数据读/写字节）/ 时间；用于判断是否撞到内存带宽上限。
- **不要只看最大倍率**：32/36t 时 aten 自己可能先饱和变慢，比值被放大。判断
  真实收益看“1-8t 比值 + 绝对 GB/s + 单线程结果”。
- 表中 `S1/P1/P2/P3/S4` 是 5 个模型形态代号（见 `bench_threads.py`）：
  S1=spconv 行 gather，P1=图消息，P2=坐标短行，P3=bool mask，S4=大排列。

### 1.3 线程数修正（关键，必读）

扩展链接 BiSheng **libomp**，而 ATen 用 **libgomp**，两者各自维护线程数 ICV：

- `torch.set_num_threads(t)` 只改 libgomp，**改不到扩展**。
- 扩展里调用的 `at::parallel_for` 是头文件内联模板，其 OpenMP 实现随扩展编译，
  因此由 libomp 执行；默认 team = 进程可见 CPU 数（affinity 到 38 核时就是 38）。
  扩展源码没有手写 `#pragma omp`，坑来自这个内联模板。

所以在 T11 之前，标着 2/4/8/16/32/36t 的 HPC 行其实都用 38 个 worker，而 aten
行确实只用 t 个，**相对加速比被严重高估**。只有 1t（`parallel_for` 串行短路）
和 38t（恰好等于默认 team）有效。

修正方式（T11）：

1. `benchmarks/bench_threads.py` 用 `ctypes` 找到已加载的 libomp，调用
   `omp_set_num_threads(t)`；
2. 内核入口 `index_impl` / `index_out_impl` 调用 `sync_omp_threads()`，把 libomp
   team 同步为 `at::get_num_threads()`。

### 1.4 噪声处理

- 共享箱上单次运行与中位数可差 ~3x；所有正式数据用 **交错 aten/hpc + 多次取
  中位数**，而不是分别跑两个进程。
- 模型场景矩阵：25 iters 中位数，固定 node 5。
- 早期 608 线程数据只作历史参考，不应作为验收依据。

---

## 2. 权威结果：T11 全量矩阵

### 2.1 31 场景 × 7 线程

比值 = aten 时间 / hpc 时间，越大越好。读法示例：`S1 8t = 3.55x` 表示 8 线程下
aten 用了 3.55 倍时间；`S4 8t = 0.87x` 表示比 aten 慢 13%。

| 场景 | 1t | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | 4.46x | 4.62x | 3.95x | 3.55x | 2.97x | 6.69x | 8.37x |
| P1 h[src] i64 C=64 E=1M | 4.42x | 4.43x | 3.83x | 4.02x | 3.52x | 6.97x | 5.96x |
| P2 pos[src] A=3 E=2M | 2.05x | 2.12x | 2.06x | 2.90x | 1.78x | 6.64x | 8.73x |
| P3 x[mask] nnz=60042 C=16 | 2.23x | 1.82x | 1.53x | 1.34x | 1.87x | 3.76x | 4.20x |
| S4 x[perm] f64 N=200k | 1.26x | 1.36x | 1.09x | 0.87x | 0.76x | 1.35x | 1.01x |
| 1D f64 K=2M | 1.26x | 1.18x | 1.15x | 1.11x | 1.03x | 1.08x | 1.51x |
| 1D f64 K=10M | 1.17x | 1.06x | 1.03x | 1.06x | 1.03x | 2.01x | 1.95x |
| 1D f32 K=2M | 1.07x | 1.41x | 1.40x | 1.35x | 1.24x | 1.48x | 2.17x |
| 1D f32 K=10M | 1.06x | 1.11x | 1.11x | 1.14x | 1.16x | 2.28x | 3.92x |
| multi x[i,j] f32 100k | 1.19x | 1.09x | 1.05x | 0.96x | 0.89x | 1.24x | 1.25x |
| rowcopy A=4 f32 K=200k | 2.41x | 1.71x | 1.52x | 1.35x | 1.23x | 1.96x | 2.10x |
| 1D pure f32 K=2M (T5) | 1.24x | 1.28x | 1.28x | 1.25x | 1.22x | 1.29x | 3.22x |
| 1D f32 K=10M idx64 | 1.20x | 1.26x | 1.26x | 1.27x | 1.27x | 2.46x | 4.50x |
| 1D f64 K=10M idx32 | 1.06x | 1.12x | 1.11x | 1.11x | 1.13x | 2.86x | 2.97x |
| 1D float16 K=2M | 1.53x | 1.56x | 1.31x | 1.36x | 1.22x | 1.34x | 2.40x |
| 1D bfloat16 K=2M | 1.35x | 1.35x | 1.48x | 1.35x | 1.26x | 1.43x | 2.05x |
| 1D int8 K=2M | 0.98x | 0.99x | 1.02x | 0.98x | 0.92x | 2.03x | 1.86x |
| 1D uint8 K=2M | 1.06x | 1.05x | 0.98x | 0.99x | 0.90x | 2.01x | 1.87x |
| 1D complex64 K=2M | 0.95x | 1.00x | 1.00x | 0.97x | 0.97x | 1.78x | 2.76x |
| 1D int32 K=2M | 1.37x | 1.39x | 1.53x | 1.43x | 1.32x | 2.49x | 3.93x |
| 1D int64 K=2M | 1.26x | 1.12x | 0.87x | 1.22x | 1.12x | 1.57x | 2.36x |
| rowcopy A=2 f32 K=200k | 1.50x | 1.55x | 1.40x | 1.33x | 1.23x | 1.91x | 3.03x |
| rowcopy A=3 f32 K=200k | 1.90x | 2.06x | 1.65x | 1.52x | 1.34x | 3.25x | 3.34x |
| rowcopy A=16 f32 K=200k | 3.58x | 4.65x | 3.92x | 3.13x | 2.34x | 4.15x | 5.80x |
| noncontig self A=16 stride2 | 1.45x | 1.20x | 1.53x | 1.42x | 1.32x | 1.73x | 3.85x |
| noncontig idx K=1M | 1.16x | 1.06x | 1.19x | 1.13x | 1.01x | 1.10x | 1.57x |
| mask 1D nnz=1199436 | 1.20x | 1.04x | 1.25x | 1.30x | 1.30x | 1.93x | 3.76x |
| mask empty N=1M | 1.69x | 1.07x | 0.74x | 1.15x | 1.01x | 1.08x | 1.12x |
| mask noncontig nnz=600108 | 0.98x | 0.96x | 0.93x | 0.89x | 0.84x | 1.25x | 1.17x |
| col x[:,i] C=64 K=64 | 2.02x | 1.88x | 1.69x | 1.43x | 1.12x | 1.55x | 1.87x |
| 1D f32 K=2M range2048 | 1.22x | 1.16x | 1.09x | 1.02x | 0.87x | 1.41x | 1.46x |

### 2.2 低于 1.00x 的单元与解释

- `x[:,i]`：**已修复**。T11 的 A==1 预校验紧凑循环把 0.18x/0.40x 提升到
  1.12-2.02x（1-36t），不再是退化项。
- `1D int8/uint8`：0.90-1.06x。1B 元素随机读，64B 行只用 1B（64x 放大）；aten
  也是标量，0.9-1.0x 即硬件上限，不值得继续投入。
- `mask noncontig`：0.84-0.98x。T8 只支持连续 mask，非连续时回退 `nonzero`
  （串行段 + 中间 int64 下标）。按 stride 扩展 T8 即可修复。
- `S4 8/16t` 0.87/0.76x、`multi 8/16t` 0.96/0.89x：每线程工作量 <100KB 的
  小规模强扩展，线程启动/调度成本大于收益；1-4t 或 32/36t 均恢复 >1x，不是
  持续趋势。
- `1D int64 4t` 0.87x、`range2048 16t` 0.87x、`mask empty 4t` 0.74x：单点抖动；
  相邻线程 >1.2x，或绝对时间只有几微秒，判为噪声。
- **高线程倍率为何很高**：S1/P1/P2 在 32/36t 出现 5-8x，一部分是 aten 在带宽
  饱和后自身变慢，不是本算子还能线性扩展。看 `degradation-analysis.md` §6.2。
- `S1 1t 4.46x`、`P1 1t 4.42x`、`rowcopy A=4 1t 2.41x`、`rowcopy A=16 1t
  3.58x`：T11 把 1 线程 A>=4 从 `gather_fused`（标量）改回 raw rowcopy（SVE）
  的结果；旧代码在 1t 对所有 A>1 都走 fused，丢掉了 SVE 整块拷贝。

### 2.3 小 K 真实路径（两进程）

K<=1024 时单次调用只有几微秒，同进程 pybind 包装会污染比值，因此用
`bench_smallk_driver.py` 交替启动 aten/hpc 两个子进程，各自跑真实 `x[idx]`，
每格取中位数。T9 的 `n*A<16384` 串行阈值修复后，2D A=16 K=4096 不再误判串行：

| 形态 | K | 1t | 8t | 36t | 全线程范围 |
|---|---|---|---|---|---|
| 1D A=1 | 0-1024 | 1.31-1.95x | 1.35-2.03x | 1.34-2.03x | 1.31-2.03x |
| 1D A=1 | 4096 | 0.88x | 0.98x | 1.00x | 0.88-1.00x |
| 1D A=1 | 8192 | 1.07x | 0.99x | 0.60x | 0.60-1.07x |
| 1D A=1 | 65536 | 0.96x | 0.88x | 0.64x | 0.64-0.96x |
| 2D A=16 | 128-1024 | 2.35-3.02x | 1.60-2.42x | 1.02-2.40x | 1.02-3.02x |
| 2D A=16 | 4096 | 3.51x | 1.78x | 1.03x | 1.03-3.51x |
| 2D A=16 | 8192 | 3.74x | 1.39x | 0.46x | 0.46-3.74x |
| 2D A=16 | 65536 | 4.23x | 2.72x | 1.15x | 1.15-4.23x |

> K=0-1024：hpc 6-14µs vs aten 12-19µs，固定开销低，稳定 1.3-2.0x。
> K>=8192 的 1D 高线程、2D K=8192 的 16-36t 出现 0.46-0.8x：总数据只有
> 32KB-512KB，每线程只有几 KB，线程启动/调度成本超过收益（绝对时间仅
> 40-90µs）。这是“小规模不该并行”的阈值问题，不是内核退化；下一步按每线程
> 字节数限制有效线程数（`degradation-analysis.md` §6.3）。

---

## 3. 历史结果（按优化轮次）

以下数据保留用于追溯优化过程。**除特别说明外，2-36t 的线程标签都受 §1.3
的双 ICV 影响，只有 1t/38t 可信**；性能结论以 §2 为准。

### 3.1 早期 1/608 线程摘要（T0 构建）

`bench_index.py` 全矩阵，3 warmup + 10 iters，每格 6 次完整运行的中位数，
`NT=0`，`PF=16`。608 线程是整机最坏情况，受 5+ 用户干扰严重。

| 场景 | 形状 | dtype | 1 线程 | 608 线程 |
|---|---|---|---|---|
| 1D gather | f64 K=2M | float64 | 0.82x | 1.77x |
| 1D gather | f64 K=10M | float64 | 0.75x | 0.36x |
| 1D gather | f32 K=2M | float32 | 0.87x | 0.49x |
| 1D gather | f32 K=10M | float32 | 0.78x | 0.24x |
| 1D gather | i64 K=2M | int64 | 0.88x | 0.27x |
| 1D gather | i32 K=2M | int32 | 1.10x | 0.58x |
| S3 列排列 | x[:,idx] f32 B=256 K=100k | float32 | 3.63x | 0.38x |
| 多索引 | x[i,j] f32 I=J=2048 | float32 | 0.77x | 0.74x |
| RowCopy | x[idx,:] f32 C=32 E=1M | float32 | 3.50x | 0.32x |
| **S1 spconv 行 gather** | features[inp_idx] f32 C=64 E=1M | float32 | **2.50x** | **1.07x** |
| **P1 图消息** | h[src] i64 f32 C=64 E=1M | float32 | **2.98x** | **1.35x** |
| **P2 坐标** | pos[src] i32 f32 A=3 E=2M | float32 | **1.94x** | **0.87x** |
| **P3 mask 选择** | x[mask] f32 C=16 | float32 | **1.12x** | **0.89x** |
| **S4 大排列** | x[perm] f64 N=200k | float64 | **1.81x** | **5.55x** |

### 3.2 早期全矩阵（含带宽，T0 构建）

| 形状 | dtype | 线程 | aten GB/s | hpc GB/s | 加速比 |
|---|---|---|---|---|---|
| 1D f64 K=2M | float64 | 1 | 0.5 | 0.4 | 0.82x |
| 1D f64 K=2M | float64 | 608 | 3.3 | 6.1 | 1.77x |
| 1D f64 K=10M | float64 | 1 | 0.4 | 0.3 | 0.75x |
| 1D f64 K=10M | float64 | 608 | 6.5 | 2.8 | 0.36x |
| 1D f32 K=2M | float32 | 1 | 0.3 | 0.2 | 0.87x |
| 1D f32 K=2M | float32 | 608 | 1.0 | 0.5 | 0.49x |
| 1D f32 K=10M | float32 | 1 | 0.2 | 0.2 | 0.78x |
| 1D f32 K=10M | float32 | 608 | 1.6 | 0.3 | 0.24x |
| 1D i64 K=2M | int64 | 1 | 0.4 | 0.4 | 0.88x |
| 1D i64 K=2M | int64 | 608 | 1.9 | 0.7 | 0.27x |
| 1D i32 K=2M | int32 | 1 | 0.2 | 0.2 | 1.10x |
| 1D i32 K=2M | int32 | 608 | 0.6 | 0.4 | 0.58x |
| x[:,idx] f32 B=256 K=100000 | float32 | 1 | 0.6 | 2.2 | 3.63x |
| x[:,idx] f32 B=256 K=100000 | float32 | 608 | 2.7 | 0.9 | 0.38x |
| x[i,j] f32 I=2048 J=2048 | float32 | 1 | 121.9 | 95.1 | 0.77x |
| x[i,j] f32 I=2048 J=2048 | float32 | 608 | 129.9 | 95.7 | 0.74x |
| x[idx,:] f32 C=32 E=1M | float32 | 1 | 0.6 | 1.9 | 3.50x |
| x[idx,:] f32 C=32 E=1M | float32 | 608 | 2.4 | 0.7 | 0.32x |
| S1 features[inp_idx] C=64 E=1M | float32 | 1 | 0.5 | 1.3 | 2.50x |
| S1 features[inp_idx] C=64 E=1M | float32 | 608 | 4.8 | 5.3 | 1.07x |
| P1 h[src] i64 C=64 E=1M | float32 | 1 | 0.3 | 1.0 | 2.98x |
| P1 h[src] i64 C=64 E=1M | float32 | 608 | 5.7 | 8.4 | 1.35x |
| P2 pos[src] i32 A=3 E=2M | float32 | 1 | 0.1 | 0.3 | 1.94x |
| P2 pos[src] i32 A=3 E=2M | float32 | 608 | 1.9 | 1.7 | 0.87x |
| P3 x[mask] nnz≈60k C=16 | float32 | 1 | 0.9 | 0.9 | 1.12x |
| P3 x[mask] nnz≈60k C=16 | float32 | 608 | 0.2 | 0.2 | 0.89x |
| S4 x[perm] f64 N=200000 | float64 | 1 | 0.3 | 0.5 | 1.81x |
| S4 x[perm] f64 N=200000 | float64 | 608 | 0.5 | 2.9 | 5.55x |

> P2 1 线程在 6 批中为 1.73/1.82/1.84/2.04/3.50/4.33，中位数 1.94x；
> 3.50/4.33 是负载离群点。§3.4 的定点门禁（同形状 30 iters × 5 轮）复现
> **1.84x**，因此 P2 的代表值是 **1.8-2.0x**，不是 3.5x。

### 3.3 NT 开关对比（608 线程，代表性行）

| 行 | NT=0 中位数 | NT=1 |
|---|---|---|
| 1D f64 K=2M | 1.77x | 2.49x |
| 1D f64 K=10M | 0.36x | 0.28x |
| 1D f32 K=2M | 0.49x | 0.84x |
| 1D i64 K=2M | 0.27x | 0.07x |
| S1 rowcopy E=1M | 1.07x | 1.05x |
| P1 h[src] E=1M | 1.35x | 0.36x |
| P2 pos[src] E=2M | 0.87x | 1.95x |
| S4 x[perm] N=200k | 5.55x | 5.47x |

NT=1 与 NT=0 的差值在同一负载下会正负翻转，且不超过本机噪声带（例如 P1 在
NT=0 单次测量可落在 0.36-2.21x）。**没有可复现收益，NT 默认关闭**（按设计 §8
保留 opt-in）。

### 3.4 聚焦重复门禁（模型场景，20-30 iters × 3-5 轮）

| 场景 | 线程 | 中位数 | 最小 | 最大 |
|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=400k i32 | 1 | 3.99x | 3.99x | 4.00x |
| S1 features[inp_idx] C=64 E=400k i32 | 608 | 1.00x | 0.91x | 1.88x |
| P1 h[src] i64 C=64 E=2M | 1 | 4.05x | 3.08x | 4.06x |
| P1 h[src] i64 C=64 E=2M | 608 | 1.28x | 1.22x | 1.37x |
| P2 pos[src] i64 A=3 E=2M | 1 | 1.95x | 1.81x | 1.99x |
| P2 pos[src] i64 A=3 E=2M | 608 | 0.38x | 0.37x | 0.53x |
| P2 pos[src] i32 A=3 E=2M | 1 | 1.84x | 1.81x | 1.85x |
| P2 pos[src] i32 A=3 E=2M | 608 | 0.92x | 0.62x | 1.92x |
| P3 x[mask] C=32 | 1 | 1.61x | 1.61x | 1.62x |
| P3 x[mask] C=32 | 608 | 0.46x | 0.41x | 1.03x |
| S4 x[perm] f64 N=1M | 1 | 0.98x | 0.98x | 0.98x |
| S4 x[perm] f64 N=1M | 608 | 2.42x | 0.36x | 2.67x |
| 1D f64 K=1M | 1 | 1.61x | 1.59x | 1.62x |
| 1D f64 K=1M | 608 | 1.73x | 1.66x | 1.82x |
| 1D f64 K=10M | 1 | 0.72x | 0.72x | 0.94x |
| 1D f64 K=10M | 608 | 0.31x | 0.20x | 0.36x |

> P2 说明：§3.2 的 P2 行用 **int32** 下标（32MB 负载），上面门禁用 **int64**
> 下标（40MB）。两者 1 线程都在 ~1.8-2.0x，说明 dtype 不是 §3.2 旧 3.50x 的
> 原因；那是 3 次采样的假象，§3.2 已改成 6 轮中位数 1.94x。
>
> 同一门禁批次内 1 线程很稳（S1/P1/P2/P3/S4-1M 波动 <0.1x），但共享箱上矩阵
> 单元会随负载摆动（如 P2 1t 在 6 批中 1.73-4.33x）；608 线程完全被噪声支配。

### 3.5 集成 / 覆盖验证

- **默认覆盖**：`import index_sve` 把 `torch.ops.hpc_index.index` 注册为
  `aten::index.Tensor` 的 CPU kernel。导入后所有 CPU 张量的 `x[idx]`、
  `x[:, i]`、`x[i, j]`、bool mask、N-D 整数下标都走本算子，调用点零改动。
  已用 dispatcher dump（CPU key 从 `RegisterCPU_1.cpp` 移到覆盖实现）和两进程
  true-aten 对拍（1D f64/f32 K=1M、S1 行 gather int32 2M、half K=5000、K<1024）
  验证。
- **一次性告警**：首次导入会打印 "Overriding a previously registered kernel..."
  （来自 `torch.library.impl`），预期且无害，故意不压制。
- **autograd**：`aten::index.Tensor` 的 backward 公式（IndexBackward）独立于
  CPU kernel，梯度仍落到被索引位置；已与 true-aten 对比值/形状/dtype。
- **运行时契约**：`LD_LIBRARY_PATH` 必须同时包含 `torch/lib` 和 BiSheng 的
  `libomp.so` 目录，否则 import 时 dlopen 失败。
- **NT 开关**：`HPC_INDEX_NT=1` 为 opt-in，默认关闭。
- **单覆盖限制**：torch 2.10 禁止同进程注册第二个 Python CPU kernel，否则抛
  `already a kernel registered from python`。这是默认覆盖设计（A6）的固有约束。
- **基准逃生开关**：`HPC_INDEX_NO_OVERRIDE=1` 在 import 时跳过覆盖，让
  benchmark/test 里的 `x[i]` 保持真 aten（`bench_pinned.py` / `bench_gate.py`
  依赖它）。

### 3.6 定点复测（1 / 38 线程）

方法：`bench_pinned.py` / `bench_gate.py`，用 `os.sched_setaffinity` 固定到
node 5（CPU 190-227，运行前 2 秒采样确认 <5% busy），20 iters × 3 轮取中位数，
全部 `HPC_INDEX_NO_OVERRIDE=1`。

| 场景 | 1 线程（min/中位/max） | 38 线程（min/中位/max） |
|---|---|---|
| S1 features[inp_idx] C=64 E=1M | 3.79/3.96/4.02x | 1.71/1.72/1.73x |
| P1 h[src] i64 C=64 E=1M | 3.75/3.86/3.87x | 1.55/1.94/1.95x |
| P2 pos[src] i32 A=3 E=2M | 1.75/1.76/1.80x | 0.85/0.86/0.87x |
| P3 x[mask] nnz≈60k C=16 | 1.11/1.11/1.11x | 0.34/0.35/0.37x |
| S4 x[perm] f64 N=200k | 0.93/0.94/0.95x | 0.51/0.53/0.71x |
| 1D f64 K=2M | 0.79/0.80/0.86x | 0.76/0.77/1.62x |
| 1D f64 K=10M | 0.73/0.74/1.69x | 0.78/0.80/2.06x |
| 1D f32 K=2M | 0.87/0.88/0.90x | 0.71/0.71/1.53x |
| 1D f32 K=10M | 0.68/0.69/0.70x | 0.18/0.18/0.19x |

> max 列的尖峰（如 1D f64 K=10M 1t 1.69x、1D f32 K=2M 38t 1.53x）是共享箱上
> 其他用户的瞬时负载，即使“静默窗口”也存在；以中位数为准。

**这一轮定下的结论**：

- **1 线程收益真实且稳定**：S1 ~4.0x、P1 ~3.9x、P2 ~1.8x、P3 ~1.1x，
  `x[:,idx]`/`x[idx,:]` ~3.5x（node 4 全矩阵：S3 3.48x、RowCopy 3.50x）。
  RowCopy 是可靠加速项，§3.4 的 S1 3.99x 在固定后完全复现。
- **单节点多线程模型行比 608 线程稳定**：S1 1.72x、P1 ~1.9x（§3.1 的
  1.07x/1.35x 是共享箱重负载下的下界）。
- **f32 K=10M 的旧 “MLP 低” 归因不成立**：历史 0.18x 混淆了两个效应——
  `gather_direct` 的逐元素 `at::Tensor` 开销（T11 已修）和线程标签失真
  （§1.3）。同构建 A/B 见 §3.7。
- **P3 mask 38 线程 0.35x 是 T8 之前的构建**：mask 走 `nonzero` → 标量
  gather，并行度差；1 线程 1.11x。T8 单遍后 T11 复测 36t 为 4.20x（§2.1）。

#### 3.6.1 旧线程扫描（2/4/8/16/32/36t，**已作废**）

> **作废说明**：扩展实际使用 libomp 默认 38 worker（§1.3），以下 2-36t 列的
> 相对收益被高估，只有 1t/38t 有效。保留表格仅用于历史追溯，不要再引用。
> 修正后的矩阵见 §2.1。

| 场景 | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | **33.7x** | 19.4x | 5.6x | 4.3x | 2.3x | 2.2x |
| P1 h[src] i64 C=64 E=1M | **30.5x** | 17.5x | 8.7x | 4.4x | 2.3x | 2.1x |
| P2 pos[src] i32 A=3 E=2M | **19.7x** | 6.3x | 5.8x | 1.9x | 3.4x | 1.1x |
| P3 x[mask] nnz≈60k C=16 | 1.2x | **1.5x** | 1.4x | 1.2x | 1.1x | 1.1x |
| S4 x[perm] f64 N=200k | **20.7x** | 13.0x | 8.0x | 4.1x | 8.6x | 11.6x |
| 1D f64 K=2M | **6.8x** | 3.0x | 1.8x | 1.9x | 2.6x | 1.9x |
| 1D f64 K=10M | **10.5x** | 5.2x | 2.7x | 1.4x | 0.9x | 0.8x |
| 1D f32 K=2M | **10.6x** | 2.8x | 1.5x | 4.1x | 1.8x | 2.6x |
| 1D f32 K=10M | 1.7x | 1.2x | 0.7x | 0.3x | 0.2x | 0.2x |

历史解读（均已撤回，仅作背景）：低线程 RowCopy 在 2-4t 就能打满单 NUMA 节点
带宽（~60GB/s），但表里 20-34x 是标签失真；修正后 S1/P1 2t 只有 4.4-4.6x。
1D f32 K=10M 在 2-4t 是 4B 纯 gather 唯一有竞争力的窗口，8t 后受 64B 行放大
回落。32/36t 波动大，且本机无 SMT，主要受共享箱邻居影响。

### 3.7 4B gather 归因复核（T11）

早期独立探针曾测到“SVE 4B ~0.75GB/s vs 标量 3.4GB/s”，据此提出“SVE 4B
MLP 低”。但该结论在真实内核上不复现。用真实内核、同一随机下标模式在 node 5
重测（交错取中位数）：

| 源大小 | 线程 | SVE 4B gather | 标量循环 |
|---|---|---|---|
| 8 MB | 8 | 1.85 ms | 1.86 ms |
| 256 MB | 38 | 7.46 ms | 7.45 ms |

注：8t 的 HPC 行同样使用 libomp 默认 38 worker（§1.3），但 SVE 与标量是同配置，
A/B 结论不受影响。

SVE 4B 与标量持平，说明瓶颈**不是** SVE gather 指令的 MLP，而是：随机取一个
元素要拉一条 64B cache line，只用到 4B（16x 放大；f64 是 8x）。成本由行流量和
DRAM 延迟决定。这也解释了为什么 8B 路径更快（同样行数、有用字节翻倍），以及
为什么换标量回退没用。真正的缓解只能减少行流量：小值域桶化（T4）、预取/软件
流水、行 gather（A>=4）。旧探针数字已撤回，仅作历史背景。

### 3.8 AI4S 负载优化轮（T0-T4）

> **历史线程标签警告**：本节 8t/32t 列在 T11 前测得，HPC 实际使用 libomp
> 默认 38 worker（§1.3），只有 1t/38t 可信；结论以 §2 为准。

背景：用户反馈真实 AI4S 模型端到端无提升甚至劣化。`bench_ai4s.py` 对
`index_ai4s_workload特点和优化方向建议.md` §3.1 的 M1-M6 六种形态做定点测试，
实现 T0-T4（自适应粒度、rowcopy 阈值 A>=4、M1 skip-linearize、T3 融合单遍、
T4 小值域桶化）。方法：aten `x[i]` 与 hpc 交错，50 iters 中位数，node 5，
`HPC_INDEX_NO_OVERRIDE=1`；GB/s =（索引字节 + 数据读写字节）/ 时间。

| 场景（形态） | dtype | 1t | 8t | 32t |
|---|---|---|---|---|
| M1 weight.view(K,C,o)[ki] (27,32,32) E=2k | f32 | 9.1x | 1.3x | 3.9x |
| M1 mid (27,64,64) E=20k i64 | f32 | 9.9x | 0.6-1.0x | 1.0x |
| M1 mid i32 (27,64,64) E=20k | f32 | 7.9x | 2.7x | 0.95x |
| M2 feat[r] (20000,64) E=100k | f32 | 3.9x | 2.6x | 0.6-7.1x |
| M2 int32 (20000,64) E=100k | f32 | 4.0x | 4.0x | 1.2x |
| M2 smallC (20000,8) E=100k | f32 | 0.93x | 0.95x | 0.92x |
| M2 largeC (20000,256) E=100k | f32 | 2.8x | 6.0x | 3.1x |
| M3 keys[order] int64[2048] E=54k i64 | i64 | 1.4x | 0.6-1.3x | 8.0x |
| M3 keys[order] int64[2048] E=54k i32 | i64 | 2.6x | 1.3-2.2x | 11.6x |
| M4 feat[inverse] (200,64) E=20k | f32 | 3.6x | 3.8x | 4.8x |
| M6 2D idx (2048,3)->flat | f32 | 2.0x | 0.9x | 1.0x |

> 8t/32t 在共享箱上逐次摆动很大（同一个 M3 场景，aten 单独测量可从 104µs
> 到 426µs）；交错的 aten/hpc 中位数才是可靠值。M1 i64 与 M3 i64 走 int64
> linearize 路径（非 int32 快路径），高线程倍率因此低于 i32 行。

**T0-T4 的结论**：

- **M1（最热，14 个模型累计 27270ms）**：i64 E=2k-20k 1 线程 8-10x、8t
  1.3-2.7x，明显好于优化前“无收益”。i32 变体在 8t 也赢。
- **M2 feat[r]（spconv）**：1t 3.9-4.0x、8t 2.6-4.0x；C=8 短行 ~0.93x
  （A=8 走 rowcopy，但行太短，收益有限）。
- **M3 keys[order] 小值域查表**：T4 raw 桶化把优化前 0.2x 变成 1t 1.4-2.6x；
  32t 的 8-12x 含标签失真，不能直接引用。
- **M4 feat[inverse] 高重复**：1t 3.6x；8-32t 3.8-4.8x（同样受标签影响）。
- **4B f32 K=10M 仍是硬件上限**：值域远大于 2048，桶化不启用，仍走 SVE
  gather；1t ~0.7x，高线程 0.2x 是旧标签，T11 实测 1.06-1.16x（§2.1）。

### 3.9 T0-T5 全线程扫描（**已作废**）

> **作废**：2-36t 列同 §3.6.1，HPC 实际使用 38 worker；只有 1t/38t 可信。
> 修正矩阵见 §2.1。下表仅作历史存档。

`bench_threads.py`，node 5（CPU 190-227），交错 aten/hpc 中位数，30 iters：

| 场景 | dtype | 1t | 2t | 4t | 8t | 16t | 32t | 36t | 38t |
|---|---|---|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | f32 | 1.06x | 30.8x | 40.2x | 21.0x | 10.6x | 8.6x | 7.5x | 7.2x |
| P1 h[src] i64 C=64 E=1M | f32 | 1.06x | 41.6x | 31.7x | 13.3x | 7.2x | 5.0x | 5.2x | 4.4x |
| P2 pos[src] A=3 E=2M | f32 | 1.90x | 65.7x | 37.4x | 8.9x | 6.4x | 9.8x | 6.4x | 7.1x |
| P3 x[mask] nnz≈60k C=16 | f32 | 0.96x | 1.90x | 1.65x | 1.33x | 1.17x | 1.33x | 1.32x | 1.32x |
| S4 x[perm] f64 N=200k | f64 | 1.31x | 6.6x | 5.4x | 3.4x | 3.0x | 3.3x | 5.0x | 5.8x |
| 1D f64 K=2M | f64 | 1.30x | 23.3x | 10.7x | 7.4x | 4.5x | 3.1x | 2.0x | 1.9x |
| 1D f64 K=10M | f64 | 1.21x | 18.9x | 12.1x | 5.3x | 3.1x | 1.6x | 2.0x | 1.5x |
| 1D f32 K=2M | f32 | 1.08x | 27.7x | 13.5x | 9.1x | 7.6x | 4.4x | 4.0x | 2.5x |
| 1D f32 K=10M | f32 | 1.07x | 13.9x | 16.6x | 6.0x | 2.6x | 2.6x | 2.0x | 2.1x |
| multi x[i,j] f32 100k | f32 | 1.19x | 2.9x | 4.6x | 5.5x | 6.9x | 5.5x | 5.3x | 4.4x |
| rowcopy A=4 f32 K=200k | f32 | 1.28x | 7.9x | 7.2x | 4.2x | 16.5x | 13.4x | 12.8x | 11.1x |
| 1D pure f32 K=2M (T5) | f32 | 1.27x | 23.5x | 10.8x | 7.3x | 2.7x | 2.1x | 1.9x | 2.8x |

历史解读（已撤回）：

- 旧结论“每个线程都 ≥1.06x”在 T11 后不成立：`x[:,i]`、非连续 mask、
  int8/uint8 仍 ~0.9x（§2.2）。`x[:,i]` 的 1t 惩罚是真实的逐元素开销，不是
  608 线程噪声。
- “低线程 13-66x”全部是标签失真；修正后 S1 2t 4.62x、P2 2t 2.12x、
  1D pure 2t 1.28x。
- P3 mask 在这版构建里 1-1.9x 且随线程基本平坦，是 `nonzero`→标量两段串行
  所致；T8 单遍后 1t 2.2x，旧路径在 2-32t 提升 3.6-5.4x（同样含标签失真）。

### 3.10 torch.compile 交互（T0-T6 spike）

`torch.compile` 默认不调用运行时 index 内核：Inductor 有 `aten.index` 的
lowering（`torch/_inductor/lowering.py:3878`），会生成自己的
`cpp_fused_index_0`，绕过 dispatcher 覆盖。用 `force_fallback(aten.index.Tensor)`
可把它钉回运行时调用，重新命中本算子。实测（f32 K=2M、int64 下标、8 线程、
node 5，交错中位数，关闭 inductor 磁盘缓存）：

| 路径 | 时间 | 相对 eager hpc |
|---|---|---|
| eager（本算子覆盖） | 2.01 ms | 1.00x |
| compile 默认（Inductor 生成） | 5.30 ms | 慢 2.64x |
| compile + force_fallback（回到本算子） | 2.57-2.67 ms | 1.28-1.33x |
| 真 aten 参考（`Tensor_out`，绕过覆盖） | 8.9 ms | — |

规模曲线（compiled-fallback / eager）：1k 4.2x、100k 3.2x、2M 1.28x、
8M **1.07x**。小 N 差距主要是 `torch.compile` 固定调用开销（同机平凡 `clone`
约 0.3ms）。

注意：**inductor 磁盘缓存会让 fallback 补丁静默失效**（图哈希不含 lowering
注册；`cache_key_tag` 只影响 autotune cache）。复现须用独立
`TORCHINDUCTOR_CACHE_DIR` 并 `force_disable_caches=True`。完整分析见
`docs/torch-compile.md`；固化实现见 §3.13。

### 3.11 T8：mask 单遍融合（跳过 nonzero）

> **历史线程标签警告**：本节 2-36t 列在 T11 前测得，HPC 实际使用 libomp
> 默认 38 worker（§1.3），只有 1t/38t 可信；结论以 §2 为准。

旧路径 `x[mask]`：`normalize_index` 先 `nonzero()` 物化 int64 位置下标，再
linearize + gather（两段串行）。T8 在入口识别“单 mask + 覆盖前 m 维 + mask/self
连续 + CPU”走 `gather_mask`：chunk 计数 → 前缀和 → 第二遍直接压实并整行拷贝；
小掩码（<1MB）计数串行，避免并行任务开销。

P3（N=200k C=16 d=0.3 nnz≈60k，node 5，交错中位数，`HPC_INDEX_NO_OVERRIDE=1`）：

| 线程 | 旧 hpc | 新 hpc | 真 aten | 新/旧 | aten/新 |
|---|---|---|---|---|---|
| 1 | 4.86 ms | 2.19 ms | 4.67 ms | 2.2x | 2.1x |
| 2 | 4.19 | 0.77 | 4.64 | 5.4x | 6.0x |
| 4 | 2.87 | 0.70 | 4.77 | 4.1x | 6.8x |
| 8 | 2.66 | 0.73 | 5.91 | 3.6x | 8.0x |
| 16 | 2.69 | 0.74 | 5.33 | 3.6x | 7.3x |
| 32 | 3.66 | 0.70 | 5.06 | 5.2x | 7.2x |
| 38 | — | 0.66 | 2.99 | — | 4.6x |

其它形态（8t，aten/新）：C=64 d=0.1 1.9-2.1x、1D A=1 1M 1.1-2.2x、全维 mask
512² 1.6-1.8x、2D mask→8 2.1-2.5x、byte 1.5-1.7x、空 mask 0.9x。共享箱 aten
波动大，hpc 0.66-0.77ms 稳定。

`index.Tensor_out` + mask 未接融合路径（out 变体罕见），仍走原 nonzero 路径。

### 3.12 T9：小 K 精简路径 + C++ 覆盖（真实 `x[idx]`）

> **历史线程标签警告**：本节是 T9 构建的历史数据，2-36t 列在 T11 前测得，
> HPC 实际使用 libomp 默认 38 worker（§1.3），只有 1t/38t 可信。T11 两进程
> 复测见 §2.3。

两处改动：入口新增 `small_single_index`（单整数下标、末维 K<=4096、连续、CPU，
绕过 normalize）；aten 覆盖从 Python `torch.library.impl` 移到 C++
`TORCH_LIBRARY_IMPL(aten, CPU, m)`，去掉每次调用的解释器往返。

真实 `x[idx]`（node 5，两进程交替 3 次取中位；共享箱 aten 绝对值波动大）：

| 形态 | 1t / 8t |
|---|---|
| 1D A=1 K=0 / 1 / 64 | 2.3x / 2.2x / 2.2x |
| 1D A=1 K=512 / 1023 / 1024 | 2.0x / 1.8x / 1.6x |
| 2D A=16 K=128 / 1023 / 1024 | 2.4x / 2.4-2.7x / 1.4x |

`K=1024` 断崖修复（8t，源 1M，随机大值域）：0.37x→1.64x；`K=2048`
0.47x→1.38x；`K=4096` 0.55x→0.99x。`K=8192..16384` 仍走原路径（0.9-1.2x），
`K=65536` 3x。C++ 注册自身：真实 `x[idx]` K=0 27.5→11.9µs、K=1023
36.9→24.4µs（同脚本前后对比）。

注意：扩展加载时 dispatcher 会打印一次 "Overriding a previously registered
kernel..."（预期）；`HPC_INDEX_NO_OVERRIDE=1` 须在 import 前设置才能关闭覆盖。

### 3.13 T10：torch.compile opt-in（HPC_INDEX_COMPILE=1）

> **历史线程标签警告**：本节 2-36t 列在 T11 前测得，HPC 实际使用 libomp
> 默认 38 worker（§1.3），只有 1t/38t 可信；结论以 §2 为准。

把 §3.10 的 spike 固化成 opt-in：`hpc_index::*` 改为 schema-only + CPU kernel
后注册 fake（整数下标推形状），再用
`register_lowering(aten.index.Tensor)(fallback_handler(...))` 永久钉成运行时
调用；Inductor 磁盘缓存切到 `<base>/hpc_index`，避免命中未打补丁的旧产物。
`HPC_INDEX_COMPILE=1` 须在 `import index_sve` 前设置，与
`HPC_INDEX_NO_OVERRIDE=1` 冲突时直接报错。

性能沿用 §3.10 的 force_fallback 测量（机制相同）：f32 K=2M/8t，
compiled-fallback 2.57-2.67ms vs eager 2.01ms（1.28-1.33x），规模 >=8M 收敛到
1.07x；compile 固定开销约 0.3ms，K≲100k 不建议 compile。默认关闭。

正确性（`tests/test_compile_optin.py`，3 例）：整数下标与 mask 两条 compile
路径输出与 eager 逐元素相等；profiler 中为 `aten::index`（无
`cpp_fused_index_*`）；cache 目录后缀为 `hpc_index`。mask 的 fake 因输出长度
数据相关不提供（compile 走 aten fallback，由 Inductor/aten 处理）。

---

## 4. 复现命令

```bash
source ~/.bashrc && conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=/home/z00889957/miniconda3/envs/torch210_env/lib/python3.12/site-packages/torch/lib:/home/envs/HPCKit_26.1.RC1/HPCKit/26.1.RC1/compiler/bisheng/lib:$LD_LIBRARY_PATH

# 权威矩阵（空载 node，31 场景 × 7 线程，约 10 分钟）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_threads.py

# 小 K 两进程真实路径
python benchmarks/bench_smallk_driver.py

# 早期全矩阵（共享箱，仅参考）
python benchmarks/bench_index.py

# 定点门禁（1/38t）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,38 \
  python benchmarks/bench_gate.py
```

注意：扩展需 `-fopenmp` 且 `LD_LIBRARY_PATH` 含 BiSheng `libomp.so`；在仓库根
目录直接 `python -c` 会被本地 `torch/` 源码遮蔽，建议在 `/tmp` 或安装后的环境
运行。

---

## 5. 与其它文档的关系

- 逐场景微架构分析、线程扩展效率、perf 证据：`docs/degradation-analysis.md`。
- 内核实现与选择门槛：`docs/kernels.md`、`docs/architecture.md`。
- 优化轮次的问题/改动/commit：`docs/optimizations.md`。
- 慢场景与“为什么不 fallback”：`docs/fallback-analysis.md`。
- 方法学摘要与使用建议：`docs/performance.md`、`docs/status.md`。
