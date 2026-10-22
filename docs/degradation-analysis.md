# 性能退化分析：线程扩展、微架构与逐案定位

本文回答三个问题：

1. 在 1/2/4/8/16/32/36 线程下，哪些场景比 torch 2.10 的 `aten::index` 慢？
2. 慢在微架构的哪一层（指令、缓存、内存、线程调度）？
3. 哪些是结构性问题，哪些只是共享机器的噪声？

适用构建：T11（`gather_direct` 去掉逐元素 `at::Tensor` 调用、2B/1B/complex
改单遍标量、非连续下标 stride 修复、A==1 紧凑循环、OpenMP 线程数同步）。
原始矩阵见 `benchmarks/RESULTS.md` §15；复现脚本见文末。

---

## 0. 结论速览

| 场景 | 现象（修正后） | 根因 | 结论 |
|---|---|---|---|
| `x[:, i]`，C=64，B=200k | 1t 2.0x，8t 1.4x，16t 1.1x，36t 1.9x | 单元素 gather；T11 A==1 预校验紧凑循环后已优于 aten | 已修复（0.18x → 1.1-2.0x）；高线程 aten 追近 |
| half/bf16 | 1.2-2.4x | 旧路径 linearize 两遍 + 标量 | 已修复（单遍标量） |
| int8/uint8 | 1-16t 0.90-1.06x，32/36t 1.9-2.0x | 1B 元素，64B 行放大 64x；无 SVE 内核 | 1-16t 基本持平，高线程 aten 变慢 |
| complex64 | 0.95-1.00x（1-16t），32/36t 1.8-2.8x | 无 SVE 内核，标量单遍 | 1-16t 持平 |
| `x[i, j]`（双 int64 下标） | 1.05-1.25x | 4B 随机 gather 行放大；8/16t 0.96/0.89 为调度噪声 | 结构性持平，非退化 |
| `x[perm]` f64，N=200k | 8/16t 0.87/0.76x，其余 ≥1.0x | 小工作量强扩展：每线程 <100KB，任务开销 > 收益 | 小规模并行阈值问题 |
| 非连续 mask | 0.85-1.26x | T8 只支持连续 mask，回退 nonzero（串行段） | 已知限制，可扩展 T8 |
| 空 mask 4t | 0.74x | 微秒级提前返回，计时/调度主导 | 噪声，无意义 |
| 非连续 idx 1t | 旧数据 1.37x 但**结果错误** | raw 路径忽略 stride | 已修复；正确性优先于假加速 |

一句话：**T11 之后没有大范围的系统性退化**；剩下的是非连续 mask（0.84-0.98x）、
1B/complex 的天然带宽限制（1-16t 0.9-1.0x），以及"每线程工作量太小"时的小规模负扩展
（S4/multi/小 K 的 0.6-0.96x）。`x[:,i]` 已从 0.18x 修到 1.1-2.0x。

---

## 1. 术语表

| 术语 | 含义 | 为什么重要 |
|---|---|---|
| **cache line（缓存行）** | CPU 与内存之间搬数据的最小单位，本机 64B | 随机读 1 个 4B 元素也要搬 64B，有效带宽只有 1/16 |
| **行放大（line amplification）** | 一次随机访问实际搬运的字节数 / 有用字节数 | 4B 元素 16x，8B 元素 8x，1B 元素 64x；这是随机 gather 的硬上限 |
| **L1d / L2 / LLC** | 一级数据缓存 / 二级缓存 / 末级缓存 | L1 命中 ~4 周期，L2 ~12-20，DRAM ~200-400；决定 gather 的成本 |
| **DRAM 延迟 / 带宽** | 内存往返时间 / 每秒可搬字节 | 随机 gather 通常先被延迟卡住，线程多了才被带宽卡住 |
| **MLP（memory-level parallelism）** | 同时处于飞行中的内存请求数 | MLP 高才能用并发掩盖延迟；但本机 4B gather 的瓶颈是行放大，不是 MLP |
| **NUMA** | 非统一内存访问；每个节点有自己的内存 | 跨节点访问延迟/带宽更差；本文固定在一个节点（38 核） |
| **SVE / SVE2 / VL** | ARM 可伸缩向量扩展 / 向量长度 | 本机 VL=512-bit：16×f32 或 8×f64 个 lane |
| **gather** | 按向量下标收集元素 | `svld1_gather_u32index_f32` 等；硬件逐 lane 发请求 |
| **prefetch** | 提前把数据拉进缓存 | `svprfd_gather_*`；隐藏随机访问延迟 |
| **NT store** | non-temporal store，绕过缓存写内存 | 适合只写一次的大输出，避免污染缓存 |
| **linearize** | 把多维/广播下标先物化成线性偏移数组 | 多一遍读写；大 K 时值得，小 K 时是固定开销 |
| **fused（融合单遍）** | 不物化偏移，边解码边 gather | 省一遍内存，但解码逻辑在热循环里 |
| **rowcopy** | 按连续行整块拷贝（A≥4） | 顺序访存，接近 memcpy，是最快的形态 |
| **bucket（桶化）** | 小值域时建 offset 查找表，再标量查表 | 表小、源块常驻缓存时比 SVE gather 快 |
| **strong scaling（强扩展）** | 固定问题规模，增加线程 | 效率 = 加速比 / 线程数；小问题很快偏离线性 |
| **parallel efficiency** | 强扩展效率 | 判断"加线程是否值得"的直接指标 |
| **grain / chunk** | 每个并行任务处理的元素数 | 太小 → 调度开销；太大 → 负载不均 |
| **false sharing** | 不同线程写同一 cache line 的不同字节 | 缓存行来回失效，性能骤降 |
| **OpenMP ICV / team** | 运行时线程数等内部状态 / 一组工作线程 | 本项目的关键坑：两个 OpenMP 运行时各有自己的 ICV |
| **SMT / 超线程** | 一个物理核跑多个逻辑线程 | 本机 1 thread/core，无 SMT；38t 就是 38 个物理核 |
| **IPC** | 每周期退休指令数 | 低 IPC + 高 stall = 等内存；高 IPC = 计算密集 |
| **dTLB miss** | 数据地址翻译缓存缺失 | 大范围随机访问会打爆 TLB，额外几十周期 |
| **warmup / median / interleaved** | 预热 / 取中位数 / 交替测量 | 共享机器上对抗漂移的标准做法 |

---

## 2. 测试平台

- CPU：鲲鹏 920F（Kunpeng 920F），aarch64，SVE2，VL=512-bit。
- 拓扑：608 个逻辑 CPU，1 thread/core（无 SMT），32 个 NUMA 节点。
- 本文固定节点 5：CPU 190-227（38 核），本地内存 64GB。
- 缓存：L1d 32KB 8-way，L1i 32KB，L2 768KB 12-way，64B cache line；
  未暴露统一 LLC（按 NUMA/簇分布）。
- 频率：1.2-2.0GHz（随负载/功耗浮动）。
- 软件：BiSheng clang 19.1.7，torch 2.10.0+cpu，OpenMP 由 BiSheng libomp 提供。

> 为什么固定一个 NUMA 节点：跨节点访存会把延迟和带宽都搅乱；38 核足够观察
> 1→36t 的扩展，又不会引入 608 线程的调度噪声。

---

## 3. 测量方法（以及一个曾经让所有线程数据失真的坑）

### 3.1 两个 OpenMP 运行时

PyTorch 用 **libgomp**，扩展用 BiSheng **libomp**，两者各自维护线程数 ICV：

- `torch.set_num_threads(t)` → 只改 libgomp 的 ICV（影响 ATen 自己的算子）。
- 扩展里的 `#pragma omp parallel` 由 libomp 执行 → 默认 team = 进程可见 CPU 数
  （本机 affinity 到 38 核时就是 38）。

因此在 T11 之前，标着 "2t/4t/8t..." 的 HPC 行实际都用了 38 个 worker，而
aten 行确实只用 t 个 → **相对加速比被严重高估**。这也是旧 RESULTS §8.3/§10
"每个线程都 ≥1.06x" 的来源。

修正：

1. 基准脚本用 `ctypes` 找到 `/proc/self/maps` 里的 libomp，调用
   `omp_set_num_threads(t)`；
2. T11 起内核入口（`index_impl` / `index_out_impl`）调用
   `sync_omp_threads()`，把 libomp 的 team 同步为 `at::get_num_threads()`，
   使用户的 `torch.set_num_threads(t)` 对扩展生效；
3. `at::parallel_for` 在 `t==1` 时本身会串行短路，所以 1t 数据一直是对的。

### 3.2 主矩阵

`benchmarks/bench_threads.py`：

- 同一进程内交替执行 aten 与 hpc：`HPC_INDEX_NO_OVERRIDE=1` 时 `x[i]` 是真
  aten，hpc 用 `index_sve.index`（pybind）直调同一内核；小 K 的 pybind 开销由
  §3.3 的两进程真实 `x[idx]` 补测。
- 每格 25 次中位数，预热 5 次；固定 node 5；`HPC_THREADS=1,2,4,8,16,32,36`。
- 31 个场景覆盖：S1/P1/P2/P3 模型形态、S4、1D f32/f64（含 idx32/idx64）、
  2B/1B/complex、A=2/3/4/16 rowcopy、非连续 self/idx、mask 三种、`x[:,i]`、
  小值域 range2048。

### 3.3 小 K 真实路径（两进程）

K≤1024 时单次调用只有几微秒，同进程的 pybind 包装（Python 调用 + list 转换）
会污染比值。`bench_smallk_driver.py` 交替启动 aten/hpc 两个子进程，各自跑
真实 `x[idx]`，取中位数后再比。覆盖 K=0/1/64/512/1023/1024/4096/8192/65536。

### 3.4 perf

`perf stat/record`（`perf_event_paranoid=-1`）在 node 5 上跑同一 case；计数
包含 Python/torch 启动（约 10-12 CPU·s），因此只做同 case 的 hpc vs aten
相对比较，不看绝对 cycles。关注指令数、IPC、frontend stalls、LLC miss。
`perf stat -D` 的单内核延迟窗口在本机不稳定（部分事件 `<not counted>`），
未用于本文结论。

### 3.5 噪声与判读

共享箱上绝对时间会漂移（同格多轮可差 2-3x）。本文规则：

- 只信**同一轮交替**得到的比值，不跨轮比较绝对值；
- 单格 <1x 但相邻线程 >1.2x、且重复后消失 → 判为噪声；
- 所有结论至少用两种线程数或两次独立运行复核；
- `mask empty` 这类微秒级 case 的比值没有意义（计时器/调度主导）。

---

## 4. 全量线程矩阵（T11）

> 下表是 `bench_threads.py` 每格 25 次交替测量的中位数比值（aten 时间 /
> hpc 时间），越大越好；`col` 行已是 A==1 紧凑循环合入后的结果。

| 场景 | 1t | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|---|
| S1 features[inp_idx] C=64 E=1M | 4.46x | 4.62x | 3.95x | 3.55x | 2.97x | 6.69x | 8.37x |
| P1 h[src] i64 C=64 E=1M | 4.42x | 4.43x | 3.83x | 4.02x | 3.52x | 6.97x | 5.96x |
| P2 pos[src] A=3 E=2M | 2.05x | 2.12x | 2.06x | 2.90x | 1.78x | 6.64x | 8.73x |
| P3 x[mask] nnz=60042 C=16 | 2.23x | 1.82x | 1.53x | 1.34x | 1.87x | 3.76x | 4.20x |
| S4 x[perm] f64 N=200k | 1.26x | 1.36x | 1.09x | 0.87x | 0.76x | 1.35x | 1.01x |
| 1D f64 K=2000000 | 1.26x | 1.18x | 1.15x | 1.11x | 1.03x | 1.08x | 1.51x |
| 1D f64 K=10000000 | 1.17x | 1.06x | 1.03x | 1.06x | 1.03x | 2.01x | 1.95x |
| 1D f32 K=2000000 | 1.07x | 1.41x | 1.40x | 1.35x | 1.24x | 1.48x | 2.17x |
| 1D f32 K=10000000 | 1.06x | 1.11x | 1.11x | 1.14x | 1.16x | 2.28x | 3.92x |
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

### 4.1 <1.00x 的单元

| S4 x[perm] f64 N=200k | 8t 0.87x, 16t 0.76x |
| multi x[i,j] f32 100k | 8t 0.96x, 16t 0.89x |
| 1D int8 K=2M | 1t 0.98x, 2t 0.99x, 8t 0.98x, 16t 0.92x |
| 1D uint8 K=2M | 4t 0.98x, 8t 0.99x, 16t 0.90x |
| 1D complex64 K=2M | 1t 0.95x, 8t 0.97x, 16t 0.97x |
| 1D int64 K=2M | 4t 0.87x |
| mask empty N=1M | 4t 0.74x |
| mask noncontig nnz=600108 | 1t 0.98x, 2t 0.96x, 4t 0.93x, 8t 0.89x, 16t 0.84x |
| 1D f32 K=2M range2048 | 16t 0.87x |

---

## 5. 逐案分析

### 5.1 `x[:, i]`：从 0.18x 修到 1.1-2.0x

**形态**：`x[200000, 64]` f32，`i[64]`，输出 `[200000, 64]`。每一行从 256B
（4 条 cache line）里按 64 个随机下标取 64 个 float。

**为什么不是"随机 DRAM"**：源行连续、每行只有 4 条 line，取完 64 个元素后
这些 line 仍在 L1/L2。所以它是**指令/循环开销 + 顺序写 51MB**，不是 DRAM
延迟问题；SVE gather 在这里帮不上忙（数据本来就在缓存里）。

**修复前**：`gather_direct` 每个元素都调用 `at::Tensor::data_ptr/strides/
element_size/dtype`（引用计数 + 间接调用），1t 315ms（0.18x）。`perf` 显示
`gather_direct<float>` 的 lambda 占 29.7% 的样本。

**修复 1**：属性循环外提取 + 增量推进 `r/k` → 1t 144ms（0.40x）。

**修复 2（T11）**：`A==1 + 单索引 + 全长连续`（`K<1024`，走 `gather_direct`）时，
先把 K 个下标校验/归一化
一次到小缓冲，再跑无检查的紧凑循环 `dst[p] = src[row_off[r] + v[k]]`，等价
aten `index_select` 的内层。K=64，预处理可忽略。结果：

| 线程 | 1t | 2t | 4t | 8t | 16t | 32t | 36t |
|---|---|---|---|---|---|---|---|
| 相对 aten | 2.02x | 1.88x | 1.69x | 1.43x | 1.12x | 1.55x | 1.87x |
| hpc 绝对时间 | 28.1ms | 15.0ms | 8.5ms | 5.3ms | 3.6ms | 2.9ms | 3.2ms |

**并行扩展**：hpc 28.1→5.3ms（1→8t，5.3x，效率 67%）；aten 56.5→7.5ms
（1→8t，7.5x），16t 时 aten 4.1ms、hpc 3.6ms 已接近饱和，相对比值在 1.1-1.9x
之间波动。**结论**：已不再是退化项；继续优化只能争取高线程的 10-20%，优先级低。

### 5.2 2B/1B/complex：从 0.5-0.9x 到 1.2-2.4x

**旧路径**：这些 dtype 没有 SVE gather 内核，统一走
`linearize`（物化偏移）+ `dispatch_scalar`（逐元素标量）。对纯 gather 来说，
linearize 是纯额外开销：小 K 时占比尤其大。

**修复**：`fast_1d`（单索引、B==1、A==1、连续）且无 SVE 内核时，所有线程数
直接走 `gather_single_1d` 单遍标量（读 idx → 检查 → 归一 → load/store）。

**结果**：

| dtype | 修复前（代表值） | 修复后（1-36t） |
|---|---|---|
| float16 | 0.75-0.94x | 1.22-2.40x |
| bfloat16 | 0.52-0.97x | 1.26-2.05x |
| int8 | 0.57-0.68x | 0.92-2.03x |
| uint8 | 0.58-0.66x | 0.90-2.01x |
| complex64 | ~0.64x | 0.95-2.76x |

**为什么 int8 仍略低于 1**：1B 元素随机读，每条 64B line 只用 1B（64x 放大）；
aten 也是标量 + 同样的行放大。两边都到硬件上限，差距只剩边界检查/归一化的
几条指令。没有 1B SVE gather 能绕过行放大，继续优化收益很低。

### 5.3 `x[i, j]`：4B 随机 gather，基本持平

`x` 2D，两个全长 int64 下标，输出 100k 个 f32。路径是 linearize（int32 偏移）
+ 4B SVE gather。最终矩阵：1-4t 1.05-1.19x，8/16t 0.96/0.89x，32/36t
1.24/1.25x。中间线程的轻微下探来自小工作量（400KB 输出）的任务切分/调度，
不是行放大变化；36t 恢复说明不是持续趋势。

### 5.4 `x[perm]` f64 N=200k：小工作量强扩展

最终矩阵：1t 1.26x、2t 1.36x、4t 1.09x、8t 0.87x、16t 0.76x、32t 1.35x、
36t 1.01x。

- 总数据 1.6MB 源 + 1.6MB 输出；16t 时每线程只有 12.5k 元素（100KB）。
- 线程启动 + 任务切分 + 内存延迟的固定成本超过并行收益；aten 在 16t
  也接近饱和。
- 32t 恢复说明 8/16t 的下探主要是放置/噪声，不是持续趋势。

**规律**：纯 gather 的每线程工作量低于 ~256KB 时，强扩展开始不划算。
`adaptive_grain` 已按 `n/(t*8)` 切分，但对"每线程字节数"没有下限；后续可加
"最小每线程字节数"限制（例如 256KB），小 K 时少开线程。

### 5.5 非连续 mask：T8 的已知边界

T8 `gather_mask` 要求 mask 与 self 都连续；非连续 mask 回退到
`nonzero`（生成 int64 下标）+ gather：

- 多一次全量扫描 + 中间下标张量（本例 600k × 8B ≈ 4.8MB）；
- `nonzero` 的输出长度依赖数据，内部有串行段/前缀和，Amdahl 限制了高线程
  收益（16t 0.84x）。

连续 mask 的 T8 路径是 1.20-3.76x。**优化方向**：让 T8 直接按 mask 的 stride
读（两遍都不用物化连续 mask），即可覆盖 `mask[::2]` 这类常见形态。

### 5.6 其余 <1x 单元：噪声/微秒级

- **空 mask 4t 0.74x**：空 mask 直接返回，耗时几微秒；4t 的 aten 值本身在
  1.97-0.79 GB/s 间跳，比值无意义。
- **range2048 16t 0.87x**：1t 1.22x、32t 1.41x，16t 是并行任务粒度的抖动。
- **uint8/complex 1-16t 0.90-1.00x**：与 int8 同因，属于"持平"。

### 5.7 非连续下标：一个被"性能"掩盖的正确性 bug

修复线程数时顺带发现：`bucket_gather_pure_raw` 与 `gather_single_1d` 原先
直接平坦读 `idx[k]` / 用 `src[v]`，忽略了索引 stride 和
`indexed_strides[0]`。后果：

- `i[::2]`（K≥1024，1t）会读到错误的 idx；
- 非连续 1D self（如 `x[::2][i]`）会读错源地址；
- 旧矩阵里 "noncontig idx 1t 1.37x" 正是这个**错误路径**跑出来的假加速。

修复：raw 路径加 `stride0`，并在 `fast_1d` 上要求 idx 连续；非连续输入回退
linearize 通用路径。修复后 1.01-1.57x（正确结果），并补了两进程回归用例。

**教训**：性能表格里的"异常快"必须先用非连续/边界用例验证正确性。

---

## 6. 并行扩展与线程选择

### 6.1 强扩展效率

以 hpc 自身 1t 为基准（GB/s，`bench_threads.py` 原始列）：

| 场景 | 1t | 8t | 16t | 32t | 36t | 8t 效率 | 36t 效率 |
|---|---|---|---|---|---|---|---|
| 1D f64 K=10M | 0.54 | 3.46 | 6.43 | 9.11 | 9.23 | 80% | 47% |
| 1D f32 K=10M | 0.24 | 1.89 | 3.69 | 6.84 | 7.11 | 98% | 82% |
| S1（模型行拷贝） | — | — | — | — | — | 高 | 高 |
| `x[:,i]`（时间） | 28.1ms | 5.3ms | 3.6ms | 2.9ms | 3.2ms | 67% | 24% |

（效率 = 加速比 / 线程数。）

读法：

- 1D f64 在 8t 为 80%、36t 47%：每线程 10M/8=1.25M 个 8B 随机读，行放大 8x，
  内存带宽开始成为主约束。
- 1D f32 在 8t 98%、36t 82%：4B 的行放大 16x，但源更大、请求更多，
  延迟掩盖更充分。
- `x[:,i]` 单线程已经很快（2.0x），但 8t 后效率降到 67%→24%；aten 同样
  饱和，所以相对比值反而在高线程回到 1.9x。

### 6.2 为什么 32/36t 有时反而更快

在 32/36t，aten 的 GB/s 经常**下降**（例如 1D f64 K=10M：16t 6.25 →
32t 4.54 → 36t 4.75，hpc 同期 6.43 → 9.11 → 9.23）。原因：

- 38 核节点上 32/36 个线程已经接近内存带宽饱和；
- aten 的标量循环在饱和后继续增加线程，只会增加争抢；
- hpc 的 SVE gather + prefetch 在高线程下仍能维持更多在飞请求。

所以 32/36t 的"高加速比"部分是 aten 退化贡献的，不能简单理解为 hpc 的
绝对性能提升。文档中的绝对带宽列就是为了区分这一点。

### 6.3 小规模并行阈值

从 S4（8/16t 0.87/0.76x）、小 K 1D（K=8192/65536 高线程 0.46-0.99x）可以看到：
**每线程工作量太小时，多线程是负收益**。建议规则：

- 纯 gather：每线程 ≥ 256KB 有效数据再并行；
- 小 K（≤4096）且 `n*A < 16384`：串行（T9 已实现，阈值用 `n*A` 而非 `n`）；
- 2D 小 K 的并行判断必须用 `n*A`：只看 `n` 会让 A=16、K=4096 误判为串行，
  8-36t 掉到 0.55-0.66x（T11 已修，修复后 1.03-3.51x）。
- 但"过小"仍然不该并行：2D A=16 K=8192 只有 512KB 数据，16/32/36t 反而
  0.83/0.50/0.46x（1t 3.74x、4t 2.06x、8t 1.39x）。小 K 的 1D 也一样
  （K=8192/65536 在高线程 0.6-0.8x，绝对时间只有 40-90µs）。
  下一步可按"总字节数 / 每线程 64KB"限制有效线程数，而不是只切 grain。

---

## 7. 微架构证据（perf）

方法：`perf stat`（`perf_event_paranoid=-1`），node 5，单线程 case 跑数百次。
注意 `perf` 计数包含 Python/torch 启动（本机启动本身约 10-12 CPU·s，且是多线程），
因此**绝对 cycles/instructions 只用于同 case 的 hpc vs aten 相对比较**；事件比例
（IPC、frontend stalled、LLC miss rate）更可靠。

代表性结果：

| case | IPC（hpc / aten） | frontend stalled（hpc / aten） | LLC load miss（hpc / aten） | 判读 |
|---|---|---|---|---|
| `x[:,i]` 1t | 0.42 / 0.41 | 47% / 47% | 69% / 70% | 两边都是低 IPC + 高写带宽；hpc 快在**每元素指令更少**（2.0x），不在 miss 更少 |
| S1 1t | 0.40 / 0.44 | 48% / 39% | 68% / 74% | hpc IPC 更低但快 4.4x：SVE rowcopy 每轮搬 256B，**bytes/cycle** 才是关键 |
| bf16 1t | 0.43 / 0.53 | 36% / 38% | 66% / 74% | hpc 的标量单遍指令更多（cycles +28%），但省掉 linearize 后仍 1.2-2.4x |
| 非连续 mask 8t | 0.47 / — | 36% / — | 67% / — | `nonzero` 串行段 + 随机 gather，IPC 低、LLC miss 高 |
| S4 f64 16t | 0.44 / — | 36% / — | 72% / — | 小工作量 + 随机行，后端停顿为主 |
| `x[i,j]` 8t | 0.45 / — | 39% / — | 75% / — | 4B 随机 gather 的 64B 行放大 |

统一结论：所有 case 的 IPC 都 ≤0.5，说明瓶颈不在算术；`frontend stalled` 高是
分支/标量循环的表现，LLC miss 高是随机 gather 的表现。**性能差异主要来自
"每字节有用数据搬了多少条 cache line"和"每元素多少条指令"**，与 §2 的模型一致。

修复前 `x[:,i]` 的 `perf record` 里，`gather_direct<float>` lambda 占 29.7% 的
样本；修复后该符号退出热点（时间减半以上），这是"去掉逐元素 `at::Tensor`
调用"的直接证据。

## 8. 优化优先级

1. **小规模并行阈值**（S4/multi/小 K 的 0.46-0.96x）：按"每线程字节数"限制
   线程（例如每线程 ≥64KB），而不是只按元素数切 grain。改动小，能消除大部分
   负扩展；这是当前最值得做的项。
2. **非连续 mask 的 T8 扩展**（0.84-0.98x → 预期 1.2x+）：两遍都按 stride 读，
   去掉 nonzero 中间量。工作量中等，收益明确。
3. **`x[:,i]`**：已修到 1.1-2.0x，剩余高线程 10-20% 收益，优先级低。
4. **1B/complex 不再投入**：行放大是硬限制，aten 同样是标量。

---

## 9. 复现

```bash
source ~/.bashrc
conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=<torch_lib>:<bisheng_lib>:$LD_LIBRARY_PATH

# 主矩阵（31 场景 × 7 线程，约 10 分钟）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36 HPC_ITERS=25 \
  python benchmarks/bench_threads.py

# 小 K 两进程
HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36 \
  HPC_REPS=2 HPC_ITERS=150 python benchmarks/bench_smallk_driver.py

# 单个 case 的 perf（示例：col 1t；aten 侧要加 HPC_INDEX_NO_OVERRIDE=1）
perf stat -e cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses \
  env MODE=hpc python benchmarks/perf_one.py col 1 5
perf stat -e cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses \
  env MODE=aten HPC_INDEX_NO_OVERRIDE=1 python benchmarks/perf_one.py col 1 5
```

---

## 10. 与其它文档的关系

- 归因/回退决策：`fallback-analysis.md`
- 内核实现与门槛：`kernels.md`
- 原始矩阵与历史数据：`benchmarks/RESULTS.md` §15（§8.3/§10 已标注作废）
- 集成/覆盖机制：`integration.md`、`override-mechanism.md`
- 语义与测试：`semantics.md`
