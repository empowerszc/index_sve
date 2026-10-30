# 术语与代号速查（Glossary）

全项目唯一的名词查询入口：读任何文档遇到不懂的术语或代号（S1、P1、M1、
shortq、A/B/K…），先查这里；各文档不再各自维护术语表。

> 简写约定：**degradation / performance / RESULTS** 分别指
> `docs/degradation-analysis.md`、`docs/performance.md`、`benchmarks/RESULTS.md`。

## 1. 技术术语表

| 术语 | 含义 | 为什么重要 |
|---|---|---|
| **cache line（缓存行）** | CPU 与内存之间搬数据的最小单位，本机 64B | 随机读 1 个 4B 元素也要搬 64B，有效带宽只有 1/16 |
| **行放大（line amplification）** | 一次随机访问实际搬运的字节数 / 有用字节数 | 4B 元素 16x，8B 元素 8x，1B 元素 64x；这是随机 gather 的硬上限 |
| **L1d / L2 / LLC** | 一级数据缓存 / 二级缓存 / 末级缓存 | L1 命中 ~4 周期，L2 ~12-20，DRAM ~200-400；决定 gather 的成本 |
| **DRAM 延迟 / 带宽** | 内存往返时间 / 每秒可搬字节 | 随机 gather 通常先被延迟卡住，线程多了才被带宽卡住 |
| **MLP（memory-level parallelism）** | 同时处于飞行中的内存请求数 | MLP 高才能用并发掩盖延迟；但本机 4B gather 的瓶颈是行放大，不是 MLP |
| **RowCopy / rowcopy（行拷贝）** | 一次访问搬一整段连续元素（A>=4），接近 memcpy | 一个 cache line 里的有用字节多，随机性被“整行”摊薄，是最快的形态 |
| **PureGather（纯 gather）** | 一次只取一个元素（A==1），地址随机 | 每个元素都可能拉一条 64B line，受行放大限制 |
| **A / B / K** | A=每次访问的连续元素数；B=索引广播的前导行数；K=索引张量的元素数 | 三者决定走 RowCopy（A 大）、纯 gather（A=1）还是小 K 快路径（K<1024） |
| **linearize（线性化）** | 把多维/广播下标换算成一维偏移并物化成数组（一遍写 + 一遍读），含越界/负数检查 | T2/T5/T9/T11 的优化大多是“跳过或合并这一遍”；大 K 时值得，小 K 时是固定开销 |
| **fused（融合单遍）** | 不物化偏移，边解码边 gather | 省一遍内存，但解码逻辑在热循环里 |
| **bucket（桶化）** | 小值域时建 offset 查找表，再标量查表 | 表小、源块常驻缓存时比 SVE gather 快 |
| **ICV** | OpenMP 的线程数配置副本；libomp/libgomp 各一份 | 旧线程数据失真的根源，见 degradation §3.1 |
| **加速比 / 强扩展效率** | 加速比 = 1t 时间 / Nt 时间；效率 = 加速比 ÷ 线程数（strong scaling：固定规模加线程） | 效率 <100% 说明并行有开销或带宽饱和；判断“加线程是否值得”的直接指标 |
| **NUMA** | 非统一内存访问；每个节点有自己的内存 | 跨节点访问延迟/带宽更差；测试固定在一个节点（38 核） |
| **SVE / SVE2 / VL** | ARM 可伸缩向量扩展 / 向量长度 | 本机 VL=512-bit：16×f32 或 8×f64 个 lane |
| **gather** | 按向量下标收集元素 | `svld1_gather_u32index_f32` 等；硬件逐 lane 发请求 |
| **prefetch** | 提前把数据拉进缓存 | `svprfd_gather_*`；隐藏随机访问延迟 |
| **NT store** | non-temporal store，绕过缓存写内存 | 适合只写一次的大输出，避免污染缓存 |
| **grain / chunk** | 每个并行任务处理的元素数 | 太小 → 调度开销；太大 → 负载不均 |
| **false sharing** | 不同线程写同一 cache line 的不同字节 | 缓存行来回失效，性能骤降 |
| **OpenMP ICV / team** | 运行时线程数等内部状态 / 一组工作线程 | 本项目的关键坑：两个 OpenMP 运行时各有自己的 ICV |
| **SMT / 超线程** | 一个物理核跑多个逻辑线程 | 本机 1 thread/core，无 SMT；38t 就是 38 个物理核 |
| **IPC** | 每周期退休指令数 | 低 IPC + 高 stall = 等内存；高 IPC = 计算密集 |
| **dTLB miss** | 数据地址翻译缓存缺失 | 大范围随机访问会打爆 TLB，额外几十周期 |
| **warmup / median / interleaved** | 预热 / 取中位数 / 交替测量 | 共享机器上对抗漂移的标准做法 |
| **fork-join** | OpenMP 并行区进入/退出：唤醒 worker + 静态分块 + barrier 同步 | 每遍 ≈5-15µs 且与工作量无关；小 K 高线程家族的共性根因（degradation §5.9.2） |
| **微码 / µop** | 一条复杂指令（如 SVE gather）在核内展开成的微操作序列 | gather 逐 lane 展开的微码/发射吞吐是它在小源上的下限（degradation §5.9.1） |
| **乱序执行（OoO）/ 在飞负载 / MSHR** | 提前执行后续指令的能力 / 已发出未返回的访存请求数 / 等待缓存填充的硬件队列 | 决定标量循环能摊掉多少内存延迟（degradation §5.9.1） |
| **DVFS** | 动态电压频率调节（主频随负载浮动） | 共享箱绝对时间漂移的来源之一（degradation §5.9.5） |
| **源（source）/ 源驻留级** | 被索引的张量 x / 它常驻的缓存层级（L1/L2/L3/DRAM） | T15 起 4B SVE 内核按源大小（>512KB）门控（degradation §5.9.1） |
| **每元素成本** | 墙钟时间 ÷ **每线程**元素数 | 并行路径除以总元素数会低估 8-38 倍（degradation §5.9.5 的历史教训） |
| **smallk / realpath** | 小 K 两进程基准 / 真实 workload 两进程基准（`benchmarks.md` §1） | 短调用结论的首选依据（RESULTS §2.3 小 K；realpath 当前值 §2.9.4，§2.5 为 T14 历史轮） |
| **对拍** | 覆盖态与非覆盖态两进程比对 true-aten 参考值 | 正确性验收口径，44/44（`tests/verify_against_aten.py`） |
| **T0-T15** | 优化/修复轮次代号（每轮一条记录） | 文中"T15 起 / T5 已修"指这些轮次，见 `optimizations.md` |

## 2. 场景与负载代号速查

文档与基准里的代号都来自真实 AI4S 模型形态（performance §3、RESULTS
§2/§3.8；定义在 `bench_threads.py` / `bench_ai4s.py` / `bench_realpath.py`）。

**模型场景（31 场景矩阵，RESULTS §2.1）**：

| 代号 | 白话含义 | 张量形状 |
|---|---|---|
| **S1** | 稀疏卷积按行取特征：`features[inp_idx]` | features (N, C=64) f32，inp_idx i32/i64，E=1M |
| **P1** | 图/点云消息传递：`h[src]` | h (N, C=64)，src i64，E=1M |
| **P2** | 点云位置取数（短行）：`pos[src]` | pos (N, A=3)，E=2M |
| **P3** | bool 掩码按行筛选：`x[mask]` | x (N, C=16)，nnz≈60k |
| **S4** | 随机置换取数（1D 纯 gather）：`x[perm]` | x (N=200k,) f64，perm 为随机排列 |

**AI4S workload（`bench_ai4s.py`，RESULTS §3.8）**：

| 代号 | 白话含义 | 张量形状 |
|---|---|---|
| **M1** | 权重块取数（最热模式，14 模型累计 27270ms）：`w[ki]` | w 三维权重视图，行 ≤16KB，E=2k-20k |
| **M2** | spconv 特征按行取数：`feat[r]` | feat (20000, 8/64/256) |
| **M3** | 小值域查表：`keys[order]` | keys (≤2048,)，order E=54k |
| **M4** | 高重复索引（同一行被反复取）：`feat[inverse]` | feat (200, 64)，idx 20000 个 |
| **M5** | 标量循环（FPS/ball_query 风格）：每次调用取 1 个元素、高频循环 | x (4096,) 或 (4096, 3)，idx (CALLS, 1) |
| **M6** | 双下标 2D 取数：`pf[vi]` | x (20000, 64)，idx (2048, 3) |

**realpath 短调用（`bench_realpath_driver.py`，当前值 RESULTS §2.9.4；§2.5 为 T14 历史轮）**：M1（真实权重块，
行 ≤16KB）、M2（代表形态）、M5（标量循环）、**A1**（单元素高频，N=200k、
K=1）、**shortq**（短行查询，A=2-4、K=64-1023，x (20000, A)）。

**矩阵场景行名读法（RESULTS §2.1=T11、§2.9.1=R10，行名相同）**：`1D f32 K=2M` = 1D f32 源按 2M 个
随机下标取元素；`rowcopy A=16 f32 K=200k` = 2D f32 源（行宽 16）按 200k
个随机行下标取行；`noncontig self` = 源是非连续视图（stride 2）、
`noncontig idx` = 下标张量非连续；`col x[:,i]` = 整列选取；`range2048` =
下标值域 ≤2048（命中桶化）；`mask empty` = 全 False 掩码（零输出边界）；
`multi x[i,j]` = 两个全长下标（广播后 2D）。

**形状记号**：A（或 C）= 每行元素数（行宽）；K（或 E）= 下标元素数
（取多少行）；B = 被索引维之前的前导维度积；nnz = mask 非零数；值域 =
下标取值的范围宽度。B·K·A 描述一次调用的输出形状。

## 3. 代号冲突警示（读文档前必看）

- **"M1" 有两套含义**：`fallback-analysis.md` §1 的 **M1-M4 是机制代号**
  （M1=行放大+DRAM 延迟、M2=两遍开销、M3=串行段、M4=无 SVE 内核）；而
  上表的 **M1-M6 是负载代号**（权重块、spconv…）。两套互不相干，按上下文
  区分："受 M1 限制"多指机制，"M1 1t 8-10x"指负载。
- **"P" 有两套含义**：场景代号 P1/P2/P3（`h[src]`、`pos[src]`、`x[mask]`）；
  路线图优先级已改用 **R1-R9**（`status.md` §7）避免混淆。
- **T0-T15** 是优化轮次（`optimizations.md`），与张量/场景代号无关。
