# benchmarks/ 目录指南

本目录只放**性能基准、性能回归扫描和性能诊断探针**；功能/正确性测试全部在
`tests/`，不要把这里的脚本当功能测试用。`bench_regression.py` 也只标记
**性能**回归（比值 <0.90），不验证结果正确性。

> **这篇讲什么**：每个 benchmark 脚本测什么、该用哪个、环境变量怎么设、结果怎么
> 和功能测试区分。
> **读完你能回答**：验收该跑哪个脚本？小 K 为什么要两进程？`HPC_AFFINITY`
> 怎么用？哪些脚本只作历史参考？
> **本篇代号**：M1-M6/shortq/A1（真实负载）、S1/P1（基准场景）、R10（当前轮次）。
> 术语见 [glossary.md](glossary.md)。

## 1. 一句话总览

下表按“正式基准 / 组件脚本 / 历史参考”三类读；不确定时先用 `bench_threads.py`。

| 文件 | 类型 | 作用 | 何时用 |
|---|---|---|---|
| `bench_index.py` | 历史基准 | 最早的 1/608 线程全矩阵，无 affinity，覆盖核心形态 | 只作历史参考，不要再用作验收 |
| `bench_pinned.py` | 历史定点 | 固定 NUMA 节点后的核心场景 A/B，按环境变量扫线程 | 快速复查旧结论，正式数据以 `bench_threads.py` 为准 |
| `bench_gate.py` | 历史门禁 | 更小的定点集合（默认 1/38t），固定 affinity，顺序测 aten/hpc | 旧门禁/快速抽查；新验收用 `bench_threads.py` |
| `bench_threads.py` | **正式基准** | 当前权威 31 场景 × 8 线程矩阵，aten/hpc 交错取中位数，修正 libomp/libgomp 双 ICV | 性能验收、复测、提交前跑 |
| `bench_smallk_real.py` | 组件脚本 | 单进程小 K 真实 `x[idx]` 计时，输出 JSON；由 driver 交替调用 | 不单独用，配合 `bench_smallk_driver.py` |
| `bench_smallk_driver.py` | **正式基准** | 交替启动 aten/hpc 两个子进程，汇总小 K 中位数，排除 pybind 包装噪声 | 小 K（K<=65536）真实路径验收 |
| `bench_realpath.py` | 组件脚本 | 单进程真实 workload（M5/shortq/A1/M1/M2）`x[idx]` 计时，输出 JSON；由 driver 交替调用 | 不单独用，配合 `bench_realpath_driver.py` |
| `bench_realpath_driver.py` | **正式基准** | 交替启动 aten/hpc 两个子进程，真实路径全 case 对拍（含短调用、高频单元素） | 短调用 / 真实 workload 收益结论的首选依据 |
| `bench_regression.py` | 回归扫描 | 多形态性能回归扫描，默认 1/8/32t，比值 <0.90 打 `REGRESS` | 改内核后快速看有没有明显性能回退 |
| `bench_ai4s.py` | 专项基准 | AI4S workload 形态（M1-M6）线程扫描，不是功能测试 | 评估 AI4S 场景收益/回归 |
| `bench_cl_family.py` | **专项基准** | cl/置换布局行索引族前后对照：`aten`/`hpc` 两进程交替（覆盖是进程级开关），cl4d 行/掩码、cl3d 置换、contig4d 箱态对照组 | R10 直写（output-layout.md §8）等布局相关优化的验收/复现（RESULTS §2.9.2） |
| `bench_embed_threads.py` | 专项基准 | 宿主多线程并发调用 `x[idx]` 的超订实验：固定总工作量，对比单线程顺序与 k 线程并发，NT/K/CALLS 环境变量扫描 | 嵌入多线程/多进程宿主程序的部署评估（integration.md §6.1） |
| `bench_t1.py` | 优化轮专项 | M2 短行 A-sweep（A=2..64），看行宽对 rowcopy 的影响 | 调整 rowcopy 阈值/短行路径时用 |
| `bench_t2.py` | 优化轮专项 | M1 权重块（A=1024/2048/4096/65536，含 ≤16KB/行真实尺寸）线程扫描 | 调整大块 rowcopy / raw 路径时用 |
| `degrade_probe.py` | 诊断探针 | 绝对带宽、小工作量强扩展、mask 分解等退化定位数据 | 分析“为什么慢”、写 degradation-analysis 时用 |
| `perf_one.py` | perf 驱动 | 单 case 循环执行，供 `perf stat/record` 采样 | 需要微架构计数（cache/TLB/指令数）时用 |
| `memprobe.c` | 硬件探针 | 纯 C 的延迟/带宽/随机 gather 微基准，与 torch 无关 | 测内存系统上限、排除算子本身因素 |
| `RESULTS.md` | 结果报告 | 正式性能结果与历史轮次数据，不是脚本 | 查数据、写结论、对齐口径 |

## 2. 该用哪个？

- **正式性能验收 / 复测**
  - 大 K、多线程矩阵：`bench_threads.py`
  - 小 K 真实路径：`bench_smallk_driver.py`
  - 真实 workload / 短调用 / 高频单元素：`bench_realpath_driver.py`
- **改完内核后的快速回归**
  - `bench_regression.py`
  - 若涉及 AI4S 场景，再跑 `bench_ai4s.py`
- **调阈值 / 专项优化**
  - 短行：`bench_t1.py`
  - 大块：`bench_t2.py`
- **分析退化原因**
  - 先看 `RESULTS.md` 和 `docs/degradation-analysis.md`
  - 需要补充数据时用 `degrade_probe.py`
  - 需要微架构计数时用 `perf_one.py` + `perf stat/record`
  - 需要硬件上限时用 `memprobe.c`
- **只看历史数据**
  - `bench_index.py` / `bench_pinned.py` / `bench_gate.py`
  - 这些脚本的 2-36t 数据可能受旧的双 OpenMP ICV 问题影响，不要直接当当前结论

## 3. 真实 workload 覆盖与预测力

**路径对称性问题**：除 `bench_smallk_real.py` / `bench_realpath.py` 外，其余脚本的
本算子侧都是 `hpc_index()` pybind 直呼，绕过了 dispatcher。大 K 下固定开销占比
可忽略，比值可信；**短调用（单次几微秒级）的比值会偏乐观**，必须以两进程
真实 `x[idx]` 对拍（`bench_realpath_driver.py` / `bench_smallk_driver.py`）为准。

各基准对“单算子 case 优化 ⇒ 真实应用收益”的预测力：

| 真实形态 | 典型来源 | 覆盖脚本 | 预测力 |
|---|---|---|---|
| M5 标量循环（K=1 重复上万次） | FPS / ball_query 逐点取元素 | `bench_realpath.py`（M5 scalar） | 高 |
| shortq 短行短调用（A=2/3/4，K=64-1023） | 图消息 / 邻域查询 | `bench_realpath.py`（shortq） | 高 |
| A1 单元素高频 gather（i64/i32） | OACNNs（~800 次/推理） | `bench_realpath.py`（A1） | 高 |
| M1 权重块（≤16KB/行，E~20k） | Sonata-v1m2 weight gather | `bench_realpath.py`（M1 real）、`bench_t2.py` | 高 |
| M2 spconv 行 gather（C=8/64/256，E~100k） | spconv features[regions] | `bench_realpath.py`（M2）、`bench_threads.py` S1 | 高 |
| 大 K 行拷贝 / pure gather（K≥200k） | 大批量重排 | `bench_threads.py` | 中（形态对，K 偏大） |
| mask / 多索引 / 列索引 | 通用数据管道 | `bench_threads.py` | 中 |
| 2B/1B/complex、K=10M | 少见边界 | `bench_threads.py` | 低（形态对但尺寸/dtype 罕见，对真实收益预测力弱） |
| 608 线程全节点 | — | 不覆盖 | 低（共享箱噪声，历史数据已作废） |

判断某个优化是否对真实应用有收益时：先看 `bench_realpath_driver.py` 对应
形态的比值，再参考 `bench_threads.py` 的全线程矩阵确认没有其它形态回退。

## 4. 公共环境变量

| 变量 | 含义 | 常用值 |
|---|---|---|
| `HPC_INDEX_NO_OVERRIDE` | `1` 时不覆盖 `aten::index.Tensor`，用于同进程测真 aten | `0` / `1` |
| `HPC_AFFINITY` | 绑定的 CPU 集合，建议固定到空载 NUMA 节点 | `190-227`（node 5） |
| `HPC_THREADS` | 逗号分隔的线程数列表 | `1,2,4,8,16,32,36,38`（38 = 单 NUMA 节点实际核数） |
| `HPC_ITERS` | 每个 case 的迭代次数 | `25` / `40` / `100` / `200` |
| `HPC_REPS` | `bench_smallk_driver.py` / `bench_realpath_driver.py` 的重复轮数 | `3` |
| `HPC_LOOP_ITERS` | `bench_realpath.py` 标量循环 case 的计时轮数（每轮内含 `HPC_CALLS` 次调用） | `5` |
| `HPC_BIG_ITERS` | `bench_realpath.py` 大输出 case（M1/M2）的计时轮数 | `20` |
| `HPC_CALLS` | `bench_realpath.py` 高频调用 case 的每轮调用次数 | `10000` |
| `HPC_INDEX_NT` | 非临时存储（non-temporal）开关 | `0` / `1` |
| `MODE` | `perf_one.py` 使用 `hpc` 还是 `aten` | `hpc` / `aten` |

注意：
- `bench_threads.py` 已同时同步 `torch.set_num_threads` 和 libomp 的
  `omp_set_num_threads`，线程标签可信。
- 旧脚本（`bench_index.py` 等）没有这层修正，2-36t 数据只能作历史参考。
- `bench_smallk_driver.py` / `bench_realpath_driver.py` 会自己在子进程里设置
  `HPC_INDEX_NO_OVERRIDE`，不需要手动设。

## 5. 典型命令

```bash
# 1. 当前权威全矩阵（31 场景 × 8 线程）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36,38 HPC_ITERS=40 \
  python benchmarks/bench_threads.py

# 2. 小 K 两进程真实路径
HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
  HPC_REPS=3 \
  python benchmarks/bench_smallk_driver.py

# 3. 真实 workload 两进程对拍（M5/shortq/A1/M1/M2，含短调用与高频单元素）
HPC_AFFINITY=190-227 HPC_THREADS=1,8,16,38 \
  HPC_REPS=3 \
  python benchmarks/bench_realpath_driver.py

# 4. 性能回归扫描
HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py

# 5. AI4S workload 专项
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_ai4s.py

# 6. 短行 A-sweep
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_t1.py

# 7. 大块 rowcopy 专项
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36,38 \
  python benchmarks/bench_t2.py

# 8. 退化定位探针
HPC_AFFINITY=190-227 python benchmarks/degrade_probe.py

# 9. 单 case perf 驱动
MODE=hpc HPC_AFFINITY=190-227 \
  python benchmarks/perf_one.py s1 1 5

MODE=aten HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  python benchmarks/perf_one.py s1 1 5

# 10. 内存系统探针
gcc -O3 -march=native -fopenmp benchmarks/memprobe.c -o /tmp/memprobe -lm
numactl --membind=5 --physcpubind=190 /tmp/memprobe latency 8192 3000000
numactl --membind=5 --physcpubind=190-227 /tmp/memprobe bandwidth 256 8 rd
numactl --membind=5 --physcpubind=190-227 /tmp/memprobe gather 256 4000000 rand
```

## 6. 与功能测试的关系

- **本目录全部是性能相关**：
  - `bench_*` 是基准/回归
  - `degrade_probe.py` / `perf_one.py` / `memprobe.c` 是诊断探针
  - `RESULTS.md` 是结果文档
- **功能/正确性测试在 `tests/`**，例如：
  - `tests/test_smoke.py`
  - `tests/test_correctness.py`
  - `tests/test_errors.py`
  - `tests/test_model.py`
  - `tests/test_smallk.py`
- 如果要验证“结果对不对”，去 `tests/`；
  如果要验证“快不快、有没有回退”，用本目录。

## 7. 相关文档

- 正式性能结果：`benchmarks/RESULTS.md`
- 方法论与结论摘要：`docs/performance.md`
- 退化与线程扩展分析：`docs/degradation-analysis.md`
- 优化轮记录：`docs/optimizations.md`
- 当前状态：`docs/status.md`
