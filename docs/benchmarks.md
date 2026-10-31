# benchmarks/ 目录指南

本目录只放**性能基准、性能回归扫描和性能诊断探针**；功能/正确性测试全部在
`tests/`，不要把这里的脚本当功能测试用。`bench_regression.py` 也只标记
**性能**回归（比值 <0.90），不验证结果正确性。

## 1. 一句话总览

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
| `bench_production_shapes.py` | 组件脚本 | 生产形态库（K 512-55296、A=1/3/64、线程 1/6/12/19、`x[idx]` 走 dispatch）单进程计时，输出 JSON；含 `--emit-cases`/`--gen-cases`/`--per-model`/`--weighted`/`--detail`/`--md`/`--abr-map`/`--total`/`--cases` | 不单独用，配合 `bench_production_driver.py`；加权/覆盖率结论见 RESULTS §2.12 |
| `trace_hotspots.example.csv` | 输入 fixture | **不是真实 trace**：按生产反馈 §11.2.5 的聚合热点表手工重建（model/path/K/A/dom/calls/threads），用来演示 `--gen-cases` 的输入格式 | 有真 trace 时换成真文件；格式说明见文件头注释 |
| `bench_production_driver.py` | **正式基准** | 交替启动 aten/hpc 两进程，汇总生产形态矩阵；`--fit` 把"固定开销 / 每元素成本"分开 | **生产收益结论的唯一依据**，任何"单算子几倍"的结论都要附这一族（RESULTS §2.10） |
| `bench_context_overhead.py` | 诊断 | 同一形态在 hot / 线程池冷（spaced）/ 源冷 / autograd 四种上下文下的单次耗时；另含固定链（K=1）、A 敏感性、K=2 小形态、mask 的 `nonzero` 与混合序列 | 回答"模型内每次调用比 bench 贵多少"，RESULTS §2.11 |
| `bench_entry_overhead.py` | 诊断 | `x[idx]` 全链路拆解：dispatcher / `torch.ops` 查表 / Python 包装 / `torch.empty` / normalize 的各路下标表示（i32/stride2/2D/双索引） | 判断"某个基准脚本的入口口径差多少"，RESULTS §2.12.1 |
| `bench_regression.py` | 回归扫描 | 多形态性能回归扫描，默认 1/8/32t，比值 <0.90 打 `REGRESS` | 改内核后快速看有没有明显性能回退 |
| `bench_ai4s.py` | 专项基准 | AI4S workload 形态（M1-M6）线程扫描，不是功能测试 | 评估 AI4S 场景收益/回归 |
| `bench_cl_family.py` | **专项基准** | cl/置换布局行索引族前后对照：`aten`/`hpc` 两进程交替（覆盖是进程级开关），cl4d 行/掩码、cl3d 置换、contig4d 箱态对照组 | R10 直写（output-layout.md §8）等布局相关优化的验收/复现（RESULTS §2.9.2） |
| `bench_embed_threads.py` | 专项基准 | 宿主多线程并发调用 `x[idx]` 的超订实验：固定总工作量，对比单线程顺序与 k 线程并发，NT/K/CALLS 环境变量扫描 | 嵌入多线程/多进程宿主程序的部署评估（integration.md §6.1） |
| `bench_t1.py` | 优化轮专项 | M2 短行 A-sweep（A=2..64），看行宽对 rowcopy 的影响 | 调整 rowcopy 阈值/短行路径时用 |
| `bench_t2.py` | 优化轮专项 | M1 权重块（A=1024/2048/4096/65536，含 ≤16KB/行真实尺寸）线程扫描 | 调整大块 rowcopy / raw 路径时用 |
| `degrade_probe.py` | 诊断探针 | 绝对带宽、小工作量强扩展、mask 分解等退化定位数据 | 分析“为什么慢”、写 degradation-analysis 时用 |
| `perf_one.py` | perf 驱动 | 单 case 循环执行，供 `perf stat/record` 采样 | 需要微架构计数（cache/TLB/指令数）时用 |
| `memprobe.c` | 硬件探针 | 纯 C 的延迟/带宽/随机 gather 微基准，与 torch 无关；带宽要带 `REPS`（单遍会把冷建队算进去） | 测内存系统上限、排除算子本身因素 |
| `bench_hw_traffic.py` | 内核线流量 | 真实 `aten::index` 1D gather 的绝对带宽，换算成 64B 行流量，用来和 `memprobe` 的上限对照 | 判断"是否撞到带宽墙" |
| `RESULTS.md` | 结果报告 | 正式性能结果与历史轮次数据，不是脚本 | 查数据、写结论、对齐口径 |

## 2. 该用哪个？

- **正式性能验收 / 复测**
  - 大 K、多线程矩阵：`bench_threads.py`
  - 小 K 真实路径：`bench_smallk_driver.py`
  - 真实 workload / 短调用 / 高频单元素：`bench_realpath_driver.py`
  - **生产形态（K 512-55296、4-19 线程）：`bench_production_driver.py`**
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

**路径对称性问题**：除 `bench_smallk_real.py` / `bench_realpath.py` /
`bench_production_shapes.py` / `bench_entry_overhead.py` 外，其余脚本的 hpc 侧都是
`hpc_index()` 直呼，绕过了 dispatcher。实测这条路的偏置方向是**偏悲观**，不是
偏乐观：Python 包装 + `torch.ops` 查表比 `x[idx]` **慢 10µs**（K=0/512/2048 都是，
`torch.ops.aten.index` 一样慢，见 RESULTS §2.12.1）。大 K 下 10µs 可忽略，比值
可信；**短调用（单次几微秒级）必须用两进程真实 `x[idx]` 对拍**
（`bench_realpath_driver.py` / `bench_smallk_driver.py` /
`bench_production_driver.py`），否则会把本算子算慢。

**K / 线程量级同样会决定结论**：`bench_index.py` 的 K=2M/10M（工作集 8-80MB）
把双方都推到行流量受限区，`bench_threads.py` 的 1/608t 是两个端点。生产的
K 512-55296（工作集 8KB-4MB）+ 4-19t 落在"每元素成本 / 固定开销"主导区，
1D A=1 在那里是 0.66-1.10x（RESULTS §2.10、degradation §6.5）。

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
| `HPC_REPS` | `bench_smallk_driver.py` / `bench_realpath_driver.py` / `bench_production_driver.py` 的重复轮数 | `3` |
| `HPC_DOM_SWEEP` | `bench_production_shapes.py` 诊断模式：固定该 K 只扫值域（桶化建表成本） | `8192` / 空 |
| `HPC_CTX_KS` | `bench_context_overhead.py` 上下文组用的 K | `512,2048,8192` |
| `HPC_SPACED_S` | `bench_context_overhead.py` `spaced` 上下文的调用间隔（秒） | `0.0002` |
| `HPC_SUBSET` | `bench_production_shapes.py` 只保留名字含任一子串的用例（逗号分隔） | 空（全量） |
| `HPC_CASES` | 用 `--gen-cases` / `--emit-cases` 的 JSON 直接当用例库（等价于 `--cases`） | 空 |
| `HPC_LOOP_ITERS` | `bench_realpath.py` 标量循环 case 的计时轮数（每轮内含 `HPC_CALLS` 次调用） | `5` |
| `HPC_BIG_ITERS` | `bench_realpath.py` 大输出 case（M1/M2）的计时轮数 | `20` |
| `HPC_CALLS` | `bench_realpath.py` 高频调用 case 的每轮调用次数 | `10000` |
| `HPC_INDEX_NT` | 非临时存储（non-temporal）开关 | `0` / `1` |
| `MODE` | `perf_one.py` 使用 `hpc` 还是 `aten` | `hpc` / `aten` |

注意：
- `bench_threads.py` 已同时同步 `torch.set_num_threads` 和 libomp 的
  `omp_set_num_threads`，线程标签可信。
- 旧脚本（`bench_index.py` 等）没有这层修正，2-36t 数据只能作历史参考。
- `bench_smallk_driver.py` / `bench_realpath_driver.py` / `bench_production_driver.py`
  会自己在子进程里设置 `HPC_INDEX_NO_OVERRIDE`，不需要手动设。
- `bench_context_overhead.py` 的 K=8192 档会受脚本内 768MB 源分配的堆扰动（aten
  多线程热循环出现 550µs 级假峰，隔离复测 32µs）；看 K≤2048 的档，或先删 `cold`
  组再跑。`forms` 组的 `nonzero` 19t 单格也可能被抖动污染，关键结论请隔离复测。
- `--weighted` 同时打印两套口径，别读反：**`aten/hpc` >1 = 本算子更快**（本仓库
  默认），**`index_sve/原生` <1 = 本算子更快**（生产反馈用的口径），互为倒数。
- `--weighted` 默认是**窄版**（一个模型一块、宽度 ≤70 列，直接看终端）；要贴进
  文档加 `--md`。`--detail` 给的排名按"调用次数 × 单次耗时"，不是按次数——
  两者结论经常不同（RESULTS §2.12.2）。
- `--gen-cases` 里的 `A` 是 **A-br 桶中值映射**，不是 trace 真值；只有 `3→64` 有
  反馈锚定，其余是几何中点假设，用 `--abr-map '4=256'` 覆盖。A=3 与 A=64 的
  单算子比值差 1.6 倍（RESULTS §2.12.4），所以这一项会影响加权结果。

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

# 3. 生产形态（K 512-55296、A=1/3/64、线程 1/6/12/19、x[idx] 走 dispatch）
HPC_AFFINITY=190-227 HPC_THREADS=1,6,12,19 HPC_REPS=3 \
  python benchmarks/bench_production_driver.py --out prod_times.json --fit
# 值域扫描：固定 K 看桶化建表的固定成本（dom>2048 回退单遍，两侧同一形态）
HPC_AFFINITY=190-227 HPC_THREADS=1,19 HPC_REPS=3 HPC_DOM_SWEEP=8192 \
  python benchmarks/bench_production_driver.py
# 按 trace 调用次数加权（无 trace 时用示例权重，只演示算法）
python benchmarks/bench_production_shapes.py --weighted \
  benchmarks/prod_weights.example.json --times prod_times.json

# 3b. 上下文开销：hot / 线程池冷 / 源冷 / autograd + 固定链 + 混合序列平均
HPC_AFFINITY=190-227 HPC_THREADS=1,12,19 \
  python benchmarks/bench_context_overhead.py --ab --json ctx.json

# 3c. 入口口径拆解（x[idx] / torch.ops / Python 包装 / empty，K=0/512/2048）
HPC_THREADS=1,12 HPC_KS=0,512,2048 \
  python benchmarks/bench_entry_overhead.py --ab --json entry.json

# 3d. trace 热点 → 用例 → 测量 → 加权的完整链路（CSV 是示例，不是真 trace）
python benchmarks/bench_production_shapes.py --gen-cases \
    benchmarks/trace_hotspots.example.csv > cases.json
HPC_CASES=cases.json HPC_THREADS=6,12,19 HPC_REPS=3 \
  python benchmarks/bench_production_driver.py --out times.json
python benchmarks/bench_production_shapes.py --gen-cases \
    benchmarks/trace_hotspots.example.csv --per-model \
    --total 'SGI=20800,PPT=4896' > models.json
# 窄版给终端看；--md 出文档表；--detail 加"按耗时"排名
python benchmarks/bench_production_shapes.py --weighted models.json --times times.json --detail
python benchmarks/bench_production_shapes.py --weighted models.json --times times.json --md \
    > weighted.md

# 4. 真实 workload 两进程对拍（M5/shortq/A1/M1/M2，含短调用与高频单元素）
HPC_AFFINITY=190-227 HPC_THREADS=1,8,16,38 \
  HPC_REPS=3 \
  python benchmarks/bench_realpath_driver.py

# 5. 性能回归扫描
HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py

# 6. AI4S workload 专项
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_ai4s.py

# 7. 短行 A-sweep
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_t1.py

# 8. 大块 rowcopy 专项
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  HPC_THREADS=1,2,4,8,16,32,36,38 \
  python benchmarks/bench_t2.py

# 9. 退化定位探针
HPC_AFFINITY=190-227 python benchmarks/degrade_probe.py

# 10. 单 case perf 驱动
MODE=hpc HPC_AFFINITY=190-227 \
  python benchmarks/perf_one.py s1 1 5

MODE=aten HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 \
  python benchmarks/perf_one.py s1 1 5

# 11. 内存系统探针
gcc -O3 -march=native -fopenmp benchmarks/memprobe.c -o /tmp/memprobe -lm
numactl --membind=5 --physcpubind=190 /tmp/memprobe latency 8192 3000000
# REPS 必填：单遍计时包含冷 OpenMP 建队（8t ~1.8ms），会把带宽读数压掉 ~40%
numactl --membind=5 --physcpubind=190-227 /tmp/memprobe bandwidth 2048 16 rd 3
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
