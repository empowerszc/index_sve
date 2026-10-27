# 当前进展快照（status）

本文档面向**新读者 / 其他 agent**：读完即可理解项目现状、已经做了什么、还剩
什么。技术细节在各专题文档（见 `index.md` 索引）；这里只给结论与指针。

> 更新规则：每轮实质变更后更新本文（测试数、commit 号、性能结论、路由）。

> **30 秒现状**：T15 完成，`test_rowcopy_routes` 路由测试 8/8、13 套件全
> PASS、真 aten 对拍 44/44。T15 修复 T13 残余高线程空档：2D A=16
> K=8192/65536 raw 放行（32-38t 0.73/0.76x → **1.26-2.17x**，hpc 绝对时间
> 砍半）；K=1024 并行阈值线程缩放（38t 1.46x → 2.17x，与 K=1023 拉平）；
> f32/i32+i64 下标大源（>512KB）直连单遍 SVE（K=10M idx64 8t 1.32→
> **2.22x**）。此前 T14 修复 realpath 对拍唯一 <1x 项（shortq A=3
> 0.84x → 2.6x）；T13 多索引小 K 快路径（K=512/A=1 0.74→1.6-2.0x，
> A=64 1t 0.67→3.4x）与 T12 strided-rows 不变。
>
> **本文怎么用**：这是“快照”，不追求完整推导；数字以 `benchmarks/RESULTS.md`
> §2 为准，原因分析看 `degradation-analysis.md`，路由看 §3 与
> `architecture.md`。发现文档间冲突时，优先级：RESULTS §2 > degradation >
> performance > 本文。

## 1. 项目一句话

为鲲鹏 920F（aarch64, SVE2 VL=512）实现的 `aten::index.Tensor` CPU 语义替代
算子。`import index_sve` 后进程中所有 `x[idx]` 自动改走本算子（Dispatcher
CPU 键覆盖，autograd 不受影响）。构建/运行契约见 `README.md` 与
`integration.md`。

## 2. 已完成（按 commit 时间序）

| 轮 | 内容 | commit | 状态 |
|---|---|---|---|
| 基础 | 三段流水线 normalize→linearize→gather；SVE512 8B/4B gather、rowcopy、scalar 回退；K<1024 `gather_direct` | （历史） | ✅ |
| T0 | 自适应并行粒度 `adaptive_grain`（grain.h） | 472f8a7 | ✅ |
| T1 | rowcopy 门槛 A>=16 → A>=4 | 09a406c | ✅ |
| T2 p1 | M1 单索引大块 skip-linearize `gather_rowcopy_raw` | 320bcb7 | ✅ |
| T3 | 融合单遍 `gather_fused`（A<4 / 非连续小块，不物化 idx） | db8251b | ✅ |
| T4 | 小值域桶化 `bucket_gather_pure` + int32 linearize 直写 | 4fd733b | ✅ |
| T4+ | raw 桶化 `bucket_gather_pure_raw`（单索引 B==1 A==1）+ 无锁归约 | 10921c1 | ✅ |
| T4++ | raw 桶化补 OOB 越界检查（曾静默错误） | 31687d4 | ✅ |
| T5 | 1 线程回归修复（单遍标量 + nb==1 快路径） | bd21485 | ✅ |
| T6 | 跨设备索引校验 + `aten::index.Tensor_out` 覆盖 | f791c39 | ✅ |
| T8 | mask 单遍融合 compaction（跳过 nonzero） | 0dd7533 | ✅ |
| T9 | 小 K（≤4096）单整数下标精简路径 + C++ 直接覆盖 aten CPU 内核 | da9a203 | ✅ |
| T10 | `torch.compile` opt-in（`HPC_INDEX_COMPILE=1`：fake + aten fallback + 独立 cache 目录） | 1cd77ce | ✅ |
| T11 | 线程矩阵方法学修正（libomp ICV 同步）、`gather_direct` A==1 紧凑循环、2B/1B/complex 单遍、非连续 stride 正确性修复、1t A>=4 恢复 raw rowcopy | fc26cf8 | ✅ |
| T12 | fastindex 差异补齐：strided-rows、≤64B memcpy、A<16 raw、B==1、grain 64、多索引 A≥4（≥16t）raw | 6603406 | ✅ |
| T13 | 多索引小 K 入口快路径：跳过 normalize、偏移预计算、dom==1 fill、逐行 memcpy；n≤65536 | a91d566 | ✅ |
| T14 | `small_single_index` 短行 memcpy 阈值 A>=4 → A>=2（realpath 对拍发现 shortq A=3 0.84x） | （未提交） | ✅ |
| T15 | 中等行宽 raw 放行 `A<=64 && B*K<=65536`（2D A=16 K=8192 32/36t 0.73/0.76x）；f32/i32+i64 下标大源（>512KB）直连单遍 SVE `pure_gather_sve_4b_i64idx`（uzp1 收窄复用 u32index）；`small_single` 并行阈值 `max(16384, 512×nt)`（38t K=1024 边界） | （未提交） | ✅ |

## 3. 路由现状（op.cpp `index_impl`）

```
单 mask && 覆盖前 m 维 && 连续 && CPU → gather_mask（单遍压实，T8）
单整数下标 && 末维 K<=4096 && 连续 && CPU → small_single_index（T9；n*A<max(16384,512×nt) 串行）
多索引 && 同 shape && 连续 && n<=65536 → small_multi_index（跳过 normalize，T13）
K < 1024                          → gather_direct（无线性化；A==1 预校验紧凑循环，T11）
单索引 && A>1 && (A>=4096||A<16||(A<=64&&B*K<=65536)) && 连续 self/idx && f32/f64
                                  → M1 raw rowcopy（skip-linearize）
单索引 && !连续 after && col_off 有 ≥4 run → strided_rows_raw（run 块拷贝）
多索引 && 连续 after && A>=4 && 线程数>=16 → 多索引 raw rowcopy
单索引 && B==1 && A==1            → T4 raw 桶化（连续 idx；小值域查表）
                                     └ 大值域 && f32/i32 数据+i64 下标 && 源>512KB && >1t
                                        → gather_single_1d_sve（T15-C 单遍 SVE）
                                     └ 其余大值域（无 SVE 内核 dtype/1t/小源）
                                        → gather_single_1d（单遍标量）
1 线程 && 单索引 && A>1           → A>=4 连续 f32/f64 ? raw rowcopy : gather_fused
A==1 || (连续 && A>=4 && f32/f64) → linearize + gather_index（SVE/桶化/scalar）
其余（A<4 / 非连续小块 / run=1）  → gather_fused
```

> 线程数：T11 起入口 `sync_omp_threads()` 把 libomp team 同步为
> `at::get_num_threads()`，`torch.set_num_threads(t)` 对扩展生效。

详细说明见 `architecture.md` §1 与 `optimizations.md` §9。

## 4. 测试

- **127/127 通过**：test_correctness 25、test_errors 6、test_model 9、
  test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
  test_smallk 7、test_compile_optin 3、test_strided_rows 6、test_rowcopy_routes 8
  （T15 增至 8：raw 路由/大 K 回退/f32+i64 路由/阈值缩放）、test_small_multi 11。
- 复跑：`for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py tests/test_errors.py tests/test_sve.py tests/test_model.py tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py tests/test_compile_optin.py tests/test_strided_rows.py tests/test_rowcopy_routes.py tests/test_small_multi.py; do python $t; done`
- 注意：测试同一进程串行跑，1 线程用例必须 `try/finally` 恢复
  `torch.get_num_threads()`（test_sve.py 内已如此）。
- **true-aten 值级对拍**（覆盖下测试自洽，须两进程法）：
  `python tests/verify_against_aten.py --mode both`（44/44，参考写
  `agent_space/`）。详见 semantics.md §6.2。

## 5. 性能结论（pin node 5，interleaved median）

**1 线程（T11 基线；T13 复测 S1/P1 约 3.0x）**：

| 形态 | 修复前 | T11 |
|---|---|---|
| S1/P1 rowcopy A=64 | 1.06x（T5 全 fused 误伤） | **4.46x / 4.42x** |
| rowcopy A=4/16 | 0.87x / — | **2.41x / 3.58x** |
| 1D pure f32 K=2M | 0.75x | **1.07x**（K=5M 1.24x） |
| x[i,j] 多索引 | 0.66x | **1.19x** |
| `x[:,i]` | 0.18x | **2.02x** |
| mask（P3 连续） | ~1.0x | **1.78x**（T8 单遍融合；非连续 0.86-0.98x） |
| 小 K 单下标 `K≤1024` | 0.5-0.7x | **1.47-2.38x**（T9/T11/T12，真实 `x[idx]`） |
| 2B/1B/complex | 0.5-0.97x | **0.95-1.53x**（1t；高线程至 2.4x） |

**T12 新增（同进程 A/B，node 5）**：

| 形态 | 旧 | 新 | 提升 |
|---|---|---|---|
| run16 A1024 1t | 10.56ms | 4.74ms | **2.23x** |
| run16 A1024 8t | 1.59ms | 0.934ms | 1.71x |
| run10 A640 1t | 4.71ms | 3.50ms | 1.36x |
| A=8 多线程 | — | — | 1.21-1.37x |
| A=15 多线程 | — | — | 1.15-1.37x |
| 多索引 A=64 16/36t | — | — | 1.12/1.16x |
| 1D K=8192 36t（相对 1t） | 0.76x | 0.98x | 小任务高线程不再负扩展 |

纯转置 run=1（tail 无连续后缀）保持 `gather_fused`，8t 仍 0.84x；块拷贝无法
改善，原因见 `index_sve_vs_fastindex_gap_20261023.md` §19.1。

**T13 多索引小 K（真实 `x[i,j]`，K=512，重复 3000 次，µs/调用）**：

| 形态 | 线程 | aten | T13 | 比值 |
|---|---|---:|---:|---:|
| B=8，A=1 | 1 / 8 / 36 | 23.0 / 29.9 / 31.4 | 14.5 / 15.1 / 18.8 | 1.59 / 1.99 / 1.67x |
| B=8，A=4 | 1 / 8 / 36 | 36.0 / 52.9 / 48.8 | 18.9 / 22.9 / 29.9 | 1.90 / 2.31 / 1.64x |
| B=8，A=64 | 1 / 8 / 36 | 184.3 / 39.1 / 45.4 | 54.0 / 29.9 / 33.9 | 3.41 / 1.31 / 1.34x |

B=32/2 同趋势。旧路径在 1t 为 0.67-0.73x；新路径把固定开销和多索引广播解码
都移出了热点，且 dom==1 时只做一次 fill。

T13 小 K 复测（`bench_smallk_driver.py`）：1D K<=1024 为 1.47-2.38x；
2D A=16 K=128-1024 为 1.37-3.15x；剩余高线程空档为 1D K=65536 16-36t
0.81-0.88x、2D A=16 K=8192 32/36t 0.73/0.76x（RESULTS §2.3）。

**T15 小 K 复测（RESULTS §2.3 新表，2026-10-26）**：2D A=16 K=8192 32-38t
**1.26-1.30x**（hpc 73-79µs 砍半到 36-37µs；aten 绝对时间随箱态漂移，以
同轮交替比值为准）、K=65536 32-38t **2.01-2.17x**；K=1024 与 K=1023 全
线程拉平（38t hpc 17.89 vs 17.86µs，T14 轮 38t 曾 1.46x）；1D K<=1024
1.83-2.36x。剩余 <1x 仅 1D 小源家族：K=8192 36/38t 0.95x、K=65536 32-38t
0.90-0.92x——T15-C 判明小源（L1 常驻）下 4B SVE gather 微码吞吐反而低于
标量，保持标量；残留为 `gather_bucket_pure_raw` 前置值域扫描 ~15µs（未来项）。

**T14 真实 workload 对拍（`bench_realpath_driver.py`，RESULTS §2.5，两进程
`x[idx]` 全链路，1/8/16/38t）**：

| 形态 | 1t | 8t | 16t | 38t |
|---|---|---|---|---|
| M1 权重块（≤16KB/行，E=20k） | 13.0-13.6x | 5.1-5.8x | 2.7-3.2x | 1.2-1.4x |
| M5 标量循环 / A1 单元素高频 | 2.6-3.8x | 2.6-3.7x | 2.6-3.7x | 2.6-3.7x |
| shortq A=2-4 K=64-1023 | 2.6-4.3x | 2.6-4.3x | 2.6-4.3x | 2.6-4.3x |
| M2 (20000,8/64/256) E=100k | 1.2-3.5x | 1.6-2.7x | 1.5-2.4x | 1.1-4.9x |

全形态 ≥1.11x（38t 的 1.1-1.4x 为带宽饱和）。T14 修复 shortq A=3（唯一 <1x
项，0.84-0.99x → 2.6x+）。aten 在 16/38t 的 K=1023×A≥3 有 539-615µs 的
fork-join 病理（我们串行免疫），对应 22-26x 格子不作收益宣传。

**T13 全量复测（RESULTS §2.4，31 场景 × 7 线程，node 5）**：绝大多数
≥1.0x，RowCopy 仍最强（S1/P1 1t 3.0x、rowcopy A=4/16 1t 1.9/2.1x）。
T13 只改多索引小 K，未引入新退化；仍 <1x 的持续项集中在五类（另加单点抖动）：

- **每线程工作量小**：S4 8/16t 0.98/0.91x、`x[i,j]` 8/16t 0.97/0.92x；
  每线程 50-100KB，fork-join/调度 > 并行收益。
- **1B 行放大**：int8 8/16t 0.85/0.82x、uint8 0.99/0.94x（focused 复测
  为 0.98/0.94、0.97/0.86x）；64B line 只用 1B，aten 同为标量。
- **无 SVE dtype**：complex64 1t 0.95x，2t+ 持平；标量单遍的边界检查开销。
- **非连续 mask**：1D stride2 mask 1-16t 0.86-0.98x；T8 只接连续 mask，
  回退 `nonzero`（物化 int64）+ linearize。
- **小 K 高线程路由空档（T15 已修大半）**：2D A=16 K=8192/65536 已修
  （raw 放行，32-38t 1.26-2.17x）；1D K=65536 大源已修（f32/i32+i64 直连
  SVE 单遍，K=10M idx64 8t 1.32→2.22x），小源判明标量更快（每元素标量
  L1 命中 ~7.1ns vs 4B gather 微码/发射下限 ~18ns，大源反转为 ~19 vs
  ~10ns，见 degradation-analysis.md §5.9），32-38t 0.90-0.92x
  残留为前置 bucket 扫描 ~15µs（未来项）。
- 单点抖动：range2048 16t 0.93x、mask 4t 0.97x；相邻线程 >1.2x，判为噪声。

32/36t 的 2-10x 多数是 aten 在带宽饱和后自身变慢，不应作为收益宣传。
逐 case 微架构分析见 `degradation-analysis.md` §5.8。

> 旧 RESULTS §3.6.1/§3.9 的 2-36t 数据因 libomp/libgomp 两套 ICV 失效（扩展实际
> 用 38 个 worker），已标注作废；不要再用其中 "≥1.06x" 的结论。

**访存上限（不可修，非 bug）**：4B（f32/i32）大 K 随机 pure gather 受 64B
cache line 放大（4B 只用 4B/行）+ DRAM 延迟 + 地址翻译限制（实测 dTLB miss
≈1.1 次/访问，本地 DRAM 115-192ns）。同构建 A/B 实测 SVE 4B 与标量持平
（8MB 1.85 vs 1.86 ms、256MB 7.46 vs 7.45 ms，8/38t），换指令/回退标量无效，
**不是 SVE MLP 缺陷**（RESULTS §3.7；早期 0.75 vs 3.4 GB/s 探针归因已撤回）。
36t 时 4B 行流量 73-76GB/s，已达单节点读带宽上限 73.3GB/s（degradation §6.4），
所以高线程倍率再高也只是把行带宽用满，不会改善“有用字节/行”。
T4 桶化在小值域规避行放大，大值域靠预取/行 gather 等待未来优化。不要据此
推断"所有场景都 ≥1x"：`x[:,i]`/1B/complex 仍有结构性差距（见
`degradation-analysis.md`）。

## 6. 已知边界 / 注意事项

- 覆盖是一次性全局副作用：一个进程只能装一个 Python CPU 覆盖
  `aten::index.Tensor`；`HPC_INDEX_NO_OVERRIDE=1` 可跳过（基准对拍用）。
- 索引张量必须在 CPU；直接调用 `hpc_index::*` 时 meta 输入会干净报错（T6 起
  不再 SIGSEGV；T10 后未开 compile 时为 PT2 "no Meta kernel"，开启后走 fake）。
  真实 `x[idx]` 的 meta 行为与原生 aten 一致。`aten::index.Tensor_out` 也已覆盖。
- 扩展链 BiSheng `libomp.so`，`LD_LIBRARY_PATH` 必须含 BiSheng lib，否则
  import 报 `libomp.so: cannot open`（`README.md` 快速开始）。
- **`torch.compile` 下默认不生效**：Inductor 把 `aten::index` 内联成自己生成的
  kernel，dispatcher 覆盖被绕开（实测慢 2.6x）。设 `HPC_INDEX_COMPILE=1`（import
  前）可 opt-in：注册 fake + 把 lowering 永久设为 fallback + 切换独立 inductor
  cache 子目录，运行期回到 dispatcher；与 `HPC_INDEX_NO_OVERRIDE=1` 冲突会报错。
  细节与限制见 `torch-compile.md` §6。
- 慢形态与"为什么不 fallback"的深度分析见 `fallback-analysis.md`：核心结论是
  4B（f32/i32）大 K 大值域 pure gather 受 64B cache line 放大 + DRAM 延迟限制
  （SVE 4B 与标量持平，非指令 MLP；RESULTS §3.7）；A<4/非连续/2B-1B 多数与 aten
  持平或略低（0.9-1.0x），`x[:,i]` 在 T11 前 0.4-0.7x、A==1 紧凑循环后 1.1-2.0x；
mask 经 T8 后 1.3-4.2x；
  fallback 的 Python wrapper 开销 ~45µs/调用对以 RowCopy 小张量为收益主体的
  本算子是不可接受的全局回归。

## 7. 演进方向（后续再看从这里开始）

> 2026-10-27 收尾时的建议排序。P1/P2 是最后两个"可修的"劣化家族（其余
> <1x 项为硬上限或使用侧问题）；P3 是验证项，其结果决定后续投入是否值得。
> 各项根因推导见 `degradation-analysis.md` §5.9。

| 优先级 | 项目 | 预期收益 | 工作量 | 详见 |
|---|---|---|---|---|
| **P1** | 1D 小源跳过 `gather_bucket_pure_raw` 前置值域扫描（并入 gather 单遍） | K=8192/65536 小源 36-38t 0.90-0.95x → ≥1x（省 ~15µs / 总 42-53µs，fork-join 占 30-50%） | 小 | degradation §5.8-C/§5.9.2、§8-1 |
| **P2** | 非连续 mask 的 T8 stride 扩展（计数/压实按 mask stride 读，去 nonzero 物化） | 0.86-0.98x → 1.2-1.5x（8t 实测可省 ~1ms） | 中 | degradation §5.8-D、§8-2 |
| **P3** | 端到端 AI4S 模型复测 | 验证 T0-T15 累计收益是否传导到真实负载（决策项） | 中 | performance §3/§4.1 |
| P4 | f64+i32 下标经 `svunpklo` 加宽走 8B gather | ≈持平 → 小幅正收益 | 小 | degradation §8-3 |
| P5 | M1 值域桶化（T2 阶段 2）：`gather_rowcopy_raw` 做下标值域分析 | 高重复下标场景再提升 | 中 | optimizations §3.5 |
| P6 | 4B 大 K 大值域：按 cache line 分桶/排序 | 受 64B 行放大硬上限，只有改访问结构有效 | 大 | degradation §5.9.3 |
| P7 | `small_multi_index` 扩广播/非连续索引（修正 `all_full_contig` + 复用偏移预计算） | 覆盖面，非性能 | 中 | — |
| P8 | dtype 白名单扩展（barebones u16/u32/u64、float8、complex32） | 覆盖面（当前抛 NotImplementedError） | 小-中 | semantics §4 |
| P9 | `HPC_INDEX_COMPILE=1` 真实模型端到端验证 | 决定是否默认开启 | 中 | torch-compile §6 |

**不建议再投入**：1B/complex（64B 行放大是硬上限，aten 同为标量，换指令
无解）；纯转置 run=1 低线程尾部（8t 0.84x，需独立"固定 col_off SVE
gather"内核，收益面窄；gap 文档 §19.1）。

**换机器注意**：多索引 raw 的 16t 阈值与 T15 各门值（源>512KB、
512×nt）均按 920F/node5 实测标定；L2 容量、内存带宽、fork-join 成本
不同则需重跑聚焦 A/B 再定门值。

## 8. 复现 / 复测命令

```bash
# 全量测试
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py \
         tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py \
         tests/test_compile_optin.py; do python $t; done

# true-aten 值级对拍（正确性，非性能）
python tests/verify_against_aten.py --mode both

# 全线程矩阵（RESULTS §2，pin node 5 静默节点，25 iters interleaved）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
  python benchmarks/bench_threads.py

# 真实 workload 对拍（RESULTS §2.5，短调用/真实路径结论首选）
HPC_AFFINITY=190-227 HPC_THREADS=1,8,16,38 HPC_REPS=3 \
  python benchmarks/bench_realpath_driver.py

# 回归扫描（pin node 5 静默节点，避免共享箱噪声）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py
```
