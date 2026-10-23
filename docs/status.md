# 当前进展快照（status）

本文档面向**新读者 / 其他 agent**：读完即可理解项目现状、已经做了什么、还剩
什么。技术细节在各专题文档（见 `index.md` 索引）；这里只给结论与指针。

> 更新规则：每轮实质变更后更新本文（测试数、commit 号、性能结论、路由）。

> **30 秒现状**：T13 完成，13 套件 123/123、真 aten 对拍 44/44。T13 新增多索引
> 小 K 入口快路径（跳过 normalize、一次预校验、dom==1 fill、逐行 memcpy）：
> K=512/A=1 从 0.74x 提升到 1.6-2.0x，A=64 1t 从 0.67x 提升到 3.4x。
> T11 的 31 场景矩阵仍是全形态口径；T12 strided-rows（run16 1.7-2.2x）不变。
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
| T13 | 多索引小 K 入口快路径：跳过 normalize、偏移预计算、dom==1 fill、逐行 memcpy；n≤65536 | 本轮 | ✅ |

## 3. 路由现状（op.cpp `index_impl`）

```
单 mask && 覆盖前 m 维 && 连续 && CPU → gather_mask（单遍压实，T8）
单整数下标 && 末维 K<=4096 && 连续 && CPU → small_single_index（T9；n*A<16384 串行）
多索引 && 同 shape && 连续 && n<=65536 → small_multi_index（跳过 normalize，T13）
K < 1024                          → gather_direct（无线性化；A==1 预校验紧凑循环，T11）
单索引 && A>1 && (A>=4096||A<16) && 连续 self/idx && f32/f64
                                  → M1 raw rowcopy（skip-linearize）
单索引 && !连续 after && col_off 有 ≥4 run → strided_rows_raw（run 块拷贝）
多索引 && 连续 after && A>=4 && 线程数>=16 → 多索引 raw rowcopy
单索引 && B==1 && A==1            → T4 raw 桶化（连续 idx；小值域查表）
                                     └ 大值域 && (!SVE 内核 || 1t) → gather_single_1d
1 线程 && 单索引 && A>1           → A>=4 连续 f32/f64 ? raw rowcopy : gather_fused
A==1 || (连续 && A>=4 && f32/f64) → linearize + gather_index（SVE/桶化/scalar）
其余（A<4 / 非连续小块 / run=1）  → gather_fused
```

> 线程数：T11 起入口 `sync_omp_threads()` 把 libomp team 同步为
> `at::get_num_threads()`，`torch.set_num_threads(t)` 对扩展生效。

详细说明见 `architecture.md` §1 与 `optimizations.md` §9。

## 4. 测试

- **123/123 通过**：test_correctness 25、test_errors 6、test_model 9、
  test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
  test_smallk 7、test_compile_optin 3、test_strided_rows 6、test_rowcopy_routes 4、
  test_small_multi 11。
- 复跑：`for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py tests/test_errors.py tests/test_sve.py tests/test_model.py tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py tests/test_compile_optin.py tests/test_strided_rows.py tests/test_rowcopy_routes.py tests/test_small_multi.py; do python $t; done`
- 注意：测试同一进程串行跑，1 线程用例必须 `try/finally` 恢复
  `torch.get_num_threads()`（test_sve.py 内已如此）。
- **true-aten 值级对拍**（覆盖下测试自洽，须两进程法）：
  `python tests/verify_against_aten.py --mode both`（44/44，参考写
  `agent_space/`）。详见 semantics.md §6.2。

## 5. 性能结论（pin node 5，interleaved median）

**1 线程（T11 矩阵）**：

| 形态 | 修复前 | T11 |
|---|---|---|
| S1/P1 rowcopy A=64 | 1.06x（T5 全 fused 误伤） | **4.46x / 4.42x** |
| rowcopy A=4/16 | 0.87x / — | **2.41x / 3.58x** |
| 1D pure f32 K=2M | 0.75x | **1.07x**（K=5M 1.24x） |
| x[i,j] 多索引 | 0.66x | **1.19x** |
| `x[:,i]` | 0.18x | **2.02x** |
| mask（P3 连续） | ~1.0x | **2.23x**（T8 单遍融合；非连续 0.84-0.98x） |
| 小 K 单下标 `K≤1024` | 0.5-0.7x | **1.31-2.03x**（T9/T11，真实 `x[idx]`） |
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

**全线程矩阵（T11 修正，RESULTS §2，1/2/4/8/16/32/36，node 5）**：31 个
场景里绝大多数 ≥1.0x，低线程最强（RowCopy/1D pure 在 2-8t 为 1.3-4.5x；
单节点内存带宽用少量核即饱和）。仍 <1x 的为：int8/uint8 0.90-0.99x、
complex 0.95-0.97x、非连续 mask 0.84-0.98x、S4 8/16t 0.87/0.76x、
`x[i,j]` 8/16t 0.96/0.89x、int64 4t 0.87x、mask empty 4t 0.74x、
range2048 16t 0.87x（单点抖动/小规模强扩展，均非持续趋势）。逐 case
微架构分析见 `degradation-analysis.md`。

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

## 7. 下一步候选（未做）

- M1 值域桶化（T2 阶段 2）：`gather_rowcopy_raw` 未对下标做值域分析，可再桶化。
- 4B 大 K 大值域 gather：软件预取 / 按 cache line 分桶排序（规避行放大；SVE 与
  标量已持平，换指令无收益）。
- 端到端 AI4S 模型复测（验证 T0-T11 在真实负载上的整体收益）。
- 纯转置 run=1 的低线程尾部（8t 0.84x）：若要继续，需要独立的“固定 col_off
  SVE gather”内核，不是 strided-rows 能解决；见 gap 文档 §19.1。
- 多索引 raw 的 16t 阈值需在其它机器复测；当前只按 920F/node5 定。
- `small_multi_index` 当前只覆盖“所有索引同 shape 且连续、self 连续”；广播、
  非连续索引仍走 `normalize + gather_direct`（后续 P1：修正 `all_full_contig`
  并复用偏移预计算）。
- 扩展 dtype 白名单到 ATen 的完整集合（barebones u16/u32/u64、float8、
  complex32 当前抛 NotImplementedError；见 semantics.md §4）。
- 在真实 AI4S 模型上验证 `HPC_INDEX_COMPILE=1` 的端到端收益与编译时间，再决定
  是否默认开启（当前默认关闭）。

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
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36 \
  python benchmarks/bench_threads.py

# 回归扫描（pin node 5 静默节点，避免共享箱噪声）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py
```
