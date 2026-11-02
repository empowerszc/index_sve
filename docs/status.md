# 当前进展快照（status）

本文档面向**新读者 / 其他 agent**：读完即可理解项目现状、已经做了什么、还剩
什么。技术细节在各专题文档（见 `index.md` 索引）；这里只给结论与指针。

> 更新规则：每轮实质变更后更新本文（测试数、commit 号、性能结论、路由）。

> **30 秒现状**：R10 直写完成（2026-10-29 第二轮，专项文档
> [output-layout.md](output-layout.md) §8）：上一轮布局修复让 cl/置换 self
> 的 `x[idx]` 输出保持原布局（正确），代价是 tmp+copy_ 一趟输出带宽
> （cl4d 行索引 0.30-0.42x vs aten）。R10 依据"cl 尾块是 dense 置换、
> rowcopy 纯块拷贝对源/出同置换语义等价"实现行块直写
> （`rowcopy_direct_ok` 资格判定 + dispatch 顶部守卫，内核零改动）：
> **cl4d 行索引 8.1-10.0ms → 1.3-2.1ms，比值 0.30-0.42x → 1.45-2.54x**；
> cl 行索引与连续行索引基本拉平。15 套件 **174/174**、E2E 差分
> **124/124** + 线程矩阵 46 例 ALL OK、对拍 44/44；normalize.cpp 编译警告清零。
> 同日第一轮完成输出布局修复（复刻 TensorIterator 置换恢复）+ E2E 差分
> harness；此前 T15/全线程档复测见 RESULTS §2.8。
>
> **再补一条按形态的结论（2026-10-31）**：把 K 换到生产区间 512-55296、线程
> 换到 6-19 后，1D A=1 从 1t 的 1.11-1.61x 掉到 **0.66-1.10x**（rowcopy 与
> K≤2048 桶化不受影响）。旧矩阵的 K=2M/10M 与 1/608t 是两个端点，没覆盖这段；
> 见 §5.1、RESULTS §2.10、degradation §6.5。
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

> 本表是**变更日志（历史）**，按轮次记录做了什么；当前性能与路由见 §3/§5.1。

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
| T14 | `small_single_index` 短行 memcpy 阈值 A>=4 → A>=2（realpath 对拍发现 shortq A=3 0.84x） | 428bafd | ✅ |
| T15 | 中等行宽 raw 放行 `A<=64 && B*K<=65536`（2D A=16 K=8192 32/36t 0.73/0.76x）；f32/i32+i64 下标大源（>512KB）直连单遍 SVE `pure_gather_sve_4b_i64idx`（uzp1 收窄复用 u32index）；`small_single` 并行阈值 `max(16384, 512×nt)`（38t K=1024 边界） | 428bafd | ✅ |
| 补测 | autograd 专项回归 `test_autograd.py`（16 项：梯度 vs `index_add_` 独立参考、负索引/重复/mask/多索引/非连续/gradcheck、路由计数器、`torch.ops` 直调无 autograd 公式） | 24cd8b3 | ✅ |
| 修复 | pybind 调试出口 `_hpc_index.index` typed 句柄补 `const&`（自初始 commit 即抛签名错，零调用方未暴露；测试扩三直调入口断言） | 0397eb5 | ✅ |
| 基准 | AI4S workload 全线程档复测（31 场景 + realpath 两进程对拍 × 1/2/4/8/16/32/36/38 全 8 档，RESULTS §2.8；summary §6.2 全档表；mask empty/complex64 已在 §2.9.1 复测） | 557cdbf | ✅ |
| 修复 | 输出内存布局与 aten 逐维一致：`aten_out_strides` 复刻 TensorIterator 置换恢复（strides/should_swap 平局规则），`empty_strided` 分配 + 非连续 tmp→`copy_`；新增 `test_output_layout.py`（25 例，期望值为无覆盖进程实测钉入）+ `verify_e2e.py`（E2E 两进程差分 120 例，值+shape+dtype+stride 比对 + 线程矩阵；R10 后增至 124 例）；专项文档 output-layout.md | 5eaa3b1 | ✅ |
| R10 | cl/置换布局行块直写：`rowcopy_direct_ok` 资格判定（源/出尾块同 dense 置换 + out 前导块行主序等）+ `IndexSpec::rowcopy_direct` + dispatch 顶部守卫（只允许块拷贝内核）+ `gather.cpp` rowcopy_sve 守卫放宽；cl4d 行索引 0.30-0.42x → **1.45-2.54x**；normalize.cpp 编译警告清零；test_rowcopy_routes 增至 14（直写 2 例含 1t/8t 双档 + 排除 3 + cl3d linearize 臂 1）；verify_e2e 增至 124 例（g5big 大 K 直写 4 例，线程矩阵 46 例） | （本轮） | ✅ |
| 基准 | 生产形态基准：`bench_production_shapes.py` + `bench_production_driver.py`（K 512-55296、A=1/3/64、线程 1/6/12/19、`x[idx]` 走 dispatch、两进程交替、`--weighted`/`--gen-cases`/`--fit`）；实测 1D A=1 在 6-19t **0.66-1.10x**（1t 1.11-1.61x），机理写入 degradation §6.5 | （本轮） | ✅ |
| 基准/裁决 | 上下文开销基准 `bench_context_overhead.py`（hot / 池冷 / 源冷 / autograd + K=1 固定链 + A 敏感性 + K=2 小形态 + mask `nonzero` + 1000 次混合序列）；对反馈"固定开销占 93-95%"逐条裁决写入 degradation §6.5.7，原始表 RESULTS §2.11 | （本轮） | ✅ |
| 测试加固 | 两个靠 `x[...]` 的套件补前置守卫（`index_sve.OVERRIDE_ACTIVE`，`NO_OVERRIDE=1` 直接 rc=1）+ 入口计数 `dispatch_count()` 给负向断言配正向控制（覆盖静默失效时 6 项全挂、假通过归零）；排查过程与检查表写入 debugging-playbook.md | （本轮） | ✅ |
| 文档核查 | 全量核对 `op.cpp` 行号引用：§4 步骤清单/小节范围/附录 A/B 共 40 余处晚 3-5 行（注册行写成 633 应为 630、`index_into` 写成 49 应为 52 等）；根因是 `check_doc_anchors.py` 把 §4 的裸引用判给了 gather.cpp（上一张表出现过 `gather.cpp:750`）。工具收紧四条规则：小节标题文件优先、`符号 :NNN` 定义精确匹配、紧邻符号必须落在函数体内或 ±3 行内被调用、表格行按符号配对；四条均做反证 | （本轮） | ✅ |
| 测试补口 | 源大小路由门补边界用例（`numel*element_size > 512KB`，判据来自 degradation §5.9.1 的 4B gather 成本交叉点）：512KB 整走标量、+4B 走 SVE，两侧各钉一点；同时订正 `test_f32_i64_1d_route` 注释里被撤回的 0.63/1.2ns 旧口径为 7.1/18ns | （本轮） | ✅ |
| 基准/裁决 2 | 基准口径审计：`bench_entry_overhead.py` 量出入口偏置（Python 包装 +10µs、`torch.ops` +9µs）；`bench_production_shapes.py` 增加 `--cases`/`--per-model`/`--total`/`--detail`/`--abr-map`/CSV fixture 与"两套口径 + ①③覆盖率区间"；按反馈热点重建 trace 复现加权，得 0.767/0.780/0.783（反馈是 1.116/1.184/0.980）。原始表 RESULTS §2.12，裁决 degradation §6.5.8 | （本轮） | ✅ |
| 分析/文档 | **路由合并与探测成本**：三层成本模型（入口必经 / O(1) 判定 / O(K) 探测）+ 四个测量实验（先确认走了哪条路 / 门边界挪一格 / 单遍扫描裸价 / 语料普查）+ 可合并清单 M1-M6（只有 M1/M2 能换到时间，其余纯维护）+ 不该合并的两类 + 新增路由的五条检查表 + 按符号名跨版本定位。实测结论：短调用（K≤4096，含 2D A=8）不进路由链；同局部性下值域门两侧是约 2 倍台阶——原始数据与复现命令在 RESULTS §2.13，新增 `benchmarks/bench_route_probe.py`。新文档 `route-merge-analysis.md`，`status` §3/§7-R1、`code-walkthrough` §4.2、`index.md` 挂指针 | （本轮） | ✅ |
| 基准/裁决 3 | **路径 × 规模 × 线程矩阵**：`bench_path_matrix.py`（35 用例 × 1/8/16t，aten/hpc 两进程交替，每用例写死期望路由并与计数器核对，98/98 一致）+ 兄弟对照（同工作量只翻一个谓词）；数据 RESULTS §2.14，裁决 `path-matrix.md`。结论：直写资格翻面即数量级（1t 22.9x / 16t 7.9x）；行拷贝族多线程仍稳定为正；两遍模型在 1D 小源8/16t 落到 1.00 以下（M1/M2 的直接依据）；`gather_direct` 不是快路径；mask 结论随规模翻转 | 见 RESULTS §2.14 | ✅ |
| T16 | **int32/int64 数据进 raw 行拷贝**：M1 raw 支 dtype 白名单加 `kInt/kLong`（另两处守卫不动——strided-rows/多索引仍要在 raw 下解码）；同时补 `copy_contig_sve` 的宽度分派（原来只有 float/double 两个 `if constexpr`，**其它 dtype 行宽 >64B 时一字不拷**，是静默正确性地雷），2B/1B/complex128 退回 memcpy 兜底。A=3 8t 100→35.5µs（0.57x→1.64x，与 f32 兄弟同档）；测试 +3（含 f16 负向对照），全量 178/178；数据 RESULTS §2.15 | 见 RESULTS §2.15 | ✅ |
| T17 | **H1 行拷贝软件预取**：`rowcopy_raw_single`/`rowcopy_sve` 提前 D 行读下标 + `prfm pldl1keep`，D 按行字节分档（≤64B→16、65-512B→8、>512B→关），运行时旋钮 `HPC_PF_DIST`；越界下标不预取、报错仍落第 1 个坏下标。1t 实测 A=3 79.1→31.9、A=16 83.0→28.5、A=32 94.5→36.7、A=64 108.2→48.0 ns/行（对 aten 1.6-2.6x → 2.1-6.0x），A=256 全档为负故关掉；perf 证明缺行数不变、周期数掉 2.5-3.1 倍。测试 +7（`test_rowcopy_pf.py`），数据 RESULTS §2.18/§2.19 | 见 RESULTS §2.18 | ✅ |
| 分析/文档 | **perf 在飞条数**：`perf_analyze.py`（两段相减规避 7.2s 启动；`L1rf/行` 当缺行数）把"aten 只有 1 条 line 在飞"从推断变实测（0.85-1.05 条，三种形态两种口径互校），`perf record` 确认热点是 `loop_2d_from_1d`；H1 的机理证据（缺行数 1.02→1.02、周期 152→49）；宽行 A=256 已有 ~16 条在飞（故预取无用）。数据 RESULTS §2.19 | 见 RESULTS §2.19 | ✅ |
| 分析/文档 | **理论空间（roofline）**：`memprobe` 新增 `mlp`（S 条独立依赖链，精确设"在飞条数"）/`idxgather`（下标从内存读）/`rowgather`（plain/memcpy/pipe/pf8/pf16 五写法）/THP 开关，新探针 `sveprobe.c`（SVE 拷贝指令下界、SVE gather、NT store），`bench_roofline.py`（aten/hpc 分进程 + `index_select` 同库参考）。结论：成本 = 137ns ÷ 在飞条数，硬件每核给 16-20 条（饱和 8.6-8.9ns），aten 行路径实测 ≈1 条在飞、本仓 1.7-1.8 条；**预取是最大杠杆（1.8-4.6x）**，逐族剩余 2.6-7.1x、已兑现 14-39%；指令/NT store/大页/排序四条为负结果或陷阱。数据 RESULTS §2.16/§2.17，新文档 `headroom.md` | 见 RESULTS §2.16/§2.17 | ✅ |
| 分析/文档 | **桶化的成本结构：三格三张账**：`idxgather` 放开小数 MB 后量出**两种下界**——L1 常驻 0.66-0.71 ns/元素（1t 就到、不随线程数变）vs DRAM 5.62-5.98（RESULTS §2.20）。用它量生产最贵三格：SGI K=13824/12t 每线程 1152 元素、28.6 ns/元素（固定成本+串行段）、PPT K=55296/19t 16.1（遍数+串行表填充）、Sonata K=256000/6t 4.53（≈L1 延迟÷1.1，**一条在飞**）。结论：上轮把三格当成"大值域 1D gather"是误归因，H2 拆成 H2a（e2e≈0）/H2b（单格 4-5x）、H4/H7 提为小 K 的第一杠杆；`headroom.md` 重写为完整的瓶颈清单+差距逐项分解+端到端上限推算 | 见 RESULTS §2.20 | ✅ |
| 分析/文档 | **本机没有 L3，1-4MB 缓坡改按 L2 命中率解释**：`lscpu -C`/sysfs 只到 L2（L1d 32KB 8-way、L2 768KB 12-way、每核私有），整机 16 个带 CPU 的 node（各 38 核）+ 16 个纯内存 node；`memprobe freq` 核对主频 **1.99GHz**（旧稿 2.07 是混合负载反推，差 3.5%）。用 `l1d_cache_refill`/`l2d_cache_refill` 量出命中率：512KB 0.7%、768KB 14%、1MB 32%、2MB 69%、4MB 85%、8MB 93%——**命中只发生在 L2**；4MB 之后 miss 率饱和而 latency 仍从 88.6 涨到 131（单次 miss 变贵，推断 DRAM row buffer）。另发现探针陷阱：指针追逐的工作集是**置换环长**，1MB 数组换种子 2339 行 vs 10040 行 → 实测 26.1 vs 46.4ns，故 1MB 那格不能当常数。数据 RESULTS §2.16.1 | 见 RESULTS §2.16.1 | ✅ |
| 分析/文档 | **headroom 重写为“可读 + 可复核”**：§1.0 补从零开始的访存背景（line/依赖链 vs 独立访问/RFO），§1.1–§1.9 每节都加“这个数怎么测出来的”，§2.1 给出 A=16 的代入算例（48 拍 vs 49 拍），§3.2 把“等效在飞 = 274 ÷ 每行周期”写成可复用的换算；新增 **§9 方法论**（五条原则、探针三种形状、一次从“怀疑是 L3”到定案的六步完整测量、十个坑、四个换算、复核清单），并把“取最好值 vs 取中位数”用同日复测 12.8/11.7/11.3ns 说清（RESULTS §2.16.4）。正文 761 → 1060 行 | 见 RESULTS §2.16.1/§2.16.4 | ✅ |

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

normalize 之后、按 out 布局分岔（2026-10-29 起）：
out 连续                          → index_into 直写
!连续 out && rowcopy_direct_ok     → 行块直写（R10；dispatch 顶部守卫只允许块拷贝内核）
!连续 out 其余                     → tmp 行主序 + copy_
```

> 线程数：T11 起入口 `sync_omp_threads()` 把 libomp team 同步为
> `at::get_num_threads()`，`torch.set_num_threads(t)` 对扩展生效。
>
> 输出分配（2026-10-29 起）：主路径（上述树中 normalize 之后的所有分支）
> 经 `aten_out_strides` 按 TensorIterator 置换恢复语义分配输出
> （`empty_strided`），连续 self 恒连续、cl/置换 self 行索引族保持原布局。
> 非连续 out 的写入：`rowcopy_direct_ok` 资格判定（单索引/idx 连续/A>=4/
> K>=1024/f32-f64/SVE512/源尾块 dense 置换/out 尾块逐维同 stride/out 前导块
> 行主序）通过则**行块直写**（dispatch 顶部守卫只允许块拷贝内核，
> R10）；否则 tmp+`copy_`。三个快路径（mask/small_single/small_multi）均有
> `self.is_contiguous()` 守卫，输出恒连续不受影响。机制与移植指南见
> `output-layout.md` §5/§8。

详细说明见 `architecture.md` §1 与 `optimizations.md` §9。**要不要把其中几条合并**、
合并后能不能省掉一遍值域扫描、新增路由前该问什么，见
`route-merge-analysis.md`（方法论 + 可合并清单 + 五条检查表）。

**每条路各值多少**——包括哪些是"仅兜底"、哪些在高线程数会变成负收益——见
`path-matrix.md`（按谓词组织的路径清单 + 兄弟对照 + 五档裁决），原始数据在
RESULTS §2.14。该文 §3 的表就是"改哪条路由 → 跑哪些 case"的索引。

## 4. 测试

- **175/175 通过**：test_correctness 25、test_errors 6、test_model 9、
  test_normalize 9、test_smoke 2、test_sve 25、test_index_out 7、test_mask_fused 9、
  test_smallk 7、test_compile_optin 3、test_strided_rows 6、test_rowcopy_routes 15
  （R10 增至 14：cl 直写路由 2 例含 1t/8t 双档 + 排除形态 3 例 +
  cl3d linearize 臂 1 例；+1 为源大小门边界）、test_small_multi 11、
  test_autograd 16、test_output_layout 25（输出布局专项，期望 stride 为无覆盖
  进程实测真值钉入）。
- 复跑：`for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py tests/test_errors.py tests/test_sve.py tests/test_model.py tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py tests/test_compile_optin.py tests/test_strided_rows.py tests/test_rowcopy_routes.py tests/test_small_multi.py tests/test_autograd.py tests/test_output_layout.py; do python $t; done`
- 注意：测试同一进程串行跑，1 线程用例必须 `try/finally` 恢复
  `torch.get_num_threads()`（test_sve.py 内已如此）。
- **true-aten 值级对拍**（覆盖下测试自洽，须两进程法）：
  `python tests/verify_against_aten.py --mode both`（44/44）；E2E 全链路
  `python tests/verify_e2e.py --mode both --threads`（124/124，真实
  `x[...]` 语法 × 布局 × autograd × 模型 workload，比对含 stride；
  参考写 `agent_space/`）。详见 semantics.md §6.2 与 output-layout.md §3。

## 5. 性能结论（pin node 5，interleaved median）

> 口径：比值 = aten ÷ 本算子。**当前值唯一出处 = `benchmarks/RESULTS.md` §2.9**：
> §5.1 是摘要，§5.2 是逐轮历史（保留用于追溯，**不代表当前状态**）。

### 5.1 当前结论（R10 轮，`330d268`，2026-10-29 下午）

> 数据轮次：R10（2026-10-29 下午）｜**当前**。全矩阵见 RESULTS §2.9.1。
> **另有按形态的一条轴（2026-10-31）**：把 K 换到 512-55296、线程换到 6-19
> （生产并集）后，1D A=1 从 1t 的 1.11-1.61x 掉到 **0.66-1.10x**，rowcopy
> 仍有 1.3-4.2x、K≤2048 桶化 1.24-1.62x。下表的 2-8x 主要来自 K=2M/10M 与
> 32/38t 端点，**不要当模型内收益**。见 RESULTS §2.10、degradation §6.5。
> **"模型内每次调用要多付 normalize/empty 固定开销"这条（2026-10-31 复测）**：
> 固定链上限是 6.5µs（K=1 全链路）+5.7µs（线程池睡下）+15µs（源冷），K=512 最坏
> 25µs；同形态 × 1/2/3/4/6/8/12/16/19t 全部 1.59-1.64x。174-193µs 的平均只能来自
> 尾部形态（1% rowcopy 就贡献 135µs，混合序列实测预测/实测 = 0.98-1.02）。
> 唯一"O(1)于 K 但很大"的是 mask 的 `nonzero`（1M 元素 1t 5.0ms）。见 RESULTS §2.11、
> degradation §6.5.7。
> **基准口径（2026-10-31 审计）**：直呼 `hpc_index()` 比 `x[idx]` 慢 10µs（偏**悲观**，
> 不是偏乐观）；`--weighted` 必须同时给 `aten/hpc` 与 `index_sve/原生` 两套口径；
> A-br 映射（A=3 vs 64）会让加权结果差 8.7%；覆盖率不足时①③区间会夹住 1.0。
> 见 RESULTS §2.12、degradation §6.5.8。

| 形态 | 1t | 8t | 38t | 备注 |
|---|---|---|---|---|
| S1 / P1 整行 rowcopy C=64 | 2.98x / 2.89x | 3.03x / 3.26x | 4.48x / 4.02x | 收益主体 |
| rowcopy A=4 / A=16 | 1.53x / 2.07x | 1.14x / 1.81x | 8.65x / 3.00x | 轮间 1.5-2.4x（T11 轻载轮曾 3.6x） |
| P3 连续 mask C=16 | 1.77x | 1.23x | 8.62x | T8 单遍压实 |
| `x[:,i]` | 2.00x | 1.41x | 2.85x | T11 修复后保持（旧 0.18x） |
| 1D f64 / f32 K=2M | 1.34x / 1.07x | 1.60x / 1.58x | 2.73x / 4.74x | 4B 受行放大墙 |
| cl4d 行索引（8t 专项） | — | **1.45-2.54x** | — | R10 直写（§2.9.2） |
| M1 权重块（realpath） | 12.7-13.6x | 5.0-5.9x | 1.22-1.47x | 集成收益主体 |
| shortq A=2-4 K=64-512 | 2.55-4.06x | 2.46-4.16x | 2.67-4.05x | µs 级随箱态 |
| M2 (20000,8/64/256) | 1.19-3.40x | 1.62-2.61x | 1.06-4.68x | 全档 ≥1.06x |

当前 <1x（RESULTS §2.9.1，31 格逐格可复算）：

- **空 mask N=1M**：2t 0.97x、4-38t 0.42-0.66x——aten 并行早退恒 ~1.44ms，
  本算子 2.9-3.5ms，**绝对差 1.5-2ms**（§2.8 曾记 ~0.3µs，已更正）；观察项，未修。
- **1B（int8/uint8）** 4-16t 0.91-0.99x、**complex64** 1-16t 0.92-0.95x——
  64B 行放大 / 无 SVE 内核，既有硬墙。
- **非连续 mask** 1-32t 0.79-0.98x——T8 只融合连续 mask（R2 待做）。
- **S4 f64 `x[perm]`** 8/16/38t 0.77-0.92x——每线程工作量 <100KB。
- **噪声级单格 4 格**（均 ≥0.88x）：multi/rowcopy 16t、range2048 16t、mask 1D 4t。

32/36t 的 5-8x 多数是 aten 带宽饱和后自身变慢，不作收益宣传；逐 case 微架构
分析见 `degradation-analysis.md` §5。

---

### 5.2 【历史】逐轮结论（2026-10-22 → 2026-10-29 上午）

#### 【历史·T11】1 线程与 rowcopy

> 数据轮次：T11（`fc26cf8`，2026-10-22）｜历史。

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

#### 【历史·T12】fastindex 差异补齐

> 数据轮次：T12（`6603406`，2026-10-23）｜历史。

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
改善，原因见 `docs/archive/index_sve_vs_fastindex_gap_20261023.md` §19.1。

#### 【历史·T13·a】多索引小 K 与小 K 复测

> 数据轮次：T13（`a91d566`，2026-10-23）｜历史。

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

#### 【历史·T15】小 K 复测

> 数据轮次：T15（`428bafd`，2026-10-26）｜历史。

**T15 小 K 复测（RESULTS §2.3 新表，2026-10-26）**：2D A=16 K=8192 32-38t
**1.26-1.30x**（hpc 73-79µs 砍半到 36-37µs；aten 绝对时间随箱态漂移，以
同轮交替比值为准）、K=65536 32-38t **2.01-2.17x**；K=1024 与 K=1023 全
线程拉平（38t hpc 17.89 vs 17.86µs，T14 轮 38t 曾 1.46x）；1D K<=1024
1.83-2.36x。剩余 <1x 仅 1D 小源家族：K=8192 36/38t 0.95x、K=65536 32-38t
0.90-0.92x——T15-C 判明小源（L1 常驻）下 4B SVE gather 微码吞吐反而低于
标量，保持标量；残留为 `gather_bucket_pure_raw` 前置值域扫描 ~15µs（未来项）。

#### 【历史·T14】真实 workload 对拍

> 数据轮次：T14（`428bafd`，2026-10-26）｜历史；当前 realpath 见 RESULTS §2.9.4。

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

#### 【前一版·2026-10-29 上午】全线程档复测

> 数据轮次：布局修复轮前（2026-10-29 上午）｜前一版；当前见 RESULTS §2.9。

**全线程档复测（RESULTS §2.8，2026-10-29，node5 空闲轻载轮）**：两脚本同日
各一轮，31 场景 + realpath 全部补齐 1/2/4/8/16/32/36/38 八档（此前 S1-P4
只有 7 档、realpath 只有 4 档），单轮自洽；AI4S 全档表格见 summary §6.2。
要点：S1-P4 与 T13 轮各档差 ≤0.3x（T11 的 S1 1t 4.46x 判轻载高点，1t 比值
轮间漂移 ~1.4x）；M1 1t 12.9-13.6x、4t 10.1-11.1x（单格噪声除外）、8t
4.1-5.5x、38t 1.17-1.40x（带宽饱和）；M2 全形态 ≥1.09x；µs 级（M5/A1/
shortq）本轮轻载 1.65-3.4x（§2.5 重载轮 2.6-4.3x——随箱态漂移，跨轮只比
形态）；§2.6 的 32-38t 高值（3.2-6.0x）未复现，与 §2.5 及本轮一致，判共享
箱噪声。当时的两个观察项在 §2.9.1 已复测：mask empty 4-38t 0.51-0.73x
（当时记的绝对差 ~0.3µs 有误，实测 1.5-2ms）、complex64 4-16t 1.68-1.71x
（T13 轮 ~1.0x，R10 复测 0.92-0.95x）。

#### 【历史·T13·b】全量复测与残余 <1x 分类

> 数据轮次：T13（2026-10-23）｜历史；当前 <1x 清单见 §5.1。

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
≈1.1 次/访问，本地 DRAM 97-138ns）。同构建 A/B 实测 SVE 4B 与标量持平
（8MB 1.85 vs 1.86 ms、256MB 7.46 vs 7.45 ms，8/38t），换指令/回退标量无效，
**不是 SVE MLP 缺陷**（RESULTS §3.7；早期 0.75 vs 3.4 GB/s 探针归因已撤回）。
36t 时 4B 行流量 ~95-103GB/s，是本机读带宽（~168GB/s）的 60%——卡在每元素成本
（~19-23ns/元素）而不是节点带宽（degradation §6.4，2026-10-30 复核；旧文档的
73.3GB/s“上限”是计时口径错误），所以倍率再高也改善不了“有用字节/行”。
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
  （SVE 4B 与标量持平，非指令 MLP；RESULTS §3.7）；**当前 <1x 的六个家族**
  见 §5.1（1B 0.91-0.99x、complex 0.92-0.95x、非连续 mask 0.79-0.98x、S4
  0.77-0.92x、空 mask 4-38t 0.42-0.66x、噪声单格）；`x[:,i]` 当前 1t 2.00x
  （T11 前曾 0.4-0.7x）；连续 mask 1t 1.77x、38t 8.62x（后者含 aten 侧退化）；
  fallback 的 Python wrapper 开销 ~45µs/调用对以 RowCopy 小张量为收益主体的
  本算子是不可接受的全局回归。

## 7. 演进方向（后续再看从这里开始）

> 2026-10-27 收尾时的建议排序（优先级代号用 **R**，避免与场景代号
> P1/P2/P3 冲突，代号总表见 `glossary.md`）。R1/R2
> 是最后两个"可修的"劣化家族（其余 <1x 项为硬上限或使用侧问题）；R3 是
> 验证项，其结果决定后续投入是否值得。各项根因推导见
> `degradation-analysis.md` §5.9。

| 优先级 | 项目 | 预期收益 | 工作量 | 详见 |
|---|---|---|---|---|
| **H1 ✅** | **行拷贝软件预取（已实做，T17）**：`rowcopy_raw_single` 与 `rowcopy_sve` 提前 D 行读下标、对行首地址发 `prfm pldl1keep`。D 只按行字节分档（≤64B→16、65-512B→8、>512B→关），运行时旋钮 `HPC_PF_DIST` | **实测 2.26-2.91x**（1t：A=3 79.1→31.9、A=16 83.0→28.5、A=32 94.5→36.7、A=64 108.2→48.0 ns/行；对 aten 从 1.6-2.6x 抬到 2.1-6.0x）。缺行数不变、周期数掉 2.5-3.1 倍；A=256 全档为负故关掉 | 小（只动循环结构） | RESULTS §2.18/§2.19、`headroom.md` §7-H1 |
| **H2** | 元素 gather「算完一批再取」。**分两个子项**（上一轮把两者混为一谈）：**H2a 大值域 1D（DRAM 下界）**——批量化越界检查 + 地址计算，现状 22.3 → 5.65 ns/元素，但 15 个生产形态里没有一例落在这一族，端到端杠杆 ≈0；**H2b 桶化输出段（L1 下界）**——`dst[k]=table[v-mn]` 现在是一条在飞（Sonata 4.53 ns/元素 ≈ L1 延迟 5.07），批量化后按 RESULTS §2.20 的 8KB 曲线可到 0.66-0.71 ns/元素。只对"每线程元素 ≥1 万"的格子有效（Sonata K=256000/6t，占该模型 39% 时间）；对小 K 桶化（SGI/PPT）无效，那两格是固定成本与串行段 | **H2b 单格 4-5x**（e2e Sonata ~1.45x）；H2a 单形态 3.9x 但 e2e ≈0 | 中（批量化 + 整数运算要一起摊） | `headroom.md` §4.2/§5.3/§7-H2、RESULTS §2.20/§2.16.3 |
| **H3** | 两遍模型的输出段：A≥128 在 16/36t 0.80-0.89x，`index_select` 是同形态 2 倍 | 1.5-2x（宽行高线程） | 中 | `headroom.md` §7-H3、§2.16.5、RESULTS §2.19.2（8t 326 指令/行、IPC 2.69 的指令墙） |
| **H4** | 去掉桶化的串行段（串行归约 + 串行建表，19t 占 4.4µs / 每元素 26%）。**小 K 桶化的第一杠杆**：SGI K=13824/12t 每线程只有 1152 元素、28.6 ns/元素，20µs 以上是固定成本与串行段 | 1.2-2x（小 K 桶化），把"高线程反而更差"翻正 | 小 | `headroom.md` §5.3/§7-H4、degradation §6.5.2 |
| **H5** | 遍数与 fork-join 的堵漏：`K<1024` 短路悬崖（同 A、K 减半慢 5.4x）、1D 小源两遍 <1.0x | 不是提速，是堵漏 | 中 | `headroom.md` §5.6/§7-H5、path-matrix §4 |
| **H6** | H1 的尾巴：预取记账在 1t 上多 26 条指令/行（80→105，占 49 周期里的 ~12），改成"每 4 行一批算 4 个地址 + 发 4 条 prfm"；顺手给 `strided_rows_raw`/`rowcopy_fused` 也加预取 | 1.1-1.2x（1 条 line 的行） | 小 | RESULTS §2.19.5、`headroom.md` §5.2/§7-H6 |
| **H7** | 每调用固定成本 ≈0.54ms（`t=F+c·K` 拟合；K≲32768 时把 ns/行 抬 2-3 倍）：输出分配 first-touch（8MB = 2048 缺页）+ 两遍 fork-join。**SGI/PPT 桶化格的第一杠杆**（每线程元素少，摊销最差） | 小 K 场景 1.5-3x | 中 | RESULTS §2.18.5、`headroom.md` §5.3/§7-H7 |
| **H8** | 路由悬崖：`K<1024` 短路排在所有行拷贝之前 → "K 小但 A 大"落到标量直取（cl4d K=512 vs 1024 差 5.2-5.4 倍）。把宽行从短路里摘出来单独判，而不是优化 `gather_direct` | 堵漏：小 K 宽行 5x 级 | 小 | `headroom.md` §5.6/§7-H8、path-matrix §5 |
| **R1** | 1D 小源跳过 `gather_bucket_pure_raw` 前置值域扫描（并入 gather 单遍） | K=8192/65536 小源 36-38t 0.90-0.95x → ≥1x（省 ~15µs / 总 42-53µs，fork-join 占 30-50%）；路径矩阵里对应的就是"两遍模型在 1D 小源 8t 落到 1.0 附近（0.96-0.98）"那两行（RESULTS §2.14.1 的 `r9_dom2049` / `r9_small_i32`），兄弟对照给出翻面差价 1.7-2.0 倍（§2.14.2） | 小 | degradation §5、path-matrix §6.8-C/§5.9.2、§8-2、route-merge §4.1/§4.2 |
| **R2** | 非连续 mask 的 T8 stride 扩展（计数/压实按 mask stride 读，去 nonzero 物化） | 0.86-0.98x → 1.2-1.5x（8t 实测可省 ~1ms） | 中 | degradation §5.8-D、§8-3 |
| **R3** | 端到端 AI4S 模型复测 | 验证 T0-T15 累计收益是否传导到真实负载（决策项） | 中 | performance §3/§4.1 |
| R4 | f64+i32 下标经 `svunpklo` 加宽走 8B gather | ≈持平 → 小幅正收益 | 小 | degradation §8-4 |
| R5 | M1 值域桶化（T2 阶段 2）：`gather_rowcopy_raw` 做下标值域分析 | 高重复下标场景再提升 | 中 | optimizations §3.5 |
| R6 | 4B 大 K 大值域：按 cache line 分桶/排序 | 受 64B 行放大硬上限，只有改访问结构有效 | 大 | degradation §5.9.3 |
| R7 | `small_multi_index` 扩广播/非连续索引（修正 `all_full_contig` + 复用偏移预计算） | 覆盖面，非性能 | 中 | — |
| R8 | dtype 白名单扩展（barebones u16/u32/u64、float8、complex32） | 覆盖面（当前抛 NotImplementedError） | 小-中 | semantics §4 |
| R9 | `HPC_INDEX_COMPILE=1` 真实模型端到端验证 | 决定是否默认开启 | 中 | torch-compile §6 |

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
         tests/test_compile_optin.py tests/test_strided_rows.py \
         tests/test_rowcopy_routes.py tests/test_small_multi.py \
         tests/test_autograd.py tests/test_output_layout.py; do python $t; done

# true-aten 值级对拍（正确性，非性能）
python tests/verify_against_aten.py --mode both
python tests/verify_e2e.py --mode both --threads

# 全线程矩阵（RESULTS §2，pin node 5 静默节点，25 iters interleaved）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
  python benchmarks/bench_threads.py

# 真实 workload 对拍（当前值 RESULTS §2.9.4；§2.5 为 T14 历史轮）
HPC_AFFINITY=190-227 HPC_THREADS=1,8,16,38 HPC_REPS=3 \
  python benchmarks/bench_realpath_driver.py

# 回归扫描（pin node 5 静默节点，避免共享箱噪声）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py
```
