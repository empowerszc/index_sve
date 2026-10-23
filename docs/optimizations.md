# 优化记录：T0-T12 性能优化轮

本文档详细记录针对 AI4S 真实负载的优化工作（T0-T12）。背景：用户实测真实 AI4S
模型端到端**无提升甚至劣化**后，按优化任务建议（T0-T4）逐项实现，每项单独
commit、全量测试 + 前后对比；T5 针对回归扫描暴露的 1 线程劣化做定向修复。
T6-T11 依次修复 index_out/autograd、mask 单遍、小 K 路径、compile opt-in、
线程数方法学；对应性能数据见 `benchmarks/RESULTS.md` §3.8、§2 与
`performance.md` §3-§4。

> **怎么读**：§1-§3 是“为什么改”；§2 是一页 commit 索引；§4 是最终路由优先级
> （改代码前先看）；§5 是汇总数字；§6-§7 是验证与复测；§9 起按轮次（T5-T11）
> 记录问题、改动、实测。**历史性能表（T0-T10）中 2-36t 的标签不可用**：旧扩展
> 用 libomp 默认 team，`torch.set_num_threads` 改的是 libgomp（详见
> `degradation-analysis.md` §3.1）；只有 1t/38t 与 T11 之后的数据有效。

## 1. 背景与目标

`index_sve` 覆盖 `aten::index.Tensor` 后，原有基准（RESULTS §3.1-§3.7）显示 RowCopy
/ 8B gather 形态有 2-4x 收益，但真实 AI4S 模型端到端收益为负。原因是：基准里的
合成形态（大 C 连续行、8B 纯 gather）不是模型里的热路径；模型实际命中：
- **M1** `weight.view(K,C,o)[ki]`（最热，14 模型累计 27270ms）；
- **M2** `feat[r]`（spconv 规则表，~28.7s）；
- **M3** `keys[order]`（int64 小值域查表）；
- **M4** `feat[inverse]`（高重复）；
- **M5/M6** 标量小块 / 2D 索引。

这些形态恰好落在旧实现的最差路径：M1 是"单索引 + A 大"但 K 小（2k-20k），旧
`linearize+gather_index` 的 linearize 解码开销占比高；M3/M4 是 int64 小值域，
线性化后走随机 4B gather（64B 行放大），单线程更慢。

优化总目标：**让新路径 ≥ 旧路径**（`linearize`+SVE 基线），优先 M1+M2 两个最热
形态，且不改平台预期（只用 `svcntb()` 判断 VL）、不动 S5/S6 通用路径、不做无脑 GRAIN 常量
调优。

## 2. 变更清单（对应 commit）

| 任务 | 变更 | 代码 | commit |
|---|---|---|---|
| T0 | 自适应并行粒度 | `index_sve/grain.h` 新增；linearize.cpp / gather.cpp 用 `adaptive_grain` 替换固定 GRAIN=4096 | 472f8a7 |
| T1 | rowcopy 门槛 A>=16 → A>=4 | gather.cpp 分支条件 | 09a406c |
| T2 p1 | M1 大块 skip-linearize | `gather.cpp:gather_rowcopy_raw` / `rowcopy_raw_single`；op.cpp 短路 | 320bcb7 |
| T3 | 融合单遍（scalar 小块） | `gather.cpp:gather_fused`；op.cpp 末支 | db8251b |
| T4 | 小值域桶化 + int32 直写 | `bucket_gather_pure`（gather_index 内）+ int32 linearize 直接写 int32 | 4fd733b |
| T4+ | raw 桶化 + 无锁归约 | `bucket_gather_pure_raw`（op.cpp 短路，单索引 B==1 A==1）；min/max 归约改 per-chunk 数组 | 10921c1 |
| T4++ | OOB 越界检查 | `bucket_gather_pure_raw` 缺小 src + 大 OOB 的边界检查（dom≤2048 曾静默错误）；补 `TORCH_CHECK_INDEX` | 31687d4 |
| T5 | 1 线程回归修复 | `gather_single_1d`（单遍标量）+ fused/linearize nb==1 全长连续快路径 + 索引属性预提取；op.cpp 1 线程路由 | bd21485 |
| T6 | 跨设备校验 + `Tensor_out` 覆盖 | normalize/gather 校验索引与 self 同设备；`TORCH_LIBRARY_IMPL(aten, CPU)` 注册 out 变体 | f791c39 |
| T7 | compile fallback 实验 | 未落地（结论并入了 T10） | — |
| T8 | mask 单遍压实 | `gather_mask`：计数 → 前缀和 → 直接压实，跳过 `nonzero()` | 0dd7533 |
| T9 | 小 K 精简路径 + C++ 覆盖 | `small_single_index`（单整数下标、K<=4096）；C++ `TORCH_LIBRARY_IMPL` 替代 Python 注册 | da9a203 |
| T10 | `torch.compile` opt-in | schema-only + fake + `register_lowering` 钉回运行时；独立 cache 目录；`HPC_INDEX_COMPILE=1` | 1cd77ce |
| T11 | 线程数同步 + 路由/正确性修复 | `sync_omp_threads()`；`gather_direct` A==1 紧凑循环；2B/1B/complex 单遍；1t A>=4 raw rowcopy；非连续 stride 修复 | fc26cf8 |
| T12 | fastindex 差异补齐 | `strided_rows_raw` run 块拷贝；`copy_contig_sve` ≤64B memcpy；A<16 raw；B==1 快路径；grain 下限 64；多索引 A>=4 raw（≥16t） | 本轮 |

## 3. 逐项详解

### 3.1 T0 自适应并行粒度（472f8a7）

**问题**：固定 `GRAIN=4096` 使小 K 场景（M1 E=2k，B*K=54k）在 8+ 线程时每个
chunk 只有 6 个元素，parallel_for 线程调度开销超过计算本身；大 K 又可能因
grain 过小生成过多 chunk。

**实现**（`grain.h`）：

```cpp
inline int64_t adaptive_grain(int64_t n) {
  constexpr int64_t MAX_GRAIN = 65536;
  const int64_t nt = at::get_num_threads();
  if (nt <= 1) return std::max<int64_t>(1, n);           // 单线程整块
  int64_t grain = std::max<int64_t>(1, n / (nt * 8));    // 每线程 ~8 chunk
  return std::min<int64_t>(grain, MAX_GRAIN);            // 上限防 chunk 过粗
}
```

`linearize`、`gather_index` 全部内核、`gather_fused`、`gather_direct`、
`rowcopy_raw_single`、桶化内核均改用 `adaptive_grain(B*K)`。

**验证**：全线程扫描（1/2/4/8/16/32/36）M1/M2/M3/M4 形态均无退化，1 线程整块
保证 M1 这类小 K 形态的收益不被 parallel_for 开销吃掉。

### 3.2 T1 rowcopy 门槛 A>=4（09a406c）

**问题**：旧门槛 `A>=16` 才启用 SVE rowcopy；M2 spconv 的 `feat[r]` 若
C=8（A=8）则落 `dispatch_scalar`，与 aten 无差甚至更慢。

**实现**：门槛放宽到 `A>=4`，分支顺序改为 `A==1 纯 gather / A>=4 rowcopy /
else scalar`。A-sweep 实测 A=4-16 短行在 1 线程即有 1.1-17x 收益，A>=16 长行
路径不变。

**代价**：A∈[2,3] 仍走标量（太短，向量化不值得）。`scalar_gather` 覆盖范围
从 A∈[2,15] 收窄到 A∈[2,3]。

### 3.3 T2 p1 M1 大块 skip-linearize（320bcb7）

**问题**：M1 `weight.view(K,C,o)[ki]` 是**单索引 + after_contiguous + A 大
（4096）** 形态，但旧实现先 `linearize`（对 K 个位置做 broadcast 解码 + 写
K×8B/4B idx 数组）再 `gather_index`。linearize 的 K 遍解码对 A 大形态纯属浪费
——rowcopy 其实只需要原始下标值。

**实现**：`gather_rowcopy_raw(spec, out, nt)` 直接遍历 `spec.indices[0]` 原始
下标（int32/int64），内联边界检查 + 负索引归一，每行 `copy_contig_sve` 拷 A
连续块。op.cpp 短路条件：`indices.size()==1 && after_contiguous && A>=4096 &&
dtype∈{f32,f64}`。

**实测**（pin node 5，interleaved median）：M1 (27,32,32) E=2k 1 线程 **9.1x**、
M1 (27,64,64) E=20k 1 线程 **9.9x**（旧路径该形态是"无收益"）。8 线程 i64
0.6-1.0x（box 噪声），i32 2.7x。

### 3.4 T3 融合单遍（db8251b）

**问题**：scalar 小块（A<4 或非连续）旧实现走 `linearize + dispatch_scalar`，
两次遍历 idx。对 P2 pos[src] A=3 这类形态，linearize 的开销占比大。

**实现**：`gather_fused` 在并行循环内先算偏移（复用 `gather_direct` 的 bfactor
broadcast 解码 + 内联检查），再就地标量拷贝，**不物化 idx 数组**。op.cpp 末支
条件：`A<4 或非连续小块`（即不满足 A==1 也不满足 A>=4 连续 rowcopy 的输入）。

**A/B 决策**：`linearize+rowcopy` 与 `gather_fused` 在 A>=4 连续块上对比——fused
单线程更慢（C=8: 0.69x vs 1.05x；C=64: 1.50x vs 3.85x），故 **fused 只用于 scalar
小块路径**，A>=4 连续块仍走 linearize+rowcopy。这与 T2 互斥（T2 只处理单索引
大块）。

### 3.5 T4 小值域桶化 + int32 直写（4fd733b）

**问题**：M3 `keys[order]` 是 int64 小值域纯 gather（2048 个可能值，E=54k），
旧实现 0.2x——linearize 的解码 + 随机 gather 的 cache-line 放大双重拖累。另有
遗留项：`linearize` 的 int32 分支先建 K×8B int64 tmp 再转换（K=1M 时 24.6ms
/ 27.7ms 全 op，占 89%）。

**实现**（两个独立修复）：

1. **int32 linearize 直接写 int32**：`linearize` 的 int32 分支在循环内直接填
   `int32_t`（`use_i32` 保证偏移 < 2^31 值域安全），删掉 K×8B int64 中间缓冲。
2. **`bucket_gather_pure`**（gather_index 内、SVE 分支之前）：先并行扫一遍
   linearize 后的 idx 求偏移 min/max，值域 ≤2048 时构建 L1/L2 常驻查找表
   （4B 表 8KB / 8B 表 16KB），标量循环 `dst[p] = table[offset - mn]`。

**动机**：小值域时表查的标量随机读命中 L1/L2，避免 64B 行放大，远优于大值域
随机 gather（T11 复核：大值域 SVE 4B 与标量持平，差异来自访存模式而非指令；
RESULTS §3.7）。值域 >2048 时返回 false，落回 SVE/scalar。

**实测**：1D f32/f64 K=1M 小值域 8-32 线程 1.2-3.1x（SVE 0.2-0.9x）。单线程
仍受 linearize 解码限制，但不再比旧路径差。

### 3.6 T4+ raw 桶化 + 无锁归约（10921c1）

**问题**：上一版 `bucket_gather_pure` 仍依赖 `linearize`（M3 i64 单线程
linearize=1.24ms / full=1.69ms）。对 **B==1 单索引 1D 纯 gather**（M3/M4
形态），偏移就是 `idx[k]` 本身，linearize 的 broadcast 解码纯属浪费。

**实现**：

1. **`bucket_gather_pure_raw`**（op.cpp 短路，条件 `单索引 && B==1 && A==1`）：
   直接对 `spec.indices[0]` 原始下标求值域（内联负索引归一），值域 ≤2048 时
   建表 + 标量填充，完全跳过 linearize。
2. **归约去锁**：两处桶化的 min/max 归约从 `std::mutex` 保护改为 **per-chunk
   局部数组 + 串行合并**（`vmn/vmx[lo/grain]`），消除多 chunk 间锁竞争；值域
   上限从 256 提到 **2048**（8B 表 16KB 仍 L1/L2 常驻）。

**实测**：M3 keys[order] int64[2048] E=54k 从 0.2x → **1.4x（i64）/ 2.6x（i32）
单线程**、**8.0x/11.6x 32 线程**；M4 feat[inverse] (200,64) E=20k 3.6-4.8x。

## 4. 路由优先级总览（优化后）

```
index_impl
  0. 单 mask && 覆盖前 m 维 && mask/self 连续 && CPU → gather_mask（单遍 compaction，T8）
  0b. 单整数下标 && 末维 K<=4096 && mask/self 连续 && CPU → small_single_index（绕 normalize，T9）
  1. K < 1024                     → gather_direct（无 linearize）
  2. 单索引 && after_contig && A>1 && (A>=4096||A<16) && f32/f64
                                     → M1 raw rowcopy（skip-linearize）
  2b. 单索引 && !after_contig && col_off 有 ≥4 run
                                     → strided rows（run 块拷贝）
  2c. 多索引 && after_contig && A>=4 && 线程数>=16
                                     → 多索引 raw rowcopy（skip-linearize）
  3. 单索引 && B==1 && A==1       → T4 raw 桶化（skip-linearize）
  4. A==1 || (after_contig && A>=4 && f32/f64)      → linearize + gather_index
       └─ gather_index：先试 bucket（A==1 值域≤2048）→ SVE 8B/4B → rowcopy → scalar
  5. 其余（A<4 / 非连续小块 / run=1）→ T3 gather_fused（不物化 idx）
```

优先级逻辑：**先试不物化 idx 的短路路径**（direct / raw rowcopy / raw 桶化），
需要 idx 时才 linearize；fused 只服务 A<4 或非连续小块。

## 5. 性能结果汇总（pin node 5，interleaved median，RESULTS §3.8）

> **历史数据警告**：下表 8t/32t 列产生于 T8 之前，扩展内 `at::parallel_for`
> 经 libomp 默认 team 运行，线程标签无效（真实 38 worker）；只有 1t 与 38t
> 可信。P3 的 8t/32t 也受影响。T11 修正后的 31 场景矩阵见 RESULTS §2，
> 逐案解释见 `degradation-analysis.md` §4-§5。

| 形态 | 1t | 8t | 32t | 优化前 |
|---|---|---|---|---|
| M1 (27,32,32) E=2k | **9.1x** | 1.3x | 3.9x | ~无收益 |
| M1 (27,64,64) E=20k i64 | **9.9x** | 0.6-1.0x | 1.0x | ~无收益 |
| M1 (27,64,64) E=20k i32 | 7.9x | **2.7x** | 0.95x | ~无收益 |
| M2 feat[r] (20000,64) E=100k | 3.9x | 2.6x | 0.6-7.1x | ~1x |
| M2 C=8（A=8 短行） | 0.93x | 0.95x | 0.92x | ~1x |
| M3 keys[order] E=54k i64 | 1.4x | 0.6-1.3x | **8.0x** | **0.2x** |
| M3 keys[order] E=54k i32 | 2.6x | 1.3-2.2x | **11.6x** | **0.2x** |
| M4 feat[inverse] E=20k | 3.6x | 3.8x | **4.8x** | ~1x |
| P3 x[mask] C=16 (T8，8t/32t 标签失效) | **2.1x** | ~~8.0x~~ | ~~7.2x~~ | 1.1-1.5x |

注：8t/32t 共享箱噪声大（aten 单格曾摆动 104→426us）；可靠数字是交错
aten/hpc 的 median。4B f32 K=10M 大值域随机 gather 仍是硬件上限（1t 0.7x、
高线程 0.2x），T4 桶化因值域 >>2048 不启用。

## 6. 验证

- 当前全 10 个测试套件 **102/102 通过**（T5 时点为 6 套件 75/75）。
- 正确性对拍：两进程法（无覆盖进程生成 true-aten 参考，覆盖进程比对），
  工具 `tests/verify_against_aten.py --mode both`（**44/44**），含 K>=1024、
  桶化、M1 raw、T5 单遍、多索引、mask、非连续 stride 用例。
- 桶化路径正确性由 test_correctness 的小值域用例覆盖（`randint(0,64/100,...)`）。

## 7. 复测

```bash
# M1-M6 workload 形态全线程扫描（pin node 5 静默节点，50 iters interleaved）
HPC_INDEX_NO_OVERRIDE=1 HPC_THREADS=1,2,4,8,16,32,36 python benchmarks/bench_ai4s.py

# T1 A 扫描
HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_t1.py
# T2 M1 扫描
HPC_INDEX_NO_OVERRIDE=1 python benchmarks/bench_t2.py

# 全量正确性
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py; do python $t; done
```

## 8. 后续可优化项（未做）

- **M1 值域桶化（T2 阶段 2）**：M1 raw rowcopy 路径未对下标做值域分析；若
  keys[order] 等值域小可再桶化。`gather_rowcopy_raw` 已留好接入入口。
- ~~**P3 mask 多线程**~~：T8 已用单遍 compaction 修复（见 §12），不再列入。
- **4B 大 K 随机 gather**：64B 行放大 + DRAM 延迟（SVE 4B 与标量持平，RESULTS
  §8.4）；可试软件预取 / 按 cache line 分桶排序。

## 9. T5：1 线程回归修复（线性化两遍模型 vs aten 单遍）

> **T11 复核修正**：本节 2/8/32t 列同样受 libomp/libgomp 双 ICV 影响（实际用
> 38 worker），相对值偏大；且 T5-2 的"1 线程 A>1 全走 fused"让 A=64 的 S1/P1
> 丢掉了 SVE rowcopy（1t 只有 1.06x）。T11 起 1 线程 A>=4 连续改回 raw rowcopy
> （S1 1t 4.46x、rowcopy A=4 1t 2.41x）。修正矩阵见 RESULTS §2。

### 背景

回归扫描（`benchmarks/bench_regression.py`，interleaved median，pin node 5）发现
1 线程下多形态劣化（0.6-0.9x），2+ 线程全部占优（rowcopy A=4 2t 7x、1D pure 2t
15-20x、x[i,j] 2t 5.98x）。1 线程劣化根因（1D f32 K=5M 分解）：

- linearize 两遍模型固定开销：`linearize` 104ms + `gather` 124ms = 228ms vs
  aten 单遍 169ms；linearize 对 1 线程无并行加速，纯属多余的一遍。
- 单线程随机 gather 的 DRAM 延迟（T11 复核：SVE 4B 与标量持平），换
  SVE→scalar 无济于事（dispatch_scalar 0.64x 与 SVE 0.71-0.79x 相当）。
- 融合单遍（`gather_fused`）每元素做 at::Tensor 方法访问（element_size/stride/
  dtype/data_ptr，引用计数 + 虚调用），5M 元素下反而更慢（0.22-0.53x）。

### 变更

| 项 | 变更 | 代码 | 1 线程效果 |
|---|---|---|---|
| T5-1 | 单索引 A==1 大值域 1 线程走 `gather_single_1d`（单遍标量，跳过 linearize） | gather.cpp 新增内核 + op.cpp 路由 | 1D f32 0.75x→**1.07x**、f64 0.71x→**1.24x** |
| T5-2 | 单索引短行 A<4 1 线程走 `gather_fused` 单遍（T11 起 A>=4 改 raw rowcopy） | op.cpp 路由 | rowcopy A=4 0.87x→1.39x（T11 2.41x） |
| T5-3 | fused/linearize 预提取索引张量属性 + nb==1 全长连续快路径（直读 idx_j[k]） | gather.cpp `gather_fused`、linearize.cpp 两变体 | x[i,j] 0.66x→**1.21x**（linearize 4.84→1.10ms）；T11 修正为 1.19x |
| T5-4 | 修正路由范围 | op.cpp（多索引 A==1 与 mask 保持 linearize+SVE，fused 实测更差） | 避免 blanket fused 误伤（x[i,j] 曾 0.49x、x[:,i] 0.28x） |

### 路由现状（op.cpp index_impl）

```
1. K < 1024                                  → gather_direct
2. 单索引 && after_contig && A>=4096 && f32/f64 → M1 raw rowcopy
3. 单索引 && B==1 && A==1                    → T4 raw 桶化
4. 1 线程 && 单索引 && A>1                   → A>=4 连续 ? raw rowcopy : T3 gather_fused
5. A==1 || (after_contig && A>=4 && f32/f64) → linearize + gather_index（含 SVE/桶化）
6. 其余                                    → T3 gather_fused
```

### 实测（pin node 5，interleaved median）

| 形态 | 1t 修复前 | 1t 修复后 | 2t | 8t | 32t |
|---|---|---|---|---|---|
| 1D pure K=5M f32 | 0.75x | **1.07-1.28x** | 33.6x | 11.9x | 5.1x |
| 1D pure K=5M f64 | 0.71x | **1.23-1.24x** | 20.8x | — | — |
| x[i,j] K=100k | 0.66x | **1.21x** | 5.98x | 3.02x | 1.13x |
| rowcopy A=4 K=200k | 0.86x | **1.39x** | 9.28x | 8.01x | 8.30x |
| mask（广播） | ~1x | ~0.97-1.13x（噪声级） | 1.20x | 1.23x | 0.97-1.13x |

mask 1 线程与高线程在 0.97-1.13x 间波动（不同密度 0.05/0.3/0.5），属噪声级，
不做 fallback（integration.md §7 论证 fallback 自身有开销）。

### 验证

T5 时点 6 套件 **75/75 通过**（新增 6 个：多索引全长/广播标量/非连续、1 线程
rowcopy 短行、1 线程大值域纯 gather、linearize 快/慢路径一致性）；当前为
10 套件 **102/102**。

## 10. T6：跨设备索引校验 + `aten::index.Tensor_out` 覆盖（正确性/覆盖）

### 问题

- **跨设备索引段错误**：索引张量不在 CPU（如 `device='meta'`）时 `data_ptr()`
  返回空指针，内核解引用直接 SIGSEGV（实测 core dump）。`normalize.cpp` 早期
  简化掉了 aten 的 `checkIndicesOnCpuOrSelfDevice`。
- **`out=` 变体未加速**：`aten::index.Tensor_out` 是独立 structured op，直接调
  `index_stub`，不经过 `aten::index.Tensor`。只覆盖 functional 时 `out=` 调用
  100% 走真 aten（实测 rowcopy A=64 K=200k：真 aten ≈5.7ms vs 本算子 ≈0.58ms，
  约 10x 收益丢失）。

### 变更

| 项 | 变更 | 代码 |
|---|---|---|
| T6-1 | 入口复刻 `checkIndicesOnCpuOrSelfDevice`：非 CPU 索引抛 aten 同文错误（`indices should be either on cpu or on the same device as the indexed tensor (cpu)`），不再段错误。T10 后 `hpc_index::*` 改为 schema-only + CPU kernel，meta 输入由 PT2 报 "no Meta kernel"（干净错误，不崩溃）；`HPC_INDEX_COMPILE=1` 时注册 fake 后可正常 tracing | normalize.cpp / op.cpp |
| T6-2 | 抽出 `index_into(self, spec, out)` 核心路由；新增 `index_out_impl` 写入调用方 out（形状不符 `resize_`；非连续 out 先算到临时连续张量再 `copy_`；校验 dtype/device/overlap） | op.cpp |
| T6-3 | 注册 `hpc_index::index_out` 并覆盖 `aten::index.Tensor_out` CPU key；`HPC_INDEX_NO_OVERRIDE=1` 仍可整体关闭，基准对拍语义不变 | op.cpp / register_aten.py |

### 验证

- 新增 `tests/test_index_out.py` 7 个用例（pure / rowcopy / mask / multi-index /
  非连续 out / dtype 不匹配 / profiler 断言内核名 `hpc_index::index_out`）；
  `test_errors.py` 增加 meta 索引子进程不崩溃用例（修复前 rc=-11）。
- T6 时点 7 套件 **83/83**；true-aten 值级对拍当时 40/40（当前 44/44）；
  autograd / vmap / jvp /
  index_put / setitem 行为不变；`HPC_INDEX_NO_OVERRIDE=1` 下 out= 仍为真 aten。

## 11. T7（实验，未落地）：`torch.compile` fallback

`torch.compile` 默认由 Inductor 内联生成 `aten::index` 的 kernel，绕过 dispatcher
覆盖（实测慢 2.6x）。用 `force_fallback(aten.index.Tensor)` 可让它回落到运行时
调用、重新命中本算子，实测 2.6x → 1.07-1.3x。含 inductor 缓存、私有 API、
FakeTensor 等坑与 opt-in 落地，见 [torch-compile.md](torch-compile.md)；
实测数据见 `benchmarks/RESULTS.md` §11。已在 T10 以 `HPC_INDEX_COMPILE=1`
opt-in 实现（§14）。

## 12. T8：mask 单遍融合（跳过 nonzero）

### 问题

旧路径 `x[mask]`：`normalize_index` 先 `mask.nonzero()` 物化 int64 位置下标，
再 linearize + gather。两段完整遍历，且 nonzero 输出长度数据相关、并行度差，P3
在高线程只有 1.0-1.9x（§5 注）。

### 变更

- `op.cpp` 入口新增 `mask_fast_path`：仅当 `indices` 恰为一个 bool/byte mask、
  mask 覆盖 self 前 `md` 维且形状逐维匹配、mask 与 self 均连续、均在 CPU 时启用；
  形状错误仍抛 aten 同文错误，byte mask 保留 deprecation 警告。
- `gather.cpp` 新增 `gather_mask`：chunk 计数（n<1MB 串行，避免并行任务开销）→
  前缀和 → 第二遍按 chunk 直接压实并 `copy_contig_sve` 整行拷贝（f32/f64），其余
  dtype 标量行拷贝；`nnz==0` / `A==0` 直接返回空输出。
- 不满足条件的 mask（非连续、跨设备、混合其它下标）仍走原路径；
  `index.Tensor_out` + mask 也仍走原路径（out 变体罕见）。

### 实测（P3 N=200k C=16 d=0.3 nnz≈60k，pin node 5，interleaved median）

| 线程 | 旧 hpc | 新 hpc | 新/旧 | aten/新 |
|---|---|---|---|---|
| 1 | 4.86 ms | 2.19 ms | 2.2x | 2.1x |
| 2 | 4.19 | 0.77 | 5.4x | 6.0x |
| 4 | 2.87 | 0.70 | 4.1x | 6.8x |
| 8 | 2.66 | 0.73 | 3.6x | 8.0x |
| 16 | 2.69 | 0.74 | 3.6x | 7.3x |
| 32 | 3.66 | 0.70 | 5.2x | 7.2x |
| 38 | — | 0.66 | — | 3.1-4.6x |

其它形态（8t，aten/新）：C=64 d=0.1 1.9-2.1x、1D A=1 1M 1.1-2.2x、全维 mask
512² 1.6-1.8x、2D mask→8 2.1-2.5x、byte 1.5-1.7x、空 mask 0.9x。共享箱上 aten
绝对时间波动大（同一 P3 8t 曾 1.1-5.9ms），hpc 侧 0.66-0.77ms 稳定。

### 验证

- 新增 `tests/test_mask_fused.py` 9 用例：1D/2D/full-dim/空/byte 掩码、
  f32/f64/int64、1/2/4/8 线程、非连续 self/mask 回退；参考值用
  `mask.nonzero()` + `index_select`（未被覆盖）。
- 全 8 套件 **92/92**；`mask_fused_count()` 调试计数断言融合路径确实命中。

## 13. T9：小 K 单整数下标精简路径 + C++ 直接覆盖

### 问题

- `x[idx]`（单整数下标、末维 K 小）固定开销高：旧路径先 `to_vec` + `normalize_index`
  （多次 vector 分配/transpose 分析），再 `gather_direct` 内层反复调用 `at::Tensor`
  方法。1D `K=1023` 1t 51.6us vs aten 24.7us（0.48x）。
- 真实调用还多一层 **Python 覆盖**：`register_aten.py` 用
  `torch.library.impl` 注册，`x[idx]` 每次往返一次 Python 解释器（~10-20us），
  小 K 被完全吃掉。
- `K=1024` 存在断崖：`K<=1023` 走 `gather_direct`，`K>=1024` 落入
  `gather_bucket_pure_raw` + `linearize` + `gather_bucket_pure` 的多次并行扫描
  （即使值域过大最终仍走 SVE），固定 ~120us；`K=1024` 8t 仅 0.37x。

### 变更

| 项 | 变更 | 代码 |
|---|---|---|
| T9-1 | 新增 `small_single_index`：单整数下标 + 末维 `K<=4096` + self/idx 连续 + CPU 时，直接按 idx 平坦顺序取行；绕过 `normalize_index`，A>=4 用 `memcpy`，n<16384 串行避免并行任务开销 | op.cpp |
| T9-2 | 覆盖注册从 Python `torch.library.impl` 移到 C++ `TORCH_LIBRARY_IMPL(aten, CPU, m)`；签名对齐 schema（`const List&`、out 返回 `Tensor&`）；`register_aten.install()` 变空实现 | op.cpp / register_aten.py |
| T9-3 | 阈值覆盖 `K=1024..4096`，消除断崖；`K>4096` 仍走原路径（SVE/桶化在更大 K 更优） | op.cpp |

### 实测（真实 `x[idx]`，pin node 5，3 次交替进程中位数）

| 形态 | 1t/8t（T9 后） |
|---|---|
| 1D A=1 K=0 | **2.3x** vs aten |
| 1D A=1 K=1 | **2.2x** |
| 1D A=1 K=64 | **2.2x** |
| 1D A=1 K=512 | **2.0x** |
| 1D A=1 K=1023 | **1.8x** |
| 1D A=1 K=1024 | **1.6x** |
| 2D A=16 K=128 | **2.4x** |
| 2D A=16 K=1023 | **2.4-2.7x** |
| 2D A=16 K=1024 | **1.4x** |

断崖修复（8t，源 1M，idx 随机大值域）：`K=1024` 0.37x→**1.64x**、`K=2048`
0.47x→**1.38x**、`K=4096` 0.55x→**0.99x**；`K=8192..16384` 仍走原路径
（0.9-1.2x），`K=65536` 3x。C++ 注册自身把真实 `x[idx]` 的 K=0 从 27.5us 降到
11.9us、K=1023 从 36.9us 降到 24.4us（同脚本前后对比）。

### 验证

- 新增 `tests/test_smallk.py` 7 用例：1D/2D/N-D/0-dim/空下标、f32/f64/int64、
  1/8 线程、非连续 self/idx 回退、越界错误逐字、路由计数（K<=4096 命中、
  多索引/mask 不命中）。
- T9 时点 9 套件 **99/99**；true-aten 对拍当时 40/40（当前 44/44）；
  `HPC_INDEX_NO_OVERRIDE=1` 仍为真 aten。
- 覆盖注册改动后 `out=` / autograd / vmap 行为不变（test_index_out + 全量回归）。

## 14. T10：`torch.compile` opt-in（HPC_INDEX_COMPILE=1）

### 问题

默认 compile 由 Inductor 内联生成第三份实现，完全绕过 CPU 覆盖（慢 2.6x，见 §11）。
需要一条"显式开启就让 compile 用回本算子"的路径，同时不改变默认行为与用户的
编译缓存。

### 变更

| 项 | 变更 | 代码 |
|---|---|---|
| T10-1 | `hpc_index::index` / `index_out` 改为 schema-only + CPU kernel，允许注册 fake/Meta（CompositeImplicitAutograd 会拒绝） | op.cpp |
| T10-2 | 新增 `compile_support.py`：`register_fake`（整数下标推形状；mask 数据相关，显式 `NotImplementedError`）、`register_lowering(aten.index.Tensor)(fallback_handler(...))`（带 API 版本守卫）、cache 目录切到 `<base>/hpc_index`（幂等） | compile_support.py |
| T10-3 | `HPC_INDEX_COMPILE=1` 在 `import index_sve` 时启用；与 `HPC_INDEX_NO_OVERRIDE=1` 冲突直接报错 | `__init__.py` |

### 实测

编译产物不再出现 `cpp_fused_index_*`，profiler 为 `aten::index`，运行时命中 CPU
覆盖；整数下标与 mask 两条路径的 compile 输出均与 eager 逐元素相等。收益见
`benchmarks/RESULTS.md` §11（0.3ms 的 compile 固定开销，K≲100k 不建议 compile）。

### 验证

- 新增 `tests/test_compile_optin.py` 3 例（子进程隔离）：默认关闭；fake 推形状 +
  整数/mask compile 数值等价 + profiler + cache 目录；与 `HPC_INDEX_NO_OVERRIDE`
  冲突报错。
- 10 套件 **102/102**；true-aten 对拍 **44/44**；`HPC_INDEX_NO_OVERRIDE=1` 仍为真 aten。

## 15. T11：线程数方法学修正 + 路由/正确性修复

### 问题

1. **线程数失真**：扩展链接 BiSheng libomp，`torch.set_num_threads(t)` 只改
   ATen 的 libgomp ICV；2-36t 实际都用 libomp 默认 team（本机 38），旧矩阵的
   低线程相对值被严重高估。
2. **1 线程 A>=4 误伤**：T5 的 "1t 单索引 A>1 全走 `gather_fused`" 让 S1/P1
   （A=64）在 1t 丢掉 SVE rowcopy，只有 1.06x。
3. **`x[:,i]` 每元素开销**：`gather_direct` 对 A==1 仍做逐元素边界检查 +
   `col_off` 内层循环。
4. **2B/1B/complex**：无 SVE 内核却仍走 linearize 两遍。
5. **非连续 stride 正确性**：`bucket_gather_pure_raw`/`gather_single_1d`/
   `rowcopy_raw` 平坦读 idx，非连续下标会算错（旧 "1.37x" 是错误路径的假加速）。

### 变更

| 项 | 变更 | 代码 |
|---|---|---|
| T11-1 | 入口 `sync_omp_threads()`：`omp_set_num_threads(at::get_num_threads())`，`torch.set_num_threads` 对扩展生效 | op.cpp |
| T11-2 | `gather_direct` 属性 hoist + A==1 预校验紧凑循环 | gather.cpp |
| T11-3 | 1t 单索引 A>=4 连续 f32/f64 → raw rowcopy；A<4 → fused | op.cpp |
| T11-4 | 无 SVE 内核 dtype 在 `fast_1d` 下所有线程走单遍标量 | op.cpp |
| T11-5 | raw 路径加 `stride0` + 连续性门控；M1/1t raw rowcopy 增加 `indices[0].is_contiguous()` | gather.cpp / op.cpp |
| T11-6 | `small_single_index` 串行阈值改用 `n*A`（2D A=16 K=4096 不再误判串行） | op.cpp |

### 实测（RESULTS §2）

| 形态 | 修复前 | T11 |
|---|---|---|
| S1 1t | 1.06x | **4.46x** |
| P1 1t | 1.06x | **4.42x** |
| rowcopy A=4 / A=16 1t | 1.35x / 1.70x | **2.41x / 3.58x** |
| `x[:,i]` 1t | 0.18→0.40x | **2.02x** |
| half/bf16 1-36t | 0.5-0.97x | **1.2-2.4x** |
| 2D A=16 K=4096 32/36t | 0.53-0.55x | **1.03-1.05x** |

仍 <1x：非连续 mask 0.84-0.98x、int8/uint8/complex 0.9-1.0x、小工作量
（S4 8/16t、小 K 高线程）0.46-0.96x。逐 case 分析见
`docs/degradation-analysis.md`。

### 验证

- 全 10 套件 **102/102**；两进程 true-aten 对拍 **44/44**（新增 4 个非连续
  stride 用例）。
- 线程矩阵 31 场景 × 1/2/4/8/16/32/36；小 K 两进程 8 个 K × 7 线程。


## 16. T12：fastindex 差异补齐（本轮）

### 变更

| 项 | 变更 | 代码 |
|---|---|---|
| T12-1 | 非连续尾维按 `col_off` 的连续 run 块拷贝；run≥4 才接管 | `gather.cpp: make_col_runs/strided_rows_raw` |
| T12-2 | ≤64B 行/run 用 `memcpy`，避免 masked SVE 谓词开销 | `gather.cpp: copy_contig_sve` |
| T12-3 | 单索引 A<16 多线程也 raw；`B==1` 去 `p/K`、`p%K` | `op.cpp` / `gather.cpp: rowcopy_raw_single` |
| T12-4 | `adaptive_grain` 下限 1→64 | `grain.h` |
| T12-5 | 多索引 A≥4 且 ≥16t 时 raw rowcopy | `gather.cpp: rowcopy_fused_sve` |
| T12-6 | 新增路由/正确性测试 9 例 | `tests/test_strided_rows.py`、`test_rowcopy_routes.py` |

### 实测（node 5，同进程交替）

| 形态 | 旧 | 新 | 提升 |
|---|---|---|---|
| run16 A1024 1t | 10.56ms | 4.74ms | **2.23x** |
| run16 A1024 8t | 1.59ms | 0.934ms | 1.71x |
| run10 A640 1t | 4.71ms | 3.50ms | 1.36x |
| A=8 多线程 | — | — | 1.21-1.37x |
| A=15 多线程 | — | — | 1.15-1.37x |
| 多索引 A=64 16/36t | — | — | 1.12/1.16x |
| 1D K=8192 36t（相对 1t） | 0.76x | 0.98x | 小任务高线程不再负扩展 |

放弃项：纯转置 run=1（无连续后缀，块拷贝不可用）、多索引 1-8t（raw 慢
9-23%）、`_offset`（0-2.3%）、mask int32 转换、slice 提前回退。系统原因见
`index_sve_vs_fastindex_gap_20261023.md` §19。

### 验证

- 12 个测试文件 **111/111**；两进程 true-aten 对拍 **44/44**。
- A/B 数据与风险见新文档 Part III。
