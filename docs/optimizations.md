# 优化记录：T0-T5 性能优化轮

本文档详细记录针对 AI4S 真实负载的优化工作（T0-T5）。背景：用户实测真实 AI4S
模型端到端**无提升甚至劣化**后，按优化任务建议（T0-T4）逐项实现，每项单独
commit、全量测试 + 前后对比；T5 针对回归扫描暴露的 1 线程劣化做定向修复。
对应性能数据见 `benchmarks/RESULTS.md` §9 与 `performance.md` §4.1。

## 1. 背景与目标

`index_sve` 覆盖 `aten::index.Tensor` 后，原有基准（RESULTS §1-§8）显示 RowCopy
/ 8B gather 形态有 2-4x 收益，但真实 AI4S 模型端到端收益为负。原因是：基准里的
合成形态（大 C 连续行、8B 纯 gather）不是模型里的热路径；模型实际命中：
- **M1** `weight.view(K,C,o)[ki]`（最热，14 模型累计 27270ms）；
- **M2** `feat[r]`（spconv 规则表，~28.7s）；
- **M3** `keys[order]`（int64 小值域查表）；
- **M4** `feat[inverse]`（高重复）；
- **M5/M6** 标量小块 / 2D 索引。

这些形态恰好落在旧实现的最差路径：M1 是"单索引 + A 大"但 K 小（2k-20k），旧
`linearize+gather_index` 的 linearize 解码开销占比高；M3/M4 是 int64 小值域，
线性化后走 low-MLP 的 SVE gather（4B 指令），单线程更慢。

优化总目标：**让新路径 ≥ 旧路径**（`linearize`+SVE 基线），优先 M1+M2 两个最热
形态，且不改硬件探测（`sve_512_ok`）、不动 S5/S6 通用路径、不做无脑 GRAIN 常量
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
旧实现 0.2x——linearize 的解码 + 4B/8B SVE gather 的 low-MLP 双重拖累。另有
遗留项：`linearize` 的 int32 分支先建 K×8B int64 tmp 再转换（K=1M 时 24.6ms
/ 27.7ms 全 op，占 89%）。

**实现**（两个独立修复）：

1. **int32 linearize 直接写 int32**：`linearize` 的 int32 分支在循环内直接填
   `int32_t`（`use_i32` 保证偏移 < 2^31 值域安全），删掉 K×8B int64 中间缓冲。
2. **`bucket_gather_pure`**（gather_index 内、SVE 分支之前）：先并行扫一遍
   linearize 后的 idx 求偏移 min/max，值域 ≤2048 时构建 L1/L2 常驻查找表
   （4B 表 8KB / 8B 表 16KB），标量循环 `dst[p] = table[offset - mn]`。

**动机**：小值域时表查的标量随机读命中缓存，优于 920F 低 MLP 的 SVE gather
指令（RESULTS §8.2：SVE 4B gather 38 线程仅 0.75 GB/s vs 标量 3.4 GB/s）。
值域 >2048 时返回 false，落回 SVE/scalar（大值域随机 gather 仍是硬件上限）。

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
  1. K < 1024                     → gather_direct（无 linearize）
  2. 单索引 && after_contig && A>=4096 && f32/f64   → M1 raw rowcopy（skip-linearize）
  3. 单索引 && B==1 && A==1       → T4 raw 桶化（skip-linearize）
  4. A==1 || (after_contig && A>=4 && f32/f64)      → linearize + gather_index
       └─ gather_index：先试 bucket（A==1 值域≤2048）→ SVE 8B/4B → rowcopy → scalar
  5. 其余（A<4 或非连续小块）    → T3 gather_fused（不物化 idx）
```

优先级逻辑：**先试不物化 idx 的短路路径**（direct / raw rowcopy / raw 桶化），
需要 idx 时才 linearize；fused 只服务 A<4 或非连续小块。

## 5. 性能结果汇总（pin node 5，interleaved median，RESULTS §9.3）

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
| P3 x[mask] C=16 (T8) | **2.1x** | **8.0x** | **7.2x** | 1.1-1.5x |

注：8t/32t 共享箱噪声大（aten 单格曾摆动 104→426us）；可靠数字是交错
aten/hpc 的 median。4B f32 K=10M 大值域随机 gather 仍是硬件上限（1t 0.7x、
高线程 0.2x），T4 桶化因值域 >>2048 不启用。

## 6. 验证

- 全 6 个测试套件 **75/75 通过**（T5 后：test_correctness 25、test_errors 5、
  test_model 9、test_normalize 9、test_smoke 2、test_sve 25）。
- 正确性对拍：两进程法（无覆盖进程生成 true-aten 参考，覆盖进程比对），
  工具 `tests/verify_against_aten.py --mode both`（40/40），含 K>=1024、桶化、
  M1 raw、T5 单遍、多索引、mask 用例。
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
- ~~**P3 mask 多线程**~~：T8 已用单遍 compaction 修复（见 §13），不再列入。
- **4B 大 K 随机 gather**：920F SVE 4B gather MLP 硬件上限（RESULTS §8.2）；
  可试软件预取 / 标量多路预取。

## 9. T5：1 线程回归修复（线性化两遍模型 vs aten 单遍）

### 背景

回归扫描（`benchmarks/bench_regression.py`，interleaved median，pin node 5）发现
1 线程下多形态劣化（0.6-0.9x），2+ 线程全部占优（rowcopy A=4 2t 7x、1D pure 2t
15-20x、x[i,j] 2t 5.98x）。1 线程劣化根因（1D f32 K=5M 分解）：

- linearize 两遍模型固定开销：`linearize` 104ms + `gather` 124ms = 228ms vs
  aten 单遍 169ms；linearize 对 1 线程无并行加速，纯属多余的一遍。
- 920F SVE gather 单线程 MLP 低（RESULTS §8.2），换 SVE→scalar 无济于事
  （dispatch_scalar 0.64x 与 SVE 0.71-0.79x 相当）。
- 融合单遍（`gather_fused`）每元素做 at::Tensor 方法访问（element_size/stride/
  dtype/data_ptr，引用计数 + 虚调用），5M 元素下反而更慢（0.22-0.53x）。

### 变更

| 项 | 变更 | 代码 | 1 线程效果 |
|---|---|---|---|
| T5-1 | 单索引 A==1 大值域 1 线程走 `gather_single_1d`（单遍标量，跳过 linearize） | gather.cpp 新增内核 + op.cpp 路由 | 1D f32 0.75x→**1.07x**、f64 0.71x→**1.24x** |
| T5-2 | 单索引 A>1 1 线程走 `gather_fused` 单遍 | op.cpp 路由收窄（仅单索引 A>1） | rowcopy A=4 0.87x→**1.39x** |
| T5-3 | fused/linearize 预提取索引张量属性 + nb==1 全长连续快路径（直读 idx_j[k]） | gather.cpp `gather_fused`、linearize.cpp 两变体 | x[i,j] 0.66x→**1.21x**（linearize 4.84→1.10ms）；rowcopy A=4 1.35x |
| T5-4 | 修正路由范围 | op.cpp（多索引 A==1 与 mask 保持 linearize+SVE，fused 实测更差） | 避免 blanket fused 误伤（x[i,j] 曾 0.49x、x[:,i] 0.28x） |

### 路由现状（op.cpp index_impl）

```
1. K < 1024                                  → gather_direct
2. 单索引 && after_contig && A>=4096 && f32/f64 → M1 raw rowcopy
3. 单索引 && B==1 && A==1                    → T4 raw 桶化
4. 1 线程 && 单索引 && A>1                   → T3 gather_fused（跳过 linearize）
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

全 6 套件 **75/75 通过**（新增 6 个：多索引全长/广播标量/非连续、1 线程 rowcopy
短行、1 线程大值域纯 gather、linearize 快/慢路径一致性）。

## 11. T6：跨设备索引校验 + `aten::index.Tensor_out` 覆盖（正确性/覆盖）

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
| T6-1 | 入口复刻 `checkIndicesOnCpuOrSelfDevice`：非 CPU 索引抛 aten 同文错误（`indices should be either on cpu or on the same device as the indexed tensor (cpu)`），不再段错误 | normalize.cpp |
| T6-2 | 抽出 `index_into(self, spec, out)` 核心路由；新增 `index_out_impl` 写入调用方 out（形状不符 `resize_`；非连续 out 先算到临时连续张量再 `copy_`；校验 dtype/device/overlap） | op.cpp |
| T6-3 | 注册 `hpc_index::index_out` 并覆盖 `aten::index.Tensor_out` CPU key；`HPC_INDEX_NO_OVERRIDE=1` 仍可整体关闭，基准对拍语义不变 | op.cpp / register_aten.py |

### 验证

- 新增 `tests/test_index_out.py` 7 个用例（pure / rowcopy / mask / multi-index /
  非连续 out / dtype 不匹配 / profiler 断言内核名 `hpc_index::index_out`）；
  `test_errors.py` 增加 meta 索引子进程不崩溃用例（修复前 rc=-11）。
- 全 7 套件 **83/83 通过**；true-aten 值级对拍 40/40；autograd / vmap / jvp /
  index_put / setitem 行为不变；`HPC_INDEX_NO_OVERRIDE=1` 下 out= 仍为真 aten。

## 12. T7（实验，未落地）：`torch.compile` fallback

`torch.compile` 默认由 Inductor 内联生成 `aten::index` 的 kernel，绕过 dispatcher
覆盖（实测慢 2.6x）。用 `force_fallback(aten.index.Tensor)` 可让它回落到运行时
调用、重新命中本算子，实测 2.6x → 1.07-1.3x。含 inductor 缓存、私有 API、
FakeTensor 等坑与 opt-in 落地计划，见 [torch-compile.md](torch-compile.md)；
实测数据见 `benchmarks/RESULTS.md` §11。**尚未实现，仅记录可行性结论。**

## 13. T8：mask 单遍融合（跳过 nonzero）

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
