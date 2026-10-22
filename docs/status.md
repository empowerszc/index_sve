# 当前进展快照（status）

本文档面向**新读者 / 其他 agent**：读完即可理解项目现状、已经做了什么、还剩
什么。技术细节在各专题文档（见 `index.md` 索引）；这里只给结论与指针。

> 更新规则：每轮实质变更后更新本文（测试数、commit 号、性能结论、路由）。

## 1. 项目一句话

为昇腾 920F（aarch64, SVE2 VL=512）实现的 `aten::index.Tensor` CPU 语义替代
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

## 3. 路由现状（op.cpp `index_impl`）

```
K < 1024                          → gather_direct（无线性化）
单索引 && A>=4096 && 连续 && f32/f64 → M1 raw rowcopy（skip-linearize）
单索引 && B==1 && A==1            → T4 raw 桶化（小值域查表）
                                     └ 大值域 && 1 线程 → gather_single_1d（单遍标量）
1 线程 && 单索引 && A>1           → gather_fused（跳过 linearize）
A==1 || (连续 && A>=4 && f32/f64) → linearize + gather_index（SVE/桶化/scalar）
其余（A<4 或非连续小块）          → gather_fused
```

详细说明见 `architecture.md` §1 与 `optimizations.md` §9。

## 4. 测试

- **75/75 通过**：test_correctness 25、test_errors 5、test_model 9、
  test_normalize 9、test_smoke 2、test_sve 25。
- 复跑：`for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py tests/test_errors.py tests/test_sve.py tests/test_model.py; do python $t; done`
- 注意：测试同一进程串行跑，1 线程用例必须 `try/finally` 恢复
  `torch.get_num_threads()`（test_sve.py 内已如此）。
- **true-aten 值级对拍**（覆盖下测试自洽，须两进程法）：
  `python tests/verify_against_aten.py --mode both`（40/40，参考写
  `agent_space/`）。详见 semantics.md §6.2。

## 5. 性能结论（pin node 5，interleaved median）

**1 线程（T5 修复后全部 ≥1.0x）**：

| 形态 | 修复前 | 修复后 |
|---|---|---|
| 1D pure f32/f64 K=5M | 0.75x / 0.71x | **1.07x / 1.24x** |
| x[i,j] 多索引 | 0.66x | **1.21x** |
| rowcopy A=4 短行 | 0.87x | **1.39x** |
| mask | ~1.0x | 噪声级 0.97-1.13x（密度相关） |

**全线程扫描（T0-T5 构建，RESULTS §10，30 iters interleaved）**：核心场景
（S1/P1/P2/P3/S4、1D f64/f32、x[i,j]、rowcopy A=4）在 1/2/4/8/16/32/36/38
全部 8 档线程 **≥1.06x**，多数大幅占优。低线程最强（RowCopy/1D pure 2-4t
13-66x，单节点内存带宽 ~60 GB/s 用少量核即饱和）。P3 mask 最弱（1-1.9x，全程
平），是 nonzero→scalar 两段式串行的结构限制。复现：`benchmarks/bench_threads.py`
（`HPC_THREADS=1,2,4,8,16,32,36,38`）。

**硬件上限（不可修，非 bug）**：4B（f32/i32）大 K 随机 pure gather 在 920F 上
SVE 4B gather 指令 MLP 极低（38t ~0.75 GB/s vs 标量 3.4 GB/s，独立探针）。当前
构建相对 aten 已无 <1x 格（T0-T5 后），但探针数据仍是硬件限制的确定性陈述
（`benchmarks/RESULTS.md` §8.2）；T4 桶化在小值域时规避，大值域靠软件预取等
待未来优化。

## 6. 已知边界 / 注意事项

- 覆盖是一次性全局副作用：一个进程只能装一个 Python CPU 覆盖
  `aten::index.Tensor`；`HPC_INDEX_NO_OVERRIDE=1` 可跳过（基准对拍用）。
- 扩展链 BiSheng `libomp.so`，`LD_LIBRARY_PATH` 必须含 BiSheng lib，否则
  import 报 `libomp.so: cannot open`（`README.md` 快速开始）。
- 慢形态与"为什么不 fallback"的深度分析见 `fallback-analysis.md`：核心结论是
  4B（f32/i32）大 K 大值域 pure gather 受 920F SVE 4B gather 指令 MLP 上限
  （探针 0.75 vs 标量 3.4 GB/s，硬件不可消除）；mask/A<4/非连续/2B-1B 等相对
  aten 通常 ≥1x 只是吃不到 SVE 加速；fallback 的 Python wrapper 开销 ~45µs/
  调用对以 RowCopy 小张量为收益主体的本算子是不可接受的全局回归。

## 7. 下一步候选（未做）

- M1 值域桶化（T2 阶段 2）：`gather_rowcopy_raw` 未对下标做值域分析，可再桶化。
- 4B 大 K 大值域 gather：软件预取 / 标量多路预取（硬件 MLP 上限的软件规避）。
- mask 多线程 nonzero→scalar 的并行性问题（38t 0.35x，独立于本内核并行性）。
- 端到端 AI4S 模型复测（验证 T0-T5 在真实负载上的整体收益）。

## 8. 复现 / 复测命令

```bash
# 全量测试
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py; do python $t; done

# true-aten 值级对拍（正确性，非性能）
python tests/verify_against_aten.py --mode both

# 全线程扫描（RESULTS §10，pin node 5 静默节点，30 iters interleaved）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,2,4,8,16,32,36,38 \
  python benchmarks/bench_threads.py

# 回归扫描（pin node 5 静默节点，避免共享箱噪声）
HPC_INDEX_NO_OVERRIDE=1 HPC_AFFINITY=190-227 HPC_THREADS=1,8,32 \
  python benchmarks/bench_regression.py
```
