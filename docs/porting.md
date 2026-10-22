# 移植指南

本算子面向鲲鹏 920F（Kunpeng 920F, aarch64, SVE2, VL=512-bit）编写。本文档列出移植到
其他平台/配置时的检查点。

## 1. 其他 VL（如 128/256/1024）

- 编译参数 `-msve-vector-bits=512` 改为目标 VL；所有内核已用 `svcntd()/svcntw()`
  动态取向量宽度，**不硬编码 8/16**，理论上可移植。
- **运行时门控** `sve_512_ok()`（gather.cpp）固定要求 `svcntb()==64`。移植到
  其他 VL 时：改为 `svcntb() == target_bytes`，或直接 `svcntb()>0`（若新平台
  保证编译 VL = 运行 VL）。否则非目标 VL 会全部落标量（正确但无加速）。
- 若目标平台的 SVE 版本非 SVE2（如仅 SVE1），`svld1_gather_*` /
  `svprfd_gather_*` 在 SVE1 均可用，但 `svldnt1/svstnt1`（NT）可能缺失——
  NT 分支需用 `__ARM_FEATURE_SVE` 相关特性宏门控。

## 2. 其他 aarch64 CPU（无 SVE 或 SVE 性能差）

- 编译期：`-march=armv8-a+sve2 -msve-vector-bits=512` 改为 native 或去掉 SVE。
- 无 SVE 时 `arm_sve.h` 相关代码全部不触发（`sve_512_ok()` false → 全部落
  `dispatch_scalar`），语义仍正确，只是退化为标量实现（含并行
  `at::parallel_for`，仍快于串行 aten）。
- 注意：**920F 的 4B 随机 gather 受 64B 行放大 + DRAM 延迟限制**（SVE 4B 与
  标量持平，kernels.md §2）。若目标平台的行/预取特性更好，4B 分支的大 K 带宽
  可能更高，可重新基准确认是否启用 4B SVE 分支。

## 3. 其他 torch 版本

- 本实现基于 torch 2.10.0+cpu 的公开 API：
  - `at::expand_outplace`（ATen/ExpandUtils.h）—— 大版本更替可能移名；
  - dispatcher typed call `findSchemaOrThrow("hpc_index::index","").typed<...>`；
  - `TORCH_LIBRARY_IMPL(aten, CPU, m)` 覆盖签名（注意 C++ 签名须与 schema 一致；2.10 的
    signature 为 `qualname, type, func`）。
- **`Tensor.index()` 陷阱**：2.10 起 `Tensor.index(positions, dims)` 是
  first-class-dims API，不再等价 `aten::index`。测试/文档一律用 `x[...]`，勿
  写 `x.index([indices])`（A1）。
- `torch::ops::hpc_index::index` 在 torch 头文件中不存在（自定义 TORCH_LIBRARY
  算子不在 torch namespace），C++ 内必须用 dispatcher typed call（A2）。

## 4. 编译环境

- 需 arm64 + SVE2 工具链；本机 BiSheng clang 19.1.7
  （`compiler/bisheng/bin/clang++`）。
- `-fopenmp` 必须保留（`at::parallel_for` 依赖），否则所有并行退化为串行
  （历史根因，architecture.md §7）。
- `use_ninja=False`（`BuildExtension` 参数）——环境无 ninja；装了 ninja 的
  环境可去掉，纯构建速度差异。
- 运行时 `libomp.so` 依赖：部署时 `LD_LIBRARY_PATH` 必须含工具链 lib 目录
  （integration.md §3）。

## 5. 语义兼容性测试清单（移植后必跑）

1. `tests/test_smoke.py`（注册/调用）；
2. `tests/test_normalize.py` + `test_linearize_values`（spec/偏移结构）；
3. `tests/test_correctness.py`（dtype × 下标矩阵，含 K<1024/K>=1024）；
4. `tests/test_errors.py`（逐字错误格式）；
5. `tests/test_sve.py`（SVE 分支门控 + 值正确性 + NT 路径）；
6. `tests/test_model.py`（模型场景）；
7. 两进程 true-aten 对拍（semantics.md §6，含 K>=1024 用例）。

全部通过 = 102/102（覆盖激活态）。

## 6. 已知可优化项（后续工作）

- ~~**P3 mask 多线程**~~：T8 已用单遍 compaction 修复（RESULTS §12）。
- **4B 大 K 随机 gather**：受 64B 行放大/DRAM 延迟限制（SVE 与标量持平）；
  可试预取加深/软件流水，或按 cache line 分桶排序（kernels.md §2）。
- **M1 值域桶化（T2 阶段 2，未做）**：M1 大块路径当前是 skip-linearize 直接
  rowcopy，未对下标做值域分析（keys[order] 等若值域小可再桶化）。文档见
  优化任务建议 T2 阶段 2；`gather_rowcopy_raw` 已为后续接入留好入口。

已完成（T4 后不再需要）：~~int32 线性化临时缓冲~~ —— `linearize.cpp` 已改为
int32 直接写（不再分配 K×8B int64 再转换）。
