# index_sve 源码走读

> **这份文档回答"代码在哪里、按什么顺序读、每段在干什么"。** 拿着源码对照读，
> 一个下午可以读完核心 ~2000 行 C++。设计的"为什么"看
> [architecture.md](architecture.md)，每个内核为什么快/慢看 [kernels.md](kernels.md)，
> 遇到名词/代号查 [glossary.md](glossary.md)。
>
> 行号对应 `index_sve/index_sve/` 当前源码：op.cpp（500 行）、gather.cpp（1093 行）、
> normalize.cpp（213 行）、linearize.cpp（162 行）、四个头文件、124 行 Python 胶水。
> 代码注释里大量出现任务/场景代号（T2 = 优化轮次、M1 = 场景形态、A7 = 手段编号、
> S1 = 负载场景），不必逐一记住——glossary.md 有全表。
>
> 用法：第一次读按 §1 的顺序走；之后当字典用——§4.2 路由表、附录 A 行号速查、
> 附录 B 调试出口。

## 0. 30 秒全局地图

一句话：**把任意 `x[i1, i2, ...]` 归一成 (B, K, A) 三段几何，然后按形态路由到
十几个专门内核之一。**

先记住三个抽象：

1. **IndexSpec**（normalize.h:5）——normalize 阶段产出的"手术单"：转置后的源张量、
   被索引维的 size/stride、广播后的下标张量、B/K/A 三个数、row_off/col_off 两张
   偏移表。**后续内核只读这张单子里的裸指针和整数，不再碰 at::Tensor 方法**
   （data_ptr/dtype 等方法带引用计数与虚调用，内层循环里是性能大敌——"循环外
   预提取指针/stride/size"的模式在 linearize、gather_direct、gather_fused、
   rowcopy_fused_sve 等内核里反复出现，见到就认识它）。
2. **B-K-A 几何**——输出元素按行优先展开为 `p = r*K + k`（r∈[0,B)、k∈[0,K)），
   每个 p 对应源里一段 A 元素的"行"。B = 前段（未索引维）积、K = 广播下标元素数、
   A = 后段（尾部维）积。§3 有逐字段表和算例。
3. **两级路由**——`index_impl`（op.cpp:375）先试三个"免 normalize"快路径
   （mask / small_single / small_multi），不中才 normalize 出 IndexSpec，进
   `index_into`（op.cpp:48）大路由；大路由按 K/A/线程数/dtype 依次选内核。

端到端主干（真实函数名 + 行号，§2 有详细版）：

```
x[idx] ─► dispatcher（aten::index.Tensor 的 CPU 键已被 op.cpp:444 覆盖）
  └► index_impl                     op.cpp:375    入口校验 + sync_omp_threads
     ├► mask_fast_path              op.cpp:168   ─► gather_mask          gather.cpp:1021
     ├► small_single_index          op.cpp:316    免 normalize
     ├► small_multi_index           op.cpp:203    免 normalize
     ├► normalize_index             normalize.cpp:126 ─► IndexSpec
     └► index_into                  op.cpp:48     大路由（§4.2 十条路由）
        ├► gather_direct                          gather.cpp:748
        ├► gather_rowcopy_raw                     gather.cpp:864
        ├► gather_strided_rows_raw                gather.cpp:979
        ├► gather_rowcopy_fused                   gather.cpp:963
        ├► gather_bucket_pure_raw                 gather.cpp:779
        ├► gather_single_1d_sve / single_1d       gather.cpp:824 / 797
        ├► linearize                linearize.cpp:85（两遍模型第一遍）
        └► gather_index                          gather.cpp:657（第二遍 + 分发树）
```

## 1. 文件清单与推荐阅读顺序

| 文件 | 行数 | 职责一句话 | 优先级 |
|---|---|---|---|
| index_sve/normalize.h | 20 | **IndexSpec 定义**（核心数据结构）+ normalize_index 声明 | ★★★ 第一个读 |
| index_sve/op.cpp | 500 | 入口校验、两级路由、免 normalize 快路径、注册三件套、调试出口 | ★★★ |
| index_sve/normalize.cpp | 213 | ATen 语义归一：mask→nonzero、广播、转置、decode_offsets → IndexSpec | ★★★ |
| index_sve/linearize.cpp | 162 | 两遍模型第一遍：把广播下标物化成扁平元素偏移（可 int32 直写） | ★★ |
| index_sve/gather.cpp | 1093 | 全部计算内核：SVE 原语 → 中层 → 顶层入口（§7 族谱） | ★★★ 主战场 |
| index_sve/gather.h / linearize.h | 16/4 | 内核与 linearize 的声明 | ★ |
| index_sve/grain.h | 16 | adaptive_grain 并行粒度（§8） | ★ |
| index_sve/__init__.py | 15 | dlopen 触发注册 + 两个环境变量门（§9） | ★ |
| index_sve/register_aten.py | 11 | 历史兼容层：install() 已是空实现（§9） | |
| index_sve/compile_support.py | 98 | torch.compile 的 opt-in 三步（§9） | |
| setup.py（仓库 index_sve/ 根） | — | 构建入口 `python setup.py build_ext --inplace`（BiSheng 工具链、`-fopenmp`、SVE512；选项见源文件） | |
| tests/（14 个文件） | — | 143 项功能/对拍/路由/autograd 测试；改路由后必跑（跑法见 README） | |

**推荐阅读顺序**（每步给一个"读完应能回答"的检查点）：

1. `normalize.h` 全文（5 分钟）+ §3 逐字段表 → *B/K/A 各是什么？after_contiguous
   怎么判定？*
2. `op.cpp:375-390`（index_impl）→ *四个快路径的尝试顺序是什么？为什么能免
   normalize？*
3. `op.cpp:48-162`（index_into）对照 §4.2 路由表 → *x(50000,64) f32、i 长 1M、
   36 线程，走哪条路？（答案见 §2）*
4. `normalize.cpp` 全文 → *mask 什么时候变成 nonzero？什么时候 permute？*
5. `linearize.cpp` 全文 → *bfactor 混合进制是什么？什么时候能 int32 直写？*
6. `gather.cpp` 按 §7 三层读：先 gather_index 分发树(:657)，再跳到感兴趣的内核
   → *为什么 raw 路径必须内联检查？*

## 2. 一次 x[idx] 的完整旅程

以 `x = torch.randn(50000, 64)`、`idx` 为 1M 个 int64、`y = x[idx]`
（f32、36 线程）为例：

```
Python __getitem__
 └► torch/csrc 的 TensorIndexing 层：整数/切片在该层解决；x[None]（新轴）也在该层
    unsqueeze 解决，从不进入本算子。传给 dispatcher 的是 aten::index.Tensor
    (self, Tensor?[] indices)；列表里的 None/undefined = "该维跳过"占位（x[:, i] 的 :）
 └► dispatcher：CPU key 已被 op.cpp:444 的 TORCH_LIBRARY_IMPL(aten, CPU) 指到 index_impl
     └► index_impl                      op.cpp:375
        ① to_vec(:377)                  List → vector
        ② 校验(:378-380)                空列表 / 索引数超过维数
        ③ sync_omp_threads(:381)        omp_set_num_threads(at::get_num_threads())
                                        （T11：双 OpenMP runtime 的 ICV 同步，§8.3）
        ④ mask_fast_path(:383)          不中——idx 是 int64 不是 mask
        ⑤ small_single_index(:384)      不中——K = 1M > 4096
        ⑥ small_multi_index(:385)      不中——单索引
        ⑦ normalize_index(:386)        → IndexSpec：B=1、K=1M、A=64、
                                        after_contiguous=true（本例不 permute）
        ⑧ at::empty(out_shape)(:387)   y (1M, 64)
        ⑨ index_into(:388)             op.cpp:48，逐条试探：
           · #1 K<1024 ?                 否
           · #2 M1 raw(:62-67)？         A=64∈[16,64] 但 B*K=1M>65536 ✗、
                                        A<16 ✗、A≥4096 ✗ → 不中
           · #3 strided(:74)?           after_contiguous → ✗
           · #4 多索引 raw(:81)?         单索引 ✗
           · #5-#7 1D 族(:96)?           A≠1 ✗
           · #8 1 线程特判(:145)?        36 线程 ✗
           · #9(:154)：after_contiguous ∧ A=64≥4 ∧ f32 ✓
             → linearize(spec, idx)     linearize.cpp:85
                 use_i32（op.cpp:134）：numel=3.2M < 2^31 且 f32 → 物化成 int32 偏移
                 all_full_contig → 直读 idx[k]，内联检查+负归一，一遍写完
             → gather_index(spec, idx, y)  gather.cpp:657
                 A>1 ∧ after_contiguous ∧ SVE512 ∧ A≥4 ∧ f32
                 → rowcopy_sve<float,int32_t,false>（gather.cpp:350）
                     [0, B*K) 并行，p=r*K+k，整行 copy_contig_sve（gather.cpp:250）
```

读这棵树的要领：**路由条件全部显式写在 index_into 里**（§4.2 表），没有隐藏开关；
每条"不中"就自然落到下一条，末尾总有兜底（gather_fused / dispatch_scalar），
任何形状/dtype 都有归宿。想自己验证，用 `debug_spec` + 计数器（附录 B）。

## 3. 核心数据结构：IndexSpec 逐字段（normalize.h:5）

| 字段 | 作用 | 产出位置（normalize.cpp） |
|---|---|---|
| transposed_self | 被索引维换到最前后的 self（无"空洞"时就是 self 本身，permute 是视图） | :141-146 |
| indexed_dims / indexed_sizes / indexed_strides | 哪些维被索引，及其 size/stride | :152-167 |
| indices | 广播后的**已定义**下标张量列表（undefined 占位不进此表，经 before/after 计数消化） | :132-165 |
| broadcast_shape | 下标广播后的形状（K 段的形状） | :175 |
| B / K / A | 前段积 / 广播元素数 / 后段积 | :176-193 |
| row_off | [B] 表：前段扁平坐标 r → 源偏移 | :202（decode_offsets） |
| col_off | [A] 表：后段扁平坐标 c → 源列偏移 | :203（decode_offsets） |
| after_contiguous | 尾段是否原样连续 ⇔ col_off[c]==c 对所有 c 成立 | :205-211 |
| out_shape | 输出形状 = before + 广播形状 + after | :195-197 |

**B-K-A 算例**（x 形状 (5,6,7)，strides (42,7,1)）：

| 用法 | B | K | A | out_shape | 说明 |
|---|---|---|---|---|---|
| `x[:, i]`，i 长 3 | 5 | 3 | 7 | (5,3,7) | dim0 是前段（undefined 占位），dim1 被索引，dim2 是后段；row_off=[0,42,84,126,168] |
| `x[i]`，i 长 3 | 1 | 3 | 42 | (3,6,7) | dim1+dim2 合成后段 A=6*7 |
| `x[i, j]`，i(3,1)、j(1,4) | 1 | 12 | 7 | (3,4,7) | 广播成 (3,4)；索引 numel≠K，linearize 走 bfactor 解码分支 |

最后是全库坐标约定：**p = r*K + k**——输出扁平编号 p，恢复 `r = p/K`、`k = p%K`
（或增量维护，见 gather_direct:520-524），源行首 =
`transposed_self + row_off[r] + Σ v_j·indexed_strides[j]`，行内第 c 列 = `col_off[c]`。

## 4. op.cpp 逐段精读

### 4.1 工具层（:1-45）

- `to_vec`(:20)：`c10::List` → `std::vector`，后续按下标访问。
- `non_temporal_enabled`(:28)：读 `HPC_INDEX_NT` 环境变量，`static` 只读一次。
- `sync_omp_threads`(:36)：**T11 的关键修复**。扩展链 BiSheng libomp、ATen 链
  libgomp，`torch.set_num_threads` 只改 libgomp 的线程数 ICV；不同步的话本算子
  永远用 libomp 默认线程数，torch 的线程设置形同虚设。ICV 按调用线程生效，所以
  每次入口调一次（:381/:411）而不是只调一次。
- `g_single_1d` / `g_single_1d_sve`(:44-45)：诊断计数器（§4.8、附录 B）。

### 4.2 index_into：大路由（:48-162）——本文件的心脏

走到这里的调用都已 normalize（有 IndexSpec）、out 已分配。十条路由**从上到下依次
试探**，条件全部显式：

| # | 行号 | 准入条件（全部同时成立） | 去向 | 一句话动机 |
|---|---|---|---|---|
| 1 | :51 | `K < 1024` | gather_direct（gather.cpp:748） | 小 K 时物化一遍偏移纯属固定开销 |
| 2 | :62 | 单索引 ∧ after_contiguous ∧ 下标连续 ∧ A>1 ∧ (A≥4096 ∨ A<16 ∨ (A≤64 ∧ B·K≤65536)) ∧ f32/f64 | gather_rowcopy_raw（:864） | M1 大块跳过 linearize，边查边整行拷 |
| 3 | :74 | 单索引 ∧ **非**after_contiguous ∧ A>1 ∧ 下标连续 ∧ f32/f64（入口内再判 max_run≥4） | gather_strided_rows_raw（:979） | 尾维有 ≥4 连续 run 时按 run 块拷 |
| 4 | :81 | 多索引 ∧ after_contiguous ∧ A≥4 ∧ **线程≥16** ∧ f32/f64 | gather_rowcopy_fused（:963） | 多索引大并行下跳过两遍模型 |
| 5 | :96 | 单索引 ∧ B==1 ∧ A==1 ∧ 下标连续，且值域≤2048（内核内判） | gather_bucket_pure_raw（:779） | 1D 小值域建查找表（M3 keys[order]、M4 形态） |
| 6 | :117 | #5 未中 ∧ i64 下标 ∧ SVE512 ∧ 线程>1 ∧ f32/i32 ∧ stride₀==1 ∧ size<2³² ∧ 源>512KB | gather_single_1d_sve（:824） | T15-C：大源 4B 单遍 SVE（uzp1 收窄 i64 下标） |
| 7 | :125 | 下标连续 ∧ (无 SVE 内核的 dtype 组合 ∨ 单线程) | gather_single_1d（:797） | 单遍标量——这种组合下两遍模型纯亏 |
| 8 | :145 | 单线程 ∧ 单索引 ∧ A>1：A≥4 连续 f32/f64 → raw；否则 fused | raw / gather_fused | 1 线程放宽 #2 的规模门槛（T15） |
| 9 | :154 | A==1 ∨ (after_contiguous ∧ A≥4 ∧ f32/f64) | linearize + gather_index | **两遍模型**：物化偏移 → SVE 族 |
| 10 | :160 | 其余（A∈[2,4)、非连续等） | gather_fused（:1014） | 兜底单遍融合，不物化 idx |

容易漏的细节：

- **顺序就是优先级**：#1 在 #2 之前——K<1024 时哪怕形状是 M1 也走 direct。
  #2 的规模门槛放行三类形状（A≥4096 超宽行 / A<16 窄行 / A≤64 且行数少），
  注释 :56-61 记录了为什么大 K 不放行——raw 逐行检查链在冷源行拷贝下反而慢
  12-32%（A/B 实测）。
- `sve_ok`(:107-109) 只描述"这个 dtype/下标组合有没有 SVE 内核"：i32 下标配
  f32/i32，i64 下标配 f64/i64。**f32 数据 + i64 下标不在内**——那是 #6（T15-C
  专用内核）的地盘；#6 不中且多线程时落 #9（两遍模型 + gather_index 的 8B/4B 分支）。
- `use_i32`(:134-135)：`self.numel() < 2³¹ && (f32 ∨ i32)` 才用 int32 偏移数组，
  省一半物化内存/带宽；**溢出自动退 int64**——gather_index 的 4B SVE 分支要求
  idx32，退了就自然落 dispatch_scalar（注释 :132-133）。
- 非 SVE512 机器（`svcntb()!=64`）一切照跑：#6 不中、gather_index 里
  `sve_512_ok()` 再拦一次，最终 dispatch_scalar。路由条件从不假设硬件，只在
  入口检查。

### 4.3 mask_fast_path（:168-195）

准入：恰好一个索引、是 bool/byte mask、两者同 CPU、mask 维数在 1..self.dim()、
两者都连续、逐维 shape 相等（不等按 aten 文案报错 :181-186）。uint8 顺带发
deprecation 警告（:188-191，与 normalize.cpp:49-52 文案一致）。命中 →
`gather_mask`（gather.cpp:1021）。

**为什么在 normalize 之前**：aten 的 mask 语义要先把 mask 转 nonzero 下标再
gather；本路径一遍扫描直接写出结果，normalize 的 expand/permute/vector 分配全
用不上（T8）。mask 覆盖前 md 维的形状校验在这里提前做掉了。

### 4.4 small_multi_index（:203-303）

T13 免 normalize 路径。准入：≥2 个索引、全部 int64/int32、同 shape、连续、
CPU、self 连续、n≤65536（:206-224）。要点：

- `ind_stride`(:235-240) 从后往前累乘——"每维 stride 从 A 起"。
- `offs`(:251-272) **串行**预计算全部 n 个偏移（内联检查+负归一），顺带探测
  `same`（所有偏移相等 → 输出整块 fill/memcpy 同一源行，:280-287）；一般情形
  A==1 逐元素、否则逐行 memcpy（:288-293）。
- 拷贝段先 `AT_DISPATCH`(:274) 再按 `n*A ≥ 2048` 决定并行（:296）——阈值是
  固定值不含线程数：offs 已经串行算过，并行段只剩纯拷贝。

### 4.5 small_single_index 与并行阈值（:305-371）

T9。准入：单 int64/int32 下标、`K = idx.size(-1) ≤ 4096`（:324-325）、self/idx
连续。A≥2 每行 memcpy（:355-358，注释解释 A=3 标量循环比 memcpy 慢 ~2x 的实测）；
A==1 纯标量（:360）。

**并行阈值** `small_single_par_threshold()`(:308-311) = `max(16384, 512·线程数)`：
small K 下 fork-join 开销可能超过收益，且按总元素数 `n*A` 而非 n 判断——
注释 :305-307 记录了只看 n 导致 2D A=16 K=4096 在 8-36t 掉到 0.55-0.66x 的教训。

### 4.6 index_impl / index_out_impl（:375-420）

- `index_impl`：§2 的九步。快路径试探顺序固定为 **mask → small_single →
  small_multi → normalize**。
- `index_out_impl`(:394)：out= 变体**不走任何免 normalize 快路径**，直接
  normalize；dtype/device 校验与 aten 结构化 out 语义对齐（:402-406），需要时
  resize（:407），然后 `assert_no_internal_overlap` / `assert_no_overlap`
  （:408-410）。**非连续 out 走临时张量 + copy_**（:414-417）——语义正确性优先
  （test_index_out.py 对拍的就是这些边界）。

### 4.7 注册三件套（:422-450）

```
TORCH_LIBRARY(hpc_index, m)          :422  只 def schema，不挂 CompositeImplicitAutograd
TORCH_LIBRARY_IMPL(hpc_index, CPU)   :428  本家 op 的 CPU 实现
TORCH_LIBRARY_IMPL(aten, CPU, m)    :444  把 aten::index.Tensor / index.Tensor_out 的
                                         CPU 键改指到 index_impl / index_out_aten
```

- hpc_index 只 def schema 的原因：挂 CompositeImplicitAutograd 的话 PyTorch 拒绝
  再注册 fake/meta（torch.compile / torch.export 需要，compile_support.py 靠这个）。
- 在 C++ 覆盖而非 Python `torch.library.impl`：Python 覆盖每次调用多一次解释器
  往返，小 K 实测 10-20µs（:433-434 注释）；register_aten.py 因此退役。
- `HPC_INDEX_NO_OVERRIDE`(:445-446)：dlopen 时决定要不要劫持——基准对拍逃生门，
  在 import 前设置才生效。

### 4.8 pybind 调试出口（:452-500）

13 个导出（附录 B 全表）。最常用的三个：`debug_spec`（看 normalize 结果——
调试路由第一步）、`linearize_spec`（看物化出的偏移数组）、各路由命中计数器。
计数器是 relaxed atomic：只做诊断，不参与控制流。

## 5. normalize.cpp 逐段精读——ATen 语义对齐

职责：把 Python 层剩下的任意索引形态**无损地**折叠成 IndexSpec。对齐目标是
aten `TensorIndexing` 的 `make_info`（逐条对拍见 semantics.md）。

### 5.1 两个校验（:14-35）

- `check_index_types`(:14)：白名单 long/int/byte/bool，复刻 IndexingUtils.h 的
  `checkIndexTensorTypes(allow_int=true)`，错误文案一致。
- `check_indices_on_cpu`(:28)：self 恒 CPU（:128 总闸），索引必须同在 CPU——
  CUDA/meta 索引的 data_ptr 不可读，不拦截会段错误。

### 5.2 expand_tensors（:38-71）

逐项处理：undefined → 保留占位（**None/undefined = "该维跳过"**：到达算子的
`aten::index(x,[None,i])` 语义就是 `x[:, i]`；用户级 `x[None]` 新轴在 torch/csrc
的 TensorIndexing 层已 unsqueeze，从不进本算子——semantics.md §None 占位语义）；
bool/byte mask → 逐维 shape 校验后 `index.nonzero()`(:63)，**再把结果的每一列
`select(1, j)` 拆成 md 个独立下标张量**（mask 覆盖前 md 维 → md 个下标）。
nonzero 是全流水线**唯一借用的 aten 计算原语**——mask 快路径（§4.3）的价值就在
于常见形态能跳过它。

### 5.3 has_contiguous_subspace / transpose_to_front（:74-100）

被索引维之间夹着未索引维时（如 `x[i, :, j]`），把被索引维 permute 到最前
（:82-100），这样"被索引块"永远落在维度的 [dims_before, dims_before+dims_indexed)
区间，B-K-A 分类才能一遍扫完。permute 是视图，无拷贝。

### 5.4 decode_offsets（:102-122）

通用工具：把 [0,num) 的扁平编号按 sizes 做混合进制拆位、乘 strides 得元素偏移。
row_off（B 个）与 col_off（A 个）都由它生成。O(num·dims)，B/A 通常不大，可接受。

### 5.5 normalize_index 主体（:126-213），按序 11 步

1. `TORCH_CHECK(self.device().is_cpu())`(:128)——CPU only 总闸。
2. 两个校验（:129-130）。
3. `expand_tensors`(:132)：mask → nonzero、undefined 占位保留。
4. `at::expand_outplace`(:134) 把所有下标广播到公共形状；失败统一转成 aten 文案
   的 "shape mismatch" 索引错误（:135-137）。
5. 尾部补 undefined 到 self.dim()（:138-140）——此后向量长度恒等于维数。
6. 无连续子空间则 transpose_to_front（:141-146）。
7. 一遍扫维分类（:152-167）：undefined 维计入 before（还没遇到索引时）或 after；
   defined 维收集 indexed_dims/sizes/strides + indices，并记录 replacement_shape
   （广播后各 defined 形状全同，循环内覆盖赋值等价）。
8. 0-size 陷阱（:168-173）：被索引维 size==0 但下标形状里没有 0 → 报 aten 文案
   "index is out of bounds for dimension with size 0"。
9. K = replacement_shape 积（:175-177）；before/after sizes → B、A（:179-193）。
10. out_shape = before + replacement + after（:195-197）；row_off/col_off（:199-203）。
11. after_contiguous 判定：col_off[c]==c 对所有 c（:205-211）。

## 6. linearize.cpp 逐段精读——两遍模型的第一遍

**两遍模型** = 第一遍把 K 个广播下标**物化**成 K 个扁平元素偏移（linearize），
第二遍按偏移 gather（gather_index）。物化的收益：第二遍内循环只剩"加偏移+一次
访存"，SVE 向量化没有分支噪声；代价：K·4/8B 的一次额外写+读。**§4.2 里所有 raw
路由（#1-#8）存在的意义就是砍掉这一遍。**

### 6.1 linearize_i64（:9-81）

- `bfactor`(:14-17)：broadcast_shape 的行优先混合进制因子——`c_d = (k/bfactor[d]) % bshape[d]`
  把 k 拆成各维坐标。这就是"广播下标 → 每维各自取元素"的解码器；gather_direct
  (:480-482)、gather_fused(:574-576)、rowcopy_fused_sve(:898-899) 里的同款是同一
  逻辑（A7 手段）。
- 预提取（:20-31）：指针/stride/esize/is_long 提到循环外——注释 :18-19 点明
  at::Tensor 方法的引用计数+虚调用是单线程开销根因。
- `all_full_contig`(:31-33)：**所有索引 numel==K 且连续**时解码退化为直读
  `idx_j[k]`，省掉整个 bfactor 展开——x[i]（i 全长下标）全走这条快路径。
- 通用路径（:54-80)：每 k 逐 j 做 bfactor 坐标解码 → **内联边界检查**（:73-74，
  全流水线对两遍模型调用的唯一检查点）→ 负归一 → 累加 v·stride。

### 6.2 linearize（:85-162）与 int32 直写

按 idx_out 的 dtype 分派：kLong → linearize_i64；kInt → **直接写 int32**
（:88-160）。早期实现是"先写 int64 tmp 再截断"，现在一步到位（:90-91 注释）。
安全性由 use_i32 保证（偏移 < 2³¹）；int32 分支与 i64 分支逻辑同构，只是写入
窄一半——gather_index 的 4B SVE gather 拿 u32 偏移正好。

## 7. gather.cpp 逐段精读——内核族谱（三层）

1093 行按"原语 → 中层 → 顶层入口"三层组织。自上而下读会迷路，建议先读
gather_index 分发树（:657），再按需跳转。

### 7.1 第 0 层：SVE 原语（:16-221）

| 函数 | 行号 | 做什么 |
|---|---|---|
| `PREFETCH_DIST` | :16 | 预取距离常量 16（微基准 {0,16,32,64} 扫出的默认） |
| `sve_512_ok()` | :20 | `svcntb()==64` 才启用全部 SVE 路径（编译按 512-bit 定长） |
| `gather8_f64` / `gather8_s64` | :26/:31 | NT 模板二选一：`svldnt1_gather`（绕缓存）或 `svld1_gather` |
| `store8_f64` / `store4_f32` / `store4_s32` | :36-58 | store 侧的同款 NT 模板 |
| `pure_gather_sve_8b_f64` / `_8b_s64` | :59/:85 | 8B gather 主循环：预取循环 + 主循环 + `svwhilelt` 尾数 |
| `pure_gather_sve_4b_f32` / `_4b_s32` | :112/:138 | 4B 同构，每迭代 svcntw()=16 路 |
| `pure_gather_sve_4b_i64idx` | :168 | T15-C：f32/i32 数据 + **原始 i64 下标**。ACLE 没有"32 位数据配 64 位下标"的 gather，先向量检查/负归一（:173-188），`uzp1` 把各 i64 低 32 位收窄成 u32 偏移（:176-177），复用 u32index gather；主循环三段式见 :197-220 |

读原语的钥匙：**三段式循环**（pf 预取 :200-208 / 主循环 :209-213 / whilelt
尾数 :214-219）+ 4b_i64idx 里"检查用 64 位谓词、取数用 32 位谓词"（:198/:216）。

### 7.2 第 1 层：中层内核（:225-470）

| 函数 | 行号 | 做什么 | 谁调用 |
|---|---|---|---|
| `pure_gather_sve_flat` | :225 | A==1 纯 gather：[0,B·K) 并行，块内按行切段（块可能跨行 :230-244） | gather_index |
| `copy_contig_sve` | :250 | A 元素连续块拷贝：≤64B 直接 memcpy（谓词构造不划算 :253-258）；否则 4×VL 展开 + whilelt 尾数；NT 模板 | rowcopy/strided/mask 各处 |
| `ColRun` / `make_col_runs` | :304/:312 | 把 col_off 里相邻 +1 的元素合并成 {dst,src,len} run 段 | strided 路径 |
| `strided_rows_raw` | :326 | 尾维非连续：行内按 run 块拷（内联检查 :337-339） | gather_strided_rows_raw |
| `rowcopy_sve` | :350 | A>1 且 after_contiguous：整行 copy_contig（**无检查**——两遍模型的 idx 已安全） | gather_index |
| `scalar_gather` / `dispatch_scalar` | :363/:379 | 兜底标量：`d[c]=s[col_off[c]]`，支持全部 dtype | gather_index |
| `bucket_gather_pure` | :389 | 两遍模型后的小值域查表：分块求 min/max → 值域≤2048 建 L1 常驻表 | gather_index |
| `bucket_gather_pure_raw` | :429 | T4 raw 版：免 linearize，直接在原始下标上求值域+查表（内联检查 :443-445） | index_into #5 |

### 7.3 第 2 层：顶层入口（:473-1093）

| 入口 | 行号 | 路由来源 | 内部结构 |
|---|---|---|---|
| `gather_direct` | :748 | #1（K<1024） | 预提取 + **A==1 单索引全长连续的预校验无检查紧凑循环**（:502-526，vbuf 先把 K 个下标查/归一一遍）+ 通用逐元素循环（:528-561，bfactor 解码 + 内联检查） |
| `gather_fused` | :1014 | #8 否则 / #10 | 与 direct 同源但**不物化 idx**；面向 A<4 或非连续小块（T3） |
| `gather_index` | :657 | #9 | **分发树**（下表） |
| `gather_single_1d` | :797 | #7 | 1 线程/无 SVE 组合的 1D 单遍标量（内联检查 :768-771） |
| `gather_single_1d_sve` | :824 | #6 | T15-C：parallel_for 直接分段调 pure_gather_sve_4b_i64idx（:816-822） |
| `gather_bucket_pure_raw` | :779 | #5 | AT_DISPATCH 后按 idx32/i64 调 bucket raw；返回是否值域命中 |
| `gather_rowcopy_raw` | :864 | #2、#8 | AT_DISPATCH + NT×idx32 四实例 → `rowcopy_raw_single`(:840)：B==1 或通用 B·K 并行，**内联检查**（:848-851），逐行 copy_contig |
| `gather_rowcopy_fused` | :963 | #4 | f32/f64 × NT → `rowcopy_fused_sve`(:892)：多索引逐行解码 + 内联检查 + copy_contig；all_full_contig 直读快路径（:914-934） |
| `gather_strided_rows_raw` | :979 | #3 | 先 make_col_runs；**max_run<4 不接管**（:984，返回 false 落回后续路由）；dtype 门 f32/f64（:986） |
| `gather_mask` | :1021 | mask_fast_path | 两遍：分块计数（**n<1MB 串行** :1040——小掩码并行实测慢 ~2x）→ offsets 前缀和（:1045-1050）→ 压实拷贝（A≥4 且 f32/f64 用 copy_contig :1071-1082，否则逐元素 :1083） |

`gather_index`(:657-746) 分发树（两遍模型的第二遍）：

| 顺序 | 条件 | 内核 |
|---|---|---|
| 1 | A==1（pure）且值域≤2048 | bucket_gather_pure（:668-672，idx32/i64 两实例） |
| 2 | pure ∧ i64 偏移 ∧ SVE512 ∧ f64/i64 | pure_gather_sve_flat 8B（:673-694） |
| 3 | pure ∧ i32 偏移 ∧ SVE512 ∧ f32/i32 | pure_gather_sve_flat 4B（:695-716） |
| 4 | A>1 ∧ after_contiguous ∧ SVE512 ∧ A≥4 ∧ f32/f64 | rowcopy_sve（:717-738） |
| 5 | 其余 | dispatch_scalar（:739-744）——**全部 dtype 的最终兜底** |

### 7.4 AT_DISPATCH 白名单 = dtype 支持面的唯一真相

9 处 `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBFloat16, kBool, ...)`：
op.cpp:274（small_multi）、:341（small_single）、gather.cpp:664（gather_index）、
:749（direct）、:783（bucket raw）、:800（single_1d）、:868（rowcopy_raw）、
:1014（fused）、:1059（mask）。白名单之外的 dtype 在这里抛
NotImplementedError——要扩 dtype 先查这 9 处（支持矩阵见 kernels.md §1）。

另有 3 个入口**没有** AT_DISPATCH：gather_rowcopy_fused(:963)、
gather_strided_rows_raw(:979)、gather_single_1d_sve(:824)——它们只收 f32/f64
（或 f32/i32），dtype 由路由条件在 index_into 里先行把关，入口内直接 cast。

## 8. grain.h 与并行约定

### 8.1 adaptive_grain 全文（grain.h）

```cpp
inline int64_t adaptive_grain(int64_t n) {
  constexpr int64_t MAX_GRAIN = 65536;
  constexpr int64_t MIN_GRAIN = 64;
  const int64_t nt = at::get_num_threads();
  if (nt <= 1) return std::max<int64_t>(1, n);   // 单线程：一整块 = 串行
  int64_t grain = std::max<int64_t>(MIN_GRAIN, n / (nt * 8));
  return std::min<int64_t>(grain, MAX_GRAIN);
}
```

- `n/(nt*8)`：每线程约 8 个 chunk——静态调度没有 work stealing，8 块/线程把
  负载不均的拖尾损失限制在 ~1/8 粒度。
- 下限 64：高线程数下极小 n 不至于切成单元素 chunk（fork-join 任务风暴）。
- 上限 65536：超大 n 的单 chunk 不过粗（cache 局部性）。
- `nt<=1` 返回 n：一块 = 串行，省掉 parallel_for 包装。

### 8.2 全库统一的并行骨架

```cpp
at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
  for (int64_t p = lo; p < hi; ++p) {   // p = r*K + k
    const int64_t r = p / K, k = p % K;
    ...
```

出现位置：pure_gather_sve_flat:229、rowcopy_sve:353、scalar_gather:367、
strided_rows_raw:333、rowcopy_raw_single:854/:859、rowcopy_fused_sve:915/:935、
gather_direct:519/:528、gather_fused:598/:619、bucket:397/:419、
gather_single_1d:766、single_1d_sve:819、linearize:35/:54/:115/:134、
gather_mask:1043/:1063、small_multi:297、small_single:365。
**读任何一个内核，先找它的 parallel_for 行——并行域和坐标约定一眼可辨。**

### 8.3 双 OpenMP runtime（T11 的背景）

本扩展用 BiSheng 工具链编译 → libomp；ATen 自身 → libgomp。两个 runtime 各有
一份线程数 ICV，`torch.set_num_threads` 只改 libgomp。因此：

- **所有并行走 `at::parallel_for`**（ATen 头在本 TU 按 `-fopenmp` 编译 → libomp）；
- **入口必须 `sync_omp_threads()`**（op.cpp:36，调用点 :381/:411）把 torch 的
  线程数灌进 libomp，否则 torch.set_num_threads 对本算子无效——历史上 2-36t
  旧数据全因此作废（degradation-analysis.md §3.1）。

宿主多进程/多线程的嵌入行为（超订实测、唯一应避免的嵌套形态）见
integration.md §6.1。

## 9. Python 胶水层

### __init__.py（15 行全文骨架）

```python
from . import _hpc_index            # dlopen 扩展 .so：TORCH_LIBRARY_IMPL 生效（覆盖发生地，
                                    # 首次 import 时 dispatcher 打印一次性告警）
from .register_aten import install  # 兼容层（见下）
import torch, os
if os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") != "1":
    _install_aten()                 # 空实现——覆盖已在上面 import 时完成
if os.environ.get("HPC_INDEX_COMPILE", "0") == "1":
    from .compile_support import enable
    _enable_compile()               # opt-in：compile 三步
def index(self, indices): ...       # torch.ops.hpc_index.index 的兼容入口
```

import 顺序即注册顺序：dlopen 先于一切。两个环境变量都必须在 import 前设置。

### register_aten.py（11 行）

`install()` 是**空实现**——Python 侧覆盖（torch.library.impl）是初版方案，每次
调用多 10-20µs 解释器往返；A6 后改由 op.cpp:444 的 C++ 注册取代。保留空壳为
兼容旧 import 代码。

### compile_support.py（98 行）

`HPC_INDEX_COMPILE=1` 时 `enable()` 三步（docstring :1-13）：

1. 为 hpc_index::index / index_out 注册 **fake**——整数下标形状可静态推导；mask
   输出依赖数据，显式 NotImplementedError（:25-34）；
2. 把 Inductor 的 aten::index **lowering 换成 fallback**——编译产物运行时回
   dispatcher 才能命中本算子；
3. 切**独立 inductor 缓存目录**——缓存 key 不含 lowering 注册信息，不改会静默
   复用未打补丁的旧缓存。

依赖 `torch._inductor` 私有 API，版本不符直接报错而不是静默失效（torch 2.10
验证过）。细节与风险见 torch-compile.md。

### setup.py

构建入口（`python setup.py build_ext --inplace`）：BiSheng 编译器、`-fopenmp`
（链 libomp）、SVE512 编译选项。运行时环境契约（LD_LIBRARY_PATH 等）见
integration.md §3。

## 10. 读代码时反复对照的十个不变量

1. **递归禁令**：任何路径不得调用会回到 `aten::index` 的算子——CPU 键已被本
   扩展覆盖，调它会回到 index_impl 自己。全流水线借用的 aten 计算原语只有
   nonzero（normalize.cpp:63）、expand_outplace（:134）与 permute/copy_/empty/
   resize_ 等视图、分配与拷贝，均不会路由回 index。
2. **raw 路径必须内联边界检查 + 负归一**：凡跳过 linearize 的内核
   （direct / rowcopy_raw / strided_raw / rowcopy_fused / single_1d 族 /
   bucket_raw），循环内必须自己 `TORCH_CHECK_INDEX(v >= -size && v < size)` 并
   `if (v < 0) v += size;`——样板见 gather.cpp:848-851。漏写 = 越界读（段错误）
   或负索引静默错数据。
3. **p = r*K + k** 是全库统一坐标（§8.2）；增量维护版本见 gather_direct:520-524。
4. **after_contiguous ⇔ col_off[c]==c**（normalize.cpp:205-211）：决定"整行块拷贝"
   还是"逐列取数"，是路由 #2/#9 与 #3 的分水岭。
5. **dtype 支持面 = 9 处 AT_DISPATCH 白名单**（§7.4）。barebones dtype 抛
   NotImplementedError 的唯一来源；扩 dtype 先改这里再补内核。
6. **双 OpenMP runtime**：一切并行走 at::parallel_for（libomp），入口
   sync_omp_threads 同步 ICV（op.cpp:36/:381/:411）。新内核忘了这两条，
   torch.set_num_threads 就管不住它。
7. **use_i32 门槛**（op.cpp:134-135）：`numel < 2³¹ ∧ (f32∨i32)`。numel≥2³¹ 时
   偏移装不进 int32，自动退 int64 → 4B SVE 分支要求 idx32 不触发 → dispatch_scalar
   兜底——正确性机制，性能代价（见 degradation-analysis.md）。
8. **计数器是 relaxed atomic**：g_* 计数器只供 pybind 诊断（附录 B），不参与任何
   控制流——删掉不影响语义。
9. **out= 语义优先**：非连续 out 走临时张量+copy_（op.cpp:414-417）；overlap
   断言（:408-410）。别在 out 变体上为性能抄近路。
10. **并行阈值是标定出来的**：small_multi 的 n·A≥2048（op.cpp:296）、
    small_single 的 max(16384, 512·nt)（:308-311）、mask 计数的 1MB
    （gather.cpp:1040）。改任何一个都要带多线程实测，否则历史重演
    （:305-307 注释里 0.55x 的教训）。

## 附录 A：函数 → 行号速查表

**op.cpp**：to_vec :20 · non_temporal_enabled :28 · sync_omp_threads :36 ·
index_into :48 · mask_fast_path :168 · small_multi_index :203 ·
small_single_par_threshold :308 · small_single_index :316 · index_impl :375 ·
index_out_impl :394 · TORCH_LIBRARY(hpc_index) :422 ·
TORCH_LIBRARY_IMPL(hpc_index,CPU) :428 · TORCH_LIBRARY_IMPL(aten,CPU) :444 ·
PYBIND11_MODULE :452

**normalize.cpp**：is_defined :9 · check_index_types :14 · check_indices_on_cpu :28 ·
expand_tensors :38 · has_contiguous_subspace :74 · transpose_to_front :82 ·
decode_offsets :102 · normalize_index :126

**linearize.cpp**：linearize_i64 :9 · linearize :85

**gather.cpp**：sve_512_ok :20 · gather8_f64 :26 · gather8_s64 :31 ·
（store 助手 :36-58）· pure_gather_sve_8b_f64 :59 · _8b_s64 :85 · _4b_f32 :112 ·
_4b_s32 :138 · pure_gather_sve_4b_i64idx :168 · pure_gather_sve_flat :225 ·
copy_contig_sve :250 · ColRun :304 · make_col_runs :312 · strided_rows_raw :326 ·
rowcopy_sve :350 · scalar_gather :363 · dispatch_scalar :379 · bucket_gather_pure :389 ·
bucket_gather_pure_raw :429 · gather_direct 模板 :473 / 包装 :748 ·
gather_fused 模板 :568 / 包装 :1014 · gather_index :657 ·
gather_single_1d 模板 :761 / 包装 :797 · gather_bucket_pure_raw 包装 :779 ·
gather_single_1d_sve_impl :816 / 包装 :824 · rowcopy_raw_single :840 ·
gather_rowcopy_raw :864 · rowcopy_fused_sve :892 · gather_rowcopy_fused :963 ·
gather_strided_rows_raw :979 · gather_mask :1021 · 计数函数 :1092-1093

**grain.h**：adaptive_grain :9

## 附录 B：pybind 调试出口（`index_sve._hpc_index.*`，op.cpp:452-500）

| 导出 | 行号 | 用途 |
|---|---|---|
| `index(self, indices)` | :453 | 绕过 __getitem__ 直达 dispatcher（对拍/计时不走 Python 切片层） |
| `linearize_spec(self, indices)` | :460 | 返回物化后的 int64 偏移数组——看 linearize 到底算了什么 |
| `debug_spec(self, indices)` | :467 | **调试路由第一步**：完整 IndexSpec（B/K/A/row_off/col_off/after_contiguous/indices…） |
| `mask_fused_count` | :474 | mask 快路径命中次数 |
| `small_single_count` / `small_multi_count` | :477/:480 | 两个免 normalize 路径命中次数 |
| `strided_rows_count` / `rowcopy_raw_count` | :483/:484 | strided / raw rowcopy（含 fused）命中次数 |
| `single_1d_count` / `single_1d_sve_count` | :485/:488 | 1D 单遍标量 / T15-C SVE 命中次数 |
| `debug_ss_parallel(n, A)` | :491 | small_single 是否并行的判定镜像（测试断言用） |
| `debug_grain(n)` | :494 | adaptive_grain 结果镜像 |
| `idx_dtype_choice(numel, dtype)` | :495 | use_i32 判定镜像（测试 f32 溢出回退） |

典型调试开场：

```python
import index_sve
S = index_sve._hpc_index.debug_spec(x, [idx])   # 先看 B/K/A 与 after_contiguous
index_sve._hpc_index.rowcopy_raw_count()          # 再看路由命中了谁
```
