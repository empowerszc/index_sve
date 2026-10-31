# 排查手记与套路：测试前置条件型失败

> **案例日期**：2026-10-31（n20，node 5，608 核 / 32 NUMA，`torch 2.10.0+cpu`）
> **对象**：`tests/test_rowcopy_routes.py`（当时 14 项，现 15 项）、
> `tests/test_output_layout.py`（25 项）
> **一句话结论**：两起"测试可能过不了"都不是随机 flaky，而是**测试自身对运行环境
> 的前置假设从未被声明**。修复分两层：先让环境不对时**大声失败**（守卫），再给
> "某条路由不该被选"的负向断言补上**正向控制**，否则覆盖静默失效时它照样全绿。

本文分两部分：§1-§2 是可复制的方法与清单，§3-§6 是这两个案例的完整复盘（含走过
的弯路），§7 之后是可以直接照做的检查表与命令。

---

## 1. 起点：先钉死基线，再谈复现

报告是"这些测试**可能**无法全部通过"。这种措辞不能直接当证据用——先要知道默认
环境下到底怎样，否则后面每一次运行都失去参照。

```bash
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py \
         tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py \
         tests/test_compile_optin.py tests/test_strided_rows.py \
         tests/test_rowcopy_routes.py tests/test_small_multi.py tests/test_autograd.py \
         tests/test_output_layout.py; do python $t; done
```

> **数据轮次：2026-10-31（n20，node 5）｜当前。** 15 套件 **175/175**。此后所有
> "复现不了"的判断都以这一行为参照。

本仓库没有 pytest（`ModuleNotFoundError: No module named 'pytest'`），每个文件自带
`tests/_harness.py`，它按**函数名字典序**收集 `test_*` 并统计 PASS/FAIL。这点很
重要：**执行顺序是函数名字典序，不是源码顺序**；跨文件则由外层 `for` 循环决定。
所以"前面某个用例改了全局状态"这类耦合不会被自动隔离——排查时要么逐条新进程跑，
要么显式逆序跑。

---

## 2. 方法论：四步，顺序不能换

### 2.1 第一步不是跑，是**读断言**

同一个文件里的 `assert` 不是同一种东西。按"被测特性被关掉时会怎样"分四型：

| 断言型 | 写法 | 特性被关掉时 | 信号质量 |
|---|---|---|---|
| **正向计数** | `assert count() == before + 1` | 计数不涨 → **失败** | 好：会叫 |
| **负向计数** | `assert count() == before` | 计数本来就不涨 → **仍然通过** | **危险：假通过** |
| **值级** | `assert_close(y, ref)` | 参考实现自己算对 → 仍然通过 | 中：只验语义不验路由 |
| **元数据** | `assert y.stride() == (...)` | 取决于参考张量是否同源 | 中 |

这一步就足以解释后面所有现象。**"全绿"和"有 fail"都不能单独当证据**——你必须先
知道这个文件里混着哪些型。

### 2.2 第二步：穷举"能让行为不同的变量"

不要随机试。列一张表，逐个问"这个变量改变时，被测代码的判据会不会变"：

| 变量 | 本例是否影响 | 判据来源 |
|---|---|---|
| 环境变量（`HPC_INDEX_*`） | **是**（决定性） | `index_sve/__init__.py:6`、`op.cpp` 的 `getenv` |
| `torch.set_num_threads` | 是（但测试自己会设） | `at::get_num_threads()` |
| 物理核数 / 亲和（`taskset`） | **否** | 全仓无 `hardware_concurrency` |
| 执行顺序（正序/逆序/单进程串跑） | 否 | 实测 174/174 |
| 工作目录 | 否 | `sys.path` 用 `__file__` 推导 |
| 默认 dtype / deterministic / mkldnn | 否 | 实测 25/25 |
| torch 版本 | **是**（布局期望值） | 见 §4.5 |
| 硬件 ISA（SVE512） | **是** | `op.cpp:125` |

### 2.3 第三步：负向对照（性价比最高的一步）

把被测特性**关掉**再跑一遍。依赖关系会立刻现形：

```bash
HPC_INDEX_NO_OVERRIDE=1 python tests/test_rowcopy_routes.py     # → 12/14
HPC_INDEX_NO_OVERRIDE=1 python tests/test_output_layout.py      # → 25/25（!!）
```

一个变成 12/14，一个纹丝不动。**这两个不同的结果才是本次排查真正的突破口**，比
"默认环境全绿"有价值得多。

### 2.4 第四步：不要把"复现不了"当结案

必须能说出**每个变量排除了什么**。"我跑了 13 种变体都绿"是证据；"我这边过了"不是。
同时把结论拆成两种，它们的处置完全不同：

- **A. 没有 bug**：那就得解释报告者为什么会看到失败（通常是环境差异）。
- **B. 有 bug 但触发条件在没试的变量上**：那就得指出**下一个该试什么**。

---

## 3. 案例一：`test_rowcopy_routes`（14 项）

### 3.1 基线

```
$ python tests/test_rowcopy_routes.py
== 14/14 passed ==
```

（本节数字按当时的 14 项计。后来这个文件又加了源大小门边界用例，现为 15 项；
结论不受影响。）

报告方说"可能无法全部通过"，所以先怀疑**环境差异**，不怀疑断言写错。

### 3.2 假设 A：机器核数/线程数不够 —— 证伪

第一反应是"用例硬编码了 16/36/38 线程，小机器上 `set_num_threads` 会被截断"。
这个假设很诱人，但**读一下判据就死了**：

| 判据 | 位置 | 读的是什么 |
|---|---|---|
| 多索引 raw 路由 | `op.cpp:104` | `at::get_num_threads() >= 16` |
| 并行台阶 | `op.cpp:330`（函数）、`:333`（返回） | `max(16384, 512 * at::get_num_threads())` |
| 自适应粒度 | `grain.h:14` | `n / (at::get_num_threads() * 8)` |

全仓 `grep hardware_concurrency` **零命中**。也就是说：判据只跟随 `torch.set_num_threads`
的**设定值**，与物理核数无关。

实测确认：`taskset -c 190-193`（4 核亲和）下 `torch.set_num_threads(36)` 后
`torch.get_num_threads()` 仍返回 36 → 公式两侧一致，不会偏移。

> **数据轮次：2026-10-31（n20，node 5）｜当前。**

| 变体 | 结果 |
|---|---|
| 默认 | 14/14 |
| 14 项逐条在新进程单独跑 | 0 失败 |
| `taskset -c 190-193`（4 核亲和） | 14/14 |
| `OMP_NUM_THREADS=4` | 14/14 |
| 同一进程连跑两遍（幂等） | 两轮 14/14 |
| `HPC_INDEX_COMPILE=1` | 14/14 |
| **`HPC_INDEX_NO_OVERRIDE=1`** | **12/14** |

### 3.3 假设 B：覆盖没生效 —— 成立

```
FAIL test_cl4d_row_index_routes_rowcopy_raw: AssertionError('nt=1: cl4d row index ...')
FAIL test_cl4d_mask_routes_rowcopy_raw:     AssertionError('cl4d mask ...')
```

这两项是整个文件里**唯一用真 `x[...]` dispatch 证明"路由确实被走到了"的用例**
（`y = x[rows]` / `y = x[m]`）。覆盖关掉后，`aten` 自己算出了正确结果，值断言照过；
但 `_hpc_index` 的计数器不涨，`rowcopy_raw_count() == before + 1` 就挂了。

### 3.4 更隐蔽的：还有 4 项是**假通过**

同一条件下，另外四项 `assert count() == before` 的用例全绿——因为覆盖关了计数器本来
就不动。它们的"通过"与路由是否正确**毫无关系**：

| 用例 | 断言 | `NO_OVERRIDE=1` 下的真实含义 |
|---|---|---|
| `test_cl4d_sliced_tail_excluded` | `count == before` | 排除条件完全没被验证 |
| `test_cl4d_small_k_excluded` | `count == before` | 同上 |
| `test_cl4d_half_excluded` | `count == before` | 同上 |
| `test_cl3d_perm_row_index_direct_values` | 只断值 + stride | 值由 aten 算对，路由未被验证 |

**这是本次最重要的发现：假通过比 fail 危险。** fail 会被人看见并追问；假通过会
进入"全绿"的统计，成为虚假信心。所以 `12/14` 这个数字本身还是**乐观的**——真实
失效面是 6/14。

诊断出这一条之后，第一版修复只加了守卫（§5.2），**没有**动断言本身。那是不够的，
补法见 §5.5。

### 3.5 还有一条真前置：SVE512 硬件

```cpp
// op.cpp:125
const bool hw_sve512 = (svcntb() == 64);
```

`test_f32_i64_1d_route` 的"大源直连 SVE 内核"断言依赖它，且**没有任何环境变量逃生门**。
本仓库只面向 920F，所以在本机不是问题；但换 SVE256 或无 SVE 机器会直接挂在那一行。
这类信息应该写在测试里，而不是靠读者去翻 C++。

---

## 4. 案例二：`test_output_layout`（25 项）

### 4.1 基线

```
$ python tests/test_output_layout.py
== 25/25 passed ==
```

文件特征：25 项**全部**是 `x[...]`，**零个计数器断言**（整个文件没有引用
`_hpc_index`，`import index_sve` 只为装覆盖）。

### 4.2 穷举式排除

> **数据轮次：2026-10-31（n20，node 5）｜当前。**

| 变体 | 结果 |
|---|---|
| 默认 | 25/25 |
| **`HPC_INDEX_NO_OVERRIDE=1`** | **25/25** |
| `OMP_NUM_THREADS=1` / `HPC_INDEX_NT=1` / `HPC_INDEX_COMPILE=1` | 各 25/25 |
| 线程数 1/2/3/4/7/8/16/36/38/64/128（逐档显式设置） | 11 档全 25/25 |
| 25 项逐条新进程 / 逆序 / 连跑 5 次 | 0 失败 |
| `numactl --membind=5 --physcpubind=190-227` | 25/25 |
| 默认 dtype=float64 / deterministic / mkldnn off / grad off | 各 25/25 |
| cwd = 仓库根 / `/tmp` / `tests/` | 各 25/25 |
| 15 套件同一进程串跑 | 174/174 |

到这里可以下"复现不了"的结论，但**不能结案**——必须解释"关掉覆盖为什么还是全绿"。

### 4.3 决定性实验：关掉被测特性，它依然全绿

文件头自己写着：

> 各期望值均为 `HPC_INDEX_NO_OVERRIDE=1` 无覆盖进程对 torch 2.10.0+cpu 的实测

即：**期望值就是真 aten 的 stride**。那么"关覆盖"等于"拿真 aten 去比对真 aten
快照"，必然全绿。这一步同时给出两个结论：

1. **好的一面**：本算子当前确实完整复现了 aten 的输出布局（否则开覆盖时会挂）。
2. **坏的一面**：这套测试**检测不到覆盖是否注册成功**。若某天注册悄悄失效、内核
   一行没跑，25 项照样全绿——它连"被测对象在不在"都不知道。

### 4.4 两种失败模式对照

| | `test_rowcopy_routes` | `test_output_layout` |
|---|---|---|
| 关覆盖时 | 修复前 12/14（2 fail + 4 假通过）；修复后 8/14，6 项全挂（当时该文件 14 项） | **25/25 全绿** |
| 信号 | 会叫，但叫得含糊（只说"路由不对"） | **不叫** |
| 风险 | 误导定位（以为内核坏了） | 虚假信心（以为验证过了） |
| 根因 | 缺"覆盖是否激活"前置声明 | 同左 |

**"会叫但叫错"好过"不叫"。** 这是两者的共同病根：都假设环境已配对，却都没声明。

### 4.5 第二层风险：期望值钉在 torch 版本上

这批 stride 来自 ATen 的 `TensorIterator` 输出分配（按 operands 的 stride 做
`should_swap` 插入排序得 `perm`，再按 `perm` 序 dense 分配）。**换 torch 版本，
`perm` 逻辑一变，25 项会一起挂**，而 `index_sve` 一行没动。文件头已声明是
2.10.0+cpu 的实测值，所以这是已知取舍，不是 bug。

### 4.6 报告方三分钟自证

先跑这条（把覆盖关掉）：

```bash
HPC_INDEX_NO_OVERRIDE=1 python tests/test_output_layout.py
```

| 结果 | 结论 |
|---|---|
| 也挂 | 期望值与你的 aten 不一致 → **版本问题**（§4.5），不是本算子 |
| 全绿 | 说明期望值 == 你真 aten；此时若开覆盖才挂，是本算子布局**真 bug** |

---

## 5. 修复（方案 A）

### 5.1 修在"真值源头"，不修在"每个测试里"

最省事的写法是在两个测试文件里各读一次 `os.environ.get("HPC_INDEX_NO_OVERRIDE")`。
坏处是**判据出现三份**（包、C++、测试），任何一处将来漂移，守卫就和实际注册脱钩。

因此先让包暴露唯一判据：

```python
# index_sve/__init__.py:6-9
OVERRIDE_ACTIVE = os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") != "1"

if OVERRIDE_ACTIVE:
    _install_aten()   # HPC_INDEX_NO_OVERRIDE=1 跳过（基准对拍用）
```

这与 C++ 侧 `TORCH_LIBRARY_IMPL` 的 `getenv` 读的是同一个变量、同一时刻（import /
dlopen），所以两者不会分裂。

### 5.2 守卫

```python
# tests/test_rowcopy_routes.py:7
if not index_sve.OVERRIDE_ACTIVE:
    raise SystemExit(
        "test_rowcopy_routes 需要覆盖激活：cl4d 六项靠 x[...] 验证路由，覆盖关闭时 "
        "6 项全挂（HPC_INDEX_NO_OVERRIDE=1？）")

# tests/test_output_layout.py:19
if not index_sve.OVERRIDE_ACTIVE:
    raise SystemExit(
        "test_output_layout 需要覆盖激活：25 项全走 x[...]，覆盖关闭时仍全绿但"
        "不验证任何东西（HPC_INDEX_NO_OVERRIDE=1？）")
```

### 5.3 三个决策的理由

1. **`SystemExit` 而不是 `assert`**：`assert` 在 `python -O` 下被剥掉，守卫会静默
   消失；而且 `assert` 混在测试断言里，读者分不清"环境错"和"被测代码错"。
   `SystemExit(msg)` 打印 msg 到 stderr 并给退出码 1，两层信息都清楚。
2. **提示语必须写"没有守卫会发生什么"**：`需要覆盖激活` 只说明规则；补上
   `6 项全挂` / `25 项全绿但不验证任何东西` 才能让人一眼判断"我是不是中招了"，
   也把本节的结论固化进代码。
3. **放在 import 之后、测试函数之前**：模块级执行，`_harness.main(globals())` 收集
   之前就退出，不会出现"先跑一半再炸"。

### 5.4 文档同步

改 `__init__.py` 会让三处引用失效，必须一起动（否则下一轮 `check_doc_anchors` 或
读者都会被误导）：

| 文件 | 改动 |
|---|---|
| `README.md` §测试 | 新增前置条件段：哪些套件依赖覆盖、`NO_OVERRIDE=1` 下会怎样 |
| `docs/integration.md` §1 | 代码骨架同步加 `OVERRIDE_ACTIVE` |
| `docs/code-walkthrough.md` §9 | 代码骨架同步，"15 行"→"17 行" |

### 5.5 补记：4 项假通过也要真修掉，守卫只是第一步

守卫解决的是"环境不对时别给假信号"，但它**不解决断言本身验不了东西**。守卫读的
是环境变量；万一注册真的失效而变量正常（例如被别的库又覆盖回去），守卫照样放行，
那 4 项仍然全绿。所以还差一件真正的事：给"某条路由**不该**被选"的负向断言配一个
**正向控制**。

做法是在两个受控入口各 +1 一个计数（`op.cpp:660` 导出为 `dispatch_count`）：

```cpp
std::atomic<uint64_t> g_dispatch{0};          // 与 g_single_1d 等同一处声明
...
g_dispatch.fetch_add(1, std::memory_order_relaxed);   // index_impl 与 index_out_impl 各一处
```

测试侧在负向断言旁边补一句：

```python
before_d = C.dispatch_count()
y = x[rows]
torch.testing.assert_close(y, ref)
assert C.rowcopy_raw_count() == before, "must not block-copy direct"
assert C.dispatch_count() == before_d + 1, "kernel did not run"   # 内核确实跑到了
```

**怎么证明这个修法真的有效**：把守卫骗过去（进程内把 `OVERRIDE_ACTIVE` 改成
`True`）而实际关掉覆盖，再跑一遍——守卫放行，剩下的全靠断言自己扛：

> **数据轮次：2026-10-31（n20，node 5）｜当前。**

| 场景 | 结果 |
|---|---|
| 修前（只有守卫） | 12/14：2 fail + **4 假通过** |
| 修后（守卫 + 正向控制） | **8/14**：6 项全挂，假通过归零（当轮该文件 14 项，现 15 项 → 9/15） |

挂掉的正好是 6 个走 `x[...]` 的用例：`test_cl4d_row_index_routes_rowcopy_raw`、
`test_cl4d_mask_routes_rowcopy_raw`、`test_cl3d_perm_row_index_direct_values`、
`test_cl4d_sliced_tail_excluded`、`test_cl4d_small_k_excluded`、
`test_cl4d_half_excluded`。这才是诚实的失败面。

**代价也要写清楚。** `op.cpp` 加了 5 行，`check_doc_anchors.py` 立刻报 **53 条
锚点漂移**（`code-walkthrough.md` 一个文件占 48 条）。按行偏移机械重映射了 68 条
显式 `op.cpp:NNN` 和 20 条裸 `:NNN`，工具复验回到 367/367（本文写完后 377/377）。

这次还撞上工具自己声明的盲区：`code-walkthrough.md` 附录 B 的"行号"列是 10 条
裸引用，每行只有一个、没有符号紧贴，属于**只做结构校验**那一桶——漂移之后结构
仍然合法（指向的还是一条 `m.def(...)`），于是**静默放行**。这 10 条只能按源码
逐个重排。教训：改高频引用的源文件，前后各跑一次
`check_doc_anchors.py`；最后一行写着 OK 不代表"仅结构校验"那 160 余条没问题。

### 5.6 再核查一轮：问题不在数字，在"校验的是哪个文件"

§5.5 之后做了一次全量自查，`check_doc_anchors.py` 报 OK，但 `code-walkthrough.md`
§4 的一批裸引用**比源码实际位置小 3-4 行**——`op.cpp` 一共长了 8 行（第 47-49 行
三行计数声明、552 和 596 各一行自增、660 起三行导出），而机械重映射只处理了
工具报出来的那 20 条，结构上仍然合法的就漏了。

更有意思的是工具为什么没报。`pick_file` 的优先级是"本段最近的符号 > 上一条显式
引用"，而 §4.2 的路由表里写着 `gather.cpp:750` 等目标行——从那一行往后，`sym_ctx`
一直是 `gather.cpp`。于是 §4.2/§4.6 里的裸 `:551`、`:126-128` 全部拿
**gather.cpp 的行号**去校验，行号落在 gather.cpp 的合法位置上，自然全绿。等于
"锚点校验器在检查错的文件"，比没检查更危险。

修法分两层。

**文档侧**：把 §4 的裸引用逐个改成源码里的实际位置（`② 校验` 549-551、
`③ sync_omp_threads` 553、`④ mask_fast_path` 555、`⑤ small_single_index` 556、
`⑥ small_multi_index` 557、`⑦ normalize_index` 558、`⑧ empty_strided` 560、
`⑨ index_into` 562，以及 `#2-#9` 六个路由条件和 §4.1-§4.8 的小节范围）。顺带把
三处注册行的 633 改成 630、附录 A 的 `index_into` 49 改成 52、
`TORCH_LIBRARY_IMPL(aten, CPU)` 的 625 改成 630、`PYBIND11_MODULE` 的 633 改成 638。

**工具侧**（`tools/check_doc_anchors.py`）四处收紧：

| 改动 | 抓的是 |
|---|---|
| 小节标题点名的文件优先于"最近的符号" | 上一节表格出现 `gather.cpp:750` 后，把 §4 的裸引用判给 gather.cpp |
| `一级小节`不点名文件时清空该上下文 | `## 9. Python 胶水层` 继承 `## 8. grain.h` 的文件 |
| `符号 :NNN` 的定义匹配从 `d-3 ≤ NNN` 收紧到 `d ≤ NNN` | 附录 A 的 `index_into` 写成 49（定义前 3 行）被放行 |
| 新增：紧邻符号必须是"函数体内"或"±3 行内被调用"；表格行按"符号 + 行号"配对 | `mask_fast_path` 写成 551、附录 B 的 `mask_fused_count` 写成 660 这类结构合法但语义错位的引用 |

同时改了三处描述性的错：本文档并行台阶的"返回行"330→333、
`override-mechanism` 的注册区间 608-612→608-613、`code-walkthrough` 里 gather.cpp
的掩码计数 1043→1042；全量计数 174→175 同步到 README/summary/porting/
override/index 等当前态声明（历史轮次里的 174 保留）。

四条新规则都做了反证：把 `④ mask_fast_path` 的行号改回 551、附录 A 的
`index_into` 改回 49、附录 B 的 `mask_fused_count` 改回 660、`single_1d` 两行
对调，工具分别报出对应的 DRIFT；改回来即 `377/377`。

---

## 6. 验证：守卫自己也要做负向对照

只验证"默认还是绿的"是不够的——那证明不了守卫有效。

> **数据轮次：2026-10-31（n20，node 5）｜当前。**

| 检查 | 期望 | 实际 |
|---|---|---|
| 默认两个套件 | 绿 | `25/25`、`15/15`，rc=0 |
| `NO_OVERRIDE=1` 两个套件 | **rc=1 + 中文提示** | 符合 |
| 骗过守卫 + 实际关覆盖 | 6 项挂、0 假通过 | `8/14`，失败的正好 6 个 `x[...]` 用例 |
| 全 15 套件 | 175/175 | 175/175 |
| `verify_against_aten.py --mode both` | 44/44 | 44/44 |
| `verify_e2e.py --mode both --threads` | 124/124 + 矩阵 OK | 符合 |
| `tools/check_doc_anchors.py` | 全过 | 改动前 361/361；§5.5 后 367/367；收紧规则后 377/377，死链 0 |

### 6.1 两个把自己骗到的坑

**坑 1：管道之后取 `$?`。** 第一版验证脚本写成

```bash
out=$(HPC_INDEX_NO_OVERRIDE=1 python $f 2>&1 | tail -2); rc=$?
```

`$?` 拿到的是 `tail` 的状态码（永远 0），于是显示"rc=0，守卫没生效"。实际守卫好好
地在退出 1。**教训：验证脚本本身也会给出假信号；关键量（退出码）要单独取、单独打印。**

**坑 2：长字符串拼接丢空格。** 提示语原本分两行写：

```python
"…覆盖关闭时"
"2 项必挂…"
```

Python 隐式拼接后是 `覆盖关闭时2 项必挂`。**教训：写中文长消息要在断行处显式留空格，
并且必须真的跑一遍看输出**——这种错编译期和静态检查都看不见。

---

## 7. 给下一个 agent 的检查表（14 步）

1. 先跑默认基线，记下总数（本例 175/175），后面一切都以它为参照。
2. **读断言**，按 §2.1 四型分类。找出"负向计数"型——它们是假通过的高危区。
3. 列变量表（§2.2），逐个标注"是否影响 + 判据来源"。**判据要读源码，不要猜。**
4. 对"影响"的变量做穷举矩阵；对"不影响"的变量，**写清是什么证据排除的**。
5. 做**负向对照**：关掉被测特性再跑。结果差异比默认结果信息量大。
6. 若某文件关掉特性后**仍然全绿**，立刻问：它有没有可能也在验证别的东西？→ 假通过。
7. 若"复现不了"，仍要给出两种结论（§2.4），并指名下一个该试的变量。
8. 修在真值源头：判据只允许有一处定义，测试引用它而不是复制它。
9. **负向断言必须配正向控制**（入口计数 / 命中计数），否则"特性没生效"和"特性
   生效但走了别的路径"在测试里长得一模一样。
10. 守卫要**大声失败**，消息里写"没守卫会发生什么"。
11. 验证守卫要做负向对照：**退出码必须是 1**，并且真的看到消息。
12. 注意验证脚本自身的假信号（管道/命令替换/`set -e`）。
13. 改了代码骨架就同步文档，最后跑 `check_doc_anchors.py` + 死链检查；改高频
    引用的源文件要前后各跑一次，并留意"仅结构校验"那一桶是真盲区。裸 `:NNN`
    尽量写成 `op.cpp:NNN` 或 `符号(:NNN)`——工具能核对语义的只有这两种写法。
14. **由成本模型导出的路由门，要在门的两侧各钉一个测试点**。这类阈值只影响
    性能、不影响正确性，写错了功能测试全绿、性能掉一截。判据变了阈值就要同步
    改文档（本例：`numel*element_size > 512KB` ↔ degradation §5.9.1 的交叉点）。

---

## 8. 这次没成立、但很诱人的假设

记录下来，避免下一个人重新走一遍：

| 假设 | 为什么诱人 | 什么证伪了它 |
|---|---|---|
| 用例硬编码 16/36/38 线程，小机器会挂 | 看起来像典型的环境耦合 | 三个 gate 全读 `at::get_num_threads()`；4 核亲和下仍返回 38 |
| 是 flaky / 多线程竞争 | 计数器和多线程测试天然让人怀疑竞争 | 单进程连跑两遍、5 次重复、逐条新进程全绿 |
| `output_layout` 挂是因为覆盖没生效 | 它全部走 `x[...]`，符合直觉 | 关掉覆盖它**仍然 25/25**，因为期望值就是 aten 真值 |
| 需要改断言 / 放宽预期让测试变绿 | 最快的止血方式 | 断言没错；错的是"没声明前置条件"，改断言会把问题藏起来 |

---

## 9. 附：完整命令

```bash
# 环境
source ~/.bashrc; conda activate torch210_env
source /home/envs/HPCKit_26.1.RC1/HPCKit/latest/setvars.sh
export LD_LIBRARY_PATH=<torch_lib>:<bisheng_lib>:$LD_LIBRARY_PATH

# 基线：全 15 套件
for t in tests/test_smoke.py tests/test_normalize.py tests/test_correctness.py \
         tests/test_errors.py tests/test_sve.py tests/test_model.py \
         tests/test_index_out.py tests/test_mask_fused.py tests/test_smallk.py \
         tests/test_compile_optin.py tests/test_strided_rows.py \
         tests/test_rowcopy_routes.py tests/test_small_multi.py tests/test_autograd.py \
         tests/test_output_layout.py; do python $t; done

# 负向对照（关键一步）
HPC_INDEX_NO_OVERRIDE=1 python tests/test_rowcopy_routes.py    # 12/14
HPC_INDEX_NO_OVERRIDE=1 python tests/test_output_layout.py     # 25/25

# 排除核数/线程数
taskset -c 190-193 python tests/test_rowcopy_routes.py
OMP_NUM_THREADS=4    python tests/test_rowcopy_routes.py

# 排除顺序/幂等
python -c "import sys; sys.path.insert(0,'tests'); import test_rowcopy_routes as m; \
  ns=[n for n in dir(m) if n.startswith('test_')]; \
  [([getattr(m,n)() for n in ns], None) for _ in range(2)]"

# 线程档穷举（1-128）
for t in 1 2 3 4 7 8 16 36 38 64 128; do
  python -c "import torch; torch.set_num_threads($t); import sys; \
    sys.path.insert(0,'tests'); import test_output_layout as m; \
    ns=[n for n in dir(m) if n.startswith('test_')]; \
    [getattr(m,n)() for n in ns]; print('nt=$t ok')"
done

# 两进程真 aten 对拍（这两个脚本自带 NO_OVERRIDE 子进程，不受新守卫影响）
python tests/verify_against_aten.py --mode both      # 44/44
python tests/verify_e2e.py --mode both --threads     # 124/124 + 线程矩阵

# 文档一致性
python tools/check_doc_anchors.py
```

---

## 10. 一句话带走

> **测试"全绿"只证明了你跑的那一次没被触发；它能证明什么，取决于它的断言在最坏
> 环境下会不会叫。** 关掉被测特性再跑一遍，是成本最低的一句话质问。
