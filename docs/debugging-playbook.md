# 排查手记与套路：测试前置条件型失败

> **案例日期**：2026-10-31（n20，node 5，608 核 / 32 NUMA，`torch 2.10.0+cpu`）
> **对象**：`tests/test_rowcopy_routes.py`（14 项）、`tests/test_output_layout.py`（25 项）
> **一句话结论**：两起"测试过不了"都不是随机 flaky，而是**测试自身对运行环境的前置
> 假设从未被声明**。修复方向不是改断言让它们变绿，而是让环境不对时**大声失败**。

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

> **数据轮次：2026-10-31（n20，node 5）｜当前。** 15 套件 **174/174**。此后所有
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
| 硬件 ISA（SVE512） | **是** | `op.cpp:122` |

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

报告方说"可能无法全部通过"，所以先怀疑**环境差异**，不怀疑断言写错。

### 3.2 假设 A：机器核数/线程数不够 —— 证伪

第一反应是"用例硬编码了 16/36/38 线程，小机器上 `set_num_threads` 会被截断"。
这个假设很诱人，但**读一下判据就死了**：

| 判据 | 位置 | 读的是什么 |
|---|---|---|
| 多索引 raw 路由 | `op.cpp:101` | `at::get_num_threads() >= 16` |
| 并行台阶 | `op.cpp:327`（函数）、`:330`（返回） | `max(16384, 512 * at::get_num_threads())` |
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

### 3.5 还有一条真前置：SVE512 硬件

```cpp
// op.cpp:122
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
| 关覆盖时 | 12/14，2 fail + 4 假通过 | **25/25 全绿** |
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
        "2 项必挂、4 项假通过（HPC_INDEX_NO_OVERRIDE=1？）")

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
   `2 项必挂、4 项假通过` / `25 项全绿但不验证任何东西` 才能让人一眼判断"我是不是
   中招了"，也把本节的结论固化进代码。
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

---

## 6. 验证：守卫自己也要做负向对照

只验证"默认还是绿的"是不够的——那证明不了守卫有效。

> **数据轮次：2026-10-31（n20，node 5）｜当前。**

| 检查 | 期望 | 实际 |
|---|---|---|
| 默认两个套件 | 绿 | `25/25`、`14/14`，rc=0 |
| `NO_OVERRIDE=1` 两个套件 | **rc=1 + 中文提示** | 符合 |
| 全 15 套件 | 174/174 | 174/174 |
| `verify_against_aten.py --mode both` | 44/44 | 44/44 |
| `verify_e2e.py --mode both --threads` | 124/124 + 矩阵 OK | 符合 |
| `tools/check_doc_anchors.py` | 全过 | 改动前 361/361；写入本文档后 367/367，死链 0 |

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

## 7. 给下一个 agent 的检查表

1. 先跑默认基线，记下总数（本例 174/174），后面一切都以它为参照。
2. **读断言**，按 §2.1 四型分类。找出"负向计数"型——它们是假通过的高危区。
3. 列变量表（§2.2），逐个标注"是否影响 + 判据来源"。**判据要读源码，不要猜。**
4. 对"影响"的变量做穷举矩阵；对"不影响"的变量，**写清是什么证据排除的**。
5. 做**负向对照**：关掉被测特性再跑。结果差异比默认结果信息量大。
6. 若某文件关掉特性后**仍然全绿**，立刻问：它有没有可能也在验证别的东西？→ 假通过。
7. 若"复现不了"，仍要给出两种结论（§2.4），并指名下一个该试的变量。
8. 修在真值源头：判据只允许有一处定义，测试引用它而不是复制它。
9. 守卫要**大声失败**，消息里写"没守卫会发生什么"。
10. 验证守卫要做负向对照：**退出码必须是 1**，并且真的看到消息。
11. 注意验证脚本自身的假信号（管道/命令替换/`set -e`）。
12. 改了代码骨架就同步文档，最后跑 `check_doc_anchors.py` + 死链检查。

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
