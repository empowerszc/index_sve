# docs 可读性重排 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `index_sve` 的 `README.md` 与 `docs/` 正文从“查表手册”改造成“能顺着读、也能跳着查”的文档，且不丢失技术事实与历史可追溯性。

**Architecture:** 以设计文档的写作规范为统一施工图，按“入口 → 总览 → 性能 → 架构/内核 → 专题 → 编年史 → reference”分批改造。每批只动一组文档，逐批用链接检查、锚点检查、数字抽查和人工通读验收，避免一次性大改引入事实漂移。

**Tech Stack:** Markdown、Python 3（链接检查脚本）、git。

**Spec:** `docs/superpowers/specs/2026-10-29-docs-readability-design.md`

## Global Constraints

- 当前数字唯一出处：`benchmarks/RESULTS.md` §2.9；正文只写代表值 + 指针，不复制全矩阵。
- 代表值的“摘要位”固定为 `README.md`、`summary.md`、`performance.md` 三篇；附录不计。
  `output-layout.md` §8 的 R10 前后对照表是该专项的实测细节（`1.45-2.54x` 的 owner），
  不算摘要复制；其余正文只写结论与指针。
- 不改任何技术结论、代码行为、测试；不改 `benchmarks/RESULTS.md`。
- 术语统一：主语用「本算子」，原生实现用「aten」，`ratio`→「加速比」，
  `line amplification`→「行放大」。**只改散文名词，不改标识符**：`hpc_index`、
  `_hpc_index`、`torch.ops.hpc_index.*`、`HPC_INDEX_*`、`MODE=hpc`，以及复现真实
  输出用的 `aten/hpc` 列名一律保持原样。
- 小节编号默认不改；必须移动时在原位置留「原 §x.y，见 §a.b」指针，并全仓 grep 更新引用。
- 正文结构：每节第一句先给结论，再展开；一段只讲一件事，超过 4 行考虑拆列表/小表；
  表格前必须有一句“怎么读”。
- 慢场景家族统一写「五个可复现家族 + 噪声级单格（另计）」，不用「六个家族」。
- 代号首次出现给白话或链到 `glossary.md`；多义代号（如 `M1`）首次出现时写明是
  「机制 M1」还是「负载 M1」。
- `R10` 不重命名；在 glossary 加一行说明它与路线图 `R1-R9` 无关。
- 每篇正文加统一导读块（`reference/` 两篇除外）；超过 300 行的文档必须加 `## 目录`，
  包括 `summary.md`、`status.md`、`fallback-analysis.md`。
- 交叉引用：正文少用打断式括号；跨篇引用写「文档名 + §x.y」，不写裸 `§x.y`（同篇除外）。
- 格式：代码块标语言；标题最多到 `###`，不再向下嵌套；表格必须有表头 + 分隔行。
- 每个实测表前保留「数据轮次」行。
- 完成后跑 `python3 tools/check_doc_anchors.py`，必须通过。

## Review Focus

1. **首屏导读缺失或叠了两块**：新读者打开任一篇，第一屏应能回答“这篇讲什么/读完能回答什么”。`status.md` 复用现有“30 秒现状”，不得再叠第二块。**Pinned by:** Task 3 Step 3（`30 秒现状` 恰 1 处）与 Task 9 Step 2（导读块覆盖）。
2. **历史数据移动后 `§x.y` 引用失效**：`§5.8`、`§5.9`、`§2.9`、`§19` 被大量跨篇引用；移动内容必须留指针并 grep 全仓。**Pinned by:** Task 4 Step 4 与 Task 9 Step 3。
3. **数字被改错或与 RESULTS §2.9.1 不一致**：改写只搬运原文数字，不新造；验收抽查 10 处。**Pinned by:** Task 9 Step 4。
4. **术语/代号首次出现无解释**：每篇首次出现的 S1/P1/M1/T/R/A 等必须有白话或 glossary 链接；`R10` 与 `R1-R9` 必须区分。**Pinned by:** Task 9 Step 6 的人工首屏检查。
5. **链接/锚点失效**：新增链接检查脚本必须覆盖相对路径与 `#anchor`；`reference/aten_index_primer.md` 的 8 个跨文件锚点不得因改标题失效。**Pinned by:** Task 1 Step 4 自测与 Task 9 Step 1。

---

### Task 1: 建立分支与文档链接检查工具

**Files:**
- Create: `tools/check_doc_links.py`
- Modify: 无（仅新增工具）
- Test: 运行 `python3 tools/check_doc_links.py`

**Interfaces:**
- Consumes: 仓库现有 `.md` 文件。
- Produces: `tools/check_doc_links.py`，可执行，退出码 0 = 全部有效，1 = 有失效链接；输出每行 `文件:行号: 失效目标`。

- [ ] **Step 1: 建工作分支**

```bash
cd /Volumes/seagate/200Code/bytime/20261029_index_sve_920F/index_sve
git switch -c docs/readability-20261029
```

- [ ] **Step 2: 写 `tools/check_doc_links.py`**

行为规范（实现者写 body）：
- 递归遍历仓库内所有 `.md`（跳过 `.git/`、`node_modules/`、`agent_space/`）。
- **先剔除 fenced code block 和 inline code**，再匹配链接；否则设计文档模板里的
  示例链接会被误报。fence 要支持嵌套与缩进：闭合串必须不短于开启串（例如 ```` 包裹
  ```），且允许行首空格（列表内的 fence）；不要用简单的“遇到 ``` 就翻转”。
- 匹配 Markdown 链接 `[text](target)`；跳过 `http://`、`https://`、`mailto:`、
  `tel:`、纯 `#` 占位。目标两端的 `<...>` 去掉。
- `target` 形如 `path#anchor` 或 `#anchor`：
  - 先解析相对路径（相对当前文件目录，`os.path.normpath`），检查目标文件存在；
  - 若带 `#anchor`，用 GitHub 风格 slug 检查目标文件有对应标题：
    `lower()` → 删除除字母/数字/中文/空格/`-`/`_` 外的字符 → 空格转 `-`
    （**不合并连续 `-`**，如 `已知限制 / 注意事项` → `已知限制--注意事项`）；
    重复标题按 GitHub 规则追加 `-1`、`-2`（按文档顺序）。
- 输出 `相对路径:行号: 失效目标`；有失效则 `sys.exit(1)`；全通过打印
  `OK: N files, 0 broken links`。
- 只依赖标准库。若脚本报某个实际上在 GitHub 有效的锚点，**先修 slug 函数，
  不要改文档**。

- [ ] **Step 3: 运行并记录基线**

Run: `python3 tools/check_doc_links.py`
Expected: 输出 `OK: N files, 0 broken links`。若列出失效链接，在本步骤内修掉
（只修路径/锚点，不改内容）；若报的是 GitHub 上实际有效的锚点，先修 slug 函数。
反复重跑至 0，把最终 `N` 记在 commit message 里。

- [ ] **Step 4: 自测（确认既要报错、也要跳过代码块）**

1. 在 `docs/index.md` 末尾加一行 `[bad](nope.md)`，运行脚本。
   Expected: 退出码 1，且打印 `docs/index.md:...: nope.md`。
2. 在同一位置再加一个 fenced block：

   ````
   ```text
   [not-a-link](also-nope.md)
   ```
   ````

   运行脚本。Expected: 仍只报 `nope.md`，**不报** `also-nope.md`。
3. 删除上面两处临时内容，重跑。Expected: `OK: N files, 0 broken links`。

- [ ] **Step 5: Commit**

```bash
git add tools/check_doc_links.py docs/superpowers/
git commit -m "docs: add markdown link checker and readability plan"
```

---

### Task 2: 入口层（README / index / glossary）

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/glossary.md`

**Interfaces:**
- Consumes: 设计文档 §4.1 导读块模板、§4.3 术语规则。
- Produces: 全项目统一的导读块三行格式；`docs/index.md` 的 6 个目标分组（先用/评估/读码/改内核/集成/移植）及锚点；glossary 的 `R10` 条目。

- [ ] **Step 1: 改造 `README.md`**

- 标题后加导读块（这篇讲什么 / 读完能回答 / 本篇代号），放在现有 `## 目录` 之前。
- “基准结论（摘要）”表只留代表行（S1、M1、shortq、R10、仍 <1x 一句），其余改为指针。
- 把「仍 <1x 的六个家族」改为「五个可复现家族 + 噪声级单格（另计）」。
- 统一术语：散文里的 `hpc`→「本算子」、`ratio`→「加速比」；首次出现解释 `B/K/A`。
  `hpc_index`、`HPC_INDEX_*`、`torch.ops.hpc_index.*` 不动。
- 保留快速开始、已知限制、测试命令。

- [ ] **Step 2: 重写 `docs/index.md` 入口**

- 加导读块。
- 顶部改成“按目标选阅读路径”：先用 / 评估性能 / 读源码 / 改内核 / 集成排障 / 移植。
- 原 25 行索引表保留，但按“入门 / 架构与内核 / 性能 / 集成与移植 / 历史”分组。
- 索引表里 R10 的 `1.45-2.54x` 改为指针（「R10 直写，见 output-layout §8 /
  RESULTS §2.9」），因为 `index.md` 不是摘要位。
- 保留“维护约定”和“速览”图（可精简，但不得删维护规则）。

- [ ] **Step 3: 补充 `docs/glossary.md`**

- 顶部加一行醒目提示：`M1` 机制 vs 负载两套含义。
- 加 `R10 = 当前轮次（2026-10-29 布局直写轮），与路线图优先级 R1-R9 无关`。
- 加“代号在文档中首次出现应给白话或链到本表”的约定。

- [ ] **Step 4: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `grep -c "这篇讲什么" README.md docs/index.md docs/glossary.md`
Expected: 每个文件 1。

Run: `grep -n "六个家族" README.md docs/index.md docs/glossary.md`
Expected: 无输出（README 若有，改为“五个家族 + 噪声级单格”）。

Run: `grep -n "1.45-2.54" docs/index.md`
Expected: 无输出（已改为指针）。

- [ ] **Step 5: Commit**

```bash
git add README.md docs/index.md docs/glossary.md
git commit -m "docs: add reading guides to entry docs"
```

---

### Task 3: 总览层（summary / status）

**Files:**
- Modify: `docs/summary.md`
- Modify: `docs/status.md`

**Interfaces:**
- Consumes: Task 2 的导读块格式与 glossary 链接。
- Produces: `status.md` 的“30 秒现状”作为导读、`§5.1` 当前结论只留摘要 + 指针；`summary.md` 校正后的测试数与统一术语。

- [ ] **Step 1: 改造 `docs/summary.md`**

- 加导读块与 `## 目录`（480 行，必须加）。
- 保留 10-15 分钟主线。
- 拆长句；把嵌套括号信息移到小表或“延伸阅读”。
- 校正过时测试数：`143 项测试` → `174 项测试`（原文写法不是 `143/143`）。
- 历史措辞收敛为“历史值见 RESULTS”，正文只留 R10。
- `summary.md` 是摘要位之一，S1/P1、M1、shortq、R10 代表值可以保留。

- [ ] **Step 2: 改造 `docs/status.md`**

- 加 `## 目录`（370 行，必须加）。
- 把现有“30 秒现状”整理成导读块（不新增第二块）。
- `§5.1` 只留当前结论 + 指针；正文里的 `2.98`、`1.45-2.54`、`12.7-13.6`
  改为「代表值见 RESULTS §2.9.1/§2.9.4」或定性结论，不复制 performance 全表。
- `§5.2` 历史逐轮内容移到文末“附录：历史结论”，正文留指针；保留 `§2` 变更日志（快照职责）。
- 拆长句；保留路由树与演进方向。

- [ ] **Step 3: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `grep -n "143/143\|143 项\|六个家族" docs/summary.md docs/status.md`
Expected: 无输出。

Run: `grep -c "^## 目录" docs/summary.md docs/status.md`
Expected: 两个文件各 1。

Run: `awk '/^## 附录/{exit} /2\.98|1\.45-2\.54|12\.7-13\.6/{print FILENAME":"NR": "$0}' docs/status.md`
Expected: 无输出（正文章节已改为指针；历史值只能在“附录”之后）。

Run: `grep -n "30 秒现状" docs/status.md`
Expected: 1 处（导读位置）。

- [ ] **Step 4: Commit**

```bash
git add docs/summary.md docs/status.md
git commit -m "docs: streamline overview docs and fix stale counts"
```

---

### Task 4: 性能组（performance / degradation / fallback）

**Files:**
- Modify: `docs/performance.md`
- Modify: `docs/degradation-analysis.md`
- Modify: `docs/fallback-analysis.md`

**Interfaces:**
- Consumes: Task 2/3 的导读块与“当前/历史”约定；`benchmarks/RESULTS.md` §2.9.1 作为唯一数字口径。
- Produces: 三篇的“正文当前 + 文末历史附录”结构；统一的“五个可复现家族 + 噪声级单格”表述。

- [ ] **Step 1: 改造 `docs/performance.md`**

- 加导读块与 `## 目录`。
- 修正开头“§2.4 T13 全量矩阵”旧句，改为“当前见 RESULTS §2.9.1”。
- 正文只留 R10 结论；T11/T13/T14/T15 表移到文末“附录：历史数据”，每张表前标数据轮次。
- 保留 §4.3 胜出根因（与时点无关）。
- `performance.md` 是摘要位之一，代表值可保留；但不要复制 RESULTS 的全矩阵。

- [ ] **Step 2: 改造 `docs/degradation-analysis.md`**

- 加导读块与 `## 目录`（823 行，必须加）。
- 正文保留：当前 <1x 家族（R10）+ §5.9 微架构根因 + §6 线程扩展机制。
- T11/T13/T15 全量矩阵与逐 case 历史数字移到文末“附录：历史矩阵”。
- 小节编号默认保持；移动处留“原 §x.y，见 §a.b”指针；§5.9、§6 不得改名。
- 拆长句；表格前补“怎么读”，表前标数据轮次。

- [ ] **Step 3: 改造 `docs/fallback-analysis.md`**

- 加导读块与 `## 目录`（395 行，必须加）。
- 正文保留：四个机制 + 当前 <1x 家族 + 四层“不 fallback”论证。
- 历史逐场景（§3.1-§3.8 的旧数据）移到“附录：历史逐场景”。
- 正文里的 `2.98`、`2.55-4.16` 改为指针（fallback 不是摘要位）；附录可保留历史值。
- 统一“五个可复现家族 + 噪声级单格”。

- [ ] **Step 4: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `grep -rn "六个家族" docs/performance.md docs/degradation-analysis.md docs/fallback-analysis.md`
Expected: 无输出。

Run: `grep -rn "§2.4 T13" docs/performance.md`
Expected: 无输出。

Run: `grep -c "^## 目录" docs/performance.md docs/degradation-analysis.md docs/fallback-analysis.md`
Expected: 三个文件各 1。

Run: `awk '/^## 附录/{exit} /2\.98|2\.55-4\.16/{print FILENAME":"NR": "$0}' docs/fallback-analysis.md`
Expected: 无输出（正文已改为指针）。

Run: `grep -rn "§5\.9\|§5\.8" README.md docs/*.md`
Expected: 每条引用都能在 `docs/degradation-analysis.md` 找到对应小节；若已移动，原位置有「原 §x.y」指针。人工逐条确认。

- [ ] **Step 5: Commit**

```bash
git add docs/performance.md docs/degradation-analysis.md docs/fallback-analysis.md
git commit -m "docs: separate current results from historical matrices"
```

---

### Task 5: 架构 / 走读 / 内核（architecture / code-walkthrough / kernels）

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/code-walkthrough.md`
- Modify: `docs/kernels.md`

**Interfaces:**
- Consumes: Task 2 术语规则；`docs/glossary.md` 的代号表。
- Produces: `architecture.md` 的“先例子后三段流水线”结构；`kernels.md` 的“选择表 + 每内核条件/实现/实测”结构。

- [ ] **Step 1: 改造 `docs/architecture.md`**

- 加导读块。
- 开头先用一个 `x[i]` 走一遍 normalize→linearize→gather：
  `x` 形状 `(100000, 64)` f32，`idx` 是长度 1M 的 int64，输出 `(1M, 64)`，
  对应 **B=1、K=1M、A=64**。先给这个数字例子，再给三段流水线定义。
- 把 §1 的超长伪码树拆成“主干 6 行 + 分支表”；术语首现给白话。
- 保留 §2-§8 的技术内容与调用图。

- [ ] **Step 2: 改造 `docs/code-walkthrough.md`**

- 加目录与导读块。
- 把“行号”与“结论”分列（表格化）；拆长句。
- 保留按文件阅读顺序、路由表、十个不变量、附录。

- [ ] **Step 3: 改造 `docs/kernels.md`**

- 加导读块与目录。
- 先给“一张选择表 + B/A/K 三个量”，再进各内核。
- 每个内核统一小标题：条件 / 实现 / 实测；历史数字标轮次。
- 保留 §8 防回归清单与 §9/§10。

- [ ] **Step 4: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `python3 tools/check_doc_anchors.py`
Expected: 全部锚点通过（docs 编辑不应改源码行号）。

Run: `grep -n "这篇讲什么" docs/architecture.md docs/code-walkthrough.md docs/kernels.md`
Expected: 每篇 1 处。

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md docs/code-walkthrough.md docs/kernels.md
git commit -m "docs: give architecture and kernel docs a narrative entry"
```

---

### Task 6: 专题组（semantics / output-layout / integration / torch-compile / override-mechanism / porting / benchmarks）

**Files:**
- Modify: `docs/semantics.md`
- Modify: `docs/output-layout.md`
- Modify: `docs/integration.md`
- Modify: `docs/torch-compile.md`
- Modify: `docs/override-mechanism.md`
- Modify: `docs/porting.md`
- Modify: `docs/benchmarks.md`

**Interfaces:**
- Consumes: Task 2 导读块与术语规则。
- Produces: 各篇统一导读块；`output-layout.md` 的“现象→根因→修复→移植→性能”顺序。

- [ ] **Step 1: 轻改六篇结构清晰文档**

`semantics.md`、`integration.md`、`torch-compile.md`、`override-mechanism.md`、`porting.md`、`benchmarks.md`：
- 加导读块；拆明显长句；表格前补“怎么读”。
- 不改技术内容、代码块、错误文案；不改任何标识符（见 Global Constraints）。
- `integration.md` 额外两处：把「六个家族」改为「五个可复现家族 + 噪声级单格」；
  正文里的 `2.98` 改为指针（integration 不是摘要位）。

- [ ] **Step 2: 重排 `docs/output-layout.md`**

- 加导读块。
- 顺序改为“现象 → 根因 → 修复 → 移植 → 性能”。
- R10 直写细节留在节末；保留 §5 移植指南与 §8 实测表
  （§8 是 `1.45-2.54x` 的 owner，不算摘要复制）。
- 小节编号保持，移动处留指针。

- [ ] **Step 3: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `for f in docs/semantics.md docs/output-layout.md docs/integration.md docs/torch-compile.md docs/override-mechanism.md docs/porting.md docs/benchmarks.md; do grep -q "这篇讲什么" "$f" || echo "MISSING $f"; done`
Expected: 无输出。

Run: `grep -n "六个家族\|2\.98" docs/integration.md`
Expected: 无输出（已改为“五个家族 + 噪声级单格”和 RESULTS 指针）。

- [ ] **Step 4: Commit**

```bash
git add docs/semantics.md docs/output-layout.md docs/integration.md docs/torch-compile.md docs/override-mechanism.md docs/porting.md docs/benchmarks.md
git commit -m "docs: add guides to topic docs and reorder output-layout"
```

---

### Task 7: 编年史（optimizations）

**Files:**
- Modify: `docs/optimizations.md`

**Interfaces:**
- Consumes: 现有 T0-T15 记录；设计文档 §4.2。
- Produces: 统一的“问题 / 改动 / 实测 / 验证”四段式；目录。

- [ ] **Step 1: 改造 `docs/optimizations.md`**

- 加导读块与目录。
- 每轮（T5-T15）统一小标题：问题 / 改动 / 实测 / 验证。
- 拆长句；把括号里的补充信息移到表格“注”列或脚注。
- 不改任何轮次的事实、commit、数字；历史表保留并标轮次。

- [ ] **Step 2: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links。

Run: `grep -n "^## 目录\|这篇讲什么" docs/optimizations.md`
Expected: 两行都存在。

- [ ] **Step 3: Commit**

```bash
git add docs/optimizations.md
git commit -m "docs: make optimization chronicle easier to scan"
```

---

### Task 8: reference 格式与锚点

**Files:**
- Modify: `docs/reference/aten_index_primer.md`
- Modify: `docs/reference/aten_index_aarch64_analysis.md`

**Interfaces:**
- Consumes: Task 1 链接检查脚本。
- Produces: 两篇 reference 的目录/格式统一；不改技术内容，不套导读块。

- [ ] **Step 1: 统一目录与格式**

- 校正目录与正文标题一致；表格补表头/分隔行；代码块标语言。
- 不新增导读块；不改章节顺序与技术内容。
- 若必须调整标题，先 grep 全仓对该标题锚点的引用并同步更新。

- [ ] **Step 2: 验证**

Run: `python3 tools/check_doc_links.py`
Expected: 0 broken links（尤其检查 primer → aarch64 的 8 个锚点）。

- [ ] **Step 3: Commit**

```bash
git add docs/reference/aten_index_primer.md docs/reference/aten_index_aarch64_analysis.md
git commit -m "docs: normalize reference formatting and anchors"
```

---

### Task 9: 全量验收

**Files:**
- Modify: 仅在修复验收发现的问题时改动相关文档。

**Interfaces:**
- Consumes: Task 1-8 的全部改动。
- Produces: 通过全部验收标准的最终文档集。

- [ ] **Step 1: 链接与锚点**

Run: `python3 tools/check_doc_links.py`
Expected: `OK: N files, 0 broken links`。

Run: `python3 tools/check_doc_anchors.py`
Expected: 全部通过。

- [ ] **Step 2: 导读块与目录覆盖**

Run:
```bash
for f in README.md docs/*.md; do
  grep -q "这篇讲什么" "$f" || echo "MISSING GUIDE $f"
  if [ "$(wc -l < "$f")" -gt 300 ] && ! grep -q "^## 目录" "$f"; then
    echo "MISSING TOC $f"
  fi
done
```
Expected: 无输出（`docs/reference/`、`docs/archive/` 不在 `docs/*.md` 范围）。

Run: `grep -rn "六个家族\|§2.4 T13" README.md docs/*.md`
Expected: 无输出。

- [ ] **Step 3: `§x.y` 引用完整性**

Run: `grep -rn "§[0-9]" README.md docs/*.md | wc -l`
记录命中数。逐条确认目标小节仍存在，重点核对：
- `degradation-analysis.md` §5.8 / §5.9 / §6；
- `RESULTS.md` §2.9 / §2.9.1 / §2.9.4；
- `performance.md` §4.3；
- `fallback-analysis.md` §4.1-§4.7；
- `status.md` §3。

Run: `grep -rn "原 §" docs/*.md`
Expected: 每个被移动的小节都有一行「原 §x.y，见 §a.b」指针。

- [ ] **Step 4: 数字抽查（10 处）与重复台账**

对照 `benchmarks/RESULTS.md` §2.9.1 / §2.9.4，抽查：
1. README 的 S1/P1 代表值
2. summary §6.1 的 S1/P1
3. performance §3 的 S1/P1
4. status §5.1 的 S1/P1（应为指针）
5. M1 12.7-13.6x
6. shortq 2.55-4.16x
7. cl4d 1.45-2.54x
8. 空 mask 0.42-0.66x
9. 1B 0.91-0.99x
10. complex 0.92-0.95x

Run:
```bash
for n in "2.98" "1.45-2.54" "12.7-13.6" "2.55-4.16"; do
  echo "-- $n --"; grep -rln "$n" README.md docs/*.md
done
```
Expected: 每个数字只出现在 `README.md`、`docs/summary.md`、`docs/performance.md`
（`1.45-2.54` 另有 `output-layout.md` §8 owner 例外）；其余出现应为附录或指针。
发现不一致先改文档、不改 RESULTS。

- [ ] **Step 5: 根因论证未删**

Run: `git diff main --stat -- README.md docs/`
Run: `git diff main -U0 -- docs/ | grep '^-## '`
Expected: 被删的 `##` 标题都能对应到“移入附录”或“合并”，且原位置有指针。逐条人工确认：
- degradation §5.9（64B 行放大 / DRAM 延迟 / 4B gather 发射下限）仍在；
- degradation §6 线程扩展机制仍在；
- fallback §4.1-§4.7 四层论证仍在；
- optimizations 每轮 T5-T15 的事实、commit、数字仍在；
- `archive/` 未删、未改内容（仅允许修路径/编号指针）。

- [ ] **Step 6: 人工通读三篇重点**

依次读 `docs/architecture.md`、`docs/performance.md`、`docs/degradation-analysis.md`
首屏 + 目录 + 第一节，确认：
- 首屏能回答“这篇讲什么/读完能回答什么”；
- 正文不再把历史矩阵与当前结论混排；
- 长句和括号明显减少；
- 首次出现的代号有白话或 glossary 链接，`R10` 与 `R1-R9` 不混。

- [ ] **Step 7: 最终提交**

```bash
git add -A
git commit -m "docs: final readability pass and verification"
```

- [ ] **Step 8: 交付**

向用户报告：改了哪些文件、每批 commit、验收结果（含 `N`、抽查的 10 个数、台账例外）；
由用户决定合并方式（当前分支 `docs/readability-20261029`）。
