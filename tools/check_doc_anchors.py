#!/usr/bin/env python3
"""校验文档里的行号锚点是否还落在正确的源码位置。

三种引用都会被检查：
  1. 显式文件：``op.cpp:543``、``gather.cpp:981``、``op.cpp:49-181``
  2. 符号限定：``rowcopy_sve:351``——文件与归属函数由该符号确定
  3. 裸行号：``(:566)``、``:326-327``——文件由同行符号 / 本段最近的符号 /
     小节标题 / 上一条显式引用推断；若紧贴 ``符号 :NNN``（附录 A 的速查表写法），
     还会校验该行号落在那个函数的函数体内

判据（任一不满足即报漂移）：
  1. 行号在该文件范围内；
  2. 区间起点不是空行 / 只有括号 / ``return 0;`` 之类残句（终点只要求非空，
     因为函数/代码块以 ``}`` 收尾是正常的）；
  3. 起点行不是注释——除非文档行明确写了"注释"，或引用的是区间（注释常是
     代码块的引导行）；
  4. 有归属函数时，引用的两端都必须落在该函数的函数体
     （定义行 → 下一个定义行）内；
  5. 单引用行里用反引号点名的符号，必须出现在 `NNN±3` 行窗口内；
  6. 引用紧邻左边点名的符号（``f(:NNN)`` / ``f :NNN``）若在该文件里有定义，
     则 NNN 必须落在 f 的函数体内，或 f 在 `NNN±3` 行内被调用——这条专抓
     "把调用点写成别处的行号"（`mask_fast_path(:551)` 这种，结构上完全合法）；
  7. 表格行 ``| 符号 | :NNN |`` 按规则 6 校验；一行 N 个引用配 N 个反引号符号
     时按出现顺序配对（附录 B 的 `a` / `b` | :x/:y）。

已知边界（刻意不装作能查）：
  - 一行写多个引用、不是表格行、也没有 ``符号 :NNN`` 紧贴写法时，只做结构校验
    且归属符号说不清（例如 §4 的步骤清单 ``① f(:N)``），函数内部小幅位移不会
    报错；这类引用计入输出里的"仅结构校验"条数，改源码后要人工按源码核一遍。
  - 宏（``TORCH_LIBRARY`` 等）与 ``struct``/变量定义不参与规则 6，带模板参数的
    调用按 ``f<`` 识别。
  - ``docs/archive/``（冻结快照）与 ``docs/reference/``（aten 源码导读）豁免。

用途：源码新增/删除几行，文档里所有"拿着代码读"的行号就整体错位。改完源码跑一遍，
按输出的建议行号修正文档再提交。

用法：
  python tools/check_doc_anchors.py [文档...]   # 默认 README + docs/**/*.md + benchmarks/*.md
退出码 0 = 通过；1 = 有漂移。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "index_sve")
TOL = 3

SRC_FILES = ["op.cpp", "normalize.cpp", "normalize.h", "linearize.cpp", "gather.cpp", "grain.h"]
EXPLICIT = re.compile(r"\b((?:op|normalize|linearize|gather|grain)\.(?:cpp|h)):(\d+)(?:-(\d+))?")
SYMBOL_REF = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_]\w{3,}):(\d+)(?:-(\d+))?")
BARE = re.compile(r"(?<![A-Za-z0-9_.:]):(\d+)(?:-(\d+))?")
OTHER_FILE = re.compile(r"\b\w+\.(?:cpp|h|hpp|cc|py|md|txt):\d+(?:-\d+)?")
HEADING_FILE = re.compile(r"^#{1,6}.*?\b((?:op|normalize|linearize|gather|grain)\.(?:cpp|h))\b")
BOLD_FILE = re.compile(r"\*\*((?:op|normalize|linearize|gather|grain)\.(?:cpp|h))\*\*")
BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
DEF = re.compile(r"^[A-Za-z_][\w:<>,\s\*&]*?\b(\w+)\s*\(")
PYBIND_DEF = re.compile(r'm\.def\("(\w+)"')
STRUCTURAL = re.compile(r"^(}|\};|\}\)|return 0;|#endif|#else|else \{|\s*)$")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def load_sources():
    return {f: open(os.path.join(SRC_DIR, f), encoding="utf-8").read().splitlines()
            for f in SRC_FILES if os.path.exists(os.path.join(SRC_DIR, f))}


def build_symbols(src):
    """符号 → 定义所在文件集合 / 定义行号列表（用于"引用是否落在该函数体内"）。"""
    sym, defs = {}, {}
    for f, lines in src.items():
        for i, line in enumerate(lines, 1):
            m = DEF.match(line) or PYBIND_DEF.search(line)
            if m:
                name = m.group(1)
                sym.setdefault(name, set()).add(f)
                defs.setdefault((f, name), []).append(i)
    return sym, defs


def inside_function(f, defs, name, lo, src):
    """lo 是否落在 name 的某个定义体（定义行 → 下一个定义行）内。"""
    starts = sorted(d for (ff, _), ds in defs.items() if ff == f for d in ds)
    for d in defs.get((f, name), []):
        end = next((x for x in starts if x > d), len(src[f]) + 1)
        if d <= lo < end:
            return True
    return False


def unique_symbols(line, sym):
    return [n for n in BACKTICK.findall(line) + IDENT.findall(line) if n in sym]


def unique_symbol_in(line, sym):
    """行内第一个只在一个文件里定义的符号 → 该文件。反引号里的符号优先。"""
    for name in BACKTICK.findall(line) + IDENT.findall(line):
        files = sym.get(name)
        if files and len(files) == 1:
            return next(iter(files))
    return None


def pick_file(line, ref_start, cur_file, sec_file, sym_ctx, sym):
    """为裸行号选目标文件。

    判据从强到弱：整行只指向一个文件 → 紧邻左边的符号 → 行首反引号主语 →
    小节标题点名的文件 → 本段最近的符号 → 上一条显式引用。

    小节标题点名文件时优先用它：`## 4. op.cpp 逐段精读` 里的裸 `:NNN` 说的是
    op.cpp，哪怕前一行表格刚出现过 `gather.cpp:750`（那会把"最近的符号"污染成
    gather.cpp，曾让 §4 的裸引用拿 gather.cpp 的行号校验，静默通过）。
    """
    cands = {f for name in unique_symbols(line, sym) for f in sym[name]}
    if len(cands) == 1:
        return cands.pop()
    near = IDENT.findall(line[max(0, ref_start - 15):ref_start])
    for name in reversed(near):
        files = sym.get(name)
        if files and len(files) == 1:
            return next(iter(files))
    for name in BACKTICK.findall(line):
        files = sym.get(name)
        if files and len(files) == 1:
            return next(iter(files))
    return sec_file or sym_ctx or cur_file


def adjacent_name(line, pos):
    """引用紧邻左边的符号名（`to_vec(:548)` / `index_into :52` 两种写法）。"""
    m = re.search(r"([A-Za-z_]\w*)\s*[`（(]*\s*$", line[:pos])
    return m.group(1) if m else None


def table_symbol(line, pos, refs):
    """表格行：`符号 | :NNN`。取引用前最后一个反引号符号；一行 N 个引用配 N 个
    符号时按出现顺序配对（附录 B 的 `a` / `b` | :x/:y 写法）。"""
    if not line.lstrip().startswith("|"):
        return None
    ticks = re.findall(r"`([A-Za-z_]\w*)`", line[:pos])
    if len(ticks) == 1:
        return ticks[0]
    if len(ticks) == len(refs):
        return ticks[[r[5] for r in refs].index(pos)]
    return None


def start_ok(body, n, hi, doc_line):
    actual = body[n - 1].strip()
    if STRUCTURAL.match(actual) or len(actual) < 3:
        return f"第 {n} 行是结构性残句 {actual!r}"
    if actual.startswith("//") and hi == n and "注释" not in doc_line:
        return f"第 {n} 行是注释（文档未声明）"
    return None


def check(docs):
    src = load_sources()
    sym, defs = build_symbols(src)
    failures, skipped = [], []
    counts = {"explicit": 0, "symbol": 0, "bare": 0, "body": 0, "weak": 0, "struct": 0}

    for doc in docs:
        if os.sep + "archive" + os.sep in doc:
            continue  # 归档快照冻结，行号随当时源码，不追新
        if os.sep + "reference" + os.sep in doc:
            continue  # aten 源码导读，行号不属于本仓库
        lines = open(doc, encoding="utf-8").read().splitlines()
        cur_file = None
        sec_file = None
        sym_ctx = None
        for i, line in enumerate(lines, 1):
            m = HEADING_FILE.match(line) or BOLD_FILE.search(line)
            if m:
                cur_file, sec_file, sym_ctx = m.group(1), m.group(1), None
            elif line.startswith("## "):
                sec_file = None  # 新的一级小节不点名文件 → 不再沿用上一节的文件
            line_sym = unique_symbol_in(line, sym)
            if line_sym:
                sym_ctx = line_sym
            refs = []
            for mm in EXPLICIT.finditer(line):
                refs.append((mm.group(1), int(mm.group(2)), int(mm.group(3) or mm.group(2)),
                             mm.group(0), "explicit", mm.start(), []))
                cur_file = mm.group(1)
                sym_ctx = None  # 显式引用是最新的强上下文，别让更早的符号盖过它
            masked = OTHER_FILE.sub(" ", EXPLICIT.sub(" ", line))
            for mm in SYMBOL_REF.finditer(masked):
                files = sym.get(mm.group(1))
                if not files:
                    continue  # 不是已知符号（例如 host:port），不当行号
                f = next(iter(files)) if len(files) == 1 else cur_file
                if f is None:
                    skipped.append((doc, i, mm.group(0), "符号在多文件定义"))
                    continue
                refs.append((f, int(mm.group(2)), int(mm.group(3) or mm.group(2)),
                             mm.group(0), "symbol", mm.start(), [mm.group(1)]))
            masked = SYMBOL_REF.sub(" ", masked)
            bare_matches = list(BARE.finditer(masked))
            for mm in bare_matches:
                f = pick_file(line, mm.start(), cur_file, sec_file, sym_ctx, sym)
                if f is None:
                    skipped.append((doc, i, mm.group(0), "无文件上下文"))
                    continue
                # "符号 :NNN"（附录 A 速查表 / 表格行）：紧贴的符号给出归属函数，
                # 引用必须落在该函数体内。一行多个引用时只认紧贴写法，避免张冠李戴。
                expect = []
                anchor = re.search(r"([A-Za-z_]\w*)\s*$", masked[:mm.start()])
                if anchor and len(sym.get(anchor.group(1), ())) == 1:
                    expect = [anchor.group(1)]
                refs.append((f, int(mm.group(1)), int(mm.group(2) or mm.group(1)),
                             mm.group(0), "bare", mm.start(), expect))

            for f, lo, hi, text, kind, pos, expect in refs:
                if f not in src:
                    failures.append((doc, i, text, f"源文件 {f} 不存在", None))
                    continue
                counts[kind] += 1
                if expect:
                    counts["body"] += 1
                elif len(refs) == 1:
                    counts["weak"] += 1
                else:
                    counts["struct"] += 1
                body = src[f]
                if not (1 <= lo <= len(body)) or hi > len(body):
                    why = f"行号超出 {f} 长度（{len(body)} 行）"
                else:
                    why = start_ok(body, lo, hi, line)
                    if why is None and not body[hi - 1].strip():
                        why = f"区间终点 {hi} 是空行"
                    if why is None and hi != lo:
                        tail = body[hi - 1].strip()
                        if tail.startswith("//") and "注释" not in line:
                            why = f"区间终点 {hi} 是注释（{tail[:40]!r}），疑越界到下一个单元"
                        elif tail.startswith("template"):
                            why = f"区间终点 {hi} 是模板声明，疑越界到下一个单元"
                    if why is None and expect and not any(
                            inside_function(f, defs, e, n, src) for e in expect for n in (lo, hi)):
                        near = defs.get((f, expect[0]), [None])[0]
                        why = f":{lo}-{hi} 不在 {expect} 的函数体内（{expect[0]} 定义在 :{near}）"
                    if why is None and len(refs) == 1:
                        # 一行多个引用时"点名的符号"属于哪一个说不清，只查结构
                        named = [s for s in BACKTICK.findall(line) if s not in SRC_FILES]
                        named = [s for s in named if any(s in t for t in body)]
                        window = "\n".join(body[max(0, lo - 1 - TOL):min(len(body), hi + TOL)])
                        if named and not any(s in window for s in named):
                            why = f"文档点名的符号 {named} 不在 ±{TOL} 行内"
                    if why is None and kind == "bare":
                        name = adjacent_name(line, pos) if len(refs) == 1 else None
                        name = name or table_symbol(line, pos, refs)
                        if name and ((f, name) in defs or any(
                                re.search(r"\b" + re.escape(name) + r"\s*[<(]", t) for t in body)):
                            near = body[max(0, lo - 1 - TOL):min(len(body), hi + TOL)]
                            if not (inside_function(f, defs, name, lo, src) or any(
                                    re.search(r"\b" + re.escape(name) + r"\s*[<(]", t) for t in near)):
                                d = defs.get((f, name), [None])[0]
                                why = (f"紧邻的 {name} 既不在 :{lo} 的函数体内"
                                       f"（定义在 :{d}），也不在 ±{TOL} 行内被调用")
                if why is None:
                    continue
                best = None
                named = [s for s in BACKTICK.findall(line) if s not in SRC_FILES]
                for j, textline in enumerate(body, 1):
                    if any(s in textline for s in named):
                        d = abs(j - lo)
                        if best is None or d < best[0]:
                            best = (d, j, textline.strip())
                failures.append((doc, i, text, f"{why}（推断文件 {f}）", best))

    for doc, ln, ref, why, best in failures:
        print(f"DRIFT {os.path.relpath(doc, ROOT)}:{ln}  {ref}\n  {why}")
        if best:
            print(f"  suggest :{best[1]}  {best[2][:88]!r}")
    for doc, ln, ref, why in skipped:
        print(f"SKIP  {os.path.relpath(doc, ROOT)}:{ln}  {ref}（{why}）")

    total = counts["explicit"] + counts["symbol"] + counts["bare"]
    summary = (f"显式 {counts['explicit']}、符号限定 {counts['symbol']}、"
               f"裸行号 {counts['bare']}、跳过 {len(skipped)}；"
               f"其中函数体已校验 {counts['body']}、单引用弱校验 {counts['weak']}、"
               f"仅结构校验 {counts['struct']}")
    if failures:
        print(f"\n{len(failures)} drift(s) / {total} refs（{summary}）")
        return 1
    print(f"OK {total}/{total} refs（{summary}）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [os.path.join(ROOT, "README.md")]
        for sub in ("docs", "benchmarks"):
            for base, _, files in os.walk(os.path.join(ROOT, sub)):
                args += [os.path.join(base, f) for f in sorted(files) if f.endswith(".md")]
    sys.exit(check(args))
