#!/usr/bin/env python3
"""校验文档里的 `file:NNN` 行号锚点是否还落在有意义的位置。

三类判据（任一不满足即报漂移）：
  1. 行号在文件范围内，且不是空行 / 只有括号 / 只有 `return 0;` 之类的残句；
  2. 该行不是注释——除非文档行明确写了"注释"；
  3. 若文档行用反引号点出了符号（如 `` `index_impl`（op.cpp:543）``），该符号
     必须在 `NNN±TOL` 行内出现。

为什么需要它：源码新增/删除几行，文档里所有"拿着代码读"的行号就整体错位。
本脚本在源码改动后跑一遍，能抓住绝大多数漂移（空行/括号/注释/符号不符），
并按源码里的实际位置给出建议行号。

用法：
  python tools/check_doc_anchors.py [文档...]     # 默认扫 docs/**/*.md + README.md
退出码 0 = 通过；1 = 有漂移。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "index_sve")
TOL = 3

SRC_FILES = ["op.cpp", "normalize.cpp", "normalize.h", "linearize.cpp", "gather.cpp", "grain.h"]
REF = re.compile(r"\b((?:op|normalize|linearize|gather|grain)\.(?:cpp|h)):(\d+)(?:-(\d+))?")
BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
STRUCTURAL = re.compile(r"^(}|\};|\}\)|return 0;|#endif|#else|else \{|\s*)$")


def check(docs):
    src = {}
    for f in SRC_FILES:
        path = os.path.join(SRC_DIR, f)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                src[f] = fh.read().splitlines()

    failures, checked = [], 0
    for doc in docs:
        if os.sep + "archive" + os.sep in doc:
            continue  # 归档快照冻结，行号随当时源码，不追新
        with open(doc, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        for i, line in enumerate(lines, 1):
            for m in REF.finditer(line):
                f, lo = m.group(1), int(m.group(2))
                hi = int(m.group(3)) if m.group(3) else lo
                checked += 1
                body = src.get(f)
                if body is None:
                    failures.append((doc, i, m.group(0), "源文件不存在", None))
                    continue
                if not (1 <= lo <= len(body)) or hi > len(body):
                    failures.append((doc, i, m.group(0), "行号超出文件长度", None))
                    continue
                actual = body[lo - 1].strip()
                window = "\n".join(body[max(0, lo - 1 - TOL):hi + TOL])

                bad = None
                if STRUCTURAL.match(actual) or len(actual) < 3:
                    bad = f"该行是结构性残句 {actual!r}"
                elif actual.startswith("//") and "注释" not in line:
                    bad = "该行是注释（文档未声明）"
                else:
                    syms = [s for s in BACKTICK.findall(line) if s not in SRC_FILES]
                    named = [s for s in syms if any(s in t for t in body)]
                    if named and not any(s in window for s in named):
                        bad = f"文档点名的符号 {named} 不在 ±{TOL} 行内"

                if bad is None:
                    continue
                best = None
                syms = [s for s in BACKTICK.findall(line) if s not in SRC_FILES]
                for j, text in enumerate(body, 1):
                    if any(s in text for s in syms):
                        d = abs(j - lo)
                        if best is None or d < best[0]:
                            best = (d, j, text.strip())
                failures.append((doc, i, m.group(0), bad, best))

    for doc, ln, ref, why, best in failures:
        print(f"DRIFT {os.path.relpath(doc, ROOT)}:{ln}  {ref}\n  {why}")
        if best:
            print(f"  suggest {ref.split(':')[0]}:{best[1]}  {best[2][:88]!r}")
    if failures:
        print(f"\n{len(failures)} drift(s) / {checked} refs")
        return 1
    print(f"OK {checked}/{checked} refs")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    else:
        args = [os.path.join(ROOT, "README.md")]
        for base, _, files in os.walk(os.path.join(ROOT, "docs")):
            args += [os.path.join(base, f) for f in sorted(files) if f.endswith(".md")]
    sys.exit(check(args))
