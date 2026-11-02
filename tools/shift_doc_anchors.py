"""把文档里的 `file:NNN` 行号按"源码改过以后的行位移"批量重定位。

用途：改了 C++ 源码（增删行）以后，文档里的行号引用会整体漂移，
`check_doc_anchors.py` 会报一堆 DRIFT。手工改不现实，这里用 difflib 把
`git show HEAD:<file>` 与工作区文件做行级对齐，得到 old→new 的映射，
再把 `check_doc_anchors.py` 报出来的每一处引用按映射改写。

只处理"纯位移"：新增/删除行导致的位置变化。**内容被改写**的目标（旧行在
新文件里没有对应）会跳过并打印，需要人工重写那一处引用。

用法：
  python tools/shift_doc_anchors.py --dry-run   # 只打印，不改文件
  python tools/shift_doc_anchors.py             # 改完后自己再跑一遍 check
  python tools/shift_doc_anchors.py --check     # 只跑 check_doc_anchors
"""
import argparse
import difflib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "index_sve")
SRC_FILES = ["op.cpp", "gather.cpp", "normalize.cpp", "linearize.cpp", "grain.h"]

DRIFT_RE = re.compile(r"^DRIFT (\S+):(\d+)\s+(\S+)$")
INFER_RE = re.compile(r"推断文件 ([\w.]+)")
REF_RE = re.compile(r"(?:([\w./]+\.(?:cpp|h)))?:(\d+)(?:-(\d+))?")


def line_map(name):
    """old_line(1-based) -> new_line(1-based)，只映射内容能对齐的行。"""
    old = subprocess.run(["git", "show", f"HEAD:index_sve/{name}"], cwd=ROOT,
                         capture_output=True, text=True)
    if old.returncode != 0:
        sys.exit(f"git show HEAD:index_sve/{name} 失败：{old.stderr[:200]}")
    with open(os.path.join(SRC_DIR, name), encoding="utf-8") as f:
        new = f.read()
    a, b = old.stdout.splitlines(), new.splitlines()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
    return m, set(range(1, len(a) + 1)) - set(m)


def check():
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                     "check_doc_anchors.py")],
                       cwd=ROOT, capture_output=True, text=True)
    return p.stdout + p.stderr


def parse(out):
    drifts = []
    pending = None
    for line in out.splitlines():
        mm = DRIFT_RE.match(line.strip())
        if mm:
            pending = {"doc": mm.group(1), "doc_line": int(mm.group(2)),
                       "ref": mm.group(3), "file": None}
            drifts.append(pending)
            continue
        if pending and pending["file"] is None:
            mi = INFER_RE.search(line)
            if mi:
                pending["file"] = mi.group(1)
    return drifts


def shift_ref(ref, default_file, maps):
    mm = REF_RE.fullmatch(ref)
    if not mm:
        return None, "无法解析"
    name = mm.group(1) or default_file
    if name not in maps:
        return None, f"未知文件 {name}"
    m, missing = maps[name]
    lo, hi = int(mm.group(2)), int(mm.group(3) or mm.group(2))
    for v in (lo, hi):
        if v in missing:
            return None, f"{name}:{v} 在改后文件中没有对应行（内容被改写）"
    nlo = m[lo]
    nhi = nlo + (hi - lo)
    new = f"{mm.group(1) + ':' if mm.group(1) else ':'}{nlo}"
    if mm.group(3):
        new += f"-{nhi}"
    return new, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        print(check())
        return

    out = check()
    drifts = parse(out)
    if not drifts:
        print("没有漂移")
        return
    maps = {n: line_map(n) for n in SRC_FILES}
    per_doc = {}
    skipped = []
    for d in drifts:
        new, why = shift_ref(d["ref"], d["file"], maps)
        if new is None:
            skipped.append((d, why))
            continue
        per_doc.setdefault(d["doc"], []).append((d["doc_line"], d["ref"], new))

    for doc, items in per_doc.items():
        path = os.path.join(ROOT, doc)
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        n_done = 0
        for doc_line, old_ref, new_ref in items:
            i = doc_line - 1
            if old_ref not in lines[i]:
                skipped.append(({"doc": doc, "doc_line": doc_line, "ref": old_ref},
                                "该行里找不到这处引用"))
                continue
            lines[i] = lines[i].replace(old_ref, new_ref, 1)
            n_done += 1
        if n_done and not a.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        print(f"{doc}: {n_done} 处{'（dry-run 未写）' if a.dry_run else ''}")

    if skipped:
        print(f"\n{len(skipped)} 处需要人工处理：")
        for d, why in skipped:
            print(f"  {d['doc']}:{d['doc_line']}  {d['ref']}  —— {why}")


if __name__ == "__main__":
    main()
