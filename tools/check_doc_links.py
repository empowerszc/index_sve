#!/usr/bin/env python3
"""Check relative Markdown links and #anchors in this repository.

Usage:
    python3 tools/check_doc_links.py [ROOT]

ROOT defaults to the current working directory. Exit code 0 means every
relative link and anchor resolved; 1 means at least one is broken.
Only the Python standard library is used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

SKIP_DIRS = {".git", "node_modules", "agent_space", ".superpowers"}
SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:")

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub("", line)


def heading_slug(text: str) -> str:
    """Approximate GitHub's heading anchor slug."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("`", "")
    text = text.lower()
    kept = []
    for ch in text:
        if ch.isalnum() or ch in (" ", "-", "_"):
            kept.append(ch)
    text = "".join(kept)
    return text.replace(" ", "-")


def collect_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_fence = False
    fence_char = ""
    fence_len = 0
    prev_text = ""
    for line in lines:
        stripped = line.lstrip()
        m = FENCE_RE.match(line)
        if m:
            token = m.group(2)
            if not in_fence:
                in_fence = True
                fence_char = token[0]
                fence_len = len(token)
            elif token[0] == fence_char and len(token) >= fence_len:
                in_fence = False
            prev_text = ""
            continue
        if in_fence:
            continue

        hm = HEADING_RE.match(line)
        if hm:
            text = hm.group(2)
        elif SETEXT_RE.match(line) and prev_text:
            text = prev_text
        else:
            if stripped:
                prev_text = stripped
            continue

        base = heading_slug(text)
        if not base:
            prev_text = ""
            continue
        n = counts.get(base, 0)
        anchors.add(base if n == 0 else f"{base}-{n}")
        counts[base] = n + 1
        prev_text = ""
    return anchors


def parse_target(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        return raw[1:raw.index(">")]
    return raw.split()[0] if raw.split() else raw


def check_file(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_fence = False
    fence_char = ""
    fence_len = 0

    for lineno, line in enumerate(lines, 1):
        m = FENCE_RE.match(line)
        if m:
            token = m.group(2)
            if not in_fence:
                in_fence = True
                fence_char = token[0]
                fence_len = len(token)
            elif token[0] == fence_char and len(token) >= fence_len:
                in_fence = False
            continue
        if in_fence:
            continue

        clean = strip_inline_code(line)
        for lm in LINK_RE.finditer(clean):
            target = parse_target(lm.group(1))
            if not target or target == "#":
                continue
            if target.startswith(SKIP_SCHEMES):
                continue

            path_part, sep, anchor = target.partition("#")
            if path_part:
                resolved = (path.parent / path_part).resolve()
                if not resolved.exists():
                    try:
                        shown = resolved.relative_to(root.resolve()).as_posix()
                    except ValueError:
                        shown = str(resolved)
                    errors.append(f"{path.relative_to(root).as_posix()}:{lineno}: {shown}")
                    continue
            else:
                resolved = path

            if sep and anchor:
                anchor = unquote(anchor)
                if resolved.suffix.lower() != ".md":
                    continue
                if anchor not in collect_anchors(resolved):
                    try:
                        shown = resolved.relative_to(root.resolve()).as_posix()
                    except ValueError:
                        shown = str(resolved)
                    errors.append(
                        f"{path.relative_to(root).as_posix()}:{lineno}: {shown}#{anchor}"
                    )
    return errors


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    root = root.resolve()
    files = []
    for p in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        files.append(p)

    errors: list[str] = []
    for p in files:
        errors.extend(check_file(p, root))

    for e in errors:
        print(e)
    if errors:
        print(f"FAIL: {len(files)} files, {len(errors)} broken links", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} files, 0 broken links")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
