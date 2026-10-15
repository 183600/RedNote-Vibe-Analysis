#!/usr/bin/env python3
"""CI gate for the RedNote-Vibe-Analysis continuous loop.

Exit 0 = pass. Any other exit code = the calling workflow must discard the
phase's changes (git reset --hard).

Layers
------
1. Syntax: every .py file parses (always run, no third-party dependencies).
2. Protected assets: README.md / README_zh.md and the core research datasets
   exist and are non-empty; every *.jsonl / *.csv.gz in the repo root is
   non-empty. The datasets are the foundation of every analysis in this
   repo -- losing or truncating them is always fatal.
3. Data sanity: the first line of every root-level *.jsonl file parses as
   JSON (cheap corruption check, stdlib only).

--syntax-only runs layer 1 only (used by the shutdown trap).
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent

FAILURES: list[str] = []

# 核心研究数据集（缺失或异常变小即整体否决）
CORE_DATASETS = (
    "training_set_human.jsonl",
    "training_set_aigc.jsonl",
    "exploring_set.jsonl",
)

# 受保护文档（缺失或为空即整体否决）
PROTECTED_DOCS = (
    "README.md",
    "README_zh.md",
)

# 核心数据集的最小合理体积（实际均为 8MB~100MB 量级）
MIN_CORE_BYTES = 1_000_000


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"[gate] FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"[gate] ok: {msg}")


# --------------------------------------------------------------------------
# Layer 1: syntax
# --------------------------------------------------------------------------
def check_syntax() -> None:
    n = 0
    for py in sorted(ROOT.rglob("*.py")):
        if any(part in (".git", "__pycache__", ".venv", "venv") for part in py.parts):
            continue
        try:
            ast.parse(py.read_text(encoding="utf-8"))
            n += 1
        except SyntaxError as e:
            fail(f"syntax error in {py.relative_to(ROOT)}: {e}")
        except (OSError, UnicodeDecodeError) as e:
            fail(f"cannot read {py.relative_to(ROOT)}: {e}")
    if n == 0:
        fail("no python files found -- the analysis codebase is gone?")
    else:
        ok(f"syntax: {n} python files parse")


# --------------------------------------------------------------------------
# Layer 2: protected assets (READMEs + research datasets)
# --------------------------------------------------------------------------
def check_protected_assets() -> None:
    for name in PROTECTED_DOCS:
        p = ROOT / name
        if not p.exists() or p.stat().st_size == 0:
            fail(f"protected doc missing or empty: {name}")

    for name in CORE_DATASETS:
        p = ROOT / name
        if not p.exists():
            fail(f"core dataset missing: {name}")
        elif p.stat().st_size < MIN_CORE_BYTES:
            fail(f"core dataset suspiciously small: {name} ({p.stat().st_size} bytes)")

    n = 0
    for pattern in ("*.jsonl", "*.csv.gz"):
        for p in sorted(ROOT.glob(pattern)):
            if p.stat().st_size == 0:
                fail(f"data file is empty: {p.name}")
            n += 1
    if n:
        ok(f"protected assets: {len(PROTECTED_DOCS)} docs + {len(CORE_DATASETS)} core "
           f"datasets + {n} root-level data files intact")


# --------------------------------------------------------------------------
# Layer 3: data sanity (first line of every root-level .jsonl parses)
# --------------------------------------------------------------------------
def check_data_sanity() -> None:
    n = 0
    for p in sorted(ROOT.glob("*.jsonl")):
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                first = f.readline()
            if first.strip():
                json.loads(first)
            n += 1
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            fail(f"data sanity: first line of {p.name} does not parse: {e}")
    if n:
        ok(f"data sanity: first line of {n} jsonl files parses")


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv
    check_syntax()
    if not syntax_only:
        check_protected_assets()
        check_data_sanity()
    if FAILURES:
        print(f"\n[gate] REJECTED -- {len(FAILURES)} failure(s)")
        return 1
    print("\n[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
