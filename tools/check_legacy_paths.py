#!/usr/bin/env python3
"""Scan workspace files for legacy CLI path usage (run_combo_wfo.py, apply_ranker.py)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

PATTERNS = {
    "python run_combo_wfo.py": "python applications/run_combo_wfo.py",
    "python apply_ranker.py": "python applications/apply_ranker.py",
    "python run_ranking_pipeline.py": "python applications/run_ranking_pipeline.py",
    "python train_ranker.py": "python applications/train_ranker.py",
}
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "_archive"}
DEFAULT_ROOT = Path.cwd()


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not any(part in EXCLUDE_DIRS for part in path.parts):
            yield path


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return findings
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern, hint in PATTERNS.items():
            if pattern in line:
                findings.append((line_no, pattern, hint))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help="工作空间根目录")
    args = parser.parse_args()

    root = args.root.resolve()
    violations = []
    for file_path in iter_files(root):
        matches = scan_file(file_path)
        if matches:
            violations.append((file_path, matches))

    if not violations:
        print("✅ 未发现 legacy CLI 路径引用")
        return 0

    print("❌ 发现 legacy CLI 引用：")
    for file_path, matches in violations:
        print(f"- {file_path}")
        for line_no, pattern, hint in matches:
            print(f"  L{line_no}: {pattern} -> {hint}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
