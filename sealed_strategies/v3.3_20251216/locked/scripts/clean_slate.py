#!/usr/bin/env python3
"""Clean-slate utility (safe):

- Moves historical generated artifacts under results/ to a timestamped backup folder
  (keeps protected ARCHIVE_* directories in place).
- Clears cache directories so cross-section/factor computations are rebuilt.

NOTE:
- Does NOT touch raw data under raw/.
- Does NOT delete sealed_strategies/.

Usage:
  uv run python scripts/clean_slate.py --yes

Optional:
  uv run python scripts/clean_slate.py --yes --keep-pattern ARCHIVE_* --keep-pattern ARCHIVE__*/
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _glob_keep_patterns(results_dir: Path, patterns: list[str]) -> set[Path]:
    kept: set[Path] = set()
    for pat in patterns:
        for p in results_dir.glob(pat):
            kept.add(p.resolve())
    return kept


def _move_children(
    results_dir: Path, backup_dir: Path, keep_patterns: list[str]
) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    backup_dir.mkdir(parents=True, exist_ok=True)

    kept = _glob_keep_patterns(results_dir, keep_patterns)
    for child in sorted(results_dir.iterdir(), key=lambda p: p.name):
        if child.resolve() == backup_dir.resolve():
            continue
        if child.resolve() in kept:
            continue

        target = backup_dir / child.name
        # Avoid overwriting if rerun
        if target.exists():
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = backup_dir / f"{child.name}.{suffix}"
        shutil.move(str(child), str(target))
        moved.append((child, target))

    return moved


def _clear_dir(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return 1

    # Directory
    count = 0
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe clean-slate: archive results + clear caches"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required: perform actions. Without this flag, only prints the plan.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (relative to project root). Default: results",
    )
    parser.add_argument(
        "--keep-pattern",
        action="append",
        default=["ARCHIVE_*"],
        help="Glob pattern(s) under results/ to keep in place. Default: ARCHIVE_* (can be repeated)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Also clear cache directories (.cache, etf_rotation_optimized/.cache).",
    )
    args = parser.parse_args()

    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results dir not found: {results_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = results_dir / f"_cleanup_backup_{ts}"

    cache_paths: list[Path] = []
    if args.clear_cache:
        cache_paths = [
            PROJECT_ROOT / ".cache",
            PROJECT_ROOT / "etf_rotation_optimized" / ".cache",
        ]

    print("=" * 100)
    print("CLEAN-SLATE PLAN")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results dir:  {results_dir}")
    print(f"Backup dir:   {backup_dir}")
    print(f"Keep patterns: {args.keep_pattern}")

    if args.clear_cache:
        print("Cache dirs to clear:")
        for p in cache_paths:
            print(f"  - {p}")

    # Dry-run listing
    kept = _glob_keep_patterns(results_dir, args.keep_pattern)
    to_move = [
        p
        for p in results_dir.iterdir()
        if p.resolve() not in kept and p.name != backup_dir.name
    ]
    print(f"Will keep in place: {[p.name for p in sorted(kept, key=lambda x: x.name)]}")
    print(f"Will move to backup ({len(to_move)} items):")
    for p in sorted(to_move, key=lambda x: x.name):
        print(f"  - {p.name}")

    if not args.yes:
        print("\n[DRY RUN] Add --yes to execute.")
        return

    moved = _move_children(results_dir, backup_dir, args.keep_pattern)
    print("\nMoved:")
    for src, dst in moved:
        print(f"  - {src.name} -> {dst.relative_to(PROJECT_ROOT)}")

    if args.clear_cache:
        print("\nClearing caches:")
        for p in cache_paths:
            n = _clear_dir(p)
            print(f"  - cleared {n} entries from {p.relative_to(PROJECT_ROOT)}")

    print("\n✅ Clean-slate completed.")


if __name__ == "__main__":
    main()
