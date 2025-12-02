#!/usr/bin/env python3
"""Utility to verify that the latest VEC and BT batch results stay within an alignment band."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_VEC_PATTERN = "vec_full_backtest_*"
DEFAULT_BT_PATTERN = "bt_backtest_full_*"


@dataclass
class AlignmentSummary:
    vec_dir: Path
    bt_dir: Path
    total_pairs: int
    avg_abs_diff_pp: float
    max_abs_diff_pp: float
    median_abs_diff_pp: float
    threshold_pp: float
    violating_pairs: int
    margin_failures: int

    def as_dict(self) -> dict:
        return {
            "vec_dir": str(self.vec_dir),
            "bt_dir": str(self.bt_dir),
            "total_pairs": self.total_pairs,
            "avg_abs_diff_pp": self.avg_abs_diff_pp,
            "max_abs_diff_pp": self.max_abs_diff_pp,
            "median_abs_diff_pp": self.median_abs_diff_pp,
            "threshold_pp": self.threshold_pp,
            "violating_pairs": self.violating_pairs,
            "margin_failures": self.margin_failures,
        }


def _resolve_latest(pattern: str, root: Path) -> Optional[Path]:
    candidates = sorted(d for d in root.glob(pattern) if d.is_dir())
    return candidates[-1] if candidates else None


def compare_vec_bt_results(
    vec_dir: Optional[Path | str] = None,
    bt_dir: Optional[Path | str] = None,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    vec_pattern: str = DEFAULT_VEC_PATTERN,
    bt_pattern: str = DEFAULT_BT_PATTERN,
    threshold_pp: float = 0.05,
    require_margin_zero: bool = True,
) -> AlignmentSummary:
    """Compare latest VEC and BT batch outputs.

    Raises:
        FileNotFoundError: if required files are missing.
        ValueError: if the merged result set is empty.
    """

    root = results_root if results_root.is_absolute() else Path(results_root)

    vec_path = Path(vec_dir) if vec_dir else _resolve_latest(vec_pattern, root)
    bt_path = Path(bt_dir) if bt_dir else _resolve_latest(bt_pattern, root)

    if vec_path is None or not vec_path.exists():
        raise FileNotFoundError(
            f"Cannot locate VEC results directory (pattern={vec_pattern}) under {root}"
        )
    if bt_path is None or not bt_path.exists():
        raise FileNotFoundError(
            f"Cannot locate BT results directory (pattern={bt_pattern}) under {root}"
        )

    vec_file = vec_path / "vec_all_combos.csv"
    bt_file = bt_path / "bt_results.csv"

    if not vec_file.exists():
        raise FileNotFoundError(f"Missing VEC results file: {vec_file}")
    if not bt_file.exists():
        raise FileNotFoundError(f"Missing BT results file: {bt_file}")

    vec_df = pd.read_csv(vec_file)
    bt_df = pd.read_csv(bt_file)

    required_vec_cols = {"combo", "vec_return"}
    required_bt_cols = {"combo", "bt_return", "bt_margin_failures"}

    if not required_vec_cols.issubset(vec_df.columns):
        raise ValueError(f"VEC file missing columns: {required_vec_cols - set(vec_df.columns)}")
    if not required_bt_cols.issubset(bt_df.columns):
        raise ValueError(f"BT file missing columns: {required_bt_cols - set(bt_df.columns)}")

    merged = pd.merge(
        vec_df[list(required_vec_cols)],
        bt_df[list(required_bt_cols)],
        on="combo",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlapping combos between VEC and BT results")

    merged["diff_pp"] = (merged["vec_return"] - merged["bt_return"]) * 100.0
    abs_diff = merged["diff_pp"].abs()

    threshold = float(threshold_pp)
    violating_pairs = int((abs_diff > threshold).sum())
    total_pairs = int(len(merged))
    margin_failures = int(merged["bt_margin_failures"].sum())

    summary = AlignmentSummary(
        vec_dir=vec_path,
        bt_dir=bt_path,
        total_pairs=total_pairs,
        avg_abs_diff_pp=float(abs_diff.mean()),
        max_abs_diff_pp=float(abs_diff.max()),
        median_abs_diff_pp=float(abs_diff.median()),
        threshold_pp=threshold,
        violating_pairs=violating_pairs,
        margin_failures=margin_failures,
    )

    if require_margin_zero and margin_failures > 0:
        raise ValueError(
            f"BT margin failures detected: {margin_failures} (expected 0). "
            "Please rerun BT to resolve capital shortfalls."
        )

    if violating_pairs > 0:
        raise ValueError(
            f"Found {violating_pairs} combos exceeding {threshold:.3f} pp diff."
        )

    return summary


def _format_summary(summary: AlignmentSummary) -> str:
    return (
        "\n".join(
            [
                f"VEC dir: {summary.vec_dir}",
                f"BT dir : {summary.bt_dir}",
                "",
                "Summary:",
                f"  Total combos:    {summary.total_pairs}",
                f"  Avg abs diff:    {summary.avg_abs_diff_pp:.6f} pp",
                f"  Median abs diff: {summary.median_abs_diff_pp:.6f} pp",
                f"  Max abs diff:    {summary.max_abs_diff_pp:.6f} pp",
                f"  Threshold:       {summary.threshold_pp:.6f} pp",
                f"  Violations:      {summary.violating_pairs}",
                f"  Margin failures: {summary.margin_failures}",
            ]
        )
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check VEC vs BT alignment band")
    parser.add_argument("--vec-dir", type=str, help="Explicit VEC results directory")
    parser.add_argument("--bt-dir", type=str, help="Explicit BT results directory")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing results (default: results)",
    )
    parser.add_argument(
        "--vec-pattern",
        type=str,
        default=DEFAULT_VEC_PATTERN,
        help="Glob pattern for VEC directories",
    )
    parser.add_argument(
        "--bt-pattern",
        type=str,
        default=DEFAULT_BT_PATTERN,
        help="Glob pattern for BT directories",
    )
    parser.add_argument(
        "--threshold-pp",
        type=float,
        default=0.05,
        help="Allowed absolute difference in percentage points",
    )
    parser.add_argument(
        "--allow-margin-failures",
        action="store_true",
        help="Do not fail when BT margin failures > 0",
    )
    args = parser.parse_args(argv)

    try:
        summary = compare_vec_bt_results(
            vec_dir=args.vec_dir,
            bt_dir=args.bt_dir,
            results_root=Path(args.results_root),
            vec_pattern=args.vec_pattern,
            bt_pattern=args.bt_pattern,
            threshold_pp=args.threshold_pp,
            require_margin_zero=not args.allow_margin_failures,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ VEC/BT alignment check failed: {exc}", file=sys.stderr)
        return 1

    print("✅ VEC/BT alignment check passed")
    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
