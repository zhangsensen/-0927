#!/usr/bin/env python3
"""Analyze holdout validation results and produce a compact markdown report.

This script is intentionally lightweight (no heavy plots) and focuses on:
- Overfitting diagnostics (train vs holdout relationship)
- Recent-window stability (21/42/63/60/120/240 trading days)
- Candidate shortlists under different constraints (e.g., choppy-market friendly)

Usage:
  uv run python scripts/analyze_holdout_results.py \
        --input results/holdout_validation_YYYYMMDD_HHMMSS/holdout_validation_results.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    # Fallback: try parquet first, then csv
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x * 100:.2f}%"


def _fmt_num(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def _top_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    view = df[cols].head(n).copy()
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="holdout_validation_results.parquet/csv path",
    )
    parser.add_argument("--output", default=None, help="output markdown path (default next to input)")
    parser.add_argument("--top", type=int, default=30, help="rows to include in each shortlist")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_path = Path(args.output) if args.output else input_path.with_name("holdout_report.md")

    df = _read_table(input_path)

    # Expected key columns
    required = [
        "combo",
        "vec_calmar_ratio",
        "vec_return",
        "vec_max_drawdown",
        "holdout_calmar_ratio",
        "holdout_return",
        "holdout_max_drawdown",
        "calmar_ratio_stability",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_total = len(df)
    n_pos_hold = int((df["holdout_return"] > 0).sum())

    corr_calmar = _safe_corr(df["vec_calmar_ratio"], df["holdout_calmar_ratio"])
    corr_ret = _safe_corr(df["vec_return"], df["holdout_return"])

    # Recent window columns (optional)
    recent_cols = [
        ("holdout_trail_21d_return", "H1M"),
        ("holdout_trail_42d_return", "H2M"),
        ("holdout_trail_63d_return", "H3M"),
        ("holdout_trail_60d_return", "H60"),
        ("holdout_trail_120d_return", "H120"),
        ("holdout_trail_240d_return", "H240"),
    ]
    have_recent = [c for c, _ in recent_cols if c in df.columns]

    # Shortlists
    df_stable = df.sort_values("calmar_ratio_stability", ascending=False)

    # Choppy-friendly: require last 1-3 months in holdout non-negative
    df_choppy = df.copy()
    for c in ["holdout_trail_21d_return", "holdout_trail_42d_return", "holdout_trail_63d_return"]:
        if c in df_choppy.columns:
            df_choppy = df_choppy[df_choppy[c] > 0]
    df_choppy = df_choppy.sort_values("calmar_ratio_stability", ascending=False)

    # Conservative risk: cap holdout drawdown
    df_conservative = df[df["holdout_max_drawdown"] <= 0.12].sort_values("calmar_ratio_stability", ascending=False)

    cols_base = [
        "combo",
        "vec_calmar_ratio",
        "holdout_calmar_ratio",
        "calmar_ratio_stability",
        "vec_return",
        "holdout_return",
        "vec_max_drawdown",
        "holdout_max_drawdown",
    ]

    cols_recent = []
    for c, tag in recent_cols:
        if c in df.columns:
            cols_recent.append(c)

    cols_show = cols_base + cols_recent

    lines: list[str] = []
    lines.append(f"# Holdout Validation Report\n")
    lines.append(f"- Input: {input_path}\n")
    lines.append(f"- Total strategies: {n_total}\n")
    lines.append(f"- Holdout positive return: {n_pos_hold} ({n_pos_hold / max(n_total, 1) * 100:.1f}%)\n")
    lines.append(f"- Corr(train_calmar, holdout_calmar): {_fmt_num(corr_calmar)}\n")
    lines.append(f"- Corr(train_return, holdout_return): {_fmt_num(corr_ret)}\n")

    if have_recent:
        lines.append("\n## Recent-window availability\n")
        lines.append("- Available columns: " + ", ".join(have_recent) + "\n")

    lines.append("\n## Top by Dual-Stability (min Train/Holdout Calmar)\n")
    lines.append(_top_table(df_stable, cols_show, n=args.top))

    lines.append("\n## Choppy-friendly shortlist (Holdout last 1-3 months all positive, then stability)\n")
    if len(df_choppy) == 0:
        lines.append("No strategies satisfy the constraints.\n")
    else:
        lines.append(_top_table(df_choppy, cols_show, n=args.top))

    lines.append("\n## Conservative risk shortlist (Holdout MDD <= 12%, then stability)\n")
    if len(df_conservative) == 0:
        lines.append("No strategies satisfy the constraints.\n")
    else:
        lines.append(_top_table(df_conservative, cols_show, n=args.top))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
