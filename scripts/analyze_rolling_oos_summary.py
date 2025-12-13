#!/usr/bin/env python3
"""Analyze rolling OOS consistency summary and propose overfitting-rejection gates.

This script reads the output of `scripts/run_rolling_oos_consistency.py`:
- rolling_oos_summary.parquet

It produces:
- rolling_oos_gate_analysis.md
- (optional) filtered candidate tables as parquet

Usage:
  uv run python scripts/analyze_rolling_oos_summary.py \
    --input results/rolling_oos_consistency_YYYYMMDD_HHMMSS/rolling_oos_summary.parquet

Notes:
- Quarterly segmentation in the current run yields 20 total segments and 2 holdout segments.
  Holdout-segment metrics therefore have coarse granularity (0/0.5/1) and should be used as a secondary check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingGate:
    name: str
    all_pos_min: float
    all_worst_ret_min: float
    full_calmar_min: float
    holdout_worst_ret_min: float | None = None


PRESETS: dict[str, RollingGate] = {
    "relaxed": RollingGate(
        name="relaxed",
        all_pos_min=0.55,
        all_worst_ret_min=-0.10,
        full_calmar_min=0.60,
        holdout_worst_ret_min=None,
    ),
    "medium": RollingGate(
        name="medium",
        all_pos_min=0.60,
        all_worst_ret_min=-0.10,
        full_calmar_min=0.60,
        holdout_worst_ret_min=None,
    ),
    "strict": RollingGate(
        name="strict",
        all_pos_min=0.60,
        all_worst_ret_min=-0.08,
        full_calmar_min=0.80,
        holdout_worst_ret_min=None,
    ),
    "ultra": RollingGate(
        name="ultra",
        all_pos_min=0.60,
        all_worst_ret_min=-0.08,
        full_calmar_min=0.80,
        holdout_worst_ret_min=0.0,
    ),
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input suffix: {suffix} ({path})")


def _quantile_table(df: pd.DataFrame, cols: list[str], qs: list[float]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q = s.quantile(qs)
        rows.append(pd.DataFrame({"metric": col, "q": q.index, "value": q.values}))
    out = pd.concat(rows, ignore_index=True)
    pivot = out.pivot(index="metric", columns="q", values="value")
    return pivot


def _apply_gate(df: pd.DataFrame, gate: RollingGate) -> pd.Series:
    mask = (
        (df["all_segment_positive_rate"] >= gate.all_pos_min)
        & (df["all_segment_worst_return"] >= gate.all_worst_ret_min)
        & (df["full_calmar_ratio"] >= gate.full_calmar_min)
    )
    if gate.holdout_worst_ret_min is not None:
        mask = mask & (df["holdout_segment_worst_return"] >= gate.holdout_worst_ret_min)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to rolling_oos_summary.parquet (or csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: alongside input with timestamp).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top-N rows to display per preset.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write filtered candidate tables as parquet.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    df = _read_table(input_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_path.parent / f"rolling_oos_gate_analysis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_cols = [
        "full_total_return",
        "full_max_drawdown",
        "full_sharpe_ratio",
        "full_calmar_ratio",
        "all_segment_positive_rate",
        "all_segment_worst_return",
        "all_segment_worst_calmar",
        "all_segment_median_return",
        "all_segment_median_calmar",
        "holdout_segment_positive_rate",
        "holdout_segment_worst_return",
        "holdout_segment_worst_calmar",
    ]

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qt = _quantile_table(df, key_cols, qs)

    md_lines: list[str] = []
    md_lines.append("# Rolling OOS Gate Analysis\n")
    md_lines.append(f"- Input: {input_path}\n")
    md_lines.append(f"- Rows: {len(df)}\n")
    md_lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md_lines.append("\n## Segment Count\n")
    md_lines.append("\n- all_segment_count (value_counts):\n")
    md_lines.append(df["all_segment_count"].value_counts().head(10).to_frame("count").to_markdown())
    md_lines.append("\n\n- holdout_segment_count (value_counts):\n")
    md_lines.append(df["holdout_segment_count"].value_counts().head(10).to_frame("count").to_markdown())

    md_lines.append("\n\n## Quantiles (key metrics)\n")
    md_lines.append(qt.to_markdown(floatfmt=".6f"))

    md_lines.append("\n\n## Preset Gates\n")
    for preset_name, gate in PRESETS.items():
        mask = _apply_gate(df, gate)
        pass_rate = float(mask.mean())
        md_lines.append(
            "\n" + "### " + preset_name + "\n"
            + f"- Definition: all_pos>={gate.all_pos_min:.2f}, all_worst_ret>={gate.all_worst_ret_min:.2f}, full_calmar>={gate.full_calmar_min:.2f}"
            + ("" if gate.holdout_worst_ret_min is None else f", holdout_worst_ret>={gate.holdout_worst_ret_min:.2f}")
            + "\n"
            + f"- Pass: {mask.sum()} / {len(df)} ({pass_rate*100:.3f}%)\n"
        )

        cols = [
            "combo",
            "full_total_return",
            "full_calmar_ratio",
            "full_max_drawdown",
            "all_segment_positive_rate",
            "all_segment_worst_return",
            "holdout_segment_worst_return",
        ]
        top = (
            df.loc[mask, cols]
            .sort_values(["full_total_return", "full_calmar_ratio"], ascending=False)
            .head(int(args.top_n))
        )
        md_lines.append(top.to_markdown(index=False, floatfmt=".6f"))

        if args.write_parquet:
            top_path = out_dir / f"candidates_{preset_name}.parquet"
            df.loc[mask].to_parquet(top_path, index=False)

    report_path = out_dir / "rolling_oos_gate_analysis.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✅ Wrote: {report_path}")
    if args.write_parquet:
        print(f"✅ Wrote candidate parquet(s) under: {out_dir}")


if __name__ == "__main__":
    main()
