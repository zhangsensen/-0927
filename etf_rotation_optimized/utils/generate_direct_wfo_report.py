#!/usr/bin/env python3
"""
Generate DIRECT_FACTOR_WFO_FINAL_REPORT.md from the latest WFO summary CSV under results/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd


def find_latest_wfo_summary(results_root: Path) -> Path:
    wfo_dir = results_root / "wfo"
    if not wfo_dir.exists():
        raise FileNotFoundError(f"WFO results root not found: {wfo_dir}")

    date_dirs = sorted([d for d in wfo_dir.iterdir() if d.is_dir()], reverse=True)
    for date_dir in date_dirs:
        time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()], reverse=True)
        for time_dir in time_dirs:
            summary = time_dir / "wfo_summary.csv"
            if summary.exists():
                return summary
    raise FileNotFoundError("No wfo_summary.csv found under results/wfo/")


def parse_selected_factors(series: pd.Series) -> List[str]:
    all_factors: List[str] = []
    for s in series.dropna().astype(str):
        parts = [p.strip() for p in s.split(",") if p.strip()]
        all_factors.extend(parts)
    return all_factors


def main():
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "results"
    summary_path = find_latest_wfo_summary(results_root)

    df = pd.read_csv(summary_path)
    avg_ic = float(df["oos_ensemble_ic"].mean())
    win_rate = float((df["oos_ensemble_ic"] > 0).mean())
    avg_sharpe = float(df["oos_ensemble_sharpe"].mean())
    avg_selected = float(df["n_selected_factors"].mean())

    # Top factors by frequency
    factors = parse_selected_factors(df["selected_factors"])
    top_counts = (
        pd.Series(factors)
        .value_counts()
        .head(10)
        .rename_axis("factor")
        .reset_index(name="count")
    )

    # Build report
    report_lines = []
    report_lines.append("# Direct Factor WFO — Final Report")
    report_lines.append("")
    report_lines.append(f"Summary CSV: `{summary_path}`")
    report_lines.append("")
    report_lines.append("## Metrics")
    report_lines.append(f"- Average OOS IC: {avg_ic:.4f}")
    report_lines.append(f"- OOS IC Win Rate: {win_rate:.1%}")
    report_lines.append(f"- Average OOS Sharpe (IC IR): {avg_sharpe:.2f}")
    report_lines.append(f"- Avg Selected Factors: {avg_selected:.1f}")
    report_lines.append("")
    report_lines.append("## Top Factors (by selection frequency)")
    for _, row in top_counts.iterrows():
        report_lines.append(f"- {row['factor']}: {int(row['count'])}")

    out_path = (
        repo_root / "etf_rotation_optimized" / "DIRECT_FACTOR_WFO_FINAL_REPORT.md"
    )
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
