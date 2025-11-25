#!/usr/bin/env python3
"""Analyse mean_oos_ic vs returns within return-ranked buckets.

This script reproduces the "高低收益分层相关" diagnostic by splitting
strategies into return quantiles (deciles by default) and reporting the
Spearman/Pearson correlation between mean_oos_ic and realised returns within
each bucket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from analysis_utils import AnalysisError, ColumnSelection, dropna_pairs, load_datasets, resolve_columns


def decile_analysis(
    frame: pd.DataFrame,
    cols: ColumnSelection,
    *,
    buckets: int,
) -> Tuple[pd.DataFrame, dict]:
    returns = frame[cols.return_column]
    try:
        labels = pd.qcut(returns, q=buckets, duplicates="drop")
    except ValueError:
        labels = pd.cut(returns, bins=buckets)
    grouped = frame.groupby(labels, observed=False)

    rows = []
    for name, group in grouped:
        ic_vals = group[cols.ic_column]
        ret_vals = group[cols.return_column]
        if len(group) == 0:
            continue
        spearman = ic_vals.corr(ret_vals, method="spearman")
        pearson = ic_vals.corr(ret_vals, method="pearson")
        rows.append(
            {
                "bucket": str(name),
                "samples": int(len(group)),
                "return_min": float(ret_vals.min()),
                "return_max": float(ret_vals.max()),
                "return_mean": float(ret_vals.mean()),
                "ic_mean": float(ic_vals.mean()),
                "spearman": float(spearman) if not np.isnan(spearman) else None,
                "pearson": float(pearson) if not np.isnan(pearson) else None,
            }
        )

    result = pd.DataFrame(rows)
    result.sort_values("return_mean", inplace=True)
    result.reset_index(drop=True, inplace=True)

    overall = {
        "rows": int(len(frame)),
        "ic_column": cols.ic_column,
        "return_column": cols.return_column,
        "buckets": int(len(result)),
        "global_spearman": float(frame[cols.ic_column].corr(frame[cols.return_column], method="spearman")),
        "global_pearson": float(frame[cols.ic_column].corr(frame[cols.return_column], method="pearson")),
    }
    return result, overall


def aggregate_extremes(result: pd.DataFrame) -> dict:
    if result.empty:
        return {}
    # Assume rows are sorted by return_mean ascending.
    bottom = result.iloc[0]
    top = result.iloc[-1]
    return {
        "bottom_bucket": bottom.to_dict(),
        "top_bucket": top.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse IC vs return correlation inside return buckets")
    parser.add_argument("--input", nargs="+", required=True, help="CSV/Parquet files or glob patterns")
    parser.add_argument("--ic-col", default=None, help="Explicit IC column name")
    parser.add_argument("--return-col", default=None, help="Explicit return column name")
    parser.add_argument("--buckets", type=int, default=10, help="Number of return quantiles (default: 10)")
    parser.add_argument("--output-dir", default="analysis_outputs/return_deciles", help="Where to write summaries")
    parser.add_argument("--prefer-net", action="store_true", help="Prefer *_net return columns if available")
    args = parser.parse_args()

    try:
        frame = load_datasets(args.input)
        cols = resolve_columns(
            frame,
            ic_column=args.ic_col,
            return_column=args.return_col,
            prefer_net=args.prefer_net,
        )
        frame = dropna_pairs(frame, [cols.ic_column, cols.return_column])
        result, overall = decile_analysis(frame, cols, buckets=args.buckets)
        extremes = aggregate_extremes(result)
    except AnalysisError as exc:
        parser.error(str(exc))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "return_bucket_correlation.csv"
    result.to_csv(table_path, index=False)

    summary = {"overall": overall, "extremes": extremes}
    summary_path = output_dir / "return_bucket_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Rows analysed: {overall['rows']}")
    print(f"IC column: {cols.ic_column}, return column: {cols.return_column}")
    print(f"Bucket table -> {table_path}")
    print(f"Summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
