#!/usr/bin/env python3
"""Quantile analysis of mean_oos_ic vs annual returns.

This script reproduces the "倒U型" observation by binning combos according to
mean out-of-sample IC (or wfo_ic) and summarising the realised annual returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional fallback
    plt = None

from analysis_utils import AnalysisError, ColumnSelection, dropna_pairs, load_datasets, resolve_columns


def _polyfit_summary(x: np.ndarray, y: np.ndarray) -> dict:
    # Fit quadratic curve y = ax^2 + bx + c
    coeffs = np.polyfit(x, y, deg=2)
    a, b, c = coeffs
    # Vertex of parabola (peak) occurs at x = -b/(2a)
    vertex_x = float(-b / (2 * a)) if a != 0 else float("nan")
    vertex_y = float(np.polyval(coeffs, vertex_x)) if a != 0 else float("nan")
    return {
        "coefficients": {"a": float(a), "b": float(b), "c": float(c)},
        "vertex_ic": vertex_x,
        "vertex_return": vertex_y,
    }


def quantile_analysis(
    frame: pd.DataFrame,
    cols: ColumnSelection,
    *,
    quantiles: int,
) -> Tuple[pd.DataFrame, dict]:
    ic = frame[cols.ic_column].to_numpy()
    returns = frame[cols.return_column].to_numpy()

    # Use qcut to guarantee equal-sized bins, fallback to cut if duplicates exist.
    try:
        buckets = pd.qcut(ic, q=quantiles, duplicates="drop")
    except ValueError:
        buckets = pd.cut(ic, bins=quantiles)

    grouped = frame.groupby(buckets, observed=False)
    stats = grouped[cols.return_column].agg(["count", "mean", "std", "min", "max"])
    stats.rename(columns={
        "count": "samples",
        "mean": "return_mean",
        "std": "return_std",
        "min": "return_min",
        "max": "return_max",
    }, inplace=True)

    stats["ic_min"] = grouped[cols.ic_column].min().values
    stats["ic_max"] = grouped[cols.ic_column].max().values
    stats["ic_mean"] = grouped[cols.ic_column].mean().values
    stats = stats.reset_index().rename(columns={cols.ic_column: "ic_bucket"})
    stats.index.name = "bucket_index"

    spearman = float(pd.Series(ic).corr(pd.Series(returns), method="spearman"))
    pearson = float(pd.Series(ic).corr(pd.Series(returns), method="pearson"))

    poly_stats = _polyfit_summary(ic, returns)

    summary = {
        "rows": int(len(frame)),
        "ic_column": cols.ic_column,
        "return_column": cols.return_column,
        "quantiles": int(len(stats)),
        "global_spearman": spearman,
        "global_pearson": pearson,
        "polyfit": poly_stats,
    }
    return stats, summary


def maybe_plot(
    frame: pd.DataFrame,
    cols: ColumnSelection,
    stats: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if plt is None:
        return None
    out_path = output_dir / "ic_return_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample for scatter to keep file reasonable if dataset is huge.
    sample = frame[[cols.ic_column, cols.return_column]].sample(
        n=min(5000, len(frame)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample[cols.ic_column], sample[cols.return_column], s=10, alpha=0.3, label="samples")

    sorted_ic = np.linspace(frame[cols.ic_column].min(), frame[cols.ic_column].max(), 200)
    coeffs = np.polyfit(frame[cols.ic_column], frame[cols.return_column], deg=2)
    ax.plot(sorted_ic, np.polyval(coeffs, sorted_ic), color="black", linewidth=2, label="quadratic fit")

    # Overlay quantile means.
    ax.scatter(stats["ic_mean"], stats["return_mean"], color="red", s=50, label="quantile means")

    ax.set_title("mean_oos_ic vs annual return")
    ax.set_xlabel(cols.ic_column)
    ax.set_ylabel(cols.return_column)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse mean_oos_ic vs annual returns")
    parser.add_argument("--input", nargs="+", required=True, help="CSV/Parquet files or glob patterns")
    parser.add_argument("--ic-col", help="Explicit IC column name", default=None)
    parser.add_argument("--return-col", help="Explicit return column name", default=None)
    parser.add_argument("--quantiles", type=int, default=10, help="Number of quantile buckets (default: 10)")
    parser.add_argument("--output-dir", default="analysis_outputs/ic_curve", help="Where to write summaries")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation")
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
        stats, summary = quantile_analysis(frame, cols, quantiles=args.quantiles)
    except AnalysisError as exc:
        parser.error(str(exc))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "ic_quantile_stats.csv"
    stats.to_csv(stats_path, index=False)

    summary_path = output_dir / "ic_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_path = None
    if not args.no_plot:
        plot_path = maybe_plot(frame, cols, stats, output_dir)

    print(f"Rows analysed: {summary['rows']}")
    print(f"IC column: {cols.ic_column}, return column: {cols.return_column}")
    print(f"Global Spearman: {summary['global_spearman']:.4f}, Pearson: {summary['global_pearson']:.4f}")
    if plot_path:
        print(f"Saved plot to {plot_path}")
    print(f"Quantile table -> {stats_path}")
    print(f"Summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
