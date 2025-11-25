#!/usr/bin/env python3
"""Compare baseline vs ML backtest outputs across multiple Top-K bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DEFAULT_TOPKS = (50, 100, 200, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two ranking backtest result files.")
    parser.add_argument("baseline", type=str, help="CSV/Parquet file with baseline backtest results")
    parser.add_argument("candidate", type=str, help="CSV/Parquet file with ML backtest results")
    parser.add_argument(
        "--topk",
        type=str,
        default="50,100,200,500,1000",
        help="Comma-separated list of K values to evaluate (default: 50,100,200,500,1000)",
    )
    parser.add_argument("--combo-column", type=str, default="combo", help="Column containing combo identifiers")
    parser.add_argument("--return-column", type=str, default="annual_ret", help="Column for annual return metric")
    parser.add_argument("--sharpe-column", type=str, default="sharpe", help="Column for Sharpe ratio")
    parser.add_argument("--max-dd-column", type=str, default="max_dd", help="Column for max drawdown")
    parser.add_argument(
        "--baseline-sort-column",
        type=str,
        default=None,
        help="Optional column to sort the baseline file before slicing Top-K",
    )
    parser.add_argument(
        "--candidate-sort-column",
        type=str,
        default=None,
        help="Optional column to sort the candidate file before slicing Top-K",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort columns in ascending order (default: descending). Applies to both datasets.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the comparison JSON report")
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="Optional path to save a Markdown summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topks = _parse_topk_values(args.topk)

    baseline = _load_frame(Path(args.baseline))
    candidate = _load_frame(Path(args.candidate))

    baseline = _optional_sort(baseline, args.baseline_sort_column, ascending=args.ascending)
    candidate = _optional_sort(candidate, args.candidate_sort_column, ascending=args.ascending)

    metrics = {
        "return": args.return_column,
        "sharpe": args.sharpe_column,
        "max_dd": args.max_dd_column,
    }

    summary = _compare(baseline, candidate, args.combo_column, metrics, topks)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.output_markdown:
        markdown = _render_markdown(summary, metrics)
        Path(args.output_markdown).expanduser().resolve().write_text(markdown, encoding="utf-8")


def _parse_topk_values(text: str) -> List[int]:
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid --topk token: {token}") from exc
    return values or list(DEFAULT_TOPKS)


def _load_frame(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    if resolved.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported file extension: {resolved.suffix}")


def _optional_sort(frame: pd.DataFrame, column: str | None, *, ascending: bool) -> pd.DataFrame:
    if column and column in frame.columns:
        return frame.sort_values(by=column, ascending=ascending).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _compare(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    combo_column: str,
    metrics: Dict[str, str],
    topks: Iterable[int],
) -> Dict:
    report: Dict[str, Dict] = {
        "config": {
            "combo_column": combo_column,
            "metrics": metrics,
            "topks": list(topks),
            "baseline_rows": len(baseline),
            "candidate_rows": len(candidate),
        },
        "bands": {},
    }

    for k in topks:
        base_slice = baseline.head(k)
        cand_slice = candidate.head(k)
        if base_slice.empty or cand_slice.empty:
            continue
        base_stats = _summarize_slice(base_slice, metrics)
        cand_stats = _summarize_slice(cand_slice, metrics)
        overlap = _compute_overlap(base_slice, cand_slice, combo_column)
        deltas = {
            key: cand_stats[key] - base_stats[key]
            for key in ("return_mean", "return_median", "sharpe_mean", "max_dd_mean", "positive_ratio")
            if key in cand_stats and key in base_stats
        }
        report["bands"][f"top{k}"] = {
            "baseline": base_stats,
            "candidate": cand_stats,
            "overlap": overlap,
            "delta": deltas,
        }
    return report


def _summarize_slice(frame: pd.DataFrame, metrics: Dict[str, str]) -> Dict[str, float]:
    stats: Dict[str, float] = {"n": len(frame)}
    ret_col = metrics.get("return")
    sharpe_col = metrics.get("sharpe")
    maxdd_col = metrics.get("max_dd")

    if ret_col not in frame.columns:
        raise KeyError(f"Return column '{ret_col}' not found in dataset")
    stats["return_mean"] = float(frame[ret_col].mean())
    stats["return_median"] = float(frame[ret_col].median())
    stats["positive_ratio"] = float((frame[ret_col] > 0).mean())

    if sharpe_col not in frame.columns:
        raise KeyError(f"Sharpe column '{sharpe_col}' not found in dataset")
    stats["sharpe_mean"] = float(frame[sharpe_col].mean())

    if maxdd_col not in frame.columns:
        raise KeyError(f"Max drawdown column '{maxdd_col}' not found in dataset")
    stats["max_dd_mean"] = float(frame[maxdd_col].mean())
    return stats


def _compute_overlap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    combo_column: str,
) -> Dict[str, float]:
    if combo_column not in baseline.columns or combo_column not in candidate.columns:
        return {"count": 0, "ratio": 0.0}
    base_set = set(baseline[combo_column].astype(str))
    cand_set = set(candidate[combo_column].astype(str))
    intersection = base_set & cand_set
    ratio = len(intersection) / max(1, len(base_set))
    return {"count": len(intersection), "ratio": ratio}


def _render_markdown(summary: Dict, metrics: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append("# Backtest Comparison Report\n")
    lines.append("| Band | Metric | Baseline | Candidate | Δ |\n")
    lines.append("|------|--------|----------|-----------|----|\n")
    for band, payload in summary.get("bands", {}).items():
        base = payload["baseline"]
        cand = payload["candidate"]
        delta = payload["delta"]
        lines.append(
            f"| {band} | Return (mean) | {base['return_mean']:.4f} | {cand['return_mean']:.4f} | {delta.get('return_mean', 0.0):+.4f} |"
        )
        lines.append(
            f"| {band} | Return (median) | {base['return_median']:.4f} | {cand['return_median']:.4f} | {delta.get('return_median', 0.0):+.4f} |"
        )
        lines.append(
            f"| {band} | Sharpe (mean) | {base['sharpe_mean']:.4f} | {cand['sharpe_mean']:.4f} | {delta.get('sharpe_mean', 0.0):+.4f} |"
        )
        lines.append(
            f"| {band} | MaxDD (mean) | {base['max_dd_mean']:.4f} | {cand['max_dd_mean']:.4f} | {delta.get('max_dd_mean', 0.0):+.4f} |"
        )
        lines.append(
            f"| {band} | Positive ratio | {base['positive_ratio']:.2%} | {cand['positive_ratio']:.2%} | {delta.get('positive_ratio', 0.0):+.2%} |"
        )
        lines.append("|------|--------|----------|-----------|----|")
    lines.append("\n_Columns used:_ "+", ".join(f"{k}={v}" for k, v in metrics.items()))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
