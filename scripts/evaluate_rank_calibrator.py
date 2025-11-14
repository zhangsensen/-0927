#!/usr/bin/env python3
"""Compare baseline and calibrated rankings and emit gating metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


RETURN_CANDIDATES = [
    "annual_return_net",
    "annual_ret_net",
    "annualized_return_net",
    "annual_return",
    "annual_ret",
    "annualized_return",
]

SHARPE_CANDIDATES = [
    "sharpe",
    "sharpe_net",
    "realized_sharpe",
    "sharpe_ratio",
    "oos_sharpe",
    "sr",
]

SCORE_CANDIDATES = [
    "rank_score",
    "calibrated_sharpe_pred",
    "ml_score",
    "mean_oos_ic",
    "baseline_score",
]

RANK_ASCENDING = ["rank_blend", "rank", "order"]

TOP_LIST = (100, 200, 500)


def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
	lower = {c.lower(): c for c in columns}
	for name in candidates:
		if name in columns:
			return name
		low = name.lower()
		if low in lower:
			return lower[low]
	raise ValueError(f"None of the columns {list(candidates)} found in dataset")


def _load_frame(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"File not found: {path}")
	frame = pd.read_parquet(path)
	return frame


def _expand_paths(patterns: Sequence[str]) -> List[Path]:
	files: List[Path] = []
	for raw in patterns:
		candidate = Path(raw)
		if any(ch in raw for ch in "*?[]"):
			files.extend(sorted(Path().glob(raw)))
		elif candidate.is_dir():
			files.extend(sorted(candidate.glob("*.parquet")))
			files.extend(sorted(candidate.glob("*.csv")))
		elif candidate.exists():
			files.append(candidate)
	return files


def _load_metrics_tables(patterns: Optional[Sequence[str]]) -> Optional[pd.DataFrame]:
	if not patterns:
		return None
	paths = _expand_paths(patterns)
	if not paths:
		raise FileNotFoundError("No metric files matched the provided patterns")
	frames: List[pd.DataFrame] = []
	for path in paths:
		suffix = path.suffix.lower()
		if suffix == ".csv":
			frames.append(pd.read_csv(path))
		elif suffix in {".parquet", ".pq"}:
			frames.append(pd.read_parquet(path))
	if not frames:
		raise ValueError("Metric files must be CSV or Parquet")
	metrics = pd.concat(frames, ignore_index=True)
	if "combo" not in metrics.columns:
		raise ValueError("Metrics dataset must include 'combo' column for merging")
	metrics = metrics.drop_duplicates(subset="combo", keep="last")
	return metrics


def _resolve_score_column(frame: pd.DataFrame) -> Tuple[str, bool]:
	for col in RANK_ASCENDING:
		if col in frame.columns:
			return col, True
	for col in SCORE_CANDIDATES:
		if col in frame.columns:
			return col, False
	raise ValueError("Unable to locate score column (rank_score / rank_blend / mean_oos_ic)")


def _sort_frame(frame: pd.DataFrame, score_col: str, ascending: bool) -> pd.DataFrame:
	return frame.sort_values(score_col, ascending=ascending).reset_index(drop=True)


def _compute_topk_stats(frame: pd.DataFrame, ret_col: str, sharpe_col: str) -> Dict[int, Dict[str, float]]:
	stats: Dict[int, Dict[str, float]] = {}
	for k in TOP_LIST:
		subset = frame.head(k)
		if subset.empty:
			stats[k] = {"annual": float("nan"), "sharpe": float("nan"), "count": 0}
			continue
		stats[k] = {
			"annual": float(subset[ret_col].mean()),
			"sharpe": float(subset[sharpe_col].mean()),
			"count": int(len(subset)),
		}
	return stats


def _compute_spearman(frame: pd.DataFrame, score_col: str, ascending: bool, ret_col: str) -> float:
	scores = frame[score_col].astype(float)
	returns = frame[ret_col].astype(float)
	mask = ~(scores.isna() | returns.isna())
	if mask.sum() < 2:
		return float("nan")
	score_series = scores[mask]
	if ascending:
		score_series = -score_series
	return float(score_series.rank(method="dense").corr(returns[mask].rank(method="dense"), method="spearman"))


def _maybe_merge_metrics(frame: pd.DataFrame, metrics: Optional[pd.DataFrame]) -> pd.DataFrame:
	if metrics is None:
		return frame
	add_cols = [c for c in metrics.columns if c != "combo" and c not in frame.columns]
	if not add_cols:
		return frame
	return frame.merge(metrics[["combo", *add_cols]], on="combo", how="left")


def evaluate(
	baseline_path: Path,
	calibrated_path: Path,
	metrics_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
	baseline = _load_frame(baseline_path)
	calibrated = _load_frame(calibrated_path)

	metrics_df = _load_metrics_tables(metrics_patterns)
	baseline = _maybe_merge_metrics(baseline, metrics_df)
	calibrated = _maybe_merge_metrics(calibrated, metrics_df)

	ret_col = _detect_column(baseline.columns, RETURN_CANDIDATES)
	sharpe_col = _detect_column(baseline.columns, SHARPE_CANDIDATES)

	baseline = baseline.dropna(subset=[ret_col, sharpe_col]).reset_index(drop=True)
	calibrated = calibrated.dropna(subset=[ret_col, sharpe_col]).reset_index(drop=True)

	base_score_col, base_asc = _resolve_score_column(baseline)
	cal_score_col, cal_asc = _resolve_score_column(calibrated)

	baseline_sorted = _sort_frame(baseline, base_score_col, base_asc)
	calibrated_sorted = _sort_frame(calibrated, cal_score_col, cal_asc)

	baseline_stats = _compute_topk_stats(baseline_sorted, ret_col, sharpe_col)
	calibrated_stats = _compute_topk_stats(calibrated_sorted, ret_col, sharpe_col)

	delta = {
		k: {
			"annual": float(calibrated_stats[k]["annual"] - baseline_stats[k]["annual"]),
			"sharpe": float(calibrated_stats[k]["sharpe"] - baseline_stats[k]["sharpe"]),
		}
		for k in TOP_LIST
	}

	spearman_baseline = _compute_spearman(baseline_sorted, base_score_col, base_asc, ret_col)
	spearman_calibrated = _compute_spearman(calibrated_sorted, cal_score_col, cal_asc, ret_col)

	passed = delta[100]["annual"] > 0 and delta[100]["sharpe"] > 0

	return {
		"baseline_path": str(baseline_path),
		"calibrated_path": str(calibrated_path),
		"return_column": ret_col,
		"sharpe_column": sharpe_col,
		"baseline": baseline_stats,
		"calibrated": calibrated_stats,
		"delta": delta,
		"spearman": {
			"baseline": spearman_baseline,
			"calibrated": spearman_calibrated,
			"delta": float(spearman_calibrated - spearman_baseline),
		},
		"gating": {
			"passed": bool(passed),
			"rule": "Top100 annual & Sharpe must both improve",
		},
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate calibrated ranking gains vs baseline")
	parser.add_argument("--baseline", type=Path, required=True, help="Path to ranking_baseline.parquet")
	parser.add_argument("--calibrated", type=Path, required=True, help="Path to ranking_lightgbm.parquet")
	parser.add_argument(
		"--metrics",
		nargs="*",
		default=None,
		help="Optional CSV/Parquet metrics with realized annual return and Sharpe",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("results/calibrator_gbdt_full_eval.json"),
		help="Where to write gating JSON",
	)
	args = parser.parse_args()

	summary = evaluate(
		args.baseline.resolve(),
		args.calibrated.resolve(),
		metrics_patterns=args.metrics,
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
	print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
