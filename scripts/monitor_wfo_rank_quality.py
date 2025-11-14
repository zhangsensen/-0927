#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor WFO ranking quality metrics against thresholds.
Metrics:
  - Pearson & Spearman between calibrated prediction and realized sharpe
  - Precision@K for K in {500, 2000}
  - Top500 HHI (equal-weight) concentration

Sources:
  1. Existing comparison/calibrated_vs_realized_metrics.json if present (fast path)
  2. Recompute from all_combos.parquet + full backtest CSV if not present

Exit codes:
  0 -> All thresholds satisfied
  2 -> Threshold breach detected (details printed)
  3 -> Metrics could not be computed (missing files)

Usage:
  python scripts/monitor_wfo_rank_quality.py --run-dir results/run_YYYYMMDD_HHMMSS \
     --thresholds config/monitor_thresholds.yaml [--backtest-csv PATH]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import sys

import numpy as np
import pandas as pd
import yaml

K_LIST = [500, 2000]


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _detect_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _compute_hhi(n: int) -> float:
    if n <= 0:
        return float("nan")
    weights = np.ones(n, dtype=float) / n
    return float(np.sum(weights ** 2))


def _precision_at_k(df: pd.DataFrame, pred_col: str, realized_col: str, k: int) -> float:
    if k > len(df):
        return float("nan")
    top_pred = df.nlargest(k, pred_col)
    top_real = df.nlargest(k, realized_col)
    inter = set(top_pred.index) & set(top_real.index)
    return len(inter) / float(k)


def _load_metrics_from_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    js = run_dir / "comparison" / "calibrated_vs_realized_metrics.json"
    if js.exists():
        try:
            return json.loads(js.read_text())
        except Exception:
            return None
    return None


def _recompute_metrics(run_dir: Path, backtest_csv: Optional[str]) -> Optional[Dict[str, Any]]:
    # Expect all_combos.parquet and full backtest CSV path from selection_summary.json or override
    all_parquet = run_dir / "all_combos.parquet"
    if not all_parquet.exists():
        return None
    try:
        df_all = pd.read_parquet(all_parquet)
    except Exception:
        return None
    # detect columns
    pred_col = _detect_col(df_all.columns, ["calibrated_sharpe_pred", "calibrated_sharpe_full", "predicted_sharpe", "score"])
    if pred_col is None:
        return None
    # backtest CSV
    bt_path = backtest_csv
    if bt_path is None:
        # attempt from selection_summary.json
        sel_summary = run_dir / "selection" / "selection_summary.json"
        if sel_summary.exists():
            try:
                bt_path = json.loads(sel_summary.read_text()).get("backtest_csv")
            except Exception:
                bt_path = None
    if bt_path is None:
        return None
    bt_file = Path(bt_path)
    if not bt_file.exists():
        return None
    try:
        df_bt = pd.read_csv(bt_file, low_memory=False)
    except Exception:
        return None
    realized_col = _detect_col(df_bt.columns, ["realized_sharpe", "sharpe", "oos_sharpe", "backtest_sharpe", "realized_sharpe_ratio", "sr"])
    id_all = _detect_col(df_all.columns, ["combo_id", "id", "combo_key", "key"])
    id_bt = _detect_col(df_bt.columns, ["combo_id", "id", "combo_key", "key"])
    if realized_col is None or id_all is None or id_bt is None:
        return None
    merged = df_all[[id_all, pred_col]].merge(df_bt[[id_bt, realized_col]], left_on=id_all, right_on=id_bt, how="inner").dropna()
    if merged.empty:
        return None
    pearson = float(np.corrcoef(merged[pred_col].to_numpy(), merged[realized_col].to_numpy())[0, 1])
    spearman = merged[pred_col].rank(ascending=False).corr(merged[realized_col].rank())
    merged = merged.sort_values(pred_col, ascending=False)
    precision_at = {k: _precision_at_k(merged.set_index(id_bt), pred_col, realized_col, k) for k in K_LIST if k <= len(merged)}
    metrics = {
        "pearson": pearson,
        "spearman": float(spearman),
        "precision_at": precision_at,
        "top500_hhi": _compute_hhi(500),  # equal-weight theoretical baseline
        "count": int(len(merged)),
        "source": "recomputed",
    }
    return metrics


def evaluate(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    breaches = []
    def chk(name: str, value: float, op: str, thresh: float):
        if np.isnan(value):
            breaches.append({"metric": name, "status": "nan", "threshold": thresh})
            return
        ok = value >= thresh if op == ">=" else value <= thresh
        if not ok:
            breaches.append({"metric": name, "value": value, "threshold": thresh, "op": op})

    chk("pearson", metrics.get("pearson", float("nan")), ">=", thresholds.get("pearson_min", 0.85))
    chk("spearman", metrics.get("spearman", float("nan")), ">=", thresholds.get("spearman_min", 0.80))
    chk("precision_at_500", metrics.get("precision_at", {}).get(500, float("nan")), ">=", thresholds.get("precision_at_500_min", 0.30))
    chk("precision_at_2000", metrics.get("precision_at", {}).get(2000, float("nan")), ">=", thresholds.get("precision_at_2000_min", 0.70))
    # HHI breaches if higher than max
    hhi_val = metrics.get("top500_hhi", float("nan"))
    max_hhi = thresholds.get("top500_hhi_max", 0.010)
    if not np.isnan(hhi_val) and hhi_val > max_hhi:
        breaches.append({"metric": "top500_hhi", "value": hhi_val, "threshold": max_hhi, "op": "<="})
    return {"breaches": breaches, "passed": len(breaches) == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--backtest-csv", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(3)

    thresholds = _read_yaml(Path(args.thresholds))
    metrics = _load_metrics_from_json(run_dir)
    if metrics is None:
        metrics = _recompute_metrics(run_dir, args.backtest_csv)
    if metrics is None:
        print("Unable to compute metrics (missing files).", file=sys.stderr)
        sys.exit(3)

    # Normalize possible key naming from precomputed JSONs
    # Support both precision_at and precision_at_k; top500_hhi and top500_hhi_pred; count and n
    if isinstance(metrics, dict):
        if "precision_at" not in metrics and "precision_at_k" in metrics:
            metrics["precision_at"] = metrics.get("precision_at_k", {})
        if "top500_hhi" not in metrics and "top500_hhi_pred" in metrics:
            metrics["top500_hhi"] = metrics.get("top500_hhi_pred")
        if "count" not in metrics and "n" in metrics:
            metrics["count"] = metrics.get("n")
        # Convert precision dict keys to int if they are strings
        if isinstance(metrics.get("precision_at"), dict):
            try:
                metrics["precision_at"] = {int(k): float(v) for k, v in metrics["precision_at"].items()}
            except Exception:
                pass

    eval_res = evaluate(metrics, thresholds)
    print(json.dumps({"metrics": metrics, "evaluation": eval_res}, ensure_ascii=False, indent=2))
    if eval_res["passed"]:
        print("✅ Rank quality within thresholds.")
        sys.exit(0)
    else:
        print("❌ Threshold breach detected.")
        sys.exit(2)


if __name__ == "__main__":
    main()
