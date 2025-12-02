#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select a diverse set of strategies from selected_top500.csv using Jaccard overlap control,
and generate a live_strategies.csv with capital allocation and rebalance frequency.

Usage:
  python scripts/select_diverse_strategies.py \
    --run-dir etf_strategy/results/run_YYYYMMDD_HHMMSS \
    --target 12 --threshold 0.6 --capital 1000000 --cash-buffer 0.15 --rebalance-freq 8

Outputs:
  {run_dir}/selection/live/live_strategies.csv
  (columns: combo_key, capital_alloc, capital_alloc_ratio, rebalance_freq_days,
            calibrated_sharpe_pred, stability_score)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd


def _detect_col(cols: List[str], candidates: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _split_combo_key(key: str) -> List[str]:
    if not isinstance(key, str):
        return []
    # combo key expected like: 'RSI_14 + SLOPE_20D + VOL_RATIO_20D'
    return [x.strip() for x in key.split("+") if x.strip()]


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def select_diverse(df: pd.DataFrame, combo_col: str, score_col: str | None, tie_col: str | None,
                   target: int, threshold: float) -> pd.DataFrame:
    # Sort by score desc, then stability desc if available
    sort_cols = []
    if score_col:
        sort_cols.append((score_col, False))
    if tie_col:
        sort_cols.append((tie_col, False))
    if sort_cols:
        df_sorted = df.sort_values([c for c, _ in sort_cols], ascending=[a for _, a in sort_cols])
    else:
        df_sorted = df

    chosen_rows = []
    chosen_sets: List[Set[str]] = []
    for _, row in df_sorted.iterrows():
        factors = set(_split_combo_key(str(row[combo_col])))
        if chosen_sets:
            max_overlap = max(jaccard(factors, s) for s in chosen_sets)
            if max_overlap >= threshold:
                continue
        chosen_rows.append(row)
        chosen_sets.append(factors)
        if len(chosen_rows) >= target:
            break
    return pd.DataFrame(chosen_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to results/run_*/ directory")
    ap.add_argument("--target", type=int, default=12, help="Number of strategies to select")
    ap.add_argument("--threshold", type=float, default=0.6, help="Max Jaccard overlap to allow (skip if >= threshold)")
    ap.add_argument("--capital", type=float, default=1_000_000.0, help="Total capital in account currency")
    ap.add_argument("--cash-buffer", type=float, default=0.15, help="Fraction of capital to keep as cash buffer (0-1)")
    ap.add_argument("--rebalance-freq", type=int, default=8, help="Rebalance frequency in days (production default)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    sel_dir = run_dir / "selection"
    in_file = sel_dir / "selected_top500.csv"
    if not in_file.exists():
        raise SystemExit(f"Missing input: {in_file}")
    df = pd.read_csv(in_file, low_memory=False)

    combo_col = _detect_col(df.columns.tolist(), ["combo_key", "combo_id", "id", "key"]) or "combo_key"
    score_col = _detect_col(df.columns.tolist(), ["calibrated_sharpe_pred", "calibrated_sharpe_full", "predicted_sharpe", "score"]) 
    stab_col = _detect_col(df.columns.tolist(), ["stability_score", "stability", "robustness"]) 

    chosen = select_diverse(df, combo_col, score_col, stab_col, args.target, args.threshold)
    n = len(chosen)
    if n == 0:
        raise SystemExit("No strategies selected; consider lowering --threshold or check input file.")

    invest_frac = max(0.0, min(1.0, 1.0 - args.cash_buffer))
    alloc_ratio = invest_frac / n
    chosen = chosen.copy()
    chosen["capital_alloc_ratio"] = alloc_ratio
    chosen["capital_alloc"] = alloc_ratio * args.capital
    chosen["rebalance_freq_days"] = int(args.rebalance_freq)

    # Keep only essential columns
    keep_cols = [c for c in [combo_col, "capital_alloc", "capital_alloc_ratio", "rebalance_freq_days", score_col, stab_col] if c in chosen.columns]
    out = chosen[keep_cols].rename(columns={combo_col: "combo_key"})

    out_dir = sel_dir / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "live_strategies.csv"
    out.to_csv(out_file, index=False)
    print(f"✅ Wrote {out_file} ({len(out)} strategies), alloc_ratio={alloc_ratio:.4f}, cash_buffer={args.cash_buffer:.2f}")


if __name__ == "__main__":
    main()
