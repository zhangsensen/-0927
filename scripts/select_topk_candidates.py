#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select TopK candidates from a WFO run by calibrated prediction, with tie-breaks,
optionally compute realized stats from latest full backtest CSV for spot-check.

Outputs under: <run_dir>/selection/
- candidate_top2000.csv
- selected_top500.csv
- selected_top200.csv
- selection_summary.json
- spotcheck_20.csv (optional if backtest CSV found)
- realized_metrics.json (optional if backtest CSV found)

Usage:
  python scripts/select_topk_candidates.py --run-dir <run_dir> [--k2000 2000 --k500 500 --k200 200]
If --run-dir is omitted, tries to use etf_strategy/results/run_latest or .latest_run.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DEFAULT_RUN_LATEST = RESULTS_DIR / "run_latest"


def _resolve_run_dir(run_dir_arg: Optional[str]) -> Path:
    if run_dir_arg:
        rd = Path(run_dir_arg)
        if not rd.is_dir():
            raise FileNotFoundError(f"run_dir not found: {rd}")
        return rd
    # try run_latest file
    if DEFAULT_RUN_LATEST.exists():
        txt = DEFAULT_RUN_LATEST.read_text().strip()
        if txt:
            rd = Path(txt)
            if rd.is_dir():
                return rd
    # fallback to newest results/run_*
    candidates = sorted((RESULTS_DIR).glob("run_*"))
    if not candidates:
        raise FileNotFoundError("No run_* directory found under results/")
    return candidates[-1]


def _find_full_backtest_csv() -> Optional[Path]:
    base = ROOT / "results_combo_wfo"
    if not base.exists():
        return None
    # pick latest subdir by mtime
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in subdirs:
        # prefer *_full.csv
        cands = sorted(d.glob("*full.csv"))
        if cands:
            return cands[0]
        # fallback any top* csv
        cands = sorted(d.glob("*.csv"))
        if cands:
            return cands[0]
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_top(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # primary: calibrated_sharpe_pred desc, tie-break: stability_score desc, then mean_oos_ic desc if exists
    sort_cols = ["calibrated_sharpe_pred", "stability_score"]
    ascending = [False, False]
    if "mean_oos_ic" in df.columns:
        sort_cols.append("mean_oos_ic")
        ascending.append(False)
    return df.sort_values(sort_cols, ascending=ascending).head(k).copy()


def _realized_metrics(rb: pd.DataFrame, key_col: str, subsets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rb2 = rb.copy()
    rb2[key_col] = rb2[key_col].str.strip()
    for name, sub in subsets.items():
        keys = set(sub[key_col].astype(str).str.strip())
        part = rb2[rb2[key_col].isin(keys)]
        m = {
            "n": int(len(part)),
            "sharpe_mean": float(part["sharpe"].mean()) if not part.empty else None,
            "sharpe_median": float(part["sharpe"].median()) if not part.empty else None,
            "max_dd_median": float(part["max_dd"].median()) if not part.empty and "max_dd" in part.columns else None,
            "annual_ret_mean": float(part["annual_ret"].mean()) if not part.empty and "annual_ret" in part.columns else None,
        }
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--k2000", type=int, default=2000)
    ap.add_argument("--k500", type=int, default=500)
    ap.add_argument("--k200", type=int, default=200)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    allp = run_dir / "all_combos.parquet"
    if not allp.exists():
        raise FileNotFoundError(f"all_combos.parquet not found in {run_dir}")

    df = pd.read_parquet(allp)
    if "calibrated_sharpe_pred" not in df.columns:
        raise RuntimeError("calibrated_sharpe_pred missing in all_combos; ensure calibrated model is active.")
    if "stability_score" not in df.columns:
        # fallback: create zero stability if absent (shouldn't happen)
        df["stability_score"] = 0.0

    # normalize key
    df["combo_key"] = df["combo"].astype(str).str.strip()

    sel_dir = run_dir / "selection"
    _ensure_dir(sel_dir)

    top2000 = _select_top(df, args.k2000)
    top500 = _select_top(top2000, args.k500)
    top200 = _select_top(top2000, args.k200)

    # save minimal columns for downstream
    keep_cols = [c for c in [
        "combo", "combo_key", "combo_size",
        "calibrated_sharpe_pred", "stability_score", "mean_oos_ic",
        "rebalance_freq", "oos_ir", "positive_rate"
    ] if c in df.columns]

    top2000[keep_cols].to_csv(sel_dir / "candidate_top2000.csv", index=False)
    top500[keep_cols].to_csv(sel_dir / "selected_top500.csv", index=False)
    top200[keep_cols].to_csv(sel_dir / "selected_top200.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "counts": {"top2000": int(len(top2000)), "top500": int(len(top500)), "top200": int(len(top200))},
        "criteria": "sort by calibrated_sharpe_pred desc, tie-break stability_score desc, then mean_oos_ic desc if present",
        "fields": keep_cols,
    }

    # try realized spot-check
    rb_csv = _find_full_backtest_csv()
    if rb_csv and rb_csv.exists():
        rb = pd.read_csv(rb_csv)
        rb["combo_key"] = rb["combo"].astype(str).str.strip()
        summary["realized_csv"] = str(rb_csv)
        summary["realized_metrics"] = _realized_metrics(rb, "combo_key", {
            "top2000": top2000,
            "top500": top500,
            "top200": top200,
        })
        # sample 20 for manual spot-check
        spot = top500.sample(n=min(20, len(top500)), random_state=17)
        spot[["combo", "combo_key"]].to_csv(sel_dir / "spotcheck_20.csv", index=False)
    else:
        summary["realized_csv"] = None

    (sel_dir / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("✅ Selection generated:", sel_dir)


if __name__ == "__main__":
    main()
