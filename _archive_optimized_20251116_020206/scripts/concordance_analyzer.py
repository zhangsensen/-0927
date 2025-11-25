#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concordance analyzer for WFO ranking vs realized backtest results.

Inputs (by default auto-discovers latest strict Top500 run dir):
  --results-dir: path like results_combo_wfo/<wfo_ts>_top500_strict_allfreq
  --k-list: comma-separated K values for Precision/Recall (default: 50,100,200)
  --z: z-score for LCB (default: 1.64 -> ~90% confidence)
  --near-freqs: comma-separated freq list for consistency penalty (default: 6,7,8,10,12,16,24)

Outputs will be written into the same results-dir:
  - concordance_report.json: all metrics + decile calibration
  - calibrated_ranking_lcb90.csv: LCB(Sharpe) based calibrated ranking (with optional freq penalty)
  - missed_gems.csv: actual top-decile not captured by WFO TopK (K in k-list)

Assumptions:
  - Top500 8D results CSV: top500_backtest_by_ic_*_full.csv (includes sharpe, winning_days, losing_days)
  - All-frequency CSV: all_freq_scan_*.csv (for frequency consistency penalty)
  - Whitelist file: etf_rotation_optimized/results/run_<wfo_ts>/whitelist_8d_wfo_qualified_<wfo_ts>.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def find_latest_results_dir(base: str = "results_combo_wfo") -> str:
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Results base dir not found: {base}")
    cands = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p) and name.endswith("_top500_strict_allfreq"):
            cands.append((os.path.getmtime(p), p))
    if not cands:
        raise FileNotFoundError("No *_top500_strict_allfreq results dirs found under results_combo_wfo/")
    cands.sort()
    return cands[-1][1]


def infer_wfo_ts_from_dir(results_dir: str) -> str:
    # Expect dirname like: results_combo_wfo/20251106_184657_top500_strict_allfreq
    base = os.path.basename(results_dir.rstrip("/"))
    m = re.match(r"(\d{8}_\d{6})_", base)
    if not m:
        raise ValueError(f"Cannot infer wfo_ts from dir name: {base}")
    return m.group(1)


def load_whitelist_rank_map(wfo_ts: str) -> Dict[str, int]:
    # Try multiple candidate paths for robustness
    candidates = [
        os.path.join("etf_rotation_optimized", "results", f"run_{wfo_ts}", f"whitelist_8d_wfo_qualified_{wfo_ts}.txt"),
        os.path.join("results", f"run_{wfo_ts}", f"whitelist_8d_wfo_qualified_{wfo_ts}.txt"),
        os.path.join("etf_rotation_optimized", "results", f"run_{wfo_ts}", f"whitelist_{wfo_ts}.txt"),
    ]
    wl_path = None
    for c in candidates:
        if os.path.exists(c):
            wl_path = c
            break
    if wl_path is None:
        raise FileNotFoundError(f"Whitelist file not found. Tried: {candidates}")

    rank_map: Dict[str, int] = {}
    with open(wl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            combo = line.strip()
            if not combo:
                continue
            # rank starts at 1
            rank_map[combo] = i + 1
    return rank_map


def compute_lcb_sharpe(df8: pd.DataFrame, z: float) -> pd.Series:
    # Use standard error of Sharpe: sqrt((1 + 0.5 * S^2) / T)
    # Approximate T as total trading days = winning_days + losing_days
    # Guard against zero/NaN
    s = df8["sharpe"].astype(float).fillna(0.0)
    t = (df8["winning_days"].astype(float).fillna(0.0) + df8["losing_days"].astype(float).fillna(0.0)).clip(lower=1.0)
    se = np.sqrt((1.0 + 0.5 * s ** 2) / t)
    lcb = s - z * se
    return lcb


def rank_within_series(values: pd.Series, ascending: bool = False) -> pd.Series:
    # Rank 1 = best (highest by default)
    return values.rank(method="min", ascending=ascending).astype(int)


def compute_precision_recall(wfo_rank: pd.Series, actual_rank: pd.Series, ks: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
    # Precision@K: |WFO_topK ∩ Actual_topK| / K
    # Recall@K (top-decile recall variant): |Actual_topK ∩ WFO_topK| / |Actual_topK|
    res_p = {}
    res_r = {}
    n = len(wfo_rank)
    actual_top_decile_k = max(1, int(0.1 * n))
    for k in ks:
        wfo_topk = set(wfo_rank.nsmallest(k).index)
        actual_topk = set(actual_rank.nsmallest(k).index)
        inter = len(wfo_topk & actual_topk)
        precision = inter / float(k)
        res_p[k] = precision

        # top-decile recall measured at WFO@K vs Actual@TopDecileK
        actual_top_decile = set(actual_rank.nsmallest(actual_top_decile_k).index)
        inter2 = len(wfo_topk & actual_top_decile)
        recall = inter2 / float(actual_top_decile_k)
        res_r[k] = recall
    return res_p, res_r


def compute_pairwise_accuracy(wfo_rank: pd.Series, actual_rank: pd.Series, window_sizes: List[int]) -> Dict[int, float]:
    # For pairs (i, j) with |i - j| <= w, count consistency of ordering in actual vs WFO
    idx = wfo_rank.sort_values().index.tolist()
    pos = {cid: i for i, cid in enumerate(idx)}
    actual_pos = actual_rank.rank(method="min", ascending=True).astype(int)  # lower rank is better
    actual_order = {cid: int(actual_pos.loc[cid]) for cid in actual_pos.index}
    res = {}
    n = len(idx)
    for w in window_sizes:
        total = 0
        ok = 0
        for i in range(n):
            for j in range(i + 1, min(n, i + w + 1)):
                ci = idx[i]
                cj = idx[j]
                total += 1
                if actual_order[ci] < actual_order[cj]:
                    ok += 1
        res[w] = ok / total if total else 0.0
    return res


def frequency_consistency_penalty(allfreq: pd.DataFrame, near_freqs: List[int]) -> pd.Series:
    # Compute std of ranks across the selected frequencies (lower std => more consistent)
    df = allfreq[allfreq["freq"].isin(near_freqs)].copy()
    # rank per freq by sharpe descending
    df["rank_in_freq"] = df.groupby("freq")["sharpe"].rank(method="min", ascending=False)
    # aggregate std across freqs per combo
    agg = df.groupby("combo")["rank_in_freq"].agg(["mean", "std", "count"]).rename(columns={"std": "std_rank"})
    return agg["std_rank"].fillna(0.0)


def main():
    parser = argparse.ArgumentParser(description="Analyze WFO vs realized results concordance and produce calibrated ranking")
    parser.add_argument("--results-dir", type=str, default=None, help="results_combo_wfo/<wfo_ts>_top500_strict_allfreq")
    parser.add_argument("--k-list", type=str, default="50,100,200", help="comma-separated K for metrics")
    parser.add_argument("--z", type=float, default=1.64, help="z-score for LCB (e.g., 1.64 ~ 90% confidence)")
    parser.add_argument("--near-freqs", type=str, default="6,7,8,10,12,16,24", help="freqs for consistency penalty")
    parser.add_argument("--lambda-freq", type=float, default=0.001, help="coefficient for frequency consistency penalty")
    parser.add_argument("--wfo-score-file", type=str, default=None, help="Path to WFO ranking file (csv/parquet) containing learned_score or wfo_score")
    parser.add_argument("--wfo-score-field", type=str, default="learned_score", help="Score field name to build true WFO ranks (descending)")
    args = parser.parse_args()

    results_dir = args.results_dir or find_latest_results_dir()
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    wfo_ts = infer_wfo_ts_from_dir(results_dir)
    wl_rank_map = load_whitelist_rank_map(wfo_ts)

    # Build true WFO ranking from provided score file (preferred) instead of whitelist order
    true_rank_map = None
    if args.wfo_score_file:
        score_path = args.wfo_score_file
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"WFO score file not found: {score_path}")
        if score_path.endswith('.parquet'):
            s_df = pd.read_parquet(score_path)
        else:
            s_df = pd.read_csv(score_path)
        if args.wfo_score_field not in s_df.columns:
            raise ValueError(f"Score field {args.wfo_score_field} not in columns: {s_df.columns.tolist()}")
        s_df = s_df[["combo", args.wfo_score_field]].dropna()
        s_df["true_wfo_rank"] = s_df[args.wfo_score_field].rank(method="min", ascending=False).astype(int)
        true_rank_map = dict(zip(s_df["combo"], s_df["true_wfo_rank"]))

    # Locate CSVs
    top500_csv = None
    allfreq_csv = None
    for name in os.listdir(results_dir):
        if name.startswith("top500_backtest_by_ic_") and name.endswith("_full.csv"):
            top500_csv = os.path.join(results_dir, name)
        if name.startswith("all_freq_scan_") and name.endswith(".csv"):
            allfreq_csv = os.path.join(results_dir, name)
    if top500_csv is None:
        raise FileNotFoundError("Top500 full CSV not found in results_dir")
    if allfreq_csv is None:
        raise FileNotFoundError("all_freq_scan CSV not found in results_dir")

    df8 = pd.read_csv(top500_csv)
    df_all = pd.read_csv(allfreq_csv)

    # Derive identifiers
    # Use combo string as unique id
    df8["combo_id"] = df8["combo"].astype(str)
    # WFO rank from whitelist (lower is better)
    if true_rank_map is not None:
        df8["wfo_rank"] = df8["combo_id"].map(true_rank_map)
    else:
        # fallback: whitelist order (may not represent true ranking)
        df8["wfo_rank"] = df8["combo_id"].map(wl_rank_map)
    if df8["wfo_rank"].isna().any():
        missing = df8[df8["wfo_rank"].isna()]["combo_id"].unique().tolist()
        raise ValueError(f"Some combos not found in whitelist map: {missing[:5]} ... total {len(missing)}")

    # Actual rank by realized Sharpe (8D fixed freq file)
    df8["actual_rank"] = rank_within_series(df8["sharpe"], ascending=False)

    # LCB for Sharpe (note: uses realized Sharpe; should be interpreted as ex-post stability, not predictive)
    df8["lcb90_sharpe"] = compute_lcb_sharpe(df8, z=args.z)

    # Frequency consistency penalty
    near_freqs = [int(x) for x in args.near_freqs.split(",") if x.strip()]
    std_rank = frequency_consistency_penalty(df_all, near_freqs)
    df8 = df8.merge(std_rank.rename("std_rank_near_freqs"), left_on="combo_id", right_index=True, how="left")
    df8["std_rank_near_freqs"].fillna(df8["std_rank_near_freqs"].median(), inplace=True)
    # Normalize by population size to be scale-invariant
    n_pop = len(df8)
    df8["freq_penalty"] = args.lambda_freq * (df8["std_rank_near_freqs"].astype(float) / max(1, n_pop))

    # Calibrated score = LCB - penalty
    df8["calibrated_score"] = df8["lcb90_sharpe"] - df8["freq_penalty"]
    df8["calibrated_rank"] = rank_within_series(df8["calibrated_score"], ascending=False)

    # Metrics
    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    # Baseline metrics (raw WFO rank vs actual)
    precision_at_k, recall_topdecile_at_k = compute_precision_recall(
        df8.set_index("combo_id")["wfo_rank"], df8.set_index("combo_id")["actual_rank"], ks
    )
    sp, sp_p = spearmanr(df8["wfo_rank"], df8["actual_rank"])
    kt, kt_p = kendalltau(df8["wfo_rank"], df8["actual_rank"])
    pair_acc = compute_pairwise_accuracy(
        df8.set_index("combo_id")["wfo_rank"], df8.set_index("combo_id")["actual_rank"], window_sizes=[1, 5, 10]
    )

    # Calibrated metrics (calibrated_rank vs actual)
    precision_at_k_cal, recall_topdecile_at_k_cal = compute_precision_recall(
        df8.set_index("combo_id")["calibrated_rank"], df8.set_index("combo_id")["actual_rank"], ks
    )
    sp_cal, sp_cal_p = spearmanr(df8["calibrated_rank"], df8["actual_rank"])
    kt_cal, kt_cal_p = kendalltau(df8["calibrated_rank"], df8["actual_rank"])
    pair_acc_cal = compute_pairwise_accuracy(
        df8.set_index("combo_id")["calibrated_rank"], df8.set_index("combo_id")["actual_rank"], window_sizes=[1, 5, 10]
    )

    # Decile calibration: bin by WFO rank deciles, report realized sharpe mean/std/count
    df8["wfo_decile"] = pd.qcut(df8["wfo_rank"], 10, labels=False, duplicates="drop")
    calib = df8.groupby("wfo_decile")["sharpe"].agg(["count", "mean", "std"]).reset_index()
    calib.rename(columns={"mean": "sharpe_mean", "std": "sharpe_std"}, inplace=True)

    # Missed gems: actual TopDecile but not in WFO TopDecile / TopK variants
    top_decile_k = max(1, int(0.1 * n_pop))
    actual_top_decile = set(df8.nsmallest(top_decile_k, "actual_rank")["combo_id"])
    wfo_top50 = set(df8.nsmallest(50, "wfo_rank")["combo_id"]) if n_pop >= 50 else set(df8["combo_id"])
    missed_top50 = sorted(list(actual_top_decile - wfo_top50), key=lambda cid: float(df8.loc[df8["combo_id"] == cid, "sharpe"].iloc[0]), reverse=True)

    # Save outputs
    report = {
        "results_dir": results_dir,
        "wfo_ts": wfo_ts,
        "n": n_pop,
        "k_list": ks,
        "z": args.z,
        "near_freqs": near_freqs,
        "lambda_freq": args.lambda_freq,
        "wfo_score_file": args.wfo_score_file,
        "wfo_score_field": args.wfo_score_field,
        "spearman": sp,
        "spearman_p": sp_p,
        "kendall_tau": kt,
        "kendall_tau_p": kt_p,
        "precision_at_k": precision_at_k,
        "recall_topdecile_at_k": recall_topdecile_at_k,
        "pairwise_accuracy": pair_acc,
        "decile_calibration": calib.to_dict(orient="records"),
        "calibrated": {
            "spearman": sp_cal,
            "spearman_p": sp_cal_p,
            "kendall_tau": kt_cal,
            "kendall_tau_p": kt_cal_p,
            "precision_at_k": precision_at_k_cal,
            "recall_topdecile_at_k": recall_topdecile_at_k_cal,
            "pairwise_accuracy": pair_acc_cal,
        },
    }

    out_report = os.path.join(results_dir, "concordance_report.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    out_calibrated = os.path.join(results_dir, "calibrated_ranking_lcb90.csv")
    df8.sort_values(["calibrated_rank", "calibrated_score"], ascending=[True, False]).to_csv(out_calibrated, index=False)

    if missed_top50:
        out_missed = os.path.join(results_dir, "missed_gems.csv")
        cols = [
            "combo_id",
            "wfo_rank",
            "actual_rank",
            "sharpe",
            "lcb90_sharpe",
            "std_rank_near_freqs",
            "freq_penalty",
        ]
        pd.DataFrame([{c: df8.loc[df8["combo_id"] == cid, c].iloc[0] for c in cols} for cid in missed_top50]).to_csv(out_missed, index=False)

    print(f"Report saved: {out_report}")
    print(f"Calibrated ranking saved: {out_calibrated}")
    if missed_top50:
        print(f"Missed gems saved: {os.path.join(results_dir, 'missed_gems.csv')}")


if __name__ == "__main__":
    main()
