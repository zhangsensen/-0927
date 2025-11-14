#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio-level backtest for selected strategies.
- Reads selected strategies from {run_dir}/selection/live/live_strategies.csv
- Locates full backtest summary CSV via {run_dir}/selection/selection_summary.json
  (or accepts --backtest-csv override)
- Merges per-strategy summary (annual_ret, vol, max_dd, n_rebalance, test_freq)
- If daily series are NOT available, runs Monte Carlo approximation to estimate
  portfolio Sharpe and max drawdown under assumptions about correlation.

Outputs:
  {run_dir}/selection/live/portfolio_report.json
  {run_dir}/selection/live/portfolio_report.md

Usage:
  python scripts/portfolio_backtest_selected.py --run-dir results/run_YYYYMMDD_HHMMSS \
    [--backtest-csv /path/to/top12597_backtest_by_ic_..._full.csv] \
    [--rho 0.0] [--mc 2000]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _load_selection(run_dir: Path) -> pd.DataFrame:
    lf = run_dir / "selection" / "live" / "live_strategies.csv"
    if not lf.exists():
        raise FileNotFoundError(f"Missing live_strategies.csv at {lf}")
    return pd.read_csv(lf, low_memory=False)


essential_cols = {
    "combo": ["combo", "combo_key", "id", "key"],
    "annual_ret": ["annual_ret", "ann_ret", "ann_return"],
    "vol": ["vol", "annual_vol", "ann_vol"],
    "sharpe": ["sharpe", "sharpe_ratio"],
    "max_dd": ["max_dd", "max_drawdown"],
    "n_rebalance": ["n_rebalance", "n_trades"],
    "test_freq": ["test_freq", "freq"],
}


def _detect_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_full_backtest(run_dir: Path, override: Optional[str]) -> pd.DataFrame:
    if override:
        return pd.read_csv(override, low_memory=False)
    sel_summary = run_dir / "selection" / "selection_summary.json"
    if not sel_summary.exists():
        raise FileNotFoundError("selection_summary.json not found; please pass --backtest-csv")
    js = json.loads(sel_summary.read_text())
    p = js.get("backtest_csv")
    if not p:
        raise FileNotFoundError("backtest_csv not recorded in selection_summary.json; please pass --backtest-csv")
    return pd.read_csv(p, low_memory=False)


def _merge(df_sel: pd.DataFrame, df_bt: pd.DataFrame) -> pd.DataFrame:
    combo_sel = _detect_col(df_sel.columns, ["combo_key", "combo", "id", "key"]) or "combo_key"
    combo_bt = _detect_col(df_bt.columns, essential_cols["combo"]) or "combo"
    df = df_sel.copy()
    df = df.merge(df_bt, left_on=combo_sel, right_on=combo_bt, how="left", suffixes=("", "_bt"))
    # rename essentials
    out = df.copy()
    for k, cands in essential_cols.items():
        c = _detect_col(out.columns, cands)
        if c and c != k:
            out = out.rename(columns={c: k})
    # keep required columns
    needed = ["combo_key", "capital_alloc", "capital_alloc_ratio", "rebalance_freq_days",
              "annual_ret", "vol", "sharpe", "max_dd", "n_rebalance", "test_freq"]
    # Resolve combo_key
    if "combo_key" not in out.columns:
        if "combo" in out.columns:
            out = out.rename(columns={"combo": "combo_key"})
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns after merge: {missing}")
    return out[needed]


def simulate_portfolio(df: pd.DataFrame, rho: float = 0.0, mc: int = 2000, seed: int = 42) -> Dict[str, Any]:
    # Convert annualized ret/vol to daily
    daily_mu = df["annual_ret"].to_numpy(dtype=float) / 252.0
    daily_sigma = df["vol"].to_numpy(dtype=float) / np.sqrt(252.0)
    w = (df["capital_alloc"].to_numpy(dtype=float))
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        # fall back to equal weights
        w = np.ones(len(df), dtype=float)
    w = w / w.sum()

    # Determine horizon from n_rebalance * test_freq (min across strategies)
    T = int(np.nanmin(df["n_rebalance"].to_numpy(dtype=float) * df["test_freq"].to_numpy(dtype=float)))
    T = max(T, 252)  # at least 1 year

    rng = np.random.default_rng(seed)
    N = len(df)
    # Build covariance matrix with constant correlation rho
    R = np.full((N, N), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    Sigma = np.outer(daily_sigma, daily_sigma) * R
    # Cholesky may fail if not PD for extreme rho; adjust slightly
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        eps = 1e-6
        L = np.linalg.cholesky(Sigma + eps * np.eye(N))

    def run_once() -> Dict[str, float]:
        z = rng.standard_normal(size=(T, N))  # iid standard normal
        eps_t = z @ L.T  # correlated shocks
        ret_t = daily_mu + eps_t  # vector per strategy, broadcast mu
        # portfolio daily return
        rp = ret_t @ w
        # equity and drawdown
        eq = np.cumprod(1.0 + rp)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1.0
        mdd = float(np.min(dd))
        ann_ret = float((eq[-1] ** (252.0 / T)) - 1.0)
        ann_vol = float(np.std(rp) * np.sqrt(252.0))
        sharpe = float((np.mean(rp) * 252.0) / (np.std(rp) * np.sqrt(252.0) + 1e-12))
        return {"mdd": mdd, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}

    sims = [run_once() for _ in range(mc)]
    mdd_arr = np.array([s["mdd"] for s in sims])
    ret_arr = np.array([s["ann_ret"] for s in sims])
    vol_arr = np.array([s["ann_vol"] for s in sims])
    sr_arr = np.array([s["sharpe"] for s in sims])

    def q(x, p):
        return float(np.quantile(x, p))

    return {
        "T_days": T,
        "weights": w.tolist(),
        "rho": rho,
        "mc": mc,
        "ann_ret_mean": float(ret_arr.mean()),
        "ann_ret_p50": q(ret_arr, 0.5),
        "ann_vol_mean": float(vol_arr.mean()),
        "sharpe_mean": float(sr_arr.mean()),
        "sharpe_p50": q(sr_arr, 0.5),
        "max_dd_p50": q(mdd_arr, 0.5),
        "max_dd_p05": q(mdd_arr, 0.05),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--backtest-csv", default=None)
    ap.add_argument("--rho", type=float, default=0.0, help="Assumed constant pairwise correlation for MC simulation (0-0.5 typical)")
    ap.add_argument("--mc", type=int, default=2000)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df_sel = _load_selection(run_dir)
    df_bt = _load_full_backtest(run_dir, args.backtest_csv)
    merged = _merge(df_sel, df_bt)

    # MC simulation (rho as provided; also compute a conservative rho=0.3 scenario)
    res_base = simulate_portfolio(merged, rho=args.rho, mc=args.mc)
    res_cons = simulate_portfolio(merged, rho=max(args.rho, 0.3), mc=args.mc)

    report = {
        "n_strategies": int(len(merged)),
        "horizon_days": res_base["T_days"],
        "inputs": {
            "weights_sum": float(np.sum(merged["capital_alloc"].to_numpy())),
            "rho_base": args.rho,
            "rho_conservative": max(args.rho, 0.3),
            "mc": args.mc,
        },
        "portfolio_estimates": {
            "base": res_base,
            "conservative": res_cons,
        },
        "note": "Estimates via Monte Carlo using per-strategy annual_ret/vol; actual daily series not available in CSV."
    }

    out_dir = run_dir / "selection" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "portfolio_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    md = []
    md.append("# Portfolio Backtest (Estimated)\n\n")
    md.append(f"Strategies: {report['n_strategies']} | Horizon(days): {report['horizon_days']} | MC: {args.mc}\n\n")
    md.append("## Base scenario\n")
    md.append(f"- rho={report['inputs']['rho_base']} | Sharpe(p50)={res_base['sharpe_p50']:.3f} | AnnRet(p50)={res_base['ann_ret_p50']:.3%} | AnnVol(mean)={res_base['ann_vol_mean']:.3%}\n")
    md.append(f"- MaxDD p50={res_base['max_dd_p50']:.2%} | p05(worse)={res_base['max_dd_p05']:.2%}\n\n")
    md.append("## Conservative scenario\n")
    md.append(f"- rho={report['inputs']['rho_conservative']} | Sharpe(p50)={res_cons['sharpe_p50']:.3f} | AnnRet(p50)={res_cons['ann_ret_p50']:.3%} | AnnVol(mean)={res_cons['ann_vol_mean']:.3%}\n")
    md.append(f"- MaxDD p50={res_cons['max_dd_p50']:.2%} | p05(worse)={res_cons['max_dd_p05']:.2%}\n\n")
    md.append("> Note: This is a Monte Carlo estimate from summary stats (annual_ret/vol) because the full backtest CSV does not contain daily series.\n")

    (out_dir / "portfolio_report.md").write_text("".join(md))
    print(f"✅ Wrote {out_dir/'portfolio_report.{json,md}'}")


if __name__ == "__main__":
    main()
