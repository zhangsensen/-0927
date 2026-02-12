#!/usr/bin/env python3
"""Differential Trade Attribution: Baseline vs Drop ADX

For each rebalance day, compare the two strategies' holdings and attribute
the return difference to specific ETF swaps. Enriched with state variables
(vol, ER, dispersion) to find the true gating condition.

Usage:
    uv run python scripts/ablation_diff_trade_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr

FREQ = 5
POS_SIZE = 2
LOOKBACK = 252
DELTA_RANK = 0.10
MIN_HOLD_DAYS = 9

COMBOS = {
    "Baseline": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "Drop ADX": "OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
}

QDII_CODES = {"513100", "513500", "159920", "513050", "513130"}


def stable_topk(scores, k):
    """Highest scores first, ties by index."""
    valid = [(scores[i], i) for i in range(len(scores)) if scores[i] > -np.inf]
    valid.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in valid[:k]]


def simulate_hysteresis(scores, holdings_mask, hold_days, top_indices, pos_size,
                        delta_rank, min_hold_days):
    """Pure-Python replica of apply_hysteresis."""
    N = len(scores)
    target_mask = np.zeros(N, dtype=bool)

    if delta_rank <= 0 and min_hold_days <= 0:
        for idx in top_indices:
            target_mask[idx] = True
        return target_mask

    held_count = int(holdings_mask.sum())
    if held_count < pos_size:
        target_mask[holdings_mask] = True
        remaining = pos_size - held_count
        for idx in top_indices:
            if remaining <= 0:
                break
            if not target_mask[idx]:
                target_mask[idx] = True
                remaining -= 1
        return target_mask

    safe_score = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(safe_score)
    rank01 = np.zeros(N, dtype=np.float64)
    denom = float(N - 1) if N > 1 else 1.0
    for j in range(N):
        rank01[order[j]] = float(j) / denom

    target_mask[holdings_mask] = True

    worst_idx = -1
    worst_rank = 2.0
    for n in range(N):
        if holdings_mask[n] and rank01[n] < worst_rank:
            worst_rank = rank01[n]
            worst_idx = n

    best_new_idx = -1
    for idx in top_indices:
        if not holdings_mask[idx]:
            best_new_idx = idx
            break

    if worst_idx >= 0 and best_new_idx >= 0:
        rank_gap = rank01[best_new_idx] - rank01[worst_idx]
        if rank_gap >= delta_rank and hold_days[worst_idx] >= min_hold_days:
            target_mask[worst_idx] = False
            target_mask[best_new_idx] = True

    return target_mask


def simulate_selection(factors_3d, factor_indices, rebalance_schedule, N, T, timing_arr):
    """Simulate full selection pipeline, return per-rebalance holdings."""
    holdings_mask = np.zeros(N, dtype=bool)
    hold_days = np.zeros(N, dtype=np.int64)
    prev_t = -1
    history = []  # list of (rb_t, set of held indices)

    for rb_t in rebalance_schedule:
        if rb_t < 1 or rb_t >= T:
            continue

        if prev_t >= 0:
            gap = rb_t - prev_t
            for n in range(N):
                if holdings_mask[n]:
                    hold_days[n] += gap
                else:
                    hold_days[n] = 0

        if timing_arr[rb_t] < 0.01:
            history.append((rb_t, set(np.where(holdings_mask)[0])))
            prev_t = rb_t
            continue

        scores = np.full(N, -np.inf)
        for n in range(N):
            s = 0.0
            has_val = False
            for idx in factor_indices:
                val = factors_3d[rb_t - 1, n, idx]
                if not np.isnan(val):
                    s += val
                    has_val = True
            if has_val and s != 0.0:
                scores[n] = s

        pre_top = stable_topk(scores, POS_SIZE)
        post_mask = simulate_hysteresis(
            scores, holdings_mask, hold_days, pre_top, POS_SIZE,
            DELTA_RANK, MIN_HOLD_DAYS,
        )

        new_held = set(np.where(post_mask)[0])
        history.append((rb_t, new_held))

        for n in range(N):
            if post_mask[n] and not holdings_mask[n]:
                hold_days[n] = 0
            holdings_mask[n] = post_mask[n]
        prev_t = rb_t

    return history


def compute_state_variables(close_df, dates, lookback=20):
    """Compute state variables for each date: ER, dispersion, vol."""
    # Use all ETFs for cross-sectional metrics, 510300 for market-level
    rets = close_df.pct_change()

    # 1. Market Efficiency Ratio (510300): |net move| / sum(|daily moves|)
    proxy = close_df["510300"] if "510300" in close_df.columns else close_df.mean(axis=1)
    er_series = pd.Series(index=dates, dtype=float)
    for i in range(lookback, len(dates)):
        window = proxy.iloc[i - lookback:i + 1]
        net_move = abs(window.iloc[-1] - window.iloc[0])
        sum_moves = sum(abs(window.diff().dropna()))
        er_series.iloc[i] = net_move / sum_moves if sum_moves > 0 else 0.0

    # 2. Cross-sectional return dispersion (std of 5D returns across ETFs)
    fwd_5d_rets = close_df.pct_change(5)
    dispersion = fwd_5d_rets.std(axis=1)

    # 3. 510300 20D realized vol (annualized)
    proxy_rets = proxy.pct_change()
    vol_20d = proxy_rets.rolling(20, min_periods=20).std() * np.sqrt(252) * 100

    # 4. Cross-sectional rank autocorrelation (do ranks persist?)
    # High autocorr = trending market; Low = mean-reverting
    # Use 5D return ranks
    rank_autocorr = pd.Series(index=dates, dtype=float)
    for i in range(10, len(dates)):
        curr_ranks = rets.iloc[i].rank()
        prev_ranks = rets.iloc[i - 5].rank()
        valid = curr_ranks.notna() & prev_ranks.notna()
        if valid.sum() > 5:
            rank_autocorr.iloc[i] = curr_ranks[valid].corr(prev_ranks[valid])

    return {
        "ER_20D": er_series,
        "dispersion_5D": dispersion,
        "vol_20D": vol_20d,
        "rank_autocorr_5D": rank_autocorr,
    }


def main():
    print("=" * 80)
    print("Differential Trade Attribution: Baseline vs Drop ADX")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv, config=config, data_dir=loader.data_dir
    )
    factors_3d = cached["factors_3d"]
    factor_names = cached["factor_names"]
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    close_df = ohlcv["close"][etf_codes].ffill().fillna(1.0)

    # Timing + regime gate
    timing_cfg = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=float(timing_cfg.get("extreme_threshold", -0.1)),
        extreme_position=float(timing_cfg.get("extreme_position", 0.1)),
    )
    timing_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates).fillna(1.0).values
    )
    timing_arr = shift_timing_signal(timing_raw)
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    # Resolve factor indices
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    combo_indices = {}
    for name, combo_str in COMBOS.items():
        factors_list = [f.strip() for f in combo_str.split("+")]
        combo_indices[name] = [factor_index_map[f] for f in factors_list]

    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T, lookback_window=LOOKBACK, freq=FREQ,
    )

    # ── Compute state variables ───────────────────────────────────────────
    print("\nComputing state variables...")
    state_vars = compute_state_variables(close_df, dates)
    print(f"  ER_20D: {state_vars['ER_20D'].notna().sum()} valid days")
    print(f"  dispersion_5D: {state_vars['dispersion_5D'].notna().sum()} valid days")

    # ── Simulate both strategies ──────────────────────────────────────────
    print("\nSimulating selection paths...")
    histories = {}
    for combo_name, f_indices in combo_indices.items():
        histories[combo_name] = simulate_selection(
            factors_3d, f_indices, rebalance_schedule, N, T, timing_arr
        )

    base_hist = dict(histories["Baseline"])
    drop_hist = dict(histories["Drop ADX"])

    # ── Build differential trade ledger ───────────────────────────────────
    print("\nBuilding differential trade ledger...")

    records = []
    for rb_t in rebalance_schedule:
        if rb_t not in base_hist or rb_t not in drop_hist:
            continue

        base_held = base_hist[rb_t]
        drop_held = drop_hist[rb_t]

        # Same holdings?
        if base_held == drop_held:
            records.append({
                "rb_t": rb_t,
                "date": dates[rb_t],
                "same": True,
                "base_only": set(),
                "drop_only": set(),
                "shared": base_held,
            })
            continue

        base_only = base_held - drop_held
        drop_only = drop_held - base_held
        shared = base_held & drop_held

        records.append({
            "rb_t": rb_t,
            "date": dates[rb_t],
            "same": False,
            "base_only": base_only,
            "drop_only": drop_only,
            "shared": shared,
        })

    # ── Compute forward returns for differential positions ────────────────
    diff_records = []
    for rec in records:
        if rec["same"]:
            continue

        rb_t = rec["rb_t"]
        dt = rec["date"]
        year = dt.year
        half = "H1" if dt.month <= 6 else "H2"
        period = f"{year} {half}"

        # State variables at this rebalance
        er_val = state_vars["ER_20D"].iloc[rb_t] if rb_t < len(state_vars["ER_20D"]) else np.nan
        disp_val = state_vars["dispersion_5D"].iloc[rb_t] if rb_t < len(state_vars["dispersion_5D"]) else np.nan
        vol_val = state_vars["vol_20D"].iloc[rb_t] if rb_t < len(state_vars["vol_20D"]) else np.nan
        gate_val = gate_arr[rb_t] if rb_t < len(gate_arr) else np.nan

        # Forward returns for differential ETFs
        for base_idx in rec["base_only"]:
            for drop_idx in rec["drop_only"]:
                # Compute forward returns (using next open as proxy entry)
                entry_t = min(rb_t + 1, T - 1)

                fwd_rets = {}
                for horizon, label in [(5, "5D"), (10, "10D"), (25, "25D")]:
                    exit_t = min(entry_t + horizon, T - 1)
                    if exit_t > entry_t:
                        base_ret = (close_prices[exit_t, base_idx] - close_prices[entry_t, base_idx]) / close_prices[entry_t, base_idx]
                        drop_ret = (close_prices[exit_t, drop_idx] - close_prices[entry_t, drop_idx]) / close_prices[entry_t, drop_idx]
                        fwd_rets[f"base_{label}"] = base_ret
                        fwd_rets[f"drop_{label}"] = drop_ret
                        fwd_rets[f"diff_{label}"] = drop_ret - base_ret
                    else:
                        fwd_rets[f"base_{label}"] = np.nan
                        fwd_rets[f"drop_{label}"] = np.nan
                        fwd_rets[f"diff_{label}"] = np.nan

                diff_records.append({
                    "rb_t": rb_t,
                    "date": str(dt.date()),
                    "period": period,
                    "base_etf": etf_codes[base_idx],
                    "drop_etf": etf_codes[drop_idx],
                    "base_is_qdii": etf_codes[base_idx] in QDII_CODES,
                    "drop_is_qdii": etf_codes[drop_idx] in QDII_CODES,
                    "ER_20D": er_val,
                    "dispersion_5D": disp_val,
                    "vol_20D": vol_val,
                    "gate": gate_val,
                    **fwd_rets,
                })

    df_diff = pd.DataFrame(diff_records)

    # ═══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    total_rebal = len(records)
    same_count = sum(1 for r in records if r["same"])
    diff_count = total_rebal - same_count

    print(f"\n{'='*80}")
    print(f"HOLDINGS COMPARISON")
    print(f"{'='*80}")
    print(f"  Total rebalances: {total_rebal}")
    print(f"  Same holdings:    {same_count} ({same_count/total_rebal*100:.0f}%)")
    print(f"  Different:        {diff_count} ({diff_count/total_rebal*100:.0f}%)")
    print(f"  Differential swap records: {len(df_diff)}")

    # ── Period breakdown ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"DIFFERENTIAL TRADES BY PERIOD")
    print(f"{'='*80}")

    if len(df_diff) > 0:
        periods = sorted(df_diff["period"].unique())
        print(f"\n  {'Period':<12} {'N_swaps':>8} {'Drop-Base 5D':>14} {'Drop-Base 10D':>15} {'Drop-Base 25D':>15}")
        print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*15} {'-'*15}")

        for period in periods:
            pdf = df_diff[df_diff["period"] == period]
            n = len(pdf)
            d5 = pdf["diff_5D"].dropna()
            d10 = pdf["diff_10D"].dropna()
            d25 = pdf["diff_25D"].dropna()

            d5_str = f"{d5.mean()*100:+.2f}%" if len(d5) > 0 else "N/A"
            d10_str = f"{d10.mean()*100:+.2f}%" if len(d10) > 0 else "N/A"
            d25_str = f"{d25.mean()*100:+.2f}%" if len(d25) > 0 else "N/A"
            print(f"  {period:<12} {n:>8} {d5_str:>14} {d10_str:>15} {d25_str:>15}")

    # ── Detailed trade log for key periods ────────────────────────────────
    for focus_period in ["2022 H2", "2025 H2"]:
        print(f"\n{'─'*80}")
        print(f"  DETAILED: {focus_period}")
        print(f"{'─'*80}")

        pdf = df_diff[df_diff["period"] == focus_period]
        if len(pdf) == 0:
            print(f"  No differential trades in {focus_period}")
            continue

        print(f"  {'Date':<12} {'Base ETF':>10} {'Drop ETF':>10}"
              f" {'Base 5D':>9} {'Drop 5D':>9} {'Diff 5D':>9}"
              f" {'ER':>6} {'Vol':>6} {'Gate':>5}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*6} {'-'*5}")

        for _, row in pdf.iterrows():
            b5 = f"{row['base_5D']*100:+.1f}%" if not np.isnan(row.get("base_5D", np.nan)) else "N/A"
            d5 = f"{row['drop_5D']*100:+.1f}%" if not np.isnan(row.get("drop_5D", np.nan)) else "N/A"
            diff5 = f"{row['diff_5D']*100:+.1f}%" if not np.isnan(row.get("diff_5D", np.nan)) else "N/A"
            er = f"{row['ER_20D']:.3f}" if not np.isnan(row.get("ER_20D", np.nan)) else "N/A"
            vol = f"{row['vol_20D']:.0f}%" if not np.isnan(row.get("vol_20D", np.nan)) else "N/A"
            gate = f"{row['gate']:.1f}" if not np.isnan(row.get("gate", np.nan)) else "N/A"
            print(f"  {row['date']:<12} {row['base_etf']:>10} {row['drop_etf']:>10}"
                  f" {b5:>9} {d5:>9} {diff5:>9}"
                  f" {er:>6} {vol:>6} {gate:>5}")

        # Summary for this period
        d5 = pdf["diff_5D"].dropna()
        d10 = pdf["diff_10D"].dropna()
        if len(d5) > 0:
            print(f"\n  Summary: {len(pdf)} diff trades,"
                  f" Drop-Base avg 5D={d5.mean()*100:+.2f}%,"
                  f" 10D={d10.mean()*100:+.2f}%,"
                  f" Drop wins {(d5>0).sum()}/{len(d5)}")

    # ── Most frequent differential ETFs ───────────────────────────────────
    print(f"\n{'='*80}")
    print(f"MOST FREQUENT DIFFERENTIAL ETFs")
    print(f"{'='*80}")

    if len(df_diff) > 0:
        print(f"\n  Base-only (Baseline has, Drop ADX doesn't):")
        base_counts = df_diff["base_etf"].value_counts().head(10)
        for etf, count in base_counts.items():
            is_qdii = "QDII" if etf in QDII_CODES else ""
            sub = df_diff[df_diff["base_etf"] == etf]
            avg_ret = sub["base_5D"].dropna().mean() * 100
            print(f"    {etf} {is_qdii:>5}: {count:>3} times, avg 5D ret={avg_ret:+.2f}%")

        print(f"\n  Drop-only (Drop ADX has, Baseline doesn't):")
        drop_counts = df_diff["drop_etf"].value_counts().head(10)
        for etf, count in drop_counts.items():
            is_qdii = "QDII" if etf in QDII_CODES else ""
            sub = df_diff[df_diff["drop_etf"] == etf]
            avg_ret = sub["drop_5D"].dropna().mean() * 100
            print(f"    {etf} {is_qdii:>5}: {count:>3} times, avg 5D ret={avg_ret:+.2f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # STATE VARIABLE CORRELATION WITH DIFFERENTIAL RETURNS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"STATE VARIABLE CORRELATION WITH Drop-vs-Base ADVANTAGE")
    print(f"{'='*80}")

    if len(df_diff) > 0 and "diff_5D" in df_diff.columns:
        state_cols = ["ER_20D", "dispersion_5D", "vol_20D", "gate"]
        print(f"\n  Spearman correlation of state var with diff_5D (Drop-Base advantage):")
        print(f"  {'State Variable':<20} {'Corr':>8} {'p-value':>10} {'N':>6} {'Interpretation'}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*6} {'-'*30}")

        from scipy import stats as sp_stats

        for col in state_cols:
            valid = df_diff[[col, "diff_5D"]].dropna()
            if len(valid) < 5:
                print(f"  {col:<20} {'N/A':>8} {'N/A':>10} {len(valid):>6}")
                continue
            corr, pval = sp_stats.spearmanr(valid[col], valid["diff_5D"])
            interp = ""
            if abs(corr) > 0.15 and pval < 0.1:
                if col == "ER_20D":
                    interp = "Low ER → Drop ADX better" if corr < 0 else "High ER → Drop ADX better"
                elif col == "vol_20D":
                    interp = "High vol → Drop ADX better" if corr > 0 else "Low vol → Drop ADX better"
                elif col == "dispersion_5D":
                    interp = "High disp → Drop ADX better" if corr > 0 else "Low disp → Drop ADX better"
                elif col == "gate":
                    interp = "Low gate → Drop ADX better" if corr < 0 else "High gate → Drop ADX better"
            print(f"  {col:<20} {corr:>+8.3f} {pval:>10.4f} {len(valid):>6} {interp}")

        # Also try 10D
        print(f"\n  Same for diff_10D:")
        for col in state_cols:
            valid = df_diff[[col, "diff_10D"]].dropna()
            if len(valid) < 5:
                continue
            corr, pval = sp_stats.spearmanr(valid[col], valid["diff_10D"])
            print(f"  {col:<20} {corr:>+8.3f} {pval:>10.4f} {len(valid):>6}")

    # ── Conditional advantage: split by ER tercile ────────────────────────
    print(f"\n{'='*80}")
    print(f"DROP ADX ADVANTAGE BY STATE VARIABLE TERCILE")
    print(f"{'='*80}")

    if len(df_diff) > 0:
        for state_var in ["ER_20D", "vol_20D", "dispersion_5D"]:
            valid = df_diff[[state_var, "diff_5D", "diff_10D"]].dropna()
            if len(valid) < 9:
                continue

            terciles = pd.qcut(valid[state_var], q=3, labels=["Low", "Mid", "High"], duplicates="drop")
            print(f"\n  {state_var} terciles (n={len(valid)}):")
            print(f"    {'Tercile':<8} {'N':>5} {'Diff_5D':>10} {'Diff_10D':>11} {'WinRate_5D':>11}")
            print(f"    {'-'*8} {'-'*5} {'-'*10} {'-'*11} {'-'*11}")

            for t_label in ["Low", "Mid", "High"]:
                mask = terciles == t_label
                sub = valid[mask]
                if len(sub) == 0:
                    continue
                d5 = sub["diff_5D"]
                d10 = sub["diff_10D"]
                wr = (d5 > 0).mean() * 100
                print(f"    {t_label:<8} {len(sub):>5}"
                      f" {d5.mean()*100:>+9.2f}%"
                      f" {d10.mean()*100:>+10.2f}%"
                      f" {wr:>10.0f}%")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
