#!/usr/bin/env python3
"""Step A Deep Diagnostics for Drop ADX finding.

Three diagnostics:
  (A) Pre-hysteresis vs Post-hysteresis QDII selection rates
  (B) Trade quality: per-swap forward returns (baseline vs drop ADX)
  (C) Regime-sliced robustness (year + volatility regime)

Usage:
    uv run python scripts/ablation_qdii_deep_diag.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
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

QDII_CODES = {"513100", "513500", "159920", "513050", "513130"}

COMBOS = {
    "Baseline": "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "Drop ADX": "OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
}


def stable_topk(scores, k):
    """Replicate kernel's stable_topk_indices: highest scores first, ties by index."""
    valid = [(scores[i], i) for i in range(len(scores)) if scores[i] > -np.inf]
    valid.sort(key=lambda x: (-x[0], x[1]))
    return [idx for _, idx in valid[:k]]


def simulate_hysteresis(
    scores, holdings_mask, hold_days, top_indices, pos_size, delta_rank, min_hold_days
):
    """Pure-Python replica of apply_hysteresis for diagnostics."""
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

    # Fully invested: compute rank01
    safe_score = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(safe_score)
    rank01 = np.zeros(N, dtype=np.float64)
    denom = float(N - 1) if N > 1 else 1.0
    for j in range(N):
        rank01[order[j]] = float(j) / denom

    # Keep all current holdings
    target_mask[holdings_mask] = True

    # Find worst held
    worst_idx = -1
    worst_rank = 2.0
    for n in range(N):
        if holdings_mask[n] and rank01[n] < worst_rank:
            worst_rank = rank01[n]
            worst_idx = n

    # Find best new candidate
    best_new_idx = -1
    for idx in top_indices:
        if not holdings_mask[idx]:
            best_new_idx = idx
            break

    if worst_idx >= 0 and best_new_idx >= 0:
        rank_gap = rank01[best_new_idx] - rank01[worst_idx]
        rank_ok = rank_gap >= delta_rank
        days_ok = hold_days[worst_idx] >= min_hold_days
        if rank_ok and days_ok:
            target_mask[worst_idx] = False
            target_mask[best_new_idx] = True

    return target_mask


def run_selection_simulation(
    factors_3d, factor_indices, rebalance_schedule, etf_codes, qdii_mask,
    close_prices, timing_arr, gate_arr,
):
    """Simulate the full selection pipeline, tracking pre/post hysteresis and trade quality."""
    T, N, _ = factors_3d.shape
    n_rebal = len(rebalance_schedule)

    holdings_mask = np.zeros(N, dtype=bool)
    hold_days = np.zeros(N, dtype=np.int64)

    # Tracking
    pre_hyst_qdii_count = 0  # rebalances where QDII in pre-hyst top-2
    post_hyst_qdii_count = 0  # rebalances where QDII in post-hyst holdings
    pre_hyst_qdii_slots = 0   # total QDII slot appearances pre-hyst
    post_hyst_qdii_slots = 0  # total QDII slot appearances post-hyst
    total_rebal = 0

    swaps = []  # list of swap events with forward returns

    prev_rebal_t = -1

    for rb_idx, rb_t in enumerate(rebalance_schedule):
        if rb_t < 1 or rb_t >= T:
            continue

        # Update hold_days
        if prev_rebal_t >= 0:
            days_between = rb_t - prev_rebal_t
            for n in range(N):
                if holdings_mask[n]:
                    hold_days[n] += days_between
                else:
                    hold_days[n] = 0

        # Check if timing allows trading
        timing_val = timing_arr[rb_t] if rb_t < len(timing_arr) else 1.0
        if timing_val < 0.01:
            prev_rebal_t = rb_t
            continue

        # Compute combined scores (same as kernel: t-1 lagged)
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

        # Pre-hysteresis top-k
        pre_top = stable_topk(scores, POS_SIZE)

        total_rebal += 1

        # Track pre-hyst QDII
        pre_qdii = [idx for idx in pre_top if qdii_mask[idx]]
        if pre_qdii:
            pre_hyst_qdii_count += 1
        pre_hyst_qdii_slots += len(pre_qdii)

        # Apply hysteresis
        post_mask = simulate_hysteresis(
            scores, holdings_mask, hold_days, pre_top, POS_SIZE,
            DELTA_RANK, MIN_HOLD_DAYS,
        )

        # Track post-hyst QDII
        post_qdii = [n for n in range(N) if post_mask[n] and qdii_mask[n]]
        if post_qdii:
            post_hyst_qdii_count += 1
        post_hyst_qdii_slots += len(post_qdii)

        # Detect swaps (for trade quality analysis)
        for n in range(N):
            if holdings_mask[n] and not post_mask[n]:
                # Sold
                sell_price = close_prices[rb_t, n] if rb_t < T else np.nan
                # Forward returns from entry
                swaps.append({
                    "rebal_t": rb_t,
                    "date": str(dates[rb_t].date()) if rb_t < len(dates) else "?",
                    "action": "SELL",
                    "etf_idx": n,
                    "etf_code": etf_codes[n],
                    "is_qdii": bool(qdii_mask[n]),
                    "score": scores[n],
                })
            if not holdings_mask[n] and post_mask[n]:
                # Bought — compute forward 5D/10D/20D returns
                buy_price = close_prices[min(rb_t + 1, T - 1), n]  # T+1 open proxy
                fwd_5d = np.nan
                fwd_10d = np.nan
                fwd_20d = np.nan
                if rb_t + 6 < T:
                    fwd_5d = (close_prices[rb_t + 6, n] - buy_price) / buy_price
                if rb_t + 11 < T:
                    fwd_10d = (close_prices[rb_t + 11, n] - buy_price) / buy_price
                if rb_t + 21 < T:
                    fwd_20d = (close_prices[rb_t + 21, n] - buy_price) / buy_price
                swaps.append({
                    "rebal_t": rb_t,
                    "date": str(dates[rb_t].date()) if rb_t < len(dates) else "?",
                    "action": "BUY",
                    "etf_idx": n,
                    "etf_code": etf_codes[n],
                    "is_qdii": bool(qdii_mask[n]),
                    "score": scores[n],
                    "fwd_5d": fwd_5d,
                    "fwd_10d": fwd_10d,
                    "fwd_20d": fwd_20d,
                })

        # Update state
        for n in range(N):
            if post_mask[n] and not holdings_mask[n]:
                hold_days[n] = 0
            holdings_mask[n] = post_mask[n]
        prev_rebal_t = rb_t

    return {
        "total_rebal": total_rebal,
        "pre_hyst_qdii_count": pre_hyst_qdii_count,
        "post_hyst_qdii_count": post_hyst_qdii_count,
        "pre_hyst_qdii_slots": pre_hyst_qdii_slots,
        "post_hyst_qdii_slots": post_hyst_qdii_slots,
        "swaps": swaps,
    }


def main():
    print("=" * 80)
    print("Step A Deep Diagnostics: Drop ADX Finding")
    print("=" * 80)

    # ── Load data (same as ablation script) ───────────────────────────────
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
    global dates
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values

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

    # QDII mask
    qdii_mask = np.array([code in QDII_CODES for code in etf_codes], dtype=bool)

    # Train/holdout
    training_end_date = pd.Timestamp(config["data"].get("training_end_date", "2025-04-30"))
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end_date:
            train_end_idx = i

    # Factor indices
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    combo_indices = {}
    for name, combo_str in COMBOS.items():
        factors_list = [f.strip() for f in combo_str.split("+")]
        combo_indices[name] = [factor_index_map[f] for f in factors_list]

    # Rebalance schedule
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T, lookback_window=LOOKBACK, freq=FREQ,
    )

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")
    print(f"  Rebalances: {len(rebalance_schedule)}")
    print(f"  Train end: {dates[train_end_idx].date()}")

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (A): Pre-hysteresis vs Post-hysteresis QDII selection
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DIAGNOSTIC (A): Pre-Hysteresis vs Post-Hysteresis QDII Selection")
    print(f"{'='*80}")

    for combo_name, f_indices in combo_indices.items():
        sim = run_selection_simulation(
            factors_3d, f_indices, rebalance_schedule, etf_codes, qdii_mask,
            close_prices, timing_arr, gate_arr,
        )

        total = sim["total_rebal"]
        total_slots = total * POS_SIZE

        print(f"\n  {combo_name} ({COMBOS[combo_name]}):")
        print(f"    Total rebalances with trading: {total}")
        print(f"    Pre-hysteresis:  QDII in top-2 at {sim['pre_hyst_qdii_count']}/{total}"
              f" ({sim['pre_hyst_qdii_count']/total*100:.1f}%) rebalances,"
              f" {sim['pre_hyst_qdii_slots']}/{total_slots}"
              f" ({sim['pre_hyst_qdii_slots']/total_slots*100:.1f}%) slots")
        print(f"    Post-hysteresis: QDII in top-2 at {sim['post_hyst_qdii_count']}/{total}"
              f" ({sim['post_hyst_qdii_count']/total*100:.1f}%) rebalances,"
              f" {sim['post_hyst_qdii_slots']}/{total_slots}"
              f" ({sim['post_hyst_qdii_slots']/total_slots*100:.1f}%) slots")

        # Split by train/holdout
        train_rebals = [t for t in rebalance_schedule if t <= train_end_idx]
        ho_rebals = [t for t in rebalance_schedule if t > train_end_idx]

        # Re-run for train/holdout separately is complex; instead count from swaps
        buy_swaps = [s for s in sim["swaps"] if s["action"] == "BUY"]
        qdii_buys = [s for s in buy_swaps if s["is_qdii"]]
        print(f"    Total BUY events: {len(buy_swaps)}, QDII BUY events: {len(qdii_buys)}")
        if qdii_buys:
            for s in qdii_buys:
                fwd5 = s.get("fwd_5d", np.nan)
                fwd10 = s.get("fwd_10d", np.nan)
                print(f"      {s['date']} BUY {s['etf_code']}"
                      f" score={s['score']:.2f}"
                      f" fwd5d={fwd5*100:+.1f}%" if not np.isnan(fwd5) else f"      {s['date']} BUY {s['etf_code']} score={s['score']:.2f} fwd5d=N/A")

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (B): Trade Quality - Forward Returns of Swaps
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DIAGNOSTIC (B): Trade Quality - Forward Returns of BUY Events")
    print(f"{'='*80}")

    for combo_name, f_indices in combo_indices.items():
        sim = run_selection_simulation(
            factors_3d, f_indices, rebalance_schedule, etf_codes, qdii_mask,
            close_prices, timing_arr, gate_arr,
        )

        buy_swaps = [s for s in sim["swaps"] if s["action"] == "BUY"]
        if not buy_swaps:
            print(f"\n  {combo_name}: No BUY events")
            continue

        fwd5_vals = [s["fwd_5d"] for s in buy_swaps if not np.isnan(s.get("fwd_5d", np.nan))]
        fwd10_vals = [s["fwd_10d"] for s in buy_swaps if not np.isnan(s.get("fwd_10d", np.nan))]
        fwd20_vals = [s["fwd_20d"] for s in buy_swaps if not np.isnan(s.get("fwd_20d", np.nan))]

        print(f"\n  {combo_name}: {len(buy_swaps)} BUY events")
        print(f"    Forward returns (mean ± std, win rate):")
        for label, vals in [("5D", fwd5_vals), ("10D", fwd10_vals), ("20D", fwd20_vals)]:
            if vals:
                arr = np.array(vals)
                wr = np.mean(arr > 0) * 100
                print(f"      {label}: mean={np.mean(arr)*100:+.2f}%"
                      f"  std={np.std(arr)*100:.2f}%"
                      f"  median={np.median(arr)*100:+.2f}%"
                      f"  WR={wr:.0f}%"
                      f"  n={len(vals)}")

        # Split into train/holdout periods
        train_buys = [s for s in buy_swaps if s["rebal_t"] <= train_end_idx]
        ho_buys = [s for s in buy_swaps if s["rebal_t"] > train_end_idx]

        for period_name, period_buys in [("Train", train_buys), ("Holdout", ho_buys)]:
            fwd5 = [s["fwd_5d"] for s in period_buys if not np.isnan(s.get("fwd_5d", np.nan))]
            fwd10 = [s["fwd_10d"] for s in period_buys if not np.isnan(s.get("fwd_10d", np.nan))]
            if fwd5:
                arr5 = np.array(fwd5)
                arr10 = np.array(fwd10) if fwd10 else np.array([])
                print(f"    {period_name} ({len(period_buys)} buys):"
                      f"  5D mean={np.mean(arr5)*100:+.2f}% WR={np.mean(arr5>0)*100:.0f}%"
                      + (f"  10D mean={np.mean(arr10)*100:+.2f}% WR={np.mean(arr10>0)*100:.0f}%" if len(arr10) > 0 else ""))

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC (C): Robustness - Regime-Sliced Performance
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DIAGNOSTIC (C): Regime-Sliced Robustness")
    print(f"{'='*80}")

    # Compute volatility regime periods from gate_arr
    # gate_arr values: 1.0 (low vol), 0.7 (mid vol), 0.4 (high vol), 0.1 (extreme)
    regime_labels = []
    for g in gate_arr:
        if g >= 0.95:
            regime_labels.append("low_vol")
        elif g >= 0.65:
            regime_labels.append("mid_vol")
        elif g >= 0.35:
            regime_labels.append("high_vol")
        else:
            regime_labels.append("extreme_vol")
    regime_labels = np.array(regime_labels)

    # Count regime days
    print(f"\n  Regime distribution:")
    for regime in ["low_vol", "mid_vol", "high_vol", "extreme_vol"]:
        count = np.sum(regime_labels == regime)
        pct = count / len(regime_labels) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

    # For each combo, run VEC backtest and compute regime-sliced returns
    from batch_vec_backtest import run_vec_backtest
    from aligned_metrics import compute_aligned_metrics
    from etf_strategy.core.cost_model import load_cost_model, build_cost_array
    from etf_strategy.core.execution_model import load_execution_model

    exec_model = load_execution_model(config)
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))
    risk_config = backtest_config.get("risk_control", {})

    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    equity_curves = {}
    for combo_name, f_indices in combo_indices.items():
        eq, ret, _, _, trades, _, risk = run_vec_backtest(
            factors_3d, close_prices, open_prices, high_prices, low_prices,
            timing_arr, f_indices,
            freq=FREQ, pos_size=POS_SIZE,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE,
            lookback=LOOKBACK,
            cost_arr=COST_ARR,
            trailing_stop_pct=risk_config.get("trailing_stop_pct", 0.0),
            stop_on_rebalance_only=risk_config.get("stop_check_on_rebalance_only", True),
            leverage_cap=risk_config.get("leverage_cap", 1.0),
            profit_ladders=risk_config.get("profit_ladders", []),
            use_t1_open=exec_model.is_t1_open,
            delta_rank=DELTA_RANK,
            min_hold_days=MIN_HOLD_DAYS,
        )
        equity_curves[combo_name] = eq

    # Compute daily returns for regime analysis
    print(f"\n  Regime-sliced daily returns (annualized):")
    print(f"  {'Regime':<14} {'Baseline':>12} {'Drop ADX':>12} {'Delta':>10}")
    print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*10}")

    for regime in ["low_vol", "mid_vol", "high_vol", "extreme_vol"]:
        r_mask = regime_labels == regime
        # Only look at post-lookback period
        r_mask[:LOOKBACK] = False
        r_indices = np.where(r_mask)[0]
        if len(r_indices) < 5:
            continue

        results_regime = {}
        for combo_name, eq in equity_curves.items():
            # Compute daily returns only on regime days
            regime_rets = []
            for idx in r_indices:
                if idx > 0 and eq[idx - 1] > 0 and np.isfinite(eq[idx]) and np.isfinite(eq[idx - 1]):
                    regime_rets.append((eq[idx] - eq[idx - 1]) / eq[idx - 1])
            if regime_rets:
                arr = np.array(regime_rets)
                ann_ret = np.mean(arr) * 252
                results_regime[combo_name] = ann_ret
            else:
                results_regime[combo_name] = np.nan

        base_val = results_regime.get("Baseline", np.nan)
        drop_val = results_regime.get("Drop ADX", np.nan)
        delta = drop_val - base_val if np.isfinite(base_val) and np.isfinite(drop_val) else np.nan
        print(f"  {regime:<14}"
              f" {base_val*100:>+11.1f}%"
              f" {drop_val*100:>+11.1f}%"
              f" {delta*100:>+9.1f}pp" if np.isfinite(delta) else
              f"  {regime:<14} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    # Half-year rolling comparison
    print(f"\n  Half-year rolling returns:")
    print(f"  {'Period':<24} {'Baseline':>10} {'Drop ADX':>10} {'Delta':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*10}")

    half_year_periods = []
    for yr in range(2021, 2027):
        for half in ["H1", "H2"]:
            if half == "H1":
                start_m, end_m = 1, 6
            else:
                start_m, end_m = 7, 12
            mask = np.array([
                d.year == yr and start_m <= d.month <= end_m
                for d in dates
            ])
            indices = np.where(mask)[0]
            indices = indices[indices >= LOOKBACK]
            if len(indices) < 10:
                continue
            label = f"{yr} {half}"
            half_year_periods.append((label, indices))

    for label, indices in half_year_periods:
        results_hy = {}
        for combo_name, eq in equity_curves.items():
            seg = eq[indices]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 2 and seg[0] > 0:
                results_hy[combo_name] = (seg[-1] - seg[0]) / seg[0]
            else:
                results_hy[combo_name] = np.nan

        base_val = results_hy.get("Baseline", np.nan)
        drop_val = results_hy.get("Drop ADX", np.nan)
        delta = drop_val - base_val if np.isfinite(base_val) and np.isfinite(drop_val) else np.nan
        b_str = f"{base_val*100:>+9.1f}%" if np.isfinite(base_val) else f"{'N/A':>10}"
        d_str = f"{drop_val*100:>+9.1f}%" if np.isfinite(drop_val) else f"{'N/A':>10}"
        dt_str = f"{delta*100:>+9.1f}pp" if np.isfinite(delta) else f"{'N/A':>10}"
        print(f"  {label:<24} {b_str} {d_str} {dt_str}")

    # Score: count periods where Drop ADX wins
    wins = 0
    losses = 0
    for label, indices in half_year_periods:
        base_eq = equity_curves["Baseline"][indices]
        drop_eq = equity_curves["Drop ADX"][indices]
        base_eq = base_eq[np.isfinite(base_eq)]
        drop_eq = drop_eq[np.isfinite(drop_eq)]
        if len(base_eq) >= 2 and len(drop_eq) >= 2:
            base_r = (base_eq[-1] - base_eq[0]) / base_eq[0]
            drop_r = (drop_eq[-1] - drop_eq[0]) / drop_eq[0]
            if drop_r > base_r:
                wins += 1
            else:
                losses += 1

    total_hy = wins + losses
    print(f"\n  Drop ADX vs Baseline: wins {wins}/{total_hy} half-years ({wins/total_hy*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")

    print("""
  (A) Pre/Post Hysteresis:
      - Check above for QDII selection rates before vs after hysteresis
      - Key question: Does hysteresis block QDII entries?

  (B) Trade Quality:
      - Compare forward returns of BUY events between Baseline and Drop ADX
      - Key question: Is improvement from "fewer bad trades" or "better selection"?

  (C) Robustness:
      - Half-year win rate tells if improvement is concentrated or distributed
      - Regime analysis shows if improvement is regime-dependent
    """)


if __name__ == "__main__":
    main()
