#!/usr/bin/env python3
"""Step A: QDII Factor Ablation Study

Tests whether ADX/OBV structural bias against QDII is the root cause of
poor QDII selection in global rotation.

4 factor combinations (all with F5+Exp4 ON, GLOBAL mode, med cost):
  1. Baseline:  ADX + OBV + SHARPE + SLOPE  (current S1)
  2. Drop ADX:  OBV + SHARPE + SLOPE
  3. Drop OBV:  ADX + SHARPE + SLOPE
  4. Drop both: SHARPE + SLOPE

Outputs:
  - QDII selection rate & rank distribution
  - Year-by-year performance (train/holdout split)
  - Overall metrics (return, Sharpe, MDD, turnover)

Usage:
    uv run python scripts/ablation_qdii_factors.py
"""

from __future__ import annotations

import sys
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

from aligned_metrics import compute_aligned_metrics
from batch_vec_backtest import run_vec_backtest

# ── Experiment definitions ────────────────────────────────────────────────
ABLATION_COMBOS = {
    "Baseline (S1)":  "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "Drop ADX":       "OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D",
    "Drop OBV":       "ADX_14D + SHARPE_RATIO_20D + SLOPE_20D",
    "Drop both":      "SHARPE_RATIO_20D + SLOPE_20D",
}

# Production params (F5 + Exp4 ON)
FREQ = 5
POS_SIZE = 2
LOOKBACK = 252
DELTA_RANK = 0.10
MIN_HOLD_DAYS = 9

QDII_CODES = {"513100", "513500", "159920", "513050", "513130"}


def compute_mdd(equity: np.ndarray) -> float:
    """Max drawdown from equity array."""
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if not np.isfinite(v):
            continue
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def compute_qdii_rank_stats(
    factors_3d: np.ndarray,
    factor_indices: list[int],
    etf_codes: list[str],
    rebalance_schedule: np.ndarray,
    qdii_mask: np.ndarray,
) -> dict:
    """Compute QDII rank statistics from factor scores at each rebalance.

    Returns dict with:
      - qdii_selection_rate: fraction of rebalance slots filled by QDII
      - qdii_rank01_mean/median/p10: rank01 distribution of QDII ETFs
      - qdii_top2_appearances: count of rebalances where QDII appears in top-2
      - per_qdii_stats: per-ETF rank01 mean
    """
    T, N, _ = factors_3d.shape
    n_rebal = len(rebalance_schedule)

    qdii_indices = [i for i in range(N) if qdii_mask[i]]
    qdii_code_list = [etf_codes[i] for i in qdii_indices]

    all_qdii_rank01 = []
    qdii_in_top2_count = 0
    qdii_slot_count = 0  # total QDII slots in top-2 across all rebalances
    total_slots = 0

    # Per-QDII tracking
    per_qdii_ranks = {code: [] for code in qdii_code_list}

    for rb_t in rebalance_schedule:
        if rb_t < 1 or rb_t >= T:
            continue

        # Compute combined score (same as kernel: sum of factor z-scores at t-1)
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

        # Compute rank01 (ascending: worst=0, best=1)
        valid_mask = scores > -np.inf
        n_valid = np.sum(valid_mask)
        if n_valid < 2:
            continue

        order = np.argsort(scores)  # ascending
        rank01 = np.zeros(N, dtype=np.float64)
        denom = float(N - 1) if N > 1 else 1.0
        for j in range(N):
            rank01[order[j]] = float(j) / denom

        # Top-2 by score
        top2 = np.argsort(scores)[-POS_SIZE:][::-1]
        total_slots += POS_SIZE

        for qi in qdii_indices:
            r01 = rank01[qi]
            if scores[qi] > -np.inf:
                all_qdii_rank01.append(r01)
                code = etf_codes[qi]
                per_qdii_ranks[code].append(r01)

            if qi in top2:
                qdii_slot_count += 1

        if any(qi in top2 for qi in qdii_indices):
            qdii_in_top2_count += 1

    all_qdii_rank01 = np.array(all_qdii_rank01)

    result = {
        "qdii_selection_rate": qdii_slot_count / total_slots if total_slots > 0 else 0,
        "qdii_top2_rebalances": qdii_in_top2_count,
        "total_rebalances": n_rebal,
        "qdii_top2_pct": qdii_in_top2_count / n_rebal * 100 if n_rebal > 0 else 0,
    }

    if len(all_qdii_rank01) > 0:
        result["qdii_rank01_mean"] = float(np.mean(all_qdii_rank01))
        result["qdii_rank01_median"] = float(np.median(all_qdii_rank01))
        result["qdii_rank01_p10"] = float(np.percentile(all_qdii_rank01, 10))
        result["qdii_rank01_p90"] = float(np.percentile(all_qdii_rank01, 90))
    else:
        result["qdii_rank01_mean"] = np.nan
        result["qdii_rank01_median"] = np.nan
        result["qdii_rank01_p10"] = np.nan
        result["qdii_rank01_p90"] = np.nan

    # Per-QDII breakdown
    per_qdii = {}
    for code in qdii_code_list:
        ranks = per_qdii_ranks[code]
        if ranks:
            per_qdii[code] = {
                "mean": float(np.mean(ranks)),
                "median": float(np.median(ranks)),
                "count": len(ranks),
            }
        else:
            per_qdii[code] = {"mean": np.nan, "median": np.nan, "count": 0}
    result["per_qdii"] = per_qdii

    return result


def compute_yearly_metrics(
    equity: np.ndarray,
    dates: pd.DatetimeIndex,
    start_idx: int,
) -> list[dict]:
    """Compute year-by-year metrics from equity curve."""
    years = sorted(set(d.year for d in dates[start_idx:]))
    results = []
    for yr in years:
        yr_mask = np.array([d.year == yr for d in dates])
        yr_indices = np.where(yr_mask)[0]
        yr_indices = yr_indices[yr_indices >= start_idx]
        if len(yr_indices) < 10:
            continue

        yr_eq = equity[yr_indices]
        yr_eq = yr_eq[np.isfinite(yr_eq)]
        if len(yr_eq) < 2:
            continue

        yr_ret = (yr_eq[-1] - yr_eq[0]) / yr_eq[0] if yr_eq[0] != 0 else 0
        yr_mdd = compute_mdd(yr_eq)

        daily_rets = np.diff(yr_eq) / yr_eq[:-1]
        valid_rets = daily_rets[np.isfinite(daily_rets)]
        yr_sharpe = 0.0
        if len(valid_rets) > 1:
            m = float(np.mean(valid_rets))
            s = float(np.std(valid_rets, ddof=1))
            if s > 1e-12:
                yr_sharpe = m / s * np.sqrt(252.0)

        # Monthly win rate
        yr_dates = dates[yr_indices]
        monthly_rets = []
        for month in range(1, 13):
            m_mask = np.array([d.month == month for d in yr_dates])
            m_idx = np.where(m_mask)[0]
            if len(m_idx) < 2:
                continue
            m_eq = equity[yr_indices[m_idx]]
            m_eq = m_eq[np.isfinite(m_eq)]
            if len(m_eq) >= 2 and m_eq[0] > 0:
                monthly_rets.append((m_eq[-1] - m_eq[0]) / m_eq[0])
        win_rate = sum(1 for r in monthly_rets if r > 0) / len(monthly_rets) if monthly_rets else 0

        results.append({
            "year": yr,
            "return": yr_ret,
            "mdd": yr_mdd,
            "sharpe": yr_sharpe,
            "monthly_wr": win_rate,
            "n_months": len(monthly_rets),
        })
    return results


def compute_factor_zscore_by_domain(
    factors_3d: np.ndarray,
    factor_names: list[str],
    etf_codes: list[str],
    qdii_mask: np.ndarray,
    target_factors: list[str],
) -> dict:
    """Compute mean Z-score of target factors for QDII vs A-share domains."""
    T, N, F = factors_3d.shape
    factor_idx_map = {name: i for i, name in enumerate(factor_names)}
    result = {}

    for fname in target_factors:
        if fname not in factor_idx_map:
            continue
        fidx = factor_idx_map[fname]

        qdii_vals = []
        ashare_vals = []
        for t in range(LOOKBACK, T):
            for n in range(N):
                v = factors_3d[t, n, fidx]
                if np.isnan(v):
                    continue
                if qdii_mask[n]:
                    qdii_vals.append(v)
                else:
                    ashare_vals.append(v)

        result[fname] = {
            "qdii_mean_zscore": float(np.mean(qdii_vals)) if qdii_vals else np.nan,
            "qdii_std_zscore": float(np.std(qdii_vals)) if qdii_vals else np.nan,
            "ashare_mean_zscore": float(np.mean(ashare_vals)) if ashare_vals else np.nan,
            "ashare_std_zscore": float(np.std(ashare_vals)) if ashare_vals else np.nan,
            "qdii_n": len(qdii_vals),
            "ashare_n": len(ashare_vals),
        }
    return result


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Step A: QDII Factor Ablation Study")
    print(f"  Mode: GLOBAL (all 43 ETFs compete)")
    print(f"  Execution: F5 + Exp4 ON (dr={DELTA_RANK}, mh={MIN_HOLD_DAYS})")
    print(f"  Combos: {len(ABLATION_COMBOS)}")
    for name, combo in ABLATION_COMBOS.items():
        print(f"    {name}: {combo}")
    print("=" * 80)

    # ── 1. Load config ────────────────────────────────────────────────────
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get("backtest", {})
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    exec_model = load_execution_model(config)
    USE_T1_OPEN = exec_model.is_t1_open

    cost_model = load_cost_model(config)
    tier = cost_model.active_tier
    qdii_set = set(FrozenETFPool().qdii_codes)

    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])

    print(f"\n  Execution: {exec_model.mode}")
    print(f"  Cost: tier={cost_model.tier}, A={tier.a_share*10000:.0f}bp, QDII={tier.qdii*10000:.0f}bp")

    # ── 2. Load data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # ── 3. Compute factors (cached) ───────────────────────────────────────
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
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    COST_ARR = build_cost_array(cost_model, list(etf_codes), qdii_set)

    # ── 4. Timing + regime gate ───────────────────────────────────────────
    timing_cfg = backtest_config.get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=float(timing_cfg.get("extreme_threshold", -0.1)),
        extreme_position=float(timing_cfg.get("extreme_position", 0.1)),
    )
    timing_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates)
        .fillna(1.0)
        .values
    )
    timing_arr = shift_timing_signal(timing_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=backtest_config
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64))

    print(f"  Data: {T} days x {N} ETFs x {len(factor_names)} factors")

    # ── 5. Train/holdout split ────────────────────────────────────────────
    training_end_date = pd.Timestamp(
        config["data"].get("training_end_date", "2025-04-30")
    )
    train_end_idx = 0
    for i, d in enumerate(dates):
        if pd.Timestamp(d) <= training_end_date:
            train_end_idx = i
    print(f"  Train:   idx 0..{train_end_idx}  ({dates[0].date()} ~ {dates[train_end_idx].date()})")
    print(f"  Holdout: idx {train_end_idx+1}..{T-1}  ({dates[train_end_idx+1].date()} ~ {dates[T-1].date()})")

    # ── 6. QDII mask ──────────────────────────────────────────────────────
    qdii_mask = np.array([code in QDII_CODES for code in etf_codes], dtype=bool)
    n_qdii = int(qdii_mask.sum())
    print(f"  QDII ETFs: {n_qdii} / {N}")
    print(f"    Codes: {[etf_codes[i] for i in range(N) if qdii_mask[i]]}")

    # ── 7. Factor domain bias diagnostic ──────────────────────────────────
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Factor Z-score by Domain (QDII vs A-share)")
    print("=" * 80)
    domain_stats = compute_factor_zscore_by_domain(
        factors_3d, factor_names, etf_codes, qdii_mask,
        ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"],
    )
    print(f"  {'Factor':<22} {'QDII mean Z':>12} {'A-share mean Z':>15} {'Gap':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*15} {'-'*8}")
    for fname, stats in domain_stats.items():
        gap = stats["qdii_mean_zscore"] - stats["ashare_mean_zscore"]
        print(
            f"  {fname:<22} {stats['qdii_mean_zscore']:>12.3f}"
            f" {stats['ashare_mean_zscore']:>15.3f}"
            f" {gap:>+8.3f}"
        )

    # ── 8. Resolve factor indices ─────────────────────────────────────────
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    ablation_indices = {}
    for name, combo_str in ABLATION_COMBOS.items():
        factors_list = [f.strip() for f in combo_str.split("+")]
        try:
            indices = [factor_index_map[f] for f in factors_list]
            ablation_indices[name] = indices
            print(f"\n  {name}: {combo_str} -> indices {indices}")
        except KeyError as e:
            print(f"  WARNING: {name}: factor {e} not found, skipping")

    # ── 9. Rebalance schedule (for QDII rank tracking) ────────────────────
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T, lookback_window=LOOKBACK, freq=FREQ,
    )

    # ── 10. Run ablation ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RUNNING ABLATION")
    print("=" * 80)

    all_results = []
    for combo_name, f_indices in ablation_indices.items():
        print(f"\n{'─'*60}")
        print(f"  {combo_name}: {ABLATION_COMBOS[combo_name]}")
        print(f"{'─'*60}")

        # Run VEC backtest
        eq, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
            factors_3d,
            close_prices,
            open_prices,
            high_prices,
            low_prices,
            timing_arr,
            f_indices,
            freq=FREQ,
            pos_size=POS_SIZE,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE,
            lookback=LOOKBACK,
            cost_arr=COST_ARR,
            trailing_stop_pct=trailing_stop_pct,
            stop_on_rebalance_only=stop_on_rebalance_only,
            leverage_cap=leverage_cap,
            profit_ladders=profit_ladders,
            use_t1_open=USE_T1_OPEN,
            delta_rank=DELTA_RANK,
            min_hold_days=MIN_HOLD_DAYS,
        )

        # Train/holdout split
        start_idx = LOOKBACK
        train_eq = eq[start_idx:train_end_idx + 1]
        holdout_eq = eq[train_end_idx:]

        train_m = compute_aligned_metrics(train_eq, start_idx=0)
        holdout_m = compute_aligned_metrics(holdout_eq, start_idx=0)
        holdout_mdd = compute_mdd(holdout_eq)

        # Year-by-year
        yearly = compute_yearly_metrics(eq, dates, start_idx)

        # QDII rank stats
        qdii_stats = compute_qdii_rank_stats(
            factors_3d, f_indices, etf_codes, rebalance_schedule, qdii_mask,
        )

        result = {
            "combo": combo_name,
            "factors": ABLATION_COMBOS[combo_name],
            "n_factors": len(f_indices),
            # Overall
            "full_return": ret,
            "full_sharpe": risk["sharpe_ratio"],
            "full_mdd": risk["max_drawdown"],
            "trades": trades,
            "turnover_ann": risk["turnover_ann"],
            "cost_drag": risk["cost_drag"],
            # Train
            "train_return": train_m["aligned_return"],
            "train_sharpe": train_m["aligned_sharpe"],
            # Holdout
            "holdout_return": holdout_m["aligned_return"],
            "holdout_sharpe": holdout_m["aligned_sharpe"],
            "holdout_mdd": holdout_mdd,
            # QDII
            "qdii_selection_rate": qdii_stats["qdii_selection_rate"],
            "qdii_top2_pct": qdii_stats["qdii_top2_pct"],
            "qdii_rank01_mean": qdii_stats["qdii_rank01_mean"],
            "qdii_rank01_median": qdii_stats["qdii_rank01_median"],
            "qdii_rank01_p10": qdii_stats["qdii_rank01_p10"],
            "qdii_rank01_p90": qdii_stats["qdii_rank01_p90"],
            # Details
            "yearly": yearly,
            "qdii_per_etf": qdii_stats["per_qdii"],
            "equity_curve": eq,
        }
        all_results.append(result)

        # Print summary
        print(f"  Full:    ret={ret*100:+6.1f}%  Sharpe={risk['sharpe_ratio']:.2f}"
              f"  MDD={risk['max_drawdown']*100:.1f}%  trades={trades}")
        print(f"  Train:   ret={train_m['aligned_return']*100:+6.1f}%"
              f"  Sharpe={train_m['aligned_sharpe']:.2f}")
        print(f"  Holdout: ret={holdout_m['aligned_return']*100:+6.1f}%"
              f"  Sharpe={holdout_m['aligned_sharpe']:.2f}"
              f"  MDD={holdout_mdd*100:.1f}%")
        print(f"  Turnover: {risk['turnover_ann']:.1f}x  CostDrag={risk['cost_drag']*100:.2f}%")
        print(f"  QDII: sel_rate={qdii_stats['qdii_selection_rate']*100:.1f}%"
              f"  top2_rebal={qdii_stats['qdii_top2_pct']:.1f}%"
              f"  rank01_mean={qdii_stats['qdii_rank01_mean']:.3f}"
              f"  rank01_med={qdii_stats['qdii_rank01_median']:.3f}")

    # ── 11. Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Main metrics
    header = (
        f"  {'Combo':<18}"
        f" {'Full%':>7}"
        f" {'Train%':>8}"
        f" {'HO%':>7}"
        f" {'HO_Shrp':>8}"
        f" {'HO_MDD%':>8}"
        f" {'TO(x)':>6}"
        f" {'Trades':>7}"
        f" {'QDII_Sel%':>10}"
        f" {'QDII_Rk':>8}"
    )
    print(header)
    print(f"  {'-'*18} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*10} {'-'*8}")

    for r in all_results:
        print(
            f"  {r['combo']:<18}"
            f" {r['full_return']*100:>+6.1f}%"
            f" {r['train_return']*100:>+7.1f}%"
            f" {r['holdout_return']*100:>+6.1f}%"
            f" {r['holdout_sharpe']:>8.2f}"
            f" {r['holdout_mdd']*100:>7.1f}%"
            f" {r['turnover_ann']:>6.1f}"
            f" {r['trades']:>7d}"
            f" {r['qdii_selection_rate']*100:>9.1f}%"
            f" {r['qdii_rank01_mean']:>8.3f}"
        )

    # QDII rank distribution
    print(f"\n  {'Combo':<18} {'Rank01_Mean':>11} {'Rank01_Med':>11} {'Rank01_P10':>11} {'Rank01_P90':>11}")
    print(f"  {'-'*18} {'-'*11} {'-'*11} {'-'*11} {'-'*11}")
    for r in all_results:
        print(
            f"  {r['combo']:<18}"
            f" {r['qdii_rank01_mean']:>11.3f}"
            f" {r['qdii_rank01_median']:>11.3f}"
            f" {r['qdii_rank01_p10']:>11.3f}"
            f" {r['qdii_rank01_p90']:>11.3f}"
        )

    # Per-QDII ETF breakdown
    print(f"\n  Per-QDII ETF Mean Rank01:")
    qdii_codes_sorted = sorted(QDII_CODES)
    header_parts = [f"  {'Combo':<18}"]
    for code in qdii_codes_sorted:
        header_parts.append(f" {code:>8}")
    print("".join(header_parts))
    print(f"  {'-'*18}" + " ".join([f"{'-'*8}" for _ in qdii_codes_sorted]))
    for r in all_results:
        parts = [f"  {r['combo']:<18}"]
        for code in qdii_codes_sorted:
            if code in r["qdii_per_etf"]:
                mean_r = r["qdii_per_etf"][code]["mean"]
                parts.append(f" {mean_r:>8.3f}")
            else:
                parts.append(f" {'N/A':>8}")
        print("".join(parts))

    # Year-by-year comparison
    print(f"\n{'='*80}")
    print("YEAR-BY-YEAR BREAKDOWN")
    print(f"{'='*80}")

    for r in all_results:
        print(f"\n  {r['combo']}:")
        print(f"  {'Year':>6} {'Return%':>9} {'MDD%':>7} {'Sharpe':>8} {'MonthWR%':>9}")
        print(f"  {'-'*6} {'-'*9} {'-'*7} {'-'*8} {'-'*9}")
        for yr in r["yearly"]:
            print(
                f"  {yr['year']:>6d}"
                f" {yr['return']*100:>+8.1f}%"
                f" {yr['mdd']*100:>6.1f}%"
                f" {yr['sharpe']:>8.2f}"
                f" {yr['monthly_wr']*100:>8.0f}%"
            )

    # ── 12. Decision summary ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("DECISION SUMMARY")
    print(f"{'='*80}")

    baseline = all_results[0]
    for r in all_results[1:]:
        ho_delta = (r["holdout_return"] - baseline["holdout_return"]) * 100
        qdii_rank_delta = r["qdii_rank01_mean"] - baseline["qdii_rank01_mean"]
        qdii_sel_delta = (r["qdii_selection_rate"] - baseline["qdii_selection_rate"]) * 100
        mdd_delta = (r["holdout_mdd"] - baseline["holdout_mdd"]) * 100

        print(f"\n  {r['combo']} vs Baseline:")
        print(f"    HO return:    {ho_delta:>+6.1f}pp")
        print(f"    HO MDD:       {mdd_delta:>+6.1f}pp")
        print(f"    QDII rank01:  {qdii_rank_delta:>+6.3f} ({'improved' if qdii_rank_delta > 0 else 'degraded'})")
        print(f"    QDII sel rate: {qdii_sel_delta:>+6.1f}pp")

    # ── 13. Save results ──────────────────────────────────────────────────
    output_dir = ROOT / "results" / f"ablation_qdii_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary as CSV (without equity curves)
    summary_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("yearly", "qdii_per_etf", "equity_curve")}
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "ablation_summary.csv", index=False)

    # Save yearly details
    yearly_rows = []
    for r in all_results:
        for yr in r["yearly"]:
            yearly_rows.append({"combo": r["combo"], **yr})
    pd.DataFrame(yearly_rows).to_csv(output_dir / "ablation_yearly.csv", index=False)

    print(f"\n  Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
