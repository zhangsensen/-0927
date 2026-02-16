#!/usr/bin/env python3
"""
S1 Degradation Isolation Test

S1 (ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D) dropped from
+42.7% to +3.8% HO return. Three confounded causes:
  (1) bounded_factors list expanded: 4 → 7 (ADX_14D now correctly NOT Winsorized)
  (2) bounded factor treatment changed: raw pass-through → rank [-0.5, 0.5]
  (3) ETF pool expansion: 43 → 49 (added 159985, 512200, 515220, 513180, 513400, 513520)

This script runs a 2x2 matrix (plus extra configs) to isolate each effect:
  A: S1 @ 49-ETF + new bounded (7) + rank    → current state, should reproduce ~3.8%
  B: S1 @ 43-ETF + new bounded (7) + rank    → isolates pool effect
  C: S1 @ 43-ETF + old bounded (4) + raw     → should approximate original ~42.7%
  D: C2 @ 43-ETF + new bounded (7) + rank    → C2 sensitivity check
  E: S1 @ 49-ETF + old bounded (4) + raw     → isolates bounded+rank effect with new pool
  F: S1 @ 43-ETF + new bounded (7) + raw     → isolates rank change effect (bounded=7, no rank)
  G: S1 @ 43-ETF + old bounded (4) + rank    → isolates bounded list change (bounded=4, with rank)

Notes on v5.0 sealed vs current cross_section_processor.py:
  - Old: BOUNDED_FACTORS = {RSI_14, PP_20D, PP_120D, PV_CORR_20D} — raw pass-through
  - New: BOUNDED_FACTORS = {+ADX_14D, +CMF_20D, +CORR_MKT} — rank to [-0.5, 0.5]
  - Key: ADX_14D was z-scored+Winsorized in old code; now rank-standardized
"""
import os
import sys
from pathlib import Path

# Force warn mode for frozen params (A/B testing)
os.environ["FROZEN_PARAMS_MODE"] = "warn"

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import copy
from datetime import datetime
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool, load_frozen_config
from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from etf_strategy.regime_gate import compute_regime_gate_arr

# Import run_vec_backtest from batch_vec_backtest via importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "batch_vec", str(ROOT / "scripts/batch_vec_backtest.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_vec_backtest = _mod.run_vec_backtest

# ── Constants ──
S1_FACTORS = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]
C2_FACTORS = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]

OLD_BOUNDED = {"RSI_14", "PRICE_POSITION_20D", "PRICE_POSITION_120D", "PV_CORR_20D"}
NEW_BOUNDED = {
    "ADX_14D", "CMF_20D", "CORRELATION_TO_MARKET_20D",
    "PRICE_POSITION_20D", "PRICE_POSITION_120D", "PV_CORR_20D", "RSI_14",
}

# 6 ETFs added between v5.0-sealed (43) and current config (49)
# Verified against sealed_strategies/v5.0_20260211/locked/configs/combo_wfo_config.yaml
ADDED_ETFS = {"159985", "512200", "515220", "513180", "513400", "513520"}

TRAINING_END = "2025-04-30"


def load_config():
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config, config_path


def get_43_etf_symbols(config):
    """Return the original 43-ETF symbol list (exclude 6 added ETFs)."""
    all_symbols = config["data"]["symbols"]
    result = [s for s in all_symbols if s not in ADDED_ETFS]
    assert len(result) == 43, f"Expected 43, got {len(result)}"
    return result


def compute_factors_manual(ohlcv, config, bounded_set, use_rank_for_bounded):
    """Compute factors_3d with explicit control over bounded treatment.

    Args:
        ohlcv: OHLCV data dict
        config: full config dict
        bounded_set: set of factor names considered bounded
        use_rank_for_bounded: if True, rank-standardize bounded factors (current behavior);
                              if False, raw pass-through (v5.0 sealed behavior)

    Returns:
        dict with std_factors, factor_names, factors_3d, dates, etf_codes
    """
    cross_section_cfg = config.get("cross_section", {})
    lower_pct = cross_section_cfg.get("winsorize_lower", 0.025) * 100
    upper_pct = cross_section_cfg.get("winsorize_upper", 0.975) * 100

    # Step 1: Compute raw factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(prices=ohlcv)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    # Step 2: Manual cross-section standardization with explicit bounded control
    processed_factors = {}
    for factor_name, factor_data in raw_factors.items():
        if factor_name in bounded_set:
            if use_rank_for_bounded:
                # Current behavior: rank standardize to [-0.5, 0.5]
                ranked = factor_data.rank(axis=1, pct=True) - 0.5
                ranked = ranked.where(factor_data.notna())
                processed_factors[factor_name] = ranked
            else:
                # v5.0 sealed behavior: raw pass-through
                processed_factors[factor_name] = factor_data
        else:
            # Unbounded: Z-score + Winsorize (same in both versions)
            mean_cs = factor_data.mean(axis=1, skipna=True)
            std_cs = factor_data.std(axis=1, skipna=True)
            standardized = factor_data.sub(mean_cs, axis=0).div(std_cs, axis=0)
            lower_bound = standardized.quantile(lower_pct / 100, axis=1)
            upper_bound = standardized.quantile(upper_pct / 100, axis=1)
            winsorized = standardized.clip(lower=lower_bound, upper=upper_bound, axis=0)
            processed_factors[factor_name] = winsorized

    # Step 3: Build factors_3d
    factor_names = sorted(processed_factors.keys())
    first_factor = processed_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    factors_3d = np.stack([processed_factors[f].values for f in factor_names], axis=-1)

    return {
        "std_factors": processed_factors,
        "factor_names": factor_names,
        "factors_3d": factors_3d,
        "dates": dates,
        "etf_codes": etf_codes,
    }


def prepare_timing_and_gate(ohlcv, dates, config):
    """Compute timing_arr with light_timing + regime gate."""
    T = len(dates)
    timing_config = config.get("backtest", {}).get("timing", {})
    extreme_threshold = timing_config.get("extreme_threshold", -0.4)
    extreme_position = timing_config.get("extreme_position", 0.3)

    timing_module = LightTimingModule(
        extreme_threshold=extreme_threshold,
        extreme_position=extreme_position,
    )
    timing_arr_raw = (
        timing_module.compute_position_ratios(ohlcv["close"])
        .reindex(dates)
        .fillna(1.0)
        .values
    )
    timing_arr = shift_timing_signal(timing_arr_raw)

    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=config.get("backtest", {})
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )
    return timing_arr


def run_single_test(label, combo_factors, config, etf_symbols, bounded_set, use_rank):
    """Run a single VEC backtest configuration and return metrics."""
    print(f"\n{'='*70}")
    print(f"  TEST: {label}")
    print(f"  Factors: {combo_factors}")
    print(f"  ETFs: {len(etf_symbols)}")
    print(f"  Bounded ({len(bounded_set)}): {sorted(bounded_set)}")
    print(f"  Rank for bounded: {use_rank}")
    print(f"{'='*70}")

    # Load OHLCV for this ETF subset
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=etf_symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Compute factors with specified bounded set and treatment
    cached = compute_factors_manual(ohlcv, config, bounded_set, use_rank)

    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    factors_3d = cached["factors_3d"]
    T = len(dates)
    N = len(etf_codes)

    # Price arrays
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing + gate
    timing_arr = prepare_timing_and_gate(ohlcv, dates, config)

    # Cost array
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, list(etf_codes), qdii_set)

    # Execution model
    exec_model = load_execution_model(config)
    use_t1_open = exec_model.is_t1_open

    # Config params
    backtest_config = config.get("backtest", {})
    freq = backtest_config.get("freq", 5)
    pos_size = backtest_config.get("pos_size", 2)
    lookback = backtest_config.get("lookback") or backtest_config.get("lookback_window", 252)
    initial_capital = float(backtest_config.get("initial_capital", 1_000_000))
    commission_rate = float(backtest_config.get("commission_rate", 0.0002))

    # Hysteresis (production: delta_rank=0.10, min_hold_days=9)
    hyst_config = backtest_config.get("hysteresis", {})
    delta_rank = float(hyst_config.get("delta_rank", 0.0))
    min_hold_days = int(hyst_config.get("min_hold_days", 0))

    # Risk params
    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.0)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", True)
    leverage_cap = risk_config.get("leverage_cap", 1.0)
    profit_ladders = risk_config.get("profit_ladders", [])
    cb_config = risk_config.get("circuit_breaker", {})
    circuit_breaker_day = cb_config.get("max_drawdown_day", 0.0)
    circuit_breaker_total = cb_config.get("max_drawdown_total", 0.0)
    circuit_recovery_days = cb_config.get("recovery_days", 5)
    cooldown_days = risk_config.get("cooldown_days", 0)

    # Factor indices
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    missing = [f for f in combo_factors if f not in factor_index_map]
    if missing:
        print(f"  WARNING: Missing factors: {missing}")
        return None
    factor_indices = [factor_index_map[f] for f in combo_factors]

    print(f"  Running VEC: T={T}, N={N}, F={len(factor_names)}")
    print(f"  freq={freq}, pos_size={pos_size}, delta_rank={delta_rank}, min_hold_days={min_hold_days}")
    print(f"  use_t1_open={use_t1_open}")

    eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        factors_3d,
        close_prices,
        open_prices,
        high_prices,
        low_prices,
        timing_arr,
        factor_indices,
        freq=freq,
        pos_size=pos_size,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        lookback=lookback,
        cost_arr=cost_arr,
        use_t1_open=use_t1_open,
        delta_rank=delta_rank,
        min_hold_days=min_hold_days,
        trailing_stop_pct=trailing_stop_pct,
        stop_on_rebalance_only=stop_on_rebalance_only,
        leverage_cap=leverage_cap,
        profit_ladders=profit_ladders,
        circuit_breaker_day=circuit_breaker_day,
        circuit_breaker_total=circuit_breaker_total,
        circuit_recovery_days=circuit_recovery_days,
        cooldown_days=cooldown_days,
    )

    # Compute holdout metrics
    dates_list = list(dates)
    training_end_dt = pd.Timestamp(TRAINING_END)
    ho_start_idx = None
    for i, d in enumerate(dates_list):
        if d > training_end_dt:
            ho_start_idx = i
            break

    ho_return = None
    ho_mdd = None
    ho_sharpe = None
    ho_worst_month = None
    if ho_start_idx is not None and ho_start_idx < len(eq_curve):
        ho_eq = eq_curve[ho_start_idx:]
        ho_eq_norm = ho_eq / ho_eq[0] if ho_eq[0] > 0 else ho_eq
        ho_return = float(ho_eq_norm[-1] / ho_eq_norm[0] - 1.0)

        # HO MDD
        running_max = np.maximum.accumulate(ho_eq_norm)
        drawdowns = (ho_eq_norm - running_max) / running_max
        ho_mdd = abs(float(drawdowns.min()))

        # HO Sharpe (annualized)
        ho_daily_ret = np.diff(ho_eq) / ho_eq[:-1]
        if len(ho_daily_ret) > 1 and np.std(ho_daily_ret) > 0:
            ho_sharpe = float(np.mean(ho_daily_ret) / np.std(ho_daily_ret) * np.sqrt(252))

        # Worst month
        ho_dates = dates[ho_start_idx:]
        ho_months = pd.Series(ho_eq, index=ho_dates).resample("ME").last()
        if len(ho_months) > 1:
            monthly_ret = ho_months.pct_change().dropna()
            ho_worst_month = float(monthly_ret.min())

    result = {
        "label": label,
        "factors": " + ".join(combo_factors),
        "n_etfs": N,
        "n_bounded": len(bounded_set),
        "rank_bounded": use_rank,
        "full_return": ret,
        "ho_return": ho_return,
        "ho_mdd": ho_mdd,
        "ho_sharpe": ho_sharpe,
        "ho_worst_month": ho_worst_month,
        "trades": trades,
        "sharpe": risk["sharpe_ratio"],
        "max_dd": risk["max_drawdown"],
    }

    print(f"\n  RESULTS for {label}:")
    print(f"    Full Return: {ret*100:+.1f}%")
    if ho_return is not None:
        print(f"    HO Return:   {ho_return*100:+.1f}%")
        print(f"    HO MDD:      {ho_mdd*100:.1f}%")
        if ho_sharpe is not None:
            print(f"    HO Sharpe:   {ho_sharpe:.2f}")
        if ho_worst_month is not None:
            print(f"    HO Worst Mo: {ho_worst_month*100:.1f}%")
    print(f"    Trades:      {trades}")
    print(f"    Full Sharpe: {risk['sharpe_ratio']:.2f}")
    print(f"    Full MDD:    {risk['max_drawdown']*100:.1f}%")

    return result


def main():
    print("=" * 80)
    print("S1 DEGRADATION ISOLATION TEST (3-way: pool x bounded_list x rank_treatment)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    config, config_path = load_config()

    all_symbols_49 = config["data"]["symbols"]
    all_symbols_43 = get_43_etf_symbols(config)

    print(f"\n49-ETF pool: {len(all_symbols_49)} ETFs")
    print(f"43-ETF pool: {len(all_symbols_43)} ETFs")
    print(f"Added ETFs: {sorted(ADDED_ETFS)}")
    print(f"Old bounded (4): {sorted(OLD_BOUNDED)}")
    print(f"New bounded (7): {sorted(NEW_BOUNDED)}")
    print(f"\nKey difference in bounded treatment:")
    print(f"  Old (v5.0 sealed): raw pass-through for bounded factors")
    print(f"  New (current):     rank [-0.5, 0.5] for bounded factors")

    results = []

    # ── Core 2x2 matrix: pool x (bounded_list + rank_treatment) ──

    # A: S1 @ 49-ETF + new bounded (7) + rank → current state
    r = run_single_test(
        "A: S1@49 new(7)+rank", S1_FACTORS, config,
        all_symbols_49, NEW_BOUNDED, use_rank=True,
    )
    if r:
        results.append(r)

    # B: S1 @ 43-ETF + new bounded (7) + rank → isolates pool effect
    r = run_single_test(
        "B: S1@43 new(7)+rank", S1_FACTORS, config,
        all_symbols_43, NEW_BOUNDED, use_rank=True,
    )
    if r:
        results.append(r)

    # C: S1 @ 43-ETF + old bounded (4) + raw → original v5.0 behavior
    r = run_single_test(
        "C: S1@43 old(4)+raw", S1_FACTORS, config,
        all_symbols_43, OLD_BOUNDED, use_rank=False,
    )
    if r:
        results.append(r)

    # D: C2 @ 43-ETF + new bounded (7) + rank → C2 sensitivity
    r = run_single_test(
        "D: C2@43 new(7)+rank", C2_FACTORS, config,
        all_symbols_43, NEW_BOUNDED, use_rank=True,
    )
    if r:
        results.append(r)

    # E: S1 @ 49-ETF + old bounded (4) + raw → bounded+rank effect at 49-pool
    r = run_single_test(
        "E: S1@49 old(4)+raw", S1_FACTORS, config,
        all_symbols_49, OLD_BOUNDED, use_rank=False,
    )
    if r:
        results.append(r)

    # ── Decomposition intermediaries ──

    # F: S1 @ 43-ETF + new bounded (7) + raw → isolates rank change
    r = run_single_test(
        "F: S1@43 new(7)+raw", S1_FACTORS, config,
        all_symbols_43, NEW_BOUNDED, use_rank=False,
    )
    if r:
        results.append(r)

    # G: S1 @ 43-ETF + old bounded (4) + rank → isolates bounded list change
    r = run_single_test(
        "G: S1@43 old(4)+rank", S1_FACTORS, config,
        all_symbols_43, OLD_BOUNDED, use_rank=True,
    )
    if r:
        results.append(r)

    # ── Summary Table ──
    print("\n\n" + "=" * 90)
    print("ISOLATION MATRIX RESULTS")
    print("=" * 90)

    header = (
        f"{'Label':<28} {'Pool':>4} {'Bnd':>3} {'Rank':>4} "
        f"{'HO Ret':>8} {'HO MDD':>8} {'HO Shrp':>8} {'HO WM':>8} {'Trades':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ho_ret_str = f"{r['ho_return']*100:+.1f}%" if r['ho_return'] is not None else "N/A"
        ho_mdd_str = f"{r['ho_mdd']*100:.1f}%" if r['ho_mdd'] is not None else "N/A"
        ho_shp_str = f"{r['ho_sharpe']:.2f}" if r['ho_sharpe'] is not None else "N/A"
        ho_wm_str = f"{r['ho_worst_month']*100:.1f}%" if r['ho_worst_month'] is not None else "N/A"
        print(
            f"{r['label']:<28} {r['n_etfs']:>4} {r['n_bounded']:>3} "
            f"{'Y' if r['rank_bounded'] else 'N':>4} "
            f"{ho_ret_str:>8} {ho_mdd_str:>8} {ho_shp_str:>8} {ho_wm_str:>8} {r['trades']:>7}"
        )

    # ── Attribution ──
    print("\n\nATTRIBUTION ANALYSIS (all pp values = HO return difference):")
    print("-" * 70)

    # Build lookup by label prefix
    by_label = {r["label"].split(":")[0].strip(): r for r in results}

    def ho(key):
        r = by_label.get(key)
        return r["ho_return"] if r and r["ho_return"] is not None else None

    a, b, c, e, f, g = ho("A"), ho("B"), ho("C"), ho("E"), ho("F"), ho("G")

    if a is not None and c is not None:
        print(f"\n  Total degradation (A-C):  {(a-c)*100:+.1f}pp")
        print(f"    (from +42.7% baseline to current)")

    if a is not None and b is not None:
        print(f"\n  1. Pool effect (A-B):     {(a-b)*100:+.1f}pp  [49-ETF vs 43-ETF, same new bounded+rank]")

    if b is not None and c is not None:
        print(f"  2. Bounded+rank effect (B-C): {(b-c)*100:+.1f}pp  [new(7)+rank vs old(4)+raw, same 43-ETF]")

    # Decompose bounded+rank into components using F and G
    if f is not None and c is not None:
        print(f"\n  Decomposition of bounded+rank effect:")
        print(f"    2a. Rank treatment (C-G): {(c-g)*100:+.1f}pp  [raw vs rank, same old bounded(4), 43-ETF]")
    if g is not None and b is not None:
        print(f"    2b. Bounded list (G-B):   {(g-b)*100:+.1f}pp  [old(4) vs new(7), both rank, 43-ETF]")
    if f is not None and c is not None:
        print(f"    2c. List only (C-F):      {(c-f)*100:+.1f}pp  [old(4) vs new(7), both raw, 43-ETF]")
    if f is not None and b is not None:
        print(f"    2d. Rank only (F-B):      {(f-b)*100:+.1f}pp  [raw vs rank, same new bounded(7), 43-ETF]")

    # Cross-check with 49-pool
    if a is not None and e is not None:
        print(f"\n  Cross-check @ 49-ETF pool:")
        print(f"    Bounded+rank effect (A-E): {(a-e)*100:+.1f}pp  [new(7)+rank vs old(4)+raw, 49-ETF]")
    if e is not None and c is not None:
        print(f"    Pool effect (E-C):         {(e-c)*100:+.1f}pp  [49-ETF vs 43-ETF, same old bounded+raw]")

    # C2 comparison
    d = ho("D")
    if d is not None and b is not None:
        print(f"\n  C2 vs S1 @ 43-ETF new bounded:")
        print(f"    C2 - S1 (D-B):            {(d-b)*100:+.1f}pp")

    # Save results
    out_dir = ROOT / "results" / "s1_isolation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"isolation_results_{ts}.csv"

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
