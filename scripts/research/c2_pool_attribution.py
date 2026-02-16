#!/usr/bin/env python3
"""
C2 Pool Expansion Attribution: 43-ETF vs 46-ETF vs 49-ETF

Investigates why C2 (AMIHUD+CALMAR+CORR_MKT) gained ~25pp from pool expansion.

Three scenarios:
  A) 43-ETF pool (sealed v5.0)
  B) 49-ETF pool (current config)
  C) 46-ETF pool (49 minus 3 new A-share: 159985, 512200, 515220)

All run with: F5, T1_OPEN, Exp4 hysteresis (dr=0.10, mh=9), regime gate ON, med cost.

Outputs:
  - Per-scenario HO return, MDD, Sharpe, trades
  - Holdings timeline comparison (which ETFs selected at each rebalance)
  - Return attribution for divergent trades
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

import numpy as np
import pandas as pd
import yaml

from etf_strategy.core.cost_model import build_cost_array, load_cost_model
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)
from etf_strategy.regime_gate import compute_regime_gate_arr, gate_stats

# Import VEC backtest functions from batch script
sys.path.insert(0, str(ROOT / "scripts"))
from batch_vec_backtest import run_vec_backtest, stable_topk_indices
from aligned_metrics import compute_aligned_metrics
from etf_strategy.core.hysteresis import apply_hysteresis

# ─────────────────────────────────────────────────────────────────────────────
# C2 factor combination
# ─────────────────────────────────────────────────────────────────────────────
C2_FACTORS = ["AMIHUD_ILLIQUIDITY", "CALMAR_RATIO_60D", "CORRELATION_TO_MARKET_20D"]

# New A-share ETFs added in 49-ETF pool (tradeable under A_SHARE_ONLY)
NEW_A_SHARE = {"159985", "512200", "515220"}

# New QDII ETFs added in 49-ETF pool (monitored, not traded under A_SHARE_ONLY)
NEW_QDII = {"513180", "513400", "513520"}


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_data(config, symbols_override=None):
    """Load OHLCV, compute factors, build timing/regime arrays for a given symbol list."""
    symbols = symbols_override if symbols_override else config["data"]["symbols"]

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # Compute factors using FactorCache
    factor_cache = FactorCache(
        cache_dir=Path(config["data"].get("cache_dir") or ".cache")
    )
    # Override symbols in config for cache computation
    config_copy = {**config, "data": {**config["data"], "symbols": symbols}}
    cached = factor_cache.get_or_compute(
        ohlcv=ohlcv,
        config=config_copy,
        data_dir=loader.data_dir,
    )

    factors_3d = cached["factors_3d"]
    factor_names = list(cached["factor_names"])
    dates = cached["dates"]
    etf_codes = cached["etf_codes"]
    T = len(dates)
    N = len(etf_codes)

    # Price arrays
    close_prices = ohlcv["close"][etf_codes].ffill().fillna(1.0).values
    open_prices = ohlcv["open"][etf_codes].ffill().fillna(1.0).values
    high_prices = ohlcv["high"][etf_codes].ffill().fillna(1.0).values
    low_prices = ohlcv["low"][etf_codes].ffill().fillna(1.0).values

    # Timing
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

    # Regime gate
    gate_arr = compute_regime_gate_arr(
        ohlcv["close"], dates, backtest_config=config.get("backtest", {})
    )
    timing_arr = (timing_arr.astype(np.float64) * gate_arr.astype(np.float64)).astype(
        np.float64
    )

    # Cost model
    cost_model = load_cost_model(config)
    qdii_set = set(FrozenETFPool().qdii_codes)
    cost_arr = build_cost_array(cost_model, list(etf_codes), qdii_set)

    return {
        "factors_3d": factors_3d,
        "factor_names": factor_names,
        "dates": dates,
        "etf_codes": etf_codes,
        "close_prices": close_prices,
        "open_prices": open_prices,
        "high_prices": high_prices,
        "low_prices": low_prices,
        "timing_arr": timing_arr,
        "cost_arr": cost_arr,
        "ohlcv": ohlcv,
        "T": T,
        "N": N,
    }


def run_c2_backtest(data, config):
    """Run VEC backtest for C2 with full production params."""
    factor_names = data["factor_names"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}

    # Validate C2 factors exist
    missing = [f for f in C2_FACTORS if f not in factor_index_map]
    if missing:
        raise ValueError(f"Missing C2 factors: {missing}")

    factor_indices = [factor_index_map[f] for f in C2_FACTORS]

    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq", 5)
    POS_SIZE = backtest_config.get("pos_size", 2)
    LOOKBACK = backtest_config.get("lookback") or backtest_config.get("lookback_window", 252)
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital", 1_000_000))
    COMMISSION_RATE = float(backtest_config.get("commission_rate", 0.0002))

    # Risk control
    risk_config = backtest_config.get("risk_control", {})
    trailing_stop_pct = risk_config.get("trailing_stop_pct", 0.08)
    stop_on_rebalance_only = risk_config.get("stop_check_on_rebalance_only", False)
    profit_ladders = risk_config.get("profit_ladders", [])
    circuit_breaker_config = risk_config.get("circuit_breaker", {})
    circuit_breaker_day = circuit_breaker_config.get("max_drawdown_day", 0.0)
    circuit_breaker_total = circuit_breaker_config.get("max_drawdown_total", 0.0)
    circuit_recovery_days = circuit_breaker_config.get("recovery_days", 5)
    cooldown_days = risk_config.get("cooldown_days", 0)
    leverage_cap = risk_config.get("leverage_cap", 1.0)

    eq_curve, ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        data["factors_3d"],
        data["close_prices"],
        data["open_prices"],
        data["high_prices"],
        data["low_prices"],
        data["timing_arr"],
        factor_indices,
        freq=FREQ,
        pos_size=POS_SIZE,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        lookback=LOOKBACK,
        cost_arr=data["cost_arr"],
        # T1_OPEN
        use_t1_open=True,
        # Hysteresis (critical)
        delta_rank=0.10,
        min_hold_days=9,
        # Stop loss
        trailing_stop_pct=trailing_stop_pct,
        stop_on_rebalance_only=stop_on_rebalance_only,
        profit_ladders=profit_ladders,
        # Circuit breaker
        circuit_breaker_day=circuit_breaker_day,
        circuit_breaker_total=circuit_breaker_total,
        circuit_recovery_days=circuit_recovery_days,
        cooldown_days=cooldown_days,
        leverage_cap=leverage_cap,
    )

    return {
        "equity_curve": eq_curve,
        "total_return": ret,
        "win_rate": wr,
        "profit_factor": pf,
        "num_trades": trades,
        "risk_metrics": risk,
    }


def trace_holdings(data, config):
    """Trace holdings at each rebalance day using Python (not Numba) to capture ETF selections.

    Replicates the VEC kernel's scoring + hysteresis logic to determine which ETFs
    are selected at each rebalance point.
    """
    factor_names = data["factor_names"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    factor_indices = [factor_index_map[f] for f in C2_FACTORS]

    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq", 5)
    POS_SIZE = backtest_config.get("pos_size", 2)
    LOOKBACK = backtest_config.get("lookback") or backtest_config.get("lookback_window", 252)

    factors_3d = data["factors_3d"]
    close_prices = data["close_prices"]
    open_prices = data["open_prices"]
    etf_codes = list(data["etf_codes"])
    dates = data["dates"]
    timing_arr = data["timing_arr"]
    cost_arr = data["cost_arr"]
    T, N = data["T"], data["N"]

    rebalance_schedule = generate_rebalance_schedule(T, LOOKBACK, FREQ)

    # State tracking
    holdings = np.full(N, -1.0)  # shares held, -1 = not held
    hold_days_arr = np.zeros(N, dtype=np.int64)
    cash = 1_000_000.0
    entry_prices = np.zeros(N)

    # Pending (T1_OPEN)
    pend_active = False
    pend_target = np.zeros(N, dtype=np.bool_)
    pend_buy_list = []
    pend_timing = 1.0

    # Holdings log: (date_idx, [etf_code1, etf_code2])
    holdings_log = []
    rebal_ptr = 0
    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else LOOKBACK

    for t in range(start_day, T):
        # Increment hold days
        for n in range(N):
            if holdings[n] > 0.0:
                hold_days_arr[n] += 1

        # Execute pending T1_OPEN orders
        if pend_active:
            # Sell
            for n in range(N):
                if holdings[n] > 0.0 and not pend_target[n]:
                    price = open_prices[t, n] if not np.isnan(open_prices[t, n]) else close_prices[t - 1, n]
                    sell_cost = holdings[n] * price * cost_arr[n]
                    cash += holdings[n] * price - sell_cost
                    holdings[n] = -1.0
                    entry_prices[n] = 0.0
                    hold_days_arr[n] = 0

            # Buy
            pend_val = cash
            for n in range(N):
                if holdings[n] > 0.0:
                    p = open_prices[t, n] if not np.isnan(open_prices[t, n]) else close_prices[t - 1, n]
                    pend_val += holdings[n] * p

            new_buys = [idx for idx in pend_buy_list if holdings[idx] < 0.0]
            if new_buys:
                kept_val = sum(
                    holdings[n] * (open_prices[t, n] if not np.isnan(open_prices[t, n]) else close_prices[t - 1, n])
                    for n in range(N) if holdings[n] > 0.0
                )
                available = max(pend_val * pend_timing - kept_val, 0.0)
                tpv = available / len(new_buys)
                for idx in new_buys:
                    price = open_prices[t, idx] if not np.isnan(open_prices[t, idx]) else 0.0
                    if price <= 0:
                        continue
                    shares = (tpv / (1.0 + cost_arr[idx])) / price
                    cost = shares * price * (1.0 + cost_arr[idx])
                    if cash >= cost - 1e-5 and cost > 0:
                        cash -= cost
                        holdings[idx] = shares
                        entry_prices[idx] = price
                        hold_days_arr[idx] = 1
            pend_active = False

        # Check rebalance
        is_rebalance_day = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance_day = True
            rebal_ptr += 1

        if is_rebalance_day:
            # Compute combined scores
            combined_score = np.full(N, -np.inf)
            for n in range(N):
                score = 0.0
                has_value = False
                for idx in factor_indices:
                    val = factors_3d[t - 1, n, idx]
                    if not np.isnan(val):
                        score += val
                        has_value = True
                if has_value and score != 0.0:
                    combined_score[n] = score

            valid = np.sum(combined_score > -np.inf)

            target_set = np.zeros(N, dtype=np.bool_)
            buy_order = []

            if valid >= POS_SIZE:
                # Top-K selection
                top_indices = stable_topk_indices(combined_score, POS_SIZE)
                for k in range(len(top_indices)):
                    idx = top_indices[k]
                    if combined_score[idx] == -np.inf:
                        break
                    target_set[idx] = True
                    buy_order.append(idx)

                # Apply hysteresis
                h_mask = np.array([holdings[n] > 0.0 for n in range(N)], dtype=np.bool_)
                if len(buy_order) > 0:
                    target_mask = apply_hysteresis(
                        combined_score,
                        h_mask,
                        hold_days_arr,
                        top_indices,
                        POS_SIZE,
                        0.10,  # delta_rank
                        9,     # min_hold_days
                    )
                    target_set = target_mask
                    buy_order = [n for n in range(N) if target_set[n]]

            # Store pending for T1_OPEN
            pend_target[:] = target_set
            pend_buy_list = buy_order
            pend_timing = timing_arr[t]
            pend_active = True

            # Log holdings (what we're targeting)
            held_etfs = sorted([etf_codes[n] for n in range(N) if target_set[n]])
            holdings_log.append({
                "date_idx": t,
                "date": str(dates[t].date()) if hasattr(dates[t], 'date') else str(dates[t])[:10],
                "holdings": held_etfs,
                "scores": {etf_codes[n]: float(combined_score[n]) for n in range(N) if target_set[n]},
            })

    return holdings_log


def compute_holdout_metrics(equity_curve, dates, training_end="2025-04-30"):
    """Compute holdout-period metrics from equity curve."""
    training_end_dt = pd.Timestamp(training_end)
    ho_start = None
    for i, d in enumerate(dates):
        if d > training_end_dt:
            ho_start = i
            break

    if ho_start is None:
        return {"ho_return": 0.0, "ho_mdd": 0.0, "ho_sharpe": 0.0}

    ho_curve = equity_curve[ho_start:]
    # Skip initial flat portion
    first_nonzero = 0
    init_val = ho_curve[0]
    for i in range(len(ho_curve)):
        if ho_curve[i] != init_val:
            first_nonzero = max(0, i - 1)
            break

    ho_curve = ho_curve[first_nonzero:]
    if len(ho_curve) < 2:
        return {"ho_return": 0.0, "ho_mdd": 0.0, "ho_sharpe": 0.0}

    ho_return = (ho_curve[-1] / ho_curve[0]) - 1.0

    # Max drawdown
    peak = ho_curve[0]
    max_dd = 0.0
    for v in ho_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    daily_rets = np.diff(ho_curve) / ho_curve[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    if len(daily_rets) > 1 and np.std(daily_rets) > 1e-8:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Worst month
    ho_dates = dates[ho_start:]
    ho_dates = ho_dates[first_nonzero:]
    monthly_rets = []
    if len(ho_dates) >= 20:
        month_start_val = ho_curve[0]
        current_month = ho_dates[0].month if hasattr(ho_dates[0], 'month') else pd.Timestamp(ho_dates[0]).month
        for i in range(1, len(ho_curve)):
            d = ho_dates[i] if i < len(ho_dates) else ho_dates[-1]
            m = d.month if hasattr(d, 'month') else pd.Timestamp(d).month
            if m != current_month:
                mret = (ho_curve[i - 1] / month_start_val) - 1.0 if month_start_val > 0 else 0
                monthly_rets.append(mret)
                month_start_val = ho_curve[i - 1]
                current_month = m
        # Last partial month
        if month_start_val > 0:
            mret = (ho_curve[-1] / month_start_val) - 1.0
            monthly_rets.append(mret)

    worst_month = min(monthly_rets) if monthly_rets else 0.0

    return {
        "ho_return": ho_return,
        "ho_mdd": max_dd,
        "ho_sharpe": sharpe,
        "ho_worst_month": worst_month,
    }


def main():
    print("=" * 80)
    print("C2 Pool Expansion Attribution: 43-ETF vs 46-ETF vs 49-ETF")
    print("=" * 80)

    # Load configs
    config_49 = load_config(ROOT / "configs/combo_wfo_config.yaml")
    config_43 = load_config(ROOT / "sealed_strategies/v5.0_20260211/locked/configs/combo_wfo_config.yaml")

    symbols_43 = config_43["data"]["symbols"]
    symbols_49 = config_49["data"]["symbols"]

    # Scenario C: 46-ETF (49 minus new A-share)
    symbols_46 = [s for s in symbols_49 if s not in NEW_A_SHARE]

    print(f"\nPool sizes:")
    print(f"  A) 43-ETF (sealed v5.0): {len(symbols_43)} symbols")
    print(f"  B) 49-ETF (current):     {len(symbols_49)} symbols")
    print(f"  C) 46-ETF (49 - A-share): {len(symbols_46)} symbols")
    print(f"\nNew A-share: {sorted(NEW_A_SHARE)}")
    print(f"New QDII:    {sorted(NEW_QDII)}")

    # Delta between 43 and 49
    set_43 = set(symbols_43)
    set_49 = set(symbols_49)
    added = sorted(set_49 - set_43)
    removed = sorted(set_43 - set_49)
    print(f"\nAdded in 49-ETF:   {added}")
    print(f"Removed from 43-ETF: {removed}")

    # ─────────────────────────────────────────────────────────────────────────
    # Prepare data for each scenario
    # Use 49-ETF config as base (has all params), override symbols only
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Preparing data for Scenario A (43-ETF) ---")
    data_43 = prepare_data(config_49, symbols_override=symbols_43)
    print(f"  T={data_43['T']}, N={data_43['N']}, factors={len(data_43['factor_names'])}")

    print("\n--- Preparing data for Scenario B (49-ETF) ---")
    data_49 = prepare_data(config_49, symbols_override=symbols_49)
    print(f"  T={data_49['T']}, N={data_49['N']}, factors={len(data_49['factor_names'])}")

    print("\n--- Preparing data for Scenario C (46-ETF) ---")
    data_46 = prepare_data(config_49, symbols_override=symbols_46)
    print(f"  T={data_46['T']}, N={data_46['N']}, factors={len(data_46['factor_names'])}")

    # ─────────────────────────────────────────────────────────────────────────
    # Run VEC backtests
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Running VEC Backtest: Scenario A (43-ETF) ---")
    result_43 = run_c2_backtest(data_43, config_49)

    print("\n--- Running VEC Backtest: Scenario B (49-ETF) ---")
    result_49 = run_c2_backtest(data_49, config_49)

    print("\n--- Running VEC Backtest: Scenario C (46-ETF) ---")
    result_46 = run_c2_backtest(data_46, config_49)

    # ─────────────────────────────────────────────────────────────────────────
    # Compute holdout metrics
    # ─────────────────────────────────────────────────────────────────────────
    training_end = config_49["data"].get("training_end_date", "2025-04-30")

    ho_43 = compute_holdout_metrics(result_43["equity_curve"], data_43["dates"], training_end)
    ho_49 = compute_holdout_metrics(result_49["equity_curve"], data_49["dates"], training_end)
    ho_46 = compute_holdout_metrics(result_46["equity_curve"], data_46["dates"], training_end)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VEC BACKTEST RESULTS")
    print("=" * 80)
    print(f"{'Metric':<25} {'A) 43-ETF':>12} {'B) 49-ETF':>12} {'C) 46-ETF':>12} {'B-A':>8} {'C-A':>8} {'B-C':>8}")
    print("-" * 85)

    metrics = [
        ("Full Return", "total_return"),
        ("Trades", "num_trades"),
        ("Win Rate", "win_rate"),
    ]
    for label, key in metrics:
        v43 = result_43[key]
        v49 = result_49[key]
        v46 = result_46[key]
        if key == "num_trades":
            print(f"{label:<25} {v43:>12.0f} {v49:>12.0f} {v46:>12.0f} {v49-v43:>+8.0f} {v46-v43:>+8.0f} {v49-v46:>+8.0f}")
        elif key == "win_rate":
            print(f"{label:<25} {v43*100:>11.1f}% {v49*100:>11.1f}% {v46*100:>11.1f}% {(v49-v43)*100:>+7.1f}% {(v46-v43)*100:>+7.1f}% {(v49-v46)*100:>+7.1f}%")
        else:
            print(f"{label:<25} {v43*100:>11.1f}% {v49*100:>11.1f}% {v46*100:>11.1f}% {(v49-v43)*100:>+7.1f}% {(v46-v43)*100:>+7.1f}% {(v49-v46)*100:>+7.1f}%")

    print()
    risk_metrics = [
        ("Max Drawdown", "max_drawdown"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Ann. Return", "annual_return"),
    ]
    for label, key in risk_metrics:
        v43 = result_43["risk_metrics"][key]
        v49 = result_49["risk_metrics"][key]
        v46 = result_46["risk_metrics"][key]
        if key == "sharpe_ratio":
            print(f"{label:<25} {v43:>12.2f} {v49:>12.2f} {v46:>12.2f} {v49-v43:>+8.2f} {v46-v43:>+8.2f} {v49-v46:>+8.2f}")
        else:
            print(f"{label:<25} {v43*100:>11.1f}% {v49*100:>11.1f}% {v46*100:>11.1f}% {(v49-v43)*100:>+7.1f}% {(v46-v43)*100:>+7.1f}% {(v49-v46)*100:>+7.1f}%")

    print()
    ho_metrics = [
        ("HO Return", "ho_return"),
        ("HO MDD", "ho_mdd"),
        ("HO Sharpe", "ho_sharpe"),
        ("HO Worst Month", "ho_worst_month"),
    ]
    for label, key in ho_metrics:
        v43 = ho_43[key]
        v49 = ho_49[key]
        v46 = ho_46[key]
        if key == "ho_sharpe":
            print(f"{label:<25} {v43:>12.2f} {v49:>12.2f} {v46:>12.2f} {v49-v43:>+8.2f} {v46-v43:>+8.2f} {v49-v46:>+8.2f}")
        else:
            print(f"{label:<25} {v43*100:>11.1f}% {v49*100:>11.1f}% {v46*100:>11.1f}% {(v49-v43)*100:>+7.1f}% {(v46-v43)*100:>+7.1f}% {(v49-v46)*100:>+7.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # Attribution decomposition
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ATTRIBUTION DECOMPOSITION (HO Return)")
    print("=" * 80)
    ba = (ho_49["ho_return"] - ho_43["ho_return"]) * 100
    ca = (ho_46["ho_return"] - ho_43["ho_return"]) * 100
    bc = (ho_49["ho_return"] - ho_46["ho_return"]) * 100
    print(f"  Total gain (B-A): {ba:+.1f}pp")
    print(f"    From QDII ranking change (C-A): {ca:+.1f}pp")
    print(f"    From new A-share ETFs (B-C):    {bc:+.1f}pp")

    # ─────────────────────────────────────────────────────────────────────────
    # Trace holdings for comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HOLDINGS TRACE: 43-ETF vs 49-ETF (Holdout Period)")
    print("=" * 80)

    print("\nTracing holdings for 43-ETF...")
    holdings_43 = trace_holdings(data_43, config_49)
    print(f"  {len(holdings_43)} rebalance points")

    print("Tracing holdings for 49-ETF...")
    holdings_49 = trace_holdings(data_49, config_49)
    print(f"  {len(holdings_49)} rebalance points")

    print("Tracing holdings for 46-ETF...")
    holdings_46 = trace_holdings(data_46, config_49)
    print(f"  {len(holdings_46)} rebalance points")

    # Filter to holdout period
    training_end_str = training_end
    ho_43_h = [h for h in holdings_43 if h["date"] > training_end_str]
    ho_49_h = [h for h in holdings_49 if h["date"] > training_end_str]
    ho_46_h = [h for h in holdings_46 if h["date"] > training_end_str]

    print(f"\nHoldout rebalance points: A={len(ho_43_h)}, B={len(ho_49_h)}, C={len(ho_46_h)}")

    # Compare holdings between A and B
    print(f"\n{'Date':<12} {'43-ETF Holdings':<30} {'49-ETF Holdings':<30} {'Same?':<6} {'New ETF?'}")
    print("-" * 100)

    # Align by date
    dates_43 = {h["date"]: h for h in ho_43_h}
    dates_49 = {h["date"]: h for h in ho_49_h}
    dates_46 = {h["date"]: h for h in ho_46_h}

    all_dates = sorted(set(list(dates_43.keys()) + list(dates_49.keys())))

    divergent_count = 0
    new_etf_selected_count = 0
    new_etf_selections = {etf: 0 for etf in sorted(NEW_A_SHARE)}

    for d in all_dates:
        h43 = dates_43.get(d, {}).get("holdings", [])
        h49 = dates_49.get(d, {}).get("holdings", [])
        same = set(h43) == set(h49)
        has_new = any(e in NEW_A_SHARE for e in h49)

        if not same:
            divergent_count += 1
            new_marker = ""
            for e in h49:
                if e in NEW_A_SHARE:
                    new_etf_selected_count += 1
                    new_etf_selections[e] = new_etf_selections.get(e, 0) + 1
                    new_marker = f"  <-- {e}"

            print(f"{d:<12} {', '.join(h43):<30} {', '.join(h49):<30} {'Y' if same else 'N':<6} {new_marker}")

    print(f"\nTotal holdout rebalance points: {len(all_dates)}")
    print(f"Divergent points: {divergent_count} ({divergent_count/max(len(all_dates),1)*100:.1f}%)")
    print(f"Points with new A-share ETF selected: {new_etf_selected_count}")
    for etf, count in sorted(new_etf_selections.items()):
        print(f"  {etf}: selected {count} times")

    # ─────────────────────────────────────────────────────────────────────────
    # Compare 43 vs 46 (QDII ranking effect)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("HOLDINGS TRACE: 43-ETF vs 46-ETF (QDII ranking effect)")
    print("=" * 80)
    print(f"\n{'Date':<12} {'43-ETF Holdings':<30} {'46-ETF Holdings':<30} {'Same?'}")
    print("-" * 80)

    divergent_qdii_count = 0
    for d in sorted(set(list(dates_43.keys()) + list(dates_46.keys()))):
        h43 = dates_43.get(d, {}).get("holdings", [])
        h46 = dates_46.get(d, {}).get("holdings", [])
        same = set(h43) == set(h46)
        if not same:
            divergent_qdii_count += 1
            print(f"{d:<12} {', '.join(h43):<30} {', '.join(h46):<30} {'Y' if same else 'N'}")

    print(f"\nDivergent points (43 vs 46): {divergent_qdii_count}")

    # ─────────────────────────────────────────────────────────────────────────
    # Robustness assessment
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ROBUSTNESS ASSESSMENT")
    print("=" * 80)

    print(f"\n1. Attribution decomposition:")
    print(f"   B-A = {ba:+.1f}pp (total pool expansion effect)")
    print(f"   C-A = {ca:+.1f}pp (QDII ranking perturbation, no new tradeable ETFs)")
    print(f"   B-C = {bc:+.1f}pp (new A-share ETF direct contribution)")

    if abs(ca) > abs(bc):
        print(f"\n   >> QDII ranking perturbation is LARGER than new A-share contribution")
        print(f"   >> Adding QDII to the pool (even untradeable) changes cross-sectional")
        print(f"   >> ranks enough to alter A-share ETF selection. This is a sensitivity concern.")
    elif abs(bc) > abs(ca):
        print(f"\n   >> New A-share ETFs contribute MORE than QDII ranking perturbation")
        print(f"   >> The gain is primarily from 159985/512200/515220 being better picks.")
    else:
        print(f"\n   >> Both effects are similar in magnitude.")

    print(f"\n2. Holdout divergence rate: {divergent_count/max(len(all_dates),1)*100:.1f}%")
    if divergent_count / max(len(all_dates), 1) > 0.3:
        print(f"   >> HIGH divergence ({divergent_count}/{len(all_dates)}): pool composition")
        print(f"   >> materially affects C2's selection behavior.")
    else:
        print(f"   >> LOW divergence: pool expansion has limited effect on most rebalances.")

    print(f"\n3. Risk impact:")
    mdd_diff_ba = (ho_49["ho_mdd"] - ho_43["ho_mdd"]) * 100
    sharpe_diff_ba = ho_49["ho_sharpe"] - ho_43["ho_sharpe"]
    print(f"   MDD change (B-A): {mdd_diff_ba:+.1f}pp")
    print(f"   Sharpe change (B-A): {sharpe_diff_ba:+.2f}")
    if ho_49["ho_mdd"] > ho_43["ho_mdd"] + 0.03:
        print(f"   >> WARNING: Pool expansion increases MDD by >3pp")
    else:
        print(f"   >> Risk profile acceptable")


if __name__ == "__main__":
    main()
