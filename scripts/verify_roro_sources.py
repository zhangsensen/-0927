#!/usr/bin/env python3
"""
RORO Source Verification Script
Isolates the contribution of Gold/Bond Beta vs. Equity Alpha.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "etf_rotation_optimized"))

import yaml
import pandas as pd
import numpy as np
from numba import njit

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule
from core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views

# Reuse the kernel from develop_vec_roro.py (copy-pasted for standalone execution)
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252
STOP_LOSS_PCT = 0.12
DD_LIMIT_SOFT = 0.15
DD_LIMIT_HARD = 0.25

@njit(cache=True)
def vec_backtest_kernel_roro(
    factors_3d,
    close_prices,
    open_prices,
    low_prices,
    timing_arr,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
    stop_loss_pct,
    dd_limit_soft,
    dd_limit_hard,
    ro_prices, 
    ro_mom,    
):
    T, N, _ = factors_3d.shape
    M = ro_prices.shape[1]

    cash = initial_capital
    eq_holdings = np.full(N, -1.0)
    eq_entry_prices = np.zeros(N)
    
    ro_holding_idx = -1 
    ro_holding_shares = 0.0
    
    peak_value = initial_capital
    max_drawdown = 0.0
    
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    stop_loss_hits = 0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    for i in range(len(rebalance_schedule)):
        t = rebalance_schedule[i]
        if i < len(rebalance_schedule) - 1:
            next_t = rebalance_schedule[i+1]
        else:
            next_t = T
        if t >= T: break

        # 1. Calculate Value
        eq_value = 0.0
        for n in range(N):
            if eq_holdings[n] > 0.0:
                eq_value += eq_holdings[n] * close_prices[t, n]
        
        ro_value = 0.0
        if ro_holding_idx >= 0:
            ro_value = ro_holding_shares * ro_prices[t, ro_holding_idx]
            
        current_total_value = cash + eq_value + ro_value
        
        if current_total_value > peak_value: peak_value = current_total_value
        drawdown = 1.0 - (current_total_value / peak_value)
        if drawdown > max_drawdown: max_drawdown = drawdown
        
        # 2. Risk Scalar
        risk_scalar = 1.0
        if drawdown > dd_limit_hard: risk_scalar = 0.0
        elif drawdown > dd_limit_soft: risk_scalar = 0.5
            
        # 3. Score Equities
        valid = 0
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
                valid += 1
            else:
                combined_score[n] = -np.inf

        for n in range(N): target_set[n] = False
        buy_count = 0
        if valid >= pos_size and risk_scalar > 0.0:
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf: break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # 4. Target Allocation
        target_eq_ratio = timing_arr[t] * risk_scalar
        target_eq_capital = current_total_value * target_eq_ratio
        
        # 5. Sell Equities
        for n in range(N):
            should_sell = (eq_holdings[n] > 0.0) and (not target_set[n] or risk_scalar == 0.0)
            if should_sell:
                price = close_prices[t, n]
                proceeds = eq_holdings[n] * price * (1.0 - commission_rate)
                cash += proceeds
                pnl = (price - eq_entry_prices[n]) / eq_entry_prices[n]
                if pnl > 0.0:
                    wins += 1
                    total_win_pnl += pnl
                else:
                    losses += 1
                    total_loss_pnl += abs(pnl)
                eq_holdings[n] = -1.0
                eq_entry_prices[n] = 0.0

        # 6. Buy Equities
        current_eq_value = 0.0
        for n in range(N):
            if eq_holdings[n] > 0.0:
                current_eq_value += eq_holdings[n] * close_prices[t, n]
        
        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if eq_holdings[idx] < 0.0:
                new_targets[new_count] = idx
                new_count += 1
        
        if new_count > 0:
            available_for_new = target_eq_capital - current_eq_value
            if available_for_new < 0.0: available_for_new = 0.0
            if available_for_new > cash: available_for_new = cash
            
            target_pos_value = available_for_new / new_count / (1.0 + commission_rate)
            if target_pos_value > 0.0:
                for k in range(new_count):
                    idx = new_targets[k]
                    price = close_prices[t, idx]
                    if np.isnan(price) or price <= 0.0: continue
                    shares = target_pos_value / price
                    cost = shares * price * (1.0 + commission_rate)
                    if cash >= cost - 1e-5:
                        actual_cost = cost
                        if actual_cost > cash: actual_cost = cash
                        cash -= actual_cost
                        eq_holdings[idx] = shares
                        eq_entry_prices[idx] = price

        # 7. Risk-Off Logic
        best_ro_idx = -1
        best_ro_mom = -999.0
        
        for m in range(M):
            mom = ro_mom[t, m]
            if not np.isnan(mom) and mom > 0.0:
                if mom > best_ro_mom:
                    best_ro_mom = mom
                    best_ro_idx = m
        
        if ro_holding_idx != -1 and ro_holding_idx != best_ro_idx:
            price = ro_prices[t, ro_holding_idx]
            proceeds = ro_holding_shares * price * (1.0 - commission_rate)
            cash += proceeds
            ro_holding_idx = -1
            ro_holding_shares = 0.0
            
        if best_ro_idx != -1 and ro_holding_idx == -1:
            invest_amt = cash * 0.99
            if invest_amt > 0:
                price = ro_prices[t, best_ro_idx]
                if not np.isnan(price) and price > 0:
                    shares = invest_amt / price
                    cost = shares * price * (1.0 + commission_rate)
                    if cash >= cost:
                        cash -= cost
                        ro_holding_idx = best_ro_idx
                        ro_holding_shares = shares

        # 8. Stop Loss
        check_start = t + 1
        check_end = next_t + 1
        if check_start < T:
            if check_end > T: check_end = T
            for n in range(N):
                if eq_holdings[n] > 0.0:
                    entry = eq_entry_prices[n]
                    stop_price = entry * (1.0 - stop_loss_pct)
                    triggered = False
                    for day in range(check_start, check_end):
                        if low_prices[day, n] < stop_price:
                            triggered = True
                            break
                    if triggered:
                        stop_loss_hits += 1
                        proceeds = eq_holdings[n] * stop_price * (1.0 - commission_rate)
                        cash += proceeds
                        pnl = (stop_price - entry) / entry
                        losses += 1
                        total_loss_pnl += abs(pnl)
                        eq_holdings[n] = -1.0
                        eq_entry_prices[n] = 0.0

    # Final Liquidation
    final_value = cash
    for n in range(N):
        if eq_holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price): price = eq_entry_prices[n]
            final_value += eq_holdings[n] * price * (1.0 - commission_rate)
    if ro_holding_idx != -1:
        price = ro_prices[T-1, ro_holding_idx]
        final_value += ro_holding_shares * price * (1.0 - commission_rate)

    total_return = (final_value - initial_capital) / initial_capital
    return total_return, max_drawdown

def run_scenario(scenario_name, ro_symbols, factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, etf_codes):
    print(f"\nRunning Scenario: {scenario_name}")
    print(f"Risk-Off Assets: {ro_symbols}")
    
    if not ro_symbols:
        ro_prices = np.zeros((close_prices.shape[0], 0))
        ro_mom = np.zeros((close_prices.shape[0], 0))
    else:
        valid_indices = []
        for sym in ro_symbols:
            if sym in etf_codes:
                valid_indices.append(etf_codes.index(sym))
            else:
                print(f"Warning: {sym} not found")
        
        if not valid_indices and ro_symbols:
            print("Error: No valid RO assets found")
            return 0.0, 0.0
            
        ro_prices = close_prices[:, valid_indices]
        ro_mom = np.zeros_like(ro_prices)
        ro_mom[20:] = ro_prices[20:] / ro_prices[:-20] - 1.0
        ro_mom[:20] = 0.0

    T = factors_3d.shape[0]
    rebalance_schedule = generate_rebalance_schedule(total_periods=T, lookback_window=LOOKBACK, freq=FREQ)
    
    ret, mdd = vec_backtest_kernel_roro(
        factors_3d, close_prices, open_prices, low_prices, timing_arr, 
        np.array(factor_indices, dtype=np.int64), rebalance_schedule, 
        POS_SIZE, INITIAL_CAPITAL, COMMISSION_RATE, 
        STOP_LOSS_PCT, DD_LIMIT_SOFT, DD_LIMIT_HARD,
        ro_prices, ro_mom
    )
    
    print(f"Result -> Return: {ret*100:.1f}%, MaxDD: {mdd*100:.1f}%")
    return ret, mdd

def main():
    # 1. Load Data
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    loader = DataLoader(data_dir=config["data"].get("data_dir"), cache_dir=config["data"].get("cache_dir"))
    ohlcv = loader.load_ohlcv(etf_codes=config["data"]["symbols"], start_date=config["data"]["start_date"], end_date=config["data"]["end_date"])

    # 2. Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    factor_names = sorted(std_factors.keys())
    etf_codes = std_factors[factor_names[0]].columns.tolist()
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_series.reindex(std_factors[factor_names[0]].index).fillna(1.0).values)

    # 3. Rank 3 Strategy
    # ADX_14D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D
    target_combo = "ADX_14D + PRICE_POSITION_20D + SHARPE_RATIO_20D + SLOPE_20D + VOL_RATIO_20D"
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    factor_indices = [factor_index_map[f.strip()] for f in target_combo.split(" + ")]
    
    print("=" * 80)
    print(f"🔍 Verifying Sources of Alpha for: {target_combo}")
    print("=" * 80)

    # Scenario 1: Gold + Bond (Current)
    run_scenario("Gold + Bond", ['518880', '511010'], factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, etf_codes)
    
    # Scenario 2: Bond Only
    run_scenario("Bond Only", ['511010'], factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, etf_codes)
    
    # Scenario 3: Cash Only
    run_scenario("Cash Only", [], factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, etf_codes)

if __name__ == "__main__":
    main()
