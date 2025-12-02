#!/usr/bin/env python3
"""
VEC Risk Control Development Script
Implements Stop-Loss and Drawdown Control in Vectorized Kernel.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from numba import njit

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views

FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# Risk Parameters
STOP_LOSS_PCT = 0.12  # 12% Stop Loss (Looser)
DD_LIMIT_SOFT = 0.15  # 15% Drawdown -> Reduce Position
DD_LIMIT_HARD = 0.25  # 25% Drawdown -> Stop Trading

@njit(cache=True)
def vec_backtest_kernel_with_risk(
    factors_3d,
    close_prices,
    open_prices,
    low_prices,  # Added for Stop Loss
    timing_arr,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
    stop_loss_pct,
    dd_limit_soft,
    dd_limit_hard
):
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    
    # Track Peak Value for Drawdown Control
    peak_value = initial_capital
    max_drawdown = 0.0
    
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    
    # Stop Loss Tracking
    stop_loss_hits = 0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    # Iterate through rebalance periods
    # We need next_t to check for stop loss in between
    for i in range(len(rebalance_schedule)):
        t = rebalance_schedule[i]
        
        # Determine next rebalance date (or end of data)
        if i < len(rebalance_schedule) - 1:
            next_t = rebalance_schedule[i+1]
        else:
            next_t = T

        if t >= T:
            break

        # 1. Calculate Portfolio Value & Drawdown
        current_value = cash
        for n in range(N):
            if holdings[n] > 0.0:
                current_value += holdings[n] * close_prices[t, n]
        
        if current_value > peak_value:
            peak_value = current_value
        
        drawdown = 1.0 - (current_value / peak_value)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        # 2. Determine Risk Scalar based on Drawdown
        risk_scalar = 1.0
        if drawdown > dd_limit_hard:
            risk_scalar = 0.0 # Stop Trading
        elif drawdown > dd_limit_soft:
            risk_scalar = 0.5 # Reduce Position
            
        # 3. Score & Select Targets
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

        for n in range(N):
            target_set[n] = False

        buy_count = 0
        if valid >= pos_size and risk_scalar > 0.0: # Only buy if risk allows
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf:
                    break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # 4. Sell Logic (Rebalance)
        timing_ratio = timing_arr[t] * risk_scalar # Apply Risk Scalar to Timing

        for n in range(N):
            # Sell if not in target OR if we need to reduce exposure (handled by re-buying less, but here we sell all non-targets first)
            # Also sell if risk_scalar is 0 (Hard Stop)
            should_sell = (holdings[n] > 0.0) and (not target_set[n] or risk_scalar == 0.0)
            
            if should_sell:
                price = close_prices[t, n]
                proceeds = holdings[n] * price * (1.0 - commission_rate)
                cash += proceeds

                pnl = (price - entry_prices[n]) / entry_prices[n]
                if pnl > 0.0:
                    wins += 1
                    total_win_pnl += pnl
                else:
                    losses += 1
                    total_loss_pnl += abs(pnl)

                holdings[n] = -1.0
                entry_prices[n] = 0.0

        # 5. Buy Logic
        # Recalculate value after sells
        current_value = cash
        kept_value = 0.0
        for n in range(N):
            if holdings[n] > 0.0:
                val = holdings[n] * close_prices[t, n]
                current_value += val
                kept_value += val

        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if holdings[idx] < 0.0: # Not held
                new_targets[new_count] = idx
                new_count += 1

        if new_count > 0:
            target_exposure = current_value * timing_ratio
            available_for_new = target_exposure - kept_value
            if available_for_new < 0.0:
                available_for_new = 0.0

            target_pos_value = available_for_new / new_count / (1.0 + commission_rate)

            if target_pos_value > 0.0:
                for k in range(new_count):
                    idx = new_targets[k]
                    price = close_prices[t, idx]
                    if np.isnan(price) or price <= 0.0:
                        continue

                    shares = target_pos_value / price
                    cost = shares * price * (1.0 + commission_rate)

                    if cash >= cost - 1e-5:
                        actual_cost = cost
                        if actual_cost > cash:
                            actual_cost = cash
                        cash -= actual_cost
                        holdings[idx] = shares
                        entry_prices[idx] = price
        
        # 6. Intra-period Stop Loss Check
        # Check from t+1 to next_t (exclusive of next_t because next_t is handled in next iteration)
        # Actually, we should check up to next_t inclusive? 
        # No, next_t is the next rebalance day. The rebalance logic happens at Close of next_t.
        # So we check Lows of [t+1, ..., next_t].
        
        check_start = t + 1
        check_end = next_t + 1 # Include next_t for stop loss check before rebalance? 
        # Usually stop loss happens intra-day. If it happens on rebalance day, we sell at stop price, not rebalance price.
        
        if check_start < T:
            if check_end > T: check_end = T
            
            for n in range(N):
                if holdings[n] > 0.0:
                    entry = entry_prices[n]
                    stop_price = entry * (1.0 - stop_loss_pct)
                    
                    # Scan days
                    triggered = False
                    trigger_day = -1
                    
                    for day in range(check_start, check_end):
                        if low_prices[day, n] < stop_price:
                            triggered = True
                            trigger_day = day
                            break
                    
                    if triggered:
                        stop_loss_hits += 1
                        # Sell at stop price (assume execution at stop price)
                        # In reality, might be lower (gap), but for VEC this is approx.
                        # Let's use stop_price as exit.
                        
                        proceeds = holdings[n] * stop_price * (1.0 - commission_rate)
                        cash += proceeds
                        
                        pnl = (stop_price - entry) / entry
                        # Stop loss is always a loss
                        losses += 1
                        total_loss_pnl += abs(pnl)
                        
                        holdings[n] = -1.0
                        entry_prices[n] = 0.0
                        # Position is now empty for the rest of the period

    # Final Liquidation
    final_value = cash
    for n in range(N):
        if holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price):
                price = entry_prices[n]

            final_value += holdings[n] * price * (1.0 - commission_rate)

            pnl = (price - entry_prices[n]) / entry_prices[n]
            if pnl > 0.0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)

    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = 0.0

    return total_return, win_rate, profit_factor, num_trades, stop_loss_hits, max_drawdown

def run_vec_backtest_risk(factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices):
    T = factors_3d.shape[0]
    
    _, open_arr, close_arr = ensure_price_views(
        close_prices,
        open_prices,
        copy_if_missing=True,
        warn_if_copied=True,
        validate=True,
        min_valid_index=LOOKBACK,
    )
    
    # Ensure low_prices
    if low_prices is None:
        low_arr = close_arr # Fallback
    else:
        low_arr = low_prices

    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )
    
    return vec_backtest_kernel_with_risk(
        factors_3d,
        close_arr,
        open_arr,
        low_arr,
        timing_arr,
        np.array(factor_indices, dtype=np.int64),
        rebalance_schedule,
        POS_SIZE,
        INITIAL_CAPITAL,
        COMMISSION_RATE,
        STOP_LOSS_PCT,
        DD_LIMIT_SOFT,
        DD_LIMIT_HARD
    )

def main():
    print("=" * 80)
    print("🛡️ VEC Risk Control Development")
    print(f"   Stop Loss: {STOP_LOSS_PCT*100}%")
    print(f"   DD Soft Limit: {DD_LIMIT_SOFT*100}% (Reduce Position)")
    print(f"   DD Hard Limit: {DD_LIMIT_HARD*100}% (Stop Trading)")
    print("=" * 80)

    # 1. Load Top 100 Scored Strategies
    input_path = ROOT / "results/vec_full_backtest_20251128_185610/wfo_vec_scored.csv"
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    df_scored = pd.read_csv(input_path)
    df_top = df_scored.nlargest(100, 'score_balanced')
    print(f"✅ Loaded Top 100 Strategies")

    # 2. Load Data
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 3. Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape

    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values # Needed for Stop Loss

    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 4. Run Backtest
    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    
    for _, row in tqdm(df_top.iterrows(), total=len(df_top), desc="Risk VEC"):
        combo_str = row['combo']
        factor_indices = [factor_index_map[f.strip()] for f in combo_str.split(" + ")]
        
        ret, wr, pf, trades, sl_hits, mdd = run_vec_backtest_risk(
            factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices
        )
        
        results.append({
            "combo": combo_str,
            "orig_return": row['vec_return'],
            "risk_return": ret,
            "orig_wr": row['vec_win_rate'],
            "risk_wr": wr,
            "sl_hits": sl_hits,
            "risk_mdd": mdd,
            "orig_pf": row['vec_profit_factor'],
            "risk_pf": pf
        })

    df_res = pd.DataFrame(results)
    
    print("\n📊 Risk Control Impact Analysis")
    print("-" * 60)
    print(f"Mean Return: {df_res['orig_return'].mean()*100:.1f}% -> {df_res['risk_return'].mean()*100:.1f}%")
    print(f"Mean WinRate: {df_res['orig_wr'].mean()*100:.1f}% -> {df_res['risk_wr'].mean()*100:.1f}%")
    print(f"Mean MaxDD:   (N/A) -> {df_res['risk_mdd'].mean()*100:.1f}%")
    print(f"Mean PF:      {df_res['orig_pf'].mean():.2f} -> {df_res['risk_pf'].mean():.2f}")
    print(f"Avg Stop Loss Hits: {df_res['sl_hits'].mean():.1f}")
    
    # Save
    output_path = ROOT / "results/vec_risk_control_test.csv"
    df_res.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
