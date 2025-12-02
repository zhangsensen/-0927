#!/usr/bin/env python3
"""
VEC Risk-On / Risk-Off (RORO) Development Script
Implements "Smart Cash" logic: When Equity exposure is reduced, invest in Gold/Bonds.

🔧 修复说明：
- 避险资产（黄金518880、国债511010）不再参与权益因子选股
- 避免双重计算问题
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
STOP_LOSS_PCT = 0.12
DD_LIMIT_SOFT = 0.15
DD_LIMIT_HARD = 0.25

# 🔧 避险资产列表（不参与权益因子选股）
SAFE_ASSET_SYMBOLS = ['518880', '511010', '518850', '511260', '511380']  # 黄金、国债、白银、十年国债、转债

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
    # RORO Params
    ro_prices, # [T, M] Risk-Off Asset Prices (Gold, Bond)
    ro_mom,    # [T, M] Risk-Off Asset Momentum (e.g. 20D Return)
    # 🔧 新增：排除列表
    excluded_indices,  # [K] 需要排除的ETF索引（避险资产）
):
    T, N, _ = factors_3d.shape
    M = ro_prices.shape[1]
    K = len(excluded_indices)  # 🔧 排除的ETF数量

    cash = initial_capital
    
    # Equity Holdings
    eq_holdings = np.full(N, -1.0)
    eq_entry_prices = np.zeros(N)
    
    # Risk-Off Holdings (Single slot for simplicity: Best of RO assets)
    ro_holding_idx = -1 # -1 means Cash
    ro_holding_shares = 0.0
    ro_entry_price = 0.0
    
    # Track Peak Value for Drawdown Control
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
    
    # 🔧 构建排除集合的快速查找（Numba兼容方式）
    is_excluded = np.zeros(N, dtype=np.bool_)
    for k in range(K):
        idx = excluded_indices[k]
        if 0 <= idx < N:
            is_excluded[idx] = True

    for i in range(len(rebalance_schedule)):
        t = rebalance_schedule[i]
        
        if i < len(rebalance_schedule) - 1:
            next_t = rebalance_schedule[i+1]
        else:
            next_t = T

        if t >= T:
            break

        # 1. Calculate Portfolio Value
        eq_value = 0.0
        for n in range(N):
            if eq_holdings[n] > 0.0:
                eq_value += eq_holdings[n] * close_prices[t, n]
        
        ro_value = 0.0
        if ro_holding_idx >= 0:
            ro_value = ro_holding_shares * ro_prices[t, ro_holding_idx]
            
        current_total_value = cash + eq_value + ro_value
        
        if current_total_value > peak_value:
            peak_value = current_total_value
        
        drawdown = 1.0 - (current_total_value / peak_value)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        # 2. Determine Risk Scalar
        risk_scalar = 1.0
        if drawdown > dd_limit_hard:
            risk_scalar = 0.0
        elif drawdown > dd_limit_soft:
            risk_scalar = 0.5
            
        # 3. Score Equities (🔧 排除避险资产)
        valid = 0
        for n in range(N):
            # 🔧 跳过避险资产
            if is_excluded[n]:
                combined_score[n] = -np.inf
                continue
                
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
        if valid >= pos_size and risk_scalar > 0.0:
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf:
                    break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # 4. Target Allocation
        # Equity Exposure = Timing * RiskScalar
        # Safe Exposure = 1.0 - Equity Exposure
        
        target_eq_ratio = timing_arr[t] * risk_scalar
        target_eq_capital = current_total_value * target_eq_ratio
        target_safe_capital = current_total_value - target_eq_capital
        
        # 5. Sell Logic (Equities)
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

        # 6. Buy Logic (Equities)
        # Recalculate cash after sells
        # We need to reserve cash for Safe Asset later? 
        # Strategy: Prioritize Equity Target. Remaining goes to Safe.
        
        current_eq_value = 0.0
        for n in range(N):
            if eq_holdings[n] > 0.0:
                current_eq_value += eq_holdings[n] * close_prices[t, n]
        
        # How much more to buy?
        # We want total equity value ~= target_eq_capital
        # But we only rebalance the "new targets". Existing targets are kept.
        # This is a simplification.
        
        new_count = 0
        for k in range(buy_count):
            idx = buy_order[k]
            if eq_holdings[idx] < 0.0:
                new_targets[new_count] = idx
                new_count += 1
        
        if new_count > 0:
            # Available for new buys = Target - Kept
            available_for_new = target_eq_capital - current_eq_value
            if available_for_new < 0.0: available_for_new = 0.0
            
            # Cap by actual cash
            if available_for_new > cash:
                available_for_new = cash
            
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

        # 7. Risk-Off Asset Logic
        # Remaining Cash is candidate for Risk-Off
        # Determine Best Risk-Off Asset
        
        best_ro_idx = -1
        best_ro_mom = -999.0
        
        # Only consider RO assets if they have positive momentum (Trend Following on Safety)
        # If all negative, stay in Cash.
        for m in range(M):
            mom = ro_mom[t, m]
            if not np.isnan(mom) and mom > 0.0: # Only buy if uptrend
                if mom > best_ro_mom:
                    best_ro_mom = mom
                    best_ro_idx = m
        
        # Rebalance Risk-Off
        # If we hold something different than best_ro_idx, sell it.
        if ro_holding_idx != -1 and ro_holding_idx != best_ro_idx:
            # Sell current RO
            price = ro_prices[t, ro_holding_idx]
            proceeds = ro_holding_shares * price * (1.0 - commission_rate)
            cash += proceeds
            ro_holding_idx = -1
            ro_holding_shares = 0.0
            ro_entry_price = 0.0
            
        # If we have a target RO and cash, buy it
        if best_ro_idx != -1 and ro_holding_idx == -1:
            # Invest all remaining cash (minus buffer?)
            invest_amt = cash * 0.99 # Leave 1% buffer just in case
            if invest_amt > 0:
                price = ro_prices[t, best_ro_idx]
                if not np.isnan(price) and price > 0:
                    shares = invest_amt / price
                    cost = shares * price * (1.0 + commission_rate)
                    if cash >= cost:
                        cash -= cost
                        ro_holding_idx = best_ro_idx
                        ro_holding_shares = shares
                        ro_entry_price = price
        
        # If we already hold best_ro_idx, do we add/trim?
        # For simplicity, let's just hold. The equity logic dominates capital usage.
        # If equity sold and generated cash, we might want to add to RO?
        # Let's add a simple "Sweep" logic: if Cash > Threshold, buy more RO.
        if best_ro_idx != -1 and ro_holding_idx == best_ro_idx:
             if cash > 10000: # Arbitrary threshold
                 invest_amt = cash * 0.99
                 price = ro_prices[t, best_ro_idx]
                 shares = invest_amt / price
                 cost = shares * price * (1.0 + commission_rate)
                 if cash >= cost:
                     cash -= cost
                     ro_holding_shares += shares
                     # Update entry price? Weighted avg? Not strictly needed for PnL calc here.

        # 8. Intra-period Stop Loss (Equities Only)
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
    # Equities
    for n in range(N):
        if eq_holdings[n] > 0.0:
            price = close_prices[T - 1, n]
            if np.isnan(price): price = eq_entry_prices[n]
            final_value += eq_holdings[n] * price * (1.0 - commission_rate)
            pnl = (price - eq_entry_prices[n]) / eq_entry_prices[n]
            if pnl > 0.0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)
    # Risk-Off
    if ro_holding_idx != -1:
        price = ro_prices[T-1, ro_holding_idx]
        final_value += ro_holding_shares * price * (1.0 - commission_rate)

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

def run_vec_roro(factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, ro_prices, ro_mom, excluded_indices):
    """🔧 修改：增加 excluded_indices 参数"""
    T = factors_3d.shape[0]
    _, open_arr, close_arr = ensure_price_views(close_prices, open_prices, copy_if_missing=True, warn_if_copied=False, validate=False)
    if low_prices is None: low_arr = close_arr
    else: low_arr = low_prices

    rebalance_schedule = generate_rebalance_schedule(total_periods=T, lookback_window=LOOKBACK, freq=FREQ)
    
    return vec_backtest_kernel_roro(
        factors_3d, close_arr, open_arr, low_arr, timing_arr, 
        np.array(factor_indices, dtype=np.int64), rebalance_schedule, 
        POS_SIZE, INITIAL_CAPITAL, COMMISSION_RATE, 
        STOP_LOSS_PCT, DD_LIMIT_SOFT, DD_LIMIT_HARD,
        ro_prices, ro_mom,
        np.array(excluded_indices, dtype=np.int64)  # 🔧 传入排除列表
    )

def main():
    print("=" * 80)
    print("🌤️ All-Weather Strategy Development (RORO)")
    print(f"   Risk-Off Assets: Gold (518880), Bond (511010)")
    print(f"   Logic: When Equity Exposure < 100%, invest Cash in Best RO Asset (if Mom > 0)")
    print("=" * 80)

    # 1. Load Top 100
    input_path = ROOT / "results/vec_full_backtest_20251128_185610/wfo_vec_scored.csv"
    df_scored = pd.read_csv(input_path)
    df_top = df_scored.nlargest(100, 'score_balanced')

    # 2. Load Data
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    loader = DataLoader(data_dir=config["data"].get("data_dir"), cache_dir=config["data"].get("cache_dir"))
    ohlcv = loader.load_ohlcv(etf_codes=config["data"]["symbols"], start_date=config["data"]["start_date"], end_date=config["data"]["end_date"])

    # 3. Factors & Prices
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
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = shift_timing_signal(timing_series.reindex(dates).fillna(1.0).values)

    # 4. Prepare Risk-Off Data
    ro_symbols = ['518880', '511010'] # Gold, Bond
    # Check if they exist in etf_codes
    valid_ro_indices = []
    for sym in ro_symbols:
        if sym in etf_codes:
            valid_ro_indices.append(etf_codes.index(sym))
        else:
            print(f"⚠️ Warning: Risk-Off Asset {sym} not found in data.")
    
    if not valid_ro_indices:
        print("❌ No Risk-Off assets found. Aborting.")
        return

    ro_prices = close_prices[:, valid_ro_indices]
    
    # Calculate RO Momentum (20D Return)
    # Simple numpy calc
    ro_mom = np.zeros_like(ro_prices)
    ro_mom[20:] = ro_prices[20:] / ro_prices[:-20] - 1.0
    ro_mom[:20] = 0.0 # No momentum initially
    
    # 🔧 构建排除列表：避险资产不参与因子选股
    excluded_indices = []
    for sym in SAFE_ASSET_SYMBOLS:
        if sym in etf_codes:
            excluded_indices.append(etf_codes.index(sym))
            print(f"🔧 排除 {sym} 不参与因子选股")
    
    print(f"🔧 共排除 {len(excluded_indices)} 个避险资产")

    # 5. Run Backtest
    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    
    for _, row in tqdm(df_top.iterrows(), total=len(df_top), desc="RORO VEC"):
        combo_str = row['combo']
        factor_indices = [factor_index_map[f.strip()] for f in combo_str.split(" + ")]
        
        ret, wr, pf, trades, sl_hits, mdd = run_vec_roro(
            factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices,
            ro_prices, ro_mom, excluded_indices  # 🔧 传入排除列表
        )
        
        results.append({
            "combo": combo_str,
            "orig_return": row['vec_return'],
            "roro_return": ret,
            "roro_mdd": mdd,
            "roro_pf": pf
        })

    df_res = pd.DataFrame(results)
    
    print("\n📊 All-Weather Strategy Impact (🔧 已排除避险资产参与选股)")
    print("-" * 60)
    print(f"Mean Return: {df_res['orig_return'].mean()*100:.1f}% -> {df_res['roro_return'].mean()*100:.1f}%")
    print(f"Mean MaxDD:   (N/A) -> {df_res['roro_mdd'].mean()*100:.1f}%")
    print(f"Mean PF:      (N/A) -> {df_res['roro_pf'].mean():.2f}")
    
    output_path = ROOT / "results/vec_roro_test.csv"
    df_res.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
