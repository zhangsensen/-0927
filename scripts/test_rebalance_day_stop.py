#!/usr/bin/env python3
"""
测试止损检查时机的影响：
1. 每天检查止损（当前方案）
2. 仅在调仓日检查止损（新方案）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np
from numba import njit

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule


@njit
def stable_topk_indices(scores, k):
    N = len(scores)
    result = np.empty(k, dtype=np.int64)
    used = np.zeros(N, dtype=np.bool_)
    
    for i in range(k):
        best_idx = -1
        best_score = -np.inf
        for n in range(N):
            if used[n]:
                continue
            if scores[n] > best_score or (scores[n] == best_score and (best_idx < 0 or n < best_idx)):
                best_score = scores[n]
                best_idx = n
        if best_idx < 0 or best_score == -np.inf:
            return result[:i]
        result[i] = best_idx
        used[best_idx] = True
    return result


@njit
def backtest_with_stop_timing(
    factors_3d,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    factor_indices,
    rebalance_schedule,
    pos_size,
    initial_capital,
    commission_rate,
    trailing_stop_pct,
    stop_check_mode,  # 0 = 每天检查, 1 = 仅调仓日检查
):
    """带止损检查时机选择的回测"""
    T, N, _ = factors_3d.shape

    cash = initial_capital
    holdings = np.full(N, -1.0)
    entry_prices = np.zeros(N)
    high_water_marks = np.zeros(N)
    
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0

    combined_score = np.empty(N)
    target_set = np.zeros(N, dtype=np.bool_)
    buy_order = np.empty(pos_size, dtype=np.int64)
    new_targets = np.empty(pos_size, dtype=np.int64)

    num_stops = 0
    
    rebal_ptr = 0
    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 252
    
    for t in range(start_day, T):
        is_rebalance_day = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance_day = True
            rebal_ptr += 1
        
        # 止损检查：根据模式决定是否检查
        should_check_stop = (stop_check_mode == 0) or (stop_check_mode == 1 and is_rebalance_day)
        
        if trailing_stop_pct > 0.0 and should_check_stop:
            for n in range(N):
                if holdings[n] > 0.0:
                    prev_hwm = high_water_marks[n]
                    stop_price = prev_hwm * (1.0 - trailing_stop_pct)
                    curr_low = low_prices[t, n]
                    curr_open = open_prices[t, n]
                    
                    if not np.isnan(curr_low) and curr_low < stop_price:
                        num_stops += 1
                        
                        if not np.isnan(curr_open) and curr_open < stop_price:
                            exec_price = curr_open
                        else:
                            exec_price = stop_price
                        
                        if not np.isnan(curr_low):
                            exec_price = max(exec_price, curr_low)
                        
                        proceeds = holdings[n] * exec_price * (1.0 - commission_rate)
                        cash += proceeds
                        
                        pnl = (exec_price - entry_prices[n]) / entry_prices[n]
                        if pnl > 0.0:
                            wins += 1
                            total_win_pnl += pnl
                        else:
                            losses += 1
                            total_loss_pnl += abs(pnl)
                        
                        holdings[n] = -1.0
                        entry_prices[n] = 0.0
                        high_water_marks[n] = 0.0
                    else:
                        curr_high = high_prices[t, n]
                        if not np.isnan(curr_high) and curr_high > high_water_marks[n]:
                            high_water_marks[n] = curr_high

        # 调仓逻辑（简化版）
        if is_rebalance_day:
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
            if valid >= pos_size:
                top_indices = stable_topk_indices(combined_score, pos_size)
                for k in range(len(top_indices)):
                    idx = top_indices[k]
                    if combined_score[idx] == -np.inf:
                        break
                    target_set[idx] = True
                    buy_order[buy_count] = idx
                    buy_count += 1

            timing_ratio = timing_arr[t]

            for n in range(N):
                if holdings[n] > 0.0 and not target_set[n]:
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
                    high_water_marks[n] = 0.0

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
                if holdings[idx] < 0.0:
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

                        if cash >= cost - 1e-5 and cost > 0.0:
                            actual_cost = cost if cost <= cash else cash
                            actual_shares = actual_cost / (price * (1.0 + commission_rate))
                            cash -= actual_cost
                            holdings[idx] = shares
                            entry_prices[idx] = price
                            high_water_marks[idx] = price

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

    return total_return, num_trades, num_stops


def main():
    print("=" * 80)
    print("止损检查时机对比测试")
    print("=" * 80)

    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq")
    POS_SIZE = backtest_config.get("pos_size")
    LOOKBACK = backtest_config.get("lookback")
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital"))
    COMMISSION_RATE = float(backtest_config.get("commission_rate"))

    # 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

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
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values

    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 加载 WFO 组合
    wfo_dirs = sorted([d for d in (ROOT / "results").glob("run_*") if d.is_dir() and not d.is_symlink()])
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "top100_by_ic.parquet"
    if not combos_path.exists():
        combos_path = latest_wfo / "all_combos.parquet"
    df_combos = pd.read_parquet(combos_path)

    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

    # 测试前 20 个组合
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    
    print("\n测试前 20 个组合:")
    print(f"{'止损率':>8s} {'检查模式':>12s} {'平均收益':>12s} {'止损次数':>10s}")
    print("-" * 80)

    for stop_pct in [0.10]:
        for mode, mode_name in [(0, "每天"), (1, "仅调仓日")]:
            returns = []
            total_stops = 0
            
            for i in range(min(20, len(df_combos))):
                combo_str = df_combos.iloc[i]["combo"]
                factor_indices = np.array(
                    [factor_index_map[f.strip()] for f in combo_str.split(" + ")],
                    dtype=np.int64
                )
                
                ret, trades, stops = backtest_with_stop_timing(
                    factors_3d,
                    close_prices,
                    open_prices,
                    high_prices,
                    low_prices,
                    timing_arr,
                    factor_indices,
                    rebalance_schedule,
                    POS_SIZE,
                    INITIAL_CAPITAL,
                    COMMISSION_RATE,
                    stop_pct,
                    mode,
                )
                
                returns.append(ret)
                total_stops += stops
            
            avg_return = np.mean(returns) * 100
            avg_stops = total_stops / len(returns)
            print(f"{stop_pct*100:7.0f}% {mode_name:>12s} {avg_return:11.2f}% {avg_stops:9.1f}")

    print("\n" + "=" * 80)
    print("结论:")
    print("  如果'仅调仓日'模式收益更高，说明频繁止损检查是有害的")
    print("  止损应该与策略节奏一致（每 8 天调仓时检查）")


if __name__ == "__main__":
    main()
