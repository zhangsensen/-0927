#!/usr/bin/env python3
"""
止损诊断工具：追踪止损触发的详细情况
分析为什么加了止损反而效果变差
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule, ensure_price_views


@njit
def stable_topk_indices(scores, k):
    """稳定排序：按 score 降序，score 相同时按 ETF 索引升序"""
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
def diagnose_backtest_kernel(
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
):
    """带诊断信息的回测内核，追踪每一次止损触发"""
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

    # 诊断信息：预分配数组记录止损事件
    max_stop_events = 10000
    stop_event_count = 0
    # 每次止损记录: [day, etf_idx, entry_price, hwm, stop_price, exec_price, low, open, return]
    stop_events_arr = np.zeros((max_stop_events, 9), dtype=np.float64)
    
    rebal_ptr = 0
    start_day = rebalance_schedule[0] if len(rebalance_schedule) > 0 else 252
    
    for t in range(start_day, T):
        # 检查是否为调仓日
        is_rebalance_day = False
        if rebal_ptr < len(rebalance_schedule) and rebalance_schedule[rebal_ptr] == t:
            is_rebalance_day = True
            rebal_ptr += 1
        
        # 止损检查（在调仓前）
        if trailing_stop_pct > 0.0:
            for n in range(N):
                if holdings[n] > 0.0:
                    prev_hwm = high_water_marks[n]
                    stop_price = prev_hwm * (1.0 - trailing_stop_pct)
                    curr_low = low_prices[t, n]
                    curr_open = open_prices[t, n]
                    
                    if not np.isnan(curr_low) and curr_low < stop_price:
                        # 止损触发！记录详细信息
                        if not np.isnan(curr_open) and curr_open < stop_price:
                            exec_price = curr_open
                        else:
                            exec_price = stop_price
                        
                        if not np.isnan(curr_low):
                            exec_price = max(exec_price, curr_low)
                        
                        # 记录止损事件
                        if stop_event_count < max_stop_events:
                            pnl = (exec_price - entry_prices[n]) / entry_prices[n]
                            stop_events_arr[stop_event_count, 0] = t
                            stop_events_arr[stop_event_count, 1] = n
                            stop_events_arr[stop_event_count, 2] = entry_prices[n]
                            stop_events_arr[stop_event_count, 3] = prev_hwm
                            stop_events_arr[stop_event_count, 4] = stop_price
                            stop_events_arr[stop_event_count, 5] = exec_price
                            stop_events_arr[stop_event_count, 6] = curr_low
                            stop_events_arr[stop_event_count, 7] = curr_open
                            stop_events_arr[stop_event_count, 8] = pnl
                            stop_event_count += 1
                        
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
                        # 未触发止损，更新 HWM
                        curr_high = high_prices[t, n]
                        if not np.isnan(curr_high) and curr_high > high_water_marks[n]:
                            high_water_marks[n] = curr_high

        # 调仓逻辑
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

            # 卖出
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

    # 平仓
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

    return total_return, num_trades, stop_events_arr[:stop_event_count]


def main():
    print("=" * 80)
    print("止损诊断分析")
    print("=" * 80)

    # 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    backtest_config = config.get("backtest", {})
    FREQ = backtest_config.get("freq")
    POS_SIZE = backtest_config.get("pos_size")
    LOOKBACK = backtest_config.get("lookback")
    INITIAL_CAPITAL = float(backtest_config.get("initial_capital"))
    COMMISSION_RATE = float(backtest_config.get("commission_rate"))

    # 加载 WFO 结果
    wfo_dirs = sorted([d for d in (ROOT / "results").glob("run_*") if d.is_dir() and not d.is_symlink()])
    latest_wfo = wfo_dirs[-1]
    combos_path = latest_wfo / "top100_by_ic.parquet"
    if not combos_path.exists():
        combos_path = latest_wfo / "all_combos.parquet"
    df_combos = pd.read_parquet(combos_path)

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

    # 计算因子
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

    # 择时
    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 测试单个组合（最佳组合）
    combo_str = df_combos.iloc[0]["combo"]
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    factor_indices = np.array(
        [factor_index_map[f.strip()] for f in combo_str.split(" + ")],
        dtype=np.int64
    )

    rebalance_schedule = generate_rebalance_schedule(
        total_periods=T,
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

    print(f"\n分析组合: {combo_str}")
    print(f"回测参数: FREQ={FREQ}, POS_SIZE={POS_SIZE}, 初始资金={INITIAL_CAPITAL:,.0f}")
    print()

    # 测试不同止损率
    stop_rates = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    print("=" * 80)
    print("不同止损率对比")
    print("=" * 80)
    print(f"{'止损率':>8s} {'总收益':>10s} {'交易次数':>10s} {'止损次数':>10s} {'止损占比':>10s} {'止损盈亏':>12s}")
    print("-" * 80)

    for stop_pct in stop_rates:
        total_return, num_trades, stop_events = diagnose_backtest_kernel(
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
        )
        
        num_stops = len(stop_events)
        stop_ratio = num_stops / num_trades * 100 if num_trades > 0 else 0
        
        # 计算止损时的平均盈亏
        if num_stops > 0:
            avg_stop_pnl = np.mean(stop_events[:, 8]) * 100
        else:
            avg_stop_pnl = 0.0
        
        print(f"{stop_pct*100:7.0f}% {total_return*100:9.2f}% {num_trades:10d} {num_stops:10d} {stop_ratio:9.1f}% {avg_stop_pnl:11.2f}%")
        
        # 详细分析 10% 止损的情况
        if stop_pct == 0.10 and num_stops > 0:
            print("\n" + "=" * 80)
            print(f"10% 止损详细分析 ({num_stops} 次止损)")
            print("=" * 80)
            
            df_stops = pd.DataFrame(
                stop_events,
                columns=['day', 'etf_idx', 'entry_price', 'hwm', 'stop_price', 
                        'exec_price', 'low', 'open', 'return']
            )
            
            # 统计分析
            print(f"\n止损时收益率分布:")
            print(f"  平均: {df_stops['return'].mean()*100:.2f}%")
            print(f"  中位: {df_stops['return'].median()*100:.2f}%")
            print(f"  最大: {df_stops['return'].max()*100:.2f}%")
            print(f"  最小: {df_stops['return'].min()*100:.2f}%")
            print(f"  盈利止损: {(df_stops['return'] > 0).sum()} / {num_stops} ({(df_stops['return'] > 0).mean()*100:.1f}%)")
            print(f"  亏损止损: {(df_stops['return'] < 0).sum()} / {num_stops} ({(df_stops['return'] < 0).mean()*100:.1f}%)")
            
            # HWM 增长分析
            df_stops['hwm_growth'] = (df_stops['hwm'] - df_stops['entry_price']) / df_stops['entry_price'] * 100
            print(f"\nHWM 相对买入价增长:")
            print(f"  平均: {df_stops['hwm_growth'].mean():.2f}%")
            print(f"  中位: {df_stops['hwm_growth'].median():.2f}%")
            print(f"  HWM > 买入价: {(df_stops['hwm_growth'] > 0).sum()} / {num_stops}")
            
            # 前 10 个止损事件
            print(f"\n前 10 个止损事件:")
            print(f"{'日期':>6s} {'ETF':>4s} {'买入价':>8s} {'HWM':>8s} {'止损价':>8s} {'执行价':>8s} {'收益':>8s}")
            print("-" * 70)
            for i in range(min(10, num_stops)):
                day = int(df_stops.iloc[i]['day'])
                etf_idx = int(df_stops.iloc[i]['etf_idx'])
                entry = df_stops.iloc[i]['entry_price']
                hwm = df_stops.iloc[i]['hwm']
                stop = df_stops.iloc[i]['stop_price']
                exec_p = df_stops.iloc[i]['exec_price']
                ret = df_stops.iloc[i]['return'] * 100
                print(f"{day:6d} {etf_idx:4d} {entry:8.3f} {hwm:8.3f} {stop:8.3f} {exec_p:8.3f} {ret:7.2f}%")


if __name__ == "__main__":
    main()
