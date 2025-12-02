#!/usr/bin/env python3
"""
全量 RORO (Risk-On/Risk-Off) 策略重评估脚本
遍历所有 12,597 个 WFO 候选组合，在 RORO 逻辑下重新回测并排名。
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

# ==================== 配置参数 ====================
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

# RORO 风控参数 (与 strategy_config.yaml 保持一致)
STOP_LOSS_PCT = 0.12   # 12% 止损
DD_LIMIT_SOFT = 0.15   # 15% 回撤 -> 减半
DD_LIMIT_HARD = 0.25   # 25% 回撤 -> 清仓

# ==================== RORO 回测内核 ====================
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
    ro_prices, # [T, M] Risk-Off Asset Prices
    ro_mom,    # [T, M] Risk-Off Asset Momentum
):
    T, N, _ = factors_3d.shape
    M = ro_prices.shape[1]

    cash = initial_capital
    
    # 权益持仓
    eq_holdings = np.full(N, -1.0)
    eq_entry_prices = np.zeros(N)
    
    # 避险持仓 (简化为只持有一个最好的)
    ro_holding_idx = -1 
    ro_holding_shares = 0.0
    
    # 峰值记录 (用于回撤控制)
    peak_value = initial_capital
    max_drawdown = 0.0
    
    wins = 0
    losses = 0
    
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

        # 1. 计算当前总资产
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
        
        # 2. 确定风控系数
        risk_scalar = 1.0
        if drawdown > dd_limit_hard:
            risk_scalar = 0.0
        elif drawdown > dd_limit_soft:
            risk_scalar = 0.5
            
        # 3. 权益选股打分
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
        if valid >= pos_size and risk_scalar > 0.0:
            sorted_idx = np.argsort(combined_score)
            for k in range(pos_size):
                idx = sorted_idx[N - 1 - k]
                if combined_score[idx] == -np.inf:
                    break
                target_set[idx] = True
                buy_order[buy_count] = idx
                buy_count += 1

        # 4. 目标仓位计算
        # 权益目标市值 = 总资产 * 择时信号 * 风控系数
        target_eq_ratio = timing_arr[t] * risk_scalar
        target_eq_capital = current_total_value * target_eq_ratio
        
        # 5. 卖出逻辑 (权益)
        for n in range(N):
            should_sell = (eq_holdings[n] > 0.0) and (not target_set[n] or risk_scalar == 0.0)
            if should_sell:
                price = close_prices[t, n]
                proceeds = eq_holdings[n] * price * (1.0 - commission_rate)
                cash += proceeds
                
                pnl = (price - eq_entry_prices[n]) / eq_entry_prices[n]
                if pnl > 0.0: wins += 1
                else: losses += 1
                
                eq_holdings[n] = -1.0
                eq_entry_prices[n] = 0.0

        # 6. 买入逻辑 (权益)
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

        # 7. 避险资产逻辑 (Risk-Off)
        # 剩余现金投资于动量最好的避险资产
        best_ro_idx = -1
        best_ro_mom = -999.0
        
        for m in range(M):
            mom = ro_mom[t, m]
            if not np.isnan(mom) and mom > 0.0: # 只买上涨趋势的
                if mom > best_ro_mom:
                    best_ro_mom = mom
                    best_ro_idx = m
        
        # 换仓避险资产
        if ro_holding_idx != -1 and ro_holding_idx != best_ro_idx:
            price = ro_prices[t, ro_holding_idx]
            proceeds = ro_holding_shares * price * (1.0 - commission_rate)
            cash += proceeds
            ro_holding_idx = -1
            ro_holding_shares = 0.0
            
        # 买入避险资产
        if best_ro_idx != -1 and ro_holding_idx == -1:
            invest_amt = cash * 0.99 # 留 1% 缓冲
            if invest_amt > 0:
                price = ro_prices[t, best_ro_idx]
                if not np.isnan(price) and price > 0:
                    shares = invest_amt / price
                    cost = shares * price * (1.0 + commission_rate)
                    if cash >= cost:
                        cash -= cost
                        ro_holding_idx = best_ro_idx
                        ro_holding_shares = shares
        
        # 补仓避险资产 (如果有闲置现金)
        if best_ro_idx != -1 and ro_holding_idx == best_ro_idx:
             if cash > 10000: 
                 invest_amt = cash * 0.99
                 price = ro_prices[t, best_ro_idx]
                 shares = invest_amt / price
                 cost = shares * price * (1.0 + commission_rate)
                 if cash >= cost:
                     cash -= cost
                     ro_holding_shares += shares

        # 8. 日内止损 (仅权益)
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
                        proceeds = eq_holdings[n] * stop_price * (1.0 - commission_rate)
                        cash += proceeds
                        pnl = (stop_price - entry) / entry
                        losses += 1
                        eq_holdings[n] = -1.0
                        eq_entry_prices[n] = 0.0

    # 最终清算
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

def run_vec_roro(factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices, ro_prices, ro_mom):
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
        ro_prices, ro_mom
    )

def main():
    print("=" * 80)
    print("🚀 全量 RORO 策略重评估 (Re-evaluating All Combos)")
    print(f"   逻辑: 权益策略 + 黄金/国债避险")
    print(f"   目标: 在 12,000+ 个组合中寻找最适合 RORO 模式的策略")
    print("=" * 80)

    # 1. 加载所有组合
    input_path = ROOT / "results/vec_full_backtest_20251128_185610/vec_all_combos.csv"
    if not input_path.exists():
        print(f"❌ 未找到输入文件: {input_path}")
        return
    df_all = pd.read_csv(input_path)
    print(f"✅ 加载组合: {len(df_all):,} 个")

    # 2. 加载数据
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    loader = DataLoader(data_dir=config["data"].get("data_dir"), cache_dir=config["data"].get("cache_dir"))
    ohlcv = loader.load_ohlcv(etf_codes=config["data"]["symbols"], start_date=config["data"]["start_date"], end_date=config["data"]["end_date"])

    # 3. 准备因子和价格
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

    # 4. 准备避险资产数据
    ro_symbols = ['518880', '511010'] # 黄金, 国债
    valid_ro_indices = []
    for sym in ro_symbols:
        if sym in etf_codes: valid_ro_indices.append(etf_codes.index(sym))
    
    if not valid_ro_indices:
        print("❌ 未找到避险资产数据")
        return

    ro_prices = close_prices[:, valid_ro_indices]
    ro_mom = np.zeros_like(ro_prices)
    ro_mom[20:] = ro_prices[20:] / ro_prices[:-20] - 1.0
    ro_mom[:20] = 0.0

    # 5. 全量回测
    results = []
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    
    # 预处理组合索引以加速
    combo_tasks = []
    for combo_str in df_all['combo']:
        indices = [factor_index_map[f.strip()] for f in combo_str.split(" + ")]
        combo_tasks.append((combo_str, indices))

    print(f"🚀 开始全量扫描...")
    for combo_str, factor_indices in tqdm(combo_tasks, total=len(combo_tasks), desc="RORO Scan"):
        ret, mdd = run_vec_roro(
            factors_3d, close_prices, open_prices, low_prices, timing_arr, factor_indices,
            ro_prices, ro_mom
        )
        results.append({
            "combo": combo_str,
            "roro_return": ret,
            "roro_mdd": mdd,
            "roro_calmar": ret / abs(mdd) if mdd != 0 else 0
        })

    # 6. 结果分析与保存
    df_res = pd.DataFrame(results)
    
    # 合并原始 VEC 结果以便对比
    df_merged = df_all.merge(df_res, on="combo")
    
    # 计算综合得分 (偏好高收益低回撤)
    # Score = 0.6 * Z(Return) - 0.4 * Z(MaxDD)
    df_merged['z_ret'] = (df_merged['roro_return'] - df_merged['roro_return'].mean()) / df_merged['roro_return'].std()
    df_merged['z_mdd'] = (df_merged['roro_mdd'] - df_merged['roro_mdd'].mean()) / df_merged['roro_mdd'].std()
    df_merged['roro_score'] = 0.6 * df_merged['z_ret'] - 0.4 * df_merged['z_mdd']
    
    output_path = ROOT / "results/roro_full_rankings.csv"
    df_merged.to_csv(output_path, index=False)
    
    print(f"\n✅ 全量扫描完成")
    print(f"   输出: {output_path}")
    
    print(f"\n🏆 RORO 模式下 Top 10 策略")
    print("-" * 100)
    print(f"{'Rank':<4} | {'Score':<6} | {'Return':<8} | {'MaxDD':<8} | {'Calmar':<6} | Combo")
    print("-" * 100)
    
    top10 = df_merged.nlargest(10, 'roro_score')
    for i, row in enumerate(top10.itertuples(), 1):
        print(f"{i:<4} | {row.roro_score:6.2f} | {row.roro_return*100:7.1f}% | {row.roro_mdd*100:7.1f}% | {row.roro_calmar:6.2f} | {row.combo[:60]}...")

if __name__ == "__main__":
    main()
