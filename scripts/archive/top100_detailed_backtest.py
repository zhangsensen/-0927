#!/usr/bin/env python3
"""
TOP 100 策略完整交易回测

生成每个策略的:
1. 完整交易日志 (买入/卖出日期、价格、收益)
2. 每日净值曲线
3. 汇总指标

输出目录: results/top100_detailed/
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
from etf_rotation_optimized.core.market_timing import LightTimingModule

# Constants
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002  # 2 bps
LOOKBACK = 252


def run_detailed_backtest(combo_name, close_prices, dates, etf_codes, factors_3d, factor_names, timing_arr):
    """
    完整回测，返回交易日志和净值曲线
    """
    T, N = close_prices.shape
    combo_factors = [f.strip() for f in combo_name.split(" + ")]
    
    try:
        factor_indices = [factor_names.index(f) for f in combo_factors]
    except ValueError:
        return None, None, None
    
    F_sel = factors_3d[:, :, factor_indices]
    
    # State
    cash = INITIAL_CAPITAL
    holdings = {}
    trade_log = []
    equity_curve = []
    daily_holdings = []
    
    for t in range(LOOKBACK, T):
        current_date = dates[t]
        
        # Mark to Market
        positions_value = sum(info['shares'] * close_prices[t, idx] for idx, info in holdings.items())
        current_value = cash + positions_value
        
        # 记录每日持仓
        holding_str = "; ".join([f"{etf_codes[idx]}:{info['shares']:.0f}股" for idx, info in holdings.items()])
        equity_curve.append({
            'date': current_date,
            'cash': cash,
            'positions_value': positions_value,
            'total_value': current_value,
            'holdings': holding_str
        })
        
        if t % FREQ == 0:
            # Signal from T-1 (严格T+1)
            combined_score = np.nansum(F_sel[t-1], axis=1)
            valid_mask = ~np.isnan(combined_score) & (combined_score != 0)
            
            if np.sum(valid_mask) >= POS_SIZE:
                sorted_indices = np.argsort(combined_score[valid_mask])
                top_k_local = sorted_indices[-POS_SIZE:]
                valid_indices = np.where(valid_mask)[0]
                target_indices = set(valid_indices[top_k_local].tolist())
            else:
                target_indices = set()
            
            timing_ratio = timing_arr[t]
            
            # Sell
            for idx in list(holdings.keys()):
                if idx not in target_indices:
                    info = holdings[idx]
                    price = close_prices[t, idx]
                    proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
                    cash += proceeds
                    
                    pnl_pct = (price - info['entry_price']) / info['entry_price']
                    pnl_amount = (price - info['entry_price']) * info['shares']
                    hold_days = (current_date - info['entry_date']).days
                    
                    trade_log.append({
                        'action': 'SELL',
                        'ticker': etf_codes[idx],
                        'date': current_date,
                        'price': price,
                        'shares': info['shares'],
                        'value': proceeds,
                        'entry_date': info['entry_date'],
                        'entry_price': info['entry_price'],
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'hold_days': hold_days,
                        'reason': 'Rebalance'
                    })
                    del holdings[idx]
            
            # Buy
            current_value = cash + sum(info['shares'] * close_prices[t, idx] for idx, info in holdings.items())
            n_to_buy = len([i for i in target_indices if i not in holdings])
            if n_to_buy > 0:
                target_pos_value = (current_value * timing_ratio) / max(len(target_indices), 1)
            
            for idx in target_indices:
                if idx in holdings:
                    continue
                price = close_prices[t, idx]
                if np.isnan(price) or price <= 0:
                    continue
                
                shares = target_pos_value / price
                cost = shares * price * (1 + COMMISSION_RATE)
                
                if cash >= cost:
                    cash -= cost
                    holdings[idx] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': current_date
                    }
                    trade_log.append({
                        'action': 'BUY',
                        'ticker': etf_codes[idx],
                        'date': current_date,
                        'price': price,
                        'shares': shares,
                        'value': cost,
                        'entry_date': current_date,
                        'entry_price': price,
                        'pnl_pct': 0,
                        'pnl_amount': 0,
                        'hold_days': 0,
                        'reason': 'Signal'
                    })
    
    # Close all at end
    final_date = dates[-1]
    for idx, info in list(holdings.items()):
        price = close_prices[-1, idx]
        if np.isnan(price):
            price = info['entry_price']
        
        proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
        cash += proceeds
        
        pnl_pct = (price - info['entry_price']) / info['entry_price']
        pnl_amount = (price - info['entry_price']) * info['shares']
        hold_days = (final_date - info['entry_date']).days
        
        trade_log.append({
            'action': 'SELL',
            'ticker': etf_codes[idx],
            'date': final_date,
            'price': price,
            'shares': info['shares'],
            'value': proceeds,
            'entry_date': info['entry_date'],
            'entry_price': info['entry_price'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'hold_days': hold_days,
            'reason': 'End'
        })
    
    # Compute metrics
    df_trades = pd.DataFrame(trade_log)
    df_equity = pd.DataFrame(equity_curve)
    
    if len(df_trades) == 0:
        return None, None, None
    
    # 只统计SELL交易的收益
    sells = df_trades[df_trades['action'] == 'SELL']
    if len(sells) < 5:
        return None, None, None
    
    wins = (sells['pnl_pct'] > 0).sum()
    total_trades = len(sells)
    win_rate = wins / total_trades
    
    avg_win = sells[sells['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = abs(sells[sells['pnl_pct'] <= 0]['pnl_pct'].mean()) if wins < total_trades else 0.0001
    profit_factor = avg_win / max(avg_loss, 0.0001)
    
    total_return = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Max DD
    if len(df_equity) > 0:
        equity = df_equity['total_value'].values
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = np.min(dd)
    else:
        max_dd = 0
    
    metrics = {
        'combo': combo_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_equity': cash,
        'avg_hold_days': sells['hold_days'].mean(),
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100
    }
    
    return metrics, df_trades, df_equity


def main():
    start_time = datetime.now()
    print("=" * 80)
    print("🎯 TOP 100 策略完整交易回测")
    print("=" * 80)
    
    # 1. 加载配置
    config_path = Path("configs/combo_wfo_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 加载数据
    print("📊 加载数据...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 3. 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 4. 横截面标准化
    print("📐 横截面标准化...")
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # 5. 准备数据
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    T, N = first_factor.shape
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"].values
    
    # 市场择时
    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = timing_series.reindex(dates).fillna(1.0).values
    
    # 6. 加载TOP 100
    df_full = pd.read_parquet("results/full_wfo_backtest_results.parquet")
    top100 = df_full.head(100)
    
    print(f"📂 回测 TOP {len(top100)} 策略...")
    
    # 7. 创建输出目录
    output_dir = Path("results/top100_detailed")
    output_dir.mkdir(exist_ok=True)
    
    # 8. 批量回测
    all_metrics = []
    
    for i, row in tqdm(top100.iterrows(), total=len(top100), desc="详细回测"):
        combo_name = row['combo']
        rank = row['real_rank']
        
        metrics, df_trades, df_equity = run_detailed_backtest(
            combo_name, close_prices, dates, etf_codes,
            factors_3d, factor_names, timing_arr
        )
        
        if metrics is None:
            continue
        
        metrics['real_rank'] = rank
        all_metrics.append(metrics)
        
        # 保存交易日志
        safe_name = combo_name.replace(" + ", "_").replace(" ", "")[:50]
        trades_path = output_dir / f"rank{int(rank):03d}_{safe_name}_trades.csv"
        df_trades.to_csv(trades_path, index=False)
        
        # 保存净值曲线
        equity_path = output_dir / f"rank{int(rank):03d}_{safe_name}_equity.csv"
        df_equity.to_csv(equity_path, index=False)
    
    # 9. 汇总
    df_summary = pd.DataFrame(all_metrics)
    df_summary = df_summary.sort_values('total_return', ascending=False)
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    
    # 10. 打印结果
    print()
    print("=" * 100)
    print("🏆 TOP 20 策略详细回测结果")
    print("=" * 100)
    print(f"{'Rank':>4} | {'WR':>6} | {'PF':>5} | {'Return':>8} | {'MaxDD':>8} | {'Trades':>6} | {'AvgHold':>7} | Combo")
    print("-" * 100)
    
    for _, row in df_summary.head(20).iterrows():
        dd_str = f"{row['max_drawdown']*100:.1f}%" if pd.notna(row['max_drawdown']) else "N/A"
        print(f"{int(row['real_rank']):>4} | {row['win_rate']*100:>5.1f}% | {row['profit_factor']:>5.2f} | "
              f"{row['total_return']*100:>7.1f}% | {dd_str:>8} | {int(row['total_trades']):>6} | "
              f"{row['avg_hold_days']:>6.1f}d | {row['combo'][:40]}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print()
    print(f"⏱️ 耗时: {elapsed:.1f}秒")
    print(f"📁 输出目录: {output_dir}")
    print(f"   - summary.csv: 汇总指标")
    print(f"   - rank*_trades.csv: 交易日志")
    print(f"   - rank*_equity.csv: 净值曲线")


if __name__ == "__main__":
    main()
