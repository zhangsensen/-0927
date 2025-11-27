#!/usr/bin/env python3
"""
全量WFO策略真实回测

对所有12597个策略进行严格T+1真实回测，不依赖WFO排名。
输出：按真实收益排序的完整结果。

用法: uv run python scripts/full_wfo_backtest.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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


def run_single_backtest(args):
    """单策略回测 - 用于并行"""
    rank, combo_name, close_prices, dates, etf_codes, factors_3d, factor_names, timing_arr = args
    
    T, N = close_prices.shape
    combo_factors = [f.strip() for f in combo_name.split(" + ")]
    
    # 获取因子索引
    try:
        factor_indices = [factor_names.index(f) for f in combo_factors]
    except ValueError:
        return None
    
    # 提取该组合的因子
    F_sel = factors_3d[:, :, factor_indices]  # (T, N, len(combo_factors))
    
    # State
    cash = INITIAL_CAPITAL
    holdings = {}
    trades = []
    equity_curve = []
    
    for t in range(LOOKBACK, T):
        current_date = dates[t]
        
        # Mark to Market
        current_value = cash
        for idx, info in holdings.items():
            current_value += info['shares'] * close_prices[t, idx]
        equity_curve.append(current_value)
        
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
                    
                    pnl = (price - info['entry_price']) / info['entry_price']
                    trades.append({
                        'entry_date': info['entry_date'],
                        'exit_date': current_date,
                        'pnl_pct': pnl
                    })
                    del holdings[idx]
            
            # Buy
            current_value = cash + sum(info['shares'] * close_prices[t, idx] for idx, info in holdings.items())
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
    
    # Close all positions
    final_date = dates[-1]
    for idx, info in holdings.items():
        price = close_prices[-1, idx]
        if np.isnan(price):
            price = info['entry_price']
        
        pnl = (price - info['entry_price']) / info['entry_price']
        trades.append({
            'entry_date': info['entry_date'],
            'exit_date': final_date,
            'pnl_pct': pnl
        })
        cash += info['shares'] * price * (1 - COMMISSION_RATE)
    
    # Metrics
    if len(trades) < 10:
        return None
    
    pnl_list = [t['pnl_pct'] for t in trades]
    wins = sum(1 for p in pnl_list if p > 0)
    win_rate = wins / len(pnl_list)
    
    avg_win = np.mean([p for p in pnl_list if p > 0]) if wins > 0 else 0
    avg_loss = abs(np.mean([p for p in pnl_list if p <= 0])) if wins < len(pnl_list) else 0.0001
    profit_factor = avg_win / max(avg_loss, 0.0001) if avg_loss > 0 else 0
    
    total_return = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # Max DD
    equity = np.array(equity_curve)
    if len(equity) > 0:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = np.min(dd)
    else:
        max_dd = 0
    
    return {
        'rank': rank,
        'combo': combo_name,
        'trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_equity': cash
    }


def main():
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("🚀 全量WFO策略真实回测 (修复VORTEX后)")
    logger.info("=" * 80)
    
    # 1. 加载配置
    config_path = Path("configs/combo_wfo_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 加载数据
    logger.info("📊 加载数据...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 3. 计算因子 (使用修复后的因子库)
    logger.info("🔧 计算因子 (修复后的VORTEX)...")
    factor_lib = PreciseFactorLibrary()
    raw_factors = factor_lib.compute_all_factors(ohlcv)
    
    # 4. 横截面标准化
    logger.info("📐 横截面标准化...")
    processor = CrossSectionProcessor()
    std_factors = processor.process_all_factors(raw_factors)
    
    # 5. 准备数据
    factor_names = sorted(std_factors.keys())
    logger.info(f"   因子列表: {factor_names}")
    
    first_factor = std_factors[factor_names[0]]
    T, N = first_factor.shape
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    # 构建3D数组
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"].values
    
    # 市场择时
    timing_module = LightTimingModule()
    timing_arr = timing_module.compute(ohlcv["close"])
    
    # 6. 加载WFO结果
    results_dir = sorted(Path("results").glob("run_*"))[-1]
    all_combos_path = results_dir / "all_combos.parquet"
    
    if not all_combos_path.exists():
        # 尝试pending目录
        results_dir = sorted(Path("results").glob("pending_run_*"))[-1]
        all_combos_path = results_dir / "all_combos.parquet"
    
    logger.info(f"📂 加载WFO结果: {all_combos_path}")
    df_combos = pd.read_parquet(all_combos_path)
    
    total_combos = len(df_combos)
    logger.info(f"   总策略数: {total_combos}")
    
    # 7. 并行回测
    logger.info("⚡ 开始全量回测...")
    
    # 准备参数
    args_list = []
    for idx, row in df_combos.iterrows():
        args_list.append((
            idx,  # rank
            row['combo'],
            close_prices,
            dates,
            etf_codes,
            factors_3d,
            factor_names,
            timing_arr
        ))
    
    # 使用进程池并行
    results = []
    n_workers = max(1, mp.cpu_count() - 2)
    batch_size = 500
    
    logger.info(f"   使用 {n_workers} 进程并行")
    
    for batch_start in range(0, len(args_list), batch_size):
        batch_end = min(batch_start + batch_size, len(args_list))
        batch = args_list[batch_start:batch_end]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(run_single_backtest, arg) for arg in batch]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        progress = (batch_end / len(args_list)) * 100
        logger.info(f"   进度: {batch_end}/{len(args_list)} ({progress:.1f}%) - 有效: {len(results)}")
    
    # 8. 汇总结果
    logger.info("📊 汇总结果...")
    df_results = pd.DataFrame(results)
    
    # 按收益排序
    df_results = df_results.sort_values('total_return', ascending=False)
    
    # 保存
    output_path = Path("results/full_wfo_backtest_results.parquet")
    output_path.parent.mkdir(exist_ok=True)
    df_results.to_parquet(output_path)
    
    csv_path = Path("results/full_wfo_backtest_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    logger.info(f"✅ 结果已保存: {output_path}")
    
    # 9. 打印TOP策略
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏆 TOP 20 策略 (按真实收益排序)")
    logger.info("=" * 80)
    
    print(f"\n{'Rank':>6} | {'WR':>6} | {'PF':>6} | {'Return':>8} | {'MaxDD':>8} | {'Trades':>6} | Combo")
    print("-" * 100)
    
    for i, row in df_results.head(20).iterrows():
        print(f"{row['rank']:>6} | {row['win_rate']*100:>5.1f}% | {row['profit_factor']:>6.2f} | "
              f"{row['total_return']*100:>7.1f}% | {row['max_drawdown']*100:>7.1f}% | "
              f"{row['trades']:>6} | {row['combo'][:50]}")
    
    # 雪球策略筛选 (WR 50-60%)
    logger.info("")
    logger.info("=" * 80)
    logger.info("❄️ 雪球策略 TOP 20 (胜率 50-60%)")
    logger.info("=" * 80)
    
    snowball = df_results[(df_results['win_rate'] >= 0.50) & (df_results['win_rate'] <= 0.60)]
    snowball = snowball.sort_values('total_return', ascending=False)
    
    print(f"\n{'Rank':>6} | {'WR':>6} | {'PF':>6} | {'Return':>8} | {'MaxDD':>8} | {'Trades':>6} | Combo")
    print("-" * 100)
    
    for i, row in snowball.head(20).iterrows():
        print(f"{row['rank']:>6} | {row['win_rate']*100:>5.1f}% | {row['profit_factor']:>6.2f} | "
              f"{row['total_return']*100:>7.1f}% | {row['max_drawdown']*100:>7.1f}% | "
              f"{row['trades']:>6} | {row['combo'][:50]}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n⏱️ 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"📊 有效策略: {len(df_results)}/{total_combos}")


if __name__ == "__main__":
    main()
