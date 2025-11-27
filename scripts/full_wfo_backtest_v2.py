#!/usr/bin/env python3
"""
全量WFO策略真实回测 v2

对所有12597个策略进行严格T+1真实回测，不依赖WFO排名。
输出：按真实收益排序的完整结果。

用法: uv run python scripts/full_wfo_backtest_v2.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

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


def run_single_backtest(combo_name, close_prices, dates, etf_codes, factors_3d, factor_names, timing_arr):
    """单策略回测"""
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
        
        # Mark to Market (处理 nan 价格)
        current_value = cash
        for idx, info in holdings.items():
            price = close_prices[t, idx]
            if np.isnan(price):
                price = info['entry_price']  # 使用入场价作为备用
            current_value += info['shares'] * price
        equity_curve.append(current_value)
        
        if t % FREQ == 0:
            # Signal from T-1 (严格T+1)
            combined_score = np.nansum(F_sel[t-1], axis=1)
            valid_mask = ~np.isnan(combined_score) & (combined_score != 0)
            
            if np.sum(valid_mask) >= POS_SIZE:
                valid_indices = np.where(valid_mask)[0]
                valid_scores = combined_score[valid_mask]
                # 按得分从高到低排序，确保确定性顺序
                sorted_order = np.argsort(valid_scores)[::-1]  # 降序
                target_list = valid_indices[sorted_order[:POS_SIZE]].tolist()
                target_indices = set(target_list)
            else:
                target_list = []
                target_indices = set()
            
            timing_ratio = timing_arr[t]
            
            # Sell (处理 nan 价格)
            for idx in list(holdings.keys()):
                if idx not in target_indices:
                    info = holdings[idx]
                    price = close_prices[t, idx]
                    if np.isnan(price):
                        price = info['entry_price']  # 使用入场价作为备用
                    proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
                    cash += proceeds
                    
                    pnl = (price - info['entry_price']) / info['entry_price']
                    trades.append({
                        'entry_date': info['entry_date'],
                        'exit_date': current_date,
                        'pnl_pct': pnl
                    })
                    del holdings[idx]
            
            # Buy (按得分从高到低顺序，确保确定性)
            current_value = cash + sum(info['shares'] * close_prices[t, idx] for idx, info in holdings.items())
            target_pos_value = (current_value * timing_ratio) / max(len(target_list), 1)
            
            for idx in target_list:  # 使用有序列表而非set
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
    logger.info("🚀 全量WFO策略真实回测 v2 (修复VORTEX后)")
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
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)  # MultiIndex DataFrame
    
    # 转换为 Dict[str, DataFrame]
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {}
    for fname in factor_names:
        raw_factors[fname] = raw_factors_df[fname]
    
    logger.info(f"   因子数: {len(factor_names)}")
    
    # 4. 横截面标准化
    logger.info("📐 横截面标准化...")
    processor = CrossSectionProcessor(verbose=False)
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
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr = timing_series.reindex(dates).fillna(1.0).values
    
    # 6. 加载WFO结果
    run_dirs = sorted([d for d in Path("results").glob("run_2*") if d.is_dir()])
    if not run_dirs:
        logger.error("找不到WFO结果目录")
        sys.exit(1)
    results_dir = run_dirs[-1]
    all_combos_path = results_dir / "all_combos.parquet"
    
    logger.info(f"📂 加载WFO结果: {all_combos_path}")
    df_combos = pd.read_parquet(all_combos_path)
    
    total_combos = len(df_combos)
    logger.info(f"   总策略数: {total_combos}")
    
    # 7. 顺序回测（避免并行的pickle问题）
    logger.info("⚡ 开始全量回测...")
    
    results = []
    for idx, row in tqdm(df_combos.iterrows(), total=total_combos, desc="回测进度"):
        result = run_single_backtest(
            row['combo'],
            close_prices,
            dates,
            etf_codes,
            factors_3d,
            factor_names,
            timing_arr
        )
        if result is not None:
            result['wfo_rank'] = idx
            results.append(result)
    
    # 8. 汇总结果
    logger.info("📊 汇总结果...")
    df_results = pd.DataFrame(results)
    
    # 按收益排序
    df_results = df_results.sort_values('total_return', ascending=False)
    df_results['real_rank'] = range(1, len(df_results) + 1)
    
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
    logger.info("🏆 TOP 30 策略 (按真实收益排序)")
    logger.info("=" * 80)
    
    print(f"\n{'Real':>4} | {'WFO':>5} | {'WR':>6} | {'PF':>6} | {'Return':>8} | {'MaxDD':>8} | {'Trades':>6} | Combo")
    print("-" * 110)
    
    for _, row in df_results.head(30).iterrows():
        print(f"{row['real_rank']:>4} | {row['wfo_rank']:>5} | {row['win_rate']*100:>5.1f}% | {row['profit_factor']:>6.2f} | "
              f"{row['total_return']*100:>7.1f}% | {row['max_drawdown']*100:>7.1f}% | "
              f"{row['trades']:>6} | {row['combo'][:45]}")
    
    # 雪球策略筛选 (WR 50-60%)
    logger.info("")
    logger.info("=" * 80)
    logger.info("❄️ 雪球策略 TOP 20 (胜率 50-60%, PF > 1.3)")
    logger.info("=" * 80)
    
    snowball = df_results[
        (df_results['win_rate'] >= 0.50) & 
        (df_results['win_rate'] <= 0.60) &
        (df_results['profit_factor'] > 1.3)
    ]
    snowball = snowball.sort_values('total_return', ascending=False)
    
    print(f"\n{'Real':>4} | {'WFO':>5} | {'WR':>6} | {'PF':>6} | {'Return':>8} | {'MaxDD':>8} | {'Trades':>6} | Combo")
    print("-" * 110)
    
    for _, row in snowball.head(20).iterrows():
        print(f"{row['real_rank']:>4} | {row['wfo_rank']:>5} | {row['win_rate']*100:>5.1f}% | {row['profit_factor']:>6.2f} | "
              f"{row['total_return']*100:>7.1f}% | {row['max_drawdown']*100:>7.1f}% | "
              f"{row['trades']:>6} | {row['combo'][:45]}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n⏱️ 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"📊 有效策略: {len(df_results)}/{total_combos}")


if __name__ == "__main__":
    main()
