#!/usr/bin/env python3
"""
统一规则 WFO 优化

核心原则：一套规则，从筛选到验证
- 筛选标准 = 验证标准 = 真实回测收益
- 无 IC，无中间层，无歧义

用法: uv run python run_unified_wfo.py
"""

import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from numba import njit

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 核心参数（硬编码，无魔数）
# ============================================================================
FREQ = 8              # 换仓频率（天）
POS_SIZE = 3          # 持仓数量
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002  # 2 bps
LOOKBACK = 252        # 回测起点（跳过前252天热身）


@njit(cache=True)
def _backtest_combo_numba(
    close_prices: np.ndarray,
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    timing_arr: np.ndarray,
    freq: int,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    lookback: int,
) -> tuple:
    """
    Numba 加速的单策略回测
    
    返回: (total_return, win_rate, profit_factor, num_trades, max_dd)
    """
    T, N = close_prices.shape
    n_factors = len(factor_indices)
    
    # 状态
    cash = initial_capital
    holdings = np.full(N, -1.0)  # -1 表示未持有
    entry_prices = np.zeros(N)
    
    # 统计
    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    
    # 净值曲线
    equity_curve = np.zeros(T - lookback)
    
    for t in range(lookback, T):
        # Mark to Market
        current_value = cash
        for n in range(N):
            if holdings[n] > 0:
                current_value += holdings[n] * close_prices[t, n]
        equity_curve[t - lookback] = current_value
        
        if t % freq == 0:
            # 信号来自 T-1（严格 T+1）
            # 使用 nansum，与旧回测一致
            combined_score = np.zeros(N)
            for n in range(N):
                score = 0.0
                for i in range(n_factors):
                    f_idx = factor_indices[i]
                    val = factors_3d[t-1, n, f_idx]
                    if not np.isnan(val):
                        score += val
                combined_score[n] = score
            
            # 有效性检查：得分非零且非NaN
            valid_count = 0
            for n in range(N):
                if combined_score[n] != 0 and not np.isnan(combined_score[n]):
                    valid_count += 1
                else:
                    combined_score[n] = -np.inf
            
            # 选 Top K
            target_set = np.zeros(N, dtype=np.bool_)
            buy_order = np.zeros(pos_size, dtype=np.int64)  # 按得分从高到低的买入顺序
            buy_count = 0
            if valid_count >= pos_size:
                sorted_indices = np.argsort(combined_score)
                for k in range(pos_size):
                    idx = sorted_indices[-(k+1)]
                    if combined_score[idx] > -np.inf:
                        target_set[idx] = True
                        buy_order[buy_count] = idx
                        buy_count += 1
            
            timing_ratio = timing_arr[t]
            
            # 卖出
            for n in range(N):
                if holdings[n] > 0 and not target_set[n]:
                    price = close_prices[t, n]
                    proceeds = holdings[n] * price * (1 - commission_rate)
                    cash += proceeds
                    
                    pnl = (price - entry_prices[n]) / entry_prices[n]
                    if pnl > 0:
                        wins += 1
                        total_win_pnl += pnl
                    else:
                        losses += 1
                        total_loss_pnl += abs(pnl)
                    
                    holdings[n] = -1.0
                    entry_prices[n] = 0.0
            
            # 买入
            current_value = cash
            for n in range(N):
                if holdings[n] > 0:
                    current_value += holdings[n] * close_prices[t, n]
            
            target_count = 0
            for n in range(N):
                if target_set[n]:
                    target_count += 1
            
            if target_count > 0:
                target_pos_value = (current_value * timing_ratio) / target_count
                
                # 按得分从高到低买入（确定性顺序）
                for k in range(buy_count):
                    n = buy_order[k]
                    if holdings[n] < 0:  # 未持有
                        price = close_prices[t, n]
                        if np.isnan(price) or price <= 0:
                            continue
                        
                        shares = target_pos_value / price
                        cost = shares * price * (1 + commission_rate)
                        
                        if cash >= cost:
                            cash -= cost
                            holdings[n] = shares
                            entry_prices[n] = price
    
    # 清仓
    final_value = cash
    for n in range(N):
        if holdings[n] > 0:
            price = close_prices[-1, n]
            if np.isnan(price):
                price = entry_prices[n]
            
            final_value += holdings[n] * price * (1 - commission_rate)
            
            pnl = (price - entry_prices[n]) / entry_prices[n]
            if pnl > 0:
                wins += 1
                total_win_pnl += pnl
            else:
                losses += 1
                total_loss_pnl += abs(pnl)
    
    # 计算指标
    num_trades = wins + losses
    total_return = (final_value - initial_capital) / initial_capital
    
    if num_trades > 0:
        win_rate = wins / num_trades
    else:
        win_rate = 0.0
    
    if losses > 0:
        avg_win = total_win_pnl / max(wins, 1)
        avg_loss = total_loss_pnl / losses
        profit_factor = avg_win / max(avg_loss, 0.0001)
    else:
        profit_factor = 0.0
    
    # 最大回撤
    max_dd = 0.0
    peak = equity_curve[0]
    for i in range(len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        dd = (equity_curve[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    return total_return, win_rate, profit_factor, num_trades, max_dd


def run_unified_wfo():
    """主函数"""
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("🎯 统一规则 WFO (Unified Rule WFO)")
    logger.info("=" * 80)
    logger.info("核心原则: 筛选标准 = 验证标准 = 真实回测收益")
    logger.info("")
    
    # 1. 加载配置
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "configs/combo_wfo_config.yaml"
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
    
    # 3. 计算因子
    logger.info("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 4. 横截面标准化
    logger.info("📐 横截面标准化...")
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
    
    logger.info(f"   数据: {T}天 × {N}只ETF × {len(factor_names)}个因子")
    
    # 6. 生成所有组合
    combo_sizes = config["combo_wfo"]["combo_sizes"]
    all_combos = []
    for size in combo_sizes:
        combos = list(combinations(range(len(factor_names)), size))
        all_combos.extend(combos)
        logger.info(f"   {size}-因子组合: {len(combos)}")
    logger.info(f"   总计: {len(all_combos)} 个组合")
    
    # 7. 回测所有组合（单线程顺序，避免并行开销）
    logger.info("")
    logger.info("⚡ 回测所有组合 (统一规则: 真实收益)")
    logger.info("-" * 80)
    
    results = []
    for combo_indices in tqdm(all_combos, desc="回测进度", ncols=80):
        factor_idx_arr = np.array(combo_indices, dtype=np.int64)
        
        ret, wr, pf, trades, dd = _backtest_combo_numba(
            close_prices,
            factors_3d,
            factor_idx_arr,
            timing_arr,
            FREQ,
            POS_SIZE,
            INITIAL_CAPITAL,
            COMMISSION_RATE,
            LOOKBACK,
        )
        
        combo_str = " + ".join([factor_names[i] for i in combo_indices])
        
        if trades >= 10:  # 最少 10 笔交易
            results.append({
                "combo": combo_str,
                "combo_size": len(combo_indices),
                "total_return": ret,
                "win_rate": wr,
                "profit_factor": pf,
                "trades": trades,
                "max_drawdown": dd,
            })
    
    # 8. 排序（唯一标准：收益）
    df = pd.DataFrame(results)
    df = df.sort_values("total_return", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # 9. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir.parent / "results" / f"unified_wfo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_dir / "all_combos.parquet", index=False)
    df.head(100).to_parquet(output_dir / "top100.parquet", index=False)
    df.to_csv(output_dir / "all_combos.csv", index=False)
    
    # 保存因子
    factors_dir = output_dir / "factors"
    factors_dir.mkdir(exist_ok=True)
    for fname in factor_names:
        std_factors[fname].to_parquet(factors_dir / f"{fname}.parquet")
    
    # 保存配置
    run_config = {
        "timestamp": timestamp,
        "rule": "UNIFIED (Return-based)",
        "parameters": {
            "freq": FREQ,
            "pos_size": POS_SIZE,
            "commission_rate": COMMISSION_RATE,
            "lookback": LOOKBACK,
        },
        "data": config["data"],
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    # 10. 输出结果
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ 完成 | 耗时: {elapsed:.1f}秒 | 有效策略: {len(df)}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🏆 TOP 20 策略 (唯一标准: 真实回测收益)")
    logger.info("-" * 80)
    print(f"{'Rank':>4} | {'Return':>8} | {'WR':>6} | {'PF':>6} | {'MaxDD':>8} | {'Trades':>6} | Combo")
    print("-" * 100)
    
    for _, row in df.head(20).iterrows():
        print(f"{row['rank']:>4} | {row['total_return']*100:>7.1f}% | "
              f"{row['win_rate']*100:>5.1f}% | {row['profit_factor']:>6.2f} | "
              f"{row['max_drawdown']*100:>7.1f}% | {row['trades']:>6} | "
              f"{row['combo'][:45]}")
    
    logger.info("")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info("")
    logger.info("💡 一套规则：筛选 = 验证 = 真实回测收益")
    
    return df, output_dir


if __name__ == "__main__":
    run_unified_wfo()
