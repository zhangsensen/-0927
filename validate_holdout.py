#!/usr/bin/env python3
"""
Holdout验证脚本：测试Top策略在2025-06-01至2025-12-08期间的表现

使用最佳策略：ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).parent

# 导入核心模块
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
    ensure_price_views,
)


# 从batch_vec_backtest.py复制核心函数
def vec_backtest_kernel(
    close_prices: np.ndarray,
    factor_scores: np.ndarray,
    rebalance_dates: np.ndarray,
    pos_size: int,
    initial_capital: float,
    commission_rate: float,
    timing_signals: np.ndarray = None,
) -> tuple:
    """向量化回测核心函数"""
    T, N = close_prices.shape
    n_rebalance = len(rebalance_dates)

    # 初始化
    portfolio_value = np.full(T, initial_capital)
    positions = np.zeros((T, N))
    cash = np.full(T, initial_capital)
    trades = 0

    # 逐个调仓日执行
    for i, rebalance_idx in enumerate(rebalance_dates):
        if rebalance_idx >= T:
            break

        # 获取当前因子得分
        current_scores = factor_scores[rebalance_idx]

        # 选择Top N
        if not np.all(np.isnan(current_scores)):
            valid_mask = ~np.isnan(current_scores)
            valid_scores = current_scores[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            if len(valid_scores) >= pos_size:
                top_indices = valid_indices[np.argsort(valid_scores)[-pos_size:]]
            else:
                top_indices = valid_indices

            # 计算目标权重
            target_weight = 1.0 / len(top_indices) if len(top_indices) > 0 else 0

            # 计算调仓
            current_positions = (
                positions[rebalance_idx - 1] if rebalance_idx > 0 else np.zeros(N)
            )
            target_positions = np.zeros(N)
            target_positions[top_indices] = (
                target_weight
                * portfolio_value[rebalance_idx]
                / close_prices[rebalance_idx, top_indices]
            )

            # 计算交易量和手续费
            trade_volume = (
                np.abs(target_positions - current_positions)
                * close_prices[rebalance_idx]
            )
            commission = np.sum(trade_volume) * commission_rate
            trades += np.sum(target_positions != current_positions)

            # 更新持仓和现金
            positions[rebalance_idx] = target_positions
            cash[rebalance_idx] = (
                cash[rebalance_idx - 1] if rebalance_idx > 0 else initial_capital
            )
            cash[rebalance_idx] -= commission

            # 前向填充持仓
            for t in range(rebalance_idx + 1, min(rebalance_idx + 3, T)):
                positions[t] = positions[rebalance_idx]
                cash[t] = cash[rebalance_idx]

    # 计算每日市值
    for t in range(T):
        if np.any(positions[t] > 0):
            portfolio_value[t] = np.sum(positions[t] * close_prices[t]) + cash[t]
        else:
            portfolio_value[t] = cash[t]

    # 计算收益指标
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    returns = np.concatenate([[0], returns])

    total_return = (portfolio_value[-1] - initial_capital) / initial_capital
    win_rate = np.mean(returns > 0)
    profit_factor = (
        np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0]))
        if np.any(returns < 0)
        else np.inf
    )
    max_drawdown = np.max(
        np.maximum.accumulate(portfolio_value) - portfolio_value
    ) / np.max(portfolio_value)

    return total_return, win_rate, profit_factor, max_drawdown, trades


def main():
    print("🔬 Holdout验证：测试最佳策略在2025-06-01至2025-12-08期间的表现")
    print("=" * 80)

    # 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Holdout期参数
    holdout_start = "2025-06-01"
    holdout_end = "2025-12-08"

    print(f"📅 Holdout期间: {holdout_start} → {holdout_end}")

    # 加载数据 (Holdout期)
    data_loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )

    print("📊 加载Holdout期数据...")
    ohlcv_data = data_loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=holdout_start,
        end_date=holdout_end,
    )
    print(
        f"✅ 数据加载完成: {len(ohlcv_data['close'])} 日期 × {len(config['data']['symbols'])} 只ETF"
    )

    # 计算因子
    print("🔧 计算因子...")
    factor_lib = PreciseFactorLibrary()
    factors = factor_lib.compute_all_factors(ohlcv_data)
    print(f"✅ 因子计算完成: {len(factors)} 个因子")

    # 横截面处理
    print("📐 横截面标准化...")
    processor = CrossSectionProcessor(
        factors=factors,
        bounded_factors=config["cross_section"]["bounded_factors"],
        winsorize_lower=config["cross_section"]["winsorize_lower"],
        winsorize_upper=config["cross_section"]["winsorize_upper"],
    )
    processed_factors = processor.process()
    print("✅ 标准化完成")

    # 最佳策略因子
    target_factors = [
        "ADX_14D",
        "MAX_DD_60D",
        "PRICE_POSITION_120D",
        "PRICE_POSITION_20D",
        "SHARPE_RATIO_20D",
    ]
    print(f"🎯 最佳策略因子: {target_factors}")

    # 检查因子是否存在
    missing_factors = [f for f in target_factors if f not in processed_factors]
    if missing_factors:
        print(f"❌ 缺失因子: {missing_factors}")
        return

    # 组合因子得分 (平均)
    factor_scores = np.mean([processed_factors[f] for f in target_factors], axis=0)
    print(f"✅ 因子得分计算完成: 形状 {factor_scores.shape}")

    # 获取价格数据
    close_prices = ohlcv_data["close"].values  # (T, N)
    print(f"✅ 价格数据: 形状 {close_prices.shape}")

    # 生成调仓日程 (每3天)
    dates = pd.date_range(start=holdout_start, end=holdout_end, freq="D")
    trading_days = dates[dates.weekday < 5]  # 周一到周五
    rebalance_dates = np.arange(0, len(trading_days), 3)  # 每3个交易日
    print(f"✅ 调仓日程: {len(rebalance_dates)} 次调仓")

    # 择时信号 (简化，无择时)
    timing_signals = np.ones(len(trading_days))

    # 回测参数
    pos_size = 2
    initial_capital = 1_000_000
    commission_rate = 0.0002

    print("⚡ 执行向量化回测...")
    total_return, win_rate, profit_factor, max_drawdown, trades = vec_backtest_kernel(
        close_prices=close_prices,
        factor_scores=factor_scores,
        rebalance_dates=rebalance_dates,
        pos_size=pos_size,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        timing_signals=timing_signals,
    )

    # 输出结果
    print("\n" + "=" * 80)
    print("📈 HOLDOUT验证结果")
    print("=" * 80)
    print(".2%")
    print(".2%")
    print(".2f")
    print(".2%")
    print(".2f")
    print(f"交易次数: {trades}")
    print(".2%")

    # 分析原因
    print("\n🔍 表现分析:")
    if total_return < -0.5:
        print("❌ 严重亏损！策略在Holdout期完全失效")
    elif total_return < -0.2:
        print("⚠️ 大幅亏损！因子在新市场环境下预测能力大幅下降")
    elif total_return < 0:
        print("⚠️ 小幅亏损！策略表现不佳，需要调整")
    else:
        print("✅ 表现尚可，但需要与训练集对比")


if __name__ == "__main__":
    main()
