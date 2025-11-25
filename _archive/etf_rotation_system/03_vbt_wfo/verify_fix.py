#!/usr/bin/env python3
"""验证BUG修复效果"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from config_loader_parallel import load_fast_config_from_args
from parallel_backtest_configurable import ConfigurableParallelBacktestEngine

print("=" * 80)
print("验证BUG修复效果 - 快速测试")
print("=" * 80)

# 加载配置
config_path = Path(__file__).parent / "simple_config.yaml"
config = load_fast_config_from_args(["-c", str(config_path)])

# 创建引擎
engine = ConfigurableParallelBacktestEngine(config)

# 加载数据（只测试IS的第一个Period）
print("\n[1] 加载测试数据...")
panel = engine._load_factor_panel()
prices = engine._load_price_data()
factors = engine._load_top_factors()

# 切分IS Period 1的数据
is_start = pd.Timestamp("2020-01-02")
is_end = pd.Timestamp("2021-01-02")

panel_dates = panel.index.get_level_values(1)
price_dates = prices.index

is_panel_mask = (panel_dates >= is_start) & (panel_dates <= is_end)
is_price_mask = (price_dates >= is_start) & (price_dates <= is_end)
is_panel = panel.loc[is_panel_mask]
is_prices = prices.loc[is_price_mask]

print(f"IS数据: Panel{is_panel.shape}, Prices{is_prices.shape}")
print(f"因子: {factors}")

# 运行小规模回测
print("\n[2] 运行快速回测 (100组合 × 1个Top-N × 1个调仓周期)...")
from dataclasses import replace

config_small = replace(
    config,
    factors=factors,
    top_n_list=[5],
    rebalance_freq_list=[10],
    weight_grid_points=[0.0, 0.5, 1.0],  # 缩小网格
    max_combinations=100,
    n_workers=2,
)

engine_small = ConfigurableParallelBacktestEngine(config_small)
results = engine_small.parallel_grid_search(
    panel=is_panel, prices=is_prices, factors=factors
)

print(f"\n[3] 回测结果分析 ({len(results)}个策略)...")
print("\n收益统计:")
print(results["total_return"].describe())

print("\nSharpe统计:")
print(results["sharpe_ratio"].describe())

print("\n最优策略:")
best = results.nlargest(1, "sharpe_ratio").iloc[0]
print(f"  Sharpe: {best['sharpe_ratio']:.3f}")
print(f"  收益率: {best['total_return']:.2f}%")
print(f"  回撤: {best['max_drawdown']:.2f}%")
print(f"  换手率: {best['turnover']:.2f}")

print("\n有效策略统计:")
positive_return = (results["total_return"] > 0).sum()
positive_sharpe = (results["sharpe_ratio"] > 0).sum()
print(
    f"  正收益策略: {positive_return}/{len(results)} ({positive_return/len(results)*100:.1f}%)"
)
print(
    f"  正Sharpe策略: {positive_sharpe}/{len(results)} ({positive_sharpe/len(results)*100:.1f}%)"
)

print("\n" + "=" * 80)
if positive_return > len(results) * 0.5 and best["total_return"] > 5:
    print("✅ 修复验证通过！收益率显著提升")
else:
    print("⚠️ 修复效果有限，可能存在其他问题")
print("=" * 80)
