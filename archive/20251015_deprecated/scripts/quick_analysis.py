#!/usr/bin/env python3
"""
快速分析8月极端收益来源
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 读取回测结果
extended_perf = pd.read_csv("rotation_output/backtest/performance_metrics_extended.csv")
core_perf = pd.read_csv("rotation_output/backtest/performance_metrics.csv")
extended_summary = pd.read_csv("rotation_output/backtest/backtest_summary_extended.csv")
core_summary = pd.read_csv("rotation_output/backtest/backtest_summary.csv")

print("=== 扩展因子系统 vs 核心系统绩效对比 ===")
print(f"扩展系统年化收益: {extended_perf['annual_return'].iloc[0]:.2%}")
print(f"核心系统年化收益: {core_perf['annual_return'].iloc[0]:.2%}")
print(
    f"收益差异: {extended_perf['annual_return'].iloc[0] - core_perf['annual_return'].iloc[0]:.2%}"
)
print()
print(f"扩展系统波动率: {extended_perf['volatility'].iloc[0]:.2%}")
print(f"核心系统波动率: {core_perf['volatility'].iloc[0]:.2%}")
print(
    f"波动率差异: {extended_perf['volatility'].iloc[0] - core_perf['volatility'].iloc[0]:.2%}"
)
print()
print(f"扩展系统夏普比率: {extended_perf['sharpe'].iloc[0]:.2f}")
print(f"核心系统夏普比率: {core_perf['sharpe'].iloc[0]:.2f}")
print()
print(f"扩展系统最大回撤: {extended_perf['max_drawdown'].iloc[0]:.2%}")
print(f"核心系统最大回撤: {core_perf['max_drawdown'].iloc[0]:.2%}")

print("\n=== 逐月组合规模对比 ===")
print("日期\t\t扩展宇宙\t扩展评分\t扩展组合\t核心宇宙\t核心评分\t核心组合")
for i, row in extended_summary.iterrows():
    date = row["trade_date"]
    core_row = core_summary[core_summary["trade_date"] == date]
    if not core_row.empty:
        core_data = core_row.iloc[0]
        print(
            f"{date}\t{row['universe_size']}\t{row['scored_size']}\t{row['portfolio_size']}\t{core_data['universe_size']}\t{core_data['scored_size']}\t{core_data['portfolio_size']}"
        )

# 重点关注8月份
august_ext = extended_summary[extended_summary["trade_date"] == 20240830]
august_core = core_summary[core_summary["trade_date"] == 20240830]

print(f"\n=== 8月份详细对比 ===")
if not august_ext.empty:
    ext_row = august_ext.iloc[0]
    print(
        f"扩展系统8月: 宇宙{ext_row['universe_size']}, 评分{ext_row['scored_size']}, 组合{ext_row['portfolio_size']}"
    )

if not august_core.empty:
    core_row = august_core.iloc[0]
    print(
        f"核心系统8月: 宇宙{core_row['universe_size']}, 评分{core_row['scored_size']}, 组合{core_row['portfolio_size']}"
    )

# 计算风险指标
ext_return = extended_perf["annual_return"].iloc[0]
ext_vol = extended_perf["volatility"].iloc[0]
core_return = core_perf["annual_return"].iloc[0]
core_vol = core_perf["volatility"].iloc[0]

print(f"\n=== 风险分析 ===")
print(f"收益提升比例: {(ext_return - core_return) / core_return:.1f}倍")
print(f"波动增加比例: {(ext_vol - core_vol) / core_vol:.1f}倍")
print(
    f"风险收益比: 每增加1%收益需要承担{((ext_vol - core_vol) / (ext_return - core_return)):.1f}%的额外波动"
)

# 检查是否有数据质量问题
print(f"\n=== 数据质量检查 ===")
print(f"扩展系统平均换手: {extended_perf['avg_turnover'].iloc[0]:.1%}")
print(f"核心系统平均换手: {core_perf['avg_turnover'].iloc[0]:.1%}")
print(f"扩展系统平均成本: {extended_perf['avg_cost'].iloc[0]:.2%}")
print(f"核心系统平均成本: {core_perf['avg_cost'].iloc[0]:.2%}")

# 年化成本
ext_annual_cost = extended_perf["avg_cost"].iloc[0] * 12
core_annual_cost = core_perf["avg_cost"].iloc[0] * 12
print(f"扩展系统年化成本: {ext_annual_cost:.2%}")
print(f"核心系统年化成本: {core_annual_cost:.2%}")

# 净收益对比
ext_net_return = ext_return - ext_annual_cost
core_net_return = core_return - core_annual_cost
print(f"扩展系统净收益: {ext_net_return:.2%}")
print(f"核心系统净收益: {core_net_return:.2%}")
print(f"净收益差异: {ext_net_return - core_net_return:.2%}")

print(f"\n=== 关键结论 ===")
if ext_vol / core_vol > 3:
    print("🚨 极高风险: 扩展系统波动率是核心系统的3倍以上")
elif ext_vol / core_vol > 2:
    print("⚠️ 高风险: 扩展系统波动率是核心系统的2倍以上")
elif ext_vol / core_vol > 1.5:
    print("⚡ 中等风险: 扩展系统波动率显著高于核心系统")
else:
    print("✅ 低风险: 扩展系统波动率与核心系统相近")

if extended_perf["sharpe"].iloc[0] < core_perf["sharpe"].iloc[0]:
    print("🚨 警告: 扩展系统风险调整后收益低于核心系统")
else:
    print("✅ 扩展系统风险调整后收益优于核心系统")

if ext_net_return < core_net_return:
    print("🚨 严重: 扩展系统扣除成本后的净收益低于核心系统")
else:
    print("✅ 扩展系统扣除成本后净收益优于核心系统")
