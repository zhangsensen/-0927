#!/usr/bin/env python3
"""深度诊断：精确定位回测收益计算BUG"""

from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("深度诊断：回测收益计算逻辑")
print("=" * 80)

# 构造最小化测试用例
np.random.seed(42)
n_dates = 10
n_symbols = 3
dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
symbols = ["A", "B", "C"]

# 1. 模拟价格数据（每天上涨1%）
prices_data = np.ones((n_dates, n_symbols)) * 100
for i in range(1, n_dates):
    prices_data[i] = prices_data[i - 1] * 1.01  # 每天固定上涨1%

prices = pd.DataFrame(prices_data, index=dates, columns=symbols)
print("\n[1] 价格矩阵（每天上涨1%）:")
print(prices)

# 2. 模拟因子得分（简单排序）
scores_data = np.array(
    [
        [2.0, 1.0, 0.0],  # Day0: A最高, B其次, C最低
        [1.0, 2.0, 0.0],  # Day1: B最高, A其次, C最低
        [0.0, 2.0, 1.0],  # Day2: B最高, C其次, A最低
        [2.0, 1.0, 0.0],  # Day3: A最高
        [1.0, 2.0, 0.0],  # Day4
        [0.0, 2.0, 1.0],  # Day5
        [2.0, 1.0, 0.0],  # Day6
        [1.0, 2.0, 0.0],  # Day7
        [0.0, 2.0, 1.0],  # Day8
        [2.0, 1.0, 0.0],  # Day9
    ]
)
scores = pd.DataFrame(scores_data, index=dates, columns=symbols)

print("\n[2] 因子得分矩阵:")
print(scores)

# 3. 应用shift(1) - 代码中的关键步骤
scores_shifted = scores.shift(1)
print("\n[3] shift(1)后的得分矩阵:")
print(scores_shifted)
print(f"注意: 第一天全部变为NaN!")

# 4. 构建权重（Top 2）
top_n = 2
ranks = scores_shifted.rank(axis=1, ascending=False, method="first")
selection = ranks <= top_n
weights = selection.astype(float)
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

print("\n[4] 等权重配置 (Top 2):")
print(weights)
print(f"第一天权重全为0: {weights.iloc[0].sum() == 0}")

# 5. 计算收益率
returns = prices.pct_change(fill_method=None).fillna(0.0)
print("\n[5] 日收益率矩阵:")
print(returns)

# 6. 当前代码的逻辑（存在问题）
print("\n" + "=" * 80)
print("[6] 当前代码逻辑分析")
print("=" * 80)

# 当前代码：
# final_weights = weights (调仓日设置权重，其他日向前填充)
# prev_weights[:, 1:, :] = final_weights[:, :-1, :]
# portfolio_returns = np.sum(prev_weights * returns, axis=1)

# 简化版本（无调仓频率）：
prev_weights = weights.shift(1).fillna(0.0)
portfolio_returns_current = (prev_weights * returns).sum(axis=1)

print("当前代码使用: prev_weights = weights.shift(1)")
print("即: T日收益 = T-1日权重 × T日收益率")
print("\nT-1日权重 (prev_weights):")
print(prev_weights)
print("\nT日组合收益:")
print(portfolio_returns_current)
print(f"累计收益: {(1 + portfolio_returns_current).prod() - 1:.2%}")
print(f"前两天收益都为0: {portfolio_returns_current.iloc[:2].sum() == 0}")

# 7. 正确的逻辑
print("\n" + "=" * 80)
print("[7] 正确逻辑分析")
print("=" * 80)

print("正确逻辑应该是:")
print("- T日开盘用T-1日收盘的因子值生成信号")
print("- T日持有T日的头寸，获得T日收益")
print("- 因此: T日收益 = T日权重 × T日收益率")
print("\n但由于shift(1):")
print("- scores_shifted[T] = scores[T-1]")
print("- weights[T]是基于scores_shifted[T]计算的，即基于T-1日因子")
print("- 所以weights[T]已经正确对应T日应持仓")
print("- 应该直接用: portfolio_returns = weights × returns")

# 使用正确逻辑
portfolio_returns_correct = (weights * returns).sum(axis=1)
print("\n正确的T日组合收益:")
print(portfolio_returns_correct)
print(f"累计收益: {(1 + portfolio_returns_correct).prod() - 1:.2%}")

# 8. 对比
print("\n" + "=" * 80)
print("[8] 对比分析")
print("=" * 80)
print(f"当前逻辑累计收益: {(1 + portfolio_returns_current).prod() - 1:.2%}")
print(f"正确逻辑累计收益: {(1 + portfolio_returns_correct).prod() - 1:.2%}")
print(
    f"差异: {((1 + portfolio_returns_correct).prod() - (1 + portfolio_returns_current).prod()):.4f}"
)

print("\n逐日对比:")
comparison = pd.DataFrame(
    {
        "当前逻辑": portfolio_returns_current,
        "正确逻辑": portfolio_returns_correct,
        "差异": portfolio_returns_correct - portfolio_returns_current,
    }
)
print(comparison)

# 9. 结论
print("\n" + "=" * 80)
print("[9] 诊断结论")
print("=" * 80)
print("🔴 严重BUG定位:")
print("   parallel_backtest_configurable.py 第454-460行:")
print("   ```python")
print("   prev_weights = np.zeros_like(final_weights)")
print("   prev_weights[:, 1:, :] = final_weights[:, :-1, :]")
print("   portfolio_returns = np.sum(prev_weights * returns, axis=2)")
print("   ```")
print("")
print("   问题: weights已经通过shift(1)延迟了，不应该再使用prev_weights!")
print("")
print("✅ 修复方案:")
print("   应该直接使用final_weights计算收益:")
print("   ```python")
print("   portfolio_returns = np.sum(final_weights * returns, axis=2)")
print("   ```")
print("")
print("💡 原因分析:")
print("   - scores.shift(1)已经让信号延迟1天（T日信号基于T-1日因子）")
print("   - weights[T]基于scores_shifted[T]，已经是T日应持有的仓位")
print("   - 再使用prev_weights会导致额外延迟1天，丢失第一天收益")
print("   - 这就是为什么收益大幅下降甚至为负的根本原因!")
