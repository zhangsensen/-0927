#!/usr/bin/env python3
"""手动验证单组合 - 老引擎 vs 新引擎"""
import glob

import numpy as np
import pandas as pd

# === 加载数据 ===
panel = pd.read_parquet("etf_rotation_system/data/panels/panel_20251018_005220.parquet")

# 加载价格
prices = []
for f in sorted(glob.glob("raw/ETF/daily/*.parquet")):
    df = pd.read_parquet(f)
    symbol = f.split("/")[-1].split("_")[0]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["trade_date"])
    prices.append(df[["date", "close", "symbol"]])

price_df = pd.concat(prices, ignore_index=True)
pivot = (
    price_df.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
)

print("数据加载完成")
print(f"面板: {panel.shape}")
print(f"价格: {pivot.shape}")
print(f"日期: {pivot.index.min()} ~ {pivot.index.max()}")

# === 测试组合: RSI_14=1.0, Top-N=8 ===
print("\n" + "=" * 80)
print("测试组合: RSI_14=1.0, Top-N=8, 每20日调仓")
print("=" * 80)

# 提取RSI_14因子
rsi14 = panel["RSI_14"].unstack(level="symbol")

# 对齐日期
common_dates = pivot.index.intersection(rsi14.index)
prices_aligned = pivot.loc[common_dates]
scores_aligned = rsi14.loc[common_dates]

print(f"\n对齐后数据: {len(common_dates)} 天")

# === 新引擎逻辑 ===
print("\n" + "-" * 80)
print("新引擎计算")
print("-" * 80)

top_n = 8
rebalance_freq = 20
fees = 0.001

# 构建权重（每日调仓，与老引擎一致）
ranks = scores_aligned.rank(axis=1, ascending=False, method="first")
selection = ranks <= top_n
weights = selection.astype(float)
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

# 计算收益
asset_returns = prices_aligned.pct_change().fillna(0.0)
prev_weights = weights.shift().fillna(0.0)
gross_returns = (prev_weights * asset_returns).sum(axis=1)

# 交易成本
weight_diff = weights.diff().abs().sum(axis=1).fillna(0.0)
turnover = 0.5 * weight_diff
net_returns = gross_returns - fees * turnover

# 净值曲线
init_cash = 1_000_000
equity = (1 + net_returns).cumprod() * init_cash

# 统计
total_return = (equity.iloc[-1] / init_cash - 1) * 100
periods_per_year = 252
n_years = len(equity) / periods_per_year
annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100

sharpe = (
    net_returns.mean() / net_returns.std() * np.sqrt(periods_per_year)
    if net_returns.std() > 0
    else 0
)
running_max = equity.cummax()
drawdown = (equity / running_max - 1) * 100
max_dd = drawdown.min()

print(f"总收益率: {total_return:.2f}%")
print(f"年化收益: {annual_return:.2f}%")
print(f"Sharpe: {sharpe:.4f}")
print(f"最大回撤: {max_dd:.2f}%")
print(f"总换手: {turnover.sum():.2f}")

# === 老引擎逻辑（从代码反推） ===
print("\n" + "-" * 80)
print("老引擎计算（理论复现）")
print("-" * 80)

# 老引擎应该也是类似逻辑，但可能调仓频率或其他细节不同
# 从CSV看老引擎的annual_return约0.156, sharpe约0.82
# 我们的annual_return应该接近

print(f"\n预期值（从老引擎CSV推算）:")
print(f"  年化收益: 约15-16%")
print(f"  Sharpe: 约0.8")
print(f"  最大回撤: 约-25%")

print("\n" + "=" * 80)
print("结论:")
if abs(annual_return - 15.68) < 2:
    print("✅ 年化收益接近，新引擎计算基本正确")
else:
    print(f"❌ 年化收益偏差较大: {annual_return:.2f}% vs 预期15.68%")
    print("   需要检查调仓频率、成本计算等细节")
