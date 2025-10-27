#!/usr/bin/env python3
"""诊断回测负收益问题的核心脚本"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 加载最新的panel和prices数据
panel_file = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/data/results/panels/panel_20251024_214306/panel.parquet"
price_dir = "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"

print("=" * 80)
print("ETF轮动系统回测诊断报告")
print("=" * 80)

# 1. 加载Panel
print("\n[1] 加载因子Panel...")
panel = pd.read_parquet(panel_file)
print(f"Panel形状: {panel.shape}")
print(f"因子列: {list(panel.columns[:10])}...")
print(
    f"日期范围: {panel.index.get_level_values(1).min()} ~ {panel.index.get_level_values(1).max()}"
)
print(f"标的数量: {panel.index.get_level_values(0).nunique()}")

# 检查NaN比例
nan_pct = panel.isna().sum().sum() / (panel.shape[0] * panel.shape[1])
print(f"Panel整体NaN比例: {nan_pct:.2%}")

# 2. 加载价格数据
print("\n[2] 加载价格数据...")
import glob

price_files = sorted(glob.glob(f"{price_dir}/*.parquet"))
print(f"价格文件数: {len(price_files)}")

prices_list = []
for f in price_files[:5]:  # 只加载前5个文件测试
    df = pd.read_parquet(f)
    symbol = Path(f).stem.split("_")[0]
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["trade_date"])
    prices_list.append(df[["date", "close", "symbol"]])

price_df = pd.concat(prices_list, ignore_index=True)
pivot_prices = price_df.pivot(index="date", columns="symbol", values="close")
pivot_prices = pivot_prices.sort_index()

print(f"价格矩阵形状: {pivot_prices.shape}")
print(f"日期范围: {pivot_prices.index.min()} ~ {pivot_prices.index.max()}")
price_nan_pct = pivot_prices.isna().sum().sum() / (
    pivot_prices.shape[0] * pivot_prices.shape[1]
)
print(f"价格矩阵NaN比例: {price_nan_pct:.2%}")

# 3. 模拟简单的回测逻辑
print("\n[3] 模拟简单回测逻辑...")

# 提取一个因子
factor_name = "MOM_20D" if "MOM_20D" in panel.columns else panel.columns[0]
print(f"使用因子: {factor_name}")

# Unstack到(date, symbol)结构
factor_scores = panel[factor_name].unstack(level="symbol")
print(f"因子得分矩阵: {factor_scores.shape}")

# 标准化
factor_normalized = (
    factor_scores - factor_scores.mean(axis=1).values[:, np.newaxis]
) / (factor_scores.std(axis=1).values[:, np.newaxis] + 1e-8)
print(
    f"标准化后NaN比例: {factor_normalized.isna().sum().sum() / (factor_normalized.shape[0] * factor_normalized.shape[1]):.2%}"
)

# 应用shift(1) - 这是关键的一步！
factor_shifted = factor_normalized.shift(1)
print(f"shift(1)后第一行全部NaN: {factor_shifted.iloc[0].isna().all()}")
print(
    f"shift(1)后NaN比例: {factor_shifted.isna().sum().sum() / (factor_shifted.shape[0] * factor_shifted.shape[1]):.2%}"
)

# 选择Top3
top_n = 3
ranks = factor_shifted.rank(axis=1, ascending=False, method="first")
selection = ranks <= top_n
weights = selection.astype(float)
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

print(f"\n权重矩阵形状: {weights.shape}")
print(f"权重矩阵前5天:")
print(weights.head())
print(f"权重非零天数: {(weights.sum(axis=1) > 0).sum()} / {len(weights)}")

# 对齐价格
common_dates = pivot_prices.index.intersection(weights.index)
common_symbols = pivot_prices.columns.intersection(weights.columns)
print(f"\n对齐后: {len(common_dates)}天, {len(common_symbols)}个标的")

aligned_prices = pivot_prices.loc[common_dates, common_symbols]
aligned_weights = weights.loc[common_dates, common_symbols]

# 计算收益率
returns = aligned_prices.pct_change(fill_method=None).fillna(0.0)
print(f"收益率矩阵: {returns.shape}")
print(f"收益率前5天:")
print(returns.head())

# 计算组合收益
# 注意：这里使用T-1日权重×T日收益
portfolio_returns = (aligned_weights.shift(1) * returns).sum(axis=1)
print(f"\n组合收益序列长度: {len(portfolio_returns)}")
print(f"组合收益前10天:")
print(portfolio_returns.head(10))
print(f"非零收益天数: {(portfolio_returns != 0).sum()}")
print(f"平均日收益: {portfolio_returns.mean():.6f}")
print(f"累计收益: {(1 + portfolio_returns).prod() - 1:.2%}")

# 4. 诊断问题
print("\n" + "=" * 80)
print("诊断结果")
print("=" * 80)

issues = []

if factor_shifted.iloc[0].isna().all():
    issues.append("⚠️ 严重问题: shift(1)导致第一天信号全部NaN，策略第一天空仓")

if (weights.sum(axis=1) == 0).sum() > len(weights) * 0.5:
    issues.append(f"⚠️ 严重问题: 超过50%的交易日权重为0（空仓）")

if portfolio_returns.mean() < 0:
    issues.append(f"⚠️ 问题: 平均日收益为负 ({portfolio_returns.mean():.6f})")

if (portfolio_returns == 0).sum() > len(portfolio_returns) * 0.3:
    issues.append(f"⚠️ 问题: 超过30%的交易日收益为0")

if len(issues) == 0:
    print("✅ 未发现明显问题")
else:
    for issue in issues:
        print(issue)

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
