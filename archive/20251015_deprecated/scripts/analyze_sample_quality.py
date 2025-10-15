#!/usr/bin/env python3
"""
评估样本期长度和数据质量
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 读取回测数据
extended_summary = pd.read_csv("rotation_output/backtest/backtest_summary_extended.csv")
core_summary = pd.read_csv("rotation_output/backtest/backtest_summary.csv")

print("=== 样本期长度评估 ===")

# 分析回测期长度
start_date = extended_summary["trade_date"].min()
end_date = extended_summary["trade_date"].max()

print(f"回测开始日期: {start_date}")
print(f"回测结束日期: {end_date}")

# 转换为日期格式
start_dt = pd.to_datetime(str(start_date), format="%Y%m%d")
end_dt = pd.to_datetime(str(end_date), format="%Y%m%d")

# 计算回测期长度
total_months = len(extended_summary)
total_days = (end_dt - start_dt).days

print(f"回测月数: {total_months} 个月")
print(f"回测天数: {total_days} 天")
print(f"回测年数: {total_months/12:.1f} 年")

# 评估样本期充分性
print(f"\n=== 样本期充分性评估 ===")

# 统计显著性要求的最小样本
min_months_stats = 30  # 最少30个月用于统计显著性检验
min_months_strategy = 60  # 最少60个月用于策略评估

if total_months < min_months_stats:
    print(f"🚨 严重不足: 回测期少于{min_months_stats}个月，无法进行有效的统计推断")
elif total_months < min_months_strategy:
    print(f"⚠️  不足: 回测期少于{min_months_strategy}个月，策略评估可靠性有限")
else:
    print(f"✅ 充分: 回测期达到{total_months}个月，满足基本策略评估要求")

# 市场环境覆盖评估
print(f"\n=== 市场环境覆盖评估 ===")

# 检查是否覆盖不同市场环境
months_2024 = len(
    [d for d in extended_summary["trade_date"] if str(d).startswith("2024")]
)
months_2025 = len(
    [d for d in extended_summary["trade_date"] if str(d).startswith("2025")]
)

print(f"2024年覆盖月数: {months_2024}")
print(f"2025年覆盖月数: {months_2025}")

if months_2024 > 0 and months_2025 > 0:
    print("✅ 覆盖多个年度")
elif months_2024 > 6 or months_2025 > 6:
    print("⚠️  主要覆盖单一年度，可能遗漏年度间差异")
else:
    print("🚨 样本期过短，无法评估跨年度稳健性")

# 季节性覆盖
months_by_quarter = {1: 0, 2: 0, 3: 0, 4: 0}  # Q1, Q2, Q3, Q4
for date_str in extended_summary["trade_date"]:
    dt = pd.to_datetime(str(date_str), format="%Y%m%d")
    quarter = (dt.month - 1) // 3 + 1
    months_by_quarter[quarter] += 1

print(f"\n季度覆盖:")
for quarter, count in months_by_quarter.items():
    print(f"Q{quarter}: {count} 个月")

quarters_covered = sum(1 for count in months_by_quarter.values() if count > 0)
if quarters_covered < 4:
    print(f"⚠️  只覆盖{quarters_covered}个季度，季节性效应评估不充分")
else:
    print("✅ 覆盖全部4个季度")

# 数据质量评估
print(f"\n=== 数据质量评估 ===")

# 检查每个月的样本完整性
print(f"扩展系统每月ETF数量统计:")
print(f"平均宇宙大小: {extended_summary['universe_size'].mean():.1f}")
print(f"最小宇宙大小: {extended_summary['universe_size'].min()}")
print(f"最大宇宙大小: {extended_summary['universe_size'].max()}")
print(f"宇宙大小标准差: {extended_summary['universe_size'].std():.1f}")

print(f"\n核心系统每月ETF数量统计:")
print(f"平均宇宙大小: {core_summary['universe_size'].mean():.1f}")
print(f"最小宇宙大小: {core_summary['universe_size'].min()}")
print(f"最大宇宙大小: {core_summary['universe_size'].max()}")
print(f"宇宙大小标准差: {core_summary['universe_size'].std():.1f}")

# 检查缺失数据
min_universe_ext = extended_summary["universe_size"].min()
min_universe_core = core_summary["universe_size"].min()

print(f"\n最小样本量检查:")
print(f"扩展系统最小ETF数量: {min_universe_ext}")
print(f"核心系统最小ETF数量: {min_universe_core}")

if min_universe_ext < 20:
    print("🚨 扩展系统样本量不足，可能影响统计显著性")
if min_universe_core < 20:
    print("🚨 核心系统样本量不足，可能影响统计显著性")

# 组合构建质量
print(f"\n=== 组合构建质量评估 ===")
print(f"扩展系统平均组合大小: {extended_summary['portfolio_size'].mean():.1f}")
print(f"核心系统平均组合大小: {core_summary['portfolio_size'].mean():.1f}")

# 检查评分通过率
ext_scored_ratio = extended_summary["scored_size"] / extended_summary["universe_size"]
core_scored_ratio = core_summary["scored_size"] / core_summary["universe_size"]

print(f"扩展系统平均评分通过率: {ext_scored_ratio.mean():.1%}")
print(f"核心系统平均评分通过率: {core_scored_ratio.mean():.1%}")

if ext_scored_ratio.mean() < 0.5:
    print("⚠️  扩展系统因子筛选过于严格，可能遗漏有效信号")
if core_scored_ratio.mean() < 0.5:
    print("⚠️  核心系统因子筛选过于严格，可能遗漏有效信号")

# 统计显著性评估
print(f"\n=== 统计显著性评估 ===")

# 估算需要的样本量（基于夏普比率差异检验）
ext_sharpe = 0.77  # 从之前分析获得
core_sharpe = 1.73
volatility = 0.5  # 年化波动率

# 使用夏普比率差异检验的样本量公式
# n = (Z_α/2 + Z_β)² * (σ₁² + σ₂²) / (μ₁ - μ₂)²
# 简化估算
desired_power = 0.8
alpha = 0.05
z_alpha = 1.96
z_beta = 0.84

# 夏普比率差异
sharpe_diff = abs(ext_sharpe - core_sharpe)
estimated_sample_size = (z_alpha + z_beta) ** 2 * 2 * volatility**2 / (sharpe_diff**2)

print(f"夏普比率差异检验所需样本量:")
print(f"当前差异: {sharpe_diff:.2f}")
print(f"估算所需月数: {estimated_sample_size:.0f}")
print(f"当前月数: {total_months}")

if total_months < estimated_sample_size:
    print(
        f"🚨 样本量不足: 需要{estimated_sample_size:.0f}个月，当前只有{total_months}个月"
    )
else:
    print(f"✅ 样本量充足: 满足统计显著性要求")

# 过拟合风险评估
print(f"\n=== 过拟合风险评估 ===")

# 因子数量与样本量比例
ext_factors = 67  # 扩展因子数
core_factors = 4  # 核心因子数

print(
    f"扩展系统: {ext_factors}个因子 / {total_months}个月 = {ext_factors/total_months:.1f} 因子/月"
)
print(
    f"核心系统: {core_factors}个因子 / {total_months}个月 = {core_factors/total_months:.1f} 因子/月"
)

# 经验法则: 每个因子至少需要10-20个观测
min_obs_per_factor = 10
required_months_ext = ext_factors * min_obs_per_factor / 8  # 8个持仓
required_months_core = core_factors * min_obs_per_factor / 8

print(f"扩展系统最小需要月数: {required_months_ext:.0f}")
print(f"核心系统最小需要月数: {required_months_core:.0f}")

if total_months < required_months_ext:
    print(f"🚨 扩展系统过拟合风险高: 因子太多，样本期太短")
if total_months < required_months_core:
    print(f"🚨 核心系统过拟合风险高: 因子评估不充分")

# 综合评估
print(f"\n=== 综合数据质量评估 ===")

issues = []

if total_months < 24:
    issues.append("样本期过短")
if min_universe_ext < 20 or min_universe_core < 20:
    issues.append("样本量不足")
if ext_scored_ratio.mean() < 0.5 or core_scored_ratio.mean() < 0.5:
    issues.append("筛选过于严格")
if total_months < estimated_sample_size:
    issues.append("统计显著性不足")
if total_months < required_months_ext:
    issues.append("扩展系统过拟合风险")

if issues:
    print("🚨 数据质量问题:")
    for issue in issues:
        print(f"  • {issue}")
else:
    print("✅ 数据质量良好")

print(f"\n建议:")
if total_months < 24:
    print("• 延长回测期至少24个月")
if min_universe_ext < 20 or min_universe_core < 20:
    print("• 增加ETF样本数量或降低筛选标准")
if ext_scored_ratio.mean() < 0.5:
    print("• 调整扩展系统因子筛选条件")
if core_scored_ratio.mean() < 0.5:
    print("• 调整核心系统因子筛选条件")
if total_months < required_months_ext:
    print("• 减少扩展因子数量或延长回测期")
if quarters_covered < 4:
    print("• 确保覆盖所有季度以评估季节性效应")
