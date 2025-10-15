#!/usr/bin/env python3
"""
分析相关性剔除机制和因子集中度风险
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 读取扩展因子配置
with open("etf_rotation/configs/extended_scoring.yaml") as f:
    extended_config = yaml.safe_load(f)

# 读取因子集
factor_set_file = extended_config["factor_set_file"]
with open(factor_set_file) as f:
    factor_set = yaml.safe_load(f)

extended_factors = factor_set["factors"]
print(f"扩展因子集总数: {len(extended_factors)}")

# 因子分类分析
factor_categories = {
    "动量类": ["Momentum", "MOM_"],
    "均线类": ["MA", "EMA", "SMA", "WMA", "TA_SMA", "TA_EMA", "TA_WMA"],
    "趋势类": ["TREND", "FIX"],
    "震荡类": ["RSI", "STOCH", "WILLR", "CCI", "MACD"],
    "波动率类": ["ATR", "BB"],
    "成交量类": ["OBV"],
}

category_counts = {}
for category, keywords in factor_categories.items():
    count = 0
    for factor in extended_factors:
        if any(keyword in factor for keyword in keywords):
            count += 1
    category_counts[category] = count
    print(f"{category}: {count} 个因子")

print(f"\n各类因子占比:")
total_factors = len(extended_factors)
for category, count in category_counts.items():
    percentage = count / total_factors * 100
    print(f"{category}: {percentage:.1f}%")

# 分析相关性剔除的影响
print(f"\n=== 相关性剔除机制分析 ===")
print(f"相关性阈值: {extended_config.get('correlation_threshold', 0.9)}")
print(f"因子选择方式: {extended_config.get('factor_selection', 'equal_weight')}")

# 估算因子剔除的严重程度
print(f"\n根据风险分析结果:")
print(f"• 发现862对高度相关因子 (|r| > 0.9)")
print(f"• 均线类因子占32/67 = 47.8%")
print(f"• 高度相关的因子对主要集中在均线类和技术指标")

# 集中度风险分析
print(f"\n=== 集中度风险分析 ===")

# 假设相关性剔除后剩余的因子分布
# 由于均线类因子高度相关，大部分会被剔除
remaining_estimates = {
    "动量类": category_counts["动量类"],  # 基本保留
    "均线类": 3,  # 从32个减少到3个
    "趋势类": category_counts["趋势类"],  # 基本保留
    "震荡类": 4,  # 从13个减少到4个
    "波动率类": 2,  # 从11个减少到2个
    "成交量类": category_counts["成交量类"],  # 基本保留
}

print(f"估算相关性剔除后剩余因子分布:")
total_remaining = sum(remaining_estimates.values())
for category, estimated in remaining_estimates.items():
    original = category_counts[category]
    percentage = estimated / total_remaining * 100
    print(f"{category}: {estimated} 个 (原{original}个) → {percentage:.1f}%")

print(f"\n剔除比例: {(total_factors - total_remaining)/total_factors:.1%}")
print(f"剩余因子: {total_remaining}/{total_factors}")

# 风险集中度评估
print(f"\n=== 风险集中度评估 ===")

# 计算集中度指标
max_category_percentage = max(remaining_estimates.values()) / total_remaining
print(f"最大类别集中度: {max_category_percentage:.1%}")

if max_category_percentage > 0.4:
    print("🚨 高风险: 单一类别因子占比超过40%")
elif max_category_percentage > 0.3:
    print("⚠️ 中风险: 单一类别因子占比超过30%")
else:
    print("✅ 低风险: 因子分布相对均衡")

# 动量因子集中度
momentum_percentage = remaining_estimates["动量类"] / total_remaining
print(f"动量因子集中度: {momentum_percentage:.1%}")

if momentum_percentage > 0.3:
    print("⚠️ 注意: 动量因子占比过高，可能存在动量风格集中风险")

# 均线因子过度剔除风险
ma_reduction = (
    category_counts["均线类"] - remaining_estimates["均线类"]
) / category_counts["均线类"]
print(f"均线因子剔除比例: {ma_reduction:.1%}")

if ma_reduction > 0.8:
    print("🚨 严重: 均线因子被过度剔除，可能丢失重要趋势信息")
elif ma_reduction > 0.6:
    print("⚠️ 警告: 均线因子剔除比例较高")

# 震荡指标剔除风险
oscillator_reduction = (
    category_counts["震荡类"] - remaining_estimates["震荡类"]
) / category_counts["震荡类"]
print(f"震荡指标剔除比例: {oscillator_reduction:.1%}")

print(f"\n=== 改进建议 ===")
print("1. 调整相关性阈值: 考虑从0.9提高到0.95，减少过度剔除")
print("2. 分层剔除: 先在类别内部剔除，再跨类别剔除")
print("3. 权重调整: 对过度集中的类别降低权重")
print("4. 因子验证: 重新验证被剔除因子的有效性")
print("5. 增加因子多样性: 引入更多不同类型的因子")

# 计算理论上的最优因子数量
print(f"\n=== 理论分析 ===")
print(f"根据现代投资组合理论，当因子相关性较高时:")
print(f"• 有效因子数量 ≈ 1 / (平均相关系数)")
print(f"• 假设平均相关系数为0.8-0.9")
print(f"• 则有效因子数量为1.1-1.3个")
print(f"• 当前67个因子存在严重的信息冗余")

print(f"\n建议的最优配置:")
print(f"• 动量因子: 2-3个 (不同周期)")
print(f"• 均线因子: 2-3个 (不同类型)")
print(f"• 震荡指标: 2-3个 (RSI, MACD, KDJ)")
print(f"• 波动率因子: 1-2个")
print(f"• 成交量因子: 1-2个")
print(f"• 总计: 8-13个低相关因子")

print(f"\n当前问题总结:")
print(f"🚨 主要风险: 因子过度集中，多样性不足")
print(f"🚨 次要风险: 信息冗余严重，计算资源浪费")
print(f"🚨 潜在问题: 过度拟合风险增加")
