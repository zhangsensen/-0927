#!/usr/bin/env python3
"""
Top1000策略深度分析
分析top1000策略的权重分布、因子重要性、策略聚类等
"""

import ast
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def load_and_parse_data():
    """加载并解析top1000策略数据"""
    print("🔄 加载Top1000策略数据...")

    df = pd.read_csv("strategies/results/top1000_strategies.csv")

    # 解析权重字符串为实际数值
    print("🔧 解析权重数据...")
    weights_list = []
    for weights_str in df["weights"]:
        weights = ast.literal_eval(weights_str)
        weights_list.append(list(weights))

    # 获取因子名称（从factors列解析）
    factor_names = ast.literal_eval(df.iloc[0]["factors"])

    # 创建权重DataFrame
    weights_df = pd.DataFrame(
        weights_list, columns=[f"weight_{i}" for i in range(len(factor_names))]
    )

    # 合并数据
    combined_df = pd.concat([df.reset_index(drop=True), weights_df], axis=1)

    print(f"✅ 加载完成: {len(combined_df)} 个策略, {len(factor_names)} 个因子")

    return combined_df, factor_names


def analyze_performance_distribution(df):
    """分析性能分布"""
    print("\n📊 策略性能分布分析")
    print("=" * 60)

    # 基本统计
    print("夏普比率分布:")
    print(f"  均值: {df['sharpe'].mean():.4f}")
    print(f"  标准差: {df['sharpe'].std():.4f}")
    print(f"  最小值: {df['sharpe'].min():.4f}")
    print(f"  最大值: {df['sharpe'].max():.4f}")
    print(f"  25%分位: {df['sharpe'].quantile(0.25):.4f}")
    print(f"  50%分位: {df['sharpe'].quantile(0.50):.4f}")
    print(f"  75%分位: {df['sharpe'].quantile(0.75):.4f}")

    print("\n年化收益分布:")
    print(
        f"  均值: {df['annual_return'].mean():.4f} ({df['annual_return'].mean()*100:.2f}%)"
    )
    print(f"  标准差: {df['annual_return'].std():.4f}")
    print(f"  范围: {df['annual_return'].min():.4f} - {df['annual_return'].max():.4f}")

    print("\n最大回撤分布:")
    print(
        f"  均值: {df['max_drawdown'].mean():.4f} ({df['max_drawdown'].mean()*100:.2f}%)"
    )
    print(f"  标准差: {df['max_drawdown'].std():.4f}")

    # Top-N分析
    top_n_counts = df["top_n"].value_counts().sort_index()
    print(f"\nTop-N分布:")
    for top_n, count in top_n_counts.items():
        sharpe_mean = df[df["top_n"] == top_n]["sharpe"].mean()
        return_mean = df[df["top_n"] == top_n]["annual_return"].mean()
        print(
            f"  Top-{int(top_n)}: {count} 个策略, 平均夏普 {sharpe_mean:.4f}, 平均收益 {return_mean:.4f}"
        )

    return df


def analyze_factor_weights(df, factor_names):
    """分析因子权重分布"""
    print(f"\n🔍 因子权重分析")
    print("=" * 60)

    # 提取权重列
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    weights_data = df[weight_cols]

    # 计算每个因子的统计信息
    factor_stats = []
    for i, factor_name in enumerate(factor_names):
        weights = weights_data[f"weight_{i}"]
        stats = {
            "factor_name": factor_name,
            "mean_weight": weights.mean(),
            "std_weight": weights.std(),
            "min_weight": weights.min(),
            "max_weight": weights.max(),
            "zero_ratio": (weights == 0).mean(),
            "positive_ratio": (weights > 0).mean(),
        }
        factor_stats.append(stats)

    factor_stats_df = pd.DataFrame(factor_stats)
    factor_stats_df = factor_stats_df.sort_values("mean_weight", ascending=False)

    print("Top 15 重要因子 (按平均权重):")
    for i, row in factor_stats_df.head(15).iterrows():
        print(
            f"  {i+1:2d}. {row['factor_name']:20s}: "
            f"均值={row['mean_weight']:.4f}, "
            f"标准差={row['std_weight']:.4f}, "
            f"非零率={1-row['zero_ratio']:.2%}"
        )

    # 分析权重分布模式
    print(f"\n权重分布模式:")
    total_weights = weights_data.values.sum(axis=1)
    print(f"  平均总权重: {total_weights.mean():.4f}")
    print(f"  总权重标准差: {total_weights.std():.4f}")

    # 非零权重数量
    non_zero_counts = (weights_data > 0).sum(axis=1)
    print(f"  平均使用因子数: {non_zero_counts.mean():.1f}")
    print(f"  因子数量范围: {non_zero_counts.min()} - {non_zero_counts.max()}")

    return factor_stats_df, weights_data


def identify_strategy_types(df, factor_names):
    """识别策略类型"""
    print(f"\n🎯 策略类型识别")
    print("=" * 60)

    # 提取权重列
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    weights_data = df[weight_cols]

    # 定义因子类别
    factor_categories = {
        "price_position": [f for f in factor_names if "PRICE_POSITION" in f],
        "volatility": [f for f in factor_names if "STDDEV" in f or "VAR" in f],
        "momentum": [f for f in factor_names if "MOMENTUM" in f],
        "rsi": [f for f in factor_names if "RSI" in f],
        "stochastic": [f for f in factor_names if "STOCH" in f],
        "other": [
            f
            for f in factor_names
            if not any(
                f.startswith(p)
                for p in [
                    "PRICE_POSITION",
                    "TA_STDDEV",
                    "TA_VAR",
                    "MOMENTUM",
                    "TA_RSI",
                    "VBT_RSI",
                    "VBT_STOCH",
                ]
            )
        ],
    }

    # 计算每个类别的权重
    strategy_types = []
    for idx, row in df.iterrows():
        type_weights = {}
        total_weight = 0

        for category, factors in factor_categories.items():
            category_weight = 0
            for factor in factors:
                if factor in factor_names:
                    factor_idx = factor_names.index(factor)
                    weight_col = f"weight_{factor_idx}"
                    if weight_col in row:
                        category_weight += row[weight_col]

            type_weights[category] = category_weight
            total_weight += category_weight

        # 标准化权重
        if total_weight > 0:
            for category in type_weights:
                type_weights[category] = type_weights[category] / total_weight

        type_weights["strategy_id"] = idx
        type_weights["sharpe"] = row["sharpe"]
        type_weights["annual_return"] = row["annual_return"]
        strategy_types.append(type_weights)

    strategy_df = pd.DataFrame(strategy_types)

    # 分析主要策略类型
    print("策略类型权重分布 (平均值):")
    for category in factor_categories.keys():
        if category in strategy_df.columns:
            avg_weight = strategy_df[category].mean()
            best_sharpe = strategy_df.loc[strategy_df[category].idxmax()]["sharpe"]
            print(
                f"  {category:15s}: 平均权重 {avg_weight:.3f}, 最高夏普 {best_sharpe:.4f}"
            )

    return strategy_df, factor_categories


def analyze_top_performers(df, factor_names, n=50):
    """分析头部表现者"""
    print(f"\n🏆 Top {n} 策略深度分析")
    print("=" * 60)

    top_strategies = df.head(n)

    print(f"Top {n} 策略性能统计:")
    print(
        f"  夏普比率: 均值={top_strategies['sharpe'].mean():.4f}, "
        f"范围=[{top_strategies['sharpe'].min():.4f}, {top_strategies['sharpe'].max():.4f}]"
    )
    print(
        f"  年化收益: 均值={top_strategies['annual_return'].mean():.4f}, "
        f"范围=[{top_strategies['annual_return'].min():.4f}, {top_strategies['annual_return'].max():.4f}]"
    )
    print(f"  最大回撤: 均值={top_strategies['max_drawdown'].mean():.4f}")
    print(f"  换手率: 均值={top_strategies['turnover'].mean():.2f}")

    # 分析Top策略的因子权重
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    top_weights = top_strategies[weight_cols]

    print(f"\nTop {n} 策略因子权重 (前15个):")
    top_factor_weights = top_weights.mean().sort_values(ascending=False).head(15)
    for i, (col, weight) in enumerate(top_factor_weights.items()):
        factor_idx = int(col.split("_")[1])
        factor_name = factor_names[factor_idx]
        print(f"  {i+1:2d}. {factor_name:20s}: {weight:.5f}")

    # 分析Top策略的多样性
    non_zero_counts = (top_weights > 0).sum(axis=1)
    print(f"\nTop {n} 策略多样性:")
    print(f"  平均使用因子数: {non_zero_counts.mean():.1f}")
    print(f"  因子数标准差: {non_zero_counts.std():.1f}")

    return top_strategies


def generate_strategy_insights(df, factor_names, factor_stats_df):
    """生成策略洞察"""
    print(f"\n💡 策略洞察与建议")
    print("=" * 60)

    # 1. 因子重要性洞察
    print("1. 因子重要性洞察:")
    top_factors = factor_stats_df.head(10)
    print(f"   - 核心因子: {', '.join(top_factors['factor_name'].head(5).tolist())}")
    print(f"   - 次要因子: {', '.join(top_factors['factor_name'].iloc[5:10].tolist())}")

    # 2. 权重分配模式
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    weights_data = df[weight_cols]

    # 计算权重集中度
    weight_concentration = []
    for idx, row in df.iterrows():
        weights = row[weight_cols].values
        weights = weights[weights > 0]  # 只考虑非零权重
        if len(weights) > 0:
            concentration = (weights**2).sum() / (weights.sum() ** 2)  # HHI指数
            weight_concentration.append(concentration)

    avg_concentration = np.mean(weight_concentration)
    print(f"\n2. 权重分配模式:")
    print(f"   - 平均权重集中度(HHI): {avg_concentration:.4f}")
    if avg_concentration < 0.3:
        print("   - 策略特征: 高度分散，风险控制良好")
    elif avg_concentration < 0.5:
        print("   - 策略特征: 适度分散，平衡收益与风险")
    else:
        print("   - 策略特征: 相对集中，追求高收益")

    # 3. 性能与因子数量关系
    non_zero_counts = (weights_data > 0).sum(axis=1)
    df["factor_count"] = non_zero_counts

    performance_by_factor_count = (
        df.groupby("factor_count")
        .agg({"sharpe": ["mean", "std", "count"], "annual_return": "mean"})
        .round(4)
    )

    print(f"\n3. 因子数量与性能关系:")
    print("   使用的因子数量越多，性能:")
    best_factor_count = performance_by_factor_count["sharpe"]["mean"].idxmax()
    best_sharpe = performance_by_factor_count["sharpe"]["mean"].max()
    print(f"   - 最佳因子数量: {best_factor_count} 个 (平均夏普 {best_sharpe:.4f})")

    # 4. 风险收益特征
    print(f"\n4. 风险收益特征:")
    print(f"   - 夏普 > 0.8 的策略: {len(df[df['sharpe'] > 0.8])} 个")
    print(f"   - 年化收益 > 15% 的策略: {len(df[df['annual_return'] > 0.15])} 个")
    print(f"   - 最大回撤 < 20% 的策略: {len(df[df['max_drawdown'] < 0.2])} 个")

    # 5. 实际应用建议
    print(f"\n5. 实际应用建议:")
    print(f"   - 推荐因子组合: 使用权重前5的因子构建核心策略")
    print(f"   - 权重分配: 建议使用0.05-0.1的权重范围，避免过度集中")
    print(f"   - Top-N设置: 8个标的组合表现最佳")
    print(f"   - 风险控制: 关注最大回撤，建议设置止损线在20%左右")


def save_analysis_results(df, factor_names, factor_stats_df):
    """保存分析结果"""
    print(f"\n💾 保存分析结果...")

    # 保存详细的因子统计
    factor_stats_df.to_csv(
        "strategies/results/factor_detailed_analysis.csv", index=False
    )

    # 保存策略洞察摘要
    insights = {
        "total_strategies": len(df),
        "factor_count": len(factor_names),
        "avg_sharpe": df["sharpe"].mean(),
        "best_sharpe": df["sharpe"].max(),
        "avg_return": df["annual_return"].mean(),
        "avg_drawdown": df["max_drawdown"].mean(),
        "optimal_top_n": df.groupby("top_n")["sharpe"].mean().idxmax(),
        "top_factors": factor_stats_df.head(10)["factor_name"].tolist(),
    }

    import json

    with open("strategies/results/strategy_insights.json", "w") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

    print(f"   - 因子详细分析: strategies/results/factor_detailed_analysis.csv")
    print(f"   - 策略洞察摘要: strategies/results/strategy_insights.json")


def main():
    """主函数"""
    print("🚀 Top1000策略深度分析")
    print("=" * 60)

    # 加载数据
    df, factor_names = load_and_parse_data()

    # 性能分布分析
    df = analyze_performance_distribution(df)

    # 因子权重分析
    factor_stats_df, weights_data = analyze_factor_weights(df, factor_names)

    # 策略类型识别
    strategy_df, factor_categories = identify_strategy_types(df, factor_names)

    # 头部表现者分析
    top_strategies = analyze_top_performers(df, factor_names)

    # 生成策略洞察
    generate_strategy_insights(df, factor_names, factor_stats_df)

    # 保存结果
    save_analysis_results(df, factor_names, factor_stats_df)

    print(f"\n✅ Top1000策略分析完成!")
    print(f"   基于 {len(df)} 个策略和 {len(factor_names)} 个因子的深度分析")
    print(f"   为实际策略构建提供了数据驱动的指导")


if __name__ == "__main__":
    main()
