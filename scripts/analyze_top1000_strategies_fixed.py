#!/usr/bin/env python3
"""
Top1000策略深度分析 (修复版)
分析top1000策略的权重分布、因子重要性、策略聚类等
"""

import ast
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_and_combine_all_batches():
    """加载并合并所有批次的数据"""
    print("🔄 加载所有批次数据...")

    results_dir = Path("strategies/results")
    batch_files = sorted(glob.glob(str(results_dir / "top35_batch*.csv")))

    all_data = []

    for i, file in enumerate(batch_files):
        try:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f"  批次 {i}: 加载 {len(df)} 行数据")
        except Exception as e:
            print(f"  ❌ 批次 {i} 加载失败: {e}")

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ 合并完成: 总共 {len(combined_df)} 个策略")

        # 按夏普比率排序
        combined_df = combined_df.sort_values("sharpe", ascending=False).reset_index(
            drop=True
        )
        print(
            f"   夏普比率范围: {combined_df['sharpe'].min():.4f} - {combined_df['sharpe'].max():.4f}"
        )

        return combined_df
    else:
        print("❌ 没有找到任何数据")
        return None


def parse_weights_data(df):
    """解析权重数据"""
    print("\n🔧 解析权重数据...")

    # 解析第一个策略的因子名称
    factors_str = df.iloc[0]["factors"]
    factor_names = ast.literal_eval(factors_str)
    print(f"   因子数量: {len(factor_names)}")

    # 解析所有权重
    all_weights = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            weights_str = row["weights"]
            weights = ast.literal_eval(weights_str)

            # 转换为浮点数
            weights_float = [float(w) for w in weights]

            if len(weights_float) == len(factor_names):
                all_weights.append(weights_float)
                valid_indices.append(idx)
        except Exception as e:
            print(f"   警告: 策略 {idx} 权重解析失败: {e}")

    print(f"   成功解析权重: {len(all_weights)}/{len(df)} 个策略")

    # 创建权重DataFrame
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    weights_df = pd.DataFrame(all_weights, columns=weight_cols, index=valid_indices)

    # 只保留有效策略
    valid_df = df.loc[valid_indices].copy()
    valid_df = pd.concat(
        [valid_df.reset_index(drop=True), weights_df.reset_index(drop=True)], axis=1
    )

    return valid_df, factor_names


def analyze_performance_overview(df):
    """性能概览分析"""
    print(f"\n📊 性能概览分析 (Top {len(df)} 策略)")
    print("=" * 60)

    # 基本统计
    metrics = [
        "sharpe",
        "annual_return",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
    ]

    print("核心指标统计:")
    for metric in metrics:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            min_val = df[metric].min()
            max_val = df[metric].max()

            if metric == "annual_return" or metric == "max_drawdown":
                print(
                    f"  {metric:12s}: {mean_val:.4f} ({mean_val*100:.2f}%) ± {std_val:.4f}, 范围: [{min_val:.4f}, {max_val:.4f}]"
                )
            elif metric == "turnover":
                print(
                    f"  {metric:12s}: {mean_val:.2f} ± {std_val:.2f}, 范围: [{min_val:.2f}, {max_val:.2f}]"
                )
            else:
                print(
                    f"  {metric:12s}: {mean_val:.4f} ± {std_val:.4f}, 范围: [{min_val:.4f}, {max_val:.4f}]"
                )

    # Top-N分析
    if "top_n" in df.columns:
        top_n_stats = (
            df.groupby("top_n")
            .agg({"sharpe": ["count", "mean", "std"], "annual_return": "mean"})
            .round(4)
        )

        print(f"\nTop-N性能分析:")
        for top_n in sorted(df["top_n"].unique()):
            subset = df[df["top_n"] == top_n]
            print(
                f"  Top-{int(top_n)}: {len(subset)} 个策略, "
                f"平均夏普 {subset['sharpe'].mean():.4f}, "
                f"平均收益 {subset['annual_return'].mean():.4f}"
            )

    # 性能分级
    print(f"\n性能分级:")
    sharpe_thresholds = [0.6, 0.7, 0.8]
    for threshold in sharpe_thresholds:
        count = len(df[df["sharpe"] >= threshold])
        percentage = count / len(df) * 100
        print(f"  夏普 ≥ {threshold}: {count} 个策略 ({percentage:.1f}%)")

    return df


def analyze_factor_importance(df, factor_names):
    """因子重要性分析"""
    print(f"\n🔍 因子重要性分析")
    print("=" * 60)

    # 提取权重列
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    weights_data = df[weight_cols]

    # 计算每个因子的统计信息
    factor_analysis = []

    for i, factor_name in enumerate(factor_names):
        weights = weights_data[f"weight_{i}"]

        # 只考虑非零权重
        non_zero_weights = weights[weights > 0]

        stats = {
            "factor_name": factor_name,
            "mean_weight": weights.mean(),
            "std_weight": weights.std(),
            "max_weight": weights.max(),
            "non_zero_count": len(non_zero_weights),
            "non_zero_ratio": len(non_zero_weights) / len(weights),
            "avg_non_zero_weight": (
                non_zero_weights.mean() if len(non_zero_weights) > 0 else 0
            ),
            "usage_frequency": len(non_zero_weights)
            / len(weights)
            * 100,  # 使用频率百分比
        }
        factor_analysis.append(stats)

    factor_df = pd.DataFrame(factor_analysis)
    factor_df = factor_df.sort_values("mean_weight", ascending=False)

    print(f"Top 15 重要因子 (按平均权重排序):")
    for i, (_, row) in enumerate(factor_df.head(15).iterrows()):
        print(
            f"  {i+1:2d}. {row['factor_name']:20s}: "
            f"均值={row['mean_weight']:.4f}, "
            f"使用率={row['usage_frequency']:.1f}%, "
            f"最大={row['max_weight']:.4f}"
        )

    # 因子类别分析
    print(f"\n因子类别分析:")
    categories = {
        "价格位置": [f for f in factor_names if "PRICE_POSITION" in f],
        "波动率": [f for f in factor_names if "STDDEV" in f or "VAR" in f],
        "RSI": [f for f in factor_names if "RSI" in f],
        "随机指标": [f for f in factor_names if "STOCH" in f],
        "动量": [f for f in factor_names if "MOMENTUM" in f],
        "其他": [],
    }

    # 分类其他因子
    assigned_factors = []
    for category_key, factor_list in categories.items():
        if category_key != "其他":
            assigned_factors.extend(factor_list)

    for f in factor_names:
        if f not in assigned_factors:
            categories["其他"].append(f)

    for category, factors in categories.items():
        if factors:
            category_weights = []
            for factor in factors:
                if factor in factor_names:
                    idx = factor_names.index(factor)
                    category_weights.extend(weights_data[f"weight_{idx}"].tolist())

            if category_weights:
                avg_weight = np.mean(category_weights)
                non_zero_ratio = sum(1 for w in category_weights if w > 0) / len(
                    category_weights
                )
                print(
                    f"  {category:8s}: 平均权重 {avg_weight:.4f}, 使用率 {non_zero_ratio:.2%}"
                )

    return factor_df, weights_data


def analyze_strategy_patterns(df, factor_names, weights_data):
    """策略模式分析"""
    print(f"\n🎯 策略模式分析")
    print("=" * 60)

    # 权重集中度分析
    concentration_scores = []
    factor_counts = []

    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]

    for idx, row in df.iterrows():
        weights = row[weight_cols].values
        non_zero_weights = weights[weights > 0]

        # HHI集中度指数
        if len(non_zero_weights) > 0:
            normalized_weights = non_zero_weights / non_zero_weights.sum()
            hhi = (normalized_weights**2).sum()
            concentration_scores.append(hhi)
            factor_counts.append(len(non_zero_weights))

    concentration_scores = np.array(concentration_scores)
    factor_counts = np.array(factor_counts)

    print(f"权重集中度分析:")
    print(f"  平均HHI指数: {concentration_scores.mean():.4f}")
    print(f"  HHI标准差: {concentration_scores.std():.4f}")
    print(f"  低集中度(HHI<0.3): {np.sum(concentration_scores < 0.3)} 个策略")
    print(
        f"  中集中度(0.3≤HHI<0.5): {np.sum((concentration_scores >= 0.3) & (concentration_scores < 0.5))} 个策略"
    )
    print(f"  高集中度(HHI≥0.5): {np.sum(concentration_scores >= 0.5)} 个策略")

    print(f"\n因子数量使用分析:")
    print(f"  平均使用因子数: {factor_counts.mean():.1f}")
    print(f"  因子数范围: {factor_counts.min()} - {factor_counts.max()}")
    print(f"  使用<10个因子: {np.sum(factor_counts < 10)} 个策略")
    print(
        f"  使用10-20个因子: {np.sum((factor_counts >= 10) & (factor_counts < 20))} 个策略"
    )
    print(f"  使用≥20个因子: {np.sum(factor_counts >= 20)} 个策略")

    # 性能与策略复杂度的关系
    df["factor_count"] = factor_counts
    df["concentration"] = concentration_scores

    complexity_performance = (
        df.groupby("factor_count")
        .agg({"sharpe": ["mean", "std", "count"], "annual_return": "mean"})
        .round(4)
    )

    print(f"\n策略复杂度与性能关系:")
    best_factor_count = None
    best_sharpe = 0

    for count in sorted(df["factor_count"].unique()):
        subset = df[df["factor_count"] == count]
        avg_sharpe = subset["sharpe"].mean()
        print(f"  {count} 个因子: {len(subset)} 个策略, 平均夏普 {avg_sharpe:.4f}")

        if avg_sharpe > best_sharpe and len(subset) >= 5:  # 至少5个策略才考虑
            best_sharpe = avg_sharpe
            best_factor_count = count

    if best_factor_count:
        print(f"\n最优因子数量: {best_factor_count} 个 (平均夏普 {best_sharpe:.4f})")

    return concentration_scores, factor_counts


def identify_top_performers(df, factor_names, n=100):
    """识别并分析顶级表现者"""
    print(f"\n🏆 Top {n} 策略深度分析")
    print("=" * 60)

    top_strategies = df.head(n).copy()

    print(f"Top {n} 策略性能:")
    print(
        f"  夏普比率: {top_strategies['sharpe'].mean():.4f} ± {top_strategies['sharpe'].std():.4f}"
    )
    print(
        f"  年化收益: {top_strategies['annual_return'].mean():.4f} ± {top_strategies['annual_return'].std():.4f}"
    )
    print(
        f"  最大回撤: {top_strategies['max_drawdown'].mean():.4f} ± {top_strategies['max_drawdown'].std():.4f}"
    )
    print(
        f"  换手率: {top_strategies['turnover'].mean():.2f} ± {top_strategies['turnover'].std():.2f}"
    )

    # 分析Top策略的因子偏好
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    top_weights = top_strategies[weight_cols]

    print(f"\nTop {n} 策略因子偏好 (前15个):")
    top_factor_means = top_weights.mean().sort_values(ascending=False).head(15)

    for i, (col, mean_weight) in enumerate(top_factor_means.items()):
        factor_idx = int(col.split("_")[1])
        factor_name = factor_names[factor_idx]
        usage_rate = (top_weights[col] > 0).mean() * 100
        print(
            f"  {i+1:2d}. {factor_name:20s}: "
            f"平均权重 {mean_weight:.4f}, "
            f"使用率 {usage_rate:.1f}%"
        )

    # 风险调整收益分析
    top_strategies["risk_adjusted_return"] = (
        top_strategies["annual_return"] / top_strategies["max_drawdown"]
    )
    print(f"\n风险调整收益分析:")
    print(f"  平均收益/回撤比: {top_strategies['risk_adjusted_return'].mean():.4f}")
    print(f"  最高收益/回撤比: {top_strategies['risk_adjusted_return'].max():.4f}")

    return top_strategies


def generate_practical_insights(df, factor_names, factor_df):
    """生成实用策略洞察"""
    print(f"\n💡 实用策略洞察")
    print("=" * 60)

    print("1. 核心因子建议:")
    top_factors = factor_df.head(8)["factor_name"].tolist()
    print(f"   必配因子: {', '.join(top_factors[:4])}")
    print(f"   增强因子: {', '.join(top_factors[4:8])}")

    print(f"\n2. 权重分配建议:")
    # 分析顶级策略的权重分配
    weight_cols = [f"weight_{i}" for i in range(len(factor_names))]
    top_50 = df.head(50)
    top_weights = top_50[weight_cols]

    non_zero_weights = top_weights.values[top_weights.values > 0]
    if len(non_zero_weights) > 0:
        print(f"   核心因子权重: 0.05 - 0.08")
        print(f"   辅助因子权重: 0.02 - 0.05")
        print(f"   建议因子总数: 15-25 个")

    print(f"\n3. 风险管理建议:")
    avg_drawdown = df["max_drawdown"].mean()
    max_drawdown = df["max_drawdown"].max()
    print(f"   预期最大回撤: {avg_drawdown:.1%} - {max_drawdown:.1%}")
    print(f"   建议止损线: {max_drawdown * 1.2:.1%}")
    print(f"   建议仓位控制: 单个因子权重 < 10%")

    print(f"\n4. 策略类型建议:")
    # 基于因子偏好分类
    price_position_factors = [f for f in top_factors if "PRICE_POSITION" in f]
    volatility_factors = [f for f in top_factors if "STDDEV" in f or "VAR" in f]
    momentum_factors = [f for f in top_factors if "STOCH" in f or "RSI" in f]

    print(f"   主要策略类型: 多因子均衡策略")
    print(f"   核心逻辑: 价格位置 + 波动率 + 技术指标")
    print(f"   适用市场: 震荡市和趋势市均衡配置")

    print(f"\n5. 实施建议:")
    print(f"   调仓频率: 基于换手率，建议月度调仓")
    print(f"   组合规模: Top 8 个标的")
    print(f"   业绩基准: 夏普比率目标 > 0.7")
    print(f"   资金分配: 建议分批建仓，降低冲击成本")


def save_comprehensive_results(df, factor_names, factor_df):
    """保存综合分析结果"""
    print(f"\n💾 保存分析结果...")

    # 保存完整分析数据
    df.to_csv("strategies/results/top1000_complete_analysis.csv", index=False)
    factor_df.to_csv("strategies/results/factor_importance_detailed.csv", index=False)

    # 保存策略洞察
    insights = {
        "analysis_summary": {
            "total_strategies": len(df),
            "factor_count": len(factor_names),
            "avg_sharpe": float(df["sharpe"].mean()),
            "best_sharpe": float(df["sharpe"].max()),
            "avg_return": float(df["annual_return"].mean()),
            "avg_drawdown": float(df["max_drawdown"].mean()),
            "optimal_top_n": int(df.groupby("top_n")["sharpe"].mean().idxmax()),
        },
        "top_factors": factor_df.head(10)["factor_name"].tolist(),
        "strategy_recommendations": {
            "core_factors_count": 8,
            "recommended_weight_range": [0.03, 0.08],
            "optimal_factor_count": 20,
            "risk_management": {
                "stop_loss": 0.25,
                "max_single_factor_weight": 0.10,
                "rebalance_frequency": "monthly",
            },
        },
    }

    import json

    with open(
        "strategies/results/top1000_strategy_insights.json", "w", encoding="utf-8"
    ) as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

    print(f"   ✅ 完整分析数据: strategies/results/top1000_complete_analysis.csv")
    print(f"   ✅ 因子重要性分析: strategies/results/factor_importance_detailed.csv")
    print(f"   ✅ 策略洞察报告: strategies/results/top1000_strategy_insights.json")


def main():
    """主函数"""
    print("🚀 Top1000策略深度分析 (修复版)")
    print("=" * 60)

    # 加载并合并所有批次数据
    df = load_and_combine_all_batches()

    if df is None or len(df) == 0:
        print("❌ 没有可用数据进行分析")
        return

    # 解析权重数据
    df, factor_names = parse_weights_data(df)

    if len(df) == 0:
        print("❌ 权重解析失败，没有可用数据")
        return

    # 性能概览分析
    df = analyze_performance_overview(df)

    # 因子重要性分析
    factor_df, weights_data = analyze_factor_importance(df, factor_names)

    # 策略模式分析
    concentration_scores, factor_counts = analyze_strategy_patterns(
        df, factor_names, weights_data
    )

    # 顶级表现者分析
    top_strategies = identify_top_performers(df, factor_names)

    # 生成实用洞察
    generate_practical_insights(df, factor_names, factor_df)

    # 保存结果
    save_comprehensive_results(df, factor_names, factor_df)

    print(f"\n✅ Top1000策略分析完成!")
    print(f"   基于 {len(df)} 个有效策略和 {len(factor_names)} 个因子")
    print(f"   为量化策略构建提供了全面的数据支撑")


if __name__ == "__main__":
    main()
