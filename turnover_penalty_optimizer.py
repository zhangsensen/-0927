#!/usr/bin/env python3
"""
基于真实成本发现，重新设计换手惩罚优化器
目标：找到在扣除现实成本后仍有正收益的策略
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_net_metrics(df, cost_rate=0.007):
    """计算扣除成本后的真实指标"""
    df = df.copy()

    # 年化换手成本
    df["annual_cost"] = cost_rate * (df["turnover"] / 100) * 252

    # 净收益
    df["net_return"] = df["annual_return"] - df["annual_cost"]

    # 净夏普（简化计算）
    df["net_sharpe"] = np.where(
        df["net_return"] > 0,
        df["sharpe"] * (df["net_return"] / df["annual_return"]),
        -abs(df["sharpe"]) * abs(df["net_return"]) / abs(df["annual_return"]),
    )

    # 净Calmar
    df["net_calmar"] = np.where(
        df["net_return"] > 0,
        df["net_return"] / df["max_drawdown"],
        df["net_return"] / df["max_drawdown"],
    )

    return df


def find_positive_net_return_strategies(df, min_net_return=0.02):
    """找到净收益为正的策略"""
    positive_strategies = df[df["net_return"] >= min_net_return].copy()
    positive_strategies = positive_strategies.sort_values("net_sharpe", ascending=False)

    print(f"净收益≥{min_net_return*100:.1f}%的策略: {len(positive_strategies)}个")

    if len(positive_strategies) > 0:
        print("\n前10名净收益策略:")
        cols = [
            "combo_idx",
            "annual_return",
            "annual_cost",
            "net_return",
            "net_sharpe",
            "turnover",
            "sharpe",
        ]
        print(positive_strategies[cols].head(10))

    return positive_strategies


def beta_optimization_pareto(df, beta_range=[0.001, 0.003, 0.005, 0.007, 0.01]):
    """β参数优化与Pareto前沿分析"""

    pareto_results = []

    print("=== β参数优化分析 ===")

    for beta in beta_range:
        # 计算综合得分：Sharpe - β×成本
        df[f"score_beta_{beta}"] = df["sharpe"] - beta * df["annual_cost"]

        # 找到最佳策略
        best_idx = df[f"score_beta_{beta}"].idxmax()
        best_strategy = df.loc[best_idx]

        pareto_results.append(
            {
                "beta": beta,
                "best_combo": best_strategy["combo_idx"],
                "score": best_strategy[f"score_beta_{beta}"],
                "gross_sharpe": best_strategy["sharpe"],
                "net_return": best_strategy["net_return"],
                "turnover": best_strategy["turnover"],
                "annual_cost": best_strategy["annual_cost"],
            }
        )

        print(
            f"β={beta}: 最佳{best_strategy['combo_idx']}, "
            f"综合得分{best_strategy[f'score_beta_{beta}']:.4f}, "
            f"毛夏普{best_strategy['sharpe']:.4f}, "
            f"净收益{best_strategy['net_return']:.3f}"
        )

    return pd.DataFrame(pareto_results)


def weight_constrained_optimization(df, max_weight=0.1, cost_rate=0.007):
    """权重上限约束优化"""

    print(f"\n=== 权重上限{max_weight}约束分析 ===")

    # 筛选集中度不超过上限的策略
    constrained_df = df[df["concentration"] <= max_weight].copy()

    if len(constrained_df) == 0:
        print(f"没有找到集中度≤{max_weight}的策略")
        return None

    # 计算净指标
    constrained_df = calculate_net_metrics(constrained_df, cost_rate)

    # 按净夏普排序
    constrained_df = constrained_df.sort_values("net_sharpe", ascending=False)

    print(f"集中度≤{max_weight}的策略: {len(constrained_df)}个")
    print(f"最佳净夏普: {constrained_df.iloc[0]['net_sharpe']:.4f}")
    print(f"最佳策略: {constrained_df.iloc[0]['combo_idx']}")
    print(f"净收益: {constrained_df.iloc[0]['net_return']:.3f}")
    print(f"换手率: {constrained_df.iloc[0]['turnover']:.1f}%")

    return constrained_df


def realistic_strategy_selection(df):
    """
    现实策略选择：考虑可执行性
    """
    print("\n=== 现实策略选择标准 ===")

    # 筛选标准
    criteria = {
        "min_net_return": 0.02,  # 净收益≥2%
        "max_turnover": 50.0,  # 换手率≤50%
        "max_concentration": 0.06,  # 集中度≤6%
        "min_sharpe": 0.6,  # 毛夏普≥0.6
        "max_cost": 0.9,  # 成本≤0.9%
    }

    realistic_df = df[
        (df["net_return"] >= criteria["min_net_return"])
        & (df["turnover"] <= criteria["max_turnover"])
        & (df["concentration"] <= criteria["max_concentration"])
        & (df["sharpe"] >= criteria["min_sharpe"])
        & (df["annual_cost"] <= criteria["max_cost"])
    ].copy()

    realistic_df = realistic_df.sort_values("net_sharpe", ascending=False)

    print(f"符合现实标准的策略: {len(realistic_df)}个")

    if len(realistic_df) > 0:
        print("\n前5名现实可行策略:")
        cols = [
            "combo_idx",
            "net_return",
            "net_sharpe",
            "turnover",
            "concentration",
            "annual_cost",
            "sharpe",
        ]
        print(realistic_df[cols].head())

        # 与原始最佳策略对比
        original_best = df.loc[df["sharpe"].idxmax()]
        realistic_best = realistic_df.iloc[0]

        print(f"\n对比分析:")
        print(
            f"原始最佳(62774): 毛夏普{original_best['sharpe']:.4f}, 净收益{original_best['net_return']:.3f}"
        )
        print(
            f"现实最佳({realistic_best['combo_idx']}): 净夏普{realistic_best['net_sharpe']:.4f}, 净收益{realistic_best['net_return']:.3f}"
        )

    return realistic_df


def main():
    """主函数：执行完整的换手惩罚优化分析"""

    print("=== 换手惩罚优化器分析 ===")

    # 读取数据
    df = pd.read_csv(
        "/Users/zhangshenshen/深度量化0927/strategies/results/top1000_complete_analysis.csv"
    )

    # 计算净指标
    df = calculate_net_metrics(df)

    # 1. 寻找正净收益策略
    positive_strategies = find_positive_net_return_strategies(df, min_net_return=0.01)

    # 2. β参数优化
    beta_results = beta_optimization_pareto(df)

    # 3. 权重约束优化
    weight_constrained = weight_constrained_optimization(df, max_weight=0.1)

    # 4. 现实策略选择
    realistic_strategies = realistic_strategy_selection(df)

    # 保存结果
    if len(positive_strategies) > 0:
        positive_strategies.to_csv(
            "/Users/zhangshenshen/深度量化0927/strategies/results/positive_net_strategies.csv",
            index=False,
        )
        print(f"\n正净收益策略已保存: {len(positive_strategies)}个")

    beta_results.to_csv(
        "/Users/zhangshenshen/深度量化0927/strategies/results/beta_optimization_results.csv",
        index=False,
    )

    if realistic_strategies is not None and len(realistic_strategies) > 0:
        realistic_strategies.to_csv(
            "/Users/zhangshenshen/深度量化0927/strategies/results/realistic_strategies.csv",
            index=False,
        )
        print(f"现实可行策略已保存: {len(realistic_strategies)}个")

    # 关键建议
    print("\n=== 核心建议 ===")
    print("1. 当前高换手策略在扣除现实成本后全部为负收益")
    print("2. 必须大幅降低换手率至35%以下才能获得正净收益")
    print("3. 建议β值设定在0.005-0.007区间，重点惩罚高换手")
    print("4. 权重上限0.1约束有助于降低集中度和成本")
    print("5. 现实可行策略稀缺，说明需要重新设计低换手策略")


if __name__ == "__main__":
    main()
