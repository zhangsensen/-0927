#!/usr/bin/env python3
"""
修正后的净收益分析：基于您的技术方案重新计算
"""

import numpy as np
import pandas as pd


def calculate_realistic_turnover_cost(turnover_percent, cost_rate=0.007):
    """
    基于您的方案计算现实的换手成本
    turnover_percent: 显示的换手率百分比 (如 43.05)
    cost_rate: 双边成本率 (默认0.7% = 0.007)
    """
    # 日换手率 = turnover_percent / 100
    daily_turnover = turnover_percent / 100

    # 年化换手率 = 日换手 * 252
    annual_turnover = daily_turnover * 252

    # 年化成本 = 年化换手 * 双边成本
    annual_cost = annual_turnover * cost_rate

    return annual_cost, daily_turnover, annual_turnover


def main():
    print("=== 修正后的净收益分析 ===")

    # 读取数据
    df = pd.read_csv(
        "/Users/zhangshenshen/深度量化0927/strategies/results/top1000_complete_analysis.csv"
    )

    # 以最佳策略62774为例
    best_strategy = df[df["combo_idx"] == 62774].iloc[0]

    print(f"最佳策略62774分析:")
    print(f"  毛收益: {best_strategy['annual_return']:.2%}")
    print(f"  夏普: {best_strategy['sharpe']:.4f}")
    print(f"  换手率: {best_strategy['turnover']:.2f}%")

    # 计算换手成本
    annual_cost, daily_turnover, annual_turnover = calculate_realistic_turnover_cost(
        best_strategy["turnover"], 0.007
    )

    print(f"  日换手率: {daily_turnover:.3f}")
    print(f"  年化换手率: {annual_turnover:.2f}")
    print(f"  年化成本: {annual_cost:.3f} ({annual_cost*100:.2f}%)")

    # 净收益
    net_return = best_strategy["annual_return"] - annual_cost
    print(f"  净收益: {net_return:.3f} ({net_return*100:.2f}%)")

    # 验证您的公式: 0.7% × 0.43 × 252
    manual_calc = 0.007 * (best_strategy["turnover"] / 100) * 252
    print(f"  手工验证: {manual_calc:.3f} ({manual_calc*100:.2f}%)")

    print(f"\n=== 成本敏感性分析 ===")

    # 不同成本率下的净收益
    cost_scenarios = [0.003, 0.005, 0.007, 0.01, 0.015]

    for cost_rate in cost_scenarios:
        annual_cost = cost_rate * (best_strategy["turnover"] / 100) * 252
        net_return = best_strategy["annual_return"] - annual_cost
        net_sharpe = best_strategy["sharpe"] - (
            annual_cost / (best_strategy["annual_return"] / best_strategy["sharpe"])
        )

        print(
            f"  成本率{cost_rate*100:.1f}%: 成本{annual_cost:.3f}, 净收益{net_return:.3f}, 净夏普{net_sharpe:.4f}"
        )

    print(f"\n=== 全样本分析 ===")

    # 计算所有策略的成本
    df["annual_turnover_cost"] = 0.007 * (df["turnover"] / 100) * 252
    df["net_annual_return"] = df["annual_return"] - df["annual_turnover_cost"]
    df["net_sharpe_approx"] = df["sharpe"] * (
        df["net_annual_return"] / df["annual_return"]
    )

    print(
        f"平均年化成本: {df['annual_turnover_cost'].mean():.3f} ({df['annual_turnover_cost'].mean()*100:.2f}%)"
    )
    print(
        f"成本中位数: {df['annual_turnover_cost'].median():.3f} ({df['annual_turnover_cost'].median()*100:.2f}%)"
    )
    print(
        f"成本范围: {df['annual_turnover_cost'].min():.3f} - {df['annual_turnover_cost'].max():.3f}"
    )

    # 排名对比（基于净夏普）
    df["gross_rank"] = df["sharpe"].rank(ascending=False)
    df["net_rank"] = df["net_sharpe_approx"].rank(ascending=False)
    df["rank_change"] = df["gross_rank"] - df["net_rank"]

    # 前10名变化
    gross_top10 = set(df.nsmallest(10, "gross_rank").index)
    net_top10 = set(df.nsmallest(10, "net_rank").index)

    print(f"\n前10名策略变化:")
    print(f"毛收益前10: {gross_top10}")
    print(f"净收益前10: {net_top10}")
    print(f"变化数量: {len(gross_top10 ^ net_top10)}")

    # 保存结果
    output_cols = [
        "combo_idx",
        "annual_return",
        "sharpe",
        "turnover",
        "annual_turnover_cost",
        "net_annual_return",
        "net_sharpe_approx",
        "gross_rank",
        "net_rank",
        "rank_change",
    ]

    df[output_cols].to_csv(
        "/Users/zhangshenshen/深度量化0927/strategies/results/corrected_net_analysis.csv",
        index=False,
    )

    # 显示净收益最佳策略
    best_net = df.loc[df["net_sharpe_approx"].idxmax()]
    print(f"\n净收益最佳策略: {best_net['combo_idx']}")
    print(
        f"  毛夏普: {best_net['sharpe']:.4f} -> 净夏普: {best_net['net_sharpe_approx']:.4f}"
    )
    print(
        f"  换手: {best_net['turnover']:.1f}%, 成本: {best_net['annual_turnover_cost']:.3f}"
    )


if __name__ == "__main__":
    main()
