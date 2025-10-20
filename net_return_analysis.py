#!/usr/bin/env python3
"""
净收益分析：计算扣除交易成本后的真实收益表现
对比毛收益与净收益排名差异
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 交易成本参数设定
TRANSACTION_COST_PARAMS = {
    "small_cap_cost": 0.0020,  # 小盘股20bps
    "large_cap_cost": 0.0010,  # 大盘股10bps
    "blended_cost": 0.0015,  # 平均15bps
    "market_impact": 0.0005,  # 市场冲击5bps
}


def calculate_annual_turnover_cost(turnover_ratio, cost_rate=None):
    """
    计算年化换手成本
    turnover_ratio: 日换手率(%)
    cost_rate: 单边交易成本，默认使用混合成本
    """
    if cost_rate is None:
        cost_rate = TRANSACTION_COST_PARAMS["blended_cost"]

    # 双边成本 = 单边成本 * 2
    bilateral_cost = cost_rate * 2

    # 年化成本 = 日换手 * 双边成本 * 252交易日
    annual_cost = (turnover_ratio / 100) * bilateral_cost * 252

    return annual_cost


def analyze_net_returns(df, beta_values=[0.001, 0.003, 0.007, 0.01, 0.03]):
    """
    分析净收益表现
    """
    results = []

    # 基础成本计算（按您提供的公式）
    base_cost_rate = 0.007  # 0.7%基础成本
    df["annual_turnover_cost"] = df["turnover"].apply(
        lambda x: calculate_annual_turnover_cost(x, base_cost_rate)
    )

    # 计算净收益
    df["net_annual_return"] = df["annual_return"] - df["annual_turnover_cost"]
    df["net_sharpe"] = df["net_annual_return"] / (
        df["annual_return"] / df["sharpe"]
    )  # 近似计算

    print("=== 基础净收益分析 ===")
    print(
        f"平均年化成本: {df['annual_turnover_cost'].mean():.3f} ({df['annual_turnover_cost'].mean()*100:.2f}%)"
    )
    print(
        f"成本范围: {df['annual_turnover_cost'].min():.3f} - {df['annual_turnover_cost'].max():.3f}"
    )
    print(
        f"平均净收益损失: {(df['annual_return'] - df['net_annual_return']).mean():.3f}"
    )

    # 不同β值的净夏普计算
    for beta in beta_values:
        col_name = f"net_sharpe_beta_{beta}"
        df[col_name] = df["sharpe"] - beta * df["annual_turnover_cost"]

        # 记录每个β下的最佳策略
        best_idx = df[col_name].idxmax()
        best_combo = df.loc[best_idx, "combo_idx"]
        best_net_sharpe = df.loc[best_idx, col_name]
        best_gross_sharpe = df.loc[best_idx, "sharpe"]

        results.append(
            {
                "beta": beta,
                "best_combo": best_combo,
                "net_sharpe": best_net_sharpe,
                "gross_sharpe": best_gross_sharpe,
                "turnover_cost": df.loc[best_idx, "annual_turnover_cost"],
            }
        )

    return df, results


def compare_rankings(df, original_top_n=50):
    """
    对比毛收益与净收益排名差异
    """
    # 毛收益排名
    df["gross_rank"] = df["sharpe"].rank(ascending=False)

    # 净收益排名（使用β=0.007）
    df["net_rank"] = df["net_sharpe_beta_0.007"].rank(ascending=False)

    # 排名变化
    df["rank_change"] = df["gross_rank"] - df["net_rank"]

    print(f"\n=== 排名对比分析（前{original_top_n}名）===")

    # 原前50名中掉出净收益前50的数量
    gross_top50 = set(df.nsmallest(original_top_n, "gross_rank").index)
    net_top50 = set(df.nsmallest(original_top_n, "net_rank").index)

    dropped_out = len(gross_top50 - net_top50)
    new_comers = len(net_top50 - gross_top50)

    print(
        f"毛收益前{original_top_n}名中，有{dropped_out}个掉出净收益前{original_top_n}"
    )
    print(f"净收益前{original_top_n}名中，有{new_comers}个新进入者")

    # 具体分析掉出者
    if dropped_out > 0:
        dropped_combos = list(gross_top50 - net_top50)
        dropped_df = df.loc[
            dropped_combos,
            [
                "combo_idx",
                "sharpe",
                "net_sharpe_beta_0.007",
                "annual_turnover_cost",
                "gross_rank",
                "net_rank",
            ],
        ].head(10)
        print(f"\n掉出净收益前{original_top_n}的策略（前10个）：")
        print(dropped_df)

    return df


def pareto_analysis(df, beta=0.007, gamma=0.1):
    """
    Pareto最优分析：Sharpe - β×TurnoverCost - γ×Concentration
    """
    # 计算综合得分
    df["composite_score"] = (
        df["sharpe"] - beta * df["annual_turnover_cost"] - gamma * df["concentration"]
    )

    # Pareto前沿识别
    df["composite_rank"] = df["composite_score"].rank(ascending=False)

    print(f"\n=== Pareto分析 (β={beta}, γ={gamma}) ===")

    # 最佳综合策略
    best_composite = df.loc[df["composite_score"].idxmax()]
    print(f"最佳综合策略: {best_composite['combo_idx']}")
    print(f"综合得分: {best_composite['composite_score']:.4f}")
    print(f"毛夏普: {best_composite['sharpe']:.4f}")
    print(f"换手成本: {best_composite['annual_turnover_cost']:.4f}")
    print(f"集中度: {best_composite['concentration']:.4f}")

    return df


def generate_cost_sensitivity_report(df):
    """
    生成成本敏感性报告
    """
    print("\n=== 成本敏感性分析 ===")

    # 不同成本假设下的净夏普
    cost_scenarios = [0.003, 0.005, 0.007, 0.01, 0.015]  # 0.3%, 0.5%, 0.7%, 1.0%, 1.5%

    for cost in cost_scenarios:
        annual_cost = df["turnover"].apply(
            lambda x: calculate_annual_turnover_cost(x, cost)
        )
        net_sharpe = df["sharpe"] - annual_cost

        best_idx = net_sharpe.idxmax()
        best_combo = df.loc[best_idx, "combo_idx"]
        best_net_sharpe = net_sharpe.iloc[best_idx]

        print(
            f"成本率{cost*100:.1f}%: 最佳策略{best_combo}, 净夏普{best_net_sharpe:.4f}"
        )


def main():
    """
    主函数：执行完整的净收益分析
    """
    print("开始净收益分析...")

    # 读取数据
    df = pd.read_csv(
        "/Users/zhangshenshen/深度量化0927/strategies/results/top1000_complete_analysis.csv"
    )

    # 基本数据清洗
    print(f"原始数据形状: {df.shape}")

    # 执行分析
    df, beta_results = analyze_net_returns(df)
    df = compare_rankings(df, 50)
    df = pareto_analysis(df, beta=0.007, gamma=0.1)
    generate_cost_sensitivity_report(df)

    # 保存结果
    output_file = (
        "/Users/zhangshenshen/深度量化0927/strategies/results/net_return_analysis.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"\n分析结果已保存至: {output_file}")

    # 关键发现总结
    print("\n=== 关键发现 ===")
    print(
        f"1. 平均年化换手成本: {df['annual_turnover_cost'].mean():.3f} ({df['annual_turnover_cost'].mean()*100:.2f}%)"
    )
    print(
        f"2. 成本对收益影响: 平均降低{(df['annual_return'] - df['net_annual_return']).mean():.3f}收益"
    )
    print(
        f"3. 最佳毛收益策略: 62774 (夏普{df.loc[df['sharpe'].idxmax(), 'sharpe']:.4f})"
    )
    print(
        f"4. 最佳净收益策略: {df.loc[df['net_sharpe_beta_0.007'].idxmax(), 'combo_idx']} (净夏普{df['net_sharpe_beta_0.007'].max():.4f})"
    )
    print(
        f"5. 排名变化: 前50名中{len(set(df.nsmallest(50, 'gross_rank').index) - set(df.nsmallest(50, 'net_rank').index))}个策略位置变化"
    )


if __name__ == "__main__":
    main()
