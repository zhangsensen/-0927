#!/usr/bin/env python3
"""
Go/No-Go决策分析
基于用户定义的生产门槛判断是否继续投入
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 用户定义的Go/No-Go标准
CRITERIA = {
    "statistical": {
        "p_threshold": 0.10,
        "min_positive_periods": 3,
        "total_periods": 4,
    },
    "robustness": {
        "min_win_rate": 0.60,
        "max_loss_asymmetry": 0.50,  # 差窗口幅度 < 好窗口幅度的一半
    },
    "practical": {
        "min_improvement_pct": 10.0,  # 最小提升百分比
        "min_negative_ic_improvement": 0.70,  # 负IC窗口最小改善率
    },
}


def load_data():
    """加载验证数据"""
    return pd.read_csv("results/wfo/prior_weighted_validation.csv")


def check_statistical_criteria(df: pd.DataFrame) -> dict:
    """检查统计标准"""
    ic_series = df["ic_weighted_ic"]
    prior_series = df["prior_weighted_ic"]

    # Wilcoxon检验
    diff = prior_series - ic_series
    _, p_wilcoxon = stats.wilcoxon(diff, alternative="greater")

    # 分时期分析（4个时期）
    n = len(df)
    period_size = n // 4
    periods = []

    for i in range(4):
        start = i * period_size
        end = (i + 1) * period_size if i < 3 else n
        period_df = df.iloc[start:end]

        period_ic_mean = period_df["ic_weighted_ic"].mean()
        period_prior_mean = period_df["prior_weighted_ic"].mean()
        period_improvement = period_prior_mean - period_ic_mean

        periods.append(
            {
                "period": i,
                "ic_mean": period_ic_mean,
                "prior_mean": period_prior_mean,
                "improvement": period_improvement,
                "positive": period_improvement > 0.0015,
            }
        )

    periods_df = pd.DataFrame(periods)
    n_positive_periods = periods_df["positive"].sum()

    # 判断
    pass_wilcoxon = p_wilcoxon < CRITERIA["statistical"]["p_threshold"]
    pass_periods = n_positive_periods >= CRITERIA["statistical"]["min_positive_periods"]

    return {
        "p_wilcoxon": p_wilcoxon,
        "pass_wilcoxon": pass_wilcoxon,
        "n_positive_periods": n_positive_periods,
        "pass_periods": pass_periods,
        "periods": periods_df,
        "pass": pass_wilcoxon or pass_periods,
    }


def check_robustness_criteria(df: pd.DataFrame) -> dict:
    """检查稳健性标准"""
    # 胜率
    win_rate = df["prior_wins"].mean()

    # 正负不对称
    positive_windows = df[df["ic_diff"] > 0]
    negative_windows = df[df["ic_diff"] < 0]

    avg_positive = (
        positive_windows["ic_diff"].mean() if len(positive_windows) > 0 else 0
    )
    avg_negative = (
        abs(negative_windows["ic_diff"].mean()) if len(negative_windows) > 0 else 0
    )

    loss_asymmetry = avg_negative / avg_positive if avg_positive > 0 else 1.0

    # 判断
    pass_win_rate = win_rate >= CRITERIA["robustness"]["min_win_rate"]
    pass_asymmetry = loss_asymmetry <= CRITERIA["robustness"]["max_loss_asymmetry"]

    return {
        "win_rate": win_rate,
        "pass_win_rate": pass_win_rate,
        "avg_positive": avg_positive,
        "avg_negative": avg_negative,
        "loss_asymmetry": loss_asymmetry,
        "pass_asymmetry": pass_asymmetry,
        "pass": pass_win_rate and pass_asymmetry,
    }


def check_practical_criteria(df: pd.DataFrame) -> dict:
    """检查实用性标准"""
    ic_mean = df["ic_weighted_ic"].mean()
    prior_mean = df["prior_weighted_ic"].mean()
    improvement_pct = (prior_mean - ic_mean) / abs(ic_mean) * 100 if ic_mean != 0 else 0

    # 负IC窗口改善
    negative_ic_windows = df[df["ic_weighted_ic"] < 0]
    if len(negative_ic_windows) > 0:
        negative_ic_improvement = negative_ic_windows["prior_wins"].mean()
    else:
        negative_ic_improvement = 0.0

    # 判断
    pass_improvement = improvement_pct >= CRITERIA["practical"]["min_improvement_pct"]
    pass_negative_ic = (
        negative_ic_improvement >= CRITERIA["practical"]["min_negative_ic_improvement"]
    )

    return {
        "improvement_pct": improvement_pct,
        "pass_improvement": pass_improvement,
        "negative_ic_improvement": negative_ic_improvement,
        "pass_negative_ic": pass_negative_ic,
        "pass": pass_improvement or pass_negative_ic,
    }


def make_decision(statistical: dict, robustness: dict, practical: dict) -> dict:
    """综合决策"""
    # 计算通过的标准数
    criteria_passed = sum(
        [
            statistical["pass"],
            robustness["pass"],
            practical["pass"],
        ]
    )

    # 决策逻辑
    if criteria_passed >= 2:
        decision = "GO"
        reason = f"通过{criteria_passed}/3项标准，建议继续投入优化"
        next_steps = [
            "实现家族收缩先验",
            "实现自适应混合（基于IS质量）",
            "实盘映射测试（成本后PnL）",
        ]
    elif criteria_passed == 1:
        decision = "CONDITIONAL_GO"
        reason = "仅通过1项标准，建议低成本快速测试"
        next_steps = [
            "测试纯稳定性先验（去掉强度）",
            "如果p<0.10，再投入家族收缩",
            "否则暂停，等待更多窗口数据",
        ]
    else:
        decision = "NO_GO"
        reason = "未通过任何标准，不建议继续投入"
        next_steps = [
            "保留为研究分支",
            "等待更多窗口数据（至少60窗口）",
            "探索其他方向（如动态因子选择）",
        ]

    return {
        "decision": decision,
        "criteria_passed": criteria_passed,
        "reason": reason,
        "next_steps": next_steps,
    }


def print_report(statistical: dict, robustness: dict, practical: dict, decision: dict):
    """打印决策报告"""
    print("=" * 80)
    print("Go/No-Go 决策分析")
    print("=" * 80)
    print()

    # 1. 统计标准
    print("## 1. 统计标准")
    print("-" * 80)
    print(
        f"Wilcoxon p值:      {statistical['p_wilcoxon']:.4f} (阈值<{CRITERIA['statistical']['p_threshold']})"
    )
    print(f"通过:              {'✅' if statistical['pass_wilcoxon'] else '❌'}")
    print()
    print(
        f"正向时期数:        {statistical['n_positive_periods']}/{CRITERIA['statistical']['total_periods']} (需≥{CRITERIA['statistical']['min_positive_periods']})"
    )
    print(f"通过:              {'✅' if statistical['pass_periods'] else '❌'}")
    print()
    print("时期分析:")
    print(statistical["periods"].to_string(index=False))
    print()
    print(f"**统计标准**: {'✅ 通过' if statistical['pass'] else '❌ 未通过'}")
    print()

    # 2. 稳健性标准
    print("## 2. 稳健性标准")
    print("-" * 80)
    print(
        f"胜率:              {robustness['win_rate']:.1%} (需≥{CRITERIA['robustness']['min_win_rate']:.0%})"
    )
    print(f"通过:              {'✅' if robustness['pass_win_rate'] else '❌'}")
    print()
    print(f"好窗口平均幅度:    {robustness['avg_positive']:.4f}")
    print(f"差窗口平均幅度:    {robustness['avg_negative']:.4f}")
    print(
        f"损失不对称比:      {robustness['loss_asymmetry']:.2f} (需≤{CRITERIA['robustness']['max_loss_asymmetry']:.2f})"
    )
    print(f"通过:              {'✅' if robustness['pass_asymmetry'] else '❌'}")
    print()
    print(f"**稳健性标准**: {'✅ 通过' if robustness['pass'] else '❌ 未通过'}")
    print()

    # 3. 实用性标准
    print("## 3. 实用性标准")
    print("-" * 80)
    print(
        f"IC提升:            {practical['improvement_pct']:+.1f}% (需≥{CRITERIA['practical']['min_improvement_pct']:.0f}%)"
    )
    print(f"通过:              {'✅' if practical['pass_improvement'] else '❌'}")
    print()
    print(
        f"负IC窗口改善率:    {practical['negative_ic_improvement']:.1%} (需≥{CRITERIA['practical']['min_negative_ic_improvement']:.0%})"
    )
    print(f"通过:              {'✅' if practical['pass_negative_ic'] else '❌'}")
    print()
    print(f"**实用性标准**: {'✅ 通过' if practical['pass'] else '❌ 未通过'}")
    print()

    # 4. 最终决策
    print("=" * 80)
    print("最终决策")
    print("=" * 80)
    print()
    print(f"通过标准: {decision['criteria_passed']}/3")
    print()

    if decision["decision"] == "GO":
        print("🟢 **GO**: " + decision["reason"])
    elif decision["decision"] == "CONDITIONAL_GO":
        print("🟡 **CONDITIONAL GO**: " + decision["reason"])
    else:
        print("🔴 **NO GO**: " + decision["reason"])

    print()
    print("下一步行动:")
    for i, step in enumerate(decision["next_steps"], 1):
        print(f"  {i}. {step}")

    print()
    print("=" * 80)


def main():
    """主函数"""
    df = load_data()

    # 检查各项标准
    statistical = check_statistical_criteria(df)
    robustness = check_robustness_criteria(df)
    practical = check_practical_criteria(df)

    # 综合决策
    decision = make_decision(statistical, robustness, practical)

    # 打印报告
    print_report(statistical, robustness, practical, decision)

    return decision


if __name__ == "__main__":
    decision = main()
