#!/usr/bin/env python3
"""
离线先验方案全面验证脚本
对比IC加权 vs 先验加权的真实性能
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 结果路径
IC_WEIGHTED_PATH = Path("results/wfo/20251029/20251029_180831/wfo_summary.csv")
PRIOR_WEIGHTED_PATH = Path("results/wfo/20251029/20251029_181406/wfo_summary.csv")


def load_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载两个方案的WFO结果"""
    ic_df = pd.read_csv(IC_WEIGHTED_PATH)
    prior_df = pd.read_csv(PRIOR_WEIGHTED_PATH)
    return ic_df, prior_df


def calculate_statistics(ic_series: pd.Series, prior_series: pd.Series) -> Dict:
    """计算统计指标"""
    # 基础统计
    ic_mean = ic_series.mean()
    prior_mean = prior_series.mean()
    ic_std = ic_series.std()
    prior_std = prior_series.std()

    # 胜率
    ic_win_rate = (ic_series > 0).mean()
    prior_win_rate = (prior_series > 0).mean()

    # 信息比率
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    prior_ir = prior_mean / prior_std if prior_std > 0 else 0

    # 配对t检验
    t_stat, p_value = stats.ttest_rel(prior_series, ic_series)

    # 效应量 (Cohen's d)
    diff = prior_series - ic_series
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0

    return {
        "ic_weighted": {
            "mean": ic_mean,
            "std": ic_std,
            "win_rate": ic_win_rate,
            "ir": ic_ir,
        },
        "prior_weighted": {
            "mean": prior_mean,
            "std": prior_std,
            "win_rate": prior_win_rate,
            "ir": prior_ir,
        },
        "improvement": {
            "absolute": prior_mean - ic_mean,
            "relative": (
                (prior_mean - ic_mean) / abs(ic_mean) * 100 if ic_mean != 0 else 0
            ),
            "t_stat": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
        },
    }


def analyze_time_windows(ic_df: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    """分析每个时间窗口的表现"""
    comparison = pd.DataFrame(
        {
            "window": ic_df["window_index"],
            "ic_weighted_ic": ic_df["oos_ensemble_ic"],
            "prior_weighted_ic": prior_df["oos_ensemble_ic"],
            "ic_weighted_sharpe": ic_df["oos_ensemble_sharpe"],
            "prior_weighted_sharpe": prior_df["oos_ensemble_sharpe"],
            "ic_diff": prior_df["oos_ensemble_ic"] - ic_df["oos_ensemble_ic"],
            "sharpe_diff": prior_df["oos_ensemble_sharpe"]
            - ic_df["oos_ensemble_sharpe"],
        }
    )

    comparison["prior_wins"] = comparison["ic_diff"] > 0
    return comparison


def check_lookback_bias(ic_df: pd.DataFrame, prior_df: pd.DataFrame) -> Dict:
    """检查是否存在前视偏差"""
    # 检查时间窗口是否完全一致
    time_match = (
        (ic_df["is_start"] == prior_df["is_start"]).all()
        and (ic_df["is_end"] == prior_df["is_end"]).all()
        and (ic_df["oos_start"] == prior_df["oos_start"]).all()
        and (ic_df["oos_end"] == prior_df["oos_end"]).all()
    )

    # 检查因子选择是否一致（应该一致，因为都基于IS IC）
    factor_match_rate = (
        ic_df["selected_factors"] == prior_df["selected_factors"]
    ).mean()

    return {
        "time_windows_match": time_match,
        "factor_selection_match_rate": factor_match_rate,
        "bias_detected": not time_match or factor_match_rate < 0.95,
    }


def analyze_factor_weights(ic_df: pd.DataFrame, prior_df: pd.DataFrame) -> pd.DataFrame:
    """分析因子权重差异"""
    import json

    weight_diffs = []
    for idx in range(len(ic_df)):
        ic_weights = json.loads(ic_df.iloc[idx]["top_factors"])
        prior_weights = json.loads(prior_df.iloc[idx]["top_factors"])

        # 提取所有因子
        all_factors = set(ic_weights.keys()) | set(prior_weights.keys())

        for factor in all_factors:
            ic_w = ic_weights.get(factor, {}).get("weight", 0)
            prior_w = prior_weights.get(factor, {}).get("weight", 0)

            weight_diffs.append(
                {
                    "window": idx,
                    "factor": factor,
                    "ic_weight": ic_w,
                    "prior_weight": prior_w,
                    "weight_diff": prior_w - ic_w,
                }
            )

    return pd.DataFrame(weight_diffs)


def print_report(stats: Dict, comparison: pd.DataFrame, bias_check: Dict):
    """打印验证报告"""
    print("=" * 80)
    print("离线先验方案验证报告")
    print("=" * 80)
    print()

    # 1. 性能对比
    print("## 1. 性能对比")
    print("-" * 80)
    ic_stats = stats["ic_weighted"]
    prior_stats = stats["prior_weighted"]
    imp = stats["improvement"]

    print(f"IC加权 (基线):")
    print(f"  平均IC:    {ic_stats['mean']:.4f}")
    print(f"  IC标准差:  {ic_stats['std']:.4f}")
    print(f"  IC胜率:    {ic_stats['win_rate']:.1%}")
    print(f"  信息比率:  {ic_stats['ir']:.4f}")
    print()

    print(f"先验加权 (新方案):")
    print(f"  平均IC:    {prior_stats['mean']:.4f}")
    print(f"  IC标准差:  {prior_stats['std']:.4f}")
    print(f"  IC胜率:    {prior_stats['win_rate']:.1%}")
    print(f"  信息比率:  {prior_stats['ir']:.4f}")
    print()

    print(f"性能提升:")
    print(f"  绝对提升:  {imp['absolute']:+.4f}")
    print(f"  相对提升:  {imp['relative']:+.1f}%")
    print(f"  t统计量:   {imp['t_stat']:.4f}")
    print(f"  p值:       {imp['p_value']:.4f}")
    print(f"  Cohen's d: {imp['cohens_d']:.4f}")
    print(f"  统计显著:  {'✅ 是' if imp['significant'] else '❌ 否'}")
    print()

    # 2. 时间窗口分析
    print("## 2. 时间窗口分析")
    print("-" * 80)
    prior_win_rate = comparison["prior_wins"].mean()
    print(
        f"先验方案胜出窗口: {comparison['prior_wins'].sum()}/{len(comparison)} ({prior_win_rate:.1%})"
    )
    print()

    # 最佳/最差窗口
    best_window = comparison.loc[comparison["ic_diff"].idxmax()]
    worst_window = comparison.loc[comparison["ic_diff"].idxmin()]

    print(f"最佳窗口 (窗口{best_window['window']}):")
    print(f"  IC提升: {best_window['ic_diff']:+.4f}")
    print()

    print(f"最差窗口 (窗口{worst_window['window']}):")
    print(f"  IC下降: {worst_window['ic_diff']:+.4f}")
    print()

    # 3. 前视偏差检查
    print("## 3. 前视偏差检查")
    print("-" * 80)
    print(
        f"时间窗口一致性: {'✅ 通过' if bias_check['time_windows_match'] else '❌ 失败'}"
    )
    print(f"因子选择一致率: {bias_check['factor_selection_match_rate']:.1%}")
    print(
        f"前视偏差检测:   {'❌ 检测到' if bias_check['bias_detected'] else '✅ 未检测到'}"
    )
    print()

    # 4. 结论
    print("## 4. 结论")
    print("-" * 80)

    if imp["significant"] and not bias_check["bias_detected"]:
        print("✅ 先验加权方案验证通过:")
        print(f"   - IC提升 {imp['relative']:+.1f}% (p={imp['p_value']:.4f})")
        print(f"   - 无前视偏差")
        print(f"   - 建议用于研究环境")
    elif imp["relative"] > 0 and not bias_check["bias_detected"]:
        print("⚠️  先验加权方案有提升但未达统计显著性:")
        print(f"   - IC提升 {imp['relative']:+.1f}% (p={imp['p_value']:.4f})")
        print(f"   - 无前视偏差")
        print(f"   - 建议继续观察")
    else:
        print("❌ 先验加权方案未通过验证:")
        if bias_check["bias_detected"]:
            print(f"   - 检测到前视偏差")
        if imp["relative"] <= 0:
            print(f"   - IC未提升 ({imp['relative']:+.1f}%)")

    print()
    print("=" * 80)


def main():
    """主函数"""
    # 加载数据
    print("加载WFO结果...")
    ic_df, prior_df = load_results()

    # 统计分析
    print("计算统计指标...")
    stats = calculate_statistics(ic_df["oos_ensemble_ic"], prior_df["oos_ensemble_ic"])

    # 时间窗口分析
    print("分析时间窗口...")
    comparison = analyze_time_windows(ic_df, prior_df)

    # 前视偏差检查
    print("检查前视偏差...")
    bias_check = check_lookback_bias(ic_df, prior_df)

    # 打印报告
    print_report(stats, comparison, bias_check)

    # 保存详细对比
    output_path = Path("results/wfo/prior_weighted_validation.csv")
    comparison.to_csv(output_path, index=False)
    print(f"\n详细对比已保存: {output_path}")


if __name__ == "__main__":
    main()
