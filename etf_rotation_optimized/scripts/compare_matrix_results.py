#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小矩阵调参结果对比分析
对比Exp7-11的性能指标，找出最优配置
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_experiment(exp_num):
    """加载实验结果"""
    results_dir = Path(__file__).parent.parent / "results" / "wfo"

    # Exp7使用特殊的文件名
    if exp_num == 7:
        pkl_file = results_dir / "exp7_max8_beta08_FIXED.pkl"
    else:
        pkl_file = results_dir / f"exp{exp_num}.pkl"

    if not pkl_file.exists():
        print(f"⚠️  文件不存在: {pkl_file}")
        return None

    with open(pkl_file, "rb") as f:
        return pickle.load(f)


def analyze_dedup_intensity(reports):
    """分析去重力度"""
    dedup_count = 0
    total_removed = 0

    for report in reports:
        for v_str in report.constraint_violations:
            if "correlation_deduplication" in v_str:
                dedup_count += 1
                # 提取移除的因子数量
                import re

                match = re.search(r"\{([^}]+)\}", v_str)
                if match:
                    factors = match.group(1).replace("'", "").split(", ")
                    total_removed += len(factors)
                break

    return dedup_count, total_removed


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  📊 小矩阵调参结果对比分析                                 ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # 实验配置定义
    experiments = {
        7: {"threshold": 0.85, "beta": 0.8, "label": "Exp7 (基线,修复后)"},
        8: {"threshold": 0.88, "beta": 0.0, "label": "Exp8 (th=0.88,β=0)"},
        9: {"threshold": 0.90, "beta": 0.0, "label": "Exp9 (th=0.90,β=0)"},
        10: {"threshold": 0.88, "beta": 0.8, "label": "Exp10 (th=0.88,β=0.8)"},
        11: {"threshold": 0.90, "beta": 0.8, "label": "Exp11 (th=0.90,β=0.8)"},
    }

    # 加载所有实验结果
    data = {}
    for exp_num in experiments.keys():
        result = load_experiment(exp_num)
        if result is not None:
            data[exp_num] = result

    if len(data) == 0:
        print("❌ 没有找到任何实验结果文件！")
        return

    print(f"✅ 成功加载 {len(data)} 个实验结果\n")

    # ===== 1. 核心指标对比 =====
    print("=" * 80)
    print("1️⃣  核心指标对比")
    print("=" * 80)
    print()

    results_table = []
    baseline_ic = None

    for exp_num in sorted(data.keys()):
        exp_data = data[exp_num]
        oos_ics = exp_data["results_df"]["oos_ic_mean"].values

        mean_ic = oos_ics.mean()
        std_ic = oos_ics.std()
        sharpe = mean_ic / std_ic

        if exp_num == 7:
            baseline_ic = mean_ic
            vs_baseline = 0.0
        else:
            vs_baseline = (mean_ic - baseline_ic) / abs(baseline_ic) * 100

        # 去重统计
        dedup_count, total_removed = analyze_dedup_intensity(
            exp_data["constraint_reports"]
        )
        avg_removed = total_removed / max(1, dedup_count)

        results_table.append(
            {
                "Exp": exp_num,
                "Label": experiments[exp_num]["label"],
                "Threshold": experiments[exp_num]["threshold"],
                "Beta": experiments[exp_num]["beta"],
                "OOS IC": mean_ic,
                "IC Std": std_ic,
                "Sharpe": sharpe,
                "vs基线(%)": vs_baseline,
                "去重窗口": dedup_count,
                "平均移除": avg_removed,
            }
        )

    df = pd.DataFrame(results_table)

    # 格式化输出
    print(
        f"{'Exp':<4} {'Threshold':<10} {'Beta':<6} {'OOS IC':<12} {'IC Std':<10} {'Sharpe':<9} {'vs基线':<10} {'去重':<6} {'移除/窗':<8}"
    )
    print("─" * 90)

    for _, row in df.iterrows():
        print(
            f"{int(row['Exp']):<4} "
            f"{row['Threshold']:<10.2f} "
            f"{row['Beta']:<6.1f} "
            f"{row['OOS IC']:<12.6f} "
            f"{row['IC Std']:<10.6f} "
            f"{row['Sharpe']:<9.3f} "
            f"{row['vs基线(%)']:>9.2f}% "
            f"{int(row['去重窗口']):<6} "
            f"{row['平均移除']:<8.2f}"
        )

    # ===== 2. 统计显著性检验 =====
    print("\n" + "=" * 80)
    print("2️⃣  统计显著性检验 (vs Exp7基线)")
    print("=" * 80)
    print()

    baseline_ic_arr = data[7]["results_df"]["oos_ic_mean"].values

    print(f"{'对比':<25} {'t值':<10} {'p值':<10} {'显著性':<15}")
    print("─" * 60)

    for exp_num in sorted(data.keys()):
        if exp_num == 7:
            continue

        test_ic_arr = data[exp_num]["results_df"]["oos_ic_mean"].values
        t_stat, p_value = stats.ttest_rel(test_ic_arr, baseline_ic_arr)

        if p_value < 0.05:
            sig = "✅ 显著" if t_stat > 0 else "❌ 显著更差"
        elif p_value < 0.10:
            sig = "🟡 边缘显著"
        else:
            sig = "⚪ 不显著"

        label = experiments[exp_num]["label"]
        print(f"{label:<25} {t_stat:>9.3f} {p_value:>9.4f} {sig:<15}")

    # ===== 3. 最优配置推荐 =====
    print("\n" + "=" * 80)
    print("3️⃣  最优配置推荐")
    print("=" * 80)
    print()

    # 按夏普比排序
    df_sorted = df.sort_values("Sharpe", ascending=False)

    print("🏆 按夏普比排名:")
    print()
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        medal = (
            "🥇"
            if rank == 1
            else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}. "
        )
        print(f"{medal} {row['Label']}")
        print(
            f"   Sharpe={row['Sharpe']:.3f}, OOS IC={row['OOS IC']:.6f}, "
            f"Std={row['IC Std']:.6f}, 去重={int(row['去重窗口'])}/55"
        )

    # ===== 4. Beta边际效应分析 =====
    print("\n" + "=" * 80)
    print("4️⃣  Beta边际效应分析")
    print("=" * 80)
    print()

    beta_pairs = [
        (8, 10, "threshold=0.88"),
        (9, 11, "threshold=0.90"),
    ]

    print(f"{'场景':<20} {'β=0 IC':<12} {'β=0.8 IC':<12} {'差异':<10} {'结论':<15}")
    print("─" * 70)

    for beta0_exp, beta08_exp, scenario in beta_pairs:
        if beta0_exp in data and beta08_exp in data:
            ic_beta0 = data[beta0_exp]["results_df"]["oos_ic_mean"].values.mean()
            ic_beta08 = data[beta08_exp]["results_df"]["oos_ic_mean"].values.mean()
            diff_pct = (ic_beta08 - ic_beta0) / abs(ic_beta0) * 100

            # t检验
            t_stat, p_value = stats.ttest_rel(
                data[beta08_exp]["results_df"]["oos_ic_mean"].values,
                data[beta0_exp]["results_df"]["oos_ic_mean"].values,
            )

            conclusion = (
                "有效"
                if p_value < 0.10 and diff_pct > 0
                else "微弱" if diff_pct > 0 else "负面"
            )

            print(
                f"{scenario:<20} {ic_beta0:<12.6f} {ic_beta08:<12.6f} {diff_pct:>9.2f}% {conclusion:<15}"
            )

    # ===== 5. 最终建议 =====
    print("\n" + "=" * 80)
    print("🎯  最终建议")
    print("=" * 80)
    print()

    best = df_sorted.iloc[0]
    best_exp = int(best["Exp"])

    print(f"【推荐配置】Exp{best_exp}")
    print(f"  threshold = {best['Threshold']}")
    print(f"  beta = {best['Beta']}")
    print(f"  max_factors = 8")
    print()
    print(f"【性能指标】")
    print(f"  OOS IC: {best['OOS IC']:.6f} (vs基线 {best['vs基线(%)']:+.2f}%)")
    print(f"  夏普比: {best['Sharpe']:.3f}")
    print(f"  IC标准差: {best['IC Std']:.6f}")
    print(f"  去重触发: {int(best['去重窗口'])}/55窗口")
    print()

    # Beta建议
    avg_beta_effect = (
        df[df["Beta"] == 0.8]["OOS IC"].mean() - df[df["Beta"] == 0.0]["OOS IC"].mean()
    )

    if avg_beta_effect < 0.0001:  # <0.01%
        print("【Meta Factor建议】")
        print("  ⚠️  Beta效应微弱 (平均<0.01%)")
        print("  💡 建议禁用Meta Factor，简化系统")
    elif avg_beta_effect > 0:
        print("【Meta Factor建议】")
        print(f"  ✅ Beta有正效应 ({avg_beta_effect*1000:.2f}‰)")
        print("  💡 建议保留Meta Factor作为增强")
    else:
        print("【Meta Factor建议】")
        print("  ❌ Beta有负效应")
        print("  💡 建议禁用Meta Factor")

    print("\n" + "=" * 80)
    print("✅ 分析完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
