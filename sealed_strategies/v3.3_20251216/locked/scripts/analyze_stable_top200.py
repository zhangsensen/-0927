#!/usr/bin/env python3
"""
深度分析双稳定 Top 200 策略
找出为什么这些策略在训练集和 Holdout 都表现优异
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 读取 Holdout 验证结果
    results_dir = Path("/home/sensen/dev/projects/-0927/results")
    latest_validation = sorted(results_dir.glob("holdout_validation_*"))[-1]
    results_path = latest_validation / "holdout_validation_results.csv"

    print("=" * 80)
    print("🔬 深度分析双稳定 Top 200 策略")
    print("=" * 80)
    print(f"数据源: {results_path}")

    df = pd.read_csv(results_path)

    # 按双稳定得分排序
    df_sorted = df.sort_values("calmar_ratio_stability", ascending=False)

    # 取 Top 200
    top200 = df_sorted.head(200)

    print(f"\n总策略数: {len(df)}")
    print(f"分析样本: Top 200 双稳定策略")

    # ========================================================================
    # 1. 整体表现统计
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 1. 整体表现统计")
    print("=" * 80)

    print("\n训练集表现:")
    print(f"  平均 Calmar:     {top200['vec_calmar_ratio'].mean():.3f}")
    print(f"  中位数 Calmar:   {top200['vec_calmar_ratio'].median():.3f}")
    print(f"  平均收益率:      {top200['vec_return'].mean()*100:.2f}%")
    print(f"  平均最大回撤:    {top200['vec_max_drawdown'].mean()*100:.2f}%")
    print(f"  平均 Sharpe:     {top200['vec_sharpe_ratio'].mean():.3f}")

    print("\nHoldout 表现:")
    print(f"  平均 Calmar:     {top200['holdout_calmar_ratio'].mean():.3f}")
    print(f"  中位数 Calmar:   {top200['holdout_calmar_ratio'].median():.3f}")
    print(f"  平均收益率:      {top200['holdout_return'].mean()*100:.2f}%")
    print(f"  平均最大回撤:    {top200['holdout_max_drawdown'].mean()*100:.2f}%")
    print(f"  平均 Sharpe:     {top200['holdout_sharpe_ratio'].mean():.3f}")

    print("\n稳定性分析:")
    print(f"  双稳定得分均值:  {top200['calmar_ratio_stability'].mean():.3f}")
    print(f"  双稳定得分中位数: {top200['calmar_ratio_stability'].median():.3f}")

    # Holdout / Train 比值
    calmar_ratio_change = top200["holdout_calmar_ratio"] / top200["vec_calmar_ratio"]
    print(f"\nHoldout/训练集 Calmar 比值:")
    print(f"  平均: {calmar_ratio_change.mean():.2f}x")
    print(f"  中位数: {calmar_ratio_change.median():.2f}x")
    print(
        f"  >1.0 (Holdout更好): {(calmar_ratio_change > 1.0).sum()} / 200 ({(calmar_ratio_change > 1.0).mean()*100:.1f}%)"
    )
    print(
        f"  0.8-1.2 (稳定): {((calmar_ratio_change >= 0.8) & (calmar_ratio_change <= 1.2)).sum()} / 200"
    )

    # ========================================================================
    # 2. 因子频率分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("🧬 2. 因子频率分析 (Top 200)")
    print("=" * 80)

    # 统计每个因子在 Top 200 中的出现次数
    factor_counts = Counter()
    for combo in top200["combo"]:
        factors = [f.strip() for f in combo.split(" + ")]
        factor_counts.update(factors)

    print("\n因子出现频率 (降序):")
    print(f"{'排名':<4} | {'因子':<40} | {'出现次数':<8} | {'出现率':<8}")
    print("-" * 80)
    for i, (factor, count) in enumerate(factor_counts.most_common(), 1):
        print(f"{i:<4} | {factor:<40} | {count:<8} | {count/2:.1f}%")

    # 核心因子（出现率 > 50%）
    core_factors = [f for f, c in factor_counts.items() if c > 100]
    print(f"\n核心因子 (出现率>50%): {len(core_factors)} 个")
    for f in core_factors:
        print(f"  - {f}: {factor_counts[f]} 次 ({factor_counts[f]/2:.1f}%)")

    # ========================================================================
    # 3. 组合大小分布
    # ========================================================================
    print("\n" + "=" * 80)
    print("📐 3. 组合大小分布")
    print("=" * 80)

    top200["combo_size"] = top200["combo"].apply(lambda x: len(x.split(" + ")))
    size_dist = top200["combo_size"].value_counts().sort_index()

    print("\n组合大小统计:")
    print(f"{'大小':<6} | {'数量':<6} | {'占比':<8} | {'平均稳定得分':<15}")
    print("-" * 50)
    for size in sorted(size_dist.index):
        count = size_dist[size]
        avg_stable = top200[top200["combo_size"] == size][
            "calmar_ratio_stability"
        ].mean()
        print(f"{size:<6} | {count:<6} | {count/2:.1f}%{'':<4} | {avg_stable:.3f}")

    print(f"\n平均组合大小: {top200['combo_size'].mean():.2f}")
    print(f"中位数组合大小: {top200['combo_size'].median():.0f}")

    # ========================================================================
    # 4. 因子共现分析（找出最强组合）
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔗 4. 因子共现分析 (Top 10 高频因子对)")
    print("=" * 80)

    # 统计因子对
    factor_pairs = Counter()
    for combo in top200["combo"]:
        factors = sorted([f.strip() for f in combo.split(" + ")])
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                factor_pairs[(factors[i], factors[j])] += 1

    print("\n最常见的因子组合:")
    print(f"{'排名':<4} | {'因子对':<80} | {'共现次数':<8}")
    print("-" * 100)
    for i, (pair, count) in enumerate(factor_pairs.most_common(10), 1):
        print(f"{i:<4} | {pair[0]:<35} + {pair[1]:<35} | {count:<8}")

    # ========================================================================
    # 5. 按表现分组分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 5. 按表现分组分析")
    print("=" * 80)

    # 分组：Top 1-50, 51-100, 101-150, 151-200
    groups = [
        ("Top 1-50", top200.iloc[0:50]),
        ("Top 51-100", top200.iloc[50:100]),
        ("Top 101-150", top200.iloc[100:150]),
        ("Top 151-200", top200.iloc[150:200]),
    ]

    for group_name, group_df in groups:
        print(f"\n{group_name}:")
        print(f"  训练集 Calmar: {group_df['vec_calmar_ratio'].mean():.3f}")
        print(f"  Holdout Calmar: {group_df['holdout_calmar_ratio'].mean():.3f}")
        print(f"  平均组合大小: {group_df['combo_size'].mean():.2f}")

        # 该组最常见因子
        group_factors = Counter()
        for combo in group_df["combo"]:
            factors = [f.strip() for f in combo.split(" + ")]
            group_factors.update(factors)
        top3_factors = group_factors.most_common(3)
        print(f"  核心因子: {', '.join([f'{f} ({c})' for f, c in top3_factors])}")

    # ========================================================================
    # 6. 识别"毒药因子"（在 Top 200 中罕见但在全样本中常见）
    # ========================================================================
    print("\n" + "=" * 80)
    print('☠️  6. "毒药因子"识别')
    print("=" * 80)

    # 统计全样本的因子频率
    all_factors = Counter()
    for combo in df["combo"]:
        factors = [f.strip() for f in combo.split(" + ")]
        all_factors.update(factors)

    # 对比 Top 200 vs 全样本
    print("\n因子在 Top 200 vs 全样本中的出现率对比:")
    print(f"{'因子':<40} | {'Top200率':<10} | {'全样本率':<10} | {'差异':<10}")
    print("-" * 80)

    all_factor_names = set(all_factors.keys())
    for factor in sorted(all_factor_names):
        top200_rate = factor_counts.get(factor, 0) / 200 * 100
        all_rate = all_factors[factor] / len(df) * 100
        diff = top200_rate - all_rate

        # 只显示差异较大的（可能是优质或毒药）
        if abs(diff) > 20:
            marker = "⭐" if diff > 0 else "☠️ "
            print(
                f"{marker} {factor:<38} | {top200_rate:>8.1f}% | {all_rate:>9.1f}% | {diff:>+9.1f}%"
            )

    # ========================================================================
    # 7. 保存 Top 200 列表供 BT 审计
    # ========================================================================
    print("\n" + "=" * 80)
    print("💾 7. 保存结果")
    print("=" * 80)

    output_dir = Path("/home/sensen/dev/projects/-0927/results/stable_top200_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 Top 200 详情
    top200_output = top200[
        [
            "combo",
            "combo_size",
            "vec_calmar_ratio",
            "holdout_calmar_ratio",
            "calmar_ratio_stability",
            "vec_return",
            "holdout_return",
            "vec_max_drawdown",
            "holdout_max_drawdown",
        ]
    ].copy()
    top200_output.to_csv(output_dir / "top200_stable_strategies.csv", index=False)
    print(f"✅ Top 200 策略列表: {output_dir / 'top200_stable_strategies.csv'}")

    # 保存因子统计
    factor_stats = pd.DataFrame(
        [
            {"factor": f, "count": c, "frequency": c / 2}
            for f, c in factor_counts.most_common()
        ]
    )
    factor_stats.to_csv(output_dir / "factor_frequency.csv", index=False)
    print(f"✅ 因子频率统计: {output_dir / 'factor_frequency.csv'}")

    # 保存因子对统计
    pair_stats = pd.DataFrame(
        [
            {"factor1": pair[0], "factor2": pair[1], "count": count}
            for pair, count in factor_pairs.most_common(50)
        ]
    )
    pair_stats.to_csv(output_dir / "factor_pairs.csv", index=False)
    print(f"✅ 因子对统计: {output_dir / 'factor_pairs.csv'}")

    # ========================================================================
    # 8. 可视化分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 8. 生成可视化图表")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 8.1 组合大小分布
    ax = axes[0, 0]
    size_dist.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 200 组合大小分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("组合大小 (因子数量)", fontsize=12)
    ax.set_ylabel("策略数量", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # 8.2 Top 15 因子频率
    ax = axes[0, 1]
    top15_factors = factor_counts.most_common(15)
    factor_names = [
        f[0].replace("_", "\n") if len(f[0]) > 20 else f[0] for f in top15_factors
    ]
    factor_values = [f[1] for f in top15_factors]
    ax.barh(factor_names, factor_values, color="coral")
    ax.set_title("Top 15 因子出现频率", fontsize=14, fontweight="bold")
    ax.set_xlabel("出现次数", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 8.3 训练集 vs Holdout Calmar 散点图
    ax = axes[1, 0]
    ax.scatter(
        top200["vec_calmar_ratio"],
        top200["holdout_calmar_ratio"],
        alpha=0.6,
        c=top200["combo_size"],
        cmap="viridis",
        s=50,
    )
    ax.plot([0, 3], [0, 3], "r--", alpha=0.5, label="y=x")
    ax.set_title("训练集 vs Holdout Calmar 对比", fontsize=14, fontweight="bold")
    ax.set_xlabel("训练集 Calmar", fontsize=12)
    ax.set_ylabel("Holdout Calmar", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("组合大小", fontsize=10)

    # 8.4 稳定性得分分布
    ax = axes[1, 1]
    ax.hist(
        top200["calmar_ratio_stability"],
        bins=30,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(
        top200["calmar_ratio_stability"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'中位数: {top200["calmar_ratio_stability"].median():.3f}',
    )
    ax.set_title("双稳定得分分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("稳定得分 (min(Train, Holdout) Calmar)", fontsize=12)
    ax.set_ylabel("策略数量", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "top200_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"✅ 可视化图表: {fig_path}")

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print(f"\n📁 所有结果保存在: {output_dir}")
    print("\n下一步:")
    print("  1. 查看因子频率统计，确定核心因子")
    print("  2. 运行 BT 审计验证 Top 200 策略")
    print("  3. 提取最优因子组合规律")


if __name__ == "__main__":
    main()
