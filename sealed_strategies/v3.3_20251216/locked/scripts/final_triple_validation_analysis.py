#!/usr/bin/env python3
"""
综合分析 Top 200 双稳定策略的 WFO/VEC/BT 三重验证结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


def main():
    root = Path("/home/sensen/dev/projects/-0927")

    # 读取三个数据源
    print("=" * 80)
    print("📊 加载 WFO/VEC/BT 三重验证数据")
    print("=" * 80)

    # VEC + Holdout 结果
    vec_path = root / "results/stable_top200_analysis/top200_stable_strategies.csv"
    df_vec = pd.read_csv(vec_path)
    print(f"✅ VEC 数据: {len(df_vec)} 个策略")

    # BT 审计结果
    bt_dirs = sorted((root / "results").glob("bt_backtest_top200_*"))
    if not bt_dirs:
        print("❌ 未找到 BT 审计结果")
        return

    bt_dir = bt_dirs[-1]
    bt_path = bt_dir / "bt_results.csv"
    df_bt = pd.read_csv(bt_path)
    print(f"✅ BT 数据: {len(df_bt)} 个策略 from {bt_dir.name}")

    # 合并数据
    df_merged = df_vec.merge(df_bt, on="combo", how="inner", suffixes=("_vec", "_bt"))
    print(f"✅ 合并后: {len(df_merged)} 个策略\n")

    # ========================================================================
    # 1. VEC vs BT 对齐验证
    # ========================================================================
    print("=" * 80)
    print("🔍 1. VEC vs BT 对齐验证（训练集期）")
    print("=" * 80)

    # 计算训练集期的对齐差异
    df_merged["train_return_diff"] = abs(
        df_merged["vec_return"] - df_merged["bt_return"]
    )
    df_merged["train_mdd_diff"] = abs(
        df_merged["vec_max_drawdown"] - df_merged["bt_max_drawdown"]
    )

    print(f"\n收益率对齐:")
    print(f"  平均差异: {df_merged['train_return_diff'].mean()*100:.3f}%")
    print(f"  中位数差异: {df_merged['train_return_diff'].median()*100:.3f}%")
    print(f"  最大差异: {df_merged['train_return_diff'].max()*100:.3f}%")
    print(
        f"  <0.5% 差异: {(df_merged['train_return_diff'] < 0.005).sum()} / {len(df_merged)} ({(df_merged['train_return_diff'] < 0.005).mean()*100:.1f}%)"
    )

    print(f"\n最大回撤对齐:")
    print(f"  平均差异: {df_merged['train_mdd_diff'].mean()*100:.3f}%")
    print(f"  中位数差异: {df_merged['train_mdd_diff'].median()*100:.3f}%")
    print(f"  最大差异: {df_merged['train_mdd_diff'].max()*100:.3f}%")
    print(
        f"  <0.5% 差异: {(df_merged['train_mdd_diff'] < 0.005).sum()} / {len(df_merged)} ({(df_merged['train_mdd_diff'] < 0.005).mean()*100:.1f}%)"
    )

    # ========================================================================
    # 2. 三重验证一致性分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 2. 三重验证一致性排名")
    print("=" * 80)

    # 计算每个策略的综合排名
    df_merged["rank_vec_train"] = df_merged["vec_calmar_ratio"].rank(ascending=False)
    df_merged["rank_holdout"] = df_merged["holdout_calmar_ratio"].rank(ascending=False)
    df_merged["rank_bt"] = df_merged["bt_calmar_ratio"].rank(ascending=False)

    # 综合稳定性得分（三期最小 Calmar）
    df_merged["triple_stable_score"] = df_merged[
        ["vec_calmar_ratio", "holdout_calmar_ratio", "bt_calmar_ratio"]
    ].min(axis=1)
    df_merged = df_merged.sort_values("triple_stable_score", ascending=False)

    print("\n🏆 三重验证 Top 20 (按最小 Calmar 排序):")
    print(
        f"{'排名':<4} | {'训练Calmar':<11} | {'Holdout':<11} | {'BT':<11} | {'最小值':<8} | {'组合'}"
    )
    print("-" * 100)

    for i, (_, row) in enumerate(df_merged.head(20).iterrows(), 1):
        print(
            f"{i:<4} | {row['vec_calmar_ratio']:>10.3f} | {row['holdout_calmar_ratio']:>10.3f} | "
            f"{row['bt_calmar_ratio']:>10.3f} | {row['triple_stable_score']:>7.3f} | {row['combo'][:60]}"
        )

    # ========================================================================
    # 3. 因子稳定性分析
    # ========================================================================
    print("\n" + "=" * 80)
    print("🧬 3. 三重验证 Top 50 因子分析")
    print("=" * 80)

    top50 = df_merged.head(50)

    factor_counts = Counter()
    for combo in top50["combo"]:
        factors = [f.strip() for f in combo.split(" + ")]
        factor_counts.update(factors)

    print("\n因子出现频率 (Top 50 策略):")
    print(f"{'排名':<4} | {'因子':<40} | {'次数':<6} | {'占比':<8}")
    print("-" * 70)
    for i, (factor, count) in enumerate(factor_counts.most_common(10), 1):
        print(f"{i:<4} | {factor:<40} | {count:<6} | {count/0.5:.1f}%")

    # ========================================================================
    # 4. 过拟合检测
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔬 4. 过拟合检测（Calmar 比值分析）")
    print("=" * 80)

    # 训练集 vs Holdout
    train_holdout_ratio = (
        df_merged["holdout_calmar_ratio"] / df_merged["vec_calmar_ratio"]
    )
    # 训练集 vs BT
    train_bt_ratio = df_merged["bt_calmar_ratio"] / df_merged["vec_calmar_ratio"]

    print("\n训练集 vs Holdout:")
    print(f"  平均比值: {train_holdout_ratio.mean():.3f}")
    print(f"  中位数比值: {train_holdout_ratio.median():.3f}")
    print(
        f"  0.8-1.2 (稳定): {((train_holdout_ratio >= 0.8) & (train_holdout_ratio <= 1.2)).sum()} / {len(df_merged)}"
    )

    print("\n训练集(VEC) vs 训练集(BT):")
    print(f"  平均比值: {train_bt_ratio.mean():.3f}")
    print(f"  中位数比值: {train_bt_ratio.median():.3f}")
    print(
        f"  0.95-1.05 (高度对齐): {((train_bt_ratio >= 0.95) & (train_bt_ratio <= 1.05)).sum()} / {len(df_merged)}"
    )

    # ========================================================================
    # 5. 最优策略推荐
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎯 5. 最优策略推荐")
    print("=" * 80)

    # 推荐标准：三期都表现优异
    top1 = df_merged.iloc[0]

    print(f"\n【推荐策略 #1】")
    print(f"因子组合: {top1['combo']}")
    print(f"\n三重验证表现:")
    print(
        f"  训练集(VEC):  Calmar={top1['vec_calmar_ratio']:.3f}, Return={top1['vec_return']*100:.2f}%, MDD={top1['vec_max_drawdown']*100:.2f}%"
    )
    print(
        f"  Holdout:      Calmar={top1['holdout_calmar_ratio']:.3f}, Return={top1['holdout_return']*100:.2f}%, MDD={top1['holdout_max_drawdown']*100:.2f}%"
    )
    print(
        f"  训练集(BT):   Calmar={top1['bt_calmar_ratio']:.3f}, Return={top1['bt_return']*100:.2f}%, MDD={top1['bt_max_drawdown']*100:.2f}%"
    )
    print(f"\n稳定性评价:")
    print(f"  最小 Calmar: {top1['triple_stable_score']:.3f}")
    print(
        f"  VEC vs BT 收益差异: {abs(top1['vec_return'] - top1['bt_return'])*100:.3f}%"
    )
    print(f"  综合排名: 1 / {len(df_merged)}")

    # ========================================================================
    # 6. 保存最终结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("💾 6. 保存最终结果")
    print("=" * 80)

    output_dir = root / "results/final_triple_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整对比
    df_output = df_merged[
        [
            "combo",
            "combo_size",
            "vec_calmar_ratio",
            "vec_return",
            "vec_max_drawdown",
            "holdout_calmar_ratio",
            "holdout_return",
            "holdout_max_drawdown",
            "bt_calmar_ratio",
            "bt_return",
            "bt_max_drawdown",
            "triple_stable_score",
            "train_return_diff",
            "train_mdd_diff",
        ]
    ].copy()

    df_output = df_output.sort_values("triple_stable_score", ascending=False)
    output_path = output_dir / "triple_validation_results.csv"
    df_output.to_csv(output_path, index=False)
    print(f"✅ 三重验证结果: {output_path}")

    # 保存 Top 10 详细报告
    top10_path = output_dir / "top10_recommendation.csv"
    df_output.head(10).to_csv(top10_path, index=False)
    print(f"✅ Top 10 推荐: {top10_path}")

    print("\n" + "=" * 80)
    print("✅ 三重验证完成！")
    print("=" * 80)
    print(f"\n核心结论:")
    print(
        f"  1. VEC vs BT 平均对齐差异: {df_merged['train_return_diff'].mean()*100:.3f}% (收益率)"
    )
    print(f"  2. 三重稳定策略数量: {len(df_merged)} 个")
    print(f"  3. Top 1 最小 Calmar: {top1['triple_stable_score']:.3f}")
    print(f"  4. 核心因子: {', '.join([f[0] for f in factor_counts.most_common(3)])}")
    print(f"\n📁 所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
