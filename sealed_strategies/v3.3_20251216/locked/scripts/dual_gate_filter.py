#!/usr/bin/env python3
"""
双重挡板+约束筛选脚本 v1.0
================================================================================
基于深度过拟合诊断结果，应用以下筛选规则：

1. 双重挡板: 训练期合格 ∩ Holdout期合格
2. 回撤约束: MaxDD < 阈值
3. 复杂度约束: 组合阶数 ≤ 阈值
4. 因子黑名单: 禁止过拟合因子(ADX_14D等)
5. 因子白名单: 优先稳定因子(MAX_DD_60D, CMF_20D等)

使用方法:
    python scripts/dual_gate_filter.py \
        --train_pct 0.70 \
        --hold_pct 0.80 \
        --max_dd 0.15 \
        --max_size 5 \
        --top_n 50
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent


def load_data():
    """加载数据"""
    df_full = pd.read_csv(
        ROOT / "results/vec_from_wfo_20251211_205649/full_space_results.csv"
    )
    df_hold = pd.read_csv(ROOT / "results/vec_from_wfo_20251211_205649/all_holdout.csv")

    # 合并
    df = df_full.merge(df_hold, on="combo", suffixes=("_train", "_hold"))

    # 计算综合得分
    df["train_composite"] = (
        0.4 * df["vec_return"]
        + 0.3 * df["vec_sharpe_ratio"]
        - 0.3 * df["vec_max_drawdown"]
    )

    df["hold_composite"] = (
        0.4 * df["hold_return"] + 0.3 * df["hold_sharpe"] - 0.3 * df["hold_max_dd"]
    )

    return df


def apply_filters(
    df,
    train_pct=0.70,
    hold_pct=0.80,
    max_dd=0.15,
    max_size=5,
    blacklist=None,
    whitelist=None,
    require_all_whitelist=False,
):
    """应用筛选条件"""

    print("=" * 80)
    print("🔍 应用筛选条件")
    print("=" * 80)

    print(f"\n初始组合数: {len(df)}")

    # 1. 双重挡板
    train_threshold = df["train_composite"].quantile(train_pct)
    hold_threshold = df["hold_composite"].quantile(hold_pct)

    mask = (df["train_composite"] >= train_threshold) & (
        df["hold_composite"] >= hold_threshold
    )
    df_filtered = df[mask].copy()
    print(
        f"✓ 双重挡板 (训练>{train_pct:.0%} ∩ Holdout>{hold_pct:.0%}): {len(df_filtered)} 个"
    )

    # 2. 回撤约束
    mask = df_filtered["hold_max_dd"] <= max_dd
    df_filtered = df_filtered[mask]
    print(f"✓ 回撤约束 (MaxDD ≤ {max_dd:.1%}): {len(df_filtered)} 个")

    # 3. 复杂度约束
    mask = df_filtered["size_train"] <= max_size
    df_filtered = df_filtered[mask]
    print(f"✓ 复杂度约束 (阶数 ≤ {max_size}): {len(df_filtered)} 个")

    # 4. 黑名单
    if blacklist:

        def has_blacklisted(combo):
            factors = [f.strip() for f in combo.split(" + ")]
            return any(f in blacklist for f in factors)

        mask = ~df_filtered["combo"].apply(has_blacklisted)
        df_filtered = df_filtered[mask]
        print(f"✓ 黑名单过滤 (禁止: {', '.join(blacklist)}): {len(df_filtered)} 个")

    # 5. 白名单
    if whitelist:

        def has_all_whitelisted(combo):
            factors = [f.strip() for f in combo.split(" + ")]
            return all(f in factors for f in whitelist)

        def has_any_whitelisted(combo):
            factors = [f.strip() for f in combo.split(" + ")]
            return any(f in whitelist for f in factors)

        if require_all_whitelist:
            mask = df_filtered["combo"].apply(has_all_whitelisted)
            print(
                f"✓ 白名单过滤 (必含全部: {', '.join(whitelist)}): {len(df_filtered[mask])} 个"
            )
        else:
            mask = df_filtered["combo"].apply(has_any_whitelisted)
            print(
                f"✓ 白名单过滤 (至少一个: {', '.join(whitelist)}): {len(df_filtered[mask])} 个"
            )

        df_filtered = df_filtered[mask]

    return df_filtered


def summarize_results(df):
    """总结筛选结果"""
    print("\n" + "=" * 80)
    print("📊 筛选结果汇总")
    print("=" * 80)

    print(f"\n【整体统计】")
    print(f"  通过组合数: {len(df)}")
    print(
        f"  训练期收益: 均值={df['vec_return'].mean():.2%}, 中位={df['vec_return'].median():.2%}"
    )
    print(
        f"  Holdout收益: 均值={df['hold_return'].mean():.2%}, 中位={df['hold_return'].median():.2%}"
    )
    print(
        f"  Holdout Sharpe: 均值={df['hold_sharpe'].mean():.4f}, 中位={df['hold_sharpe'].median():.4f}"
    )
    print(
        f"  Holdout MaxDD: 均值={df['hold_max_dd'].mean():.2%}, 中位={df['hold_max_dd'].median():.2%}"
    )

    print(f"\n【阶数分布】")
    size_dist = df["size_train"].value_counts().sort_index()
    for size, count in size_dist.items():
        print(f"  {size}因子组合: {count:4d} ({count/len(df):.1%})")

    print(f"\n【因子频率 (Top10)】")
    factor_counter = Counter()
    for combo in df["combo"]:
        factors = [f.strip() for f in combo.split(" + ")]
        factor_counter.update(factors)

    for factor, count in factor_counter.most_common(10):
        print(f"  {factor:40} {count:4d} ({count/len(df):.1%})")


def display_top_n(df, n=20):
    """显示TopN组合"""
    print("\n" + "=" * 80)
    print(f"🏆 Top{n} 组合 (按Holdout综合分排序)")
    print("=" * 80)

    # 按Holdout综合分排序
    df_sorted = df.sort_values("hold_composite", ascending=False)

    print(
        f"\n{'排名':>4} {'Holdout收益':>12} {'Holdout Sharpe':>14} {'Holdout MaxDD':>14} {'阶数':>6} 组合"
    )
    print("-" * 130)

    for idx, row in df_sorted.head(n).iterrows():
        print(
            f"{idx+1:4d} {row['hold_return']:+11.2%} {row['hold_sharpe']:13.4f} {row['hold_max_dd']:13.2%} {int(row['size_train']):6d} {row['combo']}"
        )


def main():
    parser = argparse.ArgumentParser(description="双重挡板+约束筛选")
    parser.add_argument(
        "--train_pct", type=float, default=0.70, help="训练期分位数阈值"
    )
    parser.add_argument(
        "--hold_pct", type=float, default=0.80, help="Holdout期分位数阈值"
    )
    parser.add_argument("--max_dd", type=float, default=0.15, help="最大回撤阈值")
    parser.add_argument("--max_size", type=int, default=5, help="最大组合阶数")
    parser.add_argument("--top_n", type=int, default=50, help="输出TopN")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")

    args = parser.parse_args()

    print("=" * 80)
    print("🔬 双重挡板+约束筛选 v1.0")
    print("=" * 80)

    # 加载数据
    print("\n📂 加载数据...")
    df = load_data()
    print(f"✅ 加载完成: {len(df)} 组合")

    # 定义黑名单和白名单
    blacklist = [
        "ADX_14D",  # 训练75.7% → Holdout 6.0% (失效)
        # 'SHARPE_RATIO_20D',  # 可选，但衰减严重
        # 'MOM_20D',  # 可选，但衰减严重
    ]

    whitelist = [
        "MAX_DD_60D",  # Holdout Top500中88.2%包含
    ]

    # 应用筛选
    df_filtered = apply_filters(
        df,
        train_pct=args.train_pct,
        hold_pct=args.hold_pct,
        max_dd=args.max_dd,
        max_size=args.max_size,
        blacklist=blacklist,
        whitelist=whitelist,
        require_all_whitelist=False,  # 至少包含一个白名单因子
    )

    if len(df_filtered) == 0:
        print("\n⚠️ 没有符合条件的组合，建议放宽约束")
        return

    # 总结结果
    summarize_results(df_filtered)

    # 显示TopN
    display_top_n(df_filtered, n=args.top_n)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            ROOT
            / f"results/vec_from_wfo_20251211_205649/filtered_top{args.top_n}_for_bt.csv"
        )

    # 按Holdout综合分排序后保存TopN
    df_sorted = df_filtered.sort_values("hold_composite", ascending=False)
    df_output = df_sorted.head(args.top_n)
    df_output.to_csv(output_path, index=False)

    print(f"\n💾 已保存至: {output_path}")
    print(f"   共 {len(df_output)} 个组合")

    print("\n" + "=" * 80)
    print("✅ 筛选完成")
    print("=" * 80)

    print("\n📋 下一步:")
    print(f"  1. 检查输出文件: {output_path}")
    print(
        f"  2. 执行BT小规模审计: uv run python scripts/batch_bt_backtest.py --input {output_path}"
    )
    print(f"  3. 对比VEC/BT结果，确认对齐")


if __name__ == "__main__":
    main()
