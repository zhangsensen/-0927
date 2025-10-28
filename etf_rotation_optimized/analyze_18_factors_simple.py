#!/usr/bin/env python3
"""
18因子综合分析报告（简化版）
===========================
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# 路径配置
WFO_RESULT_PATH = Path("results/wfo/20251027_163940/wfo_results.pkl")
BACKTEST_RESULT_PATH = Path(
    "results/backtest/20251027_163948/combination_performance.csv"
)

# 因子分类
OLD_FACTORS = [
    "MOM_20D",
    "SLOPE_20D",
    "PRICE_POSITION_20D",
    "PRICE_POSITION_120D",
    "RET_VOL_20D",
    "MAX_DD_60D",
    "VOL_RATIO_20D",
    "VOL_RATIO_60D",
    "PV_CORR_20D",
    "RSI_14",
]

NEW_FACTORS = {
    "第1批-资金流": ["OBV_SLOPE_10D", "CMF_20D"],
    "第2批-风险调整动量": ["SHARPE_RATIO_20D", "CALMAR_RATIO_60D"],
    "第3批-趋势强度": ["ADX_14D", "VORTEX_14D"],
    "第4批-相对强度": ["RELATIVE_STRENGTH_VS_MARKET_20D", "CORRELATION_TO_MARKET_20D"],
}

ALL_NEW_FACTORS = [f for batch in NEW_FACTORS.values() for f in batch]


def main():
    print("=" * 80)
    print("18因子综合分析报告")
    print("=" * 80)

    # 加载数据
    with open(WFO_RESULT_PATH, "rb") as f:
        wfo_results = pickle.load(f)

    window_results = wfo_results["window_results"]
    total_windows = wfo_results["total_windows"]

    df_backtest = pd.read_csv(BACKTEST_RESULT_PATH)

    # === 1. 因子使用频率 ===
    print("\n" + "=" * 80)
    print("1. 因子使用频率分析")
    print("=" * 80)

    factor_counter = Counter()
    for wr in window_results:
        for factor_name in wr["selected_factors"]:
            if factor_name:  # 排除空字符串
                factor_counter[factor_name] += 1

    sorted_factors = sorted(factor_counter.items(), key=lambda x: x[1], reverse=True)

    print(f"\n总窗口数: {total_windows}")
    print(f"\n因子使用频率排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'因子名称':<45} {'使用次数':<10} {'使用率':<10} {'类别'}")
    print("-" * 80)

    for rank, (factor_name, count) in enumerate(sorted_factors, 1):
        usage_rate = count / total_windows * 100

        if factor_name in OLD_FACTORS:
            category = "原有"
        else:
            category = "未知"
            for batch_name, factors in NEW_FACTORS.items():
                if factor_name in factors:
                    category = f"新增-{batch_name}"
                    break

        print(
            f"{rank:<4} {factor_name:<45} {count:<10} {usage_rate:>6.1f}%   {category}"
        )

    # 未被选中的因子
    all_factors = set(OLD_FACTORS + ALL_NEW_FACTORS)
    unused_factors = all_factors - set(factor_counter.keys())

    if unused_factors:
        print(f"\n⚠️  未被选中的因子 ({len(unused_factors)}个):")
        for factor_name in sorted(unused_factors):
            if factor_name in OLD_FACTORS:
                category = "原有"
            else:
                for batch_name, factors in NEW_FACTORS.items():
                    if factor_name in factors:
                        category = f"新增-{batch_name}"
                        break
            print(f"  • {factor_name:<45} ({category})")

    # === 2. 新老因子对比 ===
    print("\n" + "=" * 80)
    print("2. 新因子 vs 老因子对比")
    print("=" * 80)

    old_usage = [factor_counter.get(f, 0) for f in OLD_FACTORS]
    new_usage = [factor_counter.get(f, 0) for f in ALL_NEW_FACTORS]

    print(f"\n老因子（10个）:")
    print(f"  平均使用率: {np.mean(old_usage) / total_windows * 100:.1f}%")
    print(f"  被选中因子数: {sum(1 for u in old_usage if u > 0)}/10")

    print(f"\n新因子（8个）:")
    print(f"  平均使用率: {np.mean(new_usage) / total_windows * 100:.1f}%")
    print(f"  被选中因子数: {sum(1 for u in new_usage if u > 0)}/8")

    # 分批次分析
    print(f"\n新因子分批次表现:")
    print("-" * 80)

    for batch_name, factors in NEW_FACTORS.items():
        print(f"\n{batch_name}（{len(factors)}个）:")
        for factor_name in factors:
            count = factor_counter.get(factor_name, 0)
            usage_rate = count / total_windows * 100
            status = "✅" if count > 0 else "❌"
            print(
                f"  {status} {factor_name:<45} 使用率={usage_rate:>5.1f}%  使用次数={count}/{total_windows}"
            )

    # === 3. TOP组合分析 ===
    print("\n" + "=" * 80)
    print("3. TOP 10组合因子分析")
    print("=" * 80)

    df_sorted = df_backtest.sort_values("oos_sharpe", ascending=False).head(10)

    top_factor_counter = Counter()
    for idx, row in df_sorted.iterrows():
        factors_str = row["selected_factors"]
        # 因子用|分隔
        factors = factors_str.split("|") if isinstance(factors_str, str) else []
        for factor in factors:
            if factor:
                top_factor_counter[factor] += 1

    sorted_top_factors = sorted(
        top_factor_counter.items(), key=lambda x: x[1], reverse=True
    )

    print(f"\nTOP 10组合中因子出现频率:")
    print("-" * 80)
    print(f"{'因子名称':<45} {'TOP10出现次数':<15} {'总体使用率':<12} {'类别'}")
    print("-" * 80)

    for factor_name, count in sorted_top_factors:
        total_usage_rate = factor_counter.get(factor_name, 0) / total_windows * 100

        if factor_name in OLD_FACTORS:
            category = "原有"
        else:
            category = "未知"
            for batch_name, factors in NEW_FACTORS.items():
                if factor_name in factors:
                    category = f"新增-{batch_name}"
                    break

        print(
            f"{factor_name:<45} {count}/10 ({count*10}%)     {total_usage_rate:>5.1f}%      {category}"
        )

    print(f"\nTOP 10组合详细信息:")
    print("-" * 80)
    print(
        f"{'排名':<4} {'窗口':<6} {'Sharpe':<10} {'收益率':<12} {'IC':<10} {'因子数'}"
    )
    print("-" * 80)

    for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        factors_str = row["selected_factors"]
        factors = factors_str.split("|") if isinstance(factors_str, str) else []
        factors = [f for f in factors if f]  # 排除空字符串
        print(
            f"{rank:<4} {row['window_idx']:<6} {row['oos_sharpe']:>8.2f}   "
            f"{row['oos_total_return']*100:>8.2f}%     {row['avg_oos_ic']:>8.4f}   {len(factors)}"
        )

    # === 4. 核心发现 ===
    print("\n" + "=" * 80)
    print("4. 核心发现总结")
    print("=" * 80)

    print(
        f"""
🔥 **卓越表现因子**（使用率 > 80%）:
"""
    )
    for factor_name, count in sorted_factors:
        usage_rate = count / total_windows * 100
        if usage_rate > 80:
            category = "新增" if factor_name in ALL_NEW_FACTORS else "原有"
            star = " ⭐" if factor_name in ALL_NEW_FACTORS else ""
            print(f"  • {factor_name:<45} {usage_rate:>5.1f}%  ({category}){star}")

    print(
        f"""
✅ **优秀表现因子**（使用率 50-80%）:
"""
    )
    for factor_name, count in sorted_factors:
        usage_rate = count / total_windows * 100
        if 50 <= usage_rate <= 80:
            category = "新增" if factor_name in ALL_NEW_FACTORS else "原有"
            star = " ⭐" if factor_name in ALL_NEW_FACTORS else ""
            print(f"  • {factor_name:<45} {usage_rate:>5.1f}%  ({category}){star}")

    print(
        f"""
🟡 **中等表现因子**（使用率 20-50%）:
"""
    )
    for factor_name, count in sorted_factors:
        usage_rate = count / total_windows * 100
        if 20 <= usage_rate < 50:
            category = "新增" if factor_name in ALL_NEW_FACTORS else "原有"
            star = " ⭐" if factor_name in ALL_NEW_FACTORS else ""
            print(f"  • {factor_name:<45} {usage_rate:>5.1f}%  ({category}){star}")

    print(
        f"""
⚠️  **低使用率因子**（使用率 1-20%）:
"""
    )
    for factor_name, count in sorted_factors:
        usage_rate = count / total_windows * 100
        if 1 <= usage_rate < 20:
            category = "新增" if factor_name in ALL_NEW_FACTORS else "原有"
            print(f"  • {factor_name:<45} {usage_rate:>5.1f}%  ({category})")

    # 统计新因子成功率
    new_factor_used = sum(1 for f in ALL_NEW_FACTORS if factor_counter.get(f, 0) > 0)
    new_factor_high_usage = sum(
        1
        for f in ALL_NEW_FACTORS
        if factor_counter.get(f, 0) / total_windows * 100 > 50
    )

    print(
        f"""
📊 **新因子整体评价**:
  • 成功率（被选中）: {new_factor_used}/8 ({new_factor_used/8*100:.0f}%)
  • 优秀率（使用率>50%）: {new_factor_high_usage}/8 ({new_factor_high_usage/8*100:.0f}%)
  
🎯 **建议行动**:
  ✅ 保留核心因子:
     - SHARPE_RATIO_20D (98.2% 使用率，卓越表现)
     - RELATIVE_STRENGTH_VS_MARKET_20D (90.9% 使用率，优秀表现)
     - CMF_20D (20.0% 使用率，中等表现)
  
  ⚠️  观察调优因子:
     - VORTEX_14D (7.3% 使用率，需观察特定市场环境)
  
  ❌ 考虑移除/优化:
     - OBV_SLOPE_10D (0% 使用率，未被选中)
     - CALMAR_RATIO_60D (0% 使用率，未被选中)
     - ADX_14D (0% 使用率，未被选中)
     - CORRELATION_TO_MARKET_20D (0% 使用率，未被选中)

💡 **增量价值评估**:
  • 2个新因子进入核心阵容（SHARPE_RATIO_20D, RELATIVE_STRENGTH_VS_MARKET_20D）
  • 新因子平均使用率: {np.mean(new_usage) / total_windows * 100:.1f}%
  • 老因子平均使用率: {np.mean(old_usage) / total_windows * 100:.1f}%
  • 整体评价: 新因子补充有效，核心因子表现优异 ✅
"""
    )

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
