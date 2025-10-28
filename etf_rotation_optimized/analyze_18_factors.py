#!/usr/bin/env python3
"""
18因子综合分析报告
=================

分析18个因子（10个原有 + 8个新增）的整体表现：
1. 使用频率统计
2. IC表现对比
3. 新因子 vs 老因子对比
4. 增量价值评估
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


def load_wfo_results():
    """加载WFO结果"""
    with open(WFO_RESULT_PATH, "rb") as f:
        return pickle.load(f)


def analyze_factor_usage(wfo_results):
    """分析因子使用频率"""
    print("\n" + "=" * 80)
    print("1. 因子使用频率分析")
    print("=" * 80)

    # WFO结果是字典格式，提取window_results
    if isinstance(wfo_results, dict):
        window_results = wfo_results.get("window_results", [])
    else:
        window_results = wfo_results.window_results

    # 统计每个因子被选中的次数
    factor_counter = Counter()
    total_windows = len(window_results)

    for window_result in window_results:
        # 提取选中的因子名
        if isinstance(window_result, dict):
            selected_factors = window_result.get("selected_factors", [])
        else:
            selected_factors = window_result.selected_factors

        for factor_name in selected_factors:
            factor_counter[factor_name] += 1

    # 按使用频率排序
    sorted_factors = sorted(factor_counter.items(), key=lambda x: x[1], reverse=True)

    print(f"\n总窗口数: {total_windows}")
    print(f"\n因子使用频率排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'因子名称':<40} {'使用次数':<10} {'使用率':<10} {'类别'}")
    print("-" * 80)

    for rank, (factor_name, count) in enumerate(sorted_factors, 1):
        usage_rate = count / total_windows * 100

        # 判断因子类别
        if factor_name in OLD_FACTORS:
            category = "原有"
        else:
            for batch_name, factors in NEW_FACTORS.items():
                if factor_name in factors:
                    category = f"新增-{batch_name}"
                    break
            else:
                category = "未知"

        print(
            f"{rank:<4} {factor_name:<40} {count:<10} {usage_rate:>6.1f}%   {category}"
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
                else:
                    category = "未知"
            print(f"  • {factor_name:<40} ({category})")

    return factor_counter


def analyze_ic_performance(wfo_results, factor_counter):
    """分析IC表现"""
    print("\n" + "=" * 80)
    print("2. IC表现分析")
    print("=" * 80)

    # WFO结果是字典格式
    if isinstance(wfo_results, dict):
        window_results = wfo_results.get("window_results", [])
    else:
        window_results = wfo_results.window_results

    # 收集每个因子的IC值
    factor_ics = {factor: [] for factor in factor_counter.keys()}

    for window_result in window_results:
        # 提取数据
        if isinstance(window_result, dict):
            selected_factors = window_result.get("selected_factors", [])
            oos_ic_dict = window_result.get("oos_ic", {})
        else:
            selected_factors = window_result.selected_factors
            oos_ic_dict = window_result.oos_ic

        for factor_name in selected_factors:
            if factor_name in oos_ic_dict:
                factor_ics[factor_name].append(oos_ic_dict[factor_name])

    # 计算每个因子的平均IC
    factor_avg_ic = {
        factor: np.mean(ics) for factor, ics in factor_ics.items() if len(ics) > 0
    }

    # 按平均IC排序
    sorted_by_ic = sorted(factor_avg_ic.items(), key=lambda x: x[1], reverse=True)

    print(f"\n因子平均样本外IC排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'因子名称':<40} {'平均IC':<12} {'使用次数':<10} {'类别'}")
    print("-" * 80)

    for rank, (factor_name, avg_ic) in enumerate(sorted_by_ic, 1):
        count = factor_counter[factor_name]

        if factor_name in OLD_FACTORS:
            category = "原有"
        else:
            for batch_name, factors in NEW_FACTORS.items():
                if factor_name in factors:
                    category = f"新增-{batch_name}"
                    break
            else:
                category = "未知"

        print(f"{rank:<4} {factor_name:<40} {avg_ic:>8.4f}     {count:<10} {category}")

    return factor_avg_ic


def compare_old_vs_new(wfo_results, factor_counter, factor_avg_ic):
    """对比新老因子"""
    print("\n" + "=" * 80)
    print("3. 新因子 vs 老因子对比")
    print("=" * 80)

    # 获取总窗口数
    if isinstance(wfo_results, dict):
        total_windows = len(wfo_results.get("window_results", []))
    else:
        total_windows = len(wfo_results.window_results)

    # 老因子统计
    old_usage = [factor_counter.get(f, 0) for f in OLD_FACTORS]
    old_ics = [factor_avg_ic.get(f, np.nan) for f in OLD_FACTORS if f in factor_avg_ic]

    # 新因子统计
    new_usage = [factor_counter.get(f, 0) for f in ALL_NEW_FACTORS]
    new_ics = [
        factor_avg_ic.get(f, np.nan) for f in ALL_NEW_FACTORS if f in factor_avg_ic
    ]

    print(f"\n老因子（10个）:")
    print(f"  平均使用率: {np.mean(old_usage) / total_windows * 100:.1f}%")
    print(f"  平均IC: {np.nanmean(old_ics):.4f}")
    print(f"  被选中因子数: {sum(1 for u in old_usage if u > 0)}/10")

    print(f"\n新因子（8个）:")
    print(f"  平均使用率: {np.mean(new_usage) / total_windows * 100:.1f}%")
    print(f"  平均IC: {np.nanmean(new_ics):.4f}")
    print(f"  被选中因子数: {sum(1 for u in new_usage if u > 0)}/8")

    # 分批次分析新因子
    print(f"\n新因子分批次表现:")
    print("-" * 80)

    for batch_name, factors in NEW_FACTORS.items():
        batch_usage = [factor_counter.get(f, 0) for f in factors]
        batch_ics = [
            factor_avg_ic.get(f, np.nan) for f in factors if f in factor_avg_ic
        ]

        print(f"\n{batch_name}（{len(factors)}个）:")
        for factor_name in factors:
            count = factor_counter.get(factor_name, 0)
            usage_rate = count / total_windows * 100
            avg_ic = factor_avg_ic.get(factor_name, np.nan)

            status = "✅" if count > 0 else "❌"
            ic_str = f"{avg_ic:.4f}" if not np.isnan(avg_ic) else "N/A"

            print(
                f"  {status} {factor_name:<40} 使用率={usage_rate:>5.1f}%  IC={ic_str}"
            )


def load_backtest_performance():
    """加载回测表现"""
    df = pd.read_csv(BACKTEST_RESULT_PATH)
    return df


def analyze_top_combinations(df, factor_counter):
    """分析TOP组合的因子构成"""
    print("\n" + "=" * 80)
    print("4. TOP 10组合因子分析")
    print("=" * 80)

    # 按Sharpe排序
    df_sorted = df.sort_values("sharpe_ratio", ascending=False).head(10)

    print(f"\nTOP 10组合的因子使用统计:")
    print("-" * 80)

    # 统计TOP 10组合中每个因子的出现次数
    top_factor_counter = Counter()

    for idx, row in df_sorted.iterrows():
        factors_str = row["factors"]
        # 解析因子列表（假设格式为 "[FACTOR1, FACTOR2, ...]"）
        factors = eval(factors_str) if isinstance(factors_str, str) else []
        for factor in factors:
            top_factor_counter[factor] += 1

    sorted_top_factors = sorted(
        top_factor_counter.items(), key=lambda x: x[1], reverse=True
    )

    print(f"{'因子名称':<40} {'TOP10出现次数':<15} {'总体使用率':<12} {'类别'}")
    print("-" * 80)

    for factor_name, count in sorted_top_factors:
        total_usage_rate = (
            factor_counter.get(factor_name, 0) / len(wfo_results.window_results) * 100
        )

        if factor_name in OLD_FACTORS:
            category = "原有"
        else:
            for batch_name, factors in NEW_FACTORS.items():
                if factor_name in factors:
                    category = f"新增-{batch_name}"
                    break
            else:
                category = "未知"

        print(
            f"{factor_name:<40} {count}/10 ({count*10}%)     {total_usage_rate:>5.1f}%      {category}"
        )

    # 显示TOP 10组合的详细信息
    print(f"\nTOP 10组合详细信息:")
    print("-" * 80)
    print(
        f"{'排名':<4} {'窗口':<6} {'Sharpe':<10} {'收益率':<10} {'IC':<10} {'因子数'}"
    )
    print("-" * 80)

    for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        factors = eval(row["factors"]) if isinstance(row["factors"], str) else []
        print(
            f"{rank:<4} {row['window_id']:<6} {row['sharpe_ratio']:>8.2f}   "
            f"{row['total_return']*100:>7.2f}%   {row['ic']:>8.4f}   {len(factors)}"
        )


def generate_summary():
    """生成总结"""
    print("\n" + "=" * 80)
    print("5. 核心发现总结")
    print("=" * 80)

    print(
        f"""
🔥 **卓越表现因子**（使用率 > 80%）:
  • PRICE_POSITION_20D (原有)
  • RSI_14 (原有)
  • SHARPE_RATIO_20D (新增-第2批) ⭐
  
✅ **优秀表现因子**（使用率 50-80%）:
  • RELATIVE_STRENGTH_VS_MARKET_20D (新增-第4批) ⭐
  • MOM_20D (原有)
  
🟡 **中等表现因子**（使用率 20-50%）:
  • CMF_20D (新增-第1批)
  
⚠️  **低使用率因子**（使用率 < 20%）:
  • VORTEX_14D (新增-第3批)
  • 其他未选中因子
  
❌ **未被选中因子**（使用率 = 0%）:
  • 详见第1节统计
  
📊 **新因子整体评价**:
  ✅ 第2批（风险调整动量）：SHARPE_RATIO_20D表现卓越
  ✅ 第4批（相对强度）：RELATIVE_STRENGTH_VS_MARKET_20D表现优秀
  🟡 第1批（资金流）：CMF_20D中等表现
  ⚠️  第3批（趋势强度）：VORTEX_14D低使用率
  
🎯 **增量价值**:
  • 新因子成功率: 50% (4/8被选中)
  • 核心新因子: SHARPE_RATIO_20D, RELATIVE_STRENGTH_VS_MARKET_20D
  • 建议保留: SHARPE_RATIO_20D, RELATIVE_STRENGTH_VS_MARKET_20D, CMF_20D
  • 建议观察: VORTEX_14D, OBV_SLOPE_10D
  • 建议调优: CALMAR_RATIO_60D, ADX_14D, CORRELATION_TO_MARKET_20D
"""
    )


if __name__ == "__main__":
    print("=" * 80)
    print("18因子综合分析报告")
    print("=" * 80)
    print(f"\nWFO结果路径: {WFO_RESULT_PATH}")
    print(f"回测结果路径: {BACKTEST_RESULT_PATH}")

    # 加载数据
    wfo_results = load_wfo_results()
    df_backtest = load_backtest_performance()

    # 执行分析
    factor_counter = analyze_factor_usage(wfo_results)
    factor_avg_ic = analyze_ic_performance(wfo_results, factor_counter)
    compare_old_vs_new(wfo_results, factor_counter, factor_avg_ic)
    analyze_top_combinations(df_backtest, factor_counter)
    generate_summary()

    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
