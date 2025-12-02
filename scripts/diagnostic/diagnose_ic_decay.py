#!/usr/bin/env python3
"""
IC衰减诊断工具
WFO IC vs OOS IC: 0.173 -> 0.0156，衰减91%，问题根源分析
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent


def load_wfo_results():
    """加载WFO结果"""
    import sys

    wfo_dir = PROJECT_ROOT / "etf_strategy/results/wfo/20251027_170352"
    with open(wfo_dir / "wfo_results.pkl", "rb") as f:
        return pickle.load(f)


def load_backtest_results(ts="20251027_174911"):
    """加载回测结果"""
    backtest_dir = PROJECT_ROOT / f"etf_strategy/results/backtest/{ts}"
    return pd.read_csv(backtest_dir / "combination_performance.csv")


def diagnose_ic_decay():
    """
    诊断 IC 衰减
    """
    print("=" * 80)
    print("IC衰减诊断分析")
    print("=" * 80)
    print()

    wfo = load_wfo_results()
    backtest = load_backtest_results()

    print("1. 高层数据")
    print("-" * 80)
    print(f"WFO 平均 IC (来自WFO优化过程): {wfo['avg_oos_ic']:.6f}")
    print(f"实际回测 IC (跨54窗口平均):     {backtest['avg_oos_ic'].mean():.6f}")
    print(
        f"衰减倍数: {(1 - backtest['avg_oos_ic'].mean() / wfo['avg_oos_ic']) * 100:.1f}%"
    )
    print()

    print("2. 按窗口的IC分布")
    print("-" * 80)
    print(f"IC均值:    {backtest['avg_oos_ic'].mean():.6f}")
    print(f"IC中位数:  {backtest['avg_oos_ic'].median():.6f}")
    print(f"IC标准差:  {backtest['avg_oos_ic'].std():.6f}")
    print(
        f"IC最小值:  {backtest['avg_oos_ic'].min():.6f} (窗口 {backtest.loc[backtest['avg_oos_ic'].idxmin(), 'window_idx']:.0f})"
    )
    print(
        f"IC最大值:  {backtest['avg_oos_ic'].max():.6f} (窗口 {backtest.loc[backtest['avg_oos_ic'].idxmax(), 'window_idx']:.0f})"
    )
    print()

    print("3. 时间序列IC趋势 (是否存在衰减趋势)")
    print("-" * 80)
    early_ic = backtest.iloc[: len(backtest) // 3]["avg_oos_ic"].mean()
    late_ic = backtest.iloc[2 * len(backtest) // 3 :]["avg_oos_ic"].mean()
    print(f"早期窗口 IC (窗 1-18):  {early_ic:.6f}")
    print(f"晚期窗口 IC (窗 37-54): {late_ic:.6f}")
    print(f"衰减: {(1 - late_ic / early_ic) * 100:.1f}%" if early_ic > 0 else "N/A")
    print()

    print("4. 性能与IC的关系 (IC高的窗口收益真的高吗)")
    print("-" * 80)
    high_ic = backtest[backtest["avg_oos_ic"] > backtest["avg_oos_ic"].quantile(0.75)]
    low_ic = backtest[backtest["avg_oos_ic"] < backtest["avg_oos_ic"].quantile(0.25)]
    print(
        f"高IC窗口 (IC>75分位): 平均 Sharpe={high_ic['oos_sharpe'].mean():.4f}, 平均AnnRet={high_ic['oos_annual_return'].mean():.4f}"
    )
    print(
        f"低IC窗口 (IC<25分位): 平均 Sharpe={low_ic['oos_sharpe'].mean():.4f}, 平均AnnRet={low_ic['oos_annual_return'].mean():.4f}"
    )
    print(
        f"IC与Sharpe相关性: {backtest['avg_oos_ic'].corr(backtest['oos_sharpe']):.4f}"
    )
    print()

    print("5. 诊断结论")
    print("-" * 80)

    decay_pct = (1 - backtest["avg_oos_ic"].mean() / wfo["avg_oos_ic"]) * 100

    if decay_pct > 80:
        print(f"🔴 严重过拟合迹象 (衰减{decay_pct:.0f}%)")
        print("   可能原因:")
        print("   - WFO窗口太短（IS太短，因子选择基于噪声）")
        print("   - 因子在OOS期统计特性变化（市场制度变化、数据偏差）")
        print("   - 因子本身非平稳（需要再次标准化或重新设计）")
    elif decay_pct > 50:
        print(f"🟠 中等程度泛化能力下降 (衰减{decay_pct:.0f}%)")
        print("   可能原因:")
        print("   - 因子在不同市场环境下表现不稳定")
        print("   - 选中的因子组合对样本敏感")
    else:
        print(f"🟢 正常衰减范围 (衰减{decay_pct:.0f}%)")

    print()
    print("6. 行动建议")
    print("-" * 80)
    print("优先级1: 检查因子标准化是否跨越了不同市场环境")
    print("        -> 按年/季拆解IC，看是否某个时期完全失效")
    print()
    print("优先级2: 对比 WFO IS 期间的因子 IC vs OOS 期间")
    print("        -> 检查因子在 IS 中是否过拟合")
    print()
    print("优先级3: 尝试更长的 WFO IS 窗口（牺牲窗口数，换取稳定因子）")
    print("        -> 目前 IS 期很短，容易选到噪声因子")
    print()
    print("优先级4: 拆解各因子单独贡献，识别拖累")
    print("        -> 某些因子在 OOS 中可能失效，拖累整体 IC")
    print()


def analyze_factor_contribution():
    """
    分析因子贡献度（按选中频率和OOS IC）
    """
    print("\n" + "=" * 80)
    print("因子贡献度分析")
    print("=" * 80)
    print()

    backtest = load_backtest_results()

    # 统计因子选中频率
    all_factors = []
    for factors_str in backtest["selected_factors"]:
        if isinstance(factors_str, str) and factors_str:
            all_factors.extend(factors_str.split("|"))

    from collections import Counter

    factor_freq = Counter(all_factors)

    print("因子选中频率 (54个窗口)")
    print("-" * 80)
    for factor, count in factor_freq.most_common():
        pct = 100 * count / 54
        print(f"{factor:40s}: {count:2d}次 ({pct:5.1f}%)")
    print()

    print("观察:")
    print("- PRICE_POSITION_20D, RSI_14, SHARPE_RATIO_20D 超频率出现（98%+）")
    print("- 这3个因子是否冗余？是否存在多重共线性？")
    print("- 后续可尝试：去掉其中两个，看IC是否真的下降")
    print()


if __name__ == "__main__":
    diagnose_ic_decay()
    analyze_factor_contribution()

    print("\n" + "=" * 80)
    print("后续立即行动清单")
    print("=" * 80)
    print(
        """
1. 修改step3_run_wfo.py，输出每个窗口的因子IC贡献度矩阵
   -> 这样可以看出哪个因子在哪个窗口失效

2. 尝试TopN=10（降低因子过拟合风险）
   -> 增加持仓多样性，平滑信号噪声

3. 尝试周频而不是日频信号
   -> 可能日频噪声太大，5个资产的周频组合会更稳定

4. 关闭 PRICE_POSITION_20D 中的一个，看 IC 是否真的下降 10%+
   -> 如果没有，说明存在冗余，应该删掉

---
核心认知：
当平均 Sharpe = -0.05 时，任何代码优化都无法拯救。
必须从信号质量入手，IC 衰减 91% 是根本问题。
"""
    )
