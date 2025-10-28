#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极对比：基线 vs 增强实验全维度汇总

输出：
1. IC层指标对比
2. 组合层指标对比
3. 因子入选频率Top10对比
4. 负IC窗口成交量因子鲁棒性对比
5. 推广决策建议
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 实验路径
BASELINE_DIR = PROJECT_ROOT / "results" / "wfo" / "20251028_151333"
ENHANCED_DIR = PROJECT_ROOT / "results" / "wfo" / "20251028_151604"


def load_metadata(exp_dir: Path) -> dict:
    """加载metadata.json"""
    with open(exp_dir / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_negative_ic_diagnosis(exp_dir: Path) -> list:
    """加载负IC窗口诊断"""
    diag_path = exp_dir / "negative_ic_diagnosis.json"
    if diag_path.exists():
        with open(diag_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("终极对比：基线 vs 增强实验")
    print("=" * 80)

    # 加载metadata
    baseline_meta = load_metadata(BASELINE_DIR)
    enhanced_meta = load_metadata(ENHANCED_DIR)

    # 1. IC层对比
    print("\n📊 一、IC层指标对比")
    print("-" * 80)

    baseline_ic = baseline_meta["avg_oos_ic"]
    enhanced_ic = enhanced_meta["avg_oos_ic"]
    ic_improvement = (enhanced_ic / baseline_ic - 1) * 100

    baseline_decay = baseline_meta["avg_ic_decay"]
    enhanced_decay = enhanced_meta["avg_ic_decay"]
    decay_improvement = (enhanced_decay / baseline_decay - 1) * 100

    print(f"{'指标':<25s} {'基线':>12s} {'增强':>12s} {'差异':>12s} {'判定':>8s}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    print(
        f"{'OOS IC均值':<25s} {baseline_ic:>12.4f} {enhanced_ic:>12.4f} {ic_improvement:>11.1f}% {'✅' if ic_improvement > 0 else '❌':>8s}"
    )
    print(
        f"{'IC衰减':<25s} {baseline_decay:>12.4f} {enhanced_decay:>12.4f} {decay_improvement:>11.1f}% {'✅' if decay_improvement < 0 else '❌':>8s}"
    )

    # 2. 组合层对比
    print("\n💼 二、组合层指标对比（Top-12因子加权策略）")
    print("-" * 80)

    baseline_backtest = baseline_meta["backtest_metrics"]
    enhanced_backtest = enhanced_meta["backtest_metrics"]

    baseline_ret = baseline_backtest["annualized_return"]
    enhanced_ret = enhanced_backtest["annualized_return"]
    ret_diff = (enhanced_ret / baseline_ret - 1) * 100

    baseline_sharpe = baseline_backtest["sharpe_ratio"]
    enhanced_sharpe = enhanced_backtest["sharpe_ratio"]
    sharpe_diff = (enhanced_sharpe / baseline_sharpe - 1) * 100

    baseline_dd = baseline_backtest["max_drawdown"]
    enhanced_dd = enhanced_backtest["max_drawdown"]
    dd_diff = (enhanced_dd / baseline_dd - 1) * 100

    baseline_turnover = baseline_backtest["avg_single_turnover"]
    enhanced_turnover = enhanced_backtest["avg_single_turnover"]
    turnover_diff = (enhanced_turnover / baseline_turnover - 1) * 100

    print(
        f"{'指标':<25s} {'基线':>12s} {'增强':>12s} {'差异':>12s} {'阈值':>10s} {'判定':>8s}"
    )
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")
    print(
        f"{'年化收益':<25s} {baseline_ret:>11.2%} {enhanced_ret:>11.2%} {ret_diff:>11.1f}% {'+10%':>10s} {'❌' if ret_diff < 10 else '✅':>8s}"
    )
    print(
        f"{'夏普比率':<25s} {baseline_sharpe:>12.3f} {enhanced_sharpe:>12.3f} {sharpe_diff:>11.1f}% {'+10%':>10s} {'❌' if sharpe_diff < 10 else '✅':>8s}"
    )
    print(
        f"{'最大回撤':<25s} {baseline_dd:>11.2%} {enhanced_dd:>11.2%} {dd_diff:>11.1f}% {'不恶化':>10s} {'❌' if dd_diff > 0 else '✅':>8s}"
    )
    print(
        f"{'平均换手':<25s} {baseline_turnover:>11.2%} {enhanced_turnover:>11.2%} {turnover_diff:>11.1f}% {'-':>10s} {'➖':>8s}"
    )

    # 3. 因子频率对比
    print("\n🔍 三、因子入选频率Top10对比")
    print("-" * 80)

    baseline_freq_list = baseline_meta["factor_selection_freq"]
    enhanced_freq_list = enhanced_meta["factor_selection_freq"]

    # 转换为字典
    baseline_freq = {item[0]: item[1] for item in baseline_freq_list}
    enhanced_freq = {item[0]: item[1] for item in enhanced_freq_list}

    # 合并Top10
    all_factors = set(list(baseline_freq.keys())[:10]) | set(
        list(enhanced_freq.keys())[:10]
    )

    print(f"{'因子名':<30s} {'基线频率':>12s} {'增强频率':>12s} {'差异':>10s}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10}")

    for factor in sorted(
        all_factors, key=lambda f: enhanced_freq.get(f, 0), reverse=True
    )[:10]:
        base_f = baseline_freq.get(factor, 0)
        enh_f = enhanced_freq.get(factor, 0)
        diff = enh_f - base_f

        print(f"{factor:<30s} {base_f:>11.2%} {enh_f:>11.2%} {diff*100:>9.1f}pp")

    # 4. 负IC窗口诊断
    print("\n🩺 四、负IC窗口成交量因子鲁棒性对比")
    print("-" * 80)

    baseline_diag = load_negative_ic_diagnosis(BASELINE_DIR)
    enhanced_diag = load_negative_ic_diagnosis(ENHANCED_DIR)

    if baseline_diag and enhanced_diag:
        baseline_volume_ic = sum([d["volume_mean_ic"] for d in baseline_diag]) / len(
            baseline_diag
        )
        enhanced_volume_ic = sum([d["volume_mean_ic"] for d in enhanced_diag]) / len(
            enhanced_diag
        )

        baseline_base_ic = sum([d["base_mean_ic"] for d in baseline_diag]) / len(
            baseline_diag
        )
        enhanced_base_ic = sum([d["base_mean_ic"] for d in enhanced_diag]) / len(
            enhanced_diag
        )

        baseline_gap = baseline_volume_ic - baseline_base_ic
        enhanced_gap = enhanced_volume_ic - enhanced_base_ic

        print(f"{'指标':<30s} {'基线':>12s} {'增强':>12s}")
        print(f"{'-'*30} {'-'*12} {'-'*12}")
        print(
            f"{'成交量因子平均IC':<30s} {baseline_volume_ic:>12.4f} {enhanced_volume_ic:>12.4f}"
        )
        print(
            f"{'基础因子平均IC':<30s} {baseline_base_ic:>12.4f} {enhanced_base_ic:>12.4f}"
        )
        print(
            f"{'IC差距（成交量-基础）':<30s} {baseline_gap:>12.4f} {enhanced_gap:>12.4f}"
        )

        print(f"\n诊断结论:")
        print(
            f"  基线: {'✅ 成交量因子无系统性劣势' if baseline_gap >= 0 else '⚠️ 成交量因子劣于基础因子'}"
        )
        print(
            f"  增强: {'✅ 成交量因子无系统性劣势' if enhanced_gap >= 0 else '⚠️ 成交量因子系统性劣于基础因子（差距' + f'{abs(enhanced_gap):.4f}' + '）'}"
        )

    # 5. 推广决策
    print("\n" + "=" * 80)
    print("💡 五、推广决策建议")
    print("=" * 80)

    pass_count = 0
    total_criteria = 4

    criteria = [
        ("OOS IC改善 ≥+5%", ic_improvement >= 5, ic_improvement),
        ("夏普比率提升 ≥+10%", sharpe_diff >= 10, sharpe_diff),
        ("回撤不恶化", dd_diff <= 0, dd_diff),
        ("IC衰减改善", decay_improvement < 0, decay_improvement),
    ]

    print(f"\n决策矩阵:")
    print(f"{'准则':<30s} {'阈值':>15s} {'实际表现':>15s} {'通过':>8s}")
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*8}")

    for name, passed, value in criteria:
        status = "✅" if passed else "❌"
        if "IC改善" in name:
            print(f"{name:<30s} {'≥+5%':>15s} {f'+{value:.1f}%':>15s} {status:>8s}")
        elif "夏普" in name:
            print(f"{name:<30s} {'≥+10%':>15s} {f'{value:+.1f}%':>15s} {status:>8s}")
        elif "回撤" in name:
            print(f"{name:<30s} {'≤0%':>15s} {f'{value:+.1f}%':>15s} {status:>8s}")
        elif "衰减" in name:
            print(f"{name:<30s} {'负向':>15s} {f'{value:+.1f}%':>15s} {status:>8s}")

        if passed:
            pass_count += 1

    print(
        f"\n通过率: {pass_count}/{total_criteria} ({pass_count/total_criteria*100:.1f}%)"
    )

    if pass_count >= 3:
        decision = "✅ 建议推广"
        reason = f"通过{pass_count}/{total_criteria}项准则，满足推广要求"
    elif pass_count >= 2:
        decision = "⚠️  建议降配额试行"
        reason = "部分指标改善但组合绩效未达标，建议降低新因子族max_count后重新验证"
    else:
        decision = "🛑 暂缓推广"
        reason = "组合绩效恶化，成交量因子在极端波动期系统性失效"

    print(f"\n最终决策: {decision}")
    print(f"理由: {reason}")

    if decision == "🛑 暂缓推广":
        print(f"\n改进方向:")
        print(f"  1. 禁用成交量技术指标（RSI_14, OBV_SLOPE_10D）")
        print(f"  2. 保留TSMOM_120D + BREAKOUT_20D（稳定性好）")
        print(f"  3. 研究窗口38/46极端波动期特征，开发条件激活机制")

    print("\n" + "=" * 80)
    print("📋 详细报告: BACKTEST_UPGRADE_REPORT.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
