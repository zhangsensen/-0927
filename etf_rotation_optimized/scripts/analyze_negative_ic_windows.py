#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负IC窗口单因子剖析 - 诊断成交量因子鲁棒性

目标：
1. 定位OOS IC<0的负IC窗口
2. 提取该窗口所有候选因子的IS与OOS IC
3. 对比成交量敏感因子（RSI/OBV/TURNOVER_ACCEL）vs 基础因子的表现
4. 诊断是否存在系统性失效（极端波动期技术指标钝化）

输入：wfo_results.pkl + constraint_reports
输出：负IC窗口诊断报告
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 成交量敏感因子分组
VOLUME_FACTORS = {
    "RSI_14",
    "OBV_SLOPE_10D",
    "TURNOVER_ACCEL_5_20",
    "ADX_14D",
    "VORTEX_14D",
}
BASE_FACTORS = {
    "CALMAR_RATIO_60D",
    "CMF_20D",
    "PRICE_POSITION_120D",
    "BREAKOUT_20D",
    "TSMOM_120D",
}


def analyze_negative_ic_windows(wfo_dir: Path):
    """分析负IC窗口的单因子表现"""
    print(f"\n🔍 分析WFO负IC窗口: {wfo_dir.name}")
    print("=" * 80)

    # 加载WFO结果
    pkl_path = wfo_dir / "wfo_results.pkl"
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    window_results = results.get("window_results", [])
    constraint_reports = results.get("constraint_reports", [])

    # 筛选负IC窗口
    negative_windows = [w for w in window_results if w.get("oos_ic", 0) < 0]

    print(f"\n📊 负IC窗口统计:")
    print(f"  总窗口数: {len(window_results)}")
    print(
        f"  负IC窗口数: {len(negative_windows)} ({len(negative_windows)/len(window_results)*100:.1f}%)"
    )
    print(f"  平均负IC: {np.mean([w['oos_ic'] for w in negative_windows]):.4f}")

    # 重点窗口：28-30, 38, 46-51
    focus_windows = [
        w
        for w in negative_windows
        if w["window_id"] in [28, 29, 30, 38, 46, 47, 48, 49, 50, 51]
    ]

    print(f"\n🎯 重点负IC窗口（28-30, 38, 46-51）:")
    print(f"  匹配数: {len(focus_windows)}/{len([28,29,30,38,46,47,48,49,50,51])}")

    # 逐窗口分析
    diagnostics = []

    for w in focus_windows:
        wid = w["window_id"]
        oos_ic = w["oos_ic"]
        selected_factors = w["selected_factors"]

        # 尝试从constraint_reports获取候选因子IC
        report = None
        if constraint_reports and wid <= len(constraint_reports):
            report = constraint_reports[wid - 1]

        if report is None:
            print(f"\n⚠️  窗口{wid}: 无constraint_report，跳过")
            continue

        # 提取IS与OOS IC（使用属性访问）
        is_ic_stats = report.is_ic_stats if hasattr(report, "is_ic_stats") else {}
        candidate_factors = (
            report.candidate_factors if hasattr(report, "candidate_factors") else []
        )
        oos_performance = (
            report.oos_performance if hasattr(report, "oos_performance") else {}
        )

        # 分组统计
        volume_ic = {
            f: oos_performance.get(f, 0.0)
            for f in VOLUME_FACTORS
            if f in oos_performance
        }
        base_ic = {
            f: oos_performance.get(f, 0.0) for f in BASE_FACTORS if f in oos_performance
        }

        volume_mean = np.mean(list(volume_ic.values())) if volume_ic else 0.0
        base_mean = np.mean(list(base_ic.values())) if base_ic else 0.0

        diagnostics.append(
            {
                "window_id": wid,
                "oos_ic": oos_ic,
                "selected_factors": selected_factors,
                "volume_factors_ic": volume_ic,
                "base_factors_ic": base_ic,
                "volume_mean_ic": volume_mean,
                "base_mean_ic": base_mean,
                "ic_gap": volume_mean - base_mean,
            }
        )

        print(f"\n窗口 {wid} (OOS IC={oos_ic:.4f}):")
        print(f"  选中因子: {', '.join(selected_factors)}")
        print(f"  成交量因子平均IC: {volume_mean:.4f}")
        print(f"  基础因子平均IC:   {base_mean:.4f}")
        print(
            f"  IC差距: {volume_mean - base_mean:.4f} ({'成交量劣' if volume_mean < base_mean else '成交量优'})"
        )

        # 详细列表
        if volume_ic:
            print(f"  成交量因子明细:")
            for f, ic in sorted(volume_ic.items(), key=lambda x: x[1]):
                print(f"    {f:30s} IC={ic:7.4f}")
        if base_ic:
            print(f"  基础因子明细:")
            for f, ic in sorted(base_ic.items(), key=lambda x: x[1]):
                print(f"    {f:30s} IC={ic:7.4f}")

    # 汇总诊断
    print("\n" + "=" * 80)
    print("🩺 诊断汇总:")

    if diagnostics:
        avg_volume_ic = np.mean([d["volume_mean_ic"] for d in diagnostics])
        avg_base_ic = np.mean([d["base_mean_ic"] for d in diagnostics])
        avg_gap = np.mean([d["ic_gap"] for d in diagnostics])

        print(f"\n重点负IC窗口平均表现:")
        print(f"  成交量因子平均IC: {avg_volume_ic:.4f}")
        print(f"  基础因子平均IC:   {avg_base_ic:.4f}")
        print(f"  平均IC差距:        {avg_gap:.4f}")

        if avg_volume_ic < avg_base_ic:
            print(f"\n⚠️  结论: 成交量因子在负IC窗口表现**系统性劣于**基础因子")
            print(
                f"         差距幅度={abs(avg_gap):.4f}，提示极端波动期技术指标钝化风险"
            )
        else:
            print(f"\n✅ 结论: 成交量因子在负IC窗口无系统性劣势")

    return diagnostics


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python analyze_negative_ic_windows.py <wfo_timestamp_dir>")
        sys.exit(1)

    wfo_dir = Path(sys.argv[1])
    if not wfo_dir.exists():
        print(f"❌ 目录不存在: {wfo_dir}")
        sys.exit(1)

    diagnostics = analyze_negative_ic_windows(wfo_dir)

    # 保存诊断报告
    output_path = wfo_dir / "negative_ic_diagnosis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    print(f"\n💾 诊断报告已保存: {output_path}")


if __name__ == "__main__":
    main()
