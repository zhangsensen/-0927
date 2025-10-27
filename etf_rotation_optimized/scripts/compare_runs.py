#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFO结果对比工具
==============

用于对比两次WFO回测结果的差异

使用示例:
  python scripts/compare_runs.py results/20251025_154432 results/20251025_160000
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_run_results(run_dir: Path) -> Tuple[pd.DataFrame, dict]:
    """加载单次运行结果"""
    wfo_file = run_dir / "wfo_results.csv"
    metadata_file = run_dir / "metadata.json"

    if not wfo_file.exists():
        raise FileNotFoundError(f"找不到WFO结果: {wfo_file}")

    wfo_df = pd.read_csv(wfo_file)

    metadata = {}
    if metadata_file.exists():
        import json

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    return wfo_df, metadata


def compare_wfo_runs(run1_dir: Path, run2_dir: Path):
    """对比两次WFO运行"""
    print("=" * 80)
    print("WFO结果对比")
    print("=" * 80)

    # 加载数据
    print(f"\n加载数据...")
    print(f"  Run 1: {run1_dir.name}")
    print(f"  Run 2: {run2_dir.name}")

    wfo1, meta1 = load_run_results(run1_dir)
    wfo2, meta2 = load_run_results(run2_dir)

    # 基本信息对比
    print(f"\n{'基本信息对比':-^80}")
    print(f"{'指标':<30} {'Run 1':>20} {'Run 2':>20}")
    print(f"{'-'*80}")
    print(f"{'窗口数':<30} {len(wfo1):>20d} {len(wfo2):>20d}")

    if meta1:
        params1 = meta1.get("parameters", {})
        params2 = meta2.get("parameters", {})

        print(
            f"{'样本内天数':<30} {params1.get('is_period', 'N/A'):>20} {params2.get('oos_period', 'N/A'):>20}"
        )
        print(
            f"{'样本外天数':<30} {params1.get('oos_period', 'N/A'):>20} {params2.get('oos_period', 'N/A'):>20}"
        )
        print(
            f"{'滚动步长':<30} {params1.get('step_size', 'N/A'):>20} {params2.get('step_size', 'N/A'):>20}"
        )

    # IC对比
    print(f"\n{'IC统计对比':-^80}")
    ic1_mean = wfo1["oos_ic_mean"].mean()
    ic2_mean = wfo2["oos_ic_mean"].mean()
    ic_diff = ic2_mean - ic1_mean

    print(f"{'指标':<30} {'Run 1':>20} {'Run 2':>20} {'差值':>20}")
    print(f"{'-'*80}")
    print(f"{'平均 OOS IC':<30} {ic1_mean:>20.4f} {ic2_mean:>20.4f} {ic_diff:>+20.4f}")

    ic1_std = wfo1["oos_ic_mean"].std()
    ic2_std = wfo2["oos_ic_mean"].std()
    print(
        f"{'OOS IC 标准差':<30} {ic1_std:>20.4f} {ic2_std:>20.4f} {ic2_std-ic1_std:>+20.4f}"
    )

    drop1_mean = wfo1["ic_drop"].mean()
    drop2_mean = wfo2["ic_drop"].mean()
    print(
        f"{'平均 IC 衰减':<30} {drop1_mean:>20.4f} {drop2_mean:>20.4f} {drop2_mean-drop1_mean:>+20.4f}"
    )

    # 因子选择频率对比
    print(f"\n{'因子选择频率对比':-^80}")

    all_selected1 = ",".join(wfo1["selected_factors"].tolist()).split(",")
    all_selected2 = ",".join(wfo2["selected_factors"].tolist()).split(",")

    freq1 = pd.Series(all_selected1).value_counts()
    freq2 = pd.Series(all_selected2).value_counts()

    # 合并所有因子
    all_factors = sorted(set(list(freq1.index) + list(freq2.index)))

    print(f"{'因子':<30} {'Run 1 频率':>20} {'Run 2 频率':>20}")
    print(f"{'-'*80}")

    for factor in all_factors[:10]:  # 显示TOP 10
        f1 = freq1.get(factor, 0)
        f2 = freq2.get(factor, 0)
        print(f"{factor:<30} {f1:>20d} {f2:>20d}")

    # Sharpe对比（如果有组合结果）
    portfolio1_file = run1_dir / "top100_portfolios.csv"
    portfolio2_file = run2_dir / "top100_portfolios.csv"

    if portfolio1_file.exists() and portfolio2_file.exists():
        print(f"\n{'组合表现对比':-^80}")

        pf1 = pd.read_csv(portfolio1_file)
        pf2 = pd.read_csv(portfolio2_file)

        print(f"{'指标':<30} {'Run 1':>20} {'Run 2':>20}")
        print(f"{'-'*80}")
        print(
            f"{'TOP Sharpe':<30} {pf1['sharpe'].max():>20.4f} {pf2['sharpe'].max():>20.4f}"
        )
        print(
            f"{'平均 Sharpe':<30} {pf1['sharpe'].mean():>20.4f} {pf2['sharpe'].mean():>20.4f}"
        )
        print(
            f"{'TOP 年化收益':<30} {pf1['annual_return'].max():>20.2%} {pf2['annual_return'].max():>20.2%}"
        )
        print(
            f"{'最小 MaxDD':<30} {pf1['max_drawdown'].max():>20.2%} {pf2['max_drawdown'].max():>20.2%}"
        )

    # 结论
    print(f"\n{'结论':-^80}")
    if ic_diff > 0.01:
        print(f"✅ Run 2 显著更优 (IC提升 {ic_diff:+.4f})")
    elif ic_diff < -0.01:
        print(f"⚠️  Run 1 更优 (IC下降 {ic_diff:+.4f})")
    else:
        print(f"➡️  两次运行表现接近 (IC差异 {ic_diff:+.4f})")

    print("=" * 80)


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python compare_runs.py <run1_dir> <run2_dir>")
        print(
            "示例: python compare_runs.py results/20251025_154432 results/20251025_160000"
        )
        sys.exit(1)

    run1_dir = Path(sys.argv[1])
    run2_dir = Path(sys.argv[2])

    if not run1_dir.exists():
        print(f"错误: 找不到目录 {run1_dir}")
        sys.exit(1)

    if not run2_dir.exists():
        print(f"错误: 找不到目录 {run2_dir}")
        sys.exit(1)

    compare_wfo_runs(run1_dir, run2_dir)


if __name__ == "__main__":
    main()
