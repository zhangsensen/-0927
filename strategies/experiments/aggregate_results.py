#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验结果聚合工具

功能：
1. 聚合多次实验的 Top-N 策略
2. 生成总榜 CSV/JSON
3. 支持"历史最优 vs 最新最优"对比报表
4. 生成可视化图表（夏普-TopN 热力图、夏普-费率曲线）

用法：
    # 聚合所有 P0 实验结果
    python strategies/experiments/aggregate_results.py \\
        --pattern "strategies/results/experiments/p0_*/run_*/results.csv" \\
        --output strategies/results/experiments/p0_summary.csv
    
    # 聚合指定实验
    python strategies/experiments/aggregate_results.py \\
        --files result1.csv result2.csv result3.csv \\
        --output summary.csv
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_experiment_results(file_paths: List[Path]) -> pd.DataFrame:
    """加载多个实验结果文件并合并

    Args:
        file_paths: 结果文件路径列表

    Returns:
        合并后的 DataFrame
    """
    dfs = []
    for fp in file_paths:
        try:
            df = pd.read_csv(fp)
            df["source_file"] = str(fp)
            dfs.append(df)
            print(f"✅ 加载: {fp} ({len(df)} 条记录)")
        except Exception as e:
            print(f"⚠️ 跳过: {fp} (错误: {e})")

    if not dfs:
        raise ValueError("未成功加载任何结果文件")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 合并完成: {len(combined)} 条记录，来自 {len(dfs)} 个文件")

    return combined


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """生成汇总统计

    Args:
        df: 实验结果 DataFrame

    Returns:
        汇总统计 DataFrame
    """
    summary = (
        df.groupby(["top_n", "fee"])
        .agg(
            {
                "sharpe": ["mean", "std", "max", "min"],
                "annual_return": ["mean", "max"],
                "max_drawdown": ["mean", "min"],
                "turnover": "mean",
                "combo_idx": "count",
            }
        )
        .reset_index()
    )

    # 扁平化列名
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    return summary


def plot_sharpe_topn_heatmap(
    df: pd.DataFrame, output_path: Path, fee_value: Optional[float] = None
):
    """绘制夏普-TopN 热力图

    Args:
        df: 实验结果 DataFrame
        output_path: 输出图片路径
        fee_value: 指定费率值（None 则使用所有费率的平均值）
    """
    if fee_value is not None:
        df_filtered = df[df["fee"] == fee_value].copy()
        title = f"Sharpe vs Top-N (fee={fee_value:.4f})"
    else:
        df_filtered = df.copy()
        title = "Sharpe vs Top-N (all fees)"

    # 计算每个 Top-N 的平均夏普
    pivot = (
        df_filtered.groupby("top_n")["sharpe"].agg(["mean", "max", "std"]).reset_index()
    )

    plt.figure(figsize=(10, 6))

    # 绘制条形图
    x = pivot["top_n"]
    y_mean = pivot["mean"]
    y_std = pivot["std"]

    plt.bar(x, y_mean, yerr=y_std, capsize=5, alpha=0.7, label="Mean ± Std")
    plt.plot(x, pivot["max"], "ro-", label="Max", linewidth=2)

    plt.xlabel("Top-N", fontsize=12)
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"📈 热力图已保存: {output_path}")
    plt.close()


def plot_sharpe_fee_curve(df: pd.DataFrame, output_path: Path, top_n: int = 8):
    """绘制夏普-费率曲线

    Args:
        df: 实验结果 DataFrame
        output_path: 输出图片路径
        top_n: 指定 Top-N 值
    """
    df_filtered = df[df["top_n"] == top_n].copy()

    if df_filtered.empty:
        print(f"⚠️ 未找到 Top-N={top_n} 的数据，跳过费率曲线绘制")
        return

    # 计算每个费率的统计
    fee_stats = (
        df_filtered.groupby("fee")["sharpe"].agg(["mean", "max", "std"]).reset_index()
    )

    plt.figure(figsize=(10, 6))

    x = fee_stats["fee"]
    y_mean = fee_stats["mean"]
    y_std = fee_stats["std"]

    plt.errorbar(
        x, y_mean, yerr=y_std, fmt="o-", capsize=5, label="Mean ± Std", linewidth=2
    )
    plt.plot(x, fee_stats["max"], "s--", label="Max", linewidth=2)

    plt.xlabel("Transaction Fee", fontsize=12)
    plt.ylabel("Sharpe Ratio", fontsize=12)
    plt.title(f"Sharpe vs Fee (Top-N={top_n})", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"📈 费率曲线已保存: {output_path}")
    plt.close()


def compare_with_history(
    current_df: pd.DataFrame, history_path: Optional[Path]
) -> pd.DataFrame:
    """对比当前结果与历史最优

    Args:
        current_df: 当前实验结果
        history_path: 历史最优结果文件路径

    Returns:
        对比报表 DataFrame
    """
    if history_path is None or not history_path.exists():
        print("⚠️ 未提供历史最优文件或文件不存在，跳过对比")
        return pd.DataFrame()

    history_df = pd.read_csv(history_path)

    # 找到当前和历史的最优策略
    current_best = current_df.nlargest(1, "sharpe").iloc[0]
    history_best = history_df.nlargest(1, "sharpe").iloc[0]

    comparison = pd.DataFrame(
        {
            "Metric": ["Sharpe", "Annual Return", "Max Drawdown", "Turnover"],
            "Current Best": [
                current_best["sharpe"],
                current_best["annual_return"],
                current_best["max_drawdown"],
                current_best["turnover"],
            ],
            "History Best": [
                history_best["sharpe"],
                history_best["annual_return"],
                history_best["max_drawdown"],
                history_best["turnover"],
            ],
        }
    )

    comparison["Improvement"] = (
        (comparison["Current Best"] - comparison["History Best"])
        / comparison["History Best"].abs()
        * 100
    )

    print("\n📊 历史对比:")
    print(comparison.to_string(index=False))

    return comparison


def main():
    parser = argparse.ArgumentParser(description="实验结果聚合工具")

    parser.add_argument("--files", nargs="+", help="结果文件路径列表")
    parser.add_argument("--pattern", type=str, help="结果文件路径模式（支持通配符）")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径（CSV）")
    parser.add_argument("--top-n", type=int, default=100, help="保留 Top-N 个最优策略")
    parser.add_argument("--history", type=str, help="历史最优结果文件路径（用于对比）")
    parser.add_argument("--plot", action="store_true", help="生成可视化图表")

    args = parser.parse_args()

    # 确定文件列表
    if args.files:
        file_paths = [Path(f) for f in args.files]
    elif args.pattern:
        file_paths = [Path(f) for f in glob.glob(args.pattern)]
    else:
        print("❌ 请指定 --files 或 --pattern")
        return 1

    if not file_paths:
        print("❌ 未找到匹配的结果文件")
        return 1

    # 加载并合并结果
    combined_df = load_experiment_results(file_paths)

    # 生成汇总统计
    summary_stats = generate_summary_stats(combined_df)
    print("\n📊 汇总统计:")
    print(summary_stats.to_string(index=False))

    # 保留 Top-N 策略
    top_strategies = combined_df.nlargest(args.top_n, "sharpe")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存结果
    top_strategies.to_csv(output_path, index=False)
    print(f"\n📁 Top-{args.top_n} 策略已保存: {output_path}")

    # 保存汇总统计
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary_stats.to_csv(summary_path, index=False)
    print(f"📁 汇总统计已保存: {summary_path}")

    # 历史对比
    if args.history:
        comparison = compare_with_history(combined_df, Path(args.history))
        if not comparison.empty:
            comp_path = output_path.with_name(output_path.stem + "_comparison.csv")
            comparison.to_csv(comp_path, index=False)
            print(f"📁 历史对比已保存: {comp_path}")

    # 生成图表
    if args.plot:
        plot_dir = output_path.parent / "plots"
        plot_dir.mkdir(exist_ok=True)

        # 夏普-TopN 热力图
        heatmap_path = plot_dir / f"{output_path.stem}_sharpe_topn.png"
        plot_sharpe_topn_heatmap(combined_df, heatmap_path)

        # 夏普-费率曲线
        fee_curve_path = plot_dir / f"{output_path.stem}_sharpe_fee.png"
        plot_sharpe_fee_curve(combined_df, fee_curve_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
