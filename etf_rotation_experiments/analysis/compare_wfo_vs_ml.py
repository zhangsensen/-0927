#!/usr/bin/env python3
"""
WFO 原排序 vs ML 排序 对比分析脚本

用法:
    python analysis/compare_wfo_vs_ml.py \
        --wfo-report path/to/wfo_backtest.csv \
        --ml-report path/to/ml_backtest.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_backtest_results(csv_path: str) -> pd.DataFrame:
    """加载回测结果 CSV"""
    df = pd.read_csv(csv_path)
    return df


def extract_summary_metrics(df: pd.DataFrame) -> dict:
    """提取关键汇总指标"""
    # 假设 CSV 中每行是一个组合的回测结果
    # 计算 Top-200 的平均指标
    metrics = {
        "样本数": len(df),
        "年化收益(净)_均值": df["annual_ret_net"].mean() if "annual_ret_net" in df.columns else 0,
        "年化收益(净)_中位数": df["annual_ret_net"].median() if "annual_ret_net" in df.columns else 0,
        "年化收益(净)_Top1": df["annual_ret_net"].iloc[0] if "annual_ret_net" in df.columns and len(df) > 0 else 0,
        "Sharpe(净)_均值": df["sharpe_net"].mean() if "sharpe_net" in df.columns else 0,
        "Sharpe(净)_中位数": df["sharpe_net"].median() if "sharpe_net" in df.columns else 0,
        "Sharpe(净)_Top1": df["sharpe_net"].iloc[0] if "sharpe_net" in df.columns and len(df) > 0 else 0,
        "最大回撤(净)_均值": df["max_dd_net"].mean() if "max_dd_net" in df.columns else 0,
        "最大回撤(净)_中位数": df["max_dd_net"].median() if "max_dd_net" in df.columns else 0,
        "最大回撤(净)_Top1": df["max_dd_net"].iloc[0] if "max_dd_net" in df.columns and len(df) > 0 else 0,
    }
    
    # 添加 Calmar 比率 (如果存在)
    if "calmar_net" in df.columns:
        metrics["Calmar(净)_均值"] = df["calmar_net"].mean()
        metrics["Calmar(净)_中位数"] = df["calmar_net"].median()
        metrics["Calmar(净)_Top1"] = df["calmar_net"].iloc[0] if len(df) > 0 else 0
    
    # 添加胜率 (如果存在)
    if "win_rate" in df.columns:
        metrics["胜率_均值"] = df["win_rate"].mean()
        metrics["胜率_中位数"] = df["win_rate"].median()
        metrics["胜率_Top1"] = df["win_rate"].iloc[0] if len(df) > 0 else 0
    
    return metrics


def format_pct(value: float, decimals: int = 2) -> str:
    """格式化为百分比"""
    return f"{value*100:.{decimals}f}%"


def format_diff(wfo_val: float, ml_val: float, is_pct: bool = True, is_drawdown: bool = False) -> str:
    """格式化差异值 (改善/恶化)"""
    diff = ml_val - wfo_val
    
    if is_drawdown:
        # 回撤：负值小更好
        if abs(diff) < 0.001:
            return "持平"
        elif diff > 0:
            return f"↓ {format_pct(abs(diff))}" if is_pct else f"↓ {abs(diff):.4f}"
        else:
            return f"↑ {format_pct(abs(diff))}" if is_pct else f"↑ {abs(diff):.4f}"
    else:
        # 收益/Sharpe：正值大更好
        if abs(diff) < 0.001:
            return "持平"
        elif diff > 0:
            return f"↑ {format_pct(abs(diff))}" if is_pct else f"↑ {abs(diff):.4f}"
        else:
            return f"↓ {format_pct(abs(diff))}" if is_pct else f"↓ {abs(diff):.4f}"


def generate_markdown_report(wfo_metrics: dict, ml_metrics: dict, wfo_path: str, ml_path: str) -> str:
    """生成 Markdown 格式的对比报告"""
    
    report = []
    report.append("# WFO 原排序 vs ML 排序 对比报告")
    report.append("")
    report.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## 数据来源")
    report.append("")
    report.append(f"- **WFO 原排序**: `{Path(wfo_path).name}`")
    report.append(f"- **ML 排序**: `{Path(ml_path).name}`")
    report.append(f"- **样本数**: WFO={wfo_metrics['样本数']}, ML={ml_metrics['样本数']}")
    report.append("")
    
    report.append("## 汇总指标对比")
    report.append("")
    report.append("### 1. Top-1 组合表现")
    report.append("")
    report.append("| 指标 | WFO排序 | ML排序 | 变化 |")
    report.append("|------|---------|--------|------|")
    
    # Top-1 年化收益
    wfo_ret_top1 = wfo_metrics["年化收益(净)_Top1"]
    ml_ret_top1 = ml_metrics["年化收益(净)_Top1"]
    report.append(f"| 年化收益(净) | {format_pct(wfo_ret_top1)} | {format_pct(ml_ret_top1)} | {format_diff(wfo_ret_top1, ml_ret_top1)} |")
    
    # Top-1 Sharpe
    wfo_sharpe_top1 = wfo_metrics["Sharpe(净)_Top1"]
    ml_sharpe_top1 = ml_metrics["Sharpe(净)_Top1"]
    report.append(f"| Sharpe(净) | {wfo_sharpe_top1:.3f} | {ml_sharpe_top1:.3f} | {format_diff(wfo_sharpe_top1, ml_sharpe_top1, is_pct=False)} |")
    
    # Top-1 最大回撤
    wfo_dd_top1 = wfo_metrics["最大回撤(净)_Top1"]
    ml_dd_top1 = ml_metrics["最大回撤(净)_Top1"]
    report.append(f"| 最大回撤(净) | {format_pct(wfo_dd_top1)} | {format_pct(ml_dd_top1)} | {format_diff(wfo_dd_top1, ml_dd_top1, is_drawdown=True)} |")
    
    # Top-1 Calmar (如果存在)
    if "Calmar(净)_Top1" in wfo_metrics:
        wfo_calmar_top1 = wfo_metrics["Calmar(净)_Top1"]
        ml_calmar_top1 = ml_metrics["Calmar(净)_Top1"]
        report.append(f"| Calmar(净) | {wfo_calmar_top1:.3f} | {ml_calmar_top1:.3f} | {format_diff(wfo_calmar_top1, ml_calmar_top1, is_pct=False)} |")
    
    report.append("")
    
    report.append("### 2. Top-200 组合平均表现")
    report.append("")
    report.append("| 指标 | WFO排序 | ML排序 | 变化 |")
    report.append("|------|---------|--------|------|")
    
    # 平均年化收益
    wfo_ret_mean = wfo_metrics["年化收益(净)_均值"]
    ml_ret_mean = ml_metrics["年化收益(净)_均值"]
    report.append(f"| 年化收益(净)_均值 | {format_pct(wfo_ret_mean)} | {format_pct(ml_ret_mean)} | {format_diff(wfo_ret_mean, ml_ret_mean)} |")
    
    # 平均 Sharpe
    wfo_sharpe_mean = wfo_metrics["Sharpe(净)_均值"]
    ml_sharpe_mean = ml_metrics["Sharpe(净)_均值"]
    report.append(f"| Sharpe(净)_均值 | {wfo_sharpe_mean:.3f} | {ml_sharpe_mean:.3f} | {format_diff(wfo_sharpe_mean, ml_sharpe_mean, is_pct=False)} |")
    
    # 平均最大回撤
    wfo_dd_mean = wfo_metrics["最大回撤(净)_均值"]
    ml_dd_mean = ml_metrics["最大回撤(净)_均值"]
    report.append(f"| 最大回撤(净)_均值 | {format_pct(wfo_dd_mean)} | {format_pct(ml_dd_mean)} | {format_diff(wfo_dd_mean, ml_dd_mean, is_drawdown=True)} |")
    
    # 平均 Calmar (如果存在)
    if "Calmar(净)_均值" in wfo_metrics:
        wfo_calmar_mean = wfo_metrics["Calmar(净)_均值"]
        ml_calmar_mean = ml_metrics["Calmar(净)_均值"]
        report.append(f"| Calmar(净)_均值 | {wfo_calmar_mean:.3f} | {ml_calmar_mean:.3f} | {format_diff(wfo_calmar_mean, ml_calmar_mean, is_pct=False)} |")
    
    report.append("")
    
    report.append("### 3. Top-200 组合中位数表现")
    report.append("")
    report.append("| 指标 | WFO排序 | ML排序 | 变化 |")
    report.append("|------|---------|--------|------|")
    
    # 中位数年化收益
    wfo_ret_median = wfo_metrics["年化收益(净)_中位数"]
    ml_ret_median = ml_metrics["年化收益(净)_中位数"]
    report.append(f"| 年化收益(净)_中位数 | {format_pct(wfo_ret_median)} | {format_pct(ml_ret_median)} | {format_diff(wfo_ret_median, ml_ret_median)} |")
    
    # 中位数 Sharpe
    wfo_sharpe_median = wfo_metrics["Sharpe(净)_中位数"]
    ml_sharpe_median = ml_metrics["Sharpe(净)_中位数"]
    report.append(f"| Sharpe(净)_中位数 | {wfo_sharpe_median:.3f} | {ml_sharpe_median:.3f} | {format_diff(wfo_sharpe_median, ml_sharpe_median, is_pct=False)} |")
    
    # 中位数最大回撤
    wfo_dd_median = wfo_metrics["最大回撤(净)_中位数"]
    ml_dd_median = ml_metrics["最大回撤(净)_中位数"]
    report.append(f"| 最大回撤(净)_中位数 | {format_pct(wfo_dd_median)} | {format_pct(ml_dd_median)} | {format_diff(wfo_dd_median, ml_dd_median, is_drawdown=True)} |")
    
    report.append("")
    
    # 简单结论
    report.append("## 结论")
    report.append("")
    
    # 计算关键提升幅度
    ret_top1_improve = (ml_ret_top1 - wfo_ret_top1) * 100  # 百分点
    sharpe_top1_improve = ml_sharpe_top1 - wfo_sharpe_top1
    dd_top1_improve = (ml_dd_top1 - wfo_dd_top1) * 100  # 百分点
    
    ret_mean_improve = (ml_ret_mean - wfo_ret_mean) * 100
    sharpe_mean_improve = ml_sharpe_mean - wfo_sharpe_mean
    
    conclusions = []
    
    # Top-1 表现
    if ret_top1_improve > 1.0:
        conclusions.append(f"✅ **Top-1 组合年化收益**: ML排序 **优于** WFO排序 (+{ret_top1_improve:.2f}%)")
    elif ret_top1_improve < -1.0:
        conclusions.append(f"⚠️ **Top-1 组合年化收益**: ML排序 **劣于** WFO排序 ({ret_top1_improve:.2f}%)")
    else:
        conclusions.append(f"➡️ **Top-1 组合年化收益**: ML排序与WFO排序 **相当** ({ret_top1_improve:+.2f}%)")
    
    if sharpe_top1_improve > 0.05:
        conclusions.append(f"✅ **Top-1 组合Sharpe**: ML排序 **明显优于** WFO排序 (+{sharpe_top1_improve:.3f})")
    elif sharpe_top1_improve < -0.05:
        conclusions.append(f"⚠️ **Top-1 组合Sharpe**: ML排序 **劣于** WFO排序 ({sharpe_top1_improve:.3f})")
    else:
        conclusions.append(f"➡️ **Top-1 组合Sharpe**: ML排序与WFO排序 **相当** ({sharpe_top1_improve:+.3f})")
    
    if abs(dd_top1_improve) < 1.0:
        conclusions.append(f"➡️ **Top-1 组合最大回撤**: ML排序与WFO排序 **相当** ({dd_top1_improve:+.2f}%)")
    elif dd_top1_improve < 0:
        conclusions.append(f"✅ **Top-1 组合最大回撤**: ML排序 **更低** (改善 {abs(dd_top1_improve):.2f}%)")
    else:
        conclusions.append(f"⚠️ **Top-1 组合最大回撤**: ML排序 **更高** (恶化 {dd_top1_improve:.2f}%)")
    
    report.append("\n".join(conclusions))
    report.append("")
    
    # 整体平均表现
    report.append("### 整体平均表现")
    report.append("")
    if ret_mean_improve > 0.5:
        report.append(f"- **平均年化收益**: ML排序 **优于** WFO排序 (+{ret_mean_improve:.2f}%)")
    elif ret_mean_improve < -0.5:
        report.append(f"- **平均年化收益**: ML排序 **劣于** WFO排序 ({ret_mean_improve:.2f}%)")
    else:
        report.append(f"- **平均年化收益**: ML排序与WFO排序 **相当** ({ret_mean_improve:+.2f}%)")
    
    if sharpe_mean_improve > 0.02:
        report.append(f"- **平均Sharpe**: ML排序 **优于** WFO排序 (+{sharpe_mean_improve:.3f})")
    elif sharpe_mean_improve < -0.02:
        report.append(f"- **平均Sharpe**: ML排序 **劣于** WFO排序 ({sharpe_mean_improve:.3f})")
    else:
        report.append(f"- **平均Sharpe**: ML排序与WFO排序 **相当** ({sharpe_mean_improve:+.3f})")
    
    report.append("")
    
    # 最终建议
    report.append("### 最终建议")
    report.append("")
    
    # 综合评分: Top-1 表现 + 平均表现
    score = 0
    if ret_top1_improve > 1.0:
        score += 2
    elif ret_top1_improve > 0:
        score += 1
    
    if sharpe_top1_improve > 0.1:
        score += 2
    elif sharpe_top1_improve > 0:
        score += 1
    
    if dd_top1_improve < -1.0:  # 回撤改善
        score += 1
    elif dd_top1_improve < 0:
        score += 0.5
    
    if ret_mean_improve > 0.5:
        score += 1
    
    if sharpe_mean_improve > 0.05:
        score += 1
    
    if score >= 5:
        report.append("✅ **强烈推荐**: ML排序在 Top-1 和整体表现上均明显优于WFO排序，**建议替换为生产默认排序**。")
    elif score >= 3:
        report.append("✅ **推荐**: ML排序表现优于WFO排序，**可以考虑替换为生产默认排序**，但建议先在小规模环境验证。")
    elif score >= 1:
        report.append("➡️ **谨慎**: ML排序与WFO排序表现相当，**暂不建议替换**，可以作为备选方案继续观察。")
    else:
        report.append("⚠️ **不推荐**: ML排序表现不如WFO排序，**不建议替换为生产默认排序**，需要优化模型或特征工程。")
    
    report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="对比 WFO 原排序 vs ML 排序的回测结果"
    )
    parser.add_argument(
        "--wfo-report",
        type=str,
        required=True,
        help="WFO 原排序的回测结果 CSV 文件路径"
    )
    parser.add_argument(
        "--ml-report",
        type=str,
        required=True,
        help="ML 排序的回测结果 CSV 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存对比报告的 Markdown 文件路径 (可选)"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"📂 加载 WFO 原排序结果: {args.wfo_report}")
    wfo_df = load_backtest_results(args.wfo_report)
    print(f"   样本数: {len(wfo_df)}")
    
    print(f"📂 加载 ML 排序结果: {args.ml_report}")
    ml_df = load_backtest_results(args.ml_report)
    print(f"   样本数: {len(ml_df)}")
    print()
    
    # 提取指标
    print("📊 提取关键指标...")
    wfo_metrics = extract_summary_metrics(wfo_df)
    ml_metrics = extract_summary_metrics(ml_df)
    print("   完成")
    print()
    
    # 生成报告
    print("📝 生成对比报告...")
    report = generate_markdown_report(wfo_metrics, ml_metrics, args.wfo_report, args.ml_report)
    print()
    
    # 输出到控制台
    print("="*80)
    print(report)
    print("="*80)
    print()
    
    # 保存到文件 (可选)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"✅ 报告已保存至: {output_path}")
    else:
        print("ℹ️ 未指定 --output,报告未保存到文件")


if __name__ == "__main__":
    main()
