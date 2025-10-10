#!/usr/bin/env python3
"""
修复前后结果对比分析脚本
作者：量化首席工程师
日期：2025-10-06
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_summary_json(json_path: Path) -> Dict:
    """加载summary JSON文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_timeframe_results(before_dir: Path, after_dir: Path) -> pd.DataFrame:
    """对比分析时间框架结果"""

    results = []

    # 获取所有时间框架
    timeframes = [
        "1min",
        "2min",
        "3min",
        "5min",
        "15min",
        "30min",
        "60min",
        "2h",
        "4h",
        "1day",
    ]

    for tf in timeframes:
        before_tf_dir = before_dir / "timeframes" / tf
        after_tf_dir = after_dir / "timeframes" / tf

        if not before_tf_dir.exists() or not after_tf_dir.exists():
            print(f"⚠️ 跳过 {tf}: 目录不存在")
            continue

        # 查找screening_statistics.json
        before_stats = None
        after_stats = None

        for file in before_tf_dir.glob("*_screening_statistics_*.json"):
            with open(file, "r", encoding="utf-8") as f:
                before_stats = json.load(f)
            break

        for file in after_tf_dir.glob("*_screening_statistics_*.json"):
            with open(file, "r", encoding="utf-8") as f:
                after_stats = json.load(f)
            break

        if not before_stats or not after_stats:
            print(f"⚠️ 跳过 {tf}: 统计文件不存在")
            continue

        # 提取关键指标
        row = {
            "timeframe": tf,
            "before_total": before_stats.get("total_factors", 0),
            "after_total": after_stats.get("total_factors", 0),
            "before_significant": before_stats.get("significant_factors", 0),
            "after_significant": after_stats.get("significant_factors", 0),
            "before_high_score": before_stats.get("high_score_factors", 0),
            "after_high_score": after_stats.get("high_score_factors", 0),
            "before_tier1": before_stats.get("tier1_count", 0),
            "after_tier1": after_stats.get("tier1_count", 0),
            "before_tier2": before_stats.get("tier2_count", 0),
            "after_tier2": after_stats.get("tier2_count", 0),
            "before_avg_score": before_stats.get("average_score", 0),
            "after_avg_score": after_stats.get("average_score", 0),
            "before_alpha": before_stats.get("adaptive_alpha", 0.05),
            "after_alpha": after_stats.get("adaptive_alpha", 0.05),
        }

        # 计算改进百分比
        row["sig_improve"] = (
            (
                after_stats.get("significant_factors", 0)
                - before_stats.get("significant_factors", 0)
            )
            / max(before_stats.get("significant_factors", 1), 1)
            * 100
        )

        row["high_score_improve"] = (
            (
                after_stats.get("high_score_factors", 0)
                - before_stats.get("high_score_factors", 0)
            )
            / max(before_stats.get("high_score_factors", 1), 1)
            * 100
        )

        results.append(row)

    return pd.DataFrame(results)


def generate_comparison_report(df: pd.DataFrame, output_path: Path):
    """生成对比报告"""

    report = []
    report.append("=" * 100)
    report.append("🔍 0700.HK 因子筛选修复 - 前后对比分析报告")
    report.append("=" * 100)
    report.append("")

    report.append("## 📊 修复内容")
    report.append("1. ✅ 时间框架自适应alpha（1min:0.05 → 4h/1day:0.10）")
    report.append("2. ✅ 样本量权重修正（防止长周期虚高分）")
    report.append("3. ✅ 时间框架自适应高分阈值（1min:0.60 → 1day:0.53）")
    report.append("4. ✅ 对齐失败策略（可配置warn/fail_fast/fallback）")
    report.append("")

    report.append("## 📈 关键指标对比")
    report.append("")

    # 显著因子改进
    report.append("### 1. 显著因子数量变化")
    report.append("")
    report.append(
        f"{'时间框架':<10} {'修复前':<10} {'修复后':<10} {'改进%':<10} {'Alpha前':<10} {'Alpha后':<10}"
    )
    report.append("-" * 70)

    # Linus优化：向量化字符串生成，避免iterrows()
    report_lines = [
        f"{row.timeframe:<10} "
        f"{int(row.before_significant):<10} "
        f"{int(row.after_significant):<10} "
        f"{row.sig_improve:>+8.1f}% "
        f"{row.before_alpha:<10.3f} "
        f"{row.after_alpha:<10.3f}"
        for row in df.itertuples(index=False)
    ]
    report.extend(report_lines)

    report.append("")

    # 高分因子改进
    report.append("### 2. 高分因子数量变化")
    report.append("")
    report.append(f"{'时间框架':<10} {'修复前':<10} {'修复后':<10} {'改进%':<10}")
    report.append("-" * 50)

    # Linus优化：向量化字符串生成
    high_score_lines = [
        f"{row.timeframe:<10} "
        f"{int(row.before_high_score):<10} "
        f"{int(row.after_high_score):<10} "
        f"{row.high_score_improve:>+8.1f}%"
        for row in df.itertuples(index=False)
    ]
    report.extend(high_score_lines)

    report.append("")

    # Tier分布对比
    report.append("### 3. 因子等级分布变化")
    report.append("")
    report.append(
        f"{'时间框架':<10} {'Tier1(前)':<12} {'Tier1(后)':<12} {'Tier2(前)':<12} {'Tier2(后)':<12}"
    )
    report.append("-" * 60)

    # Linus优化：向量化字符串生成
    tier_lines = [
        f"{row.timeframe:<10} "
        f"{int(row.before_tier1):<12} "
        f"{int(row.after_tier1):<12} "
        f"{int(row.before_tier2):<12} "
        f"{int(row.after_tier2):<12}"
        for row in df.itertuples(index=False)
    ]
    report.extend(tier_lines)

    report.append("")

    # 总结
    report.append("## 🎯 修复效果总结")
    report.append("")

    total_sig_before = df["before_significant"].sum()
    total_sig_after = df["after_significant"].sum()
    total_high_before = df["before_high_score"].sum()
    total_high_after = df["after_high_score"].sum()

    report.append(
        f"✅ 显著因子总数: {int(total_sig_before)} → {int(total_sig_after)} "
        f"(+{((total_sig_after-total_sig_before)/total_sig_before*100):+.1f}%)"
    )
    report.append(
        f"✅ 高分因子总数: {int(total_high_before)} → {int(total_high_after)} "
        f"(+{((total_high_after-total_high_before)/max(total_high_before,1)*100):+.1f}%)"
    )
    report.append("")

    # 最大改进
    max_sig_improve = df.loc[df["sig_improve"].idxmax()]
    report.append(
        f"🏆 显著因子改进最大: {max_sig_improve['timeframe']} "
        f"({max_sig_improve['sig_improve']:+.1f}%)"
    )

    max_high_improve = df.loc[df["high_score_improve"].idxmax()]
    report.append(
        f"🏆 高分因子改进最大: {max_high_improve['timeframe']} "
        f"({max_high_improve['high_score_improve']:+.1f}%)"
    )

    report.append("")
    report.append("=" * 100)

    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # 同时打印到控制台
    print("\n".join(report))


def main():
    """主函数"""

    # 获取输出目录
    output_base = Path("output")

    # 查找修复前后的目录
    dirs = sorted([d for d in output_base.glob("0700.HK_multi_tf_*") if d.is_dir()])

    if len(dirs) < 2:
        print(f"❌ 需要至少2个会话目录进行对比，当前找到 {len(dirs)} 个")
        sys.exit(1)

    # 假设倒数第二个是修复前，最后一个是修复后
    before_dir = dirs[-2]
    after_dir = dirs[-1]

    print(f"📂 修复前目录: {before_dir.name}")
    print(f"📂 修复后目录: {after_dir.name}")
    print("")

    # 分析对比
    df = analyze_timeframe_results(before_dir, after_dir)

    if df.empty:
        print("❌ 无法提取对比数据")
        sys.exit(1)

    # 生成报告
    report_path = after_dir / "comparison_report.txt"
    generate_comparison_report(df, report_path)

    # 保存CSV
    csv_path = after_dir / "comparison_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📊 对比数据已保存: {csv_path}")
    print(f"📄 对比报告已保存: {report_path}")


if __name__ == "__main__":
    main()
