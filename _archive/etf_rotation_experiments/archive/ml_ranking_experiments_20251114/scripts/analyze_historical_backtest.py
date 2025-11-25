#!/usr/bin/env python3
"""
从历史回测结果中提取对照数据

目标：验证校准器排序 vs IC 排序在真实回测中的表现差异

输入：
  - 历史回测结果: results_combo_wfo/*/top*_full.csv
  - 当前 run 的两组 Top100 排序

输出：
  - 对比报告：两组策略在真实回测中的实际表现

用法：
  python scripts/analyze_historical_backtest.py \
    --backtest-csv results_combo_wfo/20251109_032515_20251110_001325/top12597_backtest_by_ic_20251109_032515_20251110_001325_full.csv \
    --ic-ranking results/run_20251112_223854/top100_ic_combos.csv \
    --calibrated-ranking results/run_20251112_223854/top100_calibrated_combos.csv \
    --output results/run_20251112_223854/historical_comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_backtest_results(backtest_csv: Path) -> pd.DataFrame:
    """加载历史回测结果"""
    logger.info(f"📂 加载回测结果: {backtest_csv}")
    df = pd.read_csv(backtest_csv, low_memory=False)
    logger.info(f"  - 总策略数: {len(df)}")
    logger.info(f"  - 列数: {len(df.columns)}")
    
    # 显示关键列
    key_cols = ['combo', 'annual_ret', 'sharpe', 'max_dd', 'vol', 'n_rebalance', 'avg_turnover']
    available_cols = [c for c in key_cols if c in df.columns]
    logger.info(f"  - 可用列: {available_cols}")
    
    return df


def merge_ranking_with_backtest(
    ranking_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """合并排序与回测结果"""
    logger.info(f"🔗 合并 {label} 排序与回测结果")
    
    # 合并
    merged = ranking_df.merge(
        backtest_df,
        on='combo',
        how='left',
        suffixes=('_rank', '_bt')
    )
    
    # 统计覆盖率
    matched = merged['annual_ret'].notna().sum()
    coverage = matched / len(ranking_df) * 100
    
    logger.info(f"  - 排序策略数: {len(ranking_df)}")
    logger.info(f"  - 匹配到回测: {matched} ({coverage:.1f}%)")
    
    if matched == 0:
        logger.warning(f"⚠️  {label} 没有策略匹配到回测结果！")
    elif coverage < 50:
        logger.warning(f"⚠️  {label} 覆盖率低于50%，结果可能不可靠")
    
    return merged


def compute_stats(df: pd.DataFrame, label: str) -> dict:
    """计算统计指标"""
    # 只计算有回测结果的行
    valid = df[df['annual_ret'].notna()]
    
    if len(valid) == 0:
        logger.error(f"❌ {label} 没有有效的回测数据")
        return {
            "n_valid": 0,
            "annual_ret_mean": None,
            "annual_ret_median": None,
            "sharpe_mean": None,
            "sharpe_median": None,
            "max_dd_mean": None,
            "vol_mean": None,
        }
    
    stats = {
        "n_valid": len(valid),
        "annual_ret_mean": float(valid['annual_ret'].mean()),
        "annual_ret_median": float(valid['annual_ret'].median()),
        "sharpe_mean": float(valid['sharpe'].mean()),
        "sharpe_median": float(valid['sharpe'].median()),
        "max_dd_mean": float(valid['max_dd'].mean()),
        "vol_mean": float(valid['vol'].mean()),
    }
    
    # 可选列
    if 'avg_turnover' in valid.columns:
        stats['avg_turnover_mean'] = float(valid['avg_turnover'].mean())
    if 'n_rebalance' in valid.columns:
        stats['n_rebalance_mean'] = float(valid['n_rebalance'].mean())
    
    logger.info(f"📊 {label} 统计:")
    logger.info(f"  - 有效样本: {stats['n_valid']}")
    logger.info(f"  - 年化收益: {stats['annual_ret_mean']:.2%} (中位数 {stats['annual_ret_median']:.2%})")
    logger.info(f"  - Sharpe: {stats['sharpe_mean']:.3f} (中位数 {stats['sharpe_median']:.3f})")
    logger.info(f"  - 最大回撤: {stats['max_dd_mean']:.2%}")
    
    return stats


def generate_report(
    ic_stats: dict,
    cal_stats: dict,
    output_dir: Path,
    backtest_source: str,
):
    """生成对比报告"""
    
    # 检查是否有有效数据
    if ic_stats['n_valid'] == 0 or cal_stats['n_valid'] == 0:
        logger.error("❌ 无法生成报告：至少一组数据无效")
        
        # 仍然生成一个基本报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "backtest_source": backtest_source,
            "error": "No valid backtest data for one or both rankings",
            "ic_ranking": ic_stats,
            "calibrated_ranking": cal_stats,
        }
        
        json_path = output_dir / "historical_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ 错误报告已保存: {json_path}")
        return
    
    # 计算增量
    deltas = {
        "annual_ret_delta": cal_stats['annual_ret_mean'] - ic_stats['annual_ret_mean'],
        "sharpe_delta": cal_stats['sharpe_mean'] - ic_stats['sharpe_mean'],
        "max_dd_delta": cal_stats['max_dd_mean'] - ic_stats['max_dd_mean'],
    }
    
    # 判定
    verdict = {
        "annual_ret_improved": deltas['annual_ret_delta'] > 0,
        "sharpe_improved": deltas['sharpe_delta'] > 0,
        "both_improved": (deltas['annual_ret_delta'] > 0) and (deltas['sharpe_delta'] > 0),
    }
    
    # 保存 JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "backtest_source": backtest_source,
        "ic_ranking": ic_stats,
        "calibrated_ranking": cal_stats,
        "deltas": deltas,
        "verdict": verdict,
    }
    
    json_path = output_dir / "historical_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ JSON 报告已保存: {json_path}")
    
    # 生成 Markdown 报告
    md_lines = [
        "# 历史回测对照分析报告",
        "",
        f"**生成时间**: {report['timestamp']}",
        f"**回测数据源**: `{backtest_source}`",
        "",
        "## 数据覆盖",
        "",
        f"- IC 排序有效样本: {ic_stats['n_valid']}/100",
        f"- 校准排序有效样本: {cal_stats['n_valid']}/100",
        "",
        "## 真实回测结果对比",
        "",
        "| 指标 | IC 排序 | 校准排序 | 增量 | 提升? |",
        "|------|---------|----------|------|-------|",
    ]
    
    # 年化收益
    md_lines.append(
        f"| 年化收益率 (均值) | {ic_stats['annual_ret_mean']:.2%} | "
        f"{cal_stats['annual_ret_mean']:.2%} | "
        f"{deltas['annual_ret_delta']:+.2%} | "
        f"{'✅' if verdict['annual_ret_improved'] else '❌'} |"
    )
    
    # Sharpe
    md_lines.append(
        f"| Sharpe (均值) | {ic_stats['sharpe_mean']:.3f} | "
        f"{cal_stats['sharpe_mean']:.3f} | "
        f"{deltas['sharpe_delta']:+.3f} | "
        f"{'✅' if verdict['sharpe_improved'] else '❌'} |"
    )
    
    # 最大回撤
    md_lines.append(
        f"| 最大回撤 (均值) | {ic_stats['max_dd_mean']:.2%} | "
        f"{cal_stats['max_dd_mean']:.2%} | "
        f"{deltas['max_dd_delta']:+.2%} | "
        f"{'✅' if deltas['max_dd_delta'] > 0 else '❌'} |"
    )
    
    # 波动率
    md_lines.append(
        f"| 波动率 (均值) | {ic_stats['vol_mean']:.2%} | "
        f"{cal_stats['vol_mean']:.2%} | - | - |"
    )
    
    # 可选指标
    if 'avg_turnover_mean' in ic_stats:
        turnover_delta = cal_stats['avg_turnover_mean'] - ic_stats['avg_turnover_mean']
        md_lines.append(
            f"| 平均换手率 | {ic_stats['avg_turnover_mean']:.2f} | "
            f"{cal_stats['avg_turnover_mean']:.2f} | "
            f"{turnover_delta:+.2f} | - |"
        )
    
    md_lines.extend([
        "",
        "## 判定结果",
        "",
    ])
    
    if verdict['both_improved']:
        md_lines.extend([
            "✅ **校准排序在真实回测中同时提升了年化收益和 Sharpe，证明校准器有效！**",
            "",
            "### 关键发现",
            f"- 年化收益提升: **{deltas['annual_ret_delta']:+.2%}**",
            f"- Sharpe 提升: **{deltas['sharpe_delta']:+.3f}**",
            f"- 最大回撤改善: **{deltas['max_dd_delta']:+.2%}**",
            "",
            "### 后续行动",
            "1. ✅ 校准器验证通过，建议在生产环境启用",
            "2. 扩展验证到 Top200/500 并评估成本敏感性",
            "3. 更新优化器配置，默认使用校准排序",
            "4. 持续监控校准器在新数据上的泛化能力",
        ])
    elif verdict['annual_ret_improved'] or verdict['sharpe_improved']:
        md_lines.extend([
            "⚠️  **校准排序仅部分改善，需权衡利弊**",
            "",
            f"- 年化收益提升: {'✅' if verdict['annual_ret_improved'] else '❌'} ({deltas['annual_ret_delta']:+.2%})",
            f"- Sharpe 提升: {'✅' if verdict['sharpe_improved'] else '❌'} ({deltas['sharpe_delta']:+.3f})",
            "",
            "### 后续行动",
            "1. 分析成本侵蚀和换手影响",
            "2. 检查样本覆盖率是否充分",
            "3. 考虑调整门控阈值（例如要求双指标同时提升）",
            "4. 在更长时间窗口或不同市场环境下验证",
        ])
    else:
        md_lines.extend([
            "❌ **校准排序在真实回测中未产生提升，建议放弃当前校准器**",
            "",
            f"- 年化收益下降: **{deltas['annual_ret_delta']:.2%}**",
            f"- Sharpe 下降: **{deltas['sharpe_delta']:.3f}**",
            "",
            "### 诊断与改进",
            "1. **分布漂移**: 训练集与测试集的特征分布可能不一致",
            "2. **过拟合**: 校准器可能记忆了训练期的噪声模式",
            "3. **特征失效**: WFO 统计特征在新数据上不再有效",
            "",
            "### 后续行动",
            "1. ❌ 停止使用当前校准器，回退到 IC 排序",
            "2. 诊断特征分布：对比训练集与当前 run 的特征分位数",
            "3. 重新训练：缩短训练窗口或增加时间衰减权重",
            "4. 简化模型：考虑用规则替代 GBDT（例如 IC>阈值 且 稳定性>阈值）",
        ])
    
    md_path = output_dir / "historical_comparison.md"
    md_path.write_text("\n".join(md_lines))
    
    logger.info(f"✅ Markdown 报告已保存: {md_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 历史回测对照摘要")
    print("="*80)
    print(f"数据源: {backtest_source}")
    print(f"覆盖率: IC排序 {ic_stats['n_valid']}/100, 校准排序 {cal_stats['n_valid']}/100")
    print("-"*80)
    print(f"年化收益率: IC {ic_stats['annual_ret_mean']:.2%} vs 校准 {cal_stats['annual_ret_mean']:.2%} (Δ {deltas['annual_ret_delta']:+.2%})")
    print(f"Sharpe:    IC {ic_stats['sharpe_mean']:.3f} vs 校准 {cal_stats['sharpe_mean']:.3f} (Δ {deltas['sharpe_delta']:+.3f})")
    print(f"最大回撤:  IC {ic_stats['max_dd_mean']:.2%} vs 校准 {cal_stats['max_dd_mean']:.2%} (Δ {deltas['max_dd_delta']:+.2%})")
    print("="*80)
    
    if verdict['both_improved']:
        print("✅ 校准器在真实回测中有效，建议采纳")
    elif not (verdict['annual_ret_improved'] or verdict['sharpe_improved']):
        print("❌ 校准器在真实回测中无效，建议放弃")
    else:
        print("⚠️  效果不明确，需权衡利弊")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="历史回测对照分析")
    parser.add_argument(
        "--backtest-csv",
        type=str,
        required=True,
        help="历史回测结果 CSV 文件",
    )
    parser.add_argument(
        "--ic-ranking",
        type=str,
        required=True,
        help="IC 排序的 Top100 策略文件",
    )
    parser.add_argument(
        "--calibrated-ranking",
        type=str,
        required=True,
        help="校准排序的 Top100 策略文件",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录",
    )
    args = parser.parse_args()
    
    # 检查文件
    backtest_csv = Path(args.backtest_csv)
    ic_ranking = Path(args.ic_ranking)
    cal_ranking = Path(args.calibrated_ranking)
    
    if not backtest_csv.exists():
        logger.error(f"❌ 回测文件不存在: {backtest_csv}")
        sys.exit(1)
    
    if not ic_ranking.exists():
        logger.error(f"❌ IC 排序文件不存在: {ic_ranking}")
        sys.exit(1)
    
    if not cal_ranking.exists():
        logger.error(f"❌ 校准排序文件不存在: {cal_ranking}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("🚀 历史回测对照分析启动")
    logger.info("="*80)
    
    # 加载数据
    backtest_df = load_backtest_results(backtest_csv)
    ic_df = pd.read_csv(ic_ranking)
    cal_df = pd.read_csv(cal_ranking)
    
    logger.info(f"✅ IC 排序策略数: {len(ic_df)}")
    logger.info(f"✅ 校准排序策略数: {len(cal_df)}")
    
    # 合并排序与回测
    ic_merged = merge_ranking_with_backtest(ic_df, backtest_df, "IC排序")
    cal_merged = merge_ranking_with_backtest(cal_df, backtest_df, "校准排序")
    
    # 保存合并结果
    ic_merged.to_csv(output_dir / "ic_ranking_with_backtest.csv", index=False)
    cal_merged.to_csv(output_dir / "calibrated_ranking_with_backtest.csv", index=False)
    
    logger.info(f"✅ 合并结果已保存到 {output_dir}")
    
    # 计算统计
    ic_stats = compute_stats(ic_merged, "IC排序")
    cal_stats = compute_stats(cal_merged, "校准排序")
    
    # 生成报告
    generate_report(ic_stats, cal_stats, output_dir, str(backtest_csv))


if __name__ == "__main__":
    main()
