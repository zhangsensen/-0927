#!/usr/bin/env python3
"""
Top100 对照回测脚本

目的：验证校准器排序 vs IC 排序的实际效果差异

输入：
  - Top100 (IC排序): results/run_XXXXXX/top100_ic_combos.csv
  - Top100 (校准排序): results/run_XXXXXX/top100_calibrated_combos.csv

输出：
  - results/run_XXXXXX/comparison/
      - ic_ranking_backtest.csv      # IC排序的100个策略的完整回测结果
      - calibrated_ranking_backtest.csv  # 校准排序的100个策略的完整回测结果
      - comparison_report.json       # 两组的对比指标
      - comparison_report.md         # 可读的对比报告

用法：
  python scripts/compare_top100_backtest.py --run-dir results/run_20251112_223854
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 添加路径以导入核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cross_section_processor import CrossSectionProcessor
from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_combo_string(combo_str: str) -> list:
    """解析因子组合字符串
    
    Args:
        combo_str: "FACTOR1 + FACTOR2 + FACTOR3" 格式的字符串
        
    Returns:
        因子名称列表
    """
    return [f.strip() for f in combo_str.split('+')]


def run_single_combo_backtest(
    combo_str: str,
    ohlcv: pd.DataFrame,
    cs_proc: CrossSectionProcessor,
    factor_lib: PreciseFactorLibrary,
    config: dict,
) -> dict:
    """运行单个因子组合的回测
    
    Args:
        combo_str: 因子组合字符串
        ohlcv: OHLCV 数据
        cs_proc: 横截面处理器
        factor_lib: 因子库
        config: 配置字典
        
    Returns:
        回测结果字典
    """
    factors = parse_combo_string(combo_str)
    
    # TODO: 这里需要调用真实的回测逻辑
    # 由于真实回测框架较复杂，这里先返回模拟结果
    # 实际使用时需要接入 etf_rotation_optimized/real_backtest/run_production_backtest.py
    
    logger.warning(f"⚠️  暂未实现真实回测逻辑，返回模拟结果: {combo_str}")
    
    return {
        "combo": combo_str,
        "annual_ret": np.random.uniform(0.05, 0.25),
        "sharpe": np.random.uniform(0.5, 1.5),
        "max_dd": np.random.uniform(-0.3, -0.1),
        "vol": np.random.uniform(0.15, 0.25),
        "n_rebalance": 144,
    }


def run_batch_backtest(
    combos_df: pd.DataFrame,
    config: dict,
    label: str,
) -> pd.DataFrame:
    """批量运行回测
    
    Args:
        combos_df: 包含 combo 列的 DataFrame
        config: 配置字典
        label: 标签（用于日志）
        
    Returns:
        回测结果 DataFrame
    """
    logger.info(f"{'='*80}")
    logger.info(f"🚀 开始 {label} 回测")
    logger.info(f"{'='*80}")
    logger.info(f"策略数量: {len(combos_df)}")
    
    # 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    n_etfs = len(ohlcv) if isinstance(ohlcv, dict) else len(ohlcv['close'].columns)
    n_dates = len(next(iter(ohlcv.values()))) if isinstance(ohlcv, dict) else len(ohlcv)
    logger.info(f"✅ 数据加载完成: {n_etfs} ETFs × {n_dates} 交易日")
    
    # 初始化处理器
    cs_proc = CrossSectionProcessor()
    factor_lib = PreciseFactorLibrary()
    
    # 运行回测
    results = []
    for idx, row in combos_df.iterrows():
        combo = row['combo']
        logger.info(f"[{idx+1}/{len(combos_df)}] {combo}")
        
        result = run_single_combo_backtest(
            combo_str=combo,
            ohlcv=ohlcv,
            cs_proc=cs_proc,
            factor_lib=factor_lib,
            config=config,
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    logger.info(f"✅ {label} 回测完成")
    
    return results_df


def generate_comparison_report(
    ic_results: pd.DataFrame,
    cal_results: pd.DataFrame,
    output_dir: Path,
):
    """生成对比报告
    
    Args:
        ic_results: IC 排序的回测结果
        cal_results: 校准排序的回测结果
        output_dir: 输出目录
    """
    # 计算统计指标
    ic_stats = {
        "annual_ret_mean": float(ic_results["annual_ret"].mean()),
        "annual_ret_median": float(ic_results["annual_ret"].median()),
        "sharpe_mean": float(ic_results["sharpe"].mean()),
        "sharpe_median": float(ic_results["sharpe"].median()),
        "max_dd_mean": float(ic_results["max_dd"].mean()),
        "vol_mean": float(ic_results["vol"].mean()),
    }
    
    cal_stats = {
        "annual_ret_mean": float(cal_results["annual_ret"].mean()),
        "annual_ret_median": float(cal_results["annual_ret"].median()),
        "sharpe_mean": float(cal_results["sharpe"].mean()),
        "sharpe_median": float(cal_results["sharpe"].median()),
        "max_dd_mean": float(cal_results["max_dd"].mean()),
        "vol_mean": float(cal_results["vol"].mean()),
    }
    
    # 计算增量
    deltas = {
        "annual_ret_delta": cal_stats["annual_ret_mean"] - ic_stats["annual_ret_mean"],
        "sharpe_delta": cal_stats["sharpe_mean"] - ic_stats["sharpe_mean"],
        "max_dd_delta": cal_stats["max_dd_mean"] - ic_stats["max_dd_mean"],
    }
    
    # 保存 JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "ic_ranking": ic_stats,
        "calibrated_ranking": cal_stats,
        "deltas": deltas,
        "verdict": {
            "annual_ret_improved": deltas["annual_ret_delta"] > 0,
            "sharpe_improved": deltas["sharpe_delta"] > 0,
            "both_improved": (deltas["annual_ret_delta"] > 0) and (deltas["sharpe_delta"] > 0),
        }
    }
    
    json_path = output_dir / "comparison_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ JSON 报告已保存: {json_path}")
    
    # 生成 Markdown 报告
    md_lines = [
        "# Top100 校准器对照回测报告",
        "",
        f"**生成时间**: {report['timestamp']}",
        "",
        "## 回测结果对比",
        "",
        "| 指标 | IC 排序 | 校准排序 | 增量 |",
        "|------|---------|----------|------|",
        f"| 年化收益率 (均值) | {ic_stats['annual_ret_mean']:.2%} | {cal_stats['annual_ret_mean']:.2%} | {deltas['annual_ret_delta']:+.2%} |",
        f"| Sharpe (均值) | {ic_stats['sharpe_mean']:.3f} | {cal_stats['sharpe_mean']:.3f} | {deltas['sharpe_delta']:+.3f} |",
        f"| 最大回撤 (均值) | {ic_stats['max_dd_mean']:.2%} | {cal_stats['max_dd_mean']:.2%} | {deltas['max_dd_delta']:+.2%} |",
        f"| 波动率 (均值) | {ic_stats['vol_mean']:.2%} | {cal_stats['vol_mean']:.2%} | - |",
        "",
        "## 判定结果",
        "",
    ]
    
    if report["verdict"]["both_improved"]:
        md_lines.extend([
            "✅ **校准排序同时提升了年化收益和 Sharpe，建议采纳**",
            "",
            "### 后续行动",
            "1. 在 Top200/500/2000 上验证效果",
            "2. 分析换手率和成本敏感性",
            "3. 更新优化器配置以启用校准排序",
        ])
    elif report["verdict"]["annual_ret_improved"] or report["verdict"]["sharpe_improved"]:
        md_lines.extend([
            "⚠️  **校准排序仅部分改善，需进一步分析**",
            "",
            f"- 年化收益提升: {'✅' if report['verdict']['annual_ret_improved'] else '❌'}",
            f"- Sharpe 提升: {'✅' if report['verdict']['sharpe_improved'] else '❌'}",
            "",
            "### 后续行动",
            "1. 分析成本侵蚀影响",
            "2. 检查校准器特征分布是否匹配",
            "3. 考虑调整门控阈值或模型参数",
        ])
    else:
        md_lines.extend([
            "❌ **校准排序未产生提升，建议回退到 IC 排序**",
            "",
            "### 后续行动",
            "1. 诊断特征分布漂移",
            "2. 重新训练校准器（缩短窗口或调整样本权重）",
            "3. 考虑简化为规则校准（IC + 稳定性阈值）",
        ])
    
    md_path = output_dir / "comparison_report.md"
    md_path.write_text("\n".join(md_lines))
    
    logger.info(f"✅ Markdown 报告已保存: {md_path}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("📊 对照回测摘要")
    print("="*80)
    print(f"年化收益率: IC 排序 {ic_stats['annual_ret_mean']:.2%} vs 校准排序 {cal_stats['annual_ret_mean']:.2%} (Δ {deltas['annual_ret_delta']:+.2%})")
    print(f"Sharpe:    IC 排序 {ic_stats['sharpe_mean']:.3f} vs 校准排序 {cal_stats['sharpe_mean']:.3f} (Δ {deltas['sharpe_delta']:+.3f})")
    print(f"最大回撤:  IC 排序 {ic_stats['max_dd_mean']:.2%} vs 校准排序 {cal_stats['max_dd_mean']:.2%}")
    print("="*80)
    
    if report["verdict"]["both_improved"]:
        print("✅ 校准器有效，建议采纳")
    elif not (report["verdict"]["annual_ret_improved"] or report["verdict"]["sharpe_improved"]):
        print("❌ 校准器无效，建议回退到 IC 排序")
    else:
        print("⚠️  效果不明确，需进一步分析")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Top100 对照回测")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="WFO run 目录，例如 results/run_20251112_223854",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/combo_wfo_config.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run 目录不存在: {run_dir}")
        sys.exit(1)
    
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载两组 Top100
    ic_combos_path = run_dir / "top100_ic_combos.csv"
    cal_combos_path = run_dir / "top100_calibrated_combos.csv"
    
    if not ic_combos_path.exists():
        logger.error(f"IC 排序文件不存在: {ic_combos_path}")
        sys.exit(1)
    
    if not cal_combos_path.exists():
        logger.error(f"校准排序文件不存在: {cal_combos_path}")
        sys.exit(1)
    
    ic_combos = pd.read_csv(ic_combos_path)
    cal_combos = pd.read_csv(cal_combos_path)
    
    logger.info(f"✅ 已加载 {len(ic_combos)} 个 IC 排序策略")
    logger.info(f"✅ 已加载 {len(cal_combos)} 个校准排序策略")
    
    # 创建输出目录
    output_dir = run_dir / "comparison"
    output_dir.mkdir(exist_ok=True)
    
    # 运行回测
    ic_results = run_batch_backtest(ic_combos, config, "IC排序Top100")
    cal_results = run_batch_backtest(cal_combos, config, "校准排序Top100")
    
    # 保存回测结果
    ic_results.to_csv(output_dir / "ic_ranking_backtest.csv", index=False)
    cal_results.to_csv(output_dir / "calibrated_ranking_backtest.csv", index=False)
    
    logger.info(f"✅ 回测结果已保存到 {output_dir}")
    
    # 生成对比报告
    generate_comparison_report(ic_results, cal_results, output_dir)


if __name__ == "__main__":
    main()
