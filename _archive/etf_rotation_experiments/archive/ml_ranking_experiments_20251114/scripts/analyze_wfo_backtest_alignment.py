#!/usr/bin/env python3
"""
WFO排序与真实回测排序一致性分析脚本

功能：
1. 按combo列合并WFO的mean_oos_sharpe与回测的sharpe_net/annual_ret_net
2. 计算Spearman/Kendall秩相关系数
3. 统计Top-K（K=100/1000）重叠度
4. 输出JSON报告与可选对照表
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from scipy.stats import spearmanr, kendalltau


def analyze_ranking_alignment(
    run_dir: Path,
    backtest_csv: Optional[Path] = None,
    strategy: str = "oos_sharpe_true",
    top_k_list: list[int] = None,
) -> Dict:
    """
    分析WFO排序与回测排序的一致性。
    
    Args:
        run_dir: WFO运行目录，包含排名文件
        backtest_csv: 回测输出CSV，若为None则自动查找
        strategy: 排序策略（ic/oos_sharpe_proxy/oos_sharpe_true）
        top_k_list: 重叠度统计的K值列表，默认[100, 1000]
    
    Returns:
        包含相关系数、重叠度统计的字典
    """
    if top_k_list is None:
        top_k_list = [100, 1000]
    
    # 1. 读取WFO排名
    wfo_rank_candidates = [
        run_dir / f"ranking_{strategy}_top5000.parquet",
        run_dir / f"ranking_{strategy}_top1000.parquet",
        run_dir / "top_combos.parquet",
        run_dir / "all_combos.parquet",
    ]
    wfo_rank_file = next((p for p in wfo_rank_candidates if p.exists()), None)
    if wfo_rank_file is None:
        raise FileNotFoundError(f"未找到WFO排名文件于: {run_dir}")
    
    wfo_df = pd.read_parquet(wfo_rank_file)
    
    # 确定WFO主指标
    if strategy == "oos_sharpe_true":
        wfo_metric = "mean_oos_sharpe"
    elif strategy == "oos_sharpe_proxy":
        wfo_metric = "oos_sharpe_proxy"
    else:
        wfo_metric = "mean_oos_ic"
    
    if wfo_metric not in wfo_df.columns:
        raise ValueError(f"WFO数据缺少主指标列: {wfo_metric}")
    
    wfo_rank = wfo_df[["combo", wfo_metric]].dropna().copy()
    
    # 2. 读取回测结果
    if backtest_csv is None:
        # 自动查找最近的回测CSV
        bt_candidates = sorted(
            run_dir.glob("*_profit_backtest_*.csv"), 
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if not bt_candidates:
            raise FileNotFoundError(f"未找到回测CSV于: {run_dir}")
        backtest_csv = bt_candidates[0]
    
    bt_df = pd.read_csv(backtest_csv)
    
    # 确定回测主指标（优先净值后，回退基准）
    bt_metric_candidates = ["sharpe_net", "annual_ret_net", "sharpe", "annual_ret"]
    bt_metric = next((c for c in bt_metric_candidates if c in bt_df.columns), None)
    if bt_metric is None:
        raise ValueError(f"回测数据缺少可用指标: {bt_metric_candidates}")
    
    bt_rank = bt_df[["combo", bt_metric]].dropna().copy()
    
    # 3. 合并数据集
    merged = wfo_rank.merge(bt_rank, on="combo", how="inner")
    n_common = len(merged)
    
    if n_common < 10:
        raise ValueError(f"共同组合数不足10个（实际: {n_common}），无法进行秩相关分析")
    
    # 4. 计算秩相关
    spearman_corr, spearman_p = spearmanr(merged[wfo_metric], merged[bt_metric])
    kendall_corr, kendall_p = kendalltau(merged[wfo_metric], merged[bt_metric])
    
    # 5. 计算Top-K重叠
    overlap_stats = {}
    for K in top_k_list:
        wfo_topk = set(wfo_rank.nlargest(K, wfo_metric)["combo"])
        bt_topk = set(merged.nlargest(K, bt_metric)["combo"])
        overlap = len(wfo_topk & bt_topk)
        overlap_rate = overlap / max(1, len(wfo_topk))
        overlap_stats[f"top{K}"] = {
            "overlap_count": overlap,
            "wfo_topk_count": len(wfo_topk),
            "bt_topk_count": len(bt_topk),
            "overlap_rate": overlap_rate,
        }
    
    # 6. 汇总报告
    report = {
        "run_dir": str(run_dir),
        "backtest_csv": str(backtest_csv),
        "strategy": strategy,
        "wfo_metric": wfo_metric,
        "bt_metric": bt_metric,
        "n_wfo_combos": len(wfo_rank),
        "n_bt_combos": len(bt_rank),
        "n_common_combos": n_common,
        "rank_correlation": {
            "spearman": {"rho": float(spearman_corr), "p_value": float(spearman_p)},
            "kendall": {"tau": float(kendall_corr), "p_value": float(kendall_p)},
        },
        "top_k_overlap": overlap_stats,
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="WFO与回测排序一致性分析")
    parser.add_argument("--run-dir", type=str, required=True, help="WFO运行目录路径")
    parser.add_argument("--backtest-csv", type=str, default=None, help="回测CSV路径（可选，自动查找）")
    parser.add_argument("--strategy", type=str, default="oos_sharpe_true", 
                       choices=["ic", "oos_sharpe_proxy", "oos_sharpe_true"],
                       help="排序策略")
    parser.add_argument("--top-k", type=int, nargs="+", default=[100, 1000],
                       help="Top-K重叠统计的K值列表")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON路径（默认：run_dir/ranking_alignment_report.json）")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    backtest_csv = Path(args.backtest_csv).resolve() if args.backtest_csv else None
    
    print(f"分析目录: {run_dir}")
    print(f"策略: {args.strategy}")
    if backtest_csv:
        print(f"回测CSV: {backtest_csv}")
    
    report = analyze_ranking_alignment(
        run_dir=run_dir,
        backtest_csv=backtest_csv,
        strategy=args.strategy,
        top_k_list=args.top_k,
    )
    
    # 输出结果
    output_path = Path(args.output) if args.output else (run_dir / "ranking_alignment_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 报告已保存: {output_path}")
    print(f"\n📊 关键指标:")
    print(f"  - 共同组合数: {report['n_common_combos']}")
    print(f"  - Spearman ρ: {report['rank_correlation']['spearman']['rho']:.4f} "
          f"(p={report['rank_correlation']['spearman']['p_value']:.4e})")
    print(f"  - Kendall τ: {report['rank_correlation']['kendall']['tau']:.4f} "
          f"(p={report['rank_correlation']['kendall']['p_value']:.4e})")
    for k, stats in report["top_k_overlap"].items():
        print(f"  - {k.upper()} 重叠: {stats['overlap_count']}/{stats['wfo_topk_count']} "
              f"({stats['overlap_rate']:.1%})")


if __name__ == "__main__":
    main()
