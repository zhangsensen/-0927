#!/usr/bin/env python3
"""
ML 排序准确度验证脚本

目标：
    在 Top3000 范围内验证 ML 校准器的排序准确度

核心指标：
    1. Spearman 排序相关性（ML预测 vs 真实Sharpe）
    2. Top-K 命中率（ML选出的TopK，有多少在真实TopK里）
    3. 分层准确度（Top100/500/1000/3000的排序质量）
    4. 排序提升幅度（ML vs IC排序的相对提升）

用法：
    python scripts/validate_ml_ranking_accuracy.py \
        --run-dir results/run_20251112_223854 \
        --topk 3000 \
        --slippage-bps 2.0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_topk_precision(
    ml_ranking: pd.DataFrame,
    true_ranking: pd.DataFrame,
    topk_list: List[int],
) -> Dict[int, float]:
    """
    计算 Top-K 命中率
    
    Args:
        ml_ranking: ML 排序结果 (按 calibrated_sharpe_pred 降序)
        true_ranking: 真实排序结果 (按 sharpe 降序)
        topk_list: 要评估的 K 值列表
    
    Returns:
        {K: precision} - 各 K 值的精确率
    """
    precisions = {}
    
    for k in topk_list:
        if k > len(ml_ranking) or k > len(true_ranking):
            continue
        
        ml_topk = set(ml_ranking.head(k)['combo'].values)
        true_topk = set(true_ranking.head(k)['combo'].values)
        
        intersection = ml_topk & true_topk
        precision = len(intersection) / k
        precisions[k] = precision
    
    return precisions


def compute_stratified_correlation(
    merged: pd.DataFrame,
    strata: List[int],
) -> Dict[str, float]:
    """
    计算分层的 Spearman 相关性
    
    Args:
        merged: 合并后的数据 (包含 calibrated_sharpe_pred 和 sharpe)
        strata: 分层边界，如 [100, 500, 1000, 3000]
    
    Returns:
        {f"top{k}": corr} - 各层级的相关系数
    """
    correlations = {}
    
    for k in strata:
        if k > len(merged):
            k = len(merged)
        
        subset = merged.head(k)
        if len(subset) < 10:  # 至少10个样本才计算
            continue
        
        corr, pval = spearmanr(
            subset['calibrated_sharpe_pred'],
            subset['sharpe']
        )
        
        correlations[f'top{k}'] = {
            'spearman': float(corr),
            'pvalue': float(pval),
            'n_samples': len(subset),
        }
    
    return correlations


def analyze_ranking_improvement(
    ic_ranking: pd.DataFrame,
    ml_ranking: pd.DataFrame,
    backtest_results: pd.DataFrame,
) -> Dict:
    """
    对比 IC 排序 vs ML 排序的实际效果提升
    
    Returns:
        改进分析结果
    """
    # 合并回测结果
    ic_merged = ic_ranking.merge(
        backtest_results[['combo', 'annual_ret', 'sharpe', 'max_dd']],
        on='combo',
        how='inner'
    )
    
    ml_merged = ml_ranking.merge(
        backtest_results[['combo', 'annual_ret', 'sharpe', 'max_dd']],
        on='combo',
        how='inner'
    )
    
    improvements = {}
    
    # 各 TopK 层级的效果对比
    for k in [10, 50, 100, 500, 1000]:
        if k > len(ic_merged) or k > len(ml_merged):
            continue
        
        ic_topk = ic_merged.head(k)
        ml_topk = ml_merged.head(k)
        
        improvements[f'top{k}'] = {
            'ic_sorting': {
                'annual_ret': float(ic_topk['annual_ret'].mean()),
                'sharpe': float(ic_topk['sharpe'].mean()),
                'max_dd': float(ic_topk['max_dd'].mean()),
            },
            'ml_sorting': {
                'annual_ret': float(ml_topk['annual_ret'].mean()),
                'sharpe': float(ml_topk['sharpe'].mean()),
                'max_dd': float(ml_topk['max_dd'].mean()),
            },
            'delta': {
                'annual_ret': float(ml_topk['annual_ret'].mean() - ic_topk['annual_ret'].mean()),
                'sharpe': float(ml_topk['sharpe'].mean() - ic_topk['sharpe'].mean()),
                'max_dd': float(ml_topk['max_dd'].mean() - ic_topk['max_dd'].mean()),
            }
        }
    
    return improvements


def main():
    parser = argparse.ArgumentParser(description='验证 ML 排序准确度')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='WFO run 目录')
    parser.add_argument('--topk', type=int, default=3000,
                        help='验证范围 (默认 3000)')
    parser.add_argument('--slippage-bps', type=float, default=2.0,
                        help='滑点 (基点)')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    topk = args.topk
    
    print("=" * 80)
    print("🔍 ML 排序准确度验证")
    print("=" * 80)
    print(f"\nRun 目录: {run_dir}")
    print(f"验证范围: Top{topk}")
    print(f"滑点设置: {args.slippage_bps} bps")
    
    # 1. 加载排名文件
    print("\n" + "=" * 80)
    print("📂 加载排名数据")
    print("=" * 80)
    
    ic_ranking_file = run_dir / 'ranking_blends/ranking_baseline.parquet'
    ml_ranking_file = run_dir / 'ranking_blends/ranking_lightgbm.parquet'
    
    if not ic_ranking_file.exists() or not ml_ranking_file.exists():
        print(f"❌ 排名文件不存在")
        print(f"   IC: {ic_ranking_file}")
        print(f"   ML: {ml_ranking_file}")
        return
    
    ic_ranking = pd.read_parquet(ic_ranking_file)
    ml_ranking = pd.read_parquet(ml_ranking_file)
    
    # 按排序字段排序并截取 TopK
    ic_ranking = ic_ranking.nlargest(topk, 'mean_oos_ic').copy()
    ml_ranking = ml_ranking.nlargest(topk, 'calibrated_sharpe_pred').copy()
    
    print(f"✅ IC 排序: {len(ic_ranking)} 个策略")
    print(f"✅ ML 排序: {len(ml_ranking)} 个策略")
    
    # 2. 检查是否需要运行回测
    print("\n" + "=" * 80)
    print("🔍 检查回测结果")
    print("=" * 80)
    
    # 查找已有的回测结果
    results_dir = Path('results_combo_wfo')
    run_id = run_dir.name.replace('run_', '')
    
    # 查找包含这个 run_id 的回测结果目录
    backtest_dirs = list(results_dir.glob(f'{run_id}_*'))
    
    if backtest_dirs:
        print(f"✅ 找到 {len(backtest_dirs)} 个回测结果目录")
        for d in backtest_dirs:
            csv_files = list(d.glob('*.csv'))
            if csv_files:
                print(f"   - {d.name}: {len(csv_files)} 个CSV文件")
    
    # 检查是否有足够的回测数据
    all_backtest_results = []
    for d in backtest_dirs:
        for csv_file in d.glob('*.csv'):
            df = pd.read_csv(csv_file)
            if 'combo' in df.columns and 'sharpe' in df.columns:
                all_backtest_results.append(df)
    
    if all_backtest_results:
        backtest_results = pd.concat(all_backtest_results, ignore_index=True)
        # 去重（可能有重复的回测）
        backtest_results = backtest_results.drop_duplicates(subset=['combo'], keep='last')
        print(f"\n✅ 已有回测结果: {len(backtest_results)} 个策略")
    else:
        backtest_results = None
        print(f"\n⚠️  未找到回测结果")
    
    # 3. 判断是否需要运行新的回测
    if backtest_results is None or len(backtest_results) < topk * 0.8:
        print("\n" + "=" * 80)
        print("🚀 准备运行回测")
        print("=" * 80)
        
        # 准备 Top3000 的排名文件
        output_dir = run_dir / 'backtest_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ic_topk_file = output_dir / f'ranking_ic_top{topk}.parquet'
        ml_topk_file = output_dir / f'ranking_ml_top{topk}.parquet'
        
        # 添加 rank_score 列（回测脚本需要）
        ic_ranking['rank_score'] = ic_ranking['mean_oos_ic']
        ml_ranking['rank_score'] = ml_ranking['calibrated_sharpe_pred']
        
        ic_ranking.to_parquet(ic_topk_file, index=False)
        ml_ranking.to_parquet(ml_topk_file, index=False)
        
        print(f"✅ 已生成排名文件:")
        print(f"   IC: {ic_topk_file}")
        print(f"   ML: {ml_topk_file}")
        
        print(f"\n" + "=" * 80)
        print(f"⏳ 请运行以下命令启动回测 (预计耗时 {topk * 1.5 / 60:.0f}-{topk * 3 / 60:.0f} 分钟):")
        print(f"=" * 80)
        
        print(f"\n# IC 排序回测")
        print(f"nohup python real_backtest/run_profit_backtest.py \\")
        print(f"  --topk {topk} \\")
        print(f"  --ranking-file {ic_topk_file} \\")
        print(f"  --slippage-bps {args.slippage_bps} \\")
        print(f"  > /tmp/ic_top{topk}_backtest.log 2>&1 &")
        
        print(f"\n# ML 排序回测")
        print(f"nohup python real_backtest/run_profit_backtest.py \\")
        print(f"  --topk {topk} \\")
        print(f"  --ranking-file {ml_topk_file} \\")
        print(f"  --slippage-bps {args.slippage_bps} \\")
        print(f"  > /tmp/ml_top{topk}_backtest.log 2>&1 &")
        
        print(f"\n回测完成后，重新运行此脚本进行分析。")
        return
    
    # 4. 执行排序准确度分析
    print("\n" + "=" * 80)
    print("📊 排序准确度分析")
    print("=" * 80)
    
    # 合并 ML 预测和真实结果
    ml_with_truth = ml_ranking.merge(
        backtest_results[['combo', 'sharpe', 'annual_ret', 'max_dd']],
        on='combo',
        how='inner'
    )
    
    print(f"\n数据覆盖: {len(ml_with_truth)} / {topk} ({len(ml_with_truth)/topk*100:.1f}%)")
    
    # 4.1 整体 Spearman 相关性
    overall_corr, overall_p = spearmanr(
        ml_with_truth['calibrated_sharpe_pred'],
        ml_with_truth['sharpe']
    )
    
    print(f"\n【整体排序相关性】")
    print(f"  Spearman 相关系数: {overall_corr:.4f} (p={overall_p:.4e})")
    
    # 4.2 分层相关性
    strata = [100, 500, 1000, 2000, 3000]
    strata = [s for s in strata if s <= topk]
    
    stratified_corr = compute_stratified_correlation(
        ml_with_truth.sort_values('calibrated_sharpe_pred', ascending=False),
        strata
    )
    
    print(f"\n【分层排序相关性】")
    for layer, stats in stratified_corr.items():
        print(f"  {layer:8s}: Spearman {stats['spearman']:+.4f} (p={stats['pvalue']:.4e}, n={stats['n_samples']})")
    
    # 4.3 Top-K 命中率
    true_ranking = backtest_results.sort_values('sharpe', ascending=False)
    ml_ranking_sorted = ml_ranking.sort_values('calibrated_sharpe_pred', ascending=False)
    
    topk_list = [10, 50, 100, 500, 1000, 2000, 3000]
    topk_list = [k for k in topk_list if k <= topk]
    
    precisions = compute_topk_precision(ml_ranking_sorted, true_ranking, topk_list)
    
    print(f"\n【Top-K 命中率】")
    print(f"  (ML 选出的 TopK 中，有多少真的在真实 TopK 里)")
    for k, prec in precisions.items():
        print(f"  Top{k:4d}: {prec*100:5.1f}%")
    
    # 4.4 对比 IC 排序的改进
    ic_with_truth = ic_ranking.merge(
        backtest_results[['combo', 'sharpe', 'annual_ret', 'max_dd']],
        on='combo',
        how='inner'
    )
    
    ic_corr, ic_p = spearmanr(
        ic_with_truth['mean_oos_ic'],
        ic_with_truth['sharpe']
    )
    
    print(f"\n【IC 排序 vs ML 排序】")
    print(f"  IC 排序 Spearman: {ic_corr:.4f}")
    print(f"  ML 排序 Spearman: {overall_corr:.4f}")
    print(f"  相关性提升: {(overall_corr - ic_corr):+.4f} ({(overall_corr - ic_corr)/abs(ic_corr)*100:+.1f}%)")
    
    # 4.5 实际效果提升
    improvements = analyze_ranking_improvement(
        ic_ranking.sort_values('mean_oos_ic', ascending=False),
        ml_ranking_sorted,
        backtest_results
    )
    
    print(f"\n【实际效果提升】")
    for layer, stats in improvements.items():
        print(f"\n  {layer}:")
        print(f"    年化收益: IC {stats['ic_sorting']['annual_ret']:6.2%} → ML {stats['ml_sorting']['annual_ret']:6.2%} (Δ {stats['delta']['annual_ret']:+.2%})")
        print(f"    Sharpe:   IC {stats['ic_sorting']['sharpe']:6.3f} → ML {stats['ml_sorting']['sharpe']:6.3f} (Δ {stats['delta']['sharpe']:+.3f})")
    
    # 5. 保存结果
    print(f"\n" + "=" * 80)
    print("💾 保存分析结果")
    print("=" * 80)
    
    result = {
        'run_dir': str(run_dir),
        'topk': topk,
        'coverage': {
            'total': topk,
            'validated': len(ml_with_truth),
            'coverage_rate': len(ml_with_truth) / topk,
        },
        'overall_correlation': {
            'ic_sorting_spearman': float(ic_corr),
            'ml_sorting_spearman': float(overall_corr),
            'improvement': float(overall_corr - ic_corr),
        },
        'stratified_correlation': stratified_corr,
        'topk_precision': {f'top{k}': prec for k, prec in precisions.items()},
        'performance_improvement': improvements,
    }
    
    output_file = run_dir / f'ml_ranking_validation_top{topk}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存: {output_file}")
    
    # 6. 生成决策建议
    print(f"\n" + "=" * 80)
    print("🎯 决策建议")
    print("=" * 80)
    
    # 判断标准
    is_good_correlation = overall_corr > 0.3
    is_better_than_ic = overall_corr > ic_corr + 0.05
    has_positive_improvement = all(
        improvements[k]['delta']['sharpe'] > 0 
        for k in ['top100', 'top500'] if k in improvements
    )
    
    if is_good_correlation and is_better_than_ic and has_positive_improvement:
        print("✅ ML 排序显著优于 IC 排序，建议采纳")
        print(f"   - 排序相关性达到 {overall_corr:.3f}（提升 {(overall_corr-ic_corr)*100:.1f}%）")
        print(f"   - 各层级效果均有改善")
    elif is_better_than_ic:
        print("⚠️  ML 排序有改善，但提升有限")
        print(f"   - 排序相关性 {overall_corr:.3f}（仅提升 {(overall_corr-ic_corr)*100:.1f}%）")
        print(f"   - 建议继续优化特征工程或模型")
    else:
        print("❌ ML 排序未达预期，需重新审视方案")
        print(f"   - 排序相关性仅 {overall_corr:.3f}")
        print(f"   - 可能问题: 特征选择不当、模型过拟合、训练数据偏差")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
