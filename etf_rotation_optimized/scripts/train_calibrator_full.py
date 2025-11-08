#!/usr/bin/env python3
"""
全量数据GBDT校准器训练脚本

使用全量12,597组合的回测结果训练校准器，相比Top2000样本:
- 样本量增加6.3倍（2000 → 12597）
- 覆盖IC全范围（-0.04 ~ 0.16），避免样本偏差
- 提升模型泛化能力，减少过拟合风险
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.wfo_realbt_calibrator import WFORealBacktestCalibrator
import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_full_backtest_csv():
    """查找最新的全量回测CSV文件"""
    results_dir = Path("results_combo_wfo")
    
    # 查找包含full或all关键字的CSV文件（按修改时间排序）
    csv_files = []
    for csv_path in results_dir.rglob("*full*.csv"):
        if "all" in csv_path.stem or "12597" in csv_path.stem or "combos" in csv_path.stem:
            csv_files.append((csv_path, csv_path.stat().st_mtime))
    
    if not csv_files:
        raise FileNotFoundError(
            f"未找到全量回测CSV文件。请确保已完成全量回测 (scripts/run_full_backtest_all_combos.sh)"
        )
    
    # 返回最新的文件
    latest_csv = sorted(csv_files, key=lambda x: x[1], reverse=True)[0][0]
    return latest_csv


def main():
    logger.info("="*80)
    logger.info("全量GBDT校准器训练")
    logger.info("="*80)
    
    # 1. 加载WFO结果
    wfo_path = Path("results/run_20251108_193712/all_combos.parquet")
    if not wfo_path.exists():
        logger.error(f"WFO结果不存在: {wfo_path}")
        return
    
    logger.info(f"\n加载WFO结果: {wfo_path}")
    wfo_df = pd.read_parquet(wfo_path)
    logger.info(f"  组合数: {len(wfo_df)}")
    logger.info(f"  IC范围: {wfo_df['mean_oos_ic'].min():.4f} ~ {wfo_df['mean_oos_ic'].max():.4f}")
    
    # 2. 查找并加载全量回测结果
    try:
        backtest_csv = find_latest_full_backtest_csv()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\n请先运行全量回测:")
        logger.info("  bash scripts/run_full_backtest_all_combos.sh")
        return
    
    logger.info(f"\n加载全量回测结果: {backtest_csv}")
    backtest_df = pd.read_csv(backtest_csv)
    logger.info(f"  回测组合数: {len(backtest_df)}")
    logger.info(f"  Sharpe范围: {backtest_df['sharpe'].min():.3f} ~ {backtest_df['sharpe'].max():.3f}")
    
    # 3. 合并数据并检查覆盖率
    merged = wfo_df.merge(backtest_df[['combo', 'sharpe', 'annual_ret', 'max_dd']], on='combo', how='inner')
    coverage = len(merged) / len(wfo_df) * 100
    
    logger.info(f"\n数据合并:")
    logger.info(f"  匹配组合数: {len(merged)}/{len(wfo_df)} ({coverage:.1f}%)")
    
    if coverage < 95:
        logger.warning(f"⚠️  覆盖率不足95%，可能回测未完成或数据缺失")
    
    # 4. 对比Top2000与全量样本的分布差异
    logger.info(f"\n{'='*80}")
    logger.info("样本分布对比（Top2000 vs 全量）")
    logger.info(f"{'='*80}")
    
    top2000_combos = set(wfo_df.nlargest(2000, 'mean_oos_ic')['combo'])
    merged['in_top2000'] = merged['combo'].isin(top2000_combos)
    
    top2000_samples = merged[merged['in_top2000']]
    rest_samples = merged[~merged['in_top2000']]
    
    logger.info(f"Top2000样本:")
    logger.info(f"  数量: {len(top2000_samples)}")
    logger.info(f"  IC: {top2000_samples['mean_oos_ic'].mean():.4f} ± {top2000_samples['mean_oos_ic'].std():.4f}")
    logger.info(f"  Sharpe: {top2000_samples['sharpe'].mean():.3f} ± {top2000_samples['sharpe'].std():.3f}")
    
    logger.info(f"\n其余样本 ({len(rest_samples)}):")
    logger.info(f"  IC: {rest_samples['mean_oos_ic'].mean():.4f} ± {rest_samples['mean_oos_ic'].std():.4f}")
    logger.info(f"  Sharpe: {rest_samples['sharpe'].mean():.3f} ± {rest_samples['sharpe'].std():.3f}")
    
    # 5. 训练全量GBDT校准器
    logger.info(f"\n{'='*80}")
    logger.info("训练GBDT校准器（全量数据）")
    logger.info(f"{'='*80}")
    
    calibrator_full = WFORealBacktestCalibrator(
        model_type="gbdt",
        n_estimators=300,  # 增加树数量（样本更多）
        max_depth=5,       # 增加深度（捕捉更复杂关系）
        cv_folds=10,       # 增加CV折数（更稳健）
    )
    
    metrics_full = calibrator_full.fit(merged, backtest_df, target_metric='sharpe')
    
    logger.info(f"\n全量模型训练评估:")
    for k, v in metrics_full.items():
        if isinstance(v, float):
            logger.info(f"  {k:25s}: {v:.4f}")
        else:
            logger.info(f"  {k:25s}: {v}")
    
    # 6. 与Top2000模型对比（若存在）
    top2000_model_path = Path("results/calibrator_gbdt_best.joblib")
    if top2000_model_path.exists():
        logger.info(f"\n{'='*80}")
        logger.info("对比Top2000模型 vs 全量模型")
        logger.info(f"{'='*80}")
        
        calibrator_top2000 = WFORealBacktestCalibrator.load(top2000_model_path)
        
        # 在全量测试集上对比两个模型
        merged['pred_top2000'] = calibrator_top2000.predict(merged)
        merged['pred_full'] = calibrator_full.predict(merged)
        
        corr_top2000, _ = spearmanr(merged['sharpe'], merged['pred_top2000'])
        corr_full, _ = spearmanr(merged['sharpe'], merged['pred_full'])
        
        logger.info(f"全量测试集（{len(merged)}组合）:")
        logger.info(f"  Top2000模型 Spearman: {corr_top2000:+.4f}")
        logger.info(f"  全量模型 Spearman:    {corr_full:+.4f}")
        logger.info(f"  提升:                 {(corr_full-corr_top2000):+.4f} ({(corr_full-corr_top2000)/abs(corr_top2000)*100:+.1f}%)")
        
        # Precision@K对比
        for k in [50, 100, 200, 500]:
            # Top2000模型
            top2000_topk = set(merged.nlargest(k, 'pred_top2000')['combo'])
            real_topk = set(merged.nlargest(k, 'sharpe')['combo'])
            prec_top2000 = len(top2000_topk & real_topk) / k
            
            # 全量模型
            full_topk = set(merged.nlargest(k, 'pred_full')['combo'])
            prec_full = len(full_topk & real_topk) / k
            
            logger.info(f"\nPrecision@{k}:")
            logger.info(f"  Top2000模型: {prec_top2000*100:5.1f}%")
            logger.info(f"  全量模型:    {prec_full*100:5.1f}%")
            logger.info(f"  提升:        {(prec_full-prec_top2000)*100:+5.1f}%")
    
    # 7. 特征重要性
    logger.info(f"\n{'='*80}")
    logger.info("特征重要性（全量模型）")
    logger.info(f"{'='*80}")
    importance = calibrator_full.analyze_feature_importance()
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # 8. 保存全量模型
    full_model_path = Path("results/calibrator_gbdt_full.joblib")
    calibrator_full.save(full_model_path)
    logger.info(f"\n✅ 全量校准器已保存: {full_model_path}")
    
    # 9. 生成可视化对比图
    logger.info(f"\n{'='*80}")
    logger.info("生成可视化对比图")
    logger.info(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: WFO IC vs 实际Sharpe（原始）
    axes[0, 0].scatter(merged['mean_oos_ic'], merged['sharpe'], alpha=0.2, s=5, c='gray')
    axes[0, 0].set_xlabel('WFO Mean OOS IC', fontsize=11)
    axes[0, 0].set_ylabel('Actual Sharpe', fontsize=11)
    corr_orig, _ = spearmanr(merged['mean_oos_ic'], merged['sharpe'])
    axes[0, 0].set_title(f'Original WFO IC vs Sharpe (ρ={corr_orig:+.3f})', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 全量模型预测 vs 实际Sharpe
    axes[0, 1].scatter(merged['pred_full'], merged['sharpe'], alpha=0.2, s=5, c='green')
    axes[0, 1].set_xlabel('Calibrated Pred (Full Model)', fontsize=11)
    axes[0, 1].set_ylabel('Actual Sharpe', fontsize=11)
    axes[0, 1].set_title(f'Full GBDT vs Sharpe (ρ={corr_full:+.3f})', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 样本分布对比（IC分布）
    axes[1, 0].hist(top2000_samples['mean_oos_ic'], bins=50, alpha=0.6, label='Top2000', color='blue')
    axes[1, 0].hist(rest_samples['mean_oos_ic'], bins=50, alpha=0.6, label=f'Rest ({len(rest_samples)})', color='orange')
    axes[1, 0].set_xlabel('Mean OOS IC', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('IC Distribution (Top2000 vs Rest)', fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 样本分布对比（Sharpe分布）
    axes[1, 1].hist(top2000_samples['sharpe'], bins=50, alpha=0.6, label='Top2000', color='blue')
    axes[1, 1].hist(rest_samples['sharpe'], bins=50, alpha=0.6, label=f'Rest ({len(rest_samples)})', color='orange')
    axes[1, 1].set_xlabel('Actual Sharpe', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Sharpe Distribution (Top2000 vs Rest)', fontsize=13)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path("results/calibrator_full_vs_top2000_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ 可视化图表已保存: {fig_path}")
    
    # 10. 校准全量WFO结果
    logger.info(f"\n{'='*80}")
    logger.info("校准全量WFO结果")
    logger.info(f"{'='*80}")
    
    wfo_df['calibrated_sharpe_full'] = calibrator_full.predict(wfo_df)
    
    calibrated_path = Path("results/run_20251108_193712/all_combos_calibrated_gbdt_full.parquet")
    wfo_df.to_parquet(calibrated_path, index=False)
    logger.info(f"✅ 校准结果已保存: {calibrated_path}")
    
    # 11. 生成校准后Top2000白名单
    calibrated_top2000 = wfo_df.nlargest(2000, 'calibrated_sharpe_full')
    whitelist_path = Path("results/run_20251108_193712/whitelist_top2000_calibrated_gbdt_full.txt")
    calibrated_top2000['combo'].to_csv(whitelist_path, index=False, header=False)
    logger.info(f"✅ 校准Top2000白名单已保存: {whitelist_path}")
    
    # 对比排序变化
    original_top2000 = set(wfo_df.nlargest(2000, 'mean_oos_ic')['combo'])
    calibrated_top2000_set = set(calibrated_top2000['combo'])
    overlap = len(original_top2000 & calibrated_top2000_set)
    
    logger.info(f"\n排序变化:")
    logger.info(f"  原始WFO Top2000 vs 校准Top2000 重叠: {overlap}/2000 ({overlap/20:.1f}%)")
    logger.info(f"  {2000-overlap} 个组合被替换")
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ 全量校准器训练完成！")
    logger.info(f"{'='*80}")
    logger.info(f"\n关键输出:")
    logger.info(f"  1. 全量模型: {full_model_path}")
    logger.info(f"  2. 校准结果: {calibrated_path}")
    logger.info(f"  3. 白名单:   {whitelist_path}")
    logger.info(f"  4. 可视化:   {fig_path}")


if __name__ == "__main__":
    main()
