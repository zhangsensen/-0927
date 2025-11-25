#!/usr/bin/env python3
"""
排序问题诊断脚本

目标：
1. 验证数据泄漏问题
2. 检查特征工程质量
3. 分析排序逻辑错误
4. 评估集成权重合理性
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_data_leakage():
    """诊断数据泄漏问题"""
    logger.info("=" * 80)
    logger.info("🔍 诊断1: 数据泄漏问题")
    logger.info("=" * 80)
    
    # 检查最新校准器训练
    calibrator_path = Path("results/calibrator_gbdt_profit.joblib")
    if not calibrator_path.exists():
        logger.warning("❌ 校准器文件不存在")
        return
    
    # 检查训练历史
    try:
        import joblib
        calibrator_data = joblib.load(calibrator_path)
        train_history = calibrator_data.get('train_history', [])
        
        if train_history:
            latest = train_history[-1]
            logger.info(f"✅ 最新训练记录:")
            logger.info(f"   - 样本数: {latest.get('n_samples', 'N/A')}")
            logger.info(f"   - 训练R²: {latest.get('train_r2', 'N/A'):.4f}")
            logger.info(f"   - CV R²: {latest.get('r2_cv_mean', 'N/A'):.4f} ± {latest.get('r2_cv_std', 'N/A'):.4f}")
            
            # 关键诊断：R²为负值表明特征无法预测目标
            if latest.get('train_r2', 0) < 0:
                logger.warning("⚠️  训练R²为负值，特征无法预测年化收益！")
                logger.warning("   这说明存在严重的数据泄漏或特征工程问题")
            
        else:
            logger.warning("❌ 无训练历史记录")
            
    except Exception as e:
        logger.error(f"❌ 读取校准器失败: {e}")

def diagnose_feature_quality():
    """诊断特征工程质量"""
    logger.info("=" * 80)
    logger.info("🔍 诊断2: 特征工程质量")
    logger.info("=" * 80)
    
    # 检查WFO结果中的特征
    latest_run = None
    results_dir = Path("results")
    for run_dir in sorted(results_dir.glob("run_*"), reverse=True):
        if run_dir.is_dir():
            latest_run = run_dir
            break
    
    if not latest_run:
        logger.error("❌ 未找到WFO运行结果")
        return
    
    all_combos_file = latest_run / "all_combos.parquet"
    if not all_combos_file.exists():
        logger.error("❌ 未找到all_combos.parquet")
        return
    
    try:
        wfo_df = pd.read_parquet(all_combos_file)
        
        # 检查关键特征
        key_features = ["mean_oos_ic", "oos_ic_std", "positive_rate", "stability_score", "combo_size", "best_rebalance_freq"]
        missing_features = [f for f in key_features if f not in wfo_df.columns]
        
        logger.info(f"✅ WFO数据: {len(wfo_df)} 个组合")
        logger.info(f"✅ 关键特征检查:")
        
        for feature in key_features:
            if feature in wfo_df.columns:
                stats = wfo_df[feature].describe()
                logger.info(f"   - {feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, missing={wfo_df[feature].isna().sum()}")
            else:
                logger.warning(f"   - {feature}: ❌ 缺失")
        
        if missing_features:
            logger.error(f"❌ 缺失特征: {missing_features}")
        
        # 检查IC与目标变量的相关性
        if "mean_oos_ic" in wfo_df.columns:
            ic_stats = wfo_df["mean_oos_ic"].describe()
            logger.info(f"\n📊 IC统计:")
            logger.info(f"   - 均值: {ic_stats['mean']:.4f}")
            logger.info(f"   - 标准差: {ic_stats['std']:.4f}")
            logger.info(f"   - 范围: [{ic_stats['min']:.4f}, {ic_stats['max']:.4f}]")
            
            # 检查IC分布是否合理
            if ic_stats['mean'] < 0.02:  # IC均值过低
                logger.warning("⚠️  IC均值过低，因子预测能力可能不足")
            
    except Exception as e:
        logger.error(f"❌ 分析WFO数据失败: {e}")

def diagnose_ranking_logic():
    """诊断排序逻辑问题"""
    logger.info("=" * 80)
    logger.info("🔍 诊断3: 排序逻辑问题")
    logger.info("=" * 80)
    
    # 检查集成排序结果
    enhanced_ranking_file = Path("test_enhanced_ranking.csv")
    stats_file = Path("stats_test_enhanced_ranking.json")
    
    if not enhanced_ranking_file.exists():
        logger.warning("❌ 增强排序结果文件不存在")
        return
    
    if not stats_file.exists():
        logger.warning("❌ 排序统计文件不存在")
        return
    
    try:
        # 读取排序结果
        df = pd.read_csv(enhanced_ranking_file)
        
        # 读取统计信息
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        logger.info(f"✅ 排序数据: {len(df)} 个策略")
        logger.info(f"✅ 排序改进: {stats.get('ranking_improvement', 'N/A')}")
        
        # 分析排序改进为负的原因
        ranking_improvement = stats.get('ranking_improvement', 0)
        if ranking_improvement < 0:
            logger.error(f"❌ 排序改进为负值: {ranking_improvement}")
            logger.error("   这说明集成方法恶化了排序质量")
            
            # 检查集成权重
            ensemble_weights = stats.get('ensemble_weights', {})
            logger.info(f"📊 集成权重: {ensemble_weights}")
            
            # 分析权重合理性
            total_weight = sum(ensemble_weights.values())
            logger.info(f"   - 总权重: {total_weight}")
            
            for name, weight in ensemble_weights.items():
                logger.info(f"   - {name}: {weight}")
        
        # 分析排序一致性
        if 'original_rank' in df.columns and 'enhanced_rank' in df.columns and 'final_rank' in df.columns:
            # 计算排序相关性
            orig_enh_corr, _ = spearmanr(df['original_rank'], df['enhanced_rank'])
            orig_final_corr, _ = spearmanr(df['original_rank'], df['final_rank'])
            enh_final_corr, _ = spearmanr(df['enhanced_rank'], df['final_rank'])
            
            logger.info(f"\n📊 排序相关性:")
            logger.info(f"   - 原始 vs 增强: {orig_enh_corr:.4f}")
            logger.info(f"   - 原始 vs 最终: {orig_final_corr:.4f}")
            logger.info(f"   - 增强 vs 最终: {enh_final_corr:.4f}")
            
            # 检查是否有排序倒置
            rank_changes = (df['original_rank'] != df['final_rank']).sum()
            logger.info(f"   - 排序变化数: {rank_changes}/{len(df)} ({rank_changes/len(df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ 分析排序逻辑失败: {e}")

def diagnose_ic_vs_returns():
    """诊断IC与真实收益的相关性"""
    logger.info("=" * 80)
    logger.info("🔍 诊断4: IC与真实收益相关性")
    logger.info("=" * 80)
    
    # 查找最新的回测结果
    results_combo_dir = Path("results_combo_wfo")
    if not results_combo_dir.exists():
        logger.warning("❌ 回测结果目录不存在")
        return
    
    # 查找最新的回测结果
    latest_backtest = None
    for backtest_dir in sorted(results_combo_dir.glob("*"), reverse=True):
        if backtest_dir.is_dir():
            csv_files = list(backtest_dir.glob("*.csv"))
            if csv_files:
                latest_backtest = backtest_dir
                break
    
    if not latest_backtest:
        logger.warning("❌ 未找到回测结果")
        return
    
    try:
        # 读取回测结果
        csv_files = list(latest_backtest.glob("*.csv"))
        backtest_dfs = []
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if 'combo' in df.columns and 'annual_ret' in df.columns and 'sharpe' in df.columns:
                backtest_dfs.append(df)
        
        if backtest_dfs:
            backtest_df = pd.concat(backtest_dfs, ignore_index=True)
            backtest_df = backtest_df.drop_duplicates(subset=['combo'], keep='last')
            
            logger.info(f"✅ 回测数据: {len(backtest_df)} 个策略")
            
            # 分析收益分布
            ret_stats = backtest_df['annual_ret'].describe()
            sharpe_stats = backtest_df['sharpe'].describe()
            
            logger.info(f"📊 年化收益统计:")
            logger.info(f"   - 均值: {ret_stats['mean']:.4f}")
            logger.info(f"   - 标准差: {ret_stats['std']:.4f}")
            logger.info(f"   - 范围: [{ret_stats['min']:.4f}, {ret_stats['max']:.4f}]")
            
            logger.info(f"📊 Sharpe统计:")
            logger.info(f"   - 均值: {sharpe_stats['mean']:.4f}")
            logger.info(f"   - 标准差: {sharpe_stats['std']:.4f}")
            logger.info(f"   - 范围: [{sharpe_stats['min']:.4f}, {sharpe_stats['max']:.4f}]")
            
            # 检查是否有足够的变化
            ret_range = ret_stats['max'] - ret_stats['min']
            sharpe_range = sharpe_stats['max'] - sharpe_stats['min']
            
            logger.info(f"\n📊 变化范围:")
            logger.info(f"   - 年化收益范围: {ret_range:.4f}")
            logger.info(f"   - Sharpe范围: {sharpe_range:.4f}")
            
            if ret_range < 0.10:  # 年化收益变化小于10%
                logger.warning("⚠️  年化收益变化范围较小，可能难以区分策略")
            
            if sharpe_range < 0.5:  # Sharpe变化小于0.5
                logger.warning("⚠️  Sharpe变化范围较小，可能难以区分策略")
        
    except Exception as e:
        logger.error(f"❌ 分析回测结果失败: {e}")

def main():
    """主诊断函数"""
    logger.info("🚀 开始排序问题诊断")
    logger.info("=" * 80)
    
    # 执行各项诊断
    diagnose_data_leakage()
    diagnose_feature_quality()
    diagnose_ranking_logic()
    diagnose_ic_vs_returns()
    
    logger.info("=" * 80)
    logger.info("✅ 诊断完成")
    logger.info("=" * 80)
    
    # 总结关键发现
    logger.info("\n🎯 关键发现总结:")
    logger.info("1. 检查校准器训练R²是否为负值")
    logger.info("2. 检查IC均值是否过低(<0.02)")
    logger.info("3. 检查排序改进是否为负值")
    logger.info("4. 检查收益变化范围是否过小")
    logger.info("5. 检查集成权重是否合理")

if __name__ == "__main__":
    main()