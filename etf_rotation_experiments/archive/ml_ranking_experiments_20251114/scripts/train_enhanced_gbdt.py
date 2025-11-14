# -*- coding: utf-8 -*-
"""
@author: Copilot
@created: 2025-11-13
@description: 使用增强特征训练 GBDT 模型
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
import json
import os

def main():
    """主函数"""
    run_id = '20251112_223854'
    
    print("="*80)
    print("🚀 GBDT 增强特征训练实验")
    print("="*80)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    
    # 加载回测结果 (真实 Sharpe)
    backtest_files = [
        f'results_combo_wfo/{run_id}_20251113_112641/top3000_profit_backtest_slip2bps_{run_id}_20251113_112641.csv',
        f'results_combo_wfo/{run_id}_20251113_112650/top3000_profit_backtest_slip2bps_{run_id}_20251113_112650.csv',
        f'results_combo_wfo/{run_id}_20251113_112657/top3000_profit_backtest_slip2bps_{run_id}_20251113_112657.csv',
        f'results_combo_wfo/{run_id}_20251113_112716/top3000_profit_backtest_slip2bps_{run_id}_20251113_112716.csv',
        f'results_combo_wfo/{run_id}_20251113_113852/top3000_profit_backtest_slip2bps_{run_id}_20251113_113852.csv',
        f'results_combo_wfo/{run_id}_20251113_114159/top3000_profit_backtest_slip2bps_{run_id}_20251113_114159.csv',
        f'results_combo_wfo/{run_id}_20251113_114610/top3000_profit_backtest_slip2bps_{run_id}_20251113_114610.csv',
    ]
    
    backtest_results = []
    for f in backtest_files:
        try:
            df = pd.read_csv(f)
            if 'combo' in df.columns and 'sharpe' in df.columns:
                backtest_results.append(df[['combo', 'sharpe']])
        except FileNotFoundError:
            pass
    
    if not backtest_results:
        print("❌ 错误: 找不到回测结果文件。")
        return
    
    backtest_df = pd.concat(backtest_results, ignore_index=True)
    backtest_df = backtest_df.drop_duplicates(subset=['combo'], keep='last')
    backtest_df.columns = ['combo', 'sharpe_real']
    
    print(f"✅ 回测结果: {len(backtest_df)} 个策略")
    
    # 加载 WFO 特征
    wfo_df = pd.read_parquet(f'results/run_{run_id}/ranking_blends/ranking_baseline.parquet')
    print(f"✅ WFO 特征: {len(wfo_df)} 个策略, {len(wfo_df.columns)} 个特征")
    
    # 加载增强特征 (注意: generate_enhanced_features.py 保存到了 results/RUNID/ 而非 results/run_RUNID/)
    enhanced_df = pd.read_parquet(f'results/{run_id}/enhanced_features.parquet')
    print(f"✅ 增强特征: {len(enhanced_df)} 个策略, {len(enhanced_df.columns)} 个特征")
    
    # 2. 合并数据
    print("\n🔗 合并数据...")
    merged_df = backtest_df.merge(wfo_df, on='combo', how='inner')
    merged_df = merged_df.merge(enhanced_df, on='combo', how='inner', suffixes=('_wfo', '_enhanced'))
    
    print(f"✅ 合并后: {len(merged_df)} 个策略, {len(merged_df.columns)} 个列")
    
    # 3. 准备特征
    print("\n📊 准备特征...")
    
    # WFO 基础特征
    wfo_features = ['mean_oos_ic', 'oos_ic_std', 'positive_rate', 'stability_score', 'combo_size']
    
    # 增强特征 (从回测摘要中提取的)
    enhanced_features = [
        'calmar_ratio',
        'sortino_ratio', 
        'profit_factor',
        'win_rate',
        'max_consecutive_wins',
        'max_consecutive_losses',
        'sharpe_calmar_product',
        'dd_recovery_ratio',
        'sortino_sharpe_ratio',
        'win_rate_profit_composite',
        'win_loss_streak_ratio'
    ]
    
    # 检查特征可用性
    available_wfo = [f for f in wfo_features if f in merged_df.columns]
    available_enhanced = [f for f in enhanced_features if f in merged_df.columns]
    
    # 处理 combo_size 列名冲突
    if 'combo_size' not in merged_df.columns:
        if 'combo_size_wfo' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_wfo']
        elif 'combo_size_enhanced' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_enhanced']
    
    all_features = available_wfo + available_enhanced
    
    print(f"   WFO 特征 ({len(available_wfo)}): {available_wfo}")
    print(f"   增强特征 ({len(available_enhanced)}): {available_enhanced}")
    print(f"   总计: {len(all_features)} 个特征")
    
    # 提取特征和目标
    X = merged_df[all_features].values
    y = merged_df['sharpe_real'].values
    
    # 处理 inf 和 nan
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 4. 训练测试划分
    print("\n✂️  划分数据...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 5. 训练模型
    print("\n" + "-"*80)
    print("🌲 训练增强 GBDT 模型...")
    print("-"*80)
    
    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=50,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # 6. 评估
    print("\n📊 模型评估...")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_spearman = spearmanr(y_train, y_train_pred)[0]
    test_spearman = spearmanr(y_test, y_test_pred)[0]
    
    print(f"   训练集 Spearman: {train_spearman:.4f}")
    print(f"   测试集 Spearman: {test_spearman:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 重要特征:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # 7. 保存模型
    output_dir = f'results/run_{run_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'gbdt_enhanced.joblib')
    joblib.dump(model, model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 保存特征列表
    feature_config = {
        'features': all_features,
        'wfo_features': available_wfo,
        'enhanced_features': available_enhanced,
        'train_spearman': float(train_spearman),
        'test_spearman': float(test_spearman)
    }
    
    config_path = os.path.join(output_dir, 'gbdt_enhanced_config.json')
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"✅ 配置已保存: {config_path}")
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, 'gbdt_enhanced_feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"✅ 特征重要性已保存: {importance_path}")
    
    print("\n" + "="*80)
    print("🎯 训练完成!")
    print("="*80)
    print(f"\n📊 性能对比 (与基础 GBDT):")
    print(f"   基础 GBDT (5 个特征):  测试集 Spearman = 0.7129")
    print(f"   增强 GBDT ({len(all_features)} 个特征): 测试集 Spearman = {test_spearman:.4f}")
    print(f"   改进幅度: {(test_spearman - 0.7129):.4f} ({(test_spearman/0.7129 - 1)*100:+.1f}%)")
    
    print("\n💡 下一步:")
    print("   1. 运行 'validate_ml_ranking_accuracy.py' 验证新模型的排序效果")
    print("   2. 如果效果提升,应用到完整的 run_20251112_223854")
    print("   3. 重新生成 ML 排序并验证 Top3000")

if __name__ == '__main__':
    main()
