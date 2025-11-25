# -*- coding: utf-8 -*-
"""
@author: Copilot  
@created: 2025-11-13
@description: 使用增强GBDT模型预测完整策略集并验证效果
"""
import pandas as pd
import numpy as np
import joblib
import json

def main():
    run_id = '20251112_223854'
    
    print("="*80)
    print("🔬 增强 GBDT 模型预测与验证")
    print("="*80)
    
    # 1. 加载模型和配置
    print("\n📂 加载模型...")
    model = joblib.load(f'results/run_{run_id}/gbdt_enhanced.joblib')
    with open(f'results/run_{run_id}/gbdt_enhanced_config.json', 'r') as f:
        config = json.load(f)
    
    feature_list = config['features']
    print(f"✅ 模型加载完成,使用 {len(feature_list)} 个特征")
    
    # 2. 加载完整策略集的特征
    print("\n📂 加载策略特征...")
    
    # WFO 特征 (12597个策略)
    wfo_df = pd.read_parquet(f'results/run_{run_id}/ranking_blends/ranking_baseline.parquet')
    print(f"✅ WFO 特征: {len(wfo_df)} 个策略")
    
    # 增强特征 (5066个已回测策略)
    enhanced_df = pd.read_parquet(f'results/{run_id}/enhanced_features.parquet')
    print(f"✅ 增强特征: {len(enhanced_df)} 个策略")
    
    # 3. 合并
    print("\n🔗 合并特征...")
    full_df = wfo_df.merge(enhanced_df, on='combo', how='left', suffixes=('_wfo', '_enhanced'))
    
    # 处理 combo_size 冲突
    if 'combo_size' not in full_df.columns:
        if 'combo_size_wfo' in full_df.columns:
            full_df['combo_size'] = full_df['combo_size_wfo']
        elif 'combo_size_enhanced' in full_df.columns:
            full_df['combo_size'] = full_df['combo_size_enhanced']
    
    print(f"✅ 合并后: {len(full_df)} 个策略")
    
    # 检查有多少策略缺少增强特征
    has_enhanced = full_df['calmar_ratio'].notna().sum()
    missing_enhanced = len(full_df) - has_enhanced
    
    print(f"\n📊 特征覆盖:")
    print(f"   有增强特征: {has_enhanced} 个策略")
    print(f"   缺少增强特征: {missing_enhanced} 个策略")
    
    if missing_enhanced > 0:
        print(f"\n⚠️  警告: {missing_enhanced} 个策略缺少增强特征!")
        print("   这些策略未参与回测,无法获得 sortino/profit_factor 等指标。")
        print("   模型将无法准确预测这些策略。")
        print("\n❌ 结论: 增强特征方案**不可行**用于预测未回测策略!")
        print("   我们只能对已回测的5066个策略进行排序,这失去了ML的意义。")
        return
    
    # 4. 提取特征并预测
    print("\n🎯 生成预测...")
    X = full_df[feature_list].values
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    
    predictions = model.predict(X)
    full_df['ml_score_enhanced'] = predictions
    
    # 5. 生成新排序
    full_df = full_df.sort_values('ml_score_enhanced', ascending=False).reset_index(drop=True)
    full_df['rank_enhanced'] = range(1, len(full_df) + 1)
    
    # 6. 保存结果
    output_path = f'results/run_{run_id}/ranking_blends/ranking_enhanced_gbdt.parquet'
    full_df.to_parquet(output_path, index=False)
    print(f"✅ 新排序已保存: {output_path}")
    
    # 7. Top-K 分析
    print("\n📈 Top-K 策略预览:")
    top100 = full_df.head(100)[['rank_enhanced', 'combo', 'ml_score_enhanced', 'mean_oos_ic']].copy()
    print(top100.head(20))
    
    print("\n✅ 完成! 现在可以:")
    print("   1. 使用新的排序文件运行真实回测")
    print("   2. 与基础GBDT和IC排序对比效果")

if __name__ == '__main__':
    main()
