# -*- coding: utf-8 -*-
"""
@author: Copilot
@created: 2025-11-13
@description: 从回测摘要文件中提取增强特征，用于 GBDT 模型优化。
"""
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    """主函数"""
    # 自动查找最新的运行目录
    run_dirs = sorted([d for d in glob.glob('results_combo_wfo/2025*') if os.path.isdir(d)])
    if not run_dirs:
        print("错误：找不到 'results_combo_wfo/2025*' 开头的运行目录。")
        return
        
    # 从最新的目录中提取 run_id，例如 '20251112_223854'
    latest_dir_basename = os.path.basename(run_dirs[-1])
    run_id = '_'.join(latest_dir_basename.split('_')[:2])
    
    print(f"🔍 正在处理最新的运行: {run_id}")
    
    # 构建正确的文件搜索模式 - 查找回测摘要文件
    file_pattern = f"results_combo_wfo/{run_id}_*/*.csv"
    print(f"📂 使用文件模式: {file_pattern}")
    
    backtest_files = glob.glob(file_pattern)
    if not backtest_files:
        print(f"错误：在模式 {file_pattern} 中找不到任何 CSV 文件。")
        return

    print(f"📊 找到 {len(backtest_files)} 个回测摘要文件。")
    
    # 读取所有回测摘要文件并合并
    all_results = []
    for file in tqdm(backtest_files, desc="读取回测摘要"):
        try:
            df = pd.read_csv(file)
            if 'combo' in df.columns and 'sharpe' in df.columns:
                all_results.append(df)
        except Exception as e:
            print(f"读取文件 {os.path.basename(file)} 时出错: {e}")
            
    if not all_results:
        print("错误：未能从任何文件中读取数据。")
        return

    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 去重，保留最新的记录
    combined_df = combined_df.drop_duplicates(subset=['combo'], keep='last')
    
    print(f"✅ 已合并 {len(combined_df)} 个唯一策略的数据。")
    
    # 选择有用的特征列
    useful_features = [
        'combo',
        'sharpe',
        'max_dd',
        'vol',
        'annual_ret',
        'calmar_ratio',
        'sortino_ratio',
        'profit_factor',
        'win_rate',
        'avg_turnover',
        'avg_n_holdings',
        'max_consecutive_wins',
        'max_consecutive_losses'
    ]
    
    # 检查哪些特征存在
    available_features = [f for f in useful_features if f in combined_df.columns]
    missing_features = [f for f in useful_features if f not in combined_df.columns]
    
    if missing_features:
        print(f"⚠️  以下特征不可用: {missing_features}")
    
    # 提取可用特征
    features_df = combined_df[available_features].copy()
    
    # 计算派生特征
    print("\n📊 计算派生特征...")
    
    # 1. 风险调整收益复合指标
    features_df['sharpe_calmar_product'] = features_df['sharpe'] * features_df.get('calmar_ratio', 1)
    
    # 2. 回撤恢复能力 (annual_ret / max_dd)
    features_df['dd_recovery_ratio'] = features_df['annual_ret'] / (features_df['max_dd'].abs() + 1e-6)
    
    # 3. 稳健性指标 (sortino / sharpe)
    features_df['sortino_sharpe_ratio'] = features_df.get('sortino_ratio', 0) / (features_df['sharpe'] + 1e-6)
    
    # 4. 胜率-盈亏比复合指标
    if 'win_rate' in features_df.columns and 'profit_factor' in features_df.columns:
        features_df['win_rate_profit_composite'] = features_df['win_rate'] * np.log1p(features_df['profit_factor'])
    
    # 5. 连胜连败比
    if 'max_consecutive_wins' in features_df.columns and 'max_consecutive_losses' in features_df.columns:
        features_df['win_loss_streak_ratio'] = features_df['max_consecutive_wins'] / (features_df['max_consecutive_losses'] + 1)
    
    # 创建输出目录
    output_dir = f"results/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存到 Parquet 文件
    output_path = os.path.join(output_dir, 'enhanced_features.parquet')
    features_df.to_parquet(output_path, index=False)
    
    print("\n✅ 增强特征提取完成！")
    print(f"   - 共处理 {len(features_df)} 个策略。")
    print(f"   - 可用特征数: {len(features_df.columns)}")
    print(f"   - 特征已保存到: {output_path}")
    
    print("\n📋 特征列表:")
    for i, col in enumerate(features_df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n💡 下一步:")
    print("   1. 将此 'enhanced_features.parquet' 文件与 'ranking_baseline.parquet' 合并。")
    print("   2. 使用合并后的数据集重新训练 GBDT 模型。")
    print("   3. 运行 'validate_ml_ranking_accuracy.py' 验证新模型的效果。")

if __name__ == '__main__':
    main()
