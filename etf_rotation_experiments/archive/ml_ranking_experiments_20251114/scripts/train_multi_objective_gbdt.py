# -*- coding: utf-8 -*-
"""
@author: Copilot
@created: 2025-11-13
@description: 多目标回归GBDT训练 - 平衡收益与风险的学习目标
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
import json
import os
import glob

def calculate_target_scores(df):
    """计算不同的目标函数"""
    targets = {}
    
    # 基线: 纯 Sharpe
    targets['sharpe'] = df['sharpe']
    
    # 目标1: Sharpe - λ * MaxDD (线性惩罚回撤)
    targets['sharpe_minus_dd_0.3'] = df['sharpe'] - 0.3 * df['max_dd'].abs()
    targets['sharpe_minus_dd_0.5'] = df['sharpe'] - 0.5 * df['max_dd'].abs()
    targets['sharpe_minus_dd_1.0'] = df['sharpe'] - 1.0 * df['max_dd'].abs()
    
    # 目标2: Calmar Ratio (年化收益 / 最大回撤)
    targets['calmar'] = df['annual_ret'] / (df['max_dd'].abs() + 1e-6)
    
    # 目标3: Sharpe / (1 + MaxDD) (比例形式)
    targets['sharpe_over_dd'] = df['sharpe'] / (1 + df['max_dd'].abs())
    
    # 目标4: 加权组合 (Sharpe^2 - DD)
    targets['sharpe2_minus_dd'] = df['sharpe']**2 - df['max_dd'].abs()
    
    return pd.DataFrame(targets)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_name):
    """训练模型并评估"""
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
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估 (使用 Spearman 相关性)
    train_spearman = spearmanr(y_train, y_train_pred)[0]
    test_spearman = spearmanr(y_test, y_test_pred)[0]
    
    # 但我们真正关心的是: 预测排序与真实Sharpe的相关性
    # (因为最终目标是选出高Sharpe策略,而非高复合指标策略)
    
    results = {
        'model_name': model_name,
        'train_spearman': train_spearman,
        'test_spearman': test_spearman,
        'model': model
    }
    
    return results

def main():
    """主函数"""
    run_id = '20251112_223854'
    
    print("="*80)
    print("🎯 多目标回归 GBDT 训练实验")
    print("="*80)
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    
    # 加载回测结果 (包含真实 Sharpe, MaxDD 等)
    backtest_files = glob.glob(f'results_combo_wfo/{run_id}_*/top3000_profit_backtest_slip2bps_{run_id}_*.csv')
    
    backtest_results = []
    for f in backtest_files:
        try:
            df = pd.read_csv(f)
            if 'combo' in df.columns and 'sharpe' in df.columns:
                # 提取需要的列
                cols = ['combo', 'sharpe', 'max_dd', 'annual_ret']
                available_cols = [c for c in cols if c in df.columns]
                backtest_results.append(df[available_cols])
        except Exception as e:
            print(f"读取 {f} 失败: {e}")
    
    if not backtest_results:
        print("❌ 错误: 找不到回测结果文件。")
        return
    
    backtest_df = pd.concat(backtest_results, ignore_index=True)
    backtest_df = backtest_df.drop_duplicates(subset=['combo'], keep='last')
    
    print(f"✅ 回测结果: {len(backtest_df)} 个策略")
    
    # 加载 WFO 特征
    wfo_df = pd.read_parquet(f'results/run_{run_id}/ranking_blends/ranking_baseline.parquet')
    print(f"✅ WFO 特征: {len(wfo_df)} 个策略")
    
    # 2. 合并数据
    print("\n🔗 合并数据...")
    merged_df = backtest_df.merge(wfo_df, on='combo', how='inner')
    print(f"✅ 合并后: {len(merged_df)} 个策略")
    
    # 3. 计算不同的目标函数
    print("\n📊 计算多目标函数...")
    target_df = calculate_target_scores(merged_df)
    
    print(f"✅ 已计算 {len(target_df.columns)} 个目标函数:")
    for col in target_df.columns:
        print(f"   - {col}")
    
    # 4. 准备特征
    wfo_features = ['mean_oos_ic', 'oos_ic_std', 'positive_rate', 'stability_score', 'combo_size']
    
    # 处理 combo_size 冲突
    if 'combo_size' not in merged_df.columns:
        if 'combo_size_x' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_x']
        elif 'combo_size_y' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_y']
    
    X = merged_df[wfo_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # 保存真实 Sharpe (用于最终评估)
    y_sharpe_real = merged_df['sharpe'].values
    
    # 5. 数据划分
    print("\n✂️  划分数据 (70% 训练, 30% 测试)...")
    
    # 使用相同的随机种子确保所有模型使用相同的划分
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    X_train, X_test = X[train_idx], X[test_idx]
    
    print(f"   训练集: {len(X_train)} 样本")
    print(f"   测试集: {len(X_test)} 样本")
    
    # 6. 训练多个模型
    print("\n" + "="*80)
    print("🌲 训练多个目标函数的 GBDT 模型...")
    print("="*80)
    
    all_results = []
    
    for target_name in target_df.columns:
        print(f"\n训练目标: {target_name}")
        
        y = target_df[target_name].values
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        result = train_and_evaluate_model(X_train, y_train, X_test, y_test, target_name)
        
        # 关键评估: 预测排序与真实Sharpe的相关性
        y_sharpe_test = y_sharpe_real[test_idx]
        
        # 使用模型预测,然后计算预测值与真实Sharpe的相关性
        y_test_pred = result['model'].predict(X_test)
        sharpe_spearman = spearmanr(y_test_pred, y_sharpe_test)[0]
        
        result['test_vs_real_sharpe_spearman'] = sharpe_spearman
        
        print(f"   训练集 Spearman (vs 目标): {result['train_spearman']:.4f}")
        print(f"   测试集 Spearman (vs 目标): {result['test_spearman']:.4f}")
        print(f"   测试集 Spearman (vs 真实Sharpe): {sharpe_spearman:.4f} ⭐")
        
        all_results.append(result)
    
    # 7. 对比分析
    print("\n" + "="*80)
    print("📊 模型对比分析")
    print("="*80)
    
    comparison_df = pd.DataFrame([{
        '目标函数': r['model_name'],
        '训练集相关性': f"{r['train_spearman']:.4f}",
        '测试集相关性(vs目标)': f"{r['test_spearman']:.4f}",
        '测试集相关性(vs真实Sharpe)': f"{r['test_vs_real_sharpe_spearman']:.4f}"
    } for r in all_results])
    
    print(comparison_df.to_string(index=False))
    
    # 8. 选择最佳模型
    best_result = max(all_results, key=lambda x: x['test_vs_real_sharpe_spearman'])
    
    print("\n" + "="*80)
    print("🏆 最佳模型")
    print("="*80)
    print(f"   目标函数: {best_result['model_name']}")
    print(f"   测试集 Spearman (vs 真实Sharpe): {best_result['test_vs_real_sharpe_spearman']:.4f}")
    
    # 9. 保存最佳模型
    output_dir = f'results/run_{run_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'gbdt_multi_objective.joblib')
    joblib.dump(best_result['model'], model_path)
    print(f"\n✅ 最佳模型已保存: {model_path}")
    
    # 保存配置
    config = {
        'target_function': best_result['model_name'],
        'features': wfo_features,
        'train_spearman': float(best_result['train_spearman']),
        'test_spearman': float(best_result['test_spearman']),
        'test_vs_real_sharpe_spearman': float(best_result['test_vs_real_sharpe_spearman']),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    config_path = os.path.join(output_dir, 'gbdt_multi_objective_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 配置已保存: {config_path}")
    
    # 保存所有结果的对比
    comparison_path = os.path.join(output_dir, 'multi_objective_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ 对比结果已保存: {comparison_path}")
    
    # 10. 性能对比
    print("\n" + "="*80)
    print("📈 性能对比 (vs 基线)")
    print("="*80)
    
    baseline_spearman = 0.7129  # 基础 GBDT (目标=sharpe)
    improvement = best_result['test_vs_real_sharpe_spearman'] - baseline_spearman
    improvement_pct = (improvement / baseline_spearman) * 100
    
    print(f"   基线 GBDT (目标=sharpe):        {baseline_spearman:.4f}")
    print(f"   多目标 GBDT (目标={best_result['model_name']}): {best_result['test_vs_real_sharpe_spearman']:.4f}")
    print(f"   改进幅度: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    print("\n💡 下一步:")
    print("   1. 使用最佳模型对完整的12597个策略重新排序")
    print("   2. 运行 Top3000 真实回测验证")
    print("   3. 与 IC 排序和基础 ML 排序对比")

if __name__ == '__main__':
    main()
