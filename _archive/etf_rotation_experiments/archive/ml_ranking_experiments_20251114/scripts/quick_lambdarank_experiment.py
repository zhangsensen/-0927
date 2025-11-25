#!/usr/bin/env python3
"""
快速 LambdaMART vs GBDT 对比实验
使用已有的真实回测数据训练和验证
"""

import glob
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_backtest_results(run_id: str):
    """加载真实回测结果"""
    pattern = f'results_combo_wfo/{run_id}_*/*.csv'
    backtest_files = glob.glob(pattern)
    
    all_results = []
    for f in backtest_files:
        try:
            df = pd.read_csv(f)
            if 'combo' in df.columns and 'sharpe' in df.columns:
                all_results.append(df)
        except:
            pass
    
    if not all_results:
        raise ValueError(f"未找到回测结果文件: {pattern}")
    
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.drop_duplicates(subset=['combo'], keep='last')
    return combined


def merge_wfo_features(backtest_df: pd.DataFrame, wfo_file: Path):
    """合并 WFO 特征"""
    wfo_df = pd.read_parquet(wfo_file)
    
    # 选择基础特征（与GBDT模型一致）
    feature_cols = ['combo', 'mean_oos_ic', 'oos_ic_std', 'positive_rate', 'stability_score', 'combo_size']
    wfo_features = wfo_df[feature_cols].copy()
    
    # 合并
    merged = backtest_df.merge(wfo_features, on='combo', how='inner')
    return merged


def to_relevance(y: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """转换为排序标签"""
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, quantiles)
    bins = np.unique(bins)
    if bins.size <= 2:
        ranks = pd.Series(y).rank(method="dense").astype(int) - 1
        return np.maximum(ranks, 0)
    thresholds = bins[1:-1]
    labels = np.digitize(y, thresholds, right=False)
    return labels.astype(int)


def calc_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    """计算排序指标"""
    # Spearman相关性
    spearman, p_value = spearmanr(y_pred, y_true)
    
    # Top-K重叠率
    metrics = {
        'model': model_name,
        'spearman': float(spearman),
        'spearman_pvalue': float(p_value),
    }
    
    for k in [10, 50, 100, 500, 1000]:
        if k > len(y_true):
            continue
        idx_true = set(np.argsort(-y_true)[:k])
        idx_pred = set(np.argsort(-y_pred)[:k])
        overlap = len(idx_true & idx_pred) / k
        metrics[f'top{k}_overlap'] = float(overlap)
    
    # NDCG
    ndcg_k = min(len(y_true), 1000)
    gains = y_true - y_true.min()
    ndcg = float(ndcg_score([gains], [y_pred], k=ndcg_k))
    metrics['ndcg@1000'] = ndcg
    
    return metrics


def main():
    print("=" * 80)
    print("🔬 LambdaMART vs GBDT 快速对比实验")
    print("=" * 80)
    
    run_id = '20251112_223854'
    run_dir = Path(f'results/run_{run_id}')
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    backtest_df = load_backtest_results(run_id)
    print(f"✅ 真实回测结果: {len(backtest_df)} 个策略")
    
    wfo_file = run_dir / 'ranking_blends/ranking_baseline.parquet'
    merged_df = merge_wfo_features(backtest_df, wfo_file)
    print(f"✅ 合并 WFO 特征后: {len(merged_df)} 个策略")
    
    # 2. 准备特征和目标
    # 处理列名冲突（combo_size 可能变成 combo_size_x 或 combo_size_y）
    if 'combo_size' not in merged_df.columns:
        if 'combo_size_y' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_y']
        elif 'combo_size_x' in merged_df.columns:
            merged_df['combo_size'] = merged_df['combo_size_x']
    
    feature_cols = ['mean_oos_ic', 'oos_ic_std', 'positive_rate', 'stability_score', 'combo_size']
    X = merged_df[feature_cols].values
    y = merged_df['sharpe'].values  # 使用真实 Sharpe 作为目标
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  特征: {feature_cols}")
    
    # 3. 训练 GBDT 回归模型
    print("\n" + "-" * 80)
    print("🌲 训练 GBDT 回归模型...")
    print("-" * 80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gbdt_model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    gbdt_model.fit(X_train_scaled, y_train)
    
    gbdt_pred_train = gbdt_model.predict(X_train_scaled)
    gbdt_pred_test = gbdt_model.predict(X_test_scaled)
    
    gbdt_metrics_train = calc_ranking_metrics(y_train, gbdt_pred_train, 'GBDT (训练集)')
    gbdt_metrics_test = calc_ranking_metrics(y_test, gbdt_pred_test, 'GBDT (测试集)')
    
    print(f"训练集 Spearman: {gbdt_metrics_train['spearman']:.4f}")
    print(f"测试集 Spearman: {gbdt_metrics_test['spearman']:.4f}")
    print(f"测试集 NDCG@1000: {gbdt_metrics_test['ndcg@1000']:.4f}")
    print(f"测试集 Top100重叠: {gbdt_metrics_test.get('top100_overlap', 0):.2%}")
    
    # 4. 训练 LambdaMART 排序模型
    print("\n" + "-" * 80)
    print("🎯 训练 LambdaMART 排序模型...")
    print("-" * 80)
    
    y_train_rank = to_relevance(y_train)
    y_test_rank = to_relevance(y_test)
    
    lambdarank_model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        n_jobs=-1,
        label_gain=list(range(32)),
        verbose=-1
    )
    
    lambdarank_model.fit(
        X_train,
        y_train_rank,
        group=[len(X_train)],
        eval_set=[(X_test, y_test_rank)],
        eval_group=[[len(X_test)]],
        callbacks=[]
    )
    
    rank_pred_train = lambdarank_model.predict(X_train)
    rank_pred_test = lambdarank_model.predict(X_test)
    
    rank_metrics_train = calc_ranking_metrics(y_train, rank_pred_train, 'LambdaMART (训练集)')
    rank_metrics_test = calc_ranking_metrics(y_test, rank_pred_test, 'LambdaMART (测试集)')
    
    print(f"训练集 Spearman: {rank_metrics_train['spearman']:.4f}")
    print(f"测试集 Spearman: {rank_metrics_test['spearman']:.4f}")
    print(f"测试集 NDCG@1000: {rank_metrics_test['ndcg@1000']:.4f}")
    print(f"测试集 Top100重叠: {rank_metrics_test.get('top100_overlap', 0):.2%}")
    
    # 5. 对比分析
    print("\n" + "=" * 80)
    print("📊 对比分析结果")
    print("=" * 80)
    
    print("\n【测试集 Spearman 相关性对比】")
    print(f"  GBDT 回归:     {gbdt_metrics_test['spearman']:+.4f}")
    print(f"  LambdaMART:    {rank_metrics_test['spearman']:+.4f}")
    improvement = rank_metrics_test['spearman'] - gbdt_metrics_test['spearman']
    print(f"  改进幅度:      {improvement:+.4f} ({improvement/abs(gbdt_metrics_test['spearman'])*100:+.1f}%)")
    
    print("\n【测试集 NDCG@1000 对比】")
    print(f"  GBDT 回归:     {gbdt_metrics_test['ndcg@1000']:.4f}")
    print(f"  LambdaMART:    {rank_metrics_test['ndcg@1000']:.4f}")
    ndcg_improvement = rank_metrics_test['ndcg@1000'] - gbdt_metrics_test['ndcg@1000']
    print(f"  改进幅度:      {ndcg_improvement:+.4f}")
    
    print("\n【Top-K 重叠率对比】")
    for k in [10, 50, 100, 500, 1000]:
        key = f'top{k}_overlap'
        if key in gbdt_metrics_test and key in rank_metrics_test:
            gbdt_val = gbdt_metrics_test[key]
            rank_val = rank_metrics_test[key]
            diff = rank_val - gbdt_val
            print(f"  Top{k:4d}:  GBDT {gbdt_val:.2%}  →  LambdaMART {rank_val:.2%}  (Δ {diff:+.2%})")
    
    # 6. 保存结果
    results = {
        'experiment': 'LambdaMART_vs_GBDT_quick_comparison',
        'run_id': run_id,
        'n_samples': len(merged_df),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'features': feature_cols,
        'gbdt_regressor': {
            'train': gbdt_metrics_train,
            'test': gbdt_metrics_test,
        },
        'lambdarank': {
            'train': rank_metrics_train,
            'test': rank_metrics_test,
        },
        'improvement': {
            'spearman_delta': improvement,
            'spearman_pct': improvement / abs(gbdt_metrics_test['spearman']) * 100,
            'ndcg_delta': ndcg_improvement,
        }
    }
    
    output_file = run_dir / 'lambdarank_vs_gbdt_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 7. 结论
    print("\n" + "=" * 80)
    print("🎯 实验结论")
    print("=" * 80)
    
    if rank_metrics_test['spearman'] > gbdt_metrics_test['spearman'] + 0.05:
        print("✅ LambdaMART 显著优于 GBDT 回归")
        print(f"   排序相关性提升 {improvement:.4f} (相对提升 {improvement/abs(gbdt_metrics_test['spearman'])*100:.1f}%)")
        print("   建议：采用 LambdaMART 替代当前 GBDT 回归模型")
    elif rank_metrics_test['spearman'] > gbdt_metrics_test['spearman']:
        print("⚠️  LambdaMART 略优于 GBDT 回归")
        print(f"   排序相关性提升 {improvement:.4f}")
        print("   建议：可以尝试采用，但提升有限")
    else:
        print("❌ LambdaMART 未显示出优势")
        print(f"   排序相关性变化 {improvement:.4f}")
        print("   建议：继续使用 GBDT 回归模型，或尝试其他优化方向")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
