#!/usr/bin/env python3
"""
训练一个简化的 LambdaMART 模型，使用与 GBDT 回归模型相同的特征集
"""

import argparse
import json
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold


def load_gbdt_features(gbdt_model_path: Path) -> List[str]:
    """从 GBDT 模型中加载特征列表"""
    gbdt_model = joblib.load(gbdt_model_path)
    return gbdt_model['feature_names']


def load_and_prepare_data(dataset_path: Path, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """加载数据并准备特征和目标"""
    df = pd.read_parquet(dataset_path)
    
    # 只保留有真实回测结果的数据
    df = df.dropna(subset=['sharpe_net']).reset_index(drop=True)
    
    # 提取特征
    X = df[feature_names].copy()
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    
    # 目标：sharpe_net
    y = df['sharpe_net'].values.astype(float)
    
    return X, y, df


def to_relevance(y: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """将连续目标转换为相关性标签"""
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, quantiles)
    bins = np.unique(bins)
    if bins.size <= 2:
        ranks = pd.Series(y).rank(method="dense").astype(int) - 1
        return np.maximum(ranks, 0)
    thresholds = bins[1:-1]
    labels = np.digitize(y, thresholds, right=False)
    return labels.astype(int)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算排序评估指标"""
    ndcg_k = min(len(y_true), 1000)
    gains = y_true
    min_val = gains.min()
    if min_val < 0:
        gains = gains - min_val
    ndcg_val = float(ndcg_score([gains], [y_pred], k=ndcg_k))
    spearman, _ = spearmanr(y_pred, y_true)
    
    metrics = {
        'ndcg@1000': ndcg_val,
        'spearman': float(spearman),
    }
    
    for k in [100, 500, 1000]:
        k_eff = min(len(y_true), k)
        idx_true = set(np.argsort(-y_true)[:k_eff])
        idx_pred = set(np.argsort(-y_pred)[:k_eff])
        overlap = len(idx_true & idx_pred) / k_eff if k_eff > 0 else 0.0
        metrics[f'top{k}_overlap'] = float(overlap)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练简化 LambdaMART 模型')
    parser.add_argument('--dataset', type=str, required=True, help='训练数据集路径')
    parser.add_argument('--gbdt-model', type=str, required=True, help='GBDT 模型路径（用于提取特征列表）')
    parser.add_argument('--output-name', type=str, default='calibrator_ranker_simple', help='输出模型名称')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = (repo_root / args.dataset).resolve()
    gbdt_model_path = (repo_root / args.gbdt_model).resolve()
    output_dir = repo_root / "results" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("简化 LambdaMART 模型训练")
    print("=" * 80)
    print(f"数据集: {dataset_path}")
    print(f"GBDT 模型: {gbdt_model_path}")
    
    # 1. 加载 GBDT 特征
    print("\n加载 GBDT 特征列表...")
    feature_names = load_gbdt_features(gbdt_model_path)
    print(f"✅ 特征数: {len(feature_names)}")
    print(f"   特征: {feature_names}")
    
    # 2. 加载和准备数据
    print("\n加载训练数据...")
    X, y, df = load_and_prepare_data(dataset_path, feature_names)
    print(f"✅ 样本数: {len(X)} | 特征数: {X.shape[1]}")
    
    # 3. 交叉验证
    print("\n" + "-" * 80)
    print("5折交叉验证...")
    print("-" * 80)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 转换为相关性标签
        y_train_rank = to_relevance(y_train)
        
        # 训练模型
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            learning_rate=0.05,
            n_estimators=300,
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
        )
        
        model.fit(
            X_train,
            y_train_rank,
            group=[len(X_train)],  # 单个查询组
            eval_set=[(X_val, to_relevance(y_val))],
            eval_group=[[len(X_val)]],
            eval_at=[100, 500, 1000],
            callbacks=[],
        )
        
        # 评估
        y_pred = model.predict(X_val)
        metrics = calc_metrics(y_val, y_pred)
        cv_metrics.append(metrics)
        
        print(f"[Fold {fold}] Spearman={metrics['spearman']:.4f} | "
              f"NDCG@1000={metrics['ndcg@1000']:.4f} | "
              f"Top100重叠={metrics['top100_overlap']:.2%}")
    
    # 平均指标
    avg_metrics = {k: float(np.mean([m[k] for m in cv_metrics])) for k in cv_metrics[0].keys()}
    print("-" * 80)
    print("CV 平均指标:")
    for k, v in avg_metrics.items():
        if 'overlap' in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:.4f}")
    
    # 4. 全量训练
    print("\n" + "-" * 80)
    print("全量训练最终模型...")
    print("-" * 80)
    
    y_rank = to_relevance(y)
    final_model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=300,
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
    )
    
    final_model.fit(X, y_rank, group=[len(X)], callbacks=[])
    
    # 5. 特征重要性
    importance = dict(zip(feature_names, final_model.booster_.feature_importance(importance_type="gain")))
    importance = {k: float(v) for k, v in importance.items()}
    
    print("\n特征重要性:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<30} {imp:>10.2f}")
    
    # 6. 保存模型和元数据
    model_file = output_dir / f"{args.output_name}.txt"
    final_model.booster_.save_model(str(model_file))
    print(f"\n✅ 模型已保存: {model_file}")
    
    importance_file = output_dir / f"{args.output_name}_importance.json"
    with open(importance_file, 'w') as f:
        json.dump(importance, f, indent=2)
    print(f"✅ 特征重要性: {importance_file}")
    
    metrics_file = output_dir / f"{args.output_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'cv_average': avg_metrics,
            'feature_names': feature_names,
            'n_samples': len(X),
            'n_features': len(feature_names),
        }, f, indent=2)
    print(f"✅ CV指标: {metrics_file}")
    
    print("\n" + "=" * 80)
    print("✅ 训练完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
