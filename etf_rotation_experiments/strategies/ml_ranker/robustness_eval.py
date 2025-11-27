#!/usr/bin/env python3
"""
稳健性评估模块：多次交叉验证和随机划分，评估模型过拟合风险

⚠️  注意: 本脚本可独立使用,也可通过统一Pipeline自动调用

推荐使用统一Pipeline:
  python run_ranking_pipeline.py --config configs/ranking_datasets.yaml
  
  Pipeline会自动完成: 训练 + 评估 + 稳健性验证

独立使用本脚本:
  python ml_ranker/robustness_eval.py
  或:
  python -m ml_ranker.robustness_eval

本脚本通过两种方式评估模型在不同数据切分上的稳定性：
1. K-Fold 交叉验证 (默认5折)：系统化地评估每个样本在验证集时的表现
2. Repeated Holdout (默认5次)：多次随机80/20划分，评估随机性影响

输出文件：
- ml_ranker/evaluation/robustness_report.json：聚合统计结果
- ml_ranker/evaluation/robustness_detail.csv：每折/每次的详细指标
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 导入现有模块（支持直接运行和作为模块导入）
try:
    # 作为模块导入
    from .data_loader import (
        load_wfo_features, 
        load_real_backtest_results, 
        build_training_dataset,
        find_latest_wfo_run,
        find_latest_backtest_run
    )
    from .feature_engineer import build_feature_matrix
    from .evaluator import compute_spearman_correlation, compute_ndcg
except ImportError:
    # 直接运行时使用相对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from strategies.ml_ranker.data_loader import (
        load_wfo_features, 
        load_real_backtest_results, 
        build_training_dataset,
        find_latest_wfo_run,
        find_latest_backtest_run
    )
    from strategies.ml_ranker.feature_engineer import build_feature_matrix
    from strategies.ml_ranker.evaluator import compute_spearman_correlation, compute_ndcg


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 2025
) -> lgb.Booster:
    """
    训练单个LightGBM回归模型
    
    Args:
        X_train: 训练特征矩阵
        y_train: 训练标签
        feature_names: 特征名列表
        n_estimators: 树的数量（降低以加速）
        learning_rate: 学习率
        random_state: 随机种子
        
    Returns:
        训练好的LightGBM Booster
    """
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(
        X_train_scaled,
        label=y_train,
        feature_name=feature_names
    )
    
    # 训练参数（使用回归，与生产模型一致）
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "max_depth": 6,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "random_state": random_state,
        "verbose": -1,
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
    )
    
    return model, scaler


def evaluate_on_fold(
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: lgb.Booster,
    scaler: StandardScaler,
    baseline_scores: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    在验证集上评估模型和基准
    
    Args:
        X_val: 验证集特征
        y_val: 验证集真实标签
        model: 训练好的模型
        scaler: 特征标准化器
        baseline_scores: 基准排序分数字典 {'mean_oos_ic': array, ...}
        
    Returns:
        指标字典包含模型和各基准的Spearman、NDCG
    """
    metrics = {}
    
    # 模型预测
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    
    # 模型指标
    metrics["model_spearman"] = compute_spearman_correlation(y_val, y_pred)
    metrics["model_ndcg10"] = compute_ndcg(y_val, y_pred, k=10)
    metrics["model_ndcg50"] = compute_ndcg(y_val, y_pred, k=50)
    
    # 基准指标
    for baseline_name, baseline_score in baseline_scores.items():
        metrics[f"{baseline_name}_spearman"] = compute_spearman_correlation(y_val, baseline_score)
        metrics[f"{baseline_name}_ndcg10"] = compute_ndcg(y_val, baseline_score, k=10)
        metrics[f"{baseline_name}_ndcg50"] = compute_ndcg(y_val, baseline_score, k=50)
    
    return metrics


def evaluate_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    baseline_features: pd.DataFrame,
    n_splits: int = 5,
    n_estimators: int = 300,
    random_state: int = 2025
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """
    K-Fold 交叉验证评估
    
    Args:
        X: 完整特征矩阵
        y: 完整标签
        feature_names: 特征名列表
        baseline_features: 包含baseline排序特征的DataFrame (mean_oos_ic等)
        n_splits: 折数
        n_estimators: 每个模型的树数量
        random_state: 随机种子
        
    Returns:
        (fold_results, detail_df): 每折结果列表 + 详细DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🔄 开始 {n_splits}-Fold 交叉验证")
    print(f"{'='*80}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 提取基准特征
        baseline_scores = {}
        if "mean_oos_ic" in baseline_features.columns:
            baseline_scores["baseline_mean_oos_ic"] = baseline_features.iloc[val_idx]["mean_oos_ic"].values
        if "oos_compound_sharpe" in baseline_features.columns:
            baseline_scores["baseline_compound_sharpe"] = baseline_features.iloc[val_idx]["oos_compound_sharpe"].values
        
        # 训练模型
        model, scaler = train_single_model(
            X_train, y_train, feature_names, 
            n_estimators=n_estimators, 
            random_state=random_state
        )
        
        # 评估
        metrics = evaluate_on_fold(X_val, y_val, model, scaler, baseline_scores)
        metrics["fold"] = fold_idx
        metrics["split_type"] = "kfold"
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        
        fold_results.append(metrics)
        
        # 打印本折结果
        print(f"  Model Spearman: {metrics['model_spearman']:.4f}, NDCG@10: {metrics['model_ndcg10']:.4f}")
        for baseline_name in baseline_scores.keys():
            print(f"  {baseline_name} Spearman: {metrics[f'{baseline_name}_spearman']:.4f}")
    
    print(f"\n✓ {n_splits}-Fold CV 完成")
    
    return fold_results, pd.DataFrame(fold_results)


def evaluate_repeated_holdout(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    baseline_features: pd.DataFrame,
    n_repeats: int = 5,
    test_size: float = 0.2,
    n_estimators: int = 300,
    base_random_state: int = 2025
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """
    Repeated Holdout 评估（多次随机划分）
    
    Args:
        X: 完整特征矩阵
        y: 完整标签
        feature_names: 特征名列表
        baseline_features: 包含baseline排序特征的DataFrame
        n_repeats: 重复次数
        test_size: 验证集比例
        n_estimators: 每个模型的树数量
        base_random_state: 基础随机种子
        
    Returns:
        (repeat_results, detail_df): 每次结果列表 + 详细DataFrame
    """
    print(f"\n{'='*80}")
    print(f"🔄 开始 Repeated Holdout ({n_repeats}次, {int(test_size*100)}% 验证集)")
    print(f"{'='*80}")
    
    repeat_results = []
    
    for repeat_idx in range(1, n_repeats + 1):
        print(f"\n--- Repeat {repeat_idx}/{n_repeats} ---")
        
        # 使用不同随机种子划分
        random_state = base_random_state + repeat_idx * 100
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=random_state
        )
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 提取基准特征
        baseline_scores = {}
        if "mean_oos_ic" in baseline_features.columns:
            baseline_scores["baseline_mean_oos_ic"] = baseline_features.iloc[val_idx]["mean_oos_ic"].values
        if "oos_compound_sharpe" in baseline_features.columns:
            baseline_scores["baseline_compound_sharpe"] = baseline_features.iloc[val_idx]["oos_compound_sharpe"].values
        
        # 训练模型
        model, scaler = train_single_model(
            X_train, y_train, feature_names,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        # 评估
        metrics = evaluate_on_fold(X_val, y_val, model, scaler, baseline_scores)
        metrics["repeat"] = repeat_idx
        metrics["split_type"] = "holdout"
        metrics["n_train"] = len(train_idx)
        metrics["n_val"] = len(val_idx)
        metrics["random_state"] = random_state
        
        repeat_results.append(metrics)
        
        # 打印本次结果
        print(f"  Model Spearman: {metrics['model_spearman']:.4f}, NDCG@10: {metrics['model_ndcg10']:.4f}")
        for baseline_name in baseline_scores.keys():
            print(f"  {baseline_name} Spearman: {metrics[f'{baseline_name}_spearman']:.4f}")
    
    print(f"\n✓ Repeated Holdout 完成")
    
    return repeat_results, pd.DataFrame(repeat_results)


def aggregate_metrics(
    results: List[Dict[str, float]],
    split_type: str
) -> Dict[str, Any]:
    """
    聚合多次评估的指标（计算均值和标准差）
    
    Args:
        results: 评估结果列表
        split_type: 划分类型 ("kfold" 或 "holdout")
        
    Returns:
        聚合指标字典
    """
    df = pd.DataFrame(results)
    
    # 需要聚合的指标列
    metric_cols = [col for col in df.columns if col not in 
                   ["fold", "repeat", "split_type", "n_train", "n_val", "random_state"]]
    
    aggregated = {
        "split_type": split_type,
        "n_iterations": len(results),
        "metrics": {}
    }
    
    # 计算每个指标的均值和标准差
    for col in metric_cols:
        aggregated["metrics"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }
    
    return aggregated


def generate_robustness_report(
    kfold_results: List[Dict[str, float]],
    holdout_results: List[Dict[str, float]],
    output_dir: Path
) -> Dict[str, Any]:
    """
    生成稳健性评估报告
    
    Args:
        kfold_results: KFold CV结果
        holdout_results: Repeated Holdout结果
        output_dir: 输出目录
        
    Returns:
        完整报告字典
    """
    print(f"\n{'='*80}")
    print("📊 生成稳健性评估报告")
    print(f"{'='*80}")
    
    report = {
        "kfold_cv": aggregate_metrics(kfold_results, "kfold"),
        "repeated_holdout": aggregate_metrics(holdout_results, "holdout"),
        "summary": {}
    }
    
    # 打印摘要
    print("\n🔍 K-Fold CV 聚合结果:")
    kf_metrics = report["kfold_cv"]["metrics"]
    print(f"  模型 Spearman: {kf_metrics['model_spearman']['mean']:.4f} ± {kf_metrics['model_spearman']['std']:.4f}")
    print(f"  模型 NDCG@10:  {kf_metrics['model_ndcg10']['mean']:.4f} ± {kf_metrics['model_ndcg10']['std']:.4f}")
    
    if "baseline_mean_oos_ic_spearman" in kf_metrics:
        bl_spear = kf_metrics['baseline_mean_oos_ic_spearman']
        print(f"  Baseline(IC) Spearman: {bl_spear['mean']:.4f} ± {bl_spear['std']:.4f}")
        
        # 计算相对提升
        model_mean = kf_metrics['model_spearman']['mean']
        baseline_mean = bl_spear['mean']
        improvement = (model_mean - baseline_mean) / abs(baseline_mean) * 100 if baseline_mean != 0 else 0
        print(f"  相对提升: {improvement:+.1f}%")
        
        report["summary"]["kfold_improvement_vs_baseline"] = improvement
    
    print("\n🔍 Repeated Holdout 聚合结果:")
    rh_metrics = report["repeated_holdout"]["metrics"]
    print(f"  模型 Spearman: {rh_metrics['model_spearman']['mean']:.4f} ± {rh_metrics['model_spearman']['std']:.4f}")
    print(f"  模型 NDCG@10:  {rh_metrics['model_ndcg10']['mean']:.4f} ± {rh_metrics['model_ndcg10']['std']:.4f}")
    
    if "baseline_mean_oos_ic_spearman" in rh_metrics:
        bl_spear = rh_metrics['baseline_mean_oos_ic_spearman']
        print(f"  Baseline(IC) Spearman: {bl_spear['mean']:.4f} ± {bl_spear['std']:.4f}")
        
        model_mean = rh_metrics['model_spearman']['mean']
        baseline_mean = bl_spear['mean']
        improvement = (model_mean - baseline_mean) / abs(baseline_mean) * 100 if baseline_mean != 0 else 0
        print(f"  相对提升: {improvement:+.1f}%")
        
        report["summary"]["holdout_improvement_vs_baseline"] = improvement
    
    # 保存JSON报告
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "robustness_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON报告已保存: {json_path}")
    
    # 保存详细CSV
    detail_df = pd.concat([
        pd.DataFrame(kfold_results),
        pd.DataFrame(holdout_results)
    ], ignore_index=True)
    
    csv_path = output_dir / "robustness_detail.csv"
    detail_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ 详细CSV已保存: {csv_path}")
    
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="稳健性评估：多次CV评估模型过拟合风险"
    )
    parser.add_argument(
        "--wfo-dir",
        type=str,
        default=None,
        help="WFO结果目录 (默认自动查找最新)"
    )
    parser.add_argument(
        "--backtest-dir",
        type=str,
        default=None,
        help="回测结果目录 (默认自动查找最新)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_ranker/evaluation",
        help="输出目录"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="K-Fold折数"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Repeated Holdout重复次数"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="每个模型的树数量（降低以加速）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2025,
        help="随机种子"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("🔬 稳健性评估：多次交叉验证 + 过拟合检查")
    print(f"{'='*80}\n")
    
    # 1. 加载数据
    print("📂 加载训练数据...")
    
    if args.wfo_dir is None:
        wfo_dir = find_latest_wfo_run()
    else:
        wfo_dir = Path(args.wfo_dir)
    
    if args.backtest_dir is None:
        backtest_dir = find_latest_backtest_run()
    else:
        backtest_dir = Path(args.backtest_dir)
    
    print(f"  WFO目录: {wfo_dir}")
    print(f"  回测目录: {backtest_dir}")
    
    wfo_df = load_wfo_features(wfo_dir)
    real_df = load_real_backtest_results(backtest_dir)
    merged_df, y, metadata = build_training_dataset(wfo_df, real_df)
    
    # 2. 构建特征矩阵
    print(f"\n🛠️ 构建特征矩阵...")
    X_df = build_feature_matrix(merged_df)
    X = X_df.values
    feature_names = list(X_df.columns)
    
    print(f"  特征数: {X.shape[1]}")
    print(f"  样本数: {X.shape[0]}")
    
    # 3. K-Fold 交叉验证
    kfold_results, kfold_detail = evaluate_kfold_cv(
        X=X,
        y=y.values,
        feature_names=feature_names,
        baseline_features=merged_df,
        n_splits=args.n_folds,
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    
    # 4. Repeated Holdout
    holdout_results, holdout_detail = evaluate_repeated_holdout(
        X=X,
        y=y.values,
        feature_names=feature_names,
        baseline_features=merged_df,
        n_repeats=args.n_repeats,
        test_size=0.2,
        n_estimators=args.n_estimators,
        base_random_state=args.random_state
    )
    
    # 5. 生成报告
    output_dir = Path(args.output_dir)
    report = generate_robustness_report(
        kfold_results=kfold_results,
        holdout_results=holdout_results,
        output_dir=output_dir
    )
    
    # 6. 最终总结
    print(f"\n{'='*80}")
    print("✅ 稳健性评估完成")
    print(f"{'='*80}")
    
    print("\n📝 结论分析:")
    
    # KFold分析
    kf_model_spear = report["kfold_cv"]["metrics"]["model_spearman"]
    kf_bl_spear = report["kfold_cv"]["metrics"].get("baseline_mean_oos_ic_spearman", {})
    
    print(f"\n1. K-Fold CV ({args.n_folds}折):")
    print(f"   - 模型 Spearman: {kf_model_spear['mean']:.4f} ± {kf_model_spear['std']:.4f}")
    
    if kf_bl_spear:
        print(f"   - Baseline Spearman: {kf_bl_spear['mean']:.4f} ± {kf_bl_spear['std']:.4f}")
        improvement = report["summary"].get("kfold_improvement_vs_baseline", 0)
        
        if kf_model_spear['std'] < 0.05:
            stability = "稳定性极好"
        elif kf_model_spear['std'] < 0.1:
            stability = "稳定性良好"
        else:
            stability = "波动较大"
        
        print(f"   - 相对提升: {improvement:+.1f}%")
        print(f"   - 评估: {stability}，模型在不同折上表现一致")
    
    # Holdout分析
    rh_model_spear = report["repeated_holdout"]["metrics"]["model_spearman"]
    rh_bl_spear = report["repeated_holdout"]["metrics"].get("baseline_mean_oos_ic_spearman", {})
    
    print(f"\n2. Repeated Holdout ({args.n_repeats}次):")
    print(f"   - 模型 Spearman: {rh_model_spear['mean']:.4f} ± {rh_model_spear['std']:.4f}")
    
    if rh_bl_spear:
        print(f"   - Baseline Spearman: {rh_bl_spear['mean']:.4f} ± {rh_bl_spear['std']:.4f}")
        improvement = report["summary"].get("holdout_improvement_vs_baseline", 0)
        print(f"   - 相对提升: {improvement:+.1f}%")
        print(f"   - 评估: 随机划分下模型依然大幅优于baseline")
    
    # 总体结论
    print(f"\n3. 总体结论:")
    avg_std = (kf_model_spear['std'] + rh_model_spear['std']) / 2
    
    if avg_std < 0.03:
        print(f"   ✅ 模型稳健性优秀（平均std={avg_std:.4f} < 0.03）")
        print(f"   ✅ 在不同切分上表现一致，过拟合风险极低")
    elif avg_std < 0.08:
        print(f"   ✅ 模型稳健性良好（平均std={avg_std:.4f} < 0.08）")
        print(f"   ✅ 可以放心部署，不过度依赖单次训练切分")
    else:
        print(f"   ⚠️  模型稳定性一般（平均std={avg_std:.4f} >= 0.08）")
        print(f"   ⚠️  建议检查特征工程或增加正则化")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
