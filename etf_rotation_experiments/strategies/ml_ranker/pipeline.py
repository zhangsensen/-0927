"""
统一训练Pipeline: 数据加载 → 训练 → 评估 → 稳健性验证

本模块封装完整的排序模型训练流程,支持:
- 单/多数据源训练
- 自动特征工程
- 交叉验证训练
- 模型评估和对比
- 稳健性验证(可选)
- 模型保存和报告生成
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# 导入现有模块
from .config import DatasetConfig, DataSource
from .data_loader import (
    load_multi_source_data, 
    build_training_dataset,
    load_wfo_features,
    load_real_backtest_results
)
from .feature_engineer import build_feature_matrix
from .ltr_model import LTRRanker
from .evaluator import generate_evaluation_report, create_ranking_comparison_df
from .robustness_eval import (
    evaluate_kfold_cv, 
    evaluate_repeated_holdout,
    generate_robustness_report
)


def run_training_pipeline(
    config: DatasetConfig,
    model_params: Optional[Dict[str, Any]] = None,
    enable_robustness: bool = True,
    robustness_params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    output_dir: Path = Path("ml_ranker"),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    统一训练Pipeline: 一键完成训练+评估+稳健性验证
    
    Args:
        config: 数据集配置(单或多数据源)
        model_params: 模型参数字典,None则使用默认值
            - n_estimators: 树数量(默认500)
            - learning_rate: 学习率(默认0.05)
            - max_depth: 最大深度(默认6)
            - 其他LightGBM参数
        enable_robustness: 是否运行稳健性评估
        robustness_params: 稳健性评估参数
            - n_splits: K-Fold折数(默认5)
            - n_repeats: Repeated Holdout次数(默认5)
            - n_estimators: 每个模型树数(默认300,减速)
            - random_state: 随机种子(默认2025)
        save_model: 是否保存模型到磁盘
        output_dir: 输出根目录
        verbose: 是否打印详细日志
        
    Returns:
        完整结果字典包含:
        - model: 训练好的LTRRanker对象
        - evaluation: 评估报告dict
        - robustness: 稳健性报告dict(如果启用)
        - metadata: 元信息dict
        - comparison_df: Top-100排序对比表
        - output_paths: 所有输出文件路径
        
    Example:
        >>> from strategies.ml_ranker.config import DatasetConfig
        >>> config = DatasetConfig.from_yaml("configs/ranking_datasets.yaml")
        >>> result = run_training_pipeline(config, enable_robustness=True)
        >>> print(f"模型Spearman: {result['evaluation']['model_metrics']['spearman_corr']:.4f}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("🚀 启动统一训练Pipeline")
        print(f"{'='*80}\n")
        print(f"配置: {len(config.datasets)} 个数据源")
        print(f"目标列: {config.target_col}")
        print(f"稳健性评估: {'启用' if enable_robustness else '禁用'}")
        print(f"输出目录: {output_dir}")
    
    # =========================================================================
    # 1. 加载数据
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("📂 STEP 1: 加载训练数据")
        print(f"{'='*80}")
    
    if len(config.datasets) == 1:
        # 单数据源: 直接加载
        ds = config.datasets[0]
        if verbose:
            print(f"  单数据源模式: {ds.display_name}")
        wfo_df = load_wfo_features(ds.wfo_dir)
        real_df = load_real_backtest_results(ds.real_dir)
        merged_df, y, metadata = build_training_dataset(wfo_df, real_df, config.target_col)
        
        # 添加rebalance_days列以保持接口一致
        merged_df['rebalance_days'] = ds.rebalance_days
        merged_df['source_label'] = ds.label or "single_source"
        metadata['n_sources'] = 1
        metadata['rebalance_days'] = np.full(len(merged_df), ds.rebalance_days)
    else:
        # 多数据源: 使用multi_source加载器
        if verbose:
            print(f"  多数据源模式: {len(config.datasets)} 个数据源")
        merged_df, y, metadata = load_multi_source_data(config, add_source_id=True, verbose=verbose)
    
    # =========================================================================
    # 2. 构建特征矩阵
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("🛠️ STEP 2: 构建特征矩阵")
        print(f"{'='*80}")
    
    X_df = build_feature_matrix(merged_df)
    X = X_df.values
    feature_names = list(X_df.columns)
    
    if verbose:
        print(f"  特征数: {X.shape[1]}")
        print(f"  样本数: {X.shape[0]}")
        print(f"  特征示例: {feature_names[:5]}")
    
    # =========================================================================
    # 3. 训练模型
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("🔥 STEP 3: 训练LTR模型")
        print(f"{'='*80}")
    
    # 默认参数
    default_params = {
        "objective": "regression",  # 使用回归避免lambdarank的query size限制
        "metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1
    }
    
    # 合并用户参数
    if model_params:
        default_params.update(model_params)
    
    if verbose:
        print(f"  模型参数:")
        print(f"    n_estimators: {default_params['n_estimators']}")
        print(f"    learning_rate: {default_params['learning_rate']}")
        print(f"    max_depth: {default_params['max_depth']}")
    
    model = LTRRanker(**default_params)
    model.train(
        X=pd.DataFrame(X, columns=feature_names),
        y=y,
        cv_folds=5
    )
    
    if verbose:
        print(f"\n  ✓ 训练完成")
        if model.cv_results:
            last_cv = model.cv_results[-1]
            print(f"  CV Spearman: {last_cv.get('spearman_corr', 0):.4f}")
    
    # =========================================================================
    # 4. 预测并评估
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("📈 STEP 4: 模型评估")
        print(f"{'='*80}")
    
    scores, ranks = model.predict(X)
    baseline_scores = merged_df["mean_oos_ic"].values
    
    eval_dir = Path(output_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    importance_df = model.get_feature_importance()
    
    if verbose:
        print(f"  Top-10重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:30s}: {row['importance']:10.0f}")
    
    eval_report = generate_evaluation_report(
        y_true=y.values,
        y_pred=scores,
        baseline_scores=baseline_scores,
        metadata=metadata,
        feature_importance=importance_df,
        output_path=eval_dir / "evaluation_report.json"
    )
    
    if verbose:
        print(f"\n  模型性能:")
        print(f"    Spearman: {eval_report['model_metrics']['spearman_corr']:.4f}")
        print(f"    NDCG@10: {eval_report['model_metrics'].get('ndcg@10', 0):.4f}")
        print(f"    NDCG@50: {eval_report['model_metrics'].get('ndcg@50', 0):.4f}")
        print(f"  Top-10分析:")
        print(f"    命中数: {eval_report['top10_analysis']['model_hits']}/10")
        print(f"    平均收益: {eval_report['top10_analysis']['model_pred_mean']:.4f}")
    
    # =========================================================================
    # 5. 稳健性评估 (可选)
    # =========================================================================
    robustness_report = None
    if enable_robustness:
        if verbose:
            print(f"\n{'='*80}")
            print("🔬 STEP 5: 稳健性评估")
            print(f"{'='*80}\n")
        
        # 默认稳健性参数
        default_rob_params = {
            'n_splits': 5,
            'n_repeats': 5,
            'n_estimators': 300,  # 减少树数以加速
            'random_state': 2025
        }
        
        if robustness_params:
            default_rob_params.update(robustness_params)
        
        if verbose:
            print(f"  K-Fold CV: {default_rob_params['n_splits']}折")
            print(f"  Repeated Holdout: {default_rob_params['n_repeats']}次")
            print(f"  每个模型树数: {default_rob_params['n_estimators']}\n")
        
        try:
            # K-Fold交叉验证
            kfold_results, _ = evaluate_kfold_cv(
                X=X,
                y=y.values,
                feature_names=feature_names,
                baseline_features=merged_df,
                n_splits=default_rob_params['n_splits'],
                n_estimators=default_rob_params['n_estimators'],
                random_state=default_rob_params['random_state']
            )
            
            # Repeated Holdout
            holdout_results, _ = evaluate_repeated_holdout(
                X=X,
                y=y.values,
                feature_names=feature_names,
                baseline_features=merged_df,
                n_repeats=default_rob_params['n_repeats'],
                test_size=0.2,
                n_estimators=default_rob_params['n_estimators'],
                base_random_state=default_rob_params['random_state']
            )
            
            # 生成报告
            robustness_report = generate_robustness_report(
                kfold_results=kfold_results,
                holdout_results=holdout_results,
                output_dir=eval_dir
            )
            
            if verbose:
                kf_spear = robustness_report['kfold_cv']['metrics']['model_spearman']
                rh_spear = robustness_report['repeated_holdout']['metrics']['model_spearman']
                print(f"\n  稳健性结果:")
                print(f"    K-Fold Spearman: {kf_spear['mean']:.4f} ± {kf_spear['std']:.4f}")
                print(f"    Holdout Spearman: {rh_spear['mean']:.4f} ± {rh_spear['std']:.4f}")
                
                avg_std = (kf_spear['std'] + rh_spear['std']) / 2
                if avg_std < 0.03:
                    print(f"    评价: ✅ 稳定性优秀 (std={avg_std:.4f} < 0.03)")
                elif avg_std < 0.08:
                    print(f"    评价: ✅ 稳定性良好 (std={avg_std:.4f} < 0.08)")
                else:
                    print(f"    评价: ⚠️  稳定性一般 (std={avg_std:.4f} >= 0.08)")
        
        except Exception as e:
            if verbose:
                print(f"  ⚠️  稳健性评估失败: {e}")
                print(f"  继续执行其他步骤...")
    
    # =========================================================================
    # 6. 保存模型
    # =========================================================================
    model_path = None
    if save_model:
        if verbose:
            print(f"\n{'='*80}")
            print("💾 STEP 6: 保存模型")
            print(f"{'='*80}")
        
        model_dir = Path(output_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "ltr_ranker"
        model.save(str(model_path))
        
        if verbose:
            print(f"  ✓ 模型已保存: {model_path}.txt")
            print(f"  ✓ 元数据已保存: {model_path}_meta.pkl")
    
    # =========================================================================
    # 7. 生成排序对比表
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("📊 STEP 7: 生成排序对比表")
        print(f"{'='*80}")
    
    comparison_df = create_ranking_comparison_df(
        y_true=y.values,
        y_pred=scores,
        baseline_scores=baseline_scores,
        combos=merged_df["combo"].values,
        top_n=100
    )
    
    comparison_path = eval_dir / "ranking_comparison_top100.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    
    if verbose:
        print(f"  ✓ Top-100对比表已保存: {comparison_path}")
    
    # =========================================================================
    # 8. 返回结果
    # =========================================================================
    result = {
        'model': model,
        'evaluation': eval_report,
        'robustness': robustness_report,
        'metadata': metadata,
        'comparison_df': comparison_df,
        'X': X,
        'y': y.values,
        'scores': scores,
        'ranks': ranks,
        'output_paths': {
            'model': str(model_path) if model_path else None,
            'evaluation': str(eval_dir / "evaluation_report.json"),
            'robustness': str(eval_dir / "robustness_report.json") if robustness_report else None,
            'robustness_detail': str(eval_dir / "robustness_detail.csv") if robustness_report else None,
            'comparison': str(comparison_path)
        }
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ Pipeline执行完成")
        print(f"{'='*80}\n")
        print(f"总结:")
        print(f"  数据源: {metadata['n_sources']} 个")
        print(f"  总样本: {len(y)}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  模型性能:")
        print(f"    Spearman: {eval_report['model_metrics']['spearman_corr']:.4f}")
        print(f"    NDCG@10: {eval_report['model_metrics'].get('ndcg@10', 0):.4f}")
        print(f"    Top-10命中: {eval_report['top10_analysis']['model_hits']}/10")
        
        if robustness_report:
            kf_spear = robustness_report['kfold_cv']['metrics']['model_spearman']
            print(f"  稳健性:")
            print(f"    K-Fold Spearman: {kf_spear['mean']:.4f} ± {kf_spear['std']:.4f}")
        
        print(f"\n输出文件:")
        for key, path in result['output_paths'].items():
            if path:
                print(f"  {key}: {path}")
        print()
    
    return result
