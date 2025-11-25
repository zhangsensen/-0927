#!/usr/bin/env python3
"""
训练LTR排序模型 (单数据源训练)

⚠️  注意: 本脚本保留用于向后兼容
推荐使用新的统一Pipeline入口: run_ranking_pipeline.py

新入口支持:
- 多数据源/多换仓周期训练
- 自动稳健性评估
- 统一配置管理

使用方式:
  # 传统单数据源训练 (本脚本)
  python train_ranker.py --wfo-dir results/run_xxx --backtest-dir results_combo_wfo/xxx
  
  # 新的统一Pipeline (推荐)
  python run_ranking_pipeline.py --config configs/ranking_datasets.yaml
"""
import argparse
from pathlib import Path
import json
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 尝试导入新的pipeline模块
try:
    from ml_ranker.config import DatasetConfig
    from ml_ranker.pipeline import run_training_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

from ml_ranker.data_loader import build_training_dataset, find_latest_wfo_run, find_latest_backtest_run
from ml_ranker.feature_engineer import build_feature_matrix
from ml_ranker.ltr_model import LTRRanker
from ml_ranker.evaluator import generate_evaluation_report, create_ranking_comparison_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练LTR排序模型 (单数据源)",
        epilog="推荐使用: python run_ranking_pipeline.py --config configs/ranking_datasets.yaml"
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
        "--model-dir",
        type=str,
        default="ml_ranker/models",
        help="模型保存目录"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="ml_ranker/evaluation",
        help="评估报告保存目录"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="交叉验证折数"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="LightGBM树数量"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率"
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="使用新的pipeline模块(推荐)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="启用 LightGBM GPU 模式（需已编译/安装 GPU 版 LightGBM）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("🚀 开始训练LTR排序模型 (单数据源模式)")
    print(f"{'='*80}\n")
    
    # 如果启用pipeline模式且可用,转而使用新pipeline
    if args.use_pipeline and PIPELINE_AVAILABLE:
        print("ℹ️  使用新的Pipeline模式...")
        
        # 确定数据路径
        if args.wfo_dir is None:
            wfo_dir = find_latest_wfo_run()
        else:
            wfo_dir = Path(args.wfo_dir)
        
        if args.backtest_dir is None:
            backtest_dir = find_latest_backtest_run()
        else:
            backtest_dir = Path(args.backtest_dir)
        
        # 创建单数据源配置
        config = DatasetConfig.from_single_source(
            wfo_dir=str(wfo_dir),
            real_dir=str(backtest_dir),
            rebalance_days=8  # 默认8天
        )
        
        # 调用pipeline
        result = run_training_pipeline(
            config=config,
            model_params={
                'n_estimators': args.n_estimators,
                'learning_rate': args.learning_rate
            },
            enable_robustness=False,  # 单数据源模式默认不做稳健性评估
            save_model=True,
            output_dir=Path("ml_ranker"),
            verbose=True
        )
        
        print(f"\n✅ 训练完成")
        print(f"💡 提示: 若要使用多数据源训练,请运行:")
        print(f"   python run_ranking_pipeline.py --config configs/ranking_datasets.yaml\n")
        
        return
    
    # 原有的训练逻辑(保持不变)
    # 1. 确定数据路径
    if args.wfo_dir is None:
        print("🔍 自动查找最新WFO结果...")
        wfo_dir = find_latest_wfo_run()
    else:
        wfo_dir = Path(args.wfo_dir)
    
    if args.backtest_dir is None:
        print("🔍 自动查找最新回测结果...")
        backtest_dir = find_latest_backtest_run()
    else:
        backtest_dir = Path(args.backtest_dir)
    
    print(f"  WFO目录: {wfo_dir}")
    print(f"  回测目录: {backtest_dir}")
    
    # 2. 加载训练数据
    print(f"\n{'='*80}")
    print("📂 加载训练数据")
    print(f"{'='*80}")
    
    from ml_ranker.data_loader import load_wfo_features, load_real_backtest_results
    
    wfo_df = load_wfo_features(wfo_dir)
    real_df = load_real_backtest_results(backtest_dir)
    
    merged_df, y, metadata = build_training_dataset(
        wfo_df=wfo_df,
        real_df=real_df
    )
    
    print(f"  样本数: {len(merged_df)}")
    print(f"  目标变量: {metadata['target_col']}")
    print(f"  均值: {y.mean():.4f}, 标准差: {y.std():.4f}")
    
    # 3. 构建特征矩阵
    print(f"\n{'='*80}")
    print("🛠️ 构建特征矩阵")
    print(f"{'='*80}")
    
    X_df = build_feature_matrix(merged_df)
    X = X_df.values
    feature_names = list(X_df.columns)
    
    print(f"  特征数: {X.shape[1]}")
    print(f"  特征维度: {X.shape}")
    print(f"  样本示例: {feature_names[:5]}")
    
    # 4. 训练模型
    print(f"\n{'='*80}")
    print("🔥 训练LTR模型")
    print(f"{'='*80}")
    
    model = LTRRanker(
        objective="regression",  # 使用回归模式避免query size限制
        metric="rmse",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=6,
        num_leaves=31,
        min_data_in_leaf=20,
        verbose=-1
    )
    # 如果传入了 --use-gpu 标志，则将 GPU 请求传递给模型
    if hasattr(args, 'use_gpu') and args.use_gpu:
        model = LTRRanker(
            objective="regression",
            metric="rmse",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=6,
            num_leaves=31,
            min_data_in_leaf=20,
            verbose=-1,
            use_gpu=True,
        )
    
    model.train(
        X=pd.DataFrame(X, columns=feature_names),
        y=y,
        cv_folds=args.n_folds
    )
    
    print(f"\n  ✓ 训练完成")
    print(f"  CV Spearman: {model.cv_results[-1]['spearman_corr']:.4f}")
    
    # 5. 预测
    print(f"\n{'='*80}")
    print("🎯 预测排序")
    print(f"{'='*80}")
    
    scores, ranks = model.predict(X)
    
    print(f"  预测分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  排名范围: [1, {ranks.max()}]")
    
    # 6. 特征重要性
    print(f"\n{'='*80}")
    print("📊 特征重要性分析")
    print(f"{'='*80}")
    
    importance_df = model.get_feature_importance()
    print(f"\n  Top-15 重要特征:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:35s}: {row['importance']:10.0f}")
    
    # 7. 评估
    print(f"\n{'='*80}")
    print("📈 模型评估")
    print(f"{'='*80}")
    
    # 使用WFO原始mean_oos_ic作为baseline
    baseline_scores = merged_df["mean_oos_ic"].values
    
    eval_report = generate_evaluation_report(
        y_true=y.values,
        y_pred=scores,
        baseline_scores=baseline_scores,
        metadata=metadata,
        feature_importance=importance_df,
        output_path=Path(args.eval_dir) / "evaluation_report.json"
    )
    
    # 8. 保存模型
    print(f"\n{'='*80}")
    print("💾 保存模型")
    print(f"{'='*80}")
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "ltr_ranker"
    model.save(str(model_path))
    
    print(f"  ✓ 模型已保存: {model_path}.txt")
    
    # 9. 保存排序对比表
    print(f"\n{'='*80}")
    print("📋 生成排序对比表")
    print(f"{'='*80}")
    
    comparison_df = create_ranking_comparison_df(
        y_true=y.values,
        y_pred=scores,
        baseline_scores=baseline_scores,
        combos=merged_df["combo"].values,
        top_n=100
    )
    
    comparison_path = Path(args.eval_dir) / "ranking_comparison_top100.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    
    print(f"  ✓ Top-100对比表已保存: {comparison_path}")
    
    # 10. 总结
    print(f"\n{'='*80}")
    print("✅ 训练完成")
    print(f"{'='*80}")
    
    print(f"\n  模型性能:")
    print(f"    Spearman相关性: {eval_report['model_metrics']['spearman_corr']:.4f}")
    print(f"    NDCG@10: {eval_report['model_metrics'].get('ndcg@10', 0):.4f}")
    print(f"    Top-10命中率: {eval_report['top10_analysis']['model_hits']}/10")
    print(f"    Top-10平均收益: {eval_report['top10_analysis']['model_pred_mean']:.4f}")
    
    print(f"\n  输出文件:")
    print(f"    模型: {model_path}.txt")
    print(f"    元数据: {model_path}.meta.pkl")
    print(f"    评估报告: {Path(args.eval_dir) / 'evaluation_report.json'}")
    print(f"    对比表: {comparison_path}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
