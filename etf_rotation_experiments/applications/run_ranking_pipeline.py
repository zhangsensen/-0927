#!/usr/bin/env python3
"""
统一训练Pipeline入口: 一键完成训练+评估+稳健性验证

本脚本是ML Ranker系统的统一训练入口,支持:
- 单/多数据源训练(通过YAML配置)
- 自动特征工程和模型训练
- 完整的模型评估和稳健性验证
- 模型保存和报告生成

使用示例:
    # 基础训练(使用默认配置)
    python applications/run_ranking_pipeline.py
    
    # 指定配置文件
    python applications/run_ranking_pipeline.py --config configs/ranking_datasets.yaml
    
    # 快速训练(跳过稳健性评估)
    python applications/run_ranking_pipeline.py --no-robustness
    
    # 自定义参数
    python applications/run_ranking_pipeline.py --n-estimators 1000 --learning-rate 0.03
"""
import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

from strategies.ml_ranker.config import DatasetConfig
from strategies.ml_ranker.pipeline import run_training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一训练Pipeline: 多数据源 + 训练 + 稳健性评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  %(prog)s --config configs/ranking_datasets.yaml
  
  # 快速训练(跳过稳健性评估)
  %(prog)s --config configs/ranking_datasets.yaml --no-robustness
  
  # 自定义模型参数
  %(prog)s --n-estimators 1000 --learning-rate 0.03
  
  # 完整参数示例
  %(prog)s \
    --config configs/ranking_datasets.yaml \
    --n-estimators 500 \
    --learning-rate 0.05 \
    --robustness-folds 10 \
    --robustness-repeats 10 \
    --output-dir ml_ranker
        """
    )
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ranking_datasets.yaml",
        help="数据源配置YAML路径 (默认: configs/ranking_datasets.yaml)"
    )
    
    # 稳健性评估
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="跳过稳健性评估,加快训练速度(约节省5分钟)"
    )
    
    parser.add_argument(
        "--robustness-folds",
        type=int,
        default=5,
        help="稳健性评估K-Fold折数 (默认: 5)"
    )
    
    parser.add_argument(
        "--robustness-repeats",
        type=int,
        default=5,
        help="稳健性评估Repeated Holdout重复次数 (默认: 5)"
    )
    
    parser.add_argument(
        "--robustness-estimators",
        type=int,
        default=300,
        help="稳健性评估每个模型的树数量 (默认: 300, 建议不超过500以控制时间)"
    )
    
    # 模型参数
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="模型树数量 (默认: 500)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率 (默认: 0.05)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="树最大深度 (默认: 6)"
    )
    
    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_ranker",
        help="输出根目录 (默认: ml_ranker)"
    )
    
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="不保存模型(仅用于测试)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式,减少日志输出"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 显示欢迎信息
    if not args.quiet:
        print(f"\n{'='*80}")
        print("🎯 ML Ranker 统一训练Pipeline")
        print(f"{'='*80}\n")
        print(f"配置文件: {args.config}")
        print(f"输出目录: {args.output_dir}")
        print(f"稳健性评估: {'禁用' if args.no_robustness else '启用'}")
        print(f"模型保存: {'禁用' if args.no_save_model else '启用'}")
    
    # 1. 加载配置
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"\n❌ 错误: 配置文件不存在: {args.config}")
            print(f"\n请确保配置文件存在,或使用--config指定正确路径")
            print(f"示例: python {sys.argv[0]} --config configs/ranking_datasets.yaml")
            return 1
        
        config = DatasetConfig.from_yaml(args.config)
        
        if not args.quiet:
            print(f"\n数据源配置:")
            print(f"  数据源数量: {len(config.datasets)}")
            print(f"  目标列: {config.target_col}")
            print(f"  次要目标: {config.secondary_target or 'None'}")
            print(f"\n数据源列表:")
            for idx, ds in enumerate(config.datasets, 1):
                print(f"  [{idx}] {ds.display_name}")
                print(f"      WFO: {ds.wfo_dir}")
                print(f"      回测: {ds.real_dir}")
    
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        return 1
    
    # 2. 准备参数
    model_params = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth
    }
    
    robustness_params = {
        'n_splits': args.robustness_folds,
        'n_repeats': args.robustness_repeats,
        'n_estimators': args.robustness_estimators
    }
    
    if not args.quiet:
        print(f"\n模型参数:")
        print(f"  n_estimators: {model_params['n_estimators']}")
        print(f"  learning_rate: {model_params['learning_rate']}")
        print(f"  max_depth: {model_params['max_depth']}")
        
        if not args.no_robustness:
            print(f"\n稳健性评估参数:")
            print(f"  K-Fold折数: {robustness_params['n_splits']}")
            print(f"  Repeated Holdout次数: {robustness_params['n_repeats']}")
            print(f"  每个模型树数: {robustness_params['n_estimators']}")
            
            total_models = robustness_params['n_splits'] + robustness_params['n_repeats']
            est_time = total_models * 30  # 每个模型约30秒
            print(f"  预计耗时: ~{est_time//60}分钟 (训练{total_models}个模型)")
    
    # 3. 运行Pipeline
    try:
        result = run_training_pipeline(
            config=config,
            model_params=model_params,
            enable_robustness=not args.no_robustness,
            robustness_params=robustness_params,
            save_model=not args.no_save_model,
            output_dir=Path(args.output_dir),
            verbose=not args.quiet
        )
    
    except Exception as e:
        print(f"\n❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. 显示总结
    if not args.quiet:
        print(f"\n{'='*80}")
        print("🎉 Pipeline执行完成!")
        print(f"{'='*80}\n")
        
        eval_report = result['evaluation']
        print(f"📊 模型性能总结:")
        print(f"  Spearman相关性: {eval_report['model_metrics']['spearman_corr']:.4f}")
        print(f"  NDCG@10: {eval_report['model_metrics'].get('ndcg@10', 0):.4f}")
        print(f"  NDCG@50: {eval_report['model_metrics'].get('ndcg@50', 0):.4f}")
        print(f"  Top-10命中率: {eval_report['top10_analysis']['model_hits']}/10")
        print(f"  Top-10平均收益: {eval_report['top10_analysis']['model_pred_mean']:.4f}")
        
        if result['robustness']:
            rob_report = result['robustness']
            kf_spear = rob_report['kfold_cv']['metrics']['model_spearman']
            rh_spear = rob_report['repeated_holdout']['metrics']['model_spearman']
            
            print(f"\n🔬 稳健性分析:")
            print(f"  K-Fold CV Spearman: {kf_spear['mean']:.4f} ± {kf_spear['std']:.4f}")
            print(f"  Repeated Holdout Spearman: {rh_spear['mean']:.4f} ± {rh_spear['std']:.4f}")
            
            avg_std = (kf_spear['std'] + rh_spear['std']) / 2
            if avg_std < 0.03:
                print(f"  稳定性评价: ✅ 优秀 (std={avg_std:.4f} < 0.03)")
            elif avg_std < 0.08:
                print(f"  稳定性评价: ✅ 良好 (std={avg_std:.4f} < 0.08)")
            else:
                print(f"  稳定性评价: ⚠️  一般 (std={avg_std:.4f} >= 0.08)")
        
        print(f"\n📁 输出文件:")
        for key, path in result['output_paths'].items():
            if path:
                print(f"  {key}: {path}")
        
        print(f"\n💡 后续操作:")
        print(f"  1. 查看详细评估报告:")
        print(f"     cat {result['output_paths']['evaluation']}")
        
        if result['output_paths']['robustness']:
            print(f"  2. 查看稳健性报告:")
            print(f"     cat {result['output_paths']['robustness']}")
        
        if result['output_paths']['model']:
            model_base = result['output_paths']['model'].replace('.txt', '')
            print(f"  3. 应用模型对新WFO排序:")
            print(f"     python applications/apply_ranker.py --model {model_base} --wfo-dir <新WFO目录> --top-k 50")
        
        print(f"  4. 查看排序对比表:")
        print(f"     open {result['output_paths']['comparison']}")
        
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
