#!/usr/bin/env python3
"""深度分析监督学习实际运行结果"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# 添加路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_ranking import evaluation, models

# 配置
DATASET_PATH = Path(__file__).parent.parent / "ml_ranking/data/training_dataset.parquet"
OUTPUT_PATH = Path(__file__).parent.parent / "ml_ranking/ACTUAL_RESULTS_ANALYSIS.json"

def analyze_results():
    """执行完整分析"""
    
    print("="*80)
    print("🔍 监督学习实际结果深度分析")
    print("="*80)
    
    # 1. 加载数据集
    print("\n[1/6] 加载数据集...")
    df = pd.read_parquet(DATASET_PATH)
    label_col = "oos_compound_sharpe"
    
    dataset_stats = {
        "rows": len(df),
        "features": len(df.columns) - 1,
        "columns": df.columns.tolist(),
        "label_stats": df[label_col].describe().to_dict(),
        "missing_rates": {
            col: float(df[col].isnull().mean()) 
            for col in df.columns if df[col].isnull().mean() > 0
        }
    }
    
    print(f"   ✅ 样本数: {dataset_stats['rows']}")
    print(f"   ✅ 特征数: {dataset_stats['features']}")
    print(f"   ✅ 标签均值: {dataset_stats['label_stats']['mean']:.4f}")
    print(f"   ✅ 标签范围: [{dataset_stats['label_stats']['min']:.4f}, {dataset_stats['label_stats']['max']:.4f}]")
    
    # 2. 数据质量检查
    print("\n[2/6] 数据质量检查...")
    
    quality_checks = {
        "label_in_features": label_col in [c for c in df.columns if c != label_col],
        "duplicates": int(df.duplicated().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
        "overall_missing_rate": float(df.isnull().mean().mean()),
        "high_missing_features": [
            col for col in df.columns 
            if col != label_col and df[col].isnull().mean() > 0.05
        ]
    }
    
    print(f"   {'✅' if not quality_checks['label_in_features'] else '🚨'} 标签泄露: {quality_checks['label_in_features']}")
    print(f"   ✅ 重复样本: {quality_checks['duplicates']}")
    print(f"   ✅ 缺失率: {quality_checks['overall_missing_rate']*100:.2f}%")
    print(f"   ⚠️  高缺失特征数: {len(quality_checks['high_missing_features'])}")
    
    # 3. 单次分割训练
    print("\n[3/6] 单次80/20分割训练...")
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy(dtype=float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 预处理
    X_train_clean = X_train.apply(pd.to_numeric, errors="coerce")
    X_test_clean = X_test.apply(pd.to_numeric, errors="coerce")
    median_vals = X_train_clean.median()
    X_train_clean = X_train_clean.fillna(median_vals).fillna(0.0)
    X_test_clean = X_test_clean.fillna(median_vals).fillna(0.0)
    
    print(f"   训练集: {len(X_train)} | 测试集: {len(X_test)}")
    
    model_registry = models.baseline_model_registry()
    single_split_results = {}
    
    for name, model in model_registry.items():
        print(f"   训练 {name:15s}...", end=" ")
        try:
            model.fit(X_train_clean, y_train)
            y_pred = model.predict(X_test_clean)
            
            eval_result = evaluation.evaluate_predictions(
                y_true=y_test, y_pred=y_pred, model_name=name
            )
            
            single_split_results[name] = eval_result.metrics
            
            spearman = eval_result.metrics['spearman']
            top50 = eval_result.metrics['top50_overlap']
            ndcg50 = eval_result.metrics['ndcg@50']
            
            print(f"Spearman={spearman:.4f}, Top50={top50:.2%}, NDCG@50={ndcg50:.4f}")
            
            # 判断是否异常
            if spearman > 0.90:
                print(f"      🚨 警告: Spearman过高 ({spearman:.4f}), 可能存在过拟合或数据泄露!")
            elif spearman > 0.75:
                print(f"      ⚠️  注意: Spearman较高 ({spearman:.4f}), 建议验证")
            elif spearman >= 0.60:
                print(f"      ✅ 正常: Spearman在合理范围")
            else:
                print(f"      ⚠️  偏低: Spearman<0.60, 特征工程可能需要改进")
                
        except Exception as e:
            print(f"失败: {e}")
    
    # 4. 5折交叉验证
    print("\n[4/6] 5折交叉验证...")
    X_all = X.apply(pd.to_numeric, errors="coerce")
    median_all = X_all.median()
    X_all_clean = X_all.fillna(median_all).fillna(0.0)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    for name in model_registry.keys():
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_all_clean)):
            X_tr = X_all_clean.iloc[train_idx]
            y_tr = y[train_idx]
            X_val = X_all_clean.iloc[val_idx]
            y_val = y[val_idx]
            
            if name == "elasticnet":
                fold_model = models.make_linear_model(models.ModelConfig())
            elif name == "decision_tree":
                fold_model = models.make_tree_model(models.ModelConfig())
            elif name == "lgbm_regressor":
                fold_model = models.make_lgbm_regressor(models.ModelConfig())
            else:
                continue
            
            try:
                fold_model.fit(X_tr, y_tr)
                y_pred_val = fold_model.predict(X_val)
                eval_res = evaluation.evaluate_predictions(
                    y_true=y_val, y_pred=y_pred_val, model_name=f"{name}_f{fold_idx}"
                )
                fold_metrics.append(eval_res.metrics)
            except Exception:
                continue
        
        if fold_metrics:
            aggregated = {}
            for metric_name in fold_metrics[0].keys():
                vals = [m[metric_name] for m in fold_metrics]
                aggregated[f"{metric_name}_mean"] = float(np.mean(vals))
                aggregated[f"{metric_name}_std"] = float(np.std(vals))
            
            cv_results[name] = {
                "n_folds": len(fold_metrics),
                "aggregated": aggregated,
            }
            
            sp_mean = aggregated['spearman_mean']
            sp_std = aggregated['spearman_std']
            top50_mean = aggregated['top50_overlap_mean']
            
            print(f"   {name:15s}: Spearman={sp_mean:.4f}±{sp_std:.4f}, Top50={top50_mean:.2%}")
    
    # 5. Top-K分析
    print("\n[5/6] Top-K预测质量分析...")
    top_ks = [50, 100, 200, 500, 1000, 2000]
    topk_analysis = {}
    
    for name, model in model_registry.items():
        if name not in single_split_results:
            continue
        
        try:
            model.fit(X_all_clean, y)
            y_pred_all = model.predict(X_all_clean)
            
            ranked_idx = np.argsort(-y_pred_all)
            true_ranked_idx = np.argsort(-y)
            
            model_topk = {}
            for k in top_ks:
                if k > len(y):
                    k = len(y)
                
                pred_topk_idx = ranked_idx[:k]
                pred_topk_actual = y[pred_topk_idx]
                
                true_topk_idx = true_ranked_idx[:k]
                oracle_actual = y[true_topk_idx]
                
                model_topk[f"top{k}"] = {
                    "mean_actual": float(np.mean(pred_topk_actual)),
                    "median_actual": float(np.median(pred_topk_actual)),
                    "std_actual": float(np.std(pred_topk_actual)),
                    "oracle_mean": float(np.mean(oracle_actual)),
                    "oracle_median": float(np.median(oracle_actual)),
                    "gap": float(np.mean(pred_topk_actual) - np.mean(oracle_actual)),
                }
            
            topk_analysis[name] = model_topk
            
            # 打印关键K值
            for k_name in ['top50', 'top2000']:
                if k_name in model_topk:
                    stats = model_topk[k_name]
                    print(f"   {name:15s} {k_name:8s}: "
                          f"pred={stats['mean_actual']:.4f}, "
                          f"oracle={stats['oracle_mean']:.4f}, "
                          f"gap={stats['gap']:.4f}")
        
        except Exception as e:
            print(f"   {name}: 失败 - {e}")
    
    # 6. 对比分析
    print("\n[6/6] 与规划目标对比...")
    
    plan_targets = {
        "phase1": {"spearman": 0.60, "top50_overlap": 0.18},
        "phase2_linear": {"spearman": 0.65, "top50_overlap": 0.25},
        "phase2_gbm": {"spearman": 0.70, "top50_overlap": 0.35},
        "phase2_lambdamart": {"spearman": 0.70, "top50_overlap": 0.40},
        "mvp": {"spearman": 0.70, "top50_overlap": 0.50},
        "ideal": {"spearman": 0.75, "top50_overlap": 0.70},
    }
    
    comparison = {}
    for name, metrics in single_split_results.items():
        actual_sp = metrics['spearman']
        actual_top50 = metrics['top50_overlap']
        
        status = "unknown"
        if actual_sp >= plan_targets["ideal"]["spearman"]:
            status = "超越理想目标"
        elif actual_sp >= plan_targets["mvp"]["spearman"]:
            status = "达到MVP标准"
        elif actual_sp >= plan_targets["phase2_gbm"]["spearman"]:
            status = "符合Phase2预期"
        elif actual_sp >= plan_targets["phase1"]["spearman"]:
            status = "符合Phase1基线"
        else:
            status = "低于Phase1基线"
        
        comparison[name] = {
            "actual_spearman": actual_sp,
            "actual_top50": actual_top50,
            "status": status,
            "vs_mvp_spearman": actual_sp - plan_targets["mvp"]["spearman"],
            "vs_mvp_top50": actual_top50 - plan_targets["mvp"]["top50_overlap"],
        }
        
        print(f"   {name:15s}: {status}")
        print(f"      Spearman: {actual_sp:.4f} (MVP目标: 0.70, 差距: {actual_sp-0.70:+.4f})")
        print(f"      Top50:    {actual_top50:.2%} (MVP目标: 50%, 差距: {actual_top50-0.50:+.2%})")
    
    # 保存完整报告
    report = {
        "dataset_stats": dataset_stats,
        "quality_checks": quality_checks,
        "single_split_results": single_split_results,
        "cross_validation_results": cv_results,
        "top_k_analysis": topk_analysis,
        "comparison_to_plan": comparison,
    }
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 完整报告已保存: {OUTPUT_PATH}")
    
    # 最终判断
    print("\n" + "="*80)
    print("🎯 最终判断")
    print("="*80)
    
    best_model = max(single_split_results.items(), key=lambda x: x[1]['spearman'])
    best_name, best_metrics = best_model
    
    print(f"\n最佳模型: {best_name}")
    print(f"  Spearman: {best_metrics['spearman']:.4f}")
    print(f"  Top50 overlap: {best_metrics['top50_overlap']:.2%}")
    print(f"  NDCG@50: {best_metrics['ndcg@50']:.4f}")
    
    if best_metrics['spearman'] > 0.95:
        print("\n🚨 严重警告: 结果异常优秀 (Spearman>0.95)")
        print("   可能原因:")
        print("   1. 数据泄露 (标签信息泄露到特征中)")
        print("   2. 样本量过小导致过拟合")
        print("   3. 特征与标签高度相关但缺乏泛化性")
        print("   建议: 立即审计特征工程代码，使用独立数据集验证")
    elif best_metrics['spearman'] > 0.85:
        print("\n⚠️  警告: 结果优于预期 (Spearman>0.85)")
        print("   这超越了Phase5理想目标(0.75)")
        print("   建议: 验证CV结果稳定性，检查是否存在轻微过拟合")
    elif best_metrics['spearman'] >= 0.70:
        print("\n✅ 优秀: 已达到MVP标准!")
        print("   可以进入Phase3特征工程深化")
    elif best_metrics['spearman'] >= 0.60:
        print("\n✅ 合格: 符合Phase1-2预期")
        print("   建议继续Phase2特征工程优化")
    else:
        print("\n⚠️  待改进: Spearman<0.60")
        print("   建议: 检查特征工程和数据预处理")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_results()
