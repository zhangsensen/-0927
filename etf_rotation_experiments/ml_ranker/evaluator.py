"""
评估模块: 排序质量评估指标和报告生成
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def compute_spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    计算Spearman排序相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Spearman相关系数
    """
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def compute_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    计算NDCG@K分数
    
    Args:
        y_true: 真实相关性分数
        y_pred: 预测排序分数
        k: Top-K, None表示使用全部
        
    Returns:
        NDCG分数
    """
    # sklearn的ndcg_score需要2D数组且不能有负值
    # 将y_true平移到非负范围
    y_true_shifted = y_true - y_true.min() + 1e-6
    
    y_true_2d = y_true_shifted.reshape(1, -1)
    y_pred_2d = y_pred.reshape(1, -1)
    
    return ndcg_score(y_true_2d, y_pred_2d, k=k)


def compute_topk_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    计算Top-K相关指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        k: Top-K
        
    Returns:
        指标dict包含命中率、平均收益等
    """
    # 获取真实Top-K索引
    true_topk_idx = np.argsort(y_true)[-k:][::-1]
    
    # 获取预测Top-K索引
    pred_topk_idx = np.argsort(y_pred)[-k:][::-1]
    
    # 命中数量
    hits = len(set(true_topk_idx) & set(pred_topk_idx))
    hit_rate = hits / k
    
    # 预测Top-K的真实平均值
    pred_topk_true_mean = y_true[pred_topk_idx].mean()
    
    # 真实Top-K的真实平均值 (理论最优)
    true_topk_true_mean = y_true[true_topk_idx].mean()
    
    # 全体平均值 (baseline)
    overall_mean = y_true.mean()
    
    # 提升倍数
    lift_vs_baseline = pred_topk_true_mean / overall_mean if overall_mean != 0 else 0
    lift_vs_optimal = pred_topk_true_mean / true_topk_true_mean if true_topk_true_mean != 0 else 0
    
    return {
        f"top{k}_hit_rate": hit_rate,
        f"top{k}_hits": hits,
        f"top{k}_pred_mean": pred_topk_true_mean,
        f"top{k}_true_mean": true_topk_true_mean,
        f"top{k}_overall_mean": overall_mean,
        f"top{k}_lift_vs_baseline": lift_vs_baseline,
        f"top{k}_lift_vs_optimal": lift_vs_optimal,
    }


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    计算完整的排序评估指标
    
    Args:
        y_true: 真实目标值
        y_pred: 预测排序分数
        metadata: 元信息 (可选包含combo等)
        
    Returns:
        完整评估指标dict
    """
    metrics = {}
    
    # 1. Spearman相关性
    metrics["spearman_corr"] = compute_spearman_correlation(y_true, y_pred)
    
    # 2. NDCG@K
    for k in [5, 10, 20, 50, 100]:
        if k <= len(y_true):
            metrics[f"ndcg@{k}"] = compute_ndcg(y_true, y_pred, k=k)
    
    # 3. Top-K指标
    for k in [10, 20, 50]:
        if k <= len(y_true):
            topk_metrics = compute_topk_metrics(y_true, y_pred, k=k)
            metrics.update(topk_metrics)
    
    # 4. 基础统计
    metrics["n_samples"] = len(y_true)
    metrics["y_true_mean"] = float(np.mean(y_true))
    metrics["y_true_std"] = float(np.std(y_true))
    metrics["y_pred_mean"] = float(np.mean(y_pred))
    metrics["y_pred_std"] = float(np.std(y_pred))
    
    return metrics


def compare_with_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_name: str = "WFO原始排序"
) -> pd.DataFrame:
    """
    对比模型预测与baseline排序
    
    Args:
        y_true: 真实目标值
        y_pred: 模型预测分数
        baseline_scores: baseline排序分数 (如WFO的mean_oos_ic)
        baseline_name: baseline名称
        
    Returns:
        对比结果DataFrame
    """
    # 计算模型指标
    model_metrics = compute_ranking_metrics(y_true, y_pred)
    
    # 计算baseline指标
    baseline_metrics = compute_ranking_metrics(y_true, baseline_scores)
    
    # 构建对比表
    comparison = []
    
    for key in sorted(model_metrics.keys()):
        if key in baseline_metrics and not key.startswith("y_") and not key.startswith("n_"):
            model_val = model_metrics[key]
            baseline_val = baseline_metrics[key]
            
            # 计算提升
            if baseline_val != 0:
                improvement = (model_val - baseline_val) / abs(baseline_val) * 100
            else:
                improvement = 0
            
            comparison.append({
                "指标": key,
                f"{baseline_name}": f"{baseline_val:.4f}",
                "LTR模型": f"{model_val:.4f}",
                "提升(%)": f"{improvement:+.2f}%"
            })
    
    return pd.DataFrame(comparison)


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_scores: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    生成完整评估报告
    
    Args:
        y_true: 真实目标值
        y_pred: 模型预测分数
        baseline_scores: baseline分数
        metadata: 元信息 (包含combo等)
        feature_importance: 特征重要性DataFrame
        output_path: 报告保存路径 (可选)
        
    Returns:
        报告dict
    """
    print(f"\n{'='*80}")
    print("生成评估报告")
    print(f"{'='*80}")
    
    # 1. 计算指标
    model_metrics = compute_ranking_metrics(y_true, y_pred, metadata)
    baseline_metrics = compute_ranking_metrics(y_true, baseline_scores, metadata)
    
    # 2. 对比表
    comparison_df = compare_with_baseline(y_true, y_pred, baseline_scores)
    
    print("\n📊 模型 vs Baseline 对比:")
    print(comparison_df.to_string(index=False))
    
    # 3. Top-K分析
    print(f"\n🏆 Top-10 策略分析:")
    
    # 真实Top-10
    true_top10_idx = np.argsort(y_true)[-10:][::-1]
    print(f"  真实Top-10平均收益: {y_true[true_top10_idx].mean():.4f}")
    
    # 模型预测Top-10
    pred_top10_idx = np.argsort(y_pred)[-10:][::-1]
    pred_top10_true_mean = y_true[pred_top10_idx].mean()
    print(f"  模型Top-10平均收益: {pred_top10_true_mean:.4f}")
    
    # Baseline预测Top-10
    baseline_top10_idx = np.argsort(baseline_scores)[-10:][::-1]
    baseline_top10_true_mean = y_true[baseline_top10_idx].mean()
    print(f"  Baseline Top-10平均收益: {baseline_top10_true_mean:.4f}")
    
    # 命中分析
    model_hits = len(set(true_top10_idx) & set(pred_top10_idx))
    baseline_hits = len(set(true_top10_idx) & set(baseline_top10_idx))
    print(f"  模型命中数: {model_hits}/10")
    print(f"  Baseline命中数: {baseline_hits}/10")
    
    # 4. 特征重要性
    if feature_importance is not None:
        print(f"\n🔍 Top-15 重要特征:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:10.0f}")
    
    # 5. 构建报告dict
    report = {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison_df.to_dict(orient="records"),
        "top10_analysis": {
            "true_mean": float(y_true[true_top10_idx].mean()),
            "model_pred_mean": float(pred_top10_true_mean),
            "baseline_pred_mean": float(baseline_top10_true_mean),
            "model_hits": int(model_hits),
            "baseline_hits": int(baseline_hits),
        }
    }
    
    if feature_importance is not None:
        report["feature_importance"] = feature_importance.head(20).to_dict(orient="records")
    
    # 6. 保存报告
    if output_path:
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 报告已保存: {output_path}")
    
    print(f"\n{'='*80}")
    
    return report


def create_ranking_comparison_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_scores: np.ndarray,
    combos: np.ndarray,
    top_n: int = 100
) -> pd.DataFrame:
    """
    创建排序对比表 (展示Top-N策略)
    
    Args:
        y_true: 真实目标值
        y_pred: 模型预测分数
        baseline_scores: baseline分数
        combos: 策略名称数组
        top_n: 展示前N个
        
    Returns:
        排序对比DataFrame
    """
    df = pd.DataFrame({
        "combo": combos,
        "true_value": y_true,
        "pred_score": y_pred,
        "baseline_score": baseline_scores,
    })
    
    # 计算各自排名
    df["true_rank"] = df["true_value"].rank(ascending=False, method="min").astype(int)
    df["pred_rank"] = df["pred_score"].rank(ascending=False, method="min").astype(int)
    df["baseline_rank"] = df["baseline_score"].rank(ascending=False, method="min").astype(int)
    
    # 按模型预测排序
    df = df.sort_values("pred_rank").reset_index(drop=True)
    
    # 只保留Top-N
    df_top = df.head(top_n).copy()
    
    # 计算排名变化
    df_top["rank_change_vs_baseline"] = df_top["baseline_rank"] - df_top["pred_rank"]
    df_top["rank_gap_to_true"] = df_top["true_rank"] - df_top["pred_rank"]
    
    return df_top
