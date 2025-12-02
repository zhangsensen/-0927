"""
WFO 校准器训练脚本
================================================================================
使用完整回测结果训练 WFO → Sharpe 校准模型。

背景
----
WFO 的 IC 排名与实际 Sharpe 相关性较弱 (r ≈ 0.1-0.4)。
通过训练回归模型，我们可以更准确地预测实际 Sharpe。

评分公式
--------
composite_score = 0.5 * IC_rank + 0.3 * stability_rank + 0.2 * simplicity_rank

其中：
- IC_rank: mean_oos_ic 的百分位排名
- stability_rank: 基于 IC 标准差的稳定性排名
- simplicity_rank: (1 / combo_size) 归一化

模型选择
--------
使用 GradientBoosting 回归器，特征包括：
- mean_oos_ic
- oos_ic_std  
- oos_ic_ir
- positive_rate
- combo_size
- best_rebalance_freq
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib

# 确保可以导入项目模块
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
for p in (_HERE, _PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)


def load_latest_backtest_results():
    """加载最新的完整回测结果"""
    results_dir = Path("results_combo_wfo")
    
    # 找到最新目录
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("202")], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No backtest results found")
    
    latest_dir = run_dirs[0]
    print(f"📂 加载结果目录: {latest_dir.name}")
    
    # 查找完整结果文件
    full_result_files = list(latest_dir.glob("*_full.csv"))
    if not full_result_files:
        raise FileNotFoundError(f"No full result CSV found in {latest_dir}")
    
    result_file = full_result_files[0]
    print(f"📄 读取结果文件: {result_file.name}")
    
    df = pd.read_csv(result_file)
    print(f"  总组合数: {len(df)}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    准备训练特征
    
    返回：
        X: 特征矩阵
        y: 目标变量 (Sharpe)
        feature_names: 特征名称列表
    """
    # 特征列
    feature_cols = [
        "wfo_ic",          # mean_oos_ic
        "combo_size",      # 组合大小
    ]
    
    # 检查可用列
    available_cols = []
    for col in feature_cols:
        if col in df.columns:
            available_cols.append(col)
        else:
            print(f"  ⚠️ 列 {col} 不存在，跳过")
    
    # 添加衍生特征
    df = df.copy()
    
    # IC 稳定性（如果没有 std，用 IC 本身的变异系数近似）
    if "wfo_ic" in df.columns:
        # 创建 IC 的排名
        df["ic_rank"] = df["wfo_ic"].rank(pct=True)
        available_cols.append("ic_rank")
    
    # 简单性分数 (1/combo_size 归一化)
    if "combo_size" in df.columns:
        df["simplicity"] = 1.0 / df["combo_size"]
        available_cols.append("simplicity")
    
    # 最终特征
    feature_names = available_cols
    X = df[feature_names].values
    
    # 目标变量
    y = df["sharpe"].values
    
    # 处理 NaN
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"  有效样本数: {len(y)}")
    print(f"  特征: {feature_names}")
    
    return X, y, feature_names


def train_calibrator(X: np.ndarray, y: np.ndarray, feature_names: list):
    """
    训练校准模型
    """
    print("\n🔧 训练校准模型...")
    
    # 使用 GradientBoosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=50,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 训练最终模型
    model.fit(X, y)
    
    # 特征重要性
    importances = model.feature_importances_
    print("\n📊 特征重要性:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
    
    return model


def create_composite_scorer(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建复合评分：0.5*IC + 0.3*Stability + 0.2*Simplicity
    """
    df = df.copy()
    
    # IC 排名 (越高越好)
    df["ic_rank_pct"] = df["wfo_ic"].rank(pct=True)
    
    # 稳定性排名（基于 IC 的绝对值，越稳定越好）
    # 这里简化为：IC 越接近 0 越不稳定，越远离 0 越稳定
    df["stability_rank_pct"] = np.abs(df["wfo_ic"]).rank(pct=True)
    
    # 简单性排名 (combo_size 越小越好)
    df["simplicity_rank_pct"] = (1.0 / df["combo_size"]).rank(pct=True)
    
    # 复合评分
    df["composite_score"] = (
        0.5 * df["ic_rank_pct"] + 
        0.3 * df["stability_rank_pct"] + 
        0.2 * df["simplicity_rank_pct"]
    )
    
    return df


def evaluate_ranking_methods(df: pd.DataFrame):
    """
    评估不同排名方法与实际 Sharpe 的相关性
    """
    from scipy.stats import spearmanr
    
    print("\n📈 排名方法评估:")
    print("-" * 60)
    
    # 原始 IC 排名
    ic_corr, ic_pval = spearmanr(df["wfo_ic"], df["sharpe"])
    print(f"  原始 WFO IC vs Sharpe: r={ic_corr:.3f}, p={ic_pval:.4f}")
    
    # 复合评分
    df_scored = create_composite_scorer(df)
    comp_corr, comp_pval = spearmanr(df_scored["composite_score"], df["sharpe"])
    print(f"  复合评分 vs Sharpe:   r={comp_corr:.3f}, p={comp_pval:.4f}")
    
    # 检验是否显著改善
    improvement = comp_corr - ic_corr
    print(f"\n  改善: {improvement:+.3f} ({'✅ 有效' if improvement > 0.05 else '⚠️ 有限'})")
    
    return df_scored


def save_calibrator(model, feature_names: list, output_path: Path):
    """保存校准器"""
    calibrator_data = {
        "model": model,
        "feature_names": feature_names,
        "version": "v2",
        "timestamp": datetime.now().isoformat(),
    }
    
    joblib.dump(calibrator_data, output_path)
    print(f"\n💾 校准器已保存: {output_path}")


def main():
    print("=" * 80)
    print("WFO 校准器训练")
    print("=" * 80)
    
    # 加载数据
    df = load_latest_backtest_results()
    
    # 评估现有排名方法
    df_scored = evaluate_ranking_methods(df)
    
    # 准备特征
    X, y, feature_names = prepare_features(df)
    
    # 训练模型
    model = train_calibrator(X, y, feature_names)
    
    # 保存模型
    output_path = Path("results/calibrator_gbdt_full.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibrator(model, feature_names, output_path)
    
    # 输出新排名的 Top 10
    print("\n🏆 使用复合评分的新 Top 10:")
    print("-" * 80)
    
    df_sorted = df_scored.sort_values("composite_score", ascending=False).head(10)
    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"  {idx:2d}. {row['combo']}")
        print(f"      Composite: {row['composite_score']:.3f} | Sharpe: {row['sharpe']:.3f} | IC: {row['wfo_ic']:.4f}")
    
    print("\n✅ 校准器训练完成")
    print("\n💡 下一步:")
    print("  1. 重新运行 WFO 优化，将自动使用新校准器")
    print("  2. 或运行回测比较新旧排名")


if __name__ == "__main__":
    main()
