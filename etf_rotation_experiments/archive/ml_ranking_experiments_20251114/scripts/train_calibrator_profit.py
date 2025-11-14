#!/usr/bin/env python3
"""
训练“利润导向”校准器（target=annual_ret），用于将 WFO 输出排序为“更可能高年化净收益”的顺序。

输入
----
1) WFO全量结果: results/run_*/all_combos.parquet
2) 全量回测结果: results_combo_wfo/<run_ts>_*/top12597_backtest_by_ic_<run_ts>_*_full.csv
   - 若不存在full，可回退到 top12597_backtest_by_ic_*.csv（但建议优先 full）

特征
----
- mean_oos_ic, oos_ic_std, positive_rate, stability_score, combo_size, best_rebalance_freq
  （全部来自 WFO all_combos.parquet）

输出
----
- results/calibrator_gbdt_profit.joblib
  {'model','scaler','feature_names','train_history'}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


def _find_latest_run() -> Path:
    roots = [Path("results").resolve()]
    # 兼容稳定仓
    repo_root = Path(__file__).resolve().parents[2]
    roots.append((repo_root / "etf_rotation_optimized" / "results").resolve())
    run_dirs = []
    for r in roots:
        if r.exists():
            run_dirs.extend([d for d in r.glob("run_*") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"未找到 run_* 目录。已检查: {roots}")
    run_dirs = sorted({d.resolve() for d in run_dirs}, reverse=True)
    return run_dirs[0]


def _find_backtest_full(run_ts: str) -> Path | None:
    base = Path("results_combo_wfo")
    if not base.exists():
        return None
    # 优先找 full.csv
    cands = list(base.glob(f"{run_ts}_*/top12597_backtest_by_ic_{run_ts}_*_full.csv"))
    if cands:
        return sorted(cands, reverse=True)[0]
    # 回退非full
    cands2 = list(base.glob(f"{run_ts}_*/top12597_backtest_by_ic_{run_ts}_*.csv"))
    if cands2:
        return sorted(cands2, reverse=True)[0]
    return None


def main():
    print("=" * 100)
    print("开始训练利润校准器 (target=annual_ret)")
    print("=" * 100)
    
    latest_run = _find_latest_run()
    run_ts = latest_run.name.replace("run_", "")
    print(f"✓ 找到最新 WFO run: {latest_run}")
    
    wfo_file = latest_run / "all_combos.parquet"
    if not wfo_file.exists():
        raise FileNotFoundError(f"缺少 WFO 全量文件: {wfo_file}")
    print(f"✓ WFO 文件: {wfo_file}")
    
    backtest_file = _find_backtest_full(run_ts)
    if backtest_file is None or not backtest_file.exists():
        raise FileNotFoundError(f"未找到全量回测CSV，预期在 results_combo_wfo 下与 {run_ts} 匹配")
    print(f"✓ 回测文件: {backtest_file}")
    print()

    print("加载数据...")
    wfo_df = pd.read_parquet(wfo_file)
    bt_df = pd.read_csv(backtest_file)
    print(f"✓ WFO 组合数: {len(wfo_df)}")
    print(f"✓ 回测组合数: {len(bt_df)}")
    print()

    # 规范列
    required = ["combo", "mean_oos_ic", "oos_ic_std", "positive_rate", "stability_score", "combo_size"]
    for c in required:
        if c not in wfo_df.columns:
            raise KeyError(f"WFO结果缺失列: {c}")
    if "best_rebalance_freq" not in wfo_df.columns:
        # 兼容旧字段名
        if "best_freq_list" in wfo_df.columns and wfo_df["best_freq_list"].notna().any():
            # 使用众数列生成频率（保底）- 安全解析
            def _safe_mode(x):
                if pd.isna(x):
                    return 8
                try:
                    import ast
                    lst = ast.literal_eval(str(x)) if isinstance(x, str) else x
                    if isinstance(lst, (list, tuple)) and len(lst) > 0:
                        from collections import Counter
                        return Counter(lst).most_common(1)[0][0]
                except Exception:
                    pass
                return 8
            wfo_df["best_rebalance_freq"] = wfo_df["best_freq_list"].apply(_safe_mode)
        else:
            wfo_df["best_rebalance_freq"] = 8

    if "annual_ret" not in bt_df.columns:
        raise KeyError("回测CSV缺失列: annual_ret")

    print("合并数据...")
    merged = wfo_df.merge(bt_df[["combo", "annual_ret"]], on="combo", how="inner")
    print(f"✓ 可用训练样本: {len(merged)}")
    if len(merged) < 500:
        raise ValueError(f"可用训练样本过少: {len(merged)} (<500)")
    print()

    feature_names = [
        "mean_oos_ic",
        "oos_ic_std",
        "positive_rate",
        "stability_score",
        "combo_size",
        "best_rebalance_freq",
    ]
    X = merged[feature_names].copy()
    for c in X.columns:
        if X[c].isna().any():
            X[c].fillna(X[c].median(), inplace=True)
    y = merged["annual_ret"].values

    print("训练 GBDT 模型...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(Xs, y)
    print("✓ 模型训练完成")
    
    print("交叉验证评估...")
    cv_scores = cross_val_score(model, Xs, y, cv=5, scoring="r2", n_jobs=-1)
    print("✓ 交叉验证完成")
    print()
    metrics = {
        "n_samples": int(len(merged)),
        "r2_cv_mean": float(cv_scores.mean()),
        "r2_cv_std": float(cv_scores.std()),
        "run_ts": run_ts,
        "wfo_file": str(wfo_file),
        "backtest_file": str(backtest_file),
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "calibrator_gbdt_profit.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "train_history": [metrics],
        },
        save_path,
    )
    print("=" * 100)
    print("✅ 利润校准器训练完成")
    print(f"保存: {save_path}")
    print("评估:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("=" * 100)


if __name__ == "__main__":
    main()


