#!/usr/bin/env python3
"""
训练 LightGBM Lambdarank 校准器，支持多指标目标融合、严格留一验证与 Holdout 评估。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, KFold


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    sort_cols = ["run_ts"]
    for candidate in ("wfo_rank_by_calibrated", "rank_score", "mean_oos_ic"):
        if candidate in df.columns:
            sort_cols.append(candidate)
            break
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "is_significant" in df.columns:
        df["is_significant"] = df["is_significant"].astype(int)
    return df


def select_features(
    df: pd.DataFrame,
    target: str,
    group_chunk: int = 500,
    preset_features: Sequence[str] | None = None,
    return_groups: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], pd.Series | None]:
    work = df.copy()
    drop_cols = {
        "run_ts",
        "group_id",
        "within_run_rank",
        "combo",
        "oos_ic_list",
        "oos_ir_list",
        "positive_rate_list",
        "best_freq_list",
        "final_value",
        "final_value_net",
        "rank",
        "wfo_freq",
        "test_freq",
        "wfo_ic",
        "wfo_score",
        "vol",
        "sharpe",
        "max_dd",
        "n_rebalance",
        "avg_turnover",
        "avg_n_holdings",
        "avg_win",
        "avg_loss",
        "win_rate",
        "winning_days",
        "losing_days",
        "profit_factor",
        "sortino_ratio",
        "calmar_ratio",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "calibrated_annual_pred",
        "rank_score",
        "rank_lgbm",
    }
    targets = {
        target,
        "total_ret",
        "total_ret_net",
        "annual_ret",
        "annual_ret_net",
        "sharpe",
        "sharpe_net",
        "max_dd",
        "max_dd_net",
        "calmar_net",
        "return_vol_ratio",
        "ret_turnover_ratio",
        "annual_ret_net_z",
        "sharpe_net_z",
        "calmar_net_z",
        "annual_ret_net_pct",
        "sharpe_net_pct",
        "calmar_net_pct",
        "annual_ret_net_decile",
        "sharpe_net_decile",
        "calmar_net_decile",
        "target_weighted",
    }
    drop_cols |= targets
    if preset_features is not None:
        feature_cols = [c for c in preset_features if c not in drop_cols]
    else:
        feature_cols = [c for c in work.columns if c not in drop_cols]

    X = work[feature_cols].copy()
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    y = work[target].values.astype(float)

    groups: pd.Series | None = None
    if return_groups:
        if "within_run_rank" not in work.columns:
            work["within_run_rank"] = work.groupby("run_ts").cumcount()
        work["group_id"] = work["run_ts"].astype(str) + "_" + (work["within_run_rank"] // group_chunk).astype(int).astype(str)
        groups = work["group_id"]

    return X, y, feature_cols, groups


def build_group_counts(labels: Iterable[str]) -> List[int]:
    counts: List[int] = []
    last = None
    acc = 0
    for val in labels:
        if last is None:
            last = val
        if val == last:
            acc += 1
        else:
            counts.append(acc)
            acc = 1
            last = val
    if acc:
        counts.append(acc)
    return counts


def make_model() -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=600,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        n_jobs=-1,
        label_gain=list(range(32)),
    )


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, topks: Tuple[int, ...] = (100, 500, 1000)) -> Dict[str, float]:
    ndcg_k = min(len(y_true), 1000)
    gains = y_true
    min_val = gains.min()
    if min_val < 0:
        gains = gains - min_val
    ndcg_val = float(ndcg_score([gains], [y_pred], k=ndcg_k))
    spearman = float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
    kendall = float(pd.Series(y_pred).corr(pd.Series(y_true), method="kendall"))
    metrics: Dict[str, float] = {
        "ndcg@1000": ndcg_val,
        "spearman": spearman,
        "kendall": kendall,
    }
    for k in topks:
        k_eff = min(len(y_true), k)
        idx_true = set(np.argsort(-y_true)[:k_eff])
        idx_pred = set(np.argsort(-y_pred)[:k_eff])
        overlap = len(idx_true & idx_pred) / k_eff if k_eff > 0 else 0.0
        metrics[f"top{k}_overlap"] = float(overlap)
    return metrics


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    keys = records[0].keys()
    return {k: float(np.mean([rec[k] for rec in records])) for k in keys}


def to_relevance(y: np.ndarray, n_bins: int = 32) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(y, quantiles)
    bins = np.unique(bins)
    if bins.size <= 2:
        ranks = pd.Series(y).rank(method="dense").astype(int) - 1
        return np.maximum(ranks, 0)
    thresholds = bins[1:-1]
    labels = np.digitize(y, thresholds, right=False)
    return labels.astype(int)


def cross_validate(X: pd.DataFrame, y: np.ndarray, groups: pd.Series, feature_names: List[str]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    unique_groups = groups.unique()
    metrics_list: List[Dict[str, float]] = []
    fold_importances: Dict[str, List[float]] = {f: [] for f in feature_names}

    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        split_iter = splitter.split(X, y, groups=groups)
        use_group = True
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
        use_group = False

    for fold, (train_idx, valid_idx) in enumerate(split_iter, 1):
        train_idx = np.sort(train_idx)
        valid_idx = np.sort(valid_idx)
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]
        y_tr_rank = to_relevance(y_tr)
        y_val_rank = to_relevance(y_val)
        if use_group:
            tr_groups = build_group_counts(groups.iloc[train_idx])
            val_groups = build_group_counts(groups.iloc[valid_idx])
        else:
            tr_groups = [1] * len(train_idx)
            val_groups = [1] * len(valid_idx)
        model = make_model()
        model.fit(
            X_tr,
            y_tr_rank,
            group=tr_groups,
            eval_set=[(X_val, y_val_rank)],
            eval_group=[val_groups],
            eval_at=[100, 500, 1000],
            callbacks=[],
        )
        preds = model.predict(X_val)
        metrics = calc_metrics(y_val, preds)
        metrics_list.append(metrics)
        importance = model.booster_.feature_importance(importance_type="gain")
        for name, imp in zip(feature_names, importance):
            fold_importances[name].append(float(imp))
        print(
            f"[Fold {fold}] NDCG@1000={metrics['ndcg@1000']:.4f} | Spearman={metrics['spearman']:.4f} | "
            f"Top1000重叠={metrics['top1000_overlap']:.2%}"
        )

    avg_metrics = aggregate_metrics(metrics_list)
    return avg_metrics, fold_importances


def save_artifacts(
    model: LGBMRanker,
    feature_names: List[str],
    avg_importance: Dict[str, float],
    metrics: Dict[str, float],
    holdout_metrics: Dict[str, float] | None,
    settings: Dict[str, object],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "calibrator_ranker.txt"
    model.booster_.save_model(str(model_path))

    importance_path = output_dir / "calibrator_ranker_importance.json"
    with open(importance_path, "w") as f:
        json.dump(avg_importance, f, indent=2)

    metrics_payload: Dict[str, object] = {"cv": metrics, "settings": settings}
    if holdout_metrics is not None:
        metrics_payload["holdout"] = holdout_metrics

    metrics_path = output_dir / "calibrator_ranker_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"模型已保存: {model_path}")
    print(f"特征重要性: {importance_path}")
    print(f"CV/Holdout 指标: {metrics_path}")


def parse_float_list(text: str) -> List[float]:
    items = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not items:
        raise ValueError("权重列表不能为空")
    return items


def normalize_weights(weights: Sequence[float]) -> List[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("目标权重之和必须为正")
    return [w / total for w in weights]


def build_weighted_target(df: pd.DataFrame, components: Sequence[str], weights: Sequence[float]) -> pd.Series:
    if "run_ts" not in df.columns:
        raise ValueError("数据集缺少 run_ts 列")
    if len(components) != len(weights):
        raise ValueError("目标组件与权重长度不一致")

    comp_series: List[pd.Series] = []
    for col in components:
        if col not in df.columns:
            raise ValueError(f"目标列缺失: {col}")
        series = df[col]
        if series.isna().any():
            series = series.groupby(df["run_ts"]).transform(lambda x: x.fillna(x.median()))
            series = series.fillna(0.0)
        comp_series.append(series.astype(float))

    weighted = sum(w * s for w, s in zip(weights, comp_series))
    return weighted


def prepare_holdout_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    feats = df.reindex(columns=feature_cols, fill_value=0.0).copy()
    for col in feats.select_dtypes(include=["bool"]).columns:
        feats[col] = feats[col].astype(int)
    for col in feats.select_dtypes(include=["object"]).columns:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")
    if feats.isna().any().any():
        feats = feats.fillna(feats.median(numeric_only=True))
    return feats


def main():
    parser = argparse.ArgumentParser(description="训练 LightGBM 排序校准器")
    parser.add_argument("--dataset", type=str, default="data/calibrator_dataset.parquet", help="训练数据文件")
    parser.add_argument(
        "--target-components",
        type=str,
        default="annual_ret_net_z,sharpe_net_z,calmar_net_z",
        help="目标指标列（逗号分隔）",
    )
    parser.add_argument(
        "--target-weights",
        type=str,
        default="0.5,0.3,0.2",
        help="目标权重（逗号分隔，将自动归一化）",
    )
    parser.add_argument("--holdout-run", type=str, help="留作最终测试的 run_ts，默认取最新 run")
    parser.add_argument("--group-chunk", type=int, default=500, help="组合多少个样本划为同一 Lambdarank group")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = (repo_root / args.dataset).resolve()
    output_dir = repo_root / "results" / "models"

    print("=" * 100)
    print("LightGBM 排序校准器训练")
    print("=" * 100)
    print(f"数据集: {dataset_path}")

    df = load_dataset(dataset_path)
    components = [c.strip() for c in args.target_components.split(",") if c.strip()]
    weights = normalize_weights(parse_float_list(args.target_weights))
    target_series = build_weighted_target(df, components, weights)

    df = df.copy()
    df["target_weighted"] = target_series
    df = df.dropna(subset=["target_weighted"]).reset_index(drop=True)

    run_order = sorted(df["run_ts"].unique())
    holdout_run = args.holdout_run or run_order[-1]
    if holdout_run not in run_order:
        raise ValueError(f"holdout_run={holdout_run} 不存在于数据集中，可选值: {run_order}")

    train_df = df[df["run_ts"] != holdout_run].reset_index(drop=True)
    holdout_df = df[df["run_ts"] == holdout_run].reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError("训练集为空，请调整 holdout_run 设置")

    X_train, y_train, feature_cols, group_labels = select_features(
        train_df,
        target="target_weighted",
        group_chunk=args.group_chunk,
    )
    if group_labels is None:
        raise RuntimeError("训练集必须提供有效的 groups")

    print(f"训练样本数: {len(X_train)} | 特征数: {len(feature_cols)} | group 数: {len(group_labels.unique())}")
    print(f"Holdout run: {holdout_run} | Holdout 样本数: {len(holdout_df)}")

    print("-" * 100)
    print("执行交叉验证...")
    avg_metrics, fold_importances = cross_validate(X_train, y_train, group_labels, feature_cols)
    print("-" * 100)
    print("CV平均指标:")
    for k, v in avg_metrics.items():
        if "overlap" in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:.4f}")

    importance_mean = {feat: float(np.mean(vals)) for feat, vals in fold_importances.items()}
    sorted_imp = sorted(importance_mean.items(), key=lambda x: x[1], reverse=True)
    print("-" * 100)
    print("特征重要性 Top 15:")
    for idx, (feat, imp) in enumerate(sorted_imp[:15], 1):
        print(f"{idx:>2}. {feat:<30} {imp:>10.2f}")

    print("-" * 100)
    print("全量训练 LightGBM 排序模型 (仅训练集)...")
    final_model = make_model()
    train_group_counts = build_group_counts(group_labels)
    train_relevance = to_relevance(y_train)
    final_model.fit(X_train, train_relevance, group=train_group_counts, callbacks=[])

    holdout_metrics: Dict[str, float] | None = None
    if not holdout_df.empty:
        holdout_feats = prepare_holdout_features(holdout_df, feature_cols)
        y_holdout = holdout_df["target_weighted"].values.astype(float)
        preds = final_model.predict(holdout_feats)
        holdout_metrics = calc_metrics(y_holdout, preds)
        print("-" * 100)
        print("Holdout 指标:")
        for k, v in holdout_metrics.items():
            if "overlap" in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.4f}")

    settings = {
        "target_components": components,
        "target_weights": weights,
        "holdout_run": holdout_run,
        "train_runs": sorted(train_df["run_ts"].unique()),
        "group_chunk": args.group_chunk,
    }

    save_artifacts(final_model, feature_cols, importance_mean, avg_metrics, holdout_metrics, settings, output_dir)

    print("=" * 100)
    print("✅ 排序校准器训练完成")
    print("=" * 100)


if __name__ == "__main__":
    main()

