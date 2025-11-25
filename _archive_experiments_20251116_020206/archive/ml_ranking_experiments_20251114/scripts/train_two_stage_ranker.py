#!/usr/bin/env python3
"""训练两阶段排序校准器：Stage1 Sharpe 筛选 + Stage2 年化收益排序"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold, KFold

from build_rank_dataset import (  # type: ignore
    add_combo_composition_features,
    add_extreme_risk_features,
    add_market_regime_features,
    add_real_derived_features,
    enrich_wfo_features,
)
from train_calibrator_ranking import (  # type: ignore
    build_group_counts,
    calc_metrics,
    load_dataset,
    prepare_holdout_features,
    select_features,
    to_relevance,
)

# 训练时可选的全局特征白名单（由 --features-file 提供）
_FEATURES_WHITELIST: List[str] | None = None


def collect_inference_features(repo_root: Path) -> Set[str]:
    results_root = repo_root / "results"
    if not results_root.exists():
        return set()
    run_dirs = sorted(results_root.glob("run_*/all_combos.parquet"))
    for combos_path in reversed(run_dirs):
        try:
            combos = pd.read_parquet(combos_path)
        except Exception:
            continue
        enriched = enrich_wfo_features(combos)
        prices_template = repo_root / "data" / "etf_prices_template.csv"
        if prices_template.exists():
            try:
                enriched = add_market_regime_features(enriched, prices_template)
            except Exception:
                pass
        enriched = add_combo_composition_features(enriched)
        enriched = add_extreme_risk_features(enriched)
        enriched = add_real_derived_features(enriched)
        return set(enriched.columns)
    return set()


def make_stage1_model() -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=31,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l1=0.2,
        lambda_l2=0.2,
        random_state=42,
        n_jobs=-1,
        label_gain=list(range(32)),
    )


def make_stage2_model() -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=600,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        n_jobs=-1,
        label_gain=list(range(32)),
    )


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.mean([rec[k] for rec in records])) for k in keys}


def summarize_predictions(preds: np.ndarray) -> Dict[str, float]:
    if preds.size == 0:
        return {}
    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
        "p05": float(np.percentile(preds, 5)),
        "p50": float(np.percentile(preds, 50)),
        "p95": float(np.percentile(preds, 95)),
    }


def run_cv(
    model_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    feature_names: List[str],
    topks: Tuple[int, ...] = (100, 500, 1000),
) -> Tuple[Dict[str, float], Dict[str, float]]:
    unique_groups = groups.unique()
    metrics_list: List[Dict[str, float]] = []
    fold_importances: Dict[str, List[float]] = {f: [] for f in feature_names}

    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        split_iter = splitter.split(X, y, groups=groups)
        use_group = True
    else:
        splitter = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
        use_group = False

    for fold, (train_idx, valid_idx) in enumerate(split_iter, 1):
        train_idx = np.asarray(train_idx)
        valid_idx = np.asarray(valid_idx)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        y_train_rank = to_relevance(y_train)
        y_valid_rank = to_relevance(y_valid)
        if use_group:
            train_groups = build_group_counts(groups.iloc[train_idx])
            valid_groups = build_group_counts(groups.iloc[valid_idx])
        else:
            train_groups = [1] * len(train_idx)
            valid_groups = [1] * len(valid_idx)

        model = model_factory()
        model.fit(
            X_train,
            y_train_rank,
            group=train_groups,
            eval_set=[(X_valid, y_valid_rank)],
            eval_group=[valid_groups],
            eval_at=list(topks),
            callbacks=[],
        )
        preds = model.predict(X_valid)
        metrics = calc_metrics(y_valid, preds, topks=topks)
        metrics_list.append(metrics)
        importance = model.booster_.feature_importance(importance_type="gain")
        for name, val in zip(feature_names, importance):
            fold_importances[name].append(float(val))

    avg_metrics = aggregate_metrics(metrics_list)
    avg_importance = {feat: float(np.mean(vals)) for feat, vals in fold_importances.items()}
    return avg_metrics, avg_importance


def train_stage(
    df: pd.DataFrame,
    target_col: str,
    model_factory,
    group_chunk: int,
    topks: Tuple[int, ...],
    allowed_features: Optional[Set[str]] = None,
) -> Tuple[LGBMRanker, List[str], Dict[str, float], Dict[str, float]]:
    X, y, feature_cols, groups = select_features(df, target=target_col, group_chunk=group_chunk, preset_features=_FEATURES_WHITELIST if _FEATURES_WHITELIST else None)
    if allowed_features:
        kept_features = [feat for feat in feature_cols if feat in allowed_features]
        if not kept_features:
            raise RuntimeError("允许特征集合与训练特征不相交，无法训练模型")
        if len(kept_features) != len(feature_cols):
            X = X[kept_features]
            feature_cols = kept_features
    if groups is None:
        raise RuntimeError("group 信息缺失")
    cv_metrics, cv_importance = run_cv(model_factory, X, y, groups, feature_cols, topks)
    model = model_factory()
    group_counts = build_group_counts(groups)
    relevance = to_relevance(y)
    model.fit(X, relevance, group=group_counts, callbacks=[])
    importance = model.booster_.feature_importance(importance_type="gain")
    feature_importance = {feat: float(val) for feat, val in zip(feature_cols, importance)}
    return model, feature_cols, {**cv_metrics}, feature_importance


def evaluate_holdout(
    model: LGBMRanker,
    holdout_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    topks: Tuple[int, ...],
) -> Dict[str, float]:
    if holdout_df.empty:
        return {}
    feats = prepare_holdout_features(holdout_df, feature_cols)
    target = holdout_df[target_col].values.astype(float)
    preds = model.predict(feats)
    metrics = calc_metrics(target, preds, topks=topks)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="训练两阶段排序校准器")
    parser.add_argument("--dataset", type=str, default="data/calibrator_dataset.parquet", help="训练数据")
    parser.add_argument("--output-dir", type=str, default="results/models", help="模型输出目录")
    parser.add_argument("--sharpe-threshold", type=float, default=0.0, help="Sharpe 筛选阈值（z-score）")
    parser.add_argument("--disable-sharpe-gate", action="store_true", help="关闭 Sharpe 门控（Stage2 不再按阈值过滤样本）")
    parser.add_argument("--features-file", type=str, help="仅使用该文件中列出的特征进行训练（每行一个列名）")
    parser.add_argument("--holdout-run", type=str, help="留作评估的 run_ts")
    parser.add_argument("--group-chunk", type=int, default=500, help="每个 Lambdarank group 的样本数")
    parser.add_argument("--topks", type=str, default="100,500,1000", help="评估用的 TopK 阈值")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = (repo_root / args.dataset).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 全局特征白名单（供 select_features 使用）
    global _FEATURES_WHITELIST  # type: ignore
    _FEATURES_WHITELIST = None
    if args.features_file:
        feat_path = (repo_root / args.features_file).resolve() if not Path(args.features_file).is_absolute() else Path(args.features_file)
        if not feat_path.exists():
            raise FileNotFoundError(f"features-file 不存在: {feat_path}")
        _FEATURES_WHITELIST = [ln.strip() for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        # 记录一份到输出目录，便于复现
        (output_dir / "features_used.txt").write_text("\n".join(_FEATURES_WHITELIST), encoding="utf-8")

    df = load_dataset(dataset_path)
    df = df.dropna(subset=["sharpe_net_z", "annual_ret_net"])
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError("训练数据为空，请先构建数据集")

    inference_feature_set = collect_inference_features(repo_root)
    if not inference_feature_set:
        print("[WARN] 未能从最新 all_combos 推断推理特征集合，训练时将使用全部特征。")

    run_order = sorted(df["run_ts"].unique())
    holdout_run: Optional[str]
    if args.holdout_run:
        holdout_run = args.holdout_run
        if holdout_run not in run_order:
            raise ValueError(f"holdout_run={holdout_run} 不存在，可选: {run_order}")
    else:
        holdout_run = run_order[-1] if len(run_order) > 1 else None

    train_df = df[df["run_ts"] != holdout_run] if holdout_run else df.copy()
    holdout_df = df[df["run_ts"] == holdout_run] if holdout_run else df.iloc[0:0]
    if train_df.empty:
        raise RuntimeError("训练集为空，无法训练模型")

    topks = tuple(int(x.strip()) for x in args.topks.split(",") if x.strip()) or (100, 500, 1000)

    print("=" * 100)
    print("Stage 1: Sharpe 筛选模型")
    print("=" * 100)
    stage1_model, stage1_features, stage1_cv_metrics, stage1_importance = train_stage(
        train_df,
        target_col="sharpe_net_z",
        model_factory=make_stage1_model,
        group_chunk=args.group_chunk,
        topks=topks,
        allowed_features=inference_feature_set if inference_feature_set else None,
    )
    stage1_train_feats = prepare_holdout_features(train_df, stage1_features)
    stage1_train_preds = stage1_model.predict(stage1_train_feats)
    stage1_pred_stats = summarize_predictions(stage1_train_preds)
    print("Stage1 CV 指标:")
    for k, v in stage1_cv_metrics.items():
        print(f"  {k}: {v:.4f}" if "overlap" not in k else f"  {k}: {v:.2%}")

    stage1_holdout_metrics = evaluate_holdout(stage1_model, holdout_df, stage1_features, "sharpe_net_z", topks)
    if stage1_holdout_metrics:
        print("Stage1 Holdout 指标:")
        for k, v in stage1_holdout_metrics.items():
            print(f"  {k}: {v:.4f}" if "overlap" not in k else f"  {k}: {v:.2%}")

    threshold = args.sharpe_threshold
    if args.disable_sharpe_gate:
        threshold = -1e9
    filtered_train = train_df[train_df["sharpe_net_z"] >= threshold].reset_index(drop=True)
    if filtered_train.empty:
        raise RuntimeError("Sharpe 阈值过高，训练样本被筛空")

    filtered_holdout = holdout_df.copy()
    if not filtered_holdout.empty:
        holdout_feats = prepare_holdout_features(filtered_holdout, stage1_features)
        holdout_preds = stage1_model.predict(holdout_feats)
        filtered_holdout = filtered_holdout[holdout_preds >= threshold].reset_index(drop=True)

    print("=" * 100)
    print("Stage 2: 年化收益排序模型")
    print("=" * 100)
    stage2_model, stage2_features, stage2_cv_metrics, stage2_importance = train_stage(
        filtered_train,
        target_col="annual_ret_net",
        model_factory=make_stage2_model,
        group_chunk=args.group_chunk,
        topks=topks,
        allowed_features=inference_feature_set if inference_feature_set else None,
    )
    stage2_train_feats = prepare_holdout_features(filtered_train, stage2_features)
    stage2_train_preds = stage2_model.predict(stage2_train_feats)
    stage2_pred_stats = summarize_predictions(stage2_train_preds)
    print("Stage2 CV 指标:")
    for k, v in stage2_cv_metrics.items():
        print(f"  {k}: {v:.4f}" if "overlap" not in k else f"  {k}: {v:.2%}")

    stage2_holdout_metrics = evaluate_holdout(stage2_model, filtered_holdout, stage2_features, "annual_ret_net", topks)
    if stage2_holdout_metrics:
        print("Stage2 Holdout 指标:")
        for k, v in stage2_holdout_metrics.items():
            print(f"  {k}: {v:.4f}" if "overlap" not in k else f"  {k}: {v:.2%}")

    # 保存模型
    stage1_path = output_dir / "calibrator_sharpe_filter.txt"
    stage1_model.booster_.save_model(str(stage1_path))
    with open(output_dir / "calibrator_sharpe_filter_importance.json", "w") as f:
        json.dump(stage1_importance, f, indent=2)
    with open(output_dir / "calibrator_sharpe_filter_metrics.json", "w") as f:
        json.dump({"cv": stage1_cv_metrics, "holdout": stage1_holdout_metrics, "features": stage1_features}, f, indent=2)
    with open(output_dir / "calibrator_sharpe_filter_scaler.json", "w") as f:
        json.dump(stage1_pred_stats, f, indent=2)

    stage2_path = output_dir / "calibrator_profit_ranker.txt"
    stage2_model.booster_.save_model(str(stage2_path))
    with open(output_dir / "calibrator_profit_ranker_importance.json", "w") as f:
        json.dump(stage2_importance, f, indent=2)
    with open(output_dir / "calibrator_profit_ranker_metrics.json", "w") as f:
        json.dump({"cv": stage2_cv_metrics, "holdout": stage2_holdout_metrics, "features": stage2_features}, f, indent=2)
    with open(output_dir / "calibrator_profit_ranker_scaler.json", "w") as f:
        json.dump(stage2_pred_stats, f, indent=2)

    report = {
        "dataset": str(dataset_path),
        "runs": run_order,
        "holdout_run": holdout_run,
        "sharpe_threshold": threshold,
        "inference_feature_count": int(len(inference_feature_set)) if inference_feature_set else 0,
        "inference_features": sorted(inference_feature_set) if inference_feature_set else [],
        "stage1": {
            "model_path": str(stage1_path),
            "cv_metrics": stage1_cv_metrics,
            "holdout_metrics": stage1_holdout_metrics,
            "n_train": int(len(train_df)),
            "features": stage1_features,
            "pred_stats": stage1_pred_stats,
        },
        "stage2": {
            "model_path": str(stage2_path),
            "cv_metrics": stage2_cv_metrics,
            "holdout_metrics": stage2_holdout_metrics,
            "n_train": int(len(filtered_train)),
            "features": stage2_features,
            "pred_stats": stage2_pred_stats,
        },
    }
    with open(output_dir / "two_stage_training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 100)
    print("✅ 两阶段模型训练完成")
    print(f"Stage1 模型: {stage1_path}")
    print(f"Stage2 模型: {stage2_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
