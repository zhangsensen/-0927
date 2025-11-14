#!/usr/bin/env python3
"""
P0 Baseline Evaluation: Generate comprehensive report with CV and Top-K analysis.

Usage:
    python run_baseline_evaluation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_ranking import evaluation, models


def main():
    print("=" * 80)
    print("P0 BASELINE EVALUATION - SUPERVISED RANKING")
    print("=" * 80)

    # Paths
    dataset_path = Path(__file__).parent.parent / "ml_ranking/data/training_dataset.parquet"
    output_path = Path(__file__).parent.parent / "ml_ranking/reports/baseline_evaluation.json"

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    df = pd.read_parquet(dataset_path)
    label_col = "oos_compound_sharpe"

    print(f"   Rows: {len(df)}")
    print(f"   Features: {len(df.columns) - 1}")
    print(f"   Label: {label_col}")

    # 2. Data quality checks
    print("\n[2/5] Data quality checks...")
    quality = {
        "total_rows": len(df),
        "total_features": len(df.columns) - 1,
        "label_in_features": label_col in [c for c in df.columns if c != label_col],
        "duplicates": int(df.duplicated().sum()),
        "label_coverage": float((~df[label_col].isnull()).mean()),
        "overall_missing_rate": float(df.isnull().mean().mean()),
        "high_missing_cols": [
            c for c in df.columns if c != label_col and df[c].isnull().mean() > 0.05
        ],
    }

    print(f"   Label leakage: {'❌ DETECTED' if quality['label_in_features'] else '✅ None'}")
    print(f"   Duplicates: {quality['duplicates']}")
    print(f"   Missing rate: {quality['overall_missing_rate']*100:.2f}%")

    # 3. Single train/test split
    print("\n[3/5] Single train/test split (80/20)...")
    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing
    X_train_clean = X_train.apply(pd.to_numeric, errors="coerce")
    X_test_clean = X_test.apply(pd.to_numeric, errors="coerce")
    median_vals = X_train_clean.median()
    X_train_clean = X_train_clean.fillna(median_vals).fillna(0.0)
    X_test_clean = X_test_clean.fillna(median_vals).fillna(0.0)

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # Train models
    model_registry = models.baseline_model_registry()
    single_split_results = {}

    for name, model in model_registry.items():
        print(f"   Training {name}...", end=" ")
        try:
            model.fit(X_train_clean, y_train)
            y_pred = model.predict(X_test_clean)
            eval_result = evaluation.evaluate_predictions(
                y_true=y_test, y_pred=y_pred, model_name=name
            )
            single_split_results[name] = eval_result.metrics
            print(
                f"Spearman={eval_result.metrics['spearman']:.4f}, "
                f"Top50={eval_result.metrics['top50_overlap']:.2%}"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # 4. 5-fold cross-validation
    print("\n[4/5] 5-fold cross-validation...")
    X_all = X.apply(pd.to_numeric, errors="coerce")
    median_all = X_all.median()
    X_all_clean = X_all.fillna(median_all).fillna(0.0)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name in model_registry.keys():
        fold_metrics = []
        print(f"   CV {name}...", end=" ")

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_all_clean)):
            X_tr = X_all_clean.iloc[train_idx]
            y_tr = y[train_idx]
            X_val = X_all_clean.iloc[val_idx]
            y_val = y[val_idx]

            # Fresh model
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
            print(
                f"Spearman={aggregated['spearman_mean']:.4f}±{aggregated['spearman_std']:.4f}"
            )

    # 5. Top-K analysis
    print("\n[5/5] Top-K prediction analysis...")
    top_ks = [50, 100, 200, 500, 1000, 2000]
    topk_analysis = {}

    for name, model in model_registry.items():
        if name not in single_split_results:
            continue

        try:
            # Retrain on full data for prediction
            model.fit(X_all_clean, y)
            y_pred_all = model.predict(X_all_clean)

            ranked_idx = np.argsort(-y_pred_all)
            true_ranked_idx = np.argsort(-y)

            model_topk = {}
            for k in top_ks:
                if k > len(y):
                    continue

                pred_topk_idx = ranked_idx[:k]
                pred_topk_actual = y[pred_topk_idx]

                true_topk_idx = true_ranked_idx[:k]
                oracle_actual = y[true_topk_idx]

                model_topk[f"top{k}"] = {
                    "mean_actual": float(np.mean(pred_topk_actual)),
                    "median_actual": float(np.median(pred_topk_actual)),
                    "std_actual": float(np.std(pred_topk_actual)),
                    "oracle_mean": float(np.mean(oracle_actual)),
                    "gap": float(np.mean(pred_topk_actual) - np.mean(oracle_actual)),
                }

            topk_analysis[name] = model_topk
            print(f"   {name}: Top50 gap={model_topk['top50']['gap']:.4f}")

        except Exception as e:
            print(f"   {name}: FAILED - {e}")

    # Save report
    report = {
        "dataset_info": {
            "total_rows": len(df),
            "total_features": len(df.columns) - 1,
            "label_column": label_col,
            "label_stats": df[label_col].describe().to_dict(),
        },
        "data_quality_checks": quality,
        "train_test_split": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_ratio": 0.2,
        },
        "single_split_results": single_split_results,
        "cross_validation_results": cv_results,
        "top_k_analysis": topk_analysis,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n📊 Single Split Performance:")
    for name, metrics in single_split_results.items():
        print(
            f"   {name:15s}: Spearman={metrics['spearman']:.4f}, "
            f"Top50={metrics['top50_overlap']:.2%}, "
            f"NDCG@50={metrics['ndcg@50']:.4f}"
        )

    print("\n📊 Cross-Validation (5-fold):")
    for name, cv in cv_results.items():
        agg = cv["aggregated"]
        print(
            f"   {name:15s}: Spearman={agg['spearman_mean']:.4f}±{agg['spearman_std']:.4f}, "
            f"Top50={agg['top50_overlap_mean']:.2%}±{agg['top50_overlap_std']:.2%}"
        )

    print("\n🎯 Top-K Quality (vs Oracle):")
    for name, topk in topk_analysis.items():
        print(f"   {name}:")
        for k_name in ["top50", "top100", "top2000"]:
            if k_name in topk:
                stats = topk[k_name]
                print(
                    f"      {k_name:8s}: pred_mean={stats['mean_actual']:.3f}, "
                    f"oracle={stats['oracle_mean']:.3f}, gap={stats['gap']:.3f}"
                )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
