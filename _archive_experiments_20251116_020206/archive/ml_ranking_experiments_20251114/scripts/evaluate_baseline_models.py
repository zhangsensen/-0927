"""Comprehensive baseline evaluation for supervised ranking models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from etf_rotation_experiments.ml_ranking import evaluation, models, training


@dataclass
class BaselineReport:
    """Complete baseline evaluation results."""

    dataset_info: Dict[str, Any]
    train_test_split_info: Dict[str, Any]
    model_results: List[Dict[str, Any]]
    cross_validation_results: Dict[str, Any]
    data_quality_checks: Dict[str, Any]


def load_and_inspect_dataset(dataset_path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataset and generate quality metrics."""

    df = pd.read_parquet(dataset_path)
    label_col = "oos_compound_sharpe"

    info = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "feature_count": len(df.columns) - 1,
        "label_column": label_col,
        "missing_rate_by_column": df.isnull().mean().to_dict(),
        "overall_missing_rate": float(df.isnull().mean().mean()),
        "label_stats": df[label_col].describe().to_dict(),
        "label_coverage": float((~df[label_col].isnull()).mean()),
        "feature_columns": [col for col in df.columns if col != label_col],
    }
    return df, info


def check_data_quality(df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
    """Perform data quality and leakage checks."""

    checks = {
        "label_in_features": label_col in [col for col in df.columns if col != label_col],
        "duplicate_rows": int(df.duplicated().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum()),
        "label_variance": float(df[label_col].var()),
        "label_range": [float(df[label_col].min()), float(df[label_col].max())],
        "high_missing_features": [
            col
            for col in df.columns
            if col != label_col and df[col].isnull().mean() > 0.05
        ],
    }
    return checks


def train_with_single_split(
    df: pd.DataFrame, label_col: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[List[training.TrainingResult], Dict[str, Any]]:
    """Train models with a single train/test split."""

    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_ratio": test_size,
        "random_state": random_state,
    }

    # Apply same preprocessing as training module
    X_train_clean = X_train.apply(pd.to_numeric, errors="coerce")
    X_test_clean = X_test.apply(pd.to_numeric, errors="coerce")
    median_vals = X_train_clean.median()
    X_train_clean = X_train_clean.fillna(median_vals).fillna(0.0)
    X_test_clean = X_test_clean.fillna(median_vals).fillna(0.0)

    # Train models
    results: List[training.TrainingResult] = []
    model_registry = models.baseline_model_registry()

    for name, model in model_registry.items():
        try:
            model.fit(X_train_clean, y_train)
            y_pred = model.predict(X_test_clean)
            eval_result = evaluation.evaluate_predictions(
                y_true=y_test, y_pred=y_pred, model_name=name
            )
            results.append(
                training.TrainingResult(model_name=name, model=model, evaluation=eval_result)
            )
        except Exception as e:
            print(f"Warning: Model {name} failed with error: {e}")
            continue

    return results, split_info


def cross_validate_models(
    df: pd.DataFrame, label_col: str, n_folds: int = 5, random_state: int = 42
) -> Dict[str, Any]:
    """Perform k-fold cross-validation."""

    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy(dtype=float)

    # Preprocessing
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    median_vals = X_numeric.median()
    X_clean = X_numeric.fillna(median_vals).fillna(0.0)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    model_registry = models.baseline_model_registry()

    cv_results: Dict[str, Any] = {}

    for name, model_template in model_registry.items():
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_clean)):
            X_train_fold = X_clean.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_clean.iloc[val_idx]
            y_val_fold = y[val_idx]

            # Create fresh model instance
            if name == "elasticnet":
                fold_model = models.make_linear_model(models.ModelConfig())
            elif name == "decision_tree":
                fold_model = models.make_tree_model(models.ModelConfig())
            elif name == "lgbm_regressor":
                fold_model = models.make_lgbm_regressor(models.ModelConfig())
            else:
                continue

            try:
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = fold_model.predict(X_val_fold)
                eval_result = evaluation.evaluate_predictions(
                    y_true=y_val_fold, y_pred=y_pred_fold, model_name=f"{name}_fold{fold_idx}"
                )
                fold_metrics.append(eval_result.metrics)
            except Exception as e:
                print(f"Warning: {name} fold {fold_idx} failed: {e}")
                continue

        if fold_metrics:
            # Aggregate metrics across folds
            aggregated = {}
            for metric_name in fold_metrics[0].keys():
                values = [m[metric_name] for m in fold_metrics if metric_name in m]
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))

            cv_results[name] = {
                "n_folds": len(fold_metrics),
                "fold_metrics": fold_metrics,
                "aggregated": aggregated,
            }

    return cv_results


def analyze_top_k_predictions(
    df: pd.DataFrame,
    trained_results: List[training.TrainingResult],
    label_col: str,
    top_ks: List[int] = [50, 100, 200, 500, 1000, 2000],
) -> Dict[str, Any]:
    """Analyze actual Sharpe distribution in predicted top-k buckets."""

    X = df.drop(columns=[label_col])
    y = df[label_col].to_numpy(dtype=float)

    # Preprocess
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    median_vals = X_numeric.median()
    X_clean = X_numeric.fillna(median_vals).fillna(0.0)

    analysis: Dict[str, Any] = {}

    for result in trained_results:
        model_name = result.model_name
        try:
            y_pred = result.model.predict(X_clean)
        except Exception:
            continue

        # Rank by prediction
        ranked_indices = np.argsort(-y_pred)

        model_analysis = {}
        for k in top_ks:
            if k > len(y):
                continue
            top_k_indices = ranked_indices[:k]
            top_k_actual = y[top_k_indices]

            model_analysis[f"top{k}"] = {
                "mean_actual_sharpe": float(np.mean(top_k_actual)),
                "median_actual_sharpe": float(np.median(top_k_actual)),
                "std_actual_sharpe": float(np.std(top_k_actual)),
                "min_actual_sharpe": float(np.min(top_k_actual)),
                "max_actual_sharpe": float(np.max(top_k_actual)),
                "count": int(k),
            }

        # Compare to oracle (true top-k)
        true_top_indices = np.argsort(-y)
        for k in top_ks:
            if k > len(y):
                continue
            oracle_top_k = y[true_top_indices[:k]]
            model_analysis[f"top{k}"]["oracle_mean"] = float(np.mean(oracle_top_k))
            model_analysis[f"top{k}"]["oracle_median"] = float(np.median(oracle_top_k))

        analysis[model_name] = model_analysis

    return analysis


def generate_baseline_report(
    dataset_path: Path, output_path: Path, n_cv_folds: int = 5
) -> BaselineReport:
    """Generate complete baseline evaluation report."""

    print("Loading dataset...")
    df, dataset_info = load_and_inspect_dataset(dataset_path)
    label_col = dataset_info["label_column"]

    print("Checking data quality...")
    quality_checks = check_data_quality(df, label_col)

    print("Training with single split...")
    trained_results, split_info = train_with_single_split(df, label_col)

    print("Running cross-validation...")
    cv_results = cross_validate_models(df, label_col, n_folds=n_cv_folds)

    print("Analyzing top-k predictions...")
    topk_analysis = analyze_top_k_predictions(df, trained_results, label_col)

    # Compile model results
    model_results = []
    for result in trained_results:
        model_results.append(
            {
                "model_name": result.model_name,
                "test_metrics": result.evaluation.metrics,
            }
        )

    report = BaselineReport(
        dataset_info=dataset_info,
        train_test_split_info=split_info,
        model_results=model_results,
        cross_validation_results=cv_results,
        data_quality_checks=quality_checks,
    )

    # Save report
    output_dict = {
        "dataset_info": report.dataset_info,
        "train_test_split_info": report.train_test_split_info,
        "model_results": report.model_results,
        "cross_validation_results": report.cross_validation_results,
        "data_quality_checks": report.data_quality_checks,
        "top_k_analysis": topk_analysis,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)

    print(f"✅ Report saved to: {output_path}")
    return report


def print_summary(report_path: Path) -> None:
    """Print human-readable summary of the report."""

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    print("\n" + "=" * 80)
    print("📊 BASELINE EVALUATION SUMMARY")
    print("=" * 80)

    # Dataset info
    ds = report["dataset_info"]
    print(f"\n📁 Dataset: {ds['total_rows']} rows × {ds['feature_count']} features")
    print(f"   Missing rate: {ds['overall_missing_rate']*100:.2f}%")
    print(f"   Label coverage: {ds['label_coverage']*100:.1f}%")
    print(f"   Label range: [{ds['label_stats']['min']:.3f}, {ds['label_stats']['max']:.3f}]")

    # Data quality
    dq = report["data_quality_checks"]
    print(f"\n🔍 Quality Checks:")
    print(f"   Label leakage: {'❌ FAIL' if dq['label_in_features'] else '✅ PASS'}")
    print(f"   Duplicates: {dq['duplicate_rows']}")
    print(f"   High-missing features: {len(dq['high_missing_features'])}")

    # Model results
    print(f"\n🤖 Model Performance (Single Split):")
    for model in report["model_results"]:
        name = model["model_name"]
        metrics = model["test_metrics"]
        print(f"\n   {name}:")
        print(f"      Spearman: {metrics.get('spearman', np.nan):.4f}")
        print(f"      Top50 overlap: {metrics.get('top50_overlap', np.nan):.2%}")
        print(f"      NDCG@50: {metrics.get('ndcg@50', np.nan):.4f}")

    # Cross-validation
    print(f"\n📊 Cross-Validation Results ({report['cross_validation_results'].keys()}):")
    for name, cv_res in report["cross_validation_results"].items():
        agg = cv_res["aggregated"]
        print(f"\n   {name} ({cv_res['n_folds']} folds):")
        print(
            f"      Spearman: {agg.get('spearman_mean', np.nan):.4f} ± {agg.get('spearman_std', np.nan):.4f}"
        )
        print(
            f"      Top50 overlap: {agg.get('top50_overlap_mean', np.nan):.2%} ± {agg.get('top50_overlap_std', np.nan):.2%}"
        )

    # Top-k analysis
    if "top_k_analysis" in report:
        print(f"\n🎯 Top-K Prediction Quality:")
        for model_name, analysis in report["top_k_analysis"].items():
            print(f"\n   {model_name}:")
            for k_name in ["top50", "top100", "top2000"]:
                if k_name in analysis:
                    stats = analysis[k_name]
                    print(
                        f"      {k_name}: mean={stats['mean_actual_sharpe']:.3f}, "
                        f"oracle={stats.get('oracle_mean', 0):.3f}, "
                        f"gap={stats['mean_actual_sharpe'] - stats.get('oracle_mean', 0):.3f}"
                    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "etf_rotation_experiments/ml_ranking/data/training_dataset.parquet"
        ),
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("etf_rotation_experiments/ml_ranking/reports/baseline_evaluation.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of cross-validation folds"
    )

    args = parser.parse_args()

    report = generate_baseline_report(args.dataset, args.output, n_cv_folds=args.cv_folds)
    print_summary(args.output)
