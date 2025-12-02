#!/usr/bin/env python3
"""
Build baseline and calibrated ranking Parquet files for a given WFO run,
so that the gating script can compare TopK metrics.

Outputs under etf_rotation_experiments/results/run_*/ranking_blends/:
  - ranking_baseline.parquet (rank by mean_oos_ic)
  - ranking_lightgbm.parquet (rank by calibrator predicted sharpe)

It also writes results/calibrator_feature_importance.csv if missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def find_latest_run(results_root: Path) -> Path:
    runs = sorted([p for p in results_root.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run_* under {results_root}")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate baseline/calibrated ranking files for gating")
    ap.add_argument("--run-ts", type=str, default=None, help="Target run_ts; default to latest run_*")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    exp_root = repo_root / "etf_rotation_experiments"
    results_root = exp_root / "results"
    run_dir = results_root / f"run_{args.run_ts}" if args.run_ts else find_latest_run(results_root)
    all_path = run_dir / "all_combos.parquet"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing all_combos.parquet: {all_path}")

    # Load WFO results
    df = pd.read_parquet(all_path)
    if "combo" not in df.columns:
        raise KeyError("Input all_combos.parquet must contain 'combo' column")

    # Prepare output dir
    blend_dir = run_dir / "ranking_blends"
    blend_dir.mkdir(parents=True, exist_ok=True)

    # Baseline ranking (mean_oos_ic fallback if calibrated_sharpe_pred not present)
    base_col = "calibrated_sharpe_pred" if "calibrated_sharpe_pred" in df.columns else "mean_oos_ic"
    base_sorted = df.sort_values(base_col, ascending=False).reset_index(drop=True).copy()
    base_sorted["rank_score"] = base_sorted[base_col].astype(float)
    (blend_dir / "ranking_baseline.parquet").write_bytes(base_sorted.to_parquet(index=False))

    # Calibrated ranking via GBDT calibrator
    sys.path.insert(0, str((exp_root / "core").resolve().parent))
    from etf_rotation_experiments.core.wfo_realbt_calibrator import WFORealBacktestCalibrator  # type: ignore

    # Locate calibrator model (try multiple common locations)
    candidates = [
        results_root / "calibrator_gbdt_full.joblib",
        repo_root / "results" / "calibrator_gbdt_full.joblib",
        repo_root / "etf_strategy" / "results" / "calibrator_gbdt_full.joblib",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Calibrator model not found in: " + ", ".join(str(p) for p in candidates))

    calibrator = WFORealBacktestCalibrator.load(model_path)
    # Ensure feature medians present for prediction
    try:
        preds = calibrator.predict(df)
    except RuntimeError as e:
        if "特征填充统计量" in str(e) or "feature" in str(e).lower():
            # Build medians from current dataset
            X_raw = calibrator.extract_features(df)
            # Align to model feature order if available
            if getattr(calibrator, "feature_names", None):
                for col in calibrator.feature_names:
                    if col not in X_raw.columns:
                        X_raw[col] = 0.0
                X_raw = X_raw[calibrator.feature_names]
            med = X_raw.median(numeric_only=True).to_dict()
            calibrator.feature_medians = {k: float(v) for k, v in med.items()}
            preds = calibrator.predict(df)
        else:
            raise
    cal_sorted = df.copy()
    cal_sorted["calibrated_sharpe_pred"] = preds.astype(float)
    cal_sorted["rank_score"] = cal_sorted["calibrated_sharpe_pred"].astype(float)
    cal_sorted = cal_sorted.sort_values("rank_score", ascending=False).reset_index(drop=True)
    (blend_dir / "ranking_lightgbm.parquet").write_bytes(cal_sorted.to_parquet(index=False))

    # Export feature importance if missing
    imp_csv = results_root / "calibrator_feature_importance.csv"
    try:
        if not imp_csv.exists():
            importance = calibrator.analyze_feature_importance()
            importance.to_csv(imp_csv, index=False)
    except Exception:
        pass

    print(f"Done. Wrote:\n  - {blend_dir / 'ranking_baseline.parquet'}\n  - {blend_dir / 'ranking_lightgbm.parquet'}")


if __name__ == "__main__":
    main()
