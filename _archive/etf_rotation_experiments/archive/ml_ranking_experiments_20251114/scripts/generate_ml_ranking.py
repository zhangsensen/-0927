#!/usr/bin/env python3
"""Generate ML ranking predictions for a specific WFO run directory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_ranking import inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and apply the ML ranker to a run_* directory.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a specific run_* directory. Defaults to the latest run under --results-root.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Root directory that contains run_* folders (default: experiments/results).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "ml_ranking/data/training_dataset.parquet"),
        help="Training dataset used to fit the inference model.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="oos_compound_sharpe",
        help="Label/target column inside the training dataset.",
    )
    parser.add_argument(
        "--preferred-model",
        type=str,
        default=None,
        help="Optionally pin the model family (default: auto select by Spearman).",
    )
    parser.add_argument(
        "--combos-file",
        type=str,
        default=None,
        help="Optional path to an alternate combos parquet (default: run_dir/all_combos.parquet).",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="calibrated_sharpe_pred_ml",
        help="Name of the score column to write into the ranking file.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Optional cap on the number of rows to persist in the output ranking.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output parquet path. Default: <run_dir>/ranking_blends/ranking_ml_supervised.parquet",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata JSON path. Default: <output>.metadata.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    run_dir = _resolve_run_dir(args.run_dir, Path(args.results_root).expanduser().resolve())
    combos_path = Path(args.combos_file).expanduser().resolve() if args.combos_file else run_dir / "all_combos.parquet"
    if not combos_path.exists():
        raise FileNotFoundError(f"Ranking source file not found: {combos_path}")

    print("=" * 80)
    print("ML RANKING - GENERATION PIPELINE")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Run dir: {run_dir}")
    print(f"Combos:  {combos_path}")

    dataset = pd.read_parquet(dataset_path)
    artifacts = inference.train_inference_model(
        dataset,
        label_column=args.label_column,
        preferred_model=args.preferred_model,
    )
    print(f"Model selected: {artifacts.model_name}")
    print(f"Training Spearman: {artifacts.evaluation_metrics.get('spearman', float('nan')):.4f}")

    combos = pd.read_parquet(combos_path).reset_index(drop=True)
    features = inference.prepare_feature_frame(combos)
    scores = inference.score_features(features, artifacts)

    ranking = combos.copy()
    ranking[args.score_column] = scores
    ranking = ranking.sort_values(by=args.score_column, ascending=False).reset_index(drop=True)
    ranking["ml_rank"] = np.arange(1, len(ranking) + 1)
    if args.topk is not None:
        ranking = ranking.head(args.topk).copy()

    output_path = _resolve_output_path(run_dir, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(output_path, index=False)

    metadata_path = _resolve_metadata_path(output_path, args.metadata)
    metadata = _build_metadata(artifacts, len(ranking), args.score_column, dataset_path, combos_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved ranking: {output_path}")
    print(f"Saved metadata: {metadata_path}")


def _resolve_run_dir(explicit: Optional[str], results_root: Path) -> Path:
    if explicit:
        run_dir = Path(explicit).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        return run_dir
    run_dirs: List[Path] = [p.resolve() for p in results_root.glob("run_*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under {results_root}")
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    return run_dirs[0]


def _resolve_output_path(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (run_dir / "ranking_blends" / "ranking_ml_supervised.parquet").resolve()


def _resolve_metadata_path(output_path: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _build_metadata(
    artifacts: inference.InferenceArtifacts,
    n_rows: int,
    score_column: str,
    dataset_path: Path,
    combos_path: Path,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": artifacts.model_name,
        "label_column": artifacts.label_column,
        "score_column": score_column,
        "n_rows": n_rows,
        "dataset": str(dataset_path),
        "combos_source": str(combos_path),
        "evaluation_metrics": artifacts.evaluation_metrics,
        "feature_columns": artifacts.feature_columns,
    }


if __name__ == "__main__":
    main()
