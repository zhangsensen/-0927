"""High-level orchestration for the supervised ETF ranking project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from . import data_prep, feature_engineering, training


@dataclass(slots=True)
class PipelineConfig:
    parquet_paths: Iterable[Path]
    output_dataset: Path
    min_sample_count: int = 60
    label_column: str = "oos_compound_sharpe"


@dataclass(slots=True)
class PipelineResult:
    dataset_path: Path
    trained_models: List[training.TrainingResult]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the end-to-end ranking pipeline."""

    load_config = data_prep.LoadConfig(
        paths=list(config.parquet_paths),
        min_oos_sample_count=config.min_sample_count,
    )
    raw_frame = data_prep.prepare_training_frame(load_config)
    features = feature_engineering.build_feature_matrix(raw_frame)
    dataset = feature_engineering.attach_standard_label(
        features, raw_frame, label_column=config.label_column
    )

    _ensure_directory(config.output_dataset.parent)
    dataset.to_parquet(config.output_dataset, index=False)

    trained_models = training.train_baseline_models(
        dataset,
        label_column=config.label_column,
    )
    return PipelineResult(dataset_path=config.output_dataset, trained_models=trained_models)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
