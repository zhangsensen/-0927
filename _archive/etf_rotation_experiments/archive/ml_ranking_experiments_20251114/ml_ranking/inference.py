"""Utilities for applying supervised ranking models to new data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping

import numpy as np
import pandas as pd

from . import data_prep, feature_engineering, training
from .models import ModelConfig, baseline_model_registry


@dataclass(slots=True)
class InferenceArtifacts:
    """Container for the trained model and feature metadata."""

    model_name: str
    model: Any
    feature_columns: List[str]
    label_column: str
    fill_values: pd.Series
    evaluation_metrics: Mapping[str, float]


def train_inference_model(
    dataset: pd.DataFrame,
    *,
    label_column: str = "oos_compound_sharpe",
    preferred_model: str | None = None,
    extra_eval_top_ks: Iterable[int] | None = (50, 100, 200, 500, 1000),
) -> InferenceArtifacts:
    """Train a model on the full dataset for inference time usage."""

    if label_column not in dataset.columns:
        raise KeyError(f"Label column '{label_column}' not present in dataset")

    feature_columns = [column for column in dataset.columns if column != label_column]
    if not feature_columns:
        raise ValueError("Dataset does not contain feature columns")

    numeric_features = dataset[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = numeric_features.median().fillna(0.0)
    filled_features = numeric_features.fillna(fill_values).fillna(0.0)
    target = dataset[label_column].to_numpy(dtype=float)

    training_results = training.train_baseline_models(
        dataset,
        label_column=label_column,
        extra_eval_top_ks=extra_eval_top_ks,
    )
    metrics_map: dict[str, Mapping[str, float]] = {
        result.model_name: result.evaluation.metrics for result in training_results
    }
    model_name = _select_model_name(metrics_map, preferred_model)

    registry = baseline_model_registry(ModelConfig())
    model = registry.get(model_name)
    if model is None:
        raise KeyError(f"Model '{model_name}' not available in registry")
    model.fit(filled_features, target)

    return InferenceArtifacts(
        model_name=model_name,
        model=model,
        feature_columns=list(filled_features.columns),
        label_column=label_column,
        fill_values=fill_values.reindex(filled_features.columns).fillna(0.0),
        evaluation_metrics=dict(metrics_map.get(model_name, {})),
    )


def prepare_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature pipeline used during training to raw combos."""

    normalized = data_prep.normalize_list_columns(raw_df)
    enriched = data_prep.expand_combo_metadata(normalized)
    features = feature_engineering.build_feature_matrix(enriched)
    features.index = raw_df.index
    return features


def align_features_for_scoring(
    features: pd.DataFrame,
    artifacts: InferenceArtifacts,
) -> pd.DataFrame:
    """Align inference features with the training schema used by the model."""

    aligned = features.copy()
    for column in artifacts.feature_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    aligned = aligned[artifacts.feature_columns]
    aligned = aligned.apply(pd.to_numeric, errors="coerce")
    fill_values = artifacts.fill_values.reindex(artifacts.feature_columns).fillna(0.0)
    aligned = aligned.fillna(fill_values).fillna(0.0)

    return aligned


def score_features(features: pd.DataFrame, artifacts: InferenceArtifacts) -> pd.Series:
    """Generate model scores for the provided feature matrix."""

    aligned = align_features_for_scoring(features, artifacts)
    scores = artifacts.model.predict(aligned)
    return pd.Series(scores, index=aligned.index, name=f"{artifacts.model_name}_score")


def _select_model_name(
    metrics_map: Mapping[str, Mapping[str, float]],
    preferred_model: str | None,
) -> str:
    if preferred_model and preferred_model in metrics_map:
        return preferred_model
    if not metrics_map:
        raise ValueError("No training metrics available for model selection")

    def _score(item: tuple[str, Mapping[str, float]]) -> float:
        metrics = item[1]
        return float(metrics.get("spearman", float("-inf")))

    best_name, _ = max(metrics_map.items(), key=_score)
    return best_name
