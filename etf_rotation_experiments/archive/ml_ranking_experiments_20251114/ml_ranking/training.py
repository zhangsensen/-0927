"""Training orchestration for supervised ETF ranking models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluation import EvaluationResult, evaluate_predictions
from .models import ModelConfig, baseline_model_registry


@dataclass(slots=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    model_config: ModelConfig = field(default_factory=ModelConfig)


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    model: Any
    evaluation: EvaluationResult


def train_baseline_models(
    features: pd.DataFrame,
    *,
    label_column: str,
    config: TrainingConfig | None = None,
    extra_eval_top_ks: Iterable[int] | None = None,
) -> List[TrainingResult]:
    """Train a collection of baseline models and evaluate them."""

    cfg = config or TrainingConfig()
    X = features.drop(columns=[label_column])
    X = X.apply(pd.to_numeric, errors="coerce")
    median_values = X.median()
    X = X.fillna(median_values)
    X = X.fillna(0.0)
    y = features[label_column].to_numpy(dtype=float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    results: List[TrainingResult] = []
    model_registry = baseline_model_registry(cfg.model_config)
    for name, model in model_registry.items():
        trained_model = _fit_model(name, model, X_train, y_train, X_valid, y_valid)
        y_pred = trained_model.predict(X_valid)
        evaluation = evaluate_predictions(
            y_true=y_valid,
            y_pred=y_pred,
            model_name=name,
            top_ks=extra_eval_top_ks if extra_eval_top_ks is not None else None,
        )
        results.append(TrainingResult(model_name=name, model=trained_model, evaluation=evaluation))
    return results


def _fit_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> Any:
    """Fit a regression or ranking model with sensible defaults."""

    if hasattr(model, "fit"):
        if name == "lgbm_ranker":
            group_train = np.array([len(X_train)], dtype=int)
            group_valid = np.array([len(X_valid)], dtype=int)
            model.fit(
                X_train,
                y_train,
                group=group_train,
                eval_set=[(X_valid, y_valid)],
                eval_group=[group_valid],
                eval_at=[5, 10, 25],
            )
        else:
            model.fit(X_train, y_train)
    else:  # pragma: no cover - defensive branch
        raise AttributeError(f"Model '{name}' does not expose a fit method")
    return model
