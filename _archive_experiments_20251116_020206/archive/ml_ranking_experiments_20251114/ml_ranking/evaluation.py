"""Model evaluation helpers for supervised ETF ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EvaluationResult:
    model_name: str
    metrics: dict[str, float]


DEFAULT_TOP_KS = (5, 10, 25, 50)


def evaluate_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    model_name: str,
    top_ks: Iterable[int] | None = DEFAULT_TOP_KS,
) -> EvaluationResult:
    """Compute validation metrics for a model."""

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    metrics: dict[str, float] = {}
    metrics["spearman"] = _spearman(y_true_arr, y_pred_arr)
    if top_ks is None:
        top_ks = DEFAULT_TOP_KS
    for k in top_ks:
        metrics[f"top{int(k)}_overlap"] = _top_k_overlap(y_true_arr, y_pred_arr, k)
        metrics[f"ndcg@{int(k)}"] = _ndcg(y_true_arr, y_pred_arr, k)
    return EvaluationResult(model_name=model_name, metrics=metrics)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    series_true = pd.Series(y_true)
    series_pred = pd.Series(y_pred)
    return float(series_true.corr(series_pred, method="spearman"))


def _top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if y_true.size == 0 or k <= 0:
        return float("nan")
    k = min(k, y_true.size)
    top_true = np.argpartition(-y_true, k - 1)[:k]
    top_pred = np.argpartition(-y_pred, k - 1)[:k]
    overlap = len(set(top_true.tolist()) & set(top_pred.tolist()))
    return float(overlap / k)


def _ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if y_true.size == 0 or k <= 0:
        return float("nan")
    k = min(k, y_true.size)
    indices = np.argsort(-y_pred)[:k]
    gains = (2 ** y_true[indices] - 1) / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal_indices = np.argsort(-y_true)[:k]
    ideal_gains = (2 ** y_true[ideal_indices] - 1) / np.log2(np.arange(2, k + 2))
    idcg = ideal_gains.sum()
    if idcg == 0:
        return float("nan")
    return float(dcg / idcg)
