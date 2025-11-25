"""Feature engineering utilities for supervised ETF ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data_prep import LIST_LIKE_COLUMNS

SUMMARY_STATS = ("mean", "std", "min", "max")


@dataclass(slots=True)
class FeatureSpec:
    base_numeric_columns: Sequence[str]
    list_like_columns: Sequence[str] = LIST_LIKE_COLUMNS
    percentile_values: Sequence[float] = (0.25, 0.5, 0.75)


def build_feature_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec | None = None,
    *,
    drop_original_lists: bool = True,
) -> pd.DataFrame:
    """Create a model-ready feature matrix from the training frame."""

    if spec is None:
        spec = FeatureSpec(
            base_numeric_columns=(
                "combo_size",
                "mean_oos_ic",
                "oos_ic_std",
                "oos_ic_ir",
                "positive_rate",
                "best_rebalance_freq",
                "stability_score",
                "mean_oos_sharpe",
                "oos_sharpe_std",
                "mean_oos_sample_count",
                # "oos_compound_mean",  # 🚨 REMOVED: 数学泄露 - 与标签直接相关
                # "oos_compound_std",   # 🚨 REMOVED: 数学泄露 - 与标签直接相关
                "oos_compound_sample_count",
                "p_value",
                "q_value",
                "oos_sharpe_proxy",
                "combo_factor_count",
                "combo_unique_factor_count",
                "combo_factor_name_avg_len",
            )
        )

    feature_frames = [_select_existing(df, spec.base_numeric_columns)]

    for column in spec.list_like_columns:
        if column not in df.columns:
            continue
        list_features = _summarize_list_column(df[column], column, spec.percentile_values)
        feature_frames.append(pd.DataFrame(list_features))
        if drop_original_lists:
            feature_frames.append(pd.DataFrame({column + "_width": df[column].apply(len)}))
    features = pd.concat(feature_frames, axis=1)
    return features


def attach_standard_label(
    features: pd.DataFrame,
    df: pd.DataFrame,
    *,
    label_column: str = "oos_compound_sharpe",
) -> pd.DataFrame:
    """Append the label column to the feature matrix."""

    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not present in training frame")
    result = features.copy()
    result[label_column] = df[label_column].values
    return result


def _select_existing(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    return df[existing].copy()


def _summarize_list_column(
    series: pd.Series,
    prefix: str,
    percentiles: Sequence[float],
) -> Mapping[str, Sequence[float]]:
    """Generate statistical summaries for a sequence column."""

    summaries: dict[str, list[float]] = {}
    percentiles = tuple(percentiles)
    for stat in SUMMARY_STATS:
        summaries[f"{prefix}_{stat}"] = []
    for percentile in percentiles:
        summaries[f"{prefix}_p{int(percentile * 100):02d}"] = []
    summaries[f"{prefix}_trend"] = []
    summaries[f"{prefix}_last"] = []
    summaries[f"{prefix}_count"] = []

    for values in series:
        arr = _safe_to_array(values)
        if arr.size == 0:
            for key in summaries:
                summaries[key].append(np.nan)
            continue
        summaries[f"{prefix}_mean"].append(float(np.mean(arr)))
        summaries[f"{prefix}_std"].append(float(np.std(arr, ddof=0)))
        summaries[f"{prefix}_min"].append(float(np.min(arr)))
        summaries[f"{prefix}_max"].append(float(np.max(arr)))
        for percentile in percentiles:
            value = float(np.quantile(arr, percentile))
            summaries[f"{prefix}_p{int(percentile * 100):02d}"].append(value)
        trend = float(arr[-1] - arr[0])
        summaries[f"{prefix}_trend"].append(trend)
        summaries[f"{prefix}_last"].append(float(arr[-1]))
        summaries[f"{prefix}_count"].append(float(arr.size))
    return summaries


def _safe_to_array(values: object) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(float)
    if isinstance(values, (list, tuple)):
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)
    return np.asarray([], dtype=float)
