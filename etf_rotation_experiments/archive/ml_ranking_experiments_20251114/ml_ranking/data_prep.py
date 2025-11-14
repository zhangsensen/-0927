"""Data preparation helpers for the supervised ETF ranking pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

LIST_LIKE_COLUMNS = (
    "oos_ic_list",
    "oos_ir_list",
    "positive_rate_list",
    "best_freq_list",
    "oos_sharpe_list",
    "oos_daily_mean_list",
    "oos_daily_std_list",
    "oos_sample_count_list",
)


@dataclass(slots=True)
class LoadConfig:
    """Configuration for loading raw ranking results."""

    paths: Sequence[Path]
    min_oos_sample_count: int = 60
    drop_duplicates: bool = True
    dedup_subset: Sequence[str] | None = ("combo",)


def load_raw_results(config: LoadConfig) -> pd.DataFrame:
    """Load and concatenate ranking result parquet files."""

    frames: List[pd.DataFrame] = []
    for path in config.paths:
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Ranking result not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df["source_path"] = parquet_path.as_posix()
        frames.append(df)
    if not frames:
        raise ValueError("No parquet files were provided for loading")
    combined = pd.concat(frames, ignore_index=True)
    if config.drop_duplicates:
        subset = config.dedup_subset
        if subset is not None:
            existing = [column for column in subset if column in combined.columns]
        else:
            existing = None
        before = len(combined)
        combined = combined.drop_duplicates(subset=existing)
        after = len(combined)
        if after < before:
            combined = combined.reset_index(drop=True)
    return combined


def normalize_list_columns(df: pd.DataFrame, list_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Ensure list-like columns contain Python lists instead of string literals."""

    columns = list_columns if list_columns is not None else LIST_LIKE_COLUMNS
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = result[column].apply(_coerce_to_list)
    return result


def filter_by_sample_count(df: pd.DataFrame, *, min_oos_sample_count: int) -> pd.DataFrame:
    """Filter rows with insufficient out-of-sample observations."""

    if "oos_compound_sample_count" not in df.columns:
        return df
    mask = df["oos_compound_sample_count"] >= min_oos_sample_count
    return df.loc[mask].reset_index(drop=True)


def expand_combo_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns derived from the factor combo description."""

    result = df.copy()
    if "combo" not in result.columns:
        return result
    tokens = result["combo"].apply(_split_combo_tokens)
    result["combo_factor_count"] = tokens.apply(len)
    result["combo_unique_factor_count"] = tokens.apply(lambda values: len(set(values)))
    result["combo_factor_name_avg_len"] = tokens.apply(_mean_token_length)
    return result


def prepare_training_frame(config: LoadConfig) -> pd.DataFrame:
    """Produce a cleaned training frame with combo metadata."""

    raw = load_raw_results(config)
    raw = filter_by_sample_count(raw, min_oos_sample_count=config.min_oos_sample_count)
    raw = normalize_list_columns(raw)
    raw = expand_combo_metadata(raw)
    return raw


def _coerce_to_list(value: object) -> List[float]:
    """Convert stored Python literal or comma-separated string to a list."""

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return [_coerce_scalar(v) for v in stripped.split(",") if v]
        if isinstance(parsed, (list, tuple)):
            return [_coerce_scalar(item) for item in parsed]
        return [_coerce_scalar(parsed)]
    return [_coerce_scalar(value)]


def _coerce_scalar(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _split_combo_tokens(combo: object) -> List[str]:
    if not isinstance(combo, str):
        return []
    return [part.strip() for part in combo.split("+") if part.strip()]


def _mean_token_length(tokens: List[str]) -> float:
    if not tokens:
        return float("nan")
    return float(np.mean([len(token) for token in tokens]))
