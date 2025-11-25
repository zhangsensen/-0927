"""Utility helpers for WFO analysis scripts.

These helpers centralize dataset loading and column inference so analysis scripts
can stay focused on the actual computations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Candidate column names ranked by preference.
IC_CANDIDATES = [
    "mean_oos_ic",
    "wfo_ic",
    "ic",
    "mean_ic",
    "ic_mean",
]

RETURN_PRIMARY = [
    "annual_ret_net",
    "annual_return_net",
    "annualized_return_net",
]

RETURN_FALLBACK = [
    "annual_ret",
    "annual_return",
    "annualized_return",
]


@dataclass
class ColumnSelection:
    ic_column: str
    return_column: str


class AnalysisError(RuntimeError):
    """Raised when required data is missing."""


def _expand_paths(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            files.extend(sorted(Path().glob(raw)))
        elif path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
            files.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            files.append(path)
    deduped: List[Path] = []
    seen = set()
    for p in files:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    if not deduped:
        raise AnalysisError("No input files found for the given --input arguments")
    return deduped


def load_datasets(inputs: Iterable[str]) -> pd.DataFrame:
    """Load one or many CSV/Parquet datasets into a single DataFrame."""

    paths = _expand_paths(inputs)
    frames: List[pd.DataFrame] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise AnalysisError(f"Unsupported file extension: {path.name}")
        frame["__source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def detect_column(
    frame: pd.DataFrame,
    explicit: Optional[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> Optional[str]:
    """Resolve a column name either by explicit argument or heuristics."""

    if explicit:
        if explicit not in frame.columns:
            raise AnalysisError(f"Column '{explicit}' not present in input data")
        return explicit
    lower_map = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name in frame.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise AnalysisError(
            "Unable to infer column. Provide it explicitly via CLI or extend candidates."
        )
    return None


def resolve_columns(
    frame: pd.DataFrame,
    *,
    ic_column: Optional[str] = None,
    return_column: Optional[str] = None,
    prefer_net: bool = True,
) -> ColumnSelection:
    """Return the effective IC and return column names to operate on."""

    ic_col = detect_column(frame, ic_column, IC_CANDIDATES)

    ret_candidates: List[str] = []
    if prefer_net:
        ret_candidates.extend(RETURN_PRIMARY)
    ret_candidates.extend(RETURN_FALLBACK)

    ret_col = detect_column(frame, return_column, ret_candidates)
    return ColumnSelection(ic_column=ic_col, return_column=ret_col)


def dropna_pairs(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop rows with NA in any of the specified columns."""

    cols = list(columns)
    return frame.dropna(subset=cols)
