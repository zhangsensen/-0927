"""Shared helpers for rebalance scheduling, timing shifts, and price views."""
from __future__ import annotations

from typing import Tuple
import warnings

import numpy as np

DEFAULT_TIMING_FILL = 1.0

__all__ = [
    "DEFAULT_TIMING_FILL",
    "compute_first_rebalance_index",
    "generate_rebalance_schedule",
    "shift_timing_signal",
    "ensure_price_views",
]


def compute_first_rebalance_index(
    lookback_window: int,
    freq: int,
    *,
    offset: int = 1,
) -> int:
    """Return the first bar index eligible for rebalancing.

    The first candidate index is ``lookback_window + offset`` (offset defaults to ``1``
    because returns on day 0 are not tradable). We then advance until the bar aligns
    with ``freq`` so that both BT and VEC engines use the exact same schedule.
    """

    if freq <= 0:
        raise ValueError("rebalance frequency must be positive")
    start_idx = lookback_window + offset
    remainder = start_idx % freq
    if remainder:
        start_idx += freq - remainder
    return start_idx


def generate_rebalance_schedule(
    total_periods: int,
    lookback_window: int,
    freq: int,
    *,
    offset: int = 1,
    dtype=np.int32,
) -> np.ndarray:
    """Generate an array of rebalance indices with the shared convention."""

    if total_periods <= 0:
        return np.empty(0, dtype=dtype)

    first_idx = compute_first_rebalance_index(lookback_window, freq, offset=offset)
    if first_idx >= total_periods:
        return np.empty(0, dtype=dtype)
    return np.arange(first_idx, total_periods, freq, dtype=dtype)


def shift_timing_signal(
    timing_signal: np.ndarray,
    *,
    fill_value: float = DEFAULT_TIMING_FILL,
) -> np.ndarray:
    """Shift timing signal back by one day to avoid lookahead."""

    timing_signal = np.asarray(timing_signal, dtype=float)
    if timing_signal.ndim != 1:
        raise ValueError("timing_signal must be a 1D array")
    if timing_signal.size == 0:
        return timing_signal.copy()

    shifted = np.empty_like(timing_signal)
    shifted[0] = fill_value
    shifted[1:] = timing_signal[:-1]
    return shifted


def ensure_price_views(
    close_prices: np.ndarray,
    open_prices: np.ndarray | None,
    *,
    copy_if_missing: bool = True,
    warn_if_copied: bool = True,
    validate: bool = True,
    min_valid_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(close_t_minus_1, open_t, close_t)`` price views.

    If ``open_prices`` is ``None`` we optionally copy ``close_prices`` so that
    downstream engines can keep operating while emitting a warning. The helper
    validates shapes to avoid silent broadcasting bugs.
    """

    close_arr = np.asarray(close_prices)
    if close_arr.ndim != 2:
        raise ValueError("close_prices must be a 2D array [T, N]")

    if open_prices is None:
        if not copy_if_missing:
            raise ValueError("open_prices is required but missing")
        open_arr = close_arr.copy()
        if warn_if_copied:
            warnings.warn(
                "open_prices not provided; falling back to close_prices."
                " Pass explicit opens to ensure correct execution pricing.",
                RuntimeWarning,
            )
    else:
        open_arr = np.asarray(open_prices)
        if open_arr.shape != close_arr.shape:
            raise ValueError("open_prices must match close_prices shape")

    if validate:
        segment = slice(min_valid_index, None)
        opens_view = open_arr[segment]
        if np.any(~np.isfinite(opens_view)):
            raise ValueError("open_prices contains NaN/inf entries beyond warmup region")
        if np.any(opens_view <= 0):
            raise ValueError("open_prices must be positive beyond warmup region")

    close_prev = np.vstack([close_arr[0:1], close_arr[:-1]])
    return close_prev, open_arr, close_arr
