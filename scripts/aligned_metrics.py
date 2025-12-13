"""Utility helpers to compute aligned return/ShARPE from the same equity curve."""

from __future__ import annotations

import numpy as np


def compute_aligned_metrics(equity_curve: np.ndarray, start_idx: int = 0) -> dict:
    """
    Calculate total return and Sharpe using a unified formula on the same equity path.

    Parameters
    ----------
    equity_curve : np.ndarray
        Portfolio equity values by day.
    start_idx : int, optional
        Index to start the evaluation window (e.g., first rebalance day / lookback end).

    Returns
    -------
    dict
        {
            "aligned_return": float,
            "aligned_sharpe": float,
            "daily_returns": np.ndarray,
        }
    """
    if equity_curve is None:
        return {
            "aligned_return": 0.0,
            "aligned_sharpe": 0.0,
            "daily_returns": np.array([], dtype=np.float64),
        }

    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return {
            "aligned_return": 0.0,
            "aligned_sharpe": 0.0,
            "daily_returns": np.array([], dtype=np.float64),
        }

    start = max(int(start_idx), 0)
    if start >= equity.size:
        start = 0

    window = equity[start:]
    # drop non-finite values to avoid propagating NaNs
    window = window[np.isfinite(window)]
    if window.size == 0:
        return {
            "aligned_return": 0.0,
            "aligned_sharpe": 0.0,
            "daily_returns": np.array([], dtype=np.float64),
        }

    daily_returns = np.diff(window) / window[:-1]
    aligned_return = (window[-1] - window[0]) / window[0] if window[0] != 0 else 0.0

    aligned_sharpe = 0.0
    valid_returns = daily_returns[np.isfinite(daily_returns)]
    if valid_returns.size > 1:
        mean_r = float(np.nanmean(valid_returns))
        std_r = float(np.nanstd(valid_returns, ddof=1))
        if std_r > 1e-12:
            aligned_sharpe = mean_r / std_r * np.sqrt(252.0)

    return {
        "aligned_return": aligned_return,
        "aligned_sharpe": aligned_sharpe,
        "daily_returns": valid_returns,
    }


