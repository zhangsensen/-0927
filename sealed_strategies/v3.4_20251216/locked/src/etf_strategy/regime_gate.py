"""Regime gate helpers (v3.2).

目标：把“坏环境降仓/停跑”的逻辑做成可复用组件，统一接入 VEC/BT/审计。

设计约束：
- 默认不启用（配置 enabled=false 时返回全 1.0，不改变现有行为）。
- 无前视偏差：信号默认使用 t-1 作用于 t（shift_days=1）。
- 尽量不侵入核心引擎：通过缩放 timing / exposure 数组实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from etf_strategy.core.regime_detector import MarketRegime, RegimeDetector
from etf_strategy.core.utils.rebalance import DEFAULT_TIMING_FILL, shift_timing_signal


def _shift_n(signal: np.ndarray, *, n: int, fill_value: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if n <= 0:
        return signal.copy()
    if signal.size == 0:
        return signal.copy()

    if n == 1:
        return shift_timing_signal(signal, fill_value=fill_value)

    shifted = np.empty_like(signal)
    shifted[:n] = fill_value
    shifted[n:] = signal[:-n]
    return shifted


def _resolve_market_proxy(close_df: pd.DataFrame, proxy_symbol: str) -> pd.Series:
    """Resolve a deterministic market proxy price series."""

    if proxy_symbol == "market_avg":
        return close_df.mean(axis=1)

    if proxy_symbol in close_df.columns:
        return close_df[proxy_symbol]

    # Common fallbacks
    for fallback in ("510300", "510050", "159919"):
        if fallback in close_df.columns:
            return close_df[fallback]

    # Last resort: equal-weight average
    return close_df.mean(axis=1)


def _validate_thresholds_and_exposures(
    thresholds_pct: Sequence[float],
    exposures: Sequence[float],
) -> None:
    if len(exposures) != len(thresholds_pct) + 1:
        raise ValueError(
            "exposures length must be len(thresholds_pct)+1 "
            f"(got thresholds={len(thresholds_pct)}, exposures={len(exposures)})"
        )
    if any(np.isnan(thresholds_pct)):
        raise ValueError("thresholds_pct contains NaN")
    if any(x < 0 for x in exposures):
        raise ValueError("exposures must be non-negative")


def compute_volatility_gate_raw(
    close_df: pd.DataFrame,
    *,
    proxy_symbol: str = "510300",
    window: int = 20,
    thresholds_pct: Sequence[float] = (25, 30, 40),
    exposures: Sequence[float] = (1.0, 0.7, 0.4, 0.1),
) -> pd.Series:
    """Compute raw (unshifted) volatility-based exposure gate.

    Returns a pd.Series indexed by close_df.index with values in [0, 1].

    Notes:
    - Uses annualized realized vol (%) of a proxy series.
    - To match existing project scripts, uses (hv + hv.shift(5)) / 2 as the regime vol.
    """

    _validate_thresholds_and_exposures(thresholds_pct, exposures)

    proxy_close = _resolve_market_proxy(close_df, proxy_symbol)
    rets = proxy_close.pct_change()

    hv = rets.rolling(window=window, min_periods=window).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2

    exp = np.full(len(regime_vol), float(exposures[0]), dtype=np.float64)
    vol_vals = regime_vol.values.astype(np.float64)

    for i, thr in enumerate(thresholds_pct):
        exp[vol_vals >= float(thr)] = float(exposures[i + 1])

    exp_s = pd.Series(exp, index=close_df.index)
    exp_s = exp_s.fillna(float(exposures[0]))
    return exp_s


def compute_market_regime_gate_raw(
    close_df: pd.DataFrame,
    *,
    window: int = 60,
    bull_exposure: float = 1.0,
    sideways_exposure: float = 0.7,
    bear_exposure: float = 0.1,
) -> pd.Series:
    """Compute raw (unshifted) market-regime exposure gate.

    This uses RegimeDetector (rule-based bull/bear/sideways) computed from equal-weight market proxy.
    """

    detector = RegimeDetector(window=window)
    regime_s, _metrics = detector.detect_regime({"close": close_df})

    mapping = {
        MarketRegime.BULL.value: float(bull_exposure),
        MarketRegime.SIDEWAYS.value: float(sideways_exposure),
        MarketRegime.BEAR.value: float(bear_exposure),
    }

    exp_s = regime_s.map(mapping).astype(float)
    exp_s = exp_s.reindex(close_df.index).fillna(float(sideways_exposure))
    return exp_s


def compute_regime_gate_arr(
    close_df: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    *,
    backtest_config: dict | None = None,
    fill_value: float = DEFAULT_TIMING_FILL,
) -> np.ndarray:
    """Compute shifted gate exposure aligned to `dates`.

    - Returns np.ndarray (T,) float64.
    - When disabled: returns ones.
    """

    if backtest_config is None:
        backtest_config = {}

    gate_cfg = backtest_config.get("regime_gate") or {}
    enabled = bool(gate_cfg.get("enabled", False))
    if not enabled:
        return np.ones(len(list(dates)), dtype=np.float64)

    mode = str(gate_cfg.get("mode", "volatility")).strip().lower()

    shift_days = 1
    raw_gate: pd.Series

    if mode == "volatility":
        vol_cfg = gate_cfg.get("volatility") or {}
        proxy_symbol = str(vol_cfg.get("proxy_symbol", "510300"))
        window = int(vol_cfg.get("window", 20))
        shift_days = int(vol_cfg.get("shift_days", 1))
        thresholds_pct = tuple(vol_cfg.get("thresholds_pct", [25, 30, 40]))
        exposures = tuple(vol_cfg.get("exposures", [1.0, 0.7, 0.4, 0.1]))

        raw_gate = compute_volatility_gate_raw(
            close_df,
            proxy_symbol=proxy_symbol,
            window=window,
            thresholds_pct=thresholds_pct,
            exposures=exposures,
        )
        default_fill = float(exposures[0]) if exposures else float(fill_value)
    elif mode == "market_regime":
        mr_cfg = gate_cfg.get("market_regime") or {}
        window = int(mr_cfg.get("window", 60))
        bull_exposure = float(mr_cfg.get("bull_exposure", 1.0))
        sideways_exposure = float(mr_cfg.get("sideways_exposure", 0.7))
        bear_exposure = float(mr_cfg.get("bear_exposure", 0.1))

        raw_gate = compute_market_regime_gate_raw(
            close_df,
            window=window,
            bull_exposure=bull_exposure,
            sideways_exposure=sideways_exposure,
            bear_exposure=bear_exposure,
        )
        default_fill = float(sideways_exposure)
        shift_days = int(mr_cfg.get("shift_days", 1))
    else:
        raise ValueError(f"Unknown regime_gate.mode: {mode}")

    dates_index = pd.DatetimeIndex(list(dates))
    raw_gate = raw_gate.reindex(dates_index).fillna(default_fill)

    raw_arr = raw_gate.values.astype(np.float64)
    shifted = _shift_n(raw_arr, n=shift_days, fill_value=default_fill)

    # Safety clamp: avoid negative exposure
    shifted = np.clip(shifted, 0.0, 1.0)
    return shifted


def gate_stats(gate_arr: np.ndarray) -> dict:
    gate_arr = np.asarray(gate_arr, dtype=float)
    if gate_arr.size == 0:
        return {"mean": 1.0, "min": 1.0, "max": 1.0}
    return {
        "mean": float(np.mean(gate_arr)),
        "min": float(np.min(gate_arr)),
        "max": float(np.max(gate_arr)),
    }
