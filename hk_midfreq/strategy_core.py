"""Strategy logic and candidate selection for the HK mid-frequency stack."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt

from hk_midfreq.config import (
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_TRADING_CONFIG,
    ExecutionConfig,
    StrategyRuntimeConfig,
    TradingConfig,
)
from hk_midfreq.factor_interface import FactorScoreLoader
from hk_midfreq.fusion import FactorFusionEngine


@dataclass
class StrategySignals:
    """Bundle of entry/exit signals and risk settings for one symbol."""

    symbol: str
    timeframe: str
    entries: pd.Series
    exits: pd.Series
    stop_loss: float
    take_profit: float

    def as_dict(self) -> Dict[str, pd.Series | float | str]:
        """Serialize the signal bundle into a mapping."""

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entries": self.entries,
            "exits": self.exits,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


def _compute_time_based_exits(entries: pd.Series, hold_days: int) -> pd.Series:
    """Generate exit signals ``hold_days`` bars after each entry."""

    cleaned_entries = entries.fillna(False).astype(bool)
    exits = pd.Series(False, index=cleaned_entries.index)
    if hold_days <= 0:
        return exits

    entry_positions = np.flatnonzero(cleaned_entries.to_numpy())
    if entry_positions.size == 0:
        return exits

    exit_positions = entry_positions + hold_days
    exit_positions = exit_positions[exit_positions < len(cleaned_entries.index)]
    if exit_positions.size == 0:
        return exits

    exits.iloc[exit_positions] = True
    return exits


def hk_reversal_logic(
    close: pd.Series,
    volume: pd.Series,
    hold_days: int,
    rsi_window: int = 14,
    bb_window: int = 20,
    volume_window: int = 5,
    rsi_threshold: float = 30.0,
    volume_multiplier: float = 1.2,
) -> StrategySignals:
    """Generate reversal signals using RSI, Bollinger Bands, and volume confirmation."""

    if close.empty:
        raise ValueError("Close series cannot be empty")

    aligned_volume = volume.reindex(close.index).fillna(method="ffill")

    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    bb = vbt.BBANDS.run(close, window=bb_window)
    rolling_volume = aligned_volume.rolling(
        window=volume_window, min_periods=volume_window
    ).mean()

    cond_rsi = rsi < rsi_threshold
    cond_bb = close <= bb.lower
    cond_vol = aligned_volume >= (rolling_volume * volume_multiplier)

    entries = (cond_rsi & cond_bb & cond_vol).fillna(False)
    exits = _compute_time_based_exits(entries, hold_days)

    return StrategySignals(
        symbol="",
        timeframe="",
        entries=entries.astype(bool),
        exits=exits,
        stop_loss=DEFAULT_EXECUTION_CONFIG.stop_loss,
        take_profit=DEFAULT_EXECUTION_CONFIG.primary_take_profit(),
    )


@dataclass(frozen=True)
class FactorDescriptor:
    """Minimal payload describing a selected factor for signal generation."""

    name: str
    timeframe: str
    metadata: Mapping[str, object] | None = None


def _normalize_timeframe_label(label: str) -> str:
    """Normalize timeframe aliases (e.g. ``60min`` -> ``60m``)."""

    lowered = label.lower()
    if lowered in {"60m", "60min", "1h"}:
        return "60m"
    if lowered in {"30m", "30min"}:
        return "30m"
    if lowered in {"15m", "15min"}:
        return "15m"
    if lowered in {"5m", "5min"}:
        return "5m"
    if lowered in {"1d", "1day", "daily", "day"}:
        return "1day"
    return lowered


def _compute_stochrsi(
    close: pd.Series,
    timeperiod: int,
    fastk_period: int,
    fastd_period: int,
) -> tuple[pd.Series, pd.Series]:
    """Return the %K and %D series for a classic StochRSI calculation."""

    if close.empty:
        raise ValueError("Close price series cannot be empty when computing StochRSI")

    rsi = vbt.RSI.run(close, window=timeperiod).rsi
    lowest = rsi.rolling(window=timeperiod, min_periods=timeperiod).min()
    highest = rsi.rolling(window=timeperiod, min_periods=timeperiod).max()
    denominator = (highest - lowest).replace(0.0, np.nan)
    stoch_rsi = (rsi - lowest) / denominator
    stoch_rsi = stoch_rsi.clip(lower=0.0, upper=1.0).ffill().fillna(0.0)

    fastk = (
        stoch_rsi.rolling(window=fastk_period, min_periods=fastk_period).mean().fillna(0.0)
    )
    fastd = (
        fastk.rolling(window=fastd_period, min_periods=fastd_period).mean().fillna(0.0)
    )
    return fastk * 100.0, fastd * 100.0


def _parse_stochrsi_params(name: str) -> tuple[int, int, int, str]:
    """Extract ``timeperiod``, ``fastk`` and ``fastd`` settings and target line from the factor name."""

    tokens = name.split("_")
    timeperiod = 14
    fastk = 5
    fastd = 3
    line = "k"

    for token in tokens:
        match = re.search(r"timeperiod(\d+)", token)
        if match:
            timeperiod = int(match.group(1))
        match = re.search(r"fastk(?:_period)?(\d+)", token)
        if match:
            fastk = int(match.group(1))
        match = re.search(r"fastd(?:_period)?(\d+)", token)
        if match:
            fastd = int(match.group(1))
        if token.lower() in {"k", "d"}:
            line = token.lower()

    if name.endswith("_K"):
        line = "k"
    elif name.endswith("_D"):
        line = "d"

    return timeperiod, fastk, fastd, line


def generate_factor_signals(
    symbol: str,
    timeframe: str,
    close: pd.Series,
    volume: Optional[pd.Series],
    descriptor: FactorDescriptor,
    hold_days: int,
    stop_loss: float,
    take_profit: float,
) -> StrategySignals:
    """Translate a factor descriptor into deterministic entry/exit signals."""

    normalized_tf = _normalize_timeframe_label(timeframe)
    factor_name = descriptor.name
    if factor_name.startswith("TA_STOCHRSI"):
        timeperiod, fastk, fastd, line = _parse_stochrsi_params(factor_name)
        k_series, d_series = _compute_stochrsi(close, timeperiod, fastk, fastd)
        target = k_series if line == "k" else d_series

        oversold = 20.0
        overbought = 80.0
        crosses_up = (target.shift(1) <= oversold) & (target > oversold)
        crosses_down = (target.shift(1) >= overbought) & (target < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(entries, hold_days)

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    raise ValueError(f"Unsupported factor for signal generation: {factor_name}")


class StrategyCore:
    """High-level orchestrator for candidate selection and signal generation."""

    def __init__(
        self,
        trading_config: TradingConfig = DEFAULT_TRADING_CONFIG,
        execution_config: ExecutionConfig = DEFAULT_EXECUTION_CONFIG,
        runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
        factor_loader: Optional[FactorScoreLoader] = None,
        fusion_engine: Optional[FactorFusionEngine] = None,
    ) -> None:
        self.trading_config = trading_config
        self.execution_config = execution_config
        self.runtime_config = runtime_config
        self.factor_loader = factor_loader or FactorScoreLoader(runtime_config)
        self.fusion_engine = fusion_engine or FactorFusionEngine(runtime_config)
        self._last_factor_panel: Optional[pd.DataFrame] = None
        self._last_fused_scores: Optional[pd.DataFrame] = None

    def select_candidates(
        self,
        universe: Iterable[str],
        timeframe: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Select candidate symbols based on aggregated factor scores."""

        symbols = list(universe)
        if not symbols:
            return []

        timeframes: Sequence[str] | None
        if timeframe is not None:
            timeframes = [timeframe]
        else:
            timeframes = self.runtime_config.fusion.ordered_timeframes()

        try:
            panel = self.factor_loader.load_factor_panels(
                symbols=symbols,
                timeframes=timeframes,
                max_factors=self.trading_config.max_positions * 2,
            )
        except FileNotFoundError:
            return []

        self._last_factor_panel = panel
        if panel.empty:
            return []

        fused = self.fusion_engine.fuse(panel)
        self._last_fused_scores = fused
        if fused.empty or "composite_score" not in fused.columns:
            return []

        composite = fused["composite_score"].dropna()
        if composite.empty:
            return []

        limit = top_n or self.trading_config.max_positions
        return composite.sort_values(ascending=False).head(limit).index.tolist()

    def _passes_trend_filter(
        self, frames: Mapping[str, pd.DataFrame]
    ) -> bool:
        trend_tf = self.runtime_config.fusion.trend_timeframe
        if not trend_tf or trend_tf not in frames:
            return True

        data = frames[trend_tf]
        if "close" not in data:
            return True

        close = data["close"].dropna()
        if close.empty:
            return True

        window = max(self.runtime_config.trend_ma_window, 1)
        if len(close) < window:
            return True

        moving_average = close.rolling(window=window).mean()
        return bool(close.iloc[-1] >= moving_average.iloc[-1])

    def _passes_confirmation_filter(
        self, frames: Mapping[str, pd.DataFrame]
    ) -> bool:
        confirmation_tf = self.runtime_config.fusion.confirmation_timeframe
        if not confirmation_tf or confirmation_tf not in frames:
            return True

        data = frames[confirmation_tf]
        if "close" not in data:
            return True

        close = data["close"].dropna()
        if close.empty:
            return True

        window = max(self.runtime_config.confirmation_ma_window, 1)
        if len(close) < window:
            return True

        moving_average = close.rolling(window=window).mean()
        return bool(close.iloc[-1] >= moving_average.iloc[-1])

    def _select_entry_timeframe(
        self, frames: Mapping[str, pd.DataFrame]
    ) -> Optional[str]:
        for timeframe in self.runtime_config.fusion.intraday_timeframes:
            if timeframe in frames:
                return timeframe
        if self.runtime_config.default_timeframe in frames:
            return self.runtime_config.default_timeframe
        return next(iter(frames.keys()), None)

    def _resolve_top_factor(
        self, symbol: str, timeframe: str
    ) -> Optional[FactorDescriptor]:
        if self._last_factor_panel is None or self._last_factor_panel.empty:
            return None

        normalized_tf = _normalize_timeframe_label(timeframe)
        panel = self._last_factor_panel.reset_index()
        panel["normalized_timeframe"] = panel["timeframe"].map(
            lambda tf: _normalize_timeframe_label(str(tf))
        )

        subset = panel[
            (panel["symbol"] == symbol)
            & (panel["normalized_timeframe"] == normalized_tf)
        ]
        if subset.empty:
            return None

        ordered = subset.sort_values(by="rank")
        best = ordered.iloc[0]
        metadata = {
            key: best[key]
            for key in ordered.columns
            if key not in {"symbol", "timeframe", "factor_name", "normalized_timeframe"}
        }
        return FactorDescriptor(
            name=str(best["factor_name"]),
            timeframe=normalized_tf,
            metadata=metadata,
        )

    def generate_signals_for_symbol(
        self,
        symbol: str,
        frames: Mapping[str, pd.DataFrame],
    ) -> Optional[StrategySignals]:
        """Create strategy signals for a single symbol using multi-timeframe data."""

        if not self._passes_trend_filter(frames):
            return None
        if not self._passes_confirmation_filter(frames):
            return None

        entry_timeframe = self._select_entry_timeframe(frames)
        if entry_timeframe is None:
            return None

        data = frames[entry_timeframe]
        if "close" not in data or "volume" not in data:
            return None

        descriptor = self._resolve_top_factor(symbol, entry_timeframe)
        if descriptor is not None:
            try:
                return generate_factor_signals(
                    symbol=symbol,
                    timeframe=entry_timeframe,
                    close=data["close"],
                    volume=data.get("volume"),
                    descriptor=descriptor,
                    hold_days=self.trading_config.hold_days,
                    stop_loss=self.execution_config.stop_loss,
                    take_profit=self.execution_config.primary_take_profit(),
                )
            except Exception:
                # Fallback to legacy logic if factor-driven path fails
                pass

        signal_bundle = hk_reversal_logic(
            close=data["close"],
            volume=data["volume"],
            hold_days=self.trading_config.hold_days,
        )
        return StrategySignals(
            symbol=symbol,
            timeframe=_normalize_timeframe_label(entry_timeframe),
            entries=signal_bundle.entries,
            exits=signal_bundle.exits,
            stop_loss=self.execution_config.stop_loss,
            take_profit=self.execution_config.primary_take_profit(),
        )

    def build_signal_universe(
        self,
        price_data: Mapping[
            str, Mapping[str, pd.DataFrame] | pd.DataFrame
        ],
        timeframe: Optional[str] = None,
    ) -> Dict[str, StrategySignals]:
        """Generate signals for the selected candidate universe."""

        if not price_data:
            return {}

        expanded: Dict[str, Mapping[str, pd.DataFrame]] = {}
        for symbol, data in price_data.items():
            if isinstance(data, Mapping):
                expanded[symbol] = data
            else:
                expanded[symbol] = {self.runtime_config.default_timeframe: data}

        universe = list(expanded.keys())
        candidates = self.select_candidates(universe, timeframe=timeframe)
        signals: Dict[str, StrategySignals] = {}
        for symbol in candidates:
            frames = expanded.get(symbol)
            if not frames:
                continue
            signal = self.generate_signals_for_symbol(symbol=symbol, frames=frames)
            if signal is not None:
                signals[symbol] = signal
        return signals


__all__ = [
    "FactorDescriptor",
    "StrategyCore",
    "StrategySignals",
    "generate_factor_signals",
    "hk_reversal_logic",
]
