"""Strategy logic and candidate selection for the HK mid-frequency stack."""

from __future__ import annotations

from dataclasses import dataclass
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

        signal_bundle = hk_reversal_logic(
            close=data["close"],
            volume=data["volume"],
            hold_days=self.trading_config.hold_days,
        )
        return StrategySignals(
            symbol=symbol,
            timeframe=entry_timeframe,
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


__all__ = ["StrategyCore", "StrategySignals", "hk_reversal_logic"]
