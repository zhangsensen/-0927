"""Strategy logic and candidate selection for the HK mid-frequency stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

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
    ) -> None:
        self.trading_config = trading_config
        self.execution_config = execution_config
        self.runtime_config = runtime_config
        self.factor_loader = factor_loader or FactorScoreLoader(runtime_config)

    def select_candidates(
        self,
        universe: Iterable[str],
        timeframe: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Select candidate symbols based on aggregated factor scores."""

        timeframe_to_use = timeframe or self.runtime_config.default_timeframe
        score_series = self.factor_loader.load_scores_as_series(
            universe, timeframe=timeframe_to_use, top_n=5, agg="mean"
        )
        if score_series.empty:
            return []

        limit = top_n or self.trading_config.max_positions
        selected = score_series.head(limit).index.tolist()
        return selected

    def generate_signals_for_symbol(
        self,
        symbol: str,
        timeframe: str,
        close: pd.Series,
        volume: pd.Series,
    ) -> StrategySignals:
        """Create strategy signals for a single symbol."""

        signal_bundle = hk_reversal_logic(
            close=close,
            volume=volume,
            hold_days=self.trading_config.hold_days,
        )
        return StrategySignals(
            symbol=symbol,
            timeframe=timeframe,
            entries=signal_bundle.entries,
            exits=signal_bundle.exits,
            stop_loss=self.execution_config.stop_loss,
            take_profit=self.execution_config.primary_take_profit(),
        )

    def build_signal_universe(
        self, price_data: Mapping[str, pd.DataFrame], timeframe: Optional[str] = None
    ) -> Dict[str, StrategySignals]:
        """Generate signals for the selected candidate universe."""

        if not price_data:
            return {}

        universe = list(price_data.keys())
        candidates = self.select_candidates(universe, timeframe=timeframe)
        signals: Dict[str, StrategySignals] = {}
        for symbol in candidates:
            data = price_data[symbol]
            if "close" not in data or "volume" not in data:
                continue
            signals[symbol] = self.generate_signals_for_symbol(
                symbol=symbol,
                timeframe=timeframe or self.runtime_config.default_timeframe,
                close=data["close"],
                volume=data["volume"],
            )
        return signals


__all__ = ["StrategyCore", "StrategySignals", "hk_reversal_logic"]
