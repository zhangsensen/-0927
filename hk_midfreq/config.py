"""Configuration objects for the HK mid-frequency strategy stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, Tuple

DEFAULT_TAKE_PROFITS = (0.006, 0.01, 0.014, 0.018)


@dataclass(frozen=True)
class FusionConfig:
    """Parameters governing how multi-timeframe factor scores are fused."""

    factor_weighting: str = "ic_weighted"
    timeframe_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "daily": 0.5,
            "60min": 0.3,
            "15min": 0.2,
        }
    )
    trend_timeframe: str = "daily"
    confirmation_timeframe: str = "60min"
    intraday_timeframes: Tuple[str, ...] = ("15min", "5min")
    trend_threshold: float = 0.0
    confirmation_threshold: float = 0.0

    def ordered_timeframes(self) -> Tuple[str, ...]:
        """Return the unique timeframes required by the fusion rules."""

        ordered: list[str] = []
        for tf in (
            self.trend_timeframe,
            self.confirmation_timeframe,
            *self.intraday_timeframes,
        ):
            if tf and tf not in ordered:
                ordered.append(tf)
        # Include any explicitly weighted timeframes so the loader can fetch them
        for tf in self.timeframe_weights.keys():
            if tf and tf not in ordered:
                ordered.append(tf)
        return tuple(ordered)


@dataclass(frozen=True)
class TradingConfig:
    """Portfolio sizing and holding horizon parameters."""

    capital: float = 1_000_000.0
    position_size: float = 100_000.0
    max_positions: int = 8
    hold_days: int = 4

    def allocation_per_position(self) -> float:
        """Return the maximum cash allocated to a single position."""

        return min(self.position_size, self.capital / max(self.max_positions, 1))


@dataclass(frozen=True)
class ExecutionConfig:
    """Trading frictions and stop management settings."""

    transaction_cost: float = 0.0038
    slippage: float = 0.001
    stop_loss: float = 0.018
    take_profit_levels: Sequence[float] = field(
        default_factory=lambda: DEFAULT_TAKE_PROFITS
    )

    def primary_take_profit(self) -> float:
        """Pick the first take-profit level for engines that only support one value."""

        return self.take_profit_levels[0] if self.take_profit_levels else 0.0


@dataclass(frozen=True)
class StrategyRuntimeConfig:
    """High-level toggles for the HK mid-frequency workflow."""

    strategy_name: str = "HK_MIDFREQ_REVERSAL"
    base_output_dir: Path = Path("因子筛选")
    default_timeframe: str = "60min"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    trend_ma_window: int = 20
    confirmation_ma_window: int = 10


DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_EXECUTION_CONFIG = ExecutionConfig()
DEFAULT_RUNTIME_CONFIG = StrategyRuntimeConfig()
