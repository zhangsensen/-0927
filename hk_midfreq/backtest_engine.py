"""Backtesting utilities built on top of vectorbt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt

from hk_midfreq.config import (
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_TRADING_CONFIG,
    ExecutionConfig,
    TradingConfig,
)
from hk_midfreq.strategy_core import StrategySignals


@dataclass
class BacktestArtifacts:
    """Container holding the generated portfolio and associated metadata."""

    portfolio: vbt.Portfolio
    signals: Dict[str, StrategySignals]


def _position_size_from_prices(
    close: pd.Series, trading_config: TradingConfig
) -> pd.Series:
    """Compute discrete share counts for a single asset."""

    allocation = trading_config.allocation_per_position()
    quantity = np.floor(allocation / close).replace([np.inf, -np.inf], 0.0)
    return quantity.fillna(0.0)


def run_single_asset_backtest(
    close: pd.Series,
    signals: StrategySignals,
    trading_config: TradingConfig = DEFAULT_TRADING_CONFIG,
    execution_config: ExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> vbt.Portfolio:
    """Execute a single-asset backtest using vectorbt."""

    if close.empty:
        raise ValueError("Close price series is empty")

    entries = signals.entries.reindex(close.index).fillna(False)
    exits = signals.exits.reindex(close.index).fillna(False)
    size = _position_size_from_prices(close, trading_config)

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=trading_config.allocation_per_position(),
        fees=execution_config.transaction_cost,
        slippage=execution_config.slippage,
        size=size,
        stop_loss=signals.stop_loss,
        take_profit=signals.take_profit,
        direction="longonly",
    )
    return portfolio


def _build_matrix(
    data: Mapping[str, pd.Series], index: pd.Index, fill_value: float | bool = 0.0
) -> pd.DataFrame:
    """Align a mapping of series to a shared index."""

    columns = {}
    for symbol, series in data.items():
        columns[symbol] = series.reindex(index).fillna(fill_value)
    return pd.DataFrame(columns, index=index)


def _coerce_signal(
    symbol: str,
    payload: StrategySignals | Mapping[str, Any],
    execution_config: ExecutionConfig,
) -> StrategySignals:
    if isinstance(payload, StrategySignals):
        return payload
    if isinstance(payload, Mapping):
        entries = payload.get("entries")
        exits = payload.get("exits")
        if entries is None or exits is None:
            raise ValueError(f"Missing entries/exits for symbol {symbol}")
        timeframe = payload.get("timeframe", "")
        stop_loss = float(payload.get("stop_loss", execution_config.stop_loss))
        take_profit = float(
            payload.get("take_profit", execution_config.primary_take_profit())
        )
        return StrategySignals(
            symbol=symbol,
            timeframe=str(timeframe),
            entries=pd.Series(entries),
            exits=pd.Series(exits),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
    raise TypeError(f"Unsupported signal payload for symbol {symbol}: {type(payload)!r}")


def run_portfolio_backtest(
    price_data: Mapping[str, Mapping[str, pd.DataFrame] | pd.DataFrame],
    signals: Mapping[str, StrategySignals | Mapping[str, Any]],
    trading_config: TradingConfig = DEFAULT_TRADING_CONFIG,
    execution_config: ExecutionConfig = DEFAULT_EXECUTION_CONFIG,
) -> Optional[BacktestArtifacts]:
    """Run a portfolio backtest for the provided signal map."""

    if not signals:
        return None

    close_columns: Dict[str, pd.Series] = {}
    entries_map: Dict[str, pd.Series] = {}
    exits_map: Dict[str, pd.Series] = {}

    for symbol, payload in signals.items():
        try:
            signal = _coerce_signal(symbol, payload, execution_config)
        except (TypeError, ValueError):
            continue

        raw_prices = price_data.get(symbol)
        if raw_prices is None:
            continue
        if isinstance(raw_prices, Mapping):
            frame = raw_prices.get(signal.timeframe)
            if frame is None and signal.timeframe:
                frame = raw_prices.get(signal.timeframe.lower())
            if frame is None and raw_prices:
                frame = next(iter(raw_prices.values()))
        else:
            frame = raw_prices
        if frame is None or "close" not in frame:
            continue
        close_series = frame["close"].dropna()
        close_columns[symbol] = close_series
        entries_map[symbol] = signal.entries
        exits_map[symbol] = signal.exits

    if not close_columns:
        return None

    combined_index = pd.Index(
        sorted(set().union(*[series.index for series in close_columns.values()]))
    )

    close_df = _build_matrix(close_columns, combined_index)
    entries_df = _build_matrix(entries_map, combined_index, fill_value=False).astype(
        bool
    )
    exits_df = _build_matrix(exits_map, combined_index, fill_value=False).astype(bool)

    allocation = trading_config.allocation_per_position()
    size_df = close_df.apply(lambda col: np.floor(allocation / col), axis=0)
    size_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    size_df.fillna(0.0, inplace=True)

    portfolio = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_df,
        exits=exits_df,
        init_cash=trading_config.capital,
        fees=execution_config.transaction_cost,
        slippage=execution_config.slippage,
        stop_loss=execution_config.stop_loss,
        take_profit=execution_config.primary_take_profit(),
        size=size_df,
        direction="longonly",
    )

    return BacktestArtifacts(portfolio=portfolio, signals=dict(signals))


__all__ = ["BacktestArtifacts", "run_single_asset_backtest", "run_portfolio_backtest"]
