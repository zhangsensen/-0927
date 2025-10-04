from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_midfreq.strategy_core import (  # noqa: E402
    FactorDescriptor,
    StrategyCore,
    generate_factor_signals,
)


def _build_synthetic_prices(periods: int = 200) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=periods, freq="D")
    values = 50 + np.sin(np.linspace(0, 12, periods)) * 5
    return pd.Series(values, index=index, name="close")


def test_generate_factor_signals_produces_entries_and_exits() -> None:
    close = _build_synthetic_prices()
    volume = pd.Series(1_000_000, index=close.index, name="volume")
    descriptor = FactorDescriptor(
        name="TA_STOCHRSI_fastd_period3_fastk_period5_timeperiod14_K",
        timeframe="daily",
    )

    signals = generate_factor_signals(
        symbol="0700.HK",
        timeframe="daily",
        close=close,
        volume=volume,
        descriptor=descriptor,
        hold_days=5,
        stop_loss=0.02,
        take_profit=0.05,
    )

    assert signals.entries.index.equals(close.index)
    assert signals.entries.any()
    assert signals.exits.any()
    assert signals.timeframe == "1day"


def test_strategy_core_prefers_factor_driven_signals() -> None:
    close = _build_synthetic_prices()
    volume = pd.Series(1_000_000, index=close.index, name="volume")
    frames = {"daily": pd.DataFrame({"close": close, "volume": volume})}

    core = StrategyCore()
    core._last_factor_panel = pd.DataFrame(
        {
            "rank": [1],
            "comprehensive_score": [0.9],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (
                    "0700.HK",
                    "daily",
                    "TA_STOCHRSI_fastd_period3_fastk_period5_timeperiod14_K",
                )
            ],
            names=["symbol", "timeframe", "factor_name"],
        ),
    )

    signals = core.generate_signals_for_symbol("0700.HK", frames)
    assert signals is not None
    assert signals.entries.any()
