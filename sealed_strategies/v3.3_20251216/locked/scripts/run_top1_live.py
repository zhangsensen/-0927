#!/usr/bin/env python3
"""
Top 1 Strategy - Live Signal Generator
======================================
Strategy: ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PV_CORR_20D + SHARPE_RATIO_20D
Freq: 3 Days
Pos: 2

Usage:
    uv run python scripts/run_top1_live.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    generate_rebalance_schedule,
    shift_timing_signal,
)

# Configuration
FREQ = 3
POS_SIZE = 2
TARGET_FACTORS = [
    "ADX_14D",
    "MAX_DD_60D",
    "PRICE_POSITION_120D",
    "PV_CORR_20D",
    "SHARPE_RATIO_20D",
]


def main():
    print(f"=== Top 1 Strategy Live Signal ({datetime.now().date()}) ===")

    # 1. Load Data
    print("Loading Data...")
    loader = DataLoader(data_dir=ROOT / "raw/ETF/daily", cache_dir=ROOT / ".cache")

    # Load recent data (enough for lookback)
    # We need at least 252 days for some factors
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Load Whitelist
    # (Simplified: Load all available in data_dir or use a fixed list)
    # For safety, let's use the list from validation script
    whitelist = [
        "510300",
        "510500",
        "510050",
        "513100",
        "513500",
        "512880",
        "512000",
        "512660",
        "512010",
        "512800",
        "512690",
        "512480",
        "512100",
        "512070",
        "515000",
        "588000",
        "159915",
        "159949",
        "518880",
        "513050",
        "513330",
    ]

    data = loader.load_ohlcv(
        etf_codes=whitelist, start_date=start_date, end_date=end_date
    )

    # 2. Compute Factors
    print("Computing Factors...")
    lib = PreciseFactorLibrary()
    data_dict = {
        "open": data["open"],
        "high": data["high"],
        "low": data["low"],
        "close": data["close"],
        "volume": data["volume"],
    }

    raw_factors_df = lib.compute_all_factors(data_dict)

    # Extract Target Factors
    raw_factors_dict = {}
    for f in TARGET_FACTORS:
        if f in raw_factors_df.columns.get_level_values(0):
            raw_factors_dict[f] = raw_factors_df[f]
        else:
            print(f"Error: Factor {f} not found in library output!")
            return

    # 3. Process Cross-Section
    print("Processing Cross-Section...")
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors_dict)

    # 4. Combine Scores
    total_score = pd.DataFrame(0.0, index=data["close"].index, columns=whitelist)
    for f in TARGET_FACTORS:
        df = std_factors[f]
        total_score = total_score.add(df.fillna(0), fill_value=0)

    # 5. Market Timing
    print("Checking Market Timing...")
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(data["close"])
    # Shift timing signal (Signal at T-1 applied to T)
    # But for "Live" signal generation, we want the signal for "Tomorrow" based on "Today" (or "Today" based on "Yesterday")
    # If we run this AFTER close, we have Today's data. We want signal for Tomorrow.
    # timing_signals index is Date.
    # timing_signals[-1] is the signal generated using data up to Today.
    # This signal applies to Tomorrow.

    current_timing = timing_signals.iloc[-1]

    # 6. Generate Signal
    print("\n=== SIGNAL REPORT ===")
    last_date = total_score.index[-1]
    print(f"Data Date: {last_date.date()}")

    # Check if today is rebalance day
    # We need to reconstruct the schedule
    # This is tricky for live trading.
    # Simple approach: Check if (index in full history) % FREQ == 0
    # But we only loaded partial history.
    # Better: Use a reference date.
    # Assume 2020-01-01 is index 0.
    # Calculate days since 2020-01-01 (trading days).
    # This requires full calendar.

    # For now, we just output the Top K and let the human/system decide if it's a rebalance day
    # OR, we can check the last few days to see if we should have rebalanced.

    scores_today = total_score.iloc[-1]
    valid_scores = scores_today[scores_today != 0].sort_values(ascending=False)

    print(f"\nMarket Timing Ratio: {current_timing:.2f}")

    print(f"\nTop {POS_SIZE} Candidates:")
    for i, (ticker, score) in enumerate(valid_scores.head(POS_SIZE).items()):
        print(f"{i+1}. {ticker}: Score {score:.4f}")

    print(f"\nNext {POS_SIZE} (Watchlist):")
    for i, (ticker, score) in enumerate(
        valid_scores.iloc[POS_SIZE : POS_SIZE * 2].items()
    ):
        print(f"{i+1}. {ticker}: Score {score:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
