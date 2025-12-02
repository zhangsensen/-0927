#!/usr/bin/env python3
"""
Single Strategy BT Verification
"""
import sys
import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData

# Configuration
COMBO_STR = "ADX_14D + MAX_DD_60D + OBV_SLOPE_10D + SLOPE_20D"
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

def run_verification():
    print(f"🚀 Running BT Verification for: {COMBO_STR}")
    
    # 1. Load Data
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )

    # 2. Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()

    # 3. Compute Timing
    timing_module = LightTimingModule()
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
    timing_series = pd.Series(timing_arr_shifted, index=dates)

    # 4. Prepare Data Feeds
    data_feeds = {}
    for ticker in etf_codes:
        df = pd.DataFrame(
            {
                "open": ohlcv["open"][ticker],
                "high": ohlcv["high"][ticker],
                "low": ohlcv["low"][ticker],
                "close": ohlcv["close"][ticker],
                "volume": ohlcv["volume"][ticker],
            }
        )
        df = df.reindex(dates)
        df = df.ffill().fillna(0.01)
        data_feeds[ticker] = df

    # 5. Prepare Strategy Inputs
    factors = [f.strip() for f in COMBO_STR.split(" + ")]
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        if f in std_factors:
            combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)
        else:
            print(f"❌ Factor not found: {f}")
            return

    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates),
        lookback_window=LOOKBACK,
        freq=FREQ,
    )

    # 6. Run Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION_RATE, leverage=1.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker, df in data_feeds.items():
        data = PandasData(dataname=df, name=ticker)
        cerebro.adddata(data)

    cerebro.addstrategy(
        GenericStrategy, 
        scores=combined_score_df, 
        timing=timing_series, 
        etf_codes=etf_codes, 
        freq=FREQ, 
        pos_size=POS_SIZE,
        rebalance_schedule=rebalance_schedule
    )

    print("⏳ Starting Backtrader engine...")
    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    bt_return = (end_val / start_val) - 1
    
    print("\n" + "=" * 40)
    print("📊 BT Verification Results")
    print("=" * 40)
    print(f"Initial Capital: {start_val:,.2f}")
    print(f"Final Value:     {end_val:,.2f}")
    print(f"Total Return:    {bt_return*100:.2f}%")
    print(f"Margin Failures: {strat.margin_failures}")
    
    # Check Final Holdings
    print("\n📦 Final Holdings:")
    for ticker, size in strat._get_current_holdings().items():
        price = strat.etf_map[ticker].close[0]
        val = size * price
        print(f"  - {ticker}: {size:.0f} shares @ {price:.3f} = {val:,.2f}")

if __name__ == "__main__":
    run_verification()
