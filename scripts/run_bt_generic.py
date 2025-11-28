#!/usr/bin/env python3
"""Run production GenericStrategy to inspect margin failures and equity."""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'etf_rotation_optimized'))

import numpy as np
import pandas as pd
import yaml
import backtrader as bt

from etf_rotation_optimized.core.data_loader import DataLoader
from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
from etf_rotation_optimized.core.market_timing import LightTimingModule
from strategy_auditor.core.engine import GenericStrategy, PandasData as EnginePandasData

FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002
LOOKBACK = 252

COMBO = "MAX_DD_60D + MOM_20D + RSI_14 + VOL_RATIO_20D + VOL_RATIO_60D"


def main():
    config_path = os.path.join(project_root, 'configs', 'combo_wfo_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    etf_codes = config['data']['symbols']
    start_date = config['data'].get('start_date', '2019-01-01')
    end_date = config['data'].get('end_date', '2024-11-25')

    loader = DataLoader()
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes, start_date=start_date, end_date=end_date)

    factor_lib = PreciseFactorLibrary()
    factor_names = COMBO.split(' + ')
    all_factors_df = factor_lib.compute_all_factors(ohlcv)

    raw_factors = {fname: all_factors_df[fname] for fname in factor_names}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    scores = next(iter(std_factors.values())).copy() * 0
    for fname in factor_names:
        scores += std_factors[fname].fillna(0)

    timing_module = LightTimingModule()
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION_RATE)
    cerebro.broker.set_coc(True)

    close_df = ohlcv['close']
    for ticker in etf_codes:
        if ticker not in close_df.columns:
            continue
        df = pd.DataFrame({
            'open': ohlcv['open'][ticker],
            'high': ohlcv['high'][ticker],
            'low': ohlcv['low'][ticker],
            'close': ohlcv['close'][ticker],
            'volume': ohlcv['volume'][ticker],
        }).dropna()
        if len(df) < LOOKBACK:
            continue
        data = EnginePandasData(dataname=df)
        cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(
        GenericStrategy,
        scores=scores,
        timing=timing_series,
        etf_codes=etf_codes,
        freq=FREQ,
        pos_size=POS_SIZE,
    )

    strat = cerebro.run()[0]
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print("Final Value:", f"{final_value:,.2f}")
    print("Total Return:", f"{total_return:.2f}%")
    print("Margin Failures:", strat.margin_failures)
    print("Orders Logged:", len(strat.orders))


if __name__ == "__main__":
    main()
