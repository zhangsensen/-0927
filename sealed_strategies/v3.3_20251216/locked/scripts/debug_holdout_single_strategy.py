#!/usr/bin/env python3
"""
Debug single strategy on Holdout to see why trades = 0
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)

from batch_vec_backtest import run_vec_backtest

# Load Configuration
config_path = ROOT / "configs/combo_wfo_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Parameters
FREQ = 3
POS_SIZE = 2
EXTREME_THRESHOLD = -0.1
EXTREME_POSITION = 0.1

# Load full data
loader = DataLoader(
    data_dir=config["data"].get("data_dir"),
    cache_dir=config["data"].get("cache_dir"),
)

ohlcv = loader.load_ohlcv(
    etf_codes=config["data"]["symbols"],
    start_date=config["data"]["start_date"],
    end_date=config["data"]["end_date"],
)

print(f"完整数据范围: {ohlcv['close'].index[0]} 至 {ohlcv['close'].index[-1]}")
print(f"数据形状: {ohlcv['close'].shape}")

# Compute Factors
factor_lib = PreciseFactorLibrary()
raw_factors_df = factor_lib.compute_all_factors(ohlcv)
factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

processor = CrossSectionProcessor(verbose=False)
std_factors = processor.process_all_factors(raw_factors)

# Prepare data
first_factor = std_factors[factor_names_list[0]]
all_dates = first_factor.index
etf_codes = first_factor.columns.tolist()

# Filter to Holdout period
training_end = pd.Timestamp(config["data"]["training_end_date"])
holdout_mask = all_dates >= training_end
dates = all_dates[holdout_mask]

print(f"\nHoldout期: {dates[0]} 至 {dates[-1]} ({len(dates)} 天)")

# Test simple 2-factor combo
test_combo = ["ADX_14D", "MAX_DD_60D"]
combo_indices = [factor_names_list.index(f) for f in test_combo]

all_factors_stack_full = np.stack(
    [std_factors[f].values for f in factor_names_list], axis=-1
)
all_factors_stack = all_factors_stack_full[holdout_mask]

print(f"\n因子数据形状: {all_factors_stack.shape}")
print(f"包含 NaN: {np.isnan(all_factors_stack).any()}")

# Price data
close_prices = ohlcv["close"][etf_codes].ffill().bfill().values[holdout_mask]
open_prices = ohlcv["open"][etf_codes].ffill().bfill().values[holdout_mask]
high_prices = ohlcv["high"][etf_codes].ffill().bfill().values[holdout_mask]
low_prices = ohlcv["low"][etf_codes].ffill().bfill().values[holdout_mask]

print(f"价格数据形状: {close_prices.shape}")

# Timing signal
timing_module = LightTimingModule(
    extreme_threshold=EXTREME_THRESHOLD,
    extreme_position=EXTREME_POSITION,
)
timing_series_full = timing_module.compute_position_ratios(ohlcv["close"])
timing_arr_raw_full = timing_series_full.reindex(all_dates).fillna(1.0).values
timing_arr_full = shift_timing_signal(timing_arr_raw_full)
timing_arr = timing_arr_full[holdout_mask]

print(f"\n择时信号统计:")
print(f"  平均: {timing_arr.mean():.2%}")
print(f"  最小: {timing_arr.min():.2%}")
print(f"  最大: {timing_arr.max():.2%}")
print(f"  满仓天数: {(timing_arr >= 0.95).sum()} / {len(timing_arr)}")

# Generate rebalance schedule
rebalance_dates = generate_rebalance_schedule(dates, freq=FREQ, lookback_window=252)
print(f"\n调仓日程:")
print(f"  总调仓次数: {len(rebalance_dates)}")
if len(rebalance_dates) > 0:
    print(f"  前5次: {rebalance_dates[:min(5, len(rebalance_dates))]}")
else:
    print("  ⚠️ 没有生成任何调仓日！")

# Check factor values on rebalance dates
print(f"\n前3个调仓日的因子值（ADX_14D）:")
for rb_date in rebalance_dates[:3]:
    rb_idx = np.where(dates == rb_date)[0]
    if len(rb_idx) > 0:
        idx = rb_idx[0]
        factor_vals = all_factors_stack[idx, :, combo_indices[0]]
        print(
            f"  {rb_date}: 非NaN数量={(~np.isnan(factor_vals)).sum()}/43, 均值={np.nanmean(factor_vals):.4f}"
        )

# Run backtest
current_factors = all_factors_stack[..., combo_indices]
current_factor_indices = list(range(len(combo_indices)))

pnl, ret, wr, pf, trades, holdings_hist, risk = run_vec_backtest(
    current_factors,
    close_prices,
    open_prices,
    high_prices,
    low_prices,
    timing_arr,
    current_factor_indices,
    freq=FREQ,
    pos_size=POS_SIZE,
    initial_capital=1000000.0,
    commission_rate=0.0002,
    lookback=252,
    trailing_stop_pct=0.0,
    stop_on_rebalance_only=True,
)

print(f"\n回测结果:")
print(f"  收益率: {ret:.2%}")
print(f"  交易次数: {trades}")
print(f"  最大回撤: {risk['max_drawdown']:.2%}")
print(f"  Calmar: {risk['calmar_ratio']:.3f}")

if trades == 0:
    print("\n⚠️ 没有交易发生！可能原因:")
    print("  1. 所有因子值都是 NaN")
    print("  2. 择时信号导致仓位过低")
    print("  3. 调仓日期计算错误")
    print(f"\n  Holdings历史长度: {len(holdings_hist)}")
    if len(holdings_hist) > 0:
        print(f"  持仓非零次数: {sum([len(h) > 0 for h in holdings_hist])}")
