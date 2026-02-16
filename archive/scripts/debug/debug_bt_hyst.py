#!/usr/bin/env python3
"""Debug: run S1_F5_ON BT with hysteresis diagnostics."""
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["FROZEN_PARAMS_MODE"] = "warn"

import yaml, numpy as np, pandas as pd
import backtrader as bt

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_cache import FactorCache
from etf_strategy.core.frozen_params import FrozenETFPool
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.cost_model import load_cost_model
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.regime_gate import compute_regime_gate_arr
from etf_strategy.core.execution_model import load_execution_model
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData

config_path = ROOT / "configs" / "combo_wfo_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

exec_model = load_execution_model(config)
USE_T1_OPEN = exec_model.is_t1_open
print(f"Execution: {exec_model.mode}, T1_OPEN={USE_T1_OPEN}")

loader = DataLoader(data_dir=config["data"].get("data_dir"), cache_dir=config["data"].get("cache_dir"))
ohlcv = loader.load_ohlcv(etf_codes=config["data"]["symbols"], start_date=config["data"]["start_date"], end_date=config["data"]["end_date"])
factor_cache = FactorCache(cache_dir=Path(config["data"].get("cache_dir") or ".cache"))
cached = factor_cache.get_or_compute(ohlcv=ohlcv, config=config, data_dir=loader.data_dir)
std_factors = cached["std_factors"]
first_factor = std_factors[sorted(std_factors.keys())[0]]
dates = first_factor.index
etf_codes = first_factor.columns.tolist()

print(f"OBV_SLOPE_10D in std_factors: {'OBV_SLOPE_10D' in std_factors}")
print(f"Factors available: {len(std_factors)}")

backtest_config = config.get("backtest", {})
cost_model = load_cost_model(config)
qdii_codes = set(FrozenETFPool().qdii_codes)

# Timing
timing_config = backtest_config.get("timing", {})
timing_module = LightTimingModule(
    extreme_threshold=timing_config.get("extreme_threshold", -0.4),
    extreme_position=timing_config.get("extreme_position", 0.3),
)
timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates).fillna(1.0).values)
timing_series = pd.Series(timing_arr_shifted, index=dates)
gate_arr = compute_regime_gate_arr(ohlcv["close"], dates, backtest_config=backtest_config)
timing_series = timing_series * pd.Series(gate_arr, index=dates)

# Vol regime
hs300 = ohlcv["close"]["510300"]
rets = hs300.pct_change()
hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
hv_5d = hv.shift(5)
regime_vol = (hv + hv_5d) / 2
exposure_s = pd.Series(1.0, index=regime_vol.index)
exposure_s[regime_vol >= 25] = 0.7
exposure_s[regime_vol >= 30] = 0.4
exposure_s[regime_vol >= 40] = 0.1
vol_regime_series = exposure_s.reindex(dates).fillna(1.0)

# S1 scores
combo_str = "ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D"
factors = [f.strip() for f in combo_str.split(" + ")]
combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
for f_name in factors:
    combined_score_df = combined_score_df.add(std_factors[f_name], fill_value=0)

# Quick score check
test_date = dates[252]
row = combined_score_df.loc[test_date]
valid = row[row.notna() & (row != 0)]
print(f"\nDate {test_date}: {len(valid)}/{len(row)} valid scores")
print(f"Top 5 scores: {valid.sort_values(ascending=False).head(5).to_dict()}")

# Rebalance schedule
sched = generate_rebalance_schedule(total_periods=len(dates), lookback_window=252, freq=5)
print(f"\nRebalance schedule: {len(sched)} events, first={sched[0]}, last={sched[-1]}")

# Data feeds
data_feeds = {}
for ticker in etf_codes:
    df = pd.DataFrame({
        "open": ohlcv["open"][ticker], "high": ohlcv["high"][ticker],
        "low": ohlcv["low"][ticker], "close": ohlcv["close"][ticker],
        "volume": ohlcv["volume"][ticker],
    }).reindex(dates).ffill().fillna(0.01)
    data_feeds[ticker] = df

# Run BT
cerebro = bt.Cerebro(cheat_on_open=USE_T1_OPEN)
cerebro.broker.setcash(1_000_000.0)
for ticker in data_feeds:
    rate = cost_model.get_cost(ticker, qdii_codes)
    cerebro.broker.setcommission(commission=rate, name=ticker, leverage=1.0)
if USE_T1_OPEN:
    cerebro.broker.set_coc(False)
    cerebro.broker.set_coo(True)
else:
    cerebro.broker.set_coc(True)
cerebro.broker.set_checksubmit(False)

for ticker, df in data_feeds.items():
    data = PandasData(dataname=df, name=ticker)
    cerebro.adddata(data)

max_comm = max(cost_model.active_tier.a_share, cost_model.active_tier.qdii)
print(f"Max commission rate for sizing: {max_comm*10000:.0f}bp")

cerebro.addstrategy(
    GenericStrategy,
    scores=combined_score_df, timing=timing_series,
    vol_regime=vol_regime_series, etf_codes=etf_codes,
    freq=5, pos_size=2, rebalance_schedule=sched,
    target_vol=0.20, vol_window=20, dynamic_leverage_enabled=True,
    use_t1_open=USE_T1_OPEN,
    delta_rank=0.10, min_hold_days=9,
    sizing_commission_rate=max_comm,
)

cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
print("\nRunning BT (suppressing BT TRADE prints)...")

# Suppress trade prints temporarily
import io
old_stdout = sys.stdout
sys.stdout = io.StringIO()
results = cerebro.run()
output = sys.stdout.getvalue()
sys.stdout = old_stdout

# Extract HYST debug lines and BT TRADE lines
hyst_lines = [l for l in output.split("\n") if "HYST" in l]
trade_lines = [l for l in output.split("\n") if "BT TRADE" in l]

print(f"\nHYST debug lines: {len(hyst_lines)}")
for line in hyst_lines[:20]:
    print(line)
if len(hyst_lines) > 20:
    print(f"... ({len(hyst_lines)} total)")

print(f"\nBT TRADE lines: {len(trade_lines)}")
for line in trade_lines[:10]:
    print(line)

strat = results[0]
ta = strat.analyzers.trades.get_analysis()
print(f"\nTotal round-trips: {ta.get('total', {}).get('total', 0)}")
print(f"Margin failures: {strat.margin_failures}")
print(f"Hyst debug count: {getattr(strat, '_hyst_debug_count', 'N/A')}")
