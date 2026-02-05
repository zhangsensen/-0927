#!/usr/bin/env python3
"""
Analyze trade distribution for the Core Strategy to explain Win Rate vs Profit Factor.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import backtrader as bt
import yaml

# Add sealed source to path
SEALED_ROOT = Path(
    "/home/sensen/dev/projects/-0927/sealed_strategies/v3.2_20251214/locked"
)
sys.path.insert(0, str(SEALED_ROOT))
sys.path.insert(0, str(SEALED_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import (
    shift_timing_signal,
    generate_rebalance_schedule,
)
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData


def analyze_distribution(combo_str):
    print(f"Analyzing Combo: {combo_str}")

    # 1. Load Config & Data
    config_path = SEALED_ROOT / "configs/combo_wfo_config.yaml"
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

    # 3. Prepare Strategy Inputs
    factors = [f.strip() for f in combo_str.split(" + ")]
    combined_score_df = pd.DataFrame(0.0, index=dates, columns=etf_codes)
    for f in factors:
        combined_score_df = combined_score_df.add(std_factors[f], fill_value=0)

    # Timing & Vol Regime
    timing_config = config.get("backtest", {}).get("timing", {})
    timing_module = LightTimingModule(
        extreme_threshold=timing_config.get("extreme_threshold", -0.4),
        extreme_position=timing_config.get("extreme_position", 0.3),
    )
    timing_series_raw = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_shifted = shift_timing_signal(
        timing_series_raw.reindex(dates).fillna(1.0).values
    )
    timing_series = pd.Series(timing_arr_shifted, index=dates)

    if "510300" in ohlcv["close"].columns:
        hs300 = ohlcv["close"]["510300"]
    else:
        hs300 = ohlcv["close"].iloc[:, 0]
    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    exposure_s = pd.Series(1.0, index=regime_vol.index)
    exposure_s[regime_vol >= 25] = 0.7
    exposure_s[regime_vol >= 30] = 0.4
    exposure_s[regime_vol >= 40] = 0.1
    vol_regime_series = exposure_s.reindex(dates).fillna(1.0)

    # Rebalance Schedule
    backtest_config = config.get("backtest", {})
    rebalance_schedule = generate_rebalance_schedule(
        total_periods=len(dates),
        lookback_window=backtest_config.get("lookback", 252),
        freq=backtest_config.get("freq", 3),
    )

    # 4. Run Backtest
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(backtest_config.get("initial_capital", 1_000_000.0)))
    cerebro.broker.setcommission(
        commission=float(backtest_config.get("commission_rate", 0.0002))
    )
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)

    for ticker in etf_codes:
        df = (
            pd.DataFrame(
                {
                    "open": ohlcv["open"][ticker],
                    "high": ohlcv["high"][ticker],
                    "low": ohlcv["low"][ticker],
                    "close": ohlcv["close"][ticker],
                    "volume": ohlcv["volume"][ticker],
                }
            )
            .reindex(dates)
            .ffill()
            .fillna(0.01)
        )
        cerebro.adddata(PandasData(dataname=df, name=ticker))

    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score_df,
        timing=timing_series,
        vol_regime=vol_regime_series,
        etf_codes=etf_codes,
        freq=backtest_config.get("freq", 3),
        pos_size=backtest_config.get("pos_size", 2),
        rebalance_schedule=rebalance_schedule,
        target_vol=0.20,
        vol_window=20,
        dynamic_leverage_enabled=True,
    )

    results = cerebro.run()
    strat = results[0]

    # 5. Analyze Trades using Custom Trade List
    # GenericStrategy populates self.trades with pre-calculated return_pct and cost

    trade_returns = []
    winning_trades = []
    losing_trades = []

    print(f"DEBUG: Strategy has {len(strat.trades)} trades in custom list.")

    for t in strat.trades:
        pnl = t["pnlcomm"]
        pct = t["return_pct"]

        trade_returns.append(pct)

        if pnl > 0:
            winning_trades.append(pct)
        else:
            losing_trades.append(pct)

    # Calculate statistics
    total_trades = len(trade_returns)
    if total_trades == 0:
        print("No trades found.")
        return

    win_rate = len(winning_trades) / total_trades
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0

    # Payoff Ratio (Avg Win / |Avg Loss|)
    payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0

    # Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
    # Note: Avg Loss is negative, so we add it.
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    print("\n📊 Trade Distribution Analysis")
    print(f"Total Trades: {total_trades}")
    print("\n--- Win/Loss Stats ---")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Avg Win:  {avg_win:.2%}")
    print(f"Avg Loss: {avg_loss:.2%}")
    print(f"Payoff Ratio (Avg Win / |Avg Loss|): {payoff_ratio:.2f}")

    print("\n--- Extremes ---")
    print(f"Max Win:  {np.max(trade_returns):.2%}")
    print(f"Max Loss: {np.min(trade_returns):.2%}")

    print("\n--- Expectancy ---")
    print(f"Expectancy per Trade: {expectancy:.2%}")

    # Histogram
    print("\n--- Distribution Histogram ---")
    bins = [-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    counts, _ = np.histogram(trade_returns, bins=bins)

    for i in range(len(counts)):
        lower = bins[i]
        upper = bins[i + 1]
        count = counts[i]
        bar = "#" * count
        print(f"({lower:>5}, {upper:>5}] : {count:>3} {bar}")


if __name__ == "__main__":
    core_combo = "CORRELATION_TO_MARKET_20D + MOM_20D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D + VOL_RATIO_20D"
    analyze_distribution(core_combo)
