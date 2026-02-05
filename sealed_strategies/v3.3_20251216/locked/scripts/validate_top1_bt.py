import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

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


class V3Strategy(bt.Strategy):
    params = (
        ("scores", None),  # DataFrame of scores
        ("vol_regime", None),  # Series of exposure (1.0, 0.7, 0.4, 0.1)
        ("timing", None),  # Series of timing (1.0 or 0.1)
        ("rebalance_schedule", None),
        ("pos_size", 2),
    )

    def __init__(self):
        self.rebalance_set = set(self.params.rebalance_schedule)
        self.inds = {}
        for d in self.datas:
            self.inds[d._name] = d.close

    def next(self):
        dt = self.datas[0].datetime.date(0)

        # Check Rebalance
        if dt not in self.rebalance_set:
            return

        # Get Exposure
        try:
            exposure = self.params.vol_regime.loc[pd.Timestamp(dt)]
            timing = self.params.timing.loc[pd.Timestamp(dt)]
        except:
            exposure = 1.0
            timing = 1.0

        target_exposure = exposure * timing

        # Get Scores
        try:
            current_scores = self.params.scores.loc[pd.Timestamp(dt)]
        except:
            return

        # Select Top N
        valid_scores = current_scores.dropna().sort_values(ascending=False)
        targets = valid_scores.head(self.params.pos_size).index.tolist()

        # Execute
        # 1. Sell non-targets
        for d in self.datas:
            name = d._name
            pos = self.getposition(d).size
            if pos > 0 and name not in targets:
                self.order_target_percent(d, target=0.0)

        # 2. Buy targets
        if not targets:
            return

        weight = target_exposure / len(targets)
        for name in targets:
            d = self.getdatabyname(name)
            self.order_target_percent(d, target=weight)


def main():
    print("🚀 Starting Backtrader Validation...")

    # Load Data & Config
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

    # Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)

    combo_str = "ADX_14D + PRICE_POSITION_20D + SLOPE_20D + VOL_RATIO_20D + VORTEX_14D"
    factors = [f.strip() for f in combo_str.split("+")]

    # Sum Scores
    scores = pd.DataFrame(
        0.0, index=raw_factors_df.index, columns=raw_factors_df.columns.levels[1]
    )
    processor = CrossSectionProcessor(verbose=False)

    # Process each factor and sum
    # Note: We need to process them individually to get standardized scores
    # But CrossSectionProcessor processes a dict of all factors.
    # Let's do it properly.
    raw_factors_dict = {f: raw_factors_df[f] for f in factors}
    std_factors = processor.process_all_factors(raw_factors_dict)

    for f in factors:
        scores += std_factors[f]

    # Vol Regime
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

    # Timing
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(ohlcv["close"])
    timing_s = shift_timing_signal(timing_signals)
    timing_s = pd.Series(timing_s, index=ohlcv["close"].index)

    # Rebalance Schedule
    dates = ohlcv["close"].index
    T = len(dates)
    sched_indices = generate_rebalance_schedule(T, 252, 3)
    sched_dates = [dates[i].date() for i in sched_indices]

    # Backtrader Setup
    cerebro = bt.Cerebro()

    # Add Data
    # Fill NaNs to match VEC
    close_df = ohlcv["close"].ffill().bfill()
    open_df = ohlcv["open"].ffill().bfill()
    high_df = ohlcv["high"].ffill().bfill()
    low_df = ohlcv["low"].ffill().bfill()
    vol_df = ohlcv["volume"].fillna(0)

    for col in ohlcv["close"].columns:
        df = pd.DataFrame(
            {
                "open": open_df[col],
                "high": high_df[col],
                "low": low_df[col],
                "close": close_df[col],
                "volume": vol_df[col],
            }
        )
        data = bt.feeds.PandasData(dataname=df, name=col)
        cerebro.adddata(data)

    cerebro.addstrategy(
        V3Strategy,
        scores=scores,
        vol_regime=exposure_s,
        timing=timing_s,
        rebalance_schedule=sched_dates,
        pos_size=2,
    )

    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=0.0002)
    cerebro.broker.set_checksubmit(False)  # Cheat-On-Close
    cerebro.broker.set_coc(True)  # Enable Cheat-On-Close execution

    print("Running Backtrader...")
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    print(f"Final Value: {final_value:.2f}")

    # Calculate Metrics
    # We need daily values
    # BT doesn't give easy daily values unless we add an analyzer
    # But we can just use final value for basic check

    ret = final_value / 1_000_000.0 - 1
    print(f"Total Return: {ret:.2%}")


if __name__ == "__main__":
    main()
