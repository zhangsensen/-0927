
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from scripts.batch_vec_backtest import run_vec_backtest

def main():
    # Configuration
    FREQ = 3
    POS_SIZE = 2
    INITIAL_CAPITAL = 1_000_000.0
    COMMISSION_RATE = 0.0002
    LOOKBACK = 252
    
    # Timing
    EXTREME_THRESHOLD = -0.1
    EXTREME_POSITION = 0.1
    
    # Combo
    COMBO = "ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D"
    FACTORS = [f.strip() for f in COMBO.split(" + ")]
    
    print(f"🔍 Debugging Combo: {COMBO}")
    print(f"⚙️ Params: FREQ={FREQ}, POS={POS_SIZE}, Timing=({EXTREME_THRESHOLD}, {EXTREME_POSITION})")
    
    # Load Data
    config_path = Path("configs/combo_wfo_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )
    # Force using the original 43 ETFs (exclude new ones if any)
    # The config might have new ones, let's filter them if needed.
    # But for now let's use what's in config, assuming the 237% run used the config at that time.
    # Wait, the 237% run was on 2025-12-01, which is TODAY (in simulation time).
    # The config currently has 43 + 3 = 46 ETFs.
    # If the 237% run was on 43 ETFs, I should restrict it.
    # The AGENTS.md says "43 ETF 高频轮动策略".
    # So I should probably use the original 43 list.
    
    # Let's load the original 43 list from a file if possible, or just hardcode the exclusion of the 3 new ones.
    all_symbols = config["data"]["symbols"]
    new_etfs = ['513180', '513400', '513520']
    symbols = [s for s in all_symbols if s not in new_etfs]
    
    print(f"📊 Universe: {len(symbols)} ETFs (Original 43 - Baseline Verification)")
    
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # Compute Factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    raw_factors = {fname: raw_factors_df[fname] for fname in raw_factors_df.columns.get_level_values(0).unique()}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values
    
    # Timing
    timing_module = LightTimingModule(
        extreme_threshold=EXTREME_THRESHOLD,
        extreme_position=EXTREME_POSITION,
    )
    timing_arr_raw = timing_module.compute_position_ratios(ohlcv["close"][etf_codes]).reindex(dates).fillna(1.0).values
    # Note: compute_position_ratios returns a Series if input is DataFrame? 
    # Wait, LightTimingModule.compute_position_ratios takes `close_prices` (DataFrame) and returns `Series` (market avg based).
    # Let's check implementation.
    
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    # Indices
    factor_indices = [factor_names.index(f) for f in FACTORS]
    
    # Run VEC
    ret, wr, pf, trades, rounding, risk = run_vec_backtest(
        factors_3d, close_prices, open_prices, high_prices, low_prices, timing_arr, factor_indices,
        freq=FREQ,
        pos_size=POS_SIZE,
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        lookback=LOOKBACK,
        target_vol=0.20, vol_window=20, dynamic_leverage_enabled=False,
        use_atr_stop=False, trailing_stop_pct=0.0, atr_arr=None, atr_multiplier=3.0, stop_on_rebalance_only=False,
        individual_trend_arr=None, individual_trend_enabled=False,
        profit_ladders=[],
        circuit_breaker_day=0.0, circuit_breaker_total=0.0, circuit_recovery_days=5,
        cooldown_days=0,
        leverage_cap=1.0
    )
    
    print("\n" + "="*40)
    print(f"🏆 Result: {ret*100:.2f}%")
    print(f"📉 MaxDD: {risk['max_drawdown']*100:.2f}%")
    print(f"📈 Sharpe: {risk['sharpe_ratio']:.4f}")
    print(f"📅 Trades: {trades}")
    print("="*40)

if __name__ == "__main__":
    main()
