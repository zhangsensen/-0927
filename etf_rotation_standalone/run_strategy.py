#!/usr/bin/env python3
"""
Standalone Strategy Runner
Runs the 18-factor ETF rotation strategy using the vectorized engine.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add current directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.precise_factor_library import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.market_timing import LightTimingModule
from core.utils.rebalance import shift_timing_signal
from core.backtester_vectorized import run_vec_backtest

def main():
    print("=" * 80)
    print("🚀 ETF Rotation Strategy - Standalone Runner")
    print("=" * 80)

    # 1. Load Configuration
    config_path = Path(__file__).parent / "configs/config.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration loaded from {config_path}")

    # 2. Load Data
    print("\n[Step 1] Loading Data...")
    
    # Ensure Risk-Off Asset (Gold: 518880) is loaded
    risk_off_asset = "518880"
    symbols = config["data"]["symbols"]
    if risk_off_asset not in symbols:
        symbols.append(risk_off_asset)
        
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # Extract Risk-Off Asset Prices
    if risk_off_asset in ohlcv["close"].columns:
        print(f"   ✅ Risk-Off Asset ({risk_off_asset}) loaded.")
        risk_off_prices = ohlcv["close"][risk_off_asset].ffill().bfill().values
        
        # Remove Risk-Off Asset from Rotation Pool
        # We don't want to select Gold as part of the "Equity" rotation, 
        # it's reserved for the Risk-Off bucket.
        rotation_symbols = [s for s in ohlcv["close"].columns if s != risk_off_asset]
        
        # Filter OHLCV for rotation
        ohlcv_rotation = {
            col: df[rotation_symbols] for col, df in ohlcv.items()
        }
    else:
        print(f"   ⚠️ Risk-Off Asset ({risk_off_asset}) NOT found! Fallback to Cash.")
        risk_off_prices = None
        ohlcv_rotation = ohlcv

    # 3. Compute Factors
    print("\n[Step 2] Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_rotation)

    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    
    print(f"   Computed {len(factor_names_list)} factors: {', '.join(factor_names_list)}")

    # 4. Process Factors (Standardization & Winsorization)
    print("\n[Step 3] Processing Factors...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=True
    )
    std_factors = processor.process_all_factors(raw_factors)

    # Prepare 3D factor array
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    T, N = first_factor.shape

    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    
    # Prepare prices (ffill/bfill for backtest continuity)
    close_prices = ohlcv_rotation["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv_rotation["open"][etf_codes].ffill().bfill().values

    # 5. Compute Timing Signal
    print("\n[Step 4] Computing Timing Signal...")
    # Use the Broad Market Index (usually 510300 or similar) for timing, 
    # or the average of the pool. The original code used ohlcv["close"].
    # We should use the rotation pool for timing signal generation to be consistent.
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"]
    )
    # Pass FULL ohlcv["close"] so it can find Gold (518880) for the signal
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    # Shift timing signal to avoid lookahead
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    print(f"   Timing signal computed. Average position ratio: {timing_arr.mean():.2f}")

    # 6. Run Backtest
    print("\n[Step 5] Running Vectorized Backtest (RORO Mode)...")
    
    # Determine factors to use
    all_factor_names = sorted(std_factors.keys())
    selected_factors = config.get("strategy", {}).get("selected_factors", [])
    
    if not selected_factors:
        print("   ℹ️ No specific factors selected in config. Using ALL available factors.")
        selected_factors = all_factor_names
    else:
        # Validate factors
        missing = [f for f in selected_factors if f not in all_factor_names]
        if missing:
            print(f"   ⚠️ Warning: The following selected factors were not found: {missing}")
            selected_factors = [f for f in selected_factors if f in all_factor_names]
            
    if not selected_factors:
        print("   ❌ No valid factors selected! Exiting.")
        return

    factor_indices = [all_factor_names.index(f) for f in selected_factors]
    
    print(f"   Strategy: Equal-weight combination of {len(selected_factors)} factors")
    print(f"   Factors: {', '.join(selected_factors)}")
    
    ret, wr, pf, trades = run_vec_backtest(
        factors_3d=factors_3d,
        close_prices=close_prices,
        open_prices=open_prices,
        timing_arr=timing_arr,
        factor_indices=factor_indices,
        risk_off_prices=risk_off_prices, # Pass Gold prices
        freq=config["backtest"]["rebalance_frequency"],
        pos_size=config["backtest"]["position_size"],
        initial_capital=config["backtest"]["initial_capital"],
        commission_rate=config["backtest"]["commission_rate"],
        lookback=config["backtest"]["lookback_window"]
    )

    # 7. Report Results
    print("\n" + "=" * 40)
    print("📊 Backtest Results")
    print("=" * 40)
    print(f"Total Return:    {ret * 100:.2f}%")
    print(f"Win Rate:        {wr * 100:.2f}%")
    print(f"Profit Factor:   {pf:.2f}")
    print(f"Total Trades:    {trades}")
    print("=" * 40)
    
    # Save results
    output_dir = Path(__file__).parent / config["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"result_{timestamp}.txt"
    
    with open(result_file, "w") as f:
        f.write(f"Run Date: {timestamp}\n")
        f.write(f"Factors: {', '.join(selected_factors)}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Return:    {ret * 100:.2f}%\n")
        f.write(f"Win Rate:        {wr * 100:.2f}%\n")
        f.write(f"Profit Factor:   {pf:.2f}\n")
        f.write(f"Total Trades:    {trades}\n")
        
    print(f"\n✅ Results saved to {result_file}")

if __name__ == "__main__":
    main()
