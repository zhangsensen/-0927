#!/usr/bin/env python3
"""
Rebalance Frequency Sweep for Standalone Strategy
Runs the strategy with rebalance frequencies from 1 to 30 days to find the optimal setting.
"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
    print("🔄 ETF Rotation Strategy - Rebalance Frequency Sweep (1-30 Days)")
    print("=" * 80)

    # 1. Load Configuration
    config_path = Path(__file__).parent / "configs/config.yaml"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    print("\n[Step 1] Loading Data...")
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
        risk_off_prices = ohlcv["close"][risk_off_asset].ffill().bfill().values
        rotation_symbols = [s for s in ohlcv["close"].columns if s != risk_off_asset]
        ohlcv_rotation = {col: df[rotation_symbols] for col, df in ohlcv.items()}
    else:
        risk_off_prices = None
        ohlcv_rotation = ohlcv

    # 3. Compute Factors
    print("\n[Step 2] Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_rotation)
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    # 4. Process Factors
    print("\n[Step 3] Processing Factors...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = processor.process_all_factors(raw_factors)

    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv_rotation["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv_rotation["open"][etf_codes].ffill().bfill().values

    # 5. Compute Timing Signal
    print("\n[Step 4] Computing Timing Signal...")
    timing_module = LightTimingModule(
        extreme_threshold=config["backtest"]["timing"]["extreme_threshold"],
        extreme_position=config["backtest"]["timing"]["extreme_position"]
    )
    timing_series = timing_module.compute_position_ratios(ohlcv["close"])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 6. Run Sweep
    print("\n[Step 5] Running Frequency Sweep (1-30 Days)...")
    
    selected_factors = config.get("strategy", {}).get("selected_factors", [])
    if not selected_factors:
        selected_factors = factor_names
    
    print(f"   Factors: {', '.join(selected_factors)}")
    factor_indices = [factor_names.index(f) for f in selected_factors]
    
    results = []
    
    for freq in tqdm(range(1, 31), desc="Sweeping Frequencies"):
        ret, wr, pf, trades = run_vec_backtest(
            factors_3d=factors_3d,
            close_prices=close_prices,
            open_prices=open_prices,
            timing_arr=timing_arr,
            factor_indices=factor_indices,
            risk_off_prices=risk_off_prices,
            freq=freq,
            pos_size=config["backtest"]["position_size"],
            initial_capital=config["backtest"]["initial_capital"],
            commission_rate=config["backtest"]["commission_rate"],
            lookback=config["backtest"]["lookback_window"]
        )
        
        # Calculate Annualized Return (approximate)
        # Assuming ~4.75 years of data (2020-01-01 to 2025-10-14)
        # Precise calculation would require days count, but total return is fine for comparison
        
        results.append({
            "Frequency": freq,
            "Total Return": ret,
            "Win Rate": wr,
            "Profit Factor": pf,
            "Trades": trades
        })

    # 7. Display Results
    df_results = pd.DataFrame(results)
    df_results["Total Return %"] = df_results["Total Return"] * 100
    df_results["Win Rate %"] = df_results["Win Rate"] * 100
    
    # Sort by Total Return
    df_sorted = df_results.sort_values("Total Return", ascending=False)
    
    print("\n" + "=" * 60)
    print("🏆 Top 10 Frequencies by Total Return")
    print("=" * 60)
    print(df_sorted[["Frequency", "Total Return %", "Win Rate %", "Profit Factor", "Trades"]].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("📉 All Frequencies (1-30)")
    print("=" * 60)
    print(df_results[["Frequency", "Total Return %", "Win Rate %", "Profit Factor", "Trades"]].to_string(index=False))

    # Save to CSV
    output_dir = Path(__file__).parent / config["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"freq_sweep_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Detailed results saved to {csv_path}")

if __name__ == "__main__":
    main()
