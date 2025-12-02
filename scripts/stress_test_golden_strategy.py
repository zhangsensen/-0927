#!/usr/bin/env python3
"""
🛡️ Golden Strategy Stress Test
================================================================================
Performs rigorous stress testing on the "Golden Strategy" to detect overfitting.

Tests:
1. Parameter Sensitivity (FREQ, POS)
2. Cost Sensitivity (Commission)
3. Universe Sensitivity (Removing key assets)
4. Robustness Check (Noise injection - optional, skipped for speed)

Target Strategy:
ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal
from scripts.batch_vec_backtest import run_vec_backtest

# 🏆 The Golden Strategy
GOLDEN_COMBO = "ADX_14D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D + SHARPE_RATIO_20D"

def load_config():
    with open(ROOT / "configs/combo_wfo_config.yaml") as f:
        return yaml.safe_load(f)

def main():
    print("🛡️  STARTING GOLDEN STRATEGY STRESS TEST")
    print("=" * 60)
    print(f"Target Strategy: {GOLDEN_COMBO}")
    
    # 1. Load Data & Config
    config = load_config()
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"],
    )
    # Load full universe
    full_symbols = config["data"]["symbols"]
    ohlcv = loader.load_ohlcv(
        etf_codes=full_symbols,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 2. Compute Factors
    print("Computing factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    # Process Cross Section
    factor_names_list = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # Prepare 3D Matrix
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    
    # Prepare Prices
    close_prices = ohlcv["close"][etf_codes].ffill().bfill().values
    open_prices = ohlcv["open"][etf_codes].ffill().bfill().values
    high_prices = ohlcv["high"][etf_codes].ffill().bfill().values
    low_prices = ohlcv["low"][etf_codes].ffill().bfill().values
    
    # Prepare Timing
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1) # Using v3.0 params
    timing_arr_raw = timing_module.compute_position_ratios(ohlcv["close"]).reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)
    
    # Prepare Factor Indices
    factor_index_map = {name: idx for idx, name in enumerate(factor_names)}
    target_factors = [f.strip() for f in GOLDEN_COMBO.split(" + ")]
    factor_indices = [factor_index_map[f] for f in target_factors]
    
    # 3. Define Scenarios
    base_params = {
        "freq": 3,
        "pos_size": 2,
        "initial_capital": 1_000_000,
        "commission_rate": 0.0002,
        "lookback": 252,
        "stop_on_rebalance_only": True, # v3.0 default
        "trailing_stop_pct": 0.0,       # v3.0 default (no stop)
    }
    
    scenarios = []
    
    # Baseline
    scenarios.append(("Baseline (v3.0)", base_params.copy(), None))
    
    # Freq Sensitivity
    for f in [1, 2, 4, 5, 8, 10]:
        p = base_params.copy()
        p["freq"] = f
        scenarios.append((f"Freq={f}", p, None))
        
    # Pos Sensitivity
    for pos in [1, 3, 4, 5]:
        p = base_params.copy()
        p["pos_size"] = pos
        scenarios.append((f"Pos={pos}", p, None))
        
    # Cost Sensitivity
    for comm in [0.0005, 0.0010, 0.0020]:
        p = base_params.copy()
        p["commission_rate"] = comm
        scenarios.append((f"Comm={comm*10000:.0f}bp", p, None))
        
    # Universe Sensitivity (The "Kill" Tests)
    # Need to map symbol to index to mask it
    symbol_to_idx = {s: i for i, s in enumerate(etf_codes)}
    
    # Kill S&P 500 (513500)
    if "513500" in symbol_to_idx:
        mask_sp500 = np.ones(len(etf_codes), dtype=bool)
        mask_sp500[symbol_to_idx["513500"]] = False
        scenarios.append(("No S&P500 (513500)", base_params.copy(), mask_sp500))
        
    # Kill Nasdaq (513100)
    if "513100" in symbol_to_idx:
        mask_nasdaq = np.ones(len(etf_codes), dtype=bool)
        mask_nasdaq[symbol_to_idx["513100"]] = False
        scenarios.append(("No Nasdaq (513100)", base_params.copy(), mask_nasdaq))
        
    # Kill All QDII
    qdii_list = ["513050", "513100", "513130", "159920", "513500"]
    mask_no_qdii = np.ones(len(etf_codes), dtype=bool)
    for q in qdii_list:
        if q in symbol_to_idx:
            mask_no_qdii[symbol_to_idx[q]] = False
    scenarios.append(("No QDII (A-Share Only)", base_params.copy(), mask_no_qdii))

    # 4. Run Tests
    results = []
    print("\nRunning Scenarios...")
    
    for name, params, mask in tqdm(scenarios):
        # Apply mask if needed
        # To mask, we can set factors to NaN for those assets
        # But factors_3d is shared. We need a copy or a way to mask inside run_vec_backtest
        # run_vec_backtest doesn't support masking.
        # Hack: Modify factors_3d temporarily? No, parallel issues if we were parallel.
        # Here we are serial.
        
        current_factors = factors_3d.copy()
        if mask is not None:
            # Set all factors for masked assets to NaN
            # mask is True for keep, False for remove
            # So we want where mask is False
            remove_indices = np.where(~mask)[0]
            current_factors[:, remove_indices, :] = np.nan
            
        ret, wr, pf, trades, _, risk = run_vec_backtest(
            current_factors, close_prices, open_prices, high_prices, low_prices, timing_arr, factor_indices,
            **params
        )
        
        results.append({
            "Scenario": name,
            "Return": ret * 100,
            "MaxDD": risk["max_drawdown"] * 100,
            "Sharpe": risk["sharpe_ratio"],
            "Calmar": risk["calmar_ratio"],
            "Trades": trades
        })

    # 5. Report
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 STRESS TEST RESULTS")
    print("="*80)
    print(df_res.to_markdown(index=False, floatfmt=".2f"))
    print("="*80)
    
    # Analysis
    baseline = df_res.iloc[0]
    print("\n🔍 ANALYSIS:")
    
    # Freq Check
    freq_res = df_res[df_res["Scenario"].str.startswith("Freq")]
    best_freq = freq_res.loc[freq_res["Return"].idxmax()]
    print(f"1. Frequency: Baseline (3) vs Best ({best_freq['Scenario']})")
    if best_freq["Return"] > baseline["Return"] * 1.1:
        print("   ⚠️  Warning: FREQ=3 is NOT the peak. Optimization might be unstable.")
    else:
        print("   ✅ FREQ=3 is close to optimal or optimal.")
        
    # QDII Check
    no_qdii = df_res[df_res["Scenario"] == "No QDII (A-Share Only)"].iloc[0]
    drop = baseline["Return"] - no_qdii["Return"]
    print(f"2. QDII Dependency: {drop:.1f}pp drop without QDII.")
    if no_qdii["Return"] < 50:
        print("   ⚠️  CRITICAL: Strategy fails without QDII. Purely a beta play on US Tech?")
    else:
        print(f"   ✅ Strategy survives without QDII ({no_qdii['Return']:.1f}%), showing alpha exists in A-shares too.")

if __name__ == "__main__":
    main()
