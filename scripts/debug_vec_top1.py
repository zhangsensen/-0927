
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule

# Top 1 Strategy
COMBO = "ADX_14D + CMF_20D + OBV_SLOPE_10D + RELATIVE_STRENGTH_VS_MARKET_20D + VORTEX_14D"
HOLDOUT_START = "2025-06-01"
HOLDOUT_END = "2025-12-08"

def run_vec_debug(factors_3d, close_prices, timing_arr, factor_indices, freq, pos_size, lookback_window, dates, etf_codes):
    n_days, n_etfs = close_prices.shape
    rebalance_schedule = generate_rebalance_schedule(n_days, lookback_window, freq)
    
    positions = np.zeros(n_etfs, dtype=np.int32)
    cash = 1_000_000.0
    commission_rate = 0.0002
    
    # 合成因子
    selected_factors = factors_3d[:, :, factor_indices]
    combined_score = np.sum(selected_factors, axis=2)
    
    print(f"DEBUG: Rebalance Schedule (first 5): {rebalance_schedule[:5]}")
    
    for t in range(lookback_window, n_days):
        if t in rebalance_schedule:
            date = dates[t]
            print(f"--- Rebalance at {date.date()} (idx={t}) ---")
            
            # Sell
            for i in range(n_etfs):
                if positions[i] > 0:
                    price = close_prices[t, i]
                    print(f"SELL {etf_codes[i]} at {price:.3f}")
                    cash += positions[i] * price * (1 - commission_rate)
            positions[:] = 0
            
            # Buy
            scores = combined_score[t-1].copy() # t-1
            valid_mask = ~np.isnan(scores) & ~np.isnan(close_prices[t])
            
            if np.any(valid_mask):
                scores[~valid_mask] = -np.inf
                top_indices = np.argsort(scores)[-pos_size:][::-1]
                
                invest_cash = cash # Simplified (no timing for debug)
                per_position = invest_cash / len(top_indices)
                
                for idx in top_indices:
                    price = close_prices[t, idx]
                    score = scores[idx]
                    print(f"BUY {etf_codes[idx]} at {price:.3f} (Score: {score:.3f})")
                    shares = int(per_position / price)
                    cost = shares * price * (1 + commission_rate)
                    if cost <= cash:
                        positions[idx] = shares
                        cash -= cost
            
            print(f"Cash: {cash:.2f}")
            if t > rebalance_schedule[4]: break # Only first 5 rebalances

def main():
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data']['data_dir'])
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    etf_codes.sort() # Ensure sorted order
    
    ohlcv_full = loader.load_ohlcv(etf_codes=etf_codes, start_date='2020-01-01', end_date=HOLDOUT_END)
    
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_full)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors({f: raw_factors_df[f] for f in factor_names})
    
    dates_full = std_factors[factor_names[0]].index
    
    # Holdout Slice
    holdout_start_idx = np.where(dates_full >= HOLDOUT_START)[0][0]
    holdout_end_idx = np.where(dates_full <= HOLDOUT_END)[0][-1]
    
    slice_start = max(0, holdout_start_idx - 1)
    slice_end = holdout_end_idx + 1
    
    dates_slice = dates_full[slice_start:slice_end]
    
    all_factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_arr = ohlcv_full['close'].values
    
    factors_slice = all_factors_3d[slice_start:slice_end]
    close_slice = close_arr[slice_start:slice_end]
    
    combo_factors = [f.strip() for f in COMBO.split(' + ')]
    factor_indices = np.array([factor_names.index(f) for f in combo_factors], dtype=np.int32)
    
    # 打印特定日期的价格
    target_dates = ['2025-06-05', '2025-06-06', '2025-06-09']
    print(f"\nDEBUG: Price Check for 1.468")
    for date_str in target_dates:
        try:
            idx = np.where(dates_full == pd.Timestamp(date_str))[0][0]
            for i, code in enumerate(etf_codes):
                for field, arr in [('Close', close_arr)]: # 只检查 Close，因为 VEC 只用 Close
                    price = arr[idx, i]
                    if abs(price - 1.468) < 0.01:
                        print(f"FOUND MATCH! {date_str} {code} {field}: {price:.3f}")
        except:
            pass

    run_vec_debug(
        factors_slice, close_slice, None, factor_indices, 
        freq=3, pos_size=2, lookback_window=1, 
        dates=dates_slice, etf_codes=etf_codes
    )

if __name__ == "__main__":
    main()
