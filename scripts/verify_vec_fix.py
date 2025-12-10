
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from numba import njit

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

@njit
def run_vec_backtest_fast(factors_3d, close_prices, timing_arr, factor_indices, rebalance_schedule, pos_size, commission_rate):
    n_days, n_etfs = close_prices.shape
    positions = np.zeros(n_etfs, dtype=np.int32)
    cash = 1_000_000.0
    portfolio_value = np.zeros(n_days)
    
    # Pre-calculate combined scores
    # factors_3d: (n_days, n_etfs, n_factors)
    # factor_indices: (n_selected_factors,)
    selected_factors = factors_3d[:, :, factor_indices]
    # Sum across the last axis (factors)
    # Handle NaNs: if any factor is NaN, sum is NaN (which is what we want for strict filtering)
    # But np.sum ignores NaNs? No, np.sum propagates NaNs unless using nansum.
    # We want to propagate NaNs.
    combined_score = np.zeros((n_days, n_etfs))
    for i in range(n_days):
        for j in range(n_etfs):
            sum_val = 0.0
            is_nan = False
            for k in range(len(factor_indices)):
                val = selected_factors[i, j, k]
                if np.isnan(val):
                    is_nan = True
                    break
                sum_val += val
            if is_nan:
                combined_score[i, j] = np.nan
            else:
                combined_score[i, j] = sum_val

    next_rebalance_idx = 0
    
    for t in range(n_days):
        # Mark to market
        current_value = cash
        for i in range(n_etfs):
            if positions[i] > 0:
                current_value += positions[i] * close_prices[t, i]
        portfolio_value[t] = current_value
        
        # Rebalance
        if next_rebalance_idx < len(rebalance_schedule) and t == rebalance_schedule[next_rebalance_idx]:
            # Sell all
            for i in range(n_etfs):
                if positions[i] > 0:
                    cash += positions[i] * close_prices[t, i] * (1 - commission_rate)
                    positions[i] = 0
            
            # Buy
            # Signal is from t-1 (already shifted in main)
            # But wait, combined_score is aligned with close_prices?
            # In main: 
            # std_factors aligned with dates
            # combined_score[t] uses factors at t.
            # We need signal from t-1 to trade at t close?
            # The standard logic is: Signal available at t-1 Close -> Trade at t Close (or t Open).
            # Here we trade at t Close. So we use signal from t-1?
            # Or signal from t (using t's Close)?
            # If we trade at t Close, we can technically use t's Close to calculate signal?
            # No, that's lookahead if we use t's Close to decide to buy at t's Close (slippage/execution lag).
            # Safe bet: use t-1 signal.
            
            if t > 0:
                scores = combined_score[t-1]
                # Filter valid
                valid_indices = []
                for i in range(n_etfs):
                    if not np.isnan(scores[i]) and not np.isnan(close_prices[t, i]):
                        valid_indices.append(i)
                
                if len(valid_indices) > 0:
                    # Sort by score descending
                    # Manual sort for numba
                    # Or just use argsort on full array and filter
                    # Numba argsort handles NaNs (put them at end)
                    
                    # Simple approach:
                    # Create temp arrays
                    valid_scores = np.zeros(len(valid_indices))
                    for k, idx in enumerate(valid_indices):
                        valid_scores[k] = scores[idx]
                    
                    # Sort
                    sorted_args = np.argsort(valid_scores)[::-1]
                    
                    top_k = min(pos_size, len(valid_indices))
                    invest_cash = cash # Full position
                    per_position = invest_cash / top_k
                    
                    for k in range(top_k):
                        idx = valid_indices[sorted_args[k]]
                        price = close_prices[t, idx]
                        shares = int(per_position / price)
                        cost = shares * price * (1 + commission_rate)
                        if cost <= cash:
                            positions[idx] = shares
                            cash -= cost
            
            next_rebalance_idx += 1
            
    return portfolio_value

def main():
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path(config['data']['data_dir'])
    loader = DataLoader(data_dir=data_dir, cache_dir=ROOT / '.cache')
    etf_files = list(data_dir.glob("*.parquet"))
    etf_codes = [f.stem.split('_')[0].split('.')[0] for f in etf_files]
    etf_codes.sort() # Ensure sorted order
    
    print("Loading data...")
    ohlcv_full = loader.load_ohlcv(etf_codes=etf_codes, start_date='2020-01-01', end_date=HOLDOUT_END)
    
    print("Computing factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv_full)
    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors({f: raw_factors_df[f] for f in factor_names})
    
    dates_full = std_factors[factor_names[0]].index
    
    # Prepare 3D factors
    n_days = len(dates_full)
    n_etfs = len(etf_codes)
    n_factors = len(factor_names)
    factors_3d = np.zeros((n_days, n_etfs, n_factors))
    
    for k, fname in enumerate(factor_names):
        factors_3d[:, :, k] = std_factors[fname].values
        
    close_prices = ohlcv_full['close'].reindex(dates_full).values
    
    # Timing
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series_raw = timing_module.compute_position_ratios(ohlcv_full["close"])
    timing_arr_shifted = shift_timing_signal(timing_series_raw.reindex(dates_full).fillna(1.0).values)
    
    # Indices
    target_factors = [f.strip() for f in COMBO.split(" + ")]
    factor_indices = np.array([factor_names.index(f) for f in target_factors])
    
    # Schedule
    rebalance_schedule = generate_rebalance_schedule(n_days, 252, 3)
    
    print("Running VEC...")
    portfolio_value = run_vec_backtest_fast(
        factors_3d, close_prices, timing_arr_shifted, factor_indices, 
        np.array(rebalance_schedule), 2, 0.0002
    )
    
    # Calculate metrics
    # Holdout Slice
    holdout_start_idx = np.where(dates_full >= HOLDOUT_START)[0][0]
    holdout_end_idx = np.where(dates_full <= HOLDOUT_END)[0][-1]
    
    holdout_values = portfolio_value[holdout_start_idx:holdout_end_idx+1]
    holdout_returns = pd.Series(holdout_values).pct_change().dropna()
    
    total_ret = (holdout_values[-1] / holdout_values[0]) - 1
    sharpe = holdout_returns.mean() / holdout_returns.std() * np.sqrt(252)
    
    print(f"VEC Holdout Return: {total_ret*100:.2f}%")
    print(f"VEC Holdout Sharpe: {sharpe:.3f}")

if __name__ == "__main__":
    main()
