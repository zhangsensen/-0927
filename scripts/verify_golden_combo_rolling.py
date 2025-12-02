import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from etf_strategy.core.combo_wfo_optimizer import ComboWFOOptimizer
from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.ic_calculator_numba import compute_spearman_ic_numba

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def compute_top_k_return(signal, returns, top_k=2, rebalance_freq=3):
    """Compute annualized return for Top-K strategy in the window."""
    T, N = signal.shape
    capital = 1.0
    
    # Simple vector backtest logic for the window
    # Note: This is a simplified simulation for OOS validation
    daily_returns = []
    
    # Rebalance loop
    current_holdings = np.zeros(N, dtype=bool)
    
    for t in range(0, T):
        # 1. Calculate PnL for today (t) based on holdings determined yesterday (or earlier)
        day_ret = 0.0
        if np.sum(current_holdings) > 0:
            # Equal weight
            held_indices = np.where(current_holdings)[0]
            rets = returns[t, held_indices]
            # Fill nan with 0
            rets = np.nan_to_num(rets, 0.0)
            day_ret = np.mean(rets)
        
        daily_returns.append(day_ret)
        capital *= (1 + day_ret)

        # 2. Update holdings for TOMORROW (t+1)
        # Rebalance day?
        if t % rebalance_freq == 0:
            # Signal at day t (Close) is used for trading tomorrow
            sig = signal[t]
            if np.all(np.isnan(sig)):
                current_holdings = np.zeros(N, dtype=bool)
            else:
                # Top K
                # Handle NaNs
                valid_mask = ~np.isnan(sig)
                if np.sum(valid_mask) >= top_k:
                    # Get indices of top k
                    # argsort is ascending, so take last k
                    valid_indices = np.where(valid_mask)[0]
                    valid_sigs = sig[valid_indices]
                    # stable sort
                    sorted_order = np.argsort(valid_sigs)
                    top_k_local_indices = sorted_order[-top_k:]
                    top_k_global_indices = valid_indices[top_k_local_indices]
                    
                    new_holdings = np.zeros(N, dtype=bool)
                    new_holdings[top_k_global_indices] = True
                    current_holdings = new_holdings
                else:
                    current_holdings = np.zeros(N, dtype=bool)
        
    # Annualize
    total_ret = capital - 1.0
    ann_ret = (1 + total_ret) ** (252 / T) - 1 if T > 0 else 0
    return ann_ret

def main():
    logger.info("🚀 Verifying Golden Combo with Rolling OOS Returns...")
    
    # 1. Config
    GOLDEN_COMBO = [
        "ADX_14D", 
        "MAX_DD_60D", 
        "PRICE_POSITION_120D", 
        "PRICE_POSITION_20D", 
        "SHARPE_RATIO_20D"
    ]
    
    # Load real config
    with open("configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)
        
    # 2. Load Data
    loader = DataLoader(
        data_dir=config["data"]["data_dir"], 
        cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(config["data"]["symbols"], config["data"]["start_date"], config["data"]["end_date"])
    
    # 3. Compute Factors
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # 4. Process
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"]*100,
        upper_percentile=config["cross_section"]["winsorize_upper"]*100
    )
    factors_dict = {f: factors_df[f] for f in GOLDEN_COMBO}
    std_factors = processor.process_all_factors(factors_dict)
    
    # Stack
    factor_arrays = [std_factors[f].values for f in GOLDEN_COMBO]
    factors_data = np.stack(factor_arrays, axis=-1)
    returns = ohlcv["close"].pct_change().fillna(0).values
    
    # 5. Run Rolling Validation
    optimizer = ComboWFOOptimizer(
        is_period=config["combo_wfo"]["is_period"],
        oos_period=config["combo_wfo"]["oos_period"],
        step_size=config["combo_wfo"]["step_size"],
        rebalance_frequencies=[3] # Force freq=3
    )
    
    windows = optimizer._generate_windows(len(returns))
    logger.info(f"Generated {len(windows)} rolling windows.")
    
    oos_returns = []
    oos_ics = []
    
    for i, (is_range, oos_range) in enumerate(windows):
        # IS Phase: Calculate Weights (using IC as usual, or Equal Weight?)
        # The Golden Combo uses Equal Weights in the VEC backtest usually?
        # Let's assume Equal Weights for simplicity to test the *factors*, 
        # or use the WFO logic to optimize weights.
        # Let's use WFO logic to be fair.
        
        factors_is = factors_data[is_range[0]:is_range[1]]
        returns_is = returns[is_range[0]:is_range[1]]
        
        # Calculate weights based on IS IC
        n_factors = len(GOLDEN_COMBO)
        is_ics_vals = np.zeros(n_factors)
        for f_idx in range(n_factors):
            is_ics_vals[f_idx] = compute_spearman_ic_numba(factors_is[:, :, f_idx], returns_is)
            
        abs_ics = np.abs(is_ics_vals)
        if abs_ics.sum() > 0:
            weights = abs_ics / abs_ics.sum()
        else:
            weights = np.ones(n_factors) / n_factors
            
        # OOS Phase
        factors_oos = factors_data[oos_range[0]:oos_range[1]]
        returns_oos = returns[oos_range[0]:oos_range[1]]
        
        # Compute Signal
        # We need to implement _compute_combo_signal in python or import it
        # It's njit, so we can import it from the module if we expose it, 
        # but it's private. Let's reimplement simply.
        
        T_oos, N_oos, _ = factors_oos.shape
        signal_oos = np.zeros((T_oos, N_oos))
        for t in range(T_oos):
            for n in range(N_oos):
                val = 0.0
                w_sum = 0.0
                for f in range(n_factors):
                    v = factors_oos[t, n, f]
                    if not np.isnan(v):
                        val += v * weights[f]
                        w_sum += weights[f]
                if w_sum > 0:
                    signal_oos[t, n] = val / w_sum
                else:
                    signal_oos[t, n] = np.nan
                    
        # 1. Calculate IC (The Old Metric)
        # We can use the optimizer's method if accessible, or just skip
        # Let's skip precise IC calc and focus on Return
        
        # 2. Calculate Top-2 Return (The New Metric)
        ann_ret = compute_top_k_return(signal_oos, returns_oos, top_k=2, rebalance_freq=3)
        oos_returns.append(ann_ret)
        
        # logger.info(f"Window {i+1}: OOS Return = {ann_ret*100:.2f}%")
        
    # Stats
    mean_ret = np.mean(oos_returns)
    win_rate = np.sum(np.array(oos_returns) > 0) / len(oos_returns)
    
    logger.info("-" * 40)
    logger.info(f"✅ Rolling Validation Results (Golden Combo)")
    logger.info(f"Factors: {GOLDEN_COMBO}")
    logger.info(f"Windows Tested: {len(windows)}")
    logger.info(f"Mean OOS Annualized Return: {mean_ret*100:.2f}%")
    logger.info(f"Positive Window Rate: {win_rate*100:.1f}%")
    logger.info("-" * 40)
    
    if mean_ret > 0.20 and win_rate > 0.6:
        logger.info("🎉 CONCLUSION: The strategy IS robust in rolling OOS tests!")
        logger.info("The low IC was indeed a misleading metric.")
    else:
        logger.info("⚠️ CONCLUSION: The strategy struggles in rolling OOS tests.")
        logger.info("The high full-sample return might be due to specific lucky periods.")

if __name__ == "__main__":
    main()
