#!/usr/bin/env python3
"""
Standalone WFO Runner (Factor Combination Selection)
Replicates the logic of `run_unified_wfo.py` for the standalone project.

Workflow:
1. Load Data & Compute Factors
2. Generate all combinations of factors (size 2-5)
3. Calculate IC/ICIR for each combination using Numba
4. Rank combinations by ICIR
5. Output top combinations for use in `run_strategy.py`
"""

import sys
import yaml
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import combinations
from numba import njit, prange

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.data_loader import DataLoader
from core.precise_factor_library import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Numba Kernels (Copied from run_unified_wfo.py)
# ============================================================================

@njit(cache=True)
def _compute_spearman_ic_single_day(scores: np.ndarray, returns: np.ndarray) -> float:
    """Compute Spearman IC for a single day."""
    mask = ~(np.isnan(scores) | np.isnan(returns))
    n_valid = np.sum(mask)
    
    if n_valid < 3:
        return np.nan
    
    s = scores[mask]
    r = returns[mask]
    
    # Rank
    s_rank = np.argsort(np.argsort(s)).astype(np.float64)
    r_rank = np.argsort(np.argsort(r)).astype(np.float64)
    
    s_mean = np.mean(s_rank)
    r_mean = np.mean(r_rank)
    
    numerator = np.sum((s_rank - s_mean) * (r_rank - r_mean))
    s_std = np.sqrt(np.sum((s_rank - s_mean) ** 2))
    r_std = np.sqrt(np.sum((r_rank - r_mean) ** 2))
    
    if s_std > 0 and r_std > 0:
        return numerator / (s_std * r_std)
    return np.nan

@njit(cache=True)
def _compute_combo_ic_series(
    factors_3d: np.ndarray,
    factor_indices: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
    min_valid_days: int,
) -> tuple:
    """Compute IC series for a single combination."""
    T, N, _ = factors_3d.shape
    n_factors = len(factor_indices)
    
    ic_values = np.zeros(T - lookback)
    valid_count = 0
    
    for t in range(lookback, T):
        # Combine scores (Equal Weight)
        combo_score = np.zeros(N)
        for n in range(N):
            score = 0.0
            for i in range(n_factors):
                f_idx = factor_indices[i]
                val = factors_3d[t-1, n, f_idx]
                if not np.isnan(val):
                    score += val
            combo_score[n] = score
        
        # Compute IC vs Forward Returns
        ic = _compute_spearman_ic_single_day(combo_score, forward_returns[t])
        ic_values[t - lookback] = ic
        if not np.isnan(ic):
            valid_count += 1
    
    # Statistics
    if valid_count >= min_valid_days:
        valid_ics = np.zeros(valid_count)
        idx = 0
        for i in range(len(ic_values)):
            if not np.isnan(ic_values[i]):
                valid_ics[idx] = ic_values[i]
                idx += 1
        
        ic_mean = np.mean(valid_ics)
        ic_std = np.std(valid_ics)
        
        ic_ir = ic_mean / ic_std if ic_std > 0.001 else 0.0
        return ic_mean, ic_std, ic_ir, valid_count
    
    return 0.0, 0.0, 0.0, valid_count

@njit(parallel=True, cache=True)
def _compute_all_combo_ics(
    factors_3d: np.ndarray,
    all_combo_indices: np.ndarray,
    combo_sizes: np.ndarray,
    forward_returns: np.ndarray,
    lookback: int,
    min_valid_days: int,
) -> np.ndarray:
    """Parallel compute IC/ICIR for all combinations."""
    n_combos = all_combo_indices.shape[0]
    results = np.zeros((n_combos, 4))
    
    for i in prange(n_combos):
        size = combo_sizes[i]
        factor_indices = all_combo_indices[i, :size]
        
        ic_mean, ic_std, ic_ir, n_valid = _compute_combo_ic_series(
            factors_3d, factor_indices, forward_returns, lookback, min_valid_days
        )
        
        results[i, 0] = ic_mean
        results[i, 1] = ic_std
        results[i, 2] = ic_ir
        results[i, 3] = n_valid
    
    return results

# ============================================================================
# Main Logic
# ============================================================================

def main():
    print("=" * 80)
    print("🚀 Standalone WFO Runner (Factor Selection)")
    print("=" * 80)
    
    # 1. Load Config
    config_path = Path(__file__).parent / "configs/wfo_config.yaml"
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Load Data
    logger.info("Loading Data...")
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    # 3. Compute Factors
    logger.info("Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    factor_names = raw_factors_df.columns.get_level_values(0).unique().tolist()
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    
    # 4. Process Factors
    logger.info("Processing Factors (Standardization)...")
    processor = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = processor.process_all_factors(raw_factors)
    
    # 5. Prepare Data for Numba
    factor_names = sorted(std_factors.keys())
    first_factor = std_factors[factor_names[0]]
    T, N = first_factor.shape
    
    factors_3d = np.stack([std_factors[f].values for f in factor_names], axis=-1)
    close_prices = ohlcv["close"][first_factor.columns].ffill().bfill().values
    
    # Compute Forward Returns (T+1)
    logger.info("Computing Forward Returns...")
    forward_returns = np.zeros((T, N))
    for t in range(T - 1):
        for n in range(N):
            if close_prices[t, n] > 0 and not np.isnan(close_prices[t + 1, n]):
                forward_returns[t + 1, n] = (close_prices[t + 1, n] - close_prices[t, n]) / close_prices[t, n]
            else:
                forward_returns[t + 1, n] = np.nan
                
    # 6. Generate Combinations
    combo_sizes = config["combo_wfo"]["combo_sizes"]
    all_combos = []
    for size in combo_sizes:
        combos = list(combinations(range(len(factor_names)), size))
        all_combos.extend([(c, size) for c in combos])
        logger.info(f"   {size}-Factor Combinations: {len(combos)}")
    
    logger.info(f"   Total Combinations: {len(all_combos)}")
    
    # Prepare Numba arrays
    max_combo_size = max(combo_sizes)
    n_combos = len(all_combos)
    all_combo_indices = np.full((n_combos, max_combo_size), -1, dtype=np.int64)
    combo_sizes_arr = np.zeros(n_combos, dtype=np.int64)
    
    for i, (combo, size) in enumerate(all_combos):
        combo_sizes_arr[i] = size
        for j, idx in enumerate(combo):
            all_combo_indices[i, j] = idx
            
    # 7. Run Calculation
    logger.info("⚡ Calculating IC/ICIR (Parallel)...")
    
    # Warmup
    _ = _compute_all_combo_ics(
        factors_3d[:100], all_combo_indices[:10], combo_sizes_arr[:10], 
        forward_returns[:100], 50, 20
    )
    
    t0 = time.time()
    ic_results = _compute_all_combo_ics(
        factors_3d,
        all_combo_indices,
        combo_sizes_arr,
        forward_returns,
        252, # Lookback
        config["combo_wfo"]["min_valid_days"]
    )
    logger.info(f"   Calculation finished in {time.time() - t0:.2f}s")
    
    # 8. Process Results
    results = []
    for i, (combo, size) in enumerate(all_combos):
        combo_str = " + ".join([factor_names[idx] for idx in combo])
        ic_mean, ic_std, ic_ir, n_valid = ic_results[i]
        
        if n_valid >= config["combo_wfo"]["min_valid_days"]:
            results.append({
                "combo": combo_str,
                "combo_size": size,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": ic_ir,
                "n_valid_days": int(n_valid),
            })
            
    df = pd.DataFrame(results)
    df = df.sort_values("icir", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    
    # 9. Save Output
    output_dir = Path(__file__).parent / config["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df.to_csv(output_dir / f"wfo_results_{timestamp}.csv", index=False)
    
    print("\n" + "=" * 80)
    print("🏆 TOP 10 Factor Combinations (by ICIR)")
    print("-" * 80)
    print(f"{'Rank':>4} | {'ICIR':>8} | {'IC_mean':>8} | {'Combo'}")
    print("-" * 80)
    
    for _, row in df.head(10).iterrows():
        print(f"{row['rank']:>4} | {row['icir']:>8.4f} | {row['ic_mean']:>8.4f} | {row['combo']}")
        
    print("=" * 80)
    print(f"✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()
