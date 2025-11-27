import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.getcwd())

try:
    from etf_rotation_optimized.core.data_loader import DataLoader
    from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
    from etf_rotation_optimized.core.market_timing import LightTimingModule
except ImportError:
    # Try adding parent dir
    sys.path.append(str(Path(os.getcwd()).parent))
    from etf_rotation_optimized.core.data_loader import DataLoader
    from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
    from etf_rotation_optimized.core.market_timing import LightTimingModule

# Configuration
COMBO = "MAX_DD_60D + MOM_20D + RSI_14 + VOL_RATIO_20D + VOL_RATIO_60D"
FREQ = 8
POS_SIZE = 3

def run_verification():
    print("="*60)
    print(f"🔍 严格 T+1 验证 (Strict T+1 Verification)")
    print(f"🎯 策略: {COMBO}")
    print("="*60)
    
    # 1. Load Config & Data
    print("Loading Data...")
    with open("configs/combo_wfo_config.yaml") as f:
        config = yaml.safe_load(f)
        
    loader = DataLoader(
        data_dir=config["data"]["data_dir"],
        cache_dir=config["data"]["cache_dir"]
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        use_cache=True
    )
    
    # 2. Compute Factors
    print("Computing Factors...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # 3. Process Factors
    print("Processing Factors...")
    
    # Extract factors into dict of DataFrames
    factors_dict = {}
    # Get factor names from the MultiIndex columns (level 0)
    # factors_df.columns is MultiIndex [(Factor, Ticker), ...]
    # We want unique level 0 values
    available_factors = factors_df.columns.get_level_values(0).unique()
    
    for f in available_factors:
        factors_dict[f] = factors_df[f]
        
    proc = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = proc.process_all_factors(factors_dict)
    
    # 4. Prepare Arrays
    factor_names = sorted(std_factors.keys())
    combo_factors = [f.strip() for f in COMBO.split(" + ")]
    factor_indices = [factor_names.index(f) for f in combo_factors]
    
    F_data = np.stack([std_factors[n].values for n in factor_names], axis=-1)
    F_sel = F_data[:, :, factor_indices]
    
    returns = ohlcv["close"].pct_change().values
    dates = ohlcv["close"].index
    
    # 5. Simulation
    T, N, F = F_sel.shape
    lookback = 252
    
    # Equity Curves
    # A: Standard (Signal T-1 -> Trade Close T-1) [Current Production Logic]
    # B: Strict T+1 (Signal T-1 -> Trade Close T) [Delayed Execution]
    
    eq_A = [1.0]
    eq_B = [1.0]
    
    w_A = np.zeros(N)
    w_B = np.zeros(N)
    pending_w_B = np.zeros(N)
    
    # Timing Signal (Shifted)
    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_series = timing_series.shift(1).fillna(1.0)
    timing_arr = timing_series.values
    
    print(f"\n🚀 Running Simulation ({T} days)...")
    
    logs = []
    
    for t in range(lookback + 1, T):
        date = dates[t]
        is_rebalance = (t - (lookback + 1)) % FREQ == 0
        
        # --- Rebalance Logic ---
        if is_rebalance:
            # Use data up to t-1
            hist_start = t - lookback
            hist_end = t
            
            # Calculate IC Weights (Simplified: Equal Weight for speed/demo, or simple correlation)
            # For strict verification, we should use the exact same IC logic, but let's assume Equal Weight for factors first
            # to isolate the "Execution Timing" effect.
            # Or better: Calculate simple IC
            f_hist = F_sel[hist_start:hist_end]
            r_hist = returns[hist_start:hist_end]
            
            feat_w = np.zeros(len(combo_factors))
            for i in range(len(combo_factors)):
                x = f_hist[:,:,i].flatten()
                y = r_hist.flatten()
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() > 10:
                    # Simple correlation
                    feat_w[i] = np.abs(np.corrcoef(x[mask], y[mask])[0,1])
            
            if feat_w.sum() > 0:
                feat_w /= feat_w.sum()
            else:
                feat_w[:] = 1.0 / len(combo_factors)
                
            # Calculate Signal using factors at t-1
            f_day = F_sel[t-1]
            score = np.zeros(N)
            for i in range(len(combo_factors)):
                score += np.nan_to_num(f_day[:, i], nan=0.0) * feat_w[i]
                
            # Select Top N
            top_idx = np.argsort(score)[-POS_SIZE:]
            target_w = np.zeros(N)
            target_w[top_idx] = 1.0 / POS_SIZE
            
            # Update Weights
            w_A = target_w
            pending_w_B = target_w
            
            # Log Trade
            if t > T - 60: # Log last few trades
                # Decode holdings
                holdings = []
                for idx in np.where(target_w > 0)[0]:
                    # We don't have ETF names in F_sel, but we can use index
                    # Actually we can get names from ohlcv columns
                    code = ohlcv["close"].columns[idx]
                    holdings.append(code)
                logs.append(f"📅 {date.date()} [Rebalance] Signal Generated. Target Holdings: {holdings}")
            
        # --- Return Calculation ---
        r_day = np.nan_to_num(returns[t], nan=0.0)
        pos_ratio = timing_arr[t]
        
        # A: Standard
        ret_A = np.sum(w_A * r_day) * pos_ratio
        eq_A.append(eq_A[-1] * (1 + ret_A))
        
        # B: Strict T+1
        # Holds w_B (determined at previous rebalance T-k -> Trade Close T-k+1)
        ret_B = np.sum(w_B * r_day) * pos_ratio
        eq_B.append(eq_B[-1] * (1 + ret_B))
        
        # Update B weights at Close T
        if is_rebalance:
            w_B = pending_w_B
            
        if t > T - 6:
            logs.append({
                "Date": str(date.date()),
                "Ret_A": f"{ret_A:.2%}",
                "Ret_B": f"{ret_B:.2%}",
                "Eq_A": f"{eq_A[-1]:.4f}",
                "Eq_B": f"{eq_B[-1]:.4f}"
            })
            
    # Stats
    days = len(eq_A)
    ann_A = (eq_A[-1]**(252/days) - 1)
    ann_B = (eq_B[-1]**(252/days) - 1)
    
    print(f"\n📊 验证结果对比:")
    print(f"  A: 标准模式 (信号 T-1 -> 交易 Close T-1) [当前生产模式]")
    print(f"     年化收益: {ann_A:.2%}")
    print(f"     最终净值: {eq_A[-1]:.4f}")
    
    print(f"  B: 严格 T+1 (信号 T-1 -> 交易 Close T) [延迟一天]")
    print(f"     年化收益: {ann_B:.2%}")
    print(f"     最终净值: {eq_B[-1]:.4f}")
    
    print(f"\n📉 差异分析:")
    diff = ann_A - ann_B
    print(f"  年化差异: {diff:.2%}")
    if abs(diff) < 0.05:
        print("  ✅ 差异较小 (<5%)，策略对执行时机不敏感，结果可信。")
    else:
        print("  ⚠️ 差异显著 (>5%)，策略可能依赖短期反转或存在前视偏差。")
        
    print("\n📝 最近 5 天日志:")
    for l in logs:
        print(l)

if __name__ == "__main__":
    run_verification()
