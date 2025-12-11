
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule

# Import the kernel directly
from scripts.batch_vec_backtest import vec_backtest_kernel

warnings.filterwarnings('ignore')

def calculate_metrics(equity_curve, dates):
    """Calculate comprehensive metrics from equity curve."""
    eq_s = pd.Series(equity_curve, index=dates)
    
    # Returns
    total_ret = eq_s.iloc[-1] / eq_s.iloc[0] - 1
    T = len(eq_s)
    ann_ret = (1 + total_ret) ** (252 / T) - 1
    
    # Drawdown
    dd = (eq_s / eq_s.cummax() - 1)
    max_dd = dd.min()
    
    # Sharpe
    daily_ret = eq_s.pct_change().fillna(0)
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    
    # Yearly Returns
    years = eq_s.index.year.unique()
    yearly_rets = {}
    for y in years:
        y_eq = eq_s[str(y)]
        if len(y_eq) > 0:
            yearly_rets[y] = y_eq.iloc[-1] / y_eq.iloc[0] - 1
        else:
            yearly_rets[y] = np.nan
            
    # Specific Date Drawdown (2024-10-08)
    try:
        dd_oct8 = dd.loc['2024-10-08']
    except:
        dd_oct8 = np.nan
        
    return {
        'ann_ret': ann_ret,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'yearly_rets': yearly_rets,
        'dd_oct8': dd_oct8,
        'equity_curve': eq_s,
        'drawdown_curve': dd
    }

def run_vec_simulation(factors_3d, close_arr, open_arr, high_arr, low_arr, timing_arr, vol_regime_arr, indices, rebalance_schedule, commission=0.0002):
    """Run VEC simulation for a specific setup."""
    T, N, _ = factors_3d.shape
    
    # Dummy inputs
    atr_arr = np.zeros((T, N))
    individual_trend_arr = np.ones((T, N), dtype=bool)
    profit_ladder_thresholds = np.array([np.inf, np.inf, np.inf])
    profit_ladder_stops = np.array([0.0, 0.0, 0.0])
    profit_ladder_multipliers = np.array([0.0, 0.0, 0.0])
    
    kernel_res = vec_backtest_kernel(
        factors_3d,
        close_arr,
        open_arr,
        high_arr,
        low_arr,
        timing_arr,
        vol_regime_arr,
        indices,
        rebalance_schedule,
        2, # POS_SIZE
        1_000_000.0, # INITIAL_CAPITAL
        commission,
        0.20, 20, False, # Dynamic Leverage
        False, 0.0, atr_arr, 3.0, False, # Stop Loss
        individual_trend_arr, False, # Individual Trend
        profit_ladder_thresholds, profit_ladder_stops, profit_ladder_multipliers,
        0.0, 0.0, 5, # Circuit Breaker
        0, # Cooldown
        1.0 # Leverage Cap
    )
    
    equity_raw = kernel_res[0]
    
    # Fix equity_raw before start
    non_zero_idx = np.argmax(equity_raw > 0)
    if non_zero_idx > 0:
        equity_raw[:non_zero_idx] = 1_000_000.0
        
    # Cash adjustment
    # timing_v3 = timing_arr * vol_regime_arr
    current_exposure = timing_arr * vol_regime_arr
    cash_weight = 1.0 - current_exposure
    cash_ret = cash_weight * (0.02 / 252)
    
    equity_raw_safe = equity_raw.copy()
    equity_raw_safe[equity_raw_safe == 0] = 1.0
    
    r_raw = np.zeros(T)
    r_raw[1:] = equity_raw[1:] / equity_raw_safe[:-1] - 1
    r_adj = r_raw + cash_ret
    
    equity_adj = np.cumprod(1 + r_adj) * 1_000_000.0
    equity_adj[0] = 1_000_000.0
    
    return equity_adj

def main():
    print("🚀 Starting Top 1 Strategy Validation...")
    
    # 1. Setup
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    loader = DataLoader(data_dir=config['data'].get('data_dir'), cache_dir=config['data'].get('cache_dir'))
    ohlcv = loader.load_ohlcv(etf_codes=config['data']['symbols'], start_date=config['data']['start_date'], end_date=config['data']['end_date'])
    
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    factor_map = {name: i for i, name in enumerate(factor_names_list)}
    
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors({fname: raw_factors_df[fname] for fname in factor_names_list})
    
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    T, N = first_factor.shape
    
    factors_3d = np.zeros((T, N, len(factor_names_list)), dtype=np.float64)
    for i, name in enumerate(factor_names_list):
        factors_3d[:, :, i] = std_factors[name].values
        
    # Vol Regime
    if '510300' in ohlcv['close'].columns:
        hs300 = ohlcv['close']['510300']
    else:
        hs300 = ohlcv['close'].iloc[:, 0]
    rets = hs300.pct_change()
    hv = rets.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
    hv_5d = hv.shift(5)
    regime_vol = (hv + hv_5d) / 2
    regime_vol = regime_vol.fillna(0.0)
    
    vol_regime_arr = np.ones(T, dtype=np.float64)
    vol_vals = regime_vol.values
    mask_yellow = (vol_vals >= 25) & (vol_vals < 30)
    vol_regime_arr[mask_yellow] = 0.7
    mask_orange = (vol_vals >= 30) & (vol_vals < 40)
    vol_regime_arr[mask_orange] = 0.4
    mask_red = (vol_vals >= 40)
    vol_regime_arr[mask_red] = 0.1
    
    # Timing
    timing_module = LightTimingModule()
    timing_signals = timing_module.compute_position_ratios(ohlcv['close'])
    timing_arr = shift_timing_signal(timing_signals)
    
    # Arrays
    close_arr = ohlcv['close'].values
    open_arr = ohlcv['open'].values
    high_arr = ohlcv['high'].values
    low_arr = ohlcv['low'].values
    rebalance_schedule = generate_rebalance_schedule(T, 252, 3)
    
    # Strategy
    combo_str = "ADX_14D + PRICE_POSITION_20D + SLOPE_20D + VOL_RATIO_20D + VORTEX_14D"
    factors = [f.strip() for f in combo_str.split('+')]
    indices = np.array([factor_map[f] for f in factors], dtype=np.int64)
    
    # ==========================================================================
    # Task 1: V0 vs V3
    # ==========================================================================
    print("\n[Task 1] V0 vs V3 Comparison")
    
    # V0 Run (Vol Regime = 1.0)
    eq_v0 = run_vec_simulation(factors_3d, close_arr, open_arr, high_arr, low_arr, timing_arr, np.ones(T), indices, rebalance_schedule)
    metrics_v0 = calculate_metrics(eq_v0, dates)
    
    # V3 Run
    eq_v3 = run_vec_simulation(factors_3d, close_arr, open_arr, high_arr, low_arr, timing_arr, vol_regime_arr, indices, rebalance_schedule)
    metrics_v3 = calculate_metrics(eq_v3, dates)
    
    print(f"| Metric | V0 | V3 | Improvement |")
    print(f"|---|---|---|---|")
    print(f"| Ann Ret | {metrics_v0['ann_ret']:.1%} | {metrics_v3['ann_ret']:.1%} | {(metrics_v3['ann_ret'] - metrics_v0['ann_ret']):.1%} |")
    print(f"| Max DD | {metrics_v0['max_dd']:.1%} | {metrics_v3['max_dd']:.1%} | {(metrics_v3['max_dd'] - metrics_v0['max_dd']):.1%} |")
    print(f"| Sharpe | {metrics_v0['sharpe']:.2f} | {metrics_v3['sharpe']:.2f} | {(metrics_v3['sharpe'] - metrics_v0['sharpe']):.2f} |")
    print(f"| 2022 Ret | {metrics_v0['yearly_rets'][2022]:.1%} | {metrics_v3['yearly_rets'][2022]:.1%} | - |")
    print(f"| 2024 Ret | {metrics_v0['yearly_rets'][2024]:.1%} | {metrics_v3['yearly_rets'][2024]:.1%} | - |")
    print(f"| Oct 8 DD | {metrics_v0['dd_oct8']:.1%} | {metrics_v3['dd_oct8']:.1%} | - |")
    
    # ==========================================================================
    # Task 3: Cost Sensitivity
    # ==========================================================================
    print("\n[Task 3] Cost Sensitivity")
    
    eq_7bp = run_vec_simulation(factors_3d, close_arr, open_arr, high_arr, low_arr, timing_arr, vol_regime_arr, indices, rebalance_schedule, commission=0.0007)
    m_7bp = calculate_metrics(eq_7bp, dates)
    
    eq_15bp = run_vec_simulation(factors_3d, close_arr, open_arr, high_arr, low_arr, timing_arr, vol_regime_arr, indices, rebalance_schedule, commission=0.0015)
    m_15bp = calculate_metrics(eq_15bp, dates)
    
    decay = (m_7bp['ann_ret'] - m_15bp['ann_ret']) / m_7bp['ann_ret']
    
    print(f"| Cost | Ann Ret | Max DD | Sharpe |")
    print(f"|---|---|---|---|")
    print(f"| 7bp | {m_7bp['ann_ret']:.1%} | {m_7bp['max_dd']:.1%} | {m_7bp['sharpe']:.2f} |")
    print(f"| 12bp | {metrics_v3['ann_ret']:.1%} | {metrics_v3['max_dd']:.1%} | {metrics_v3['sharpe']:.2f} |")
    print(f"| 15bp | {m_15bp['ann_ret']:.1%} | {m_15bp['max_dd']:.1%} | {m_15bp['sharpe']:.2f} |")
    print(f"Decay Rate: {decay:.1%}")
    
    # ==========================================================================
    # Task 5: Yearly Breakdown
    # ==========================================================================
    print("\n[Task 5] Yearly Breakdown")
    print(f"| Year | Return | Max DD |")
    print(f"|---|---|---|")
    for y in sorted(metrics_v3['yearly_rets'].keys()):
        # Calculate yearly max dd
        y_dd = metrics_v3['drawdown_curve'][str(y)].min()
        print(f"| {y} | {metrics_v3['yearly_rets'][y]:.1%} | {y_dd:.1%} |")
        
    # ==========================================================================
    # Task 6: 2024-10-08 Analysis
    # ==========================================================================
    print("\n[Task 6] 2024-10-08 Analysis")
    day = '2024-10-08'
    try:
        idx = dates.get_loc(day)
        vol_val = regime_vol.loc[day]
        exposure = vol_regime_arr[idx]
        
        print(f"Date: {day}")
        print(f"Regime Vol: {vol_val:.1f}%")
        print(f"Exposure: {exposure:.0%}")
        print(f"V3 Equity: {metrics_v3['equity_curve'].loc[day]:.0f}")
        print(f"V0 Equity: {metrics_v0['equity_curve'].loc[day]:.0f}")
        
        # Holdings (Need to simulate selection)
        # This is hard to get from VEC kernel output directly without re-running selection logic
        # But we can infer from VEC equity change if we assume we know the holdings
        # For now, just report macro stats
    except Exception as e:
        print(f"Error analyzing {day}: {e}")
        
    # ==========================================================================
    # Task 7: Monte Carlo VaR
    # ==========================================================================
    print("\n[Task 7] Monte Carlo VaR")
    daily_rets = metrics_v3['equity_curve'].pct_change().dropna().values
    
    # Block Bootstrap
    n_sims = 10000
    block_size = 21
    n_days = 252
    
    sim_max_dds = []
    
    for _ in range(n_sims):
        # Generate path
        path_rets = []
        while len(path_rets) < n_days:
            start_idx = np.random.randint(0, len(daily_rets) - block_size)
            block = daily_rets[start_idx : start_idx + block_size]
            path_rets.extend(block)
        
        path_rets = np.array(path_rets[:n_days])
        path_eq = np.cumprod(1 + path_rets)
        path_dd = (path_eq / np.maximum.accumulate(path_eq) - 1).min()
        sim_max_dds.append(path_dd)
        
    var_99 = np.percentile(sim_max_dds, 1)
    print(f"Block VaR 99%: {var_99:.1%}")
    
    # ==========================================================================
    # Task 8: Alternatives
    # ==========================================================================
    print("\n[Task 8] Alternatives Selection")
    df_all = pd.read_csv(ROOT / 'results/full_space_v3_results.csv')
    
    # Filter All Weather
    candidates = df_all[
        (df_all['ret_2022'] > 0) & 
        (df_all['ret_2024'] > 0.20) &
        (df_all['sharpe'] >= 0.90) # Relaxed slightly to find enough candidates
    ].copy()
    
    # Calculate Correlation (Proxy: Jaccard Similarity of Factors)
    top1_factors = set(factors)
    
    def calc_similarity(combo_str):
        fs = set([f.strip() for f in combo_str.split('+')])
        intersection = len(top1_factors.intersection(fs))
        union = len(top1_factors.union(fs))
        return intersection / union
        
    candidates['similarity'] = candidates['combo'].apply(calc_similarity)
    
    # Filter low similarity
    alternatives = candidates[candidates['similarity'] < 0.5].sort_values('sharpe', ascending=False).head(5)
    
    print(alternatives[['combo', 'ann_return', 'max_dd', 'sharpe', 'similarity']].to_markdown())
    
    # ==========================================================================
    # Task 9: Dry Run Data
    # ==========================================================================
    print("\n[Task 9] Dry Run Data Prep")
    last_date = dates[-1]
    print(f"Last Date: {last_date}")
    
    # Get last day factors
    last_idx = -1
    scores = np.zeros(N)
    for k in indices:
        scores += factors_3d[last_idx, :, k]
        
    # Rank
    # Filter valid prices
    valid_mask = ~np.isnan(close_arr[last_idx]) & (close_arr[last_idx] > 0)
    scores[~valid_mask] = -np.inf
    
    # Top 2
    top_indices = np.argsort(scores)[::-1][:2]
    
    print(f"Top 2 ETFs for {last_date}:")
    for idx in top_indices:
        code = ohlcv['close'].columns[idx]
        score = scores[idx]
        print(f"  {code}: Score {score:.2f}")
        
    # Vol Regime
    last_vol = regime_vol.iloc[-1]
    last_exposure = vol_regime_arr[-1]
    print(f"Last Vol: {last_vol:.1f}%")
    print(f"Last Exposure: {last_exposure:.0%}")

if __name__ == "__main__":
    main()
