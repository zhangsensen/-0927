
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import warnings

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal

# Import the backtest engine
from batch_vec_backtest import run_vec_backtest

warnings.filterwarnings('ignore')

def main():
    print('=' * 80)
    print('🚀 FULL SPACE VECTORIZED BACKTEST (FREQ=3, POS=2)')
    print('=' * 80)

    # 1. Load Configuration
    config_path = ROOT / 'configs/combo_wfo_config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    backtest_config = config.get('backtest', {})
    
    # Override with optimized parameters
    FREQ = 3
    POS_SIZE = 2
    EXTREME_THRESHOLD = -0.1
    EXTREME_POSITION = 0.1
    
    print(f"Configuration:")
    print(f"  FREQ: {FREQ}")
    print(f"  POS_SIZE: {POS_SIZE}")
    print(f"  Timing Threshold: {EXTREME_THRESHOLD}")
    print(f"  Timing Position: {EXTREME_POSITION}")

    # 2. Load Data
    print("\nLoading Data...")
    loader = DataLoader(
        data_dir=config['data'].get('data_dir'),
        cache_dir=config['data'].get('cache_dir'),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
    )

    # 3. Compute Factors
    print("Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    factor_names_list = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names_list}

    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)

    # 4. Prepare Backtest Data
    first_factor = std_factors[factor_names_list[0]]
    dates = first_factor.index
    etf_codes = first_factor.columns.tolist()
    
    # Stack all factors into a 3D array (Factors, Time, Assets)
    # Note: run_vec_backtest expects (Time, Assets, Factors) or similar?
    # Let's check batch_vec_backtest.py signature.
    # It takes `factors_3d` which is usually (Time, Assets, Factors) or (Factors, Time, Assets)?
    # In batch_vec_backtest.py:
    # factors_3d = np.stack([std_factors[f].values for f in factor_names_in_combo], axis=-1)
    # So it is (Time, Assets, Factors).
    
    # Here we will pre-compute the full stack of ALL factors: (Time, Assets, All_Factors)
    all_factors_stack = np.stack([std_factors[f].values for f in factor_names_list], axis=-1)
    
    close_prices = ohlcv['close'][etf_codes].ffill().bfill().values
    open_prices = ohlcv['open'][etf_codes].ffill().bfill().values
    high_prices = ohlcv['high'][etf_codes].ffill().bfill().values
    low_prices = ohlcv['low'][etf_codes].ffill().bfill().values

    # 5. Compute Timing Signal
    timing_module = LightTimingModule(
        extreme_threshold=EXTREME_THRESHOLD,
        extreme_position=EXTREME_POSITION,
    )
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_arr_raw = timing_series.reindex(dates).fillna(1.0).values
    timing_arr = shift_timing_signal(timing_arr_raw)

    # 6. Generate Combinations
    print("\nGenerating Combinations...")
    combos = []
    for r in [2, 3, 4, 5]:
        for c in combinations(range(len(factor_names_list)), r):
            combos.append(c)
    
    print(f"Total Combinations: {len(combos)}")

    # 7. Run Backtest Loop
    results = []
    
    # We can pass the full stack and just indices to the kernel?
    # run_vec_backtest takes `factors_3d` which corresponds to the specific combo.
    # To avoid memory allocation in loop, we can try to slice.
    # But run_vec_backtest might expect the last dim to be exactly the factors in combo.
    # Let's just slice it. It's fast enough.
    
    for combo_indices in tqdm(combos, desc="Running Backtests"):
        # Slice the specific factors for this combo
        # all_factors_stack is (T, N, F_total)
        # We want (T, N, F_subset)
        current_factors = all_factors_stack[..., list(combo_indices)]
        
        # We pass indices 0..k-1 because current_factors only has k factors
        current_factor_indices = list(range(len(combo_indices)))
        
        try:
            ret, wr, pf, trades, _, risk = run_vec_backtest(
                current_factors, close_prices, open_prices, high_prices, low_prices,
                timing_arr, current_factor_indices,
                freq=FREQ,
                pos_size=POS_SIZE,
                initial_capital=float(backtest_config['initial_capital']),
                commission_rate=float(backtest_config['commission_rate']),
                lookback=backtest_config['lookback'],
                trailing_stop_pct=0.0,
                stop_on_rebalance_only=True,
            )
            
            combo_name = " + ".join([factor_names_list[i] for i in combo_indices])
            
            results.append({
                'combo': combo_name,
                'size': len(combo_indices),
                'vec_return': ret,
                'vec_max_drawdown': risk['max_drawdown'],
                'vec_calmar_ratio': risk['calmar_ratio'],
                'vec_sharpe_ratio': risk['sharpe_ratio'],
                'vec_trades': trades
            })
        except Exception as e:
            # print(f"Error in combo {combo_indices}: {e}")
            continue

    # 8. Save and Report
    df = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / 'results' / f'vec_full_space_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'full_space_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Top 20 Analysis
    df_sorted = df.sort_values('vec_calmar_ratio', ascending=False)
    
    print('\n' + '=' * 80)
    print('🏆 TOP 20 STRATEGIES (Sorted by Calmar)')
    print('=' * 80)
    print(f"{'Rank':<4} | {'Return':<8} | {'MDD':<8} | {'Calmar':<8} | {'Sharpe':<8} | {'Combo'}")
    print('-' * 80)
    
    for i, (_, row) in enumerate(df_sorted.head(20).iterrows()):
        print(f"{i+1:<4} | {row['vec_return']*100:>7.2f}% | {row['vec_max_drawdown']*100:>7.2f}% | {row['vec_calmar_ratio']:>8.3f} | {row['vec_sharpe_ratio']:>8.3f} | {row['combo'][:50]}")

    # Check specific target
    target_combo = 'CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D'
    target_row = df[df['combo'] == target_combo]
    
    print('\n' + '=' * 80)
    print('🔍 TARGET STRATEGY PERFORMANCE')
    print('=' * 80)
    if not target_row.empty:
        row = target_row.iloc[0]
        rank = df_sorted.index.get_loc(target_row.index[0]) + 1
        print(f"Rank: {rank} / {len(df)}")
        print(f"Return: {row['vec_return']*100:.2f}%")
        print(f"MDD:    {row['vec_max_drawdown']*100:.2f}%")
        print(f"Calmar: {row['vec_calmar_ratio']:.3f}")
    else:
        print("Target strategy not found (check factor names matching).")

if __name__ == '__main__':
    main()
