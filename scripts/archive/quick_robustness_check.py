#!/usr/bin/env python3
"""
快速稳健性验证脚本
对候选策略组合进行不同step_size的OOS滑窗测试
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.factor_library import FactorLibrary
from etf_strategy.real_backtest.run_production_backtest import backtest_no_lookahead


def quick_robustness_check(
    candidates_csv: str,
    step_sizes: list = [80, 100],
    is_period: int = 252,
    oos_period: int = 60,
    position_size: int = 5,
):
    """
    对候选组合进行快速稳健性验证
    
    Args:
        candidates_csv: 候选组合CSV路径
        step_sizes: OOS步长列表（天数）
        is_period: IS期长度
        oos_period: OOS期长度
        position_size: 持仓数
    """
    # Load candidates
    candidates = pd.read_csv(candidates_csv)
    print(f"Loaded {len(candidates)} candidate combos from {candidates_csv}\n")
    
    # Load data once
    print("Loading data...")
    loader = DataLoader()
    prices_df = loader.load_prices()
    factor_lib = FactorLibrary(prices_df)
    factors_data = factor_lib.get_all_factors()
    print(f"Data loaded: {len(prices_df)} days, {factors_data.shape[1]} factors\n")
    
    results = []
    
    for idx, row in candidates.iterrows():
        combo_str = row['combo']
        factors = [f.strip() for f in combo_str.split('+')]
        freq = int(row['freq'])
        
        print(f"\n{'='*80}")
        print(f"[{idx+1}/{len(candidates)}] {combo_str[:70]}...")
        print(f"Freq={freq}d, Original Sharpe={row['sharpe']:.3f}, MaxDD={row['max_dd']:.3f}")
        print(f"{'='*80}")
        
        for step_size in step_sizes:
            try:
                # Run backtest with different step_size
                metrics = backtest_no_lookahead(
                    factors_data=factors_data,
                    prices_df=prices_df,
                    factors=factors,
                    freq=freq,
                    position_size=position_size,
                    ic_window=20,
                    is_period=is_period,
                    oos_period=oos_period,
                    step_size=step_size,
                    commission=0.0015,
                    slippage=0.001,
                )
                
                print(f"  Step={step_size}d: Sharpe={metrics['sharpe']:.3f}, "
                      f"MaxDD={metrics['max_dd']:.3f}, "
                      f"AnnRet={metrics['annual_ret']:.3f}, "
                      f"Rebal={metrics['n_rebalance']}")
                
                results.append({
                    'rank': row['rank'],
                    'combo': combo_str,
                    'family': row['family'],
                    'freq': freq,
                    'step_size': step_size,
                    'sharpe': metrics['sharpe'],
                    'max_dd': metrics['max_dd'],
                    'annual_ret': metrics['annual_ret'],
                    'vol': metrics['vol'],
                    'calmar': metrics['calmar_ratio'],
                    'n_rebalance': metrics['n_rebalance'],
                    'win_rate': metrics['win_rate'],
                    'original_sharpe': row['sharpe'],
                    'sharpe_diff': metrics['sharpe'] - row['sharpe'],
                })
                
            except Exception as e:
                print(f"  ❌ Step={step_size}d failed: {e}")
                results.append({
                    'rank': row['rank'],
                    'combo': combo_str,
                    'family': row['family'],
                    'freq': freq,
                    'step_size': step_size,
                    'sharpe': np.nan,
                    'max_dd': np.nan,
                    'annual_ret': np.nan,
                    'vol': np.nan,
                    'calmar': np.nan,
                    'n_rebalance': 0,
                    'win_rate': np.nan,
                    'original_sharpe': row['sharpe'],
                    'sharpe_diff': np.nan,
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = 'production/robustness_check_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\n✅ Results saved to {output_path}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("ROBUSTNESS SUMMARY")
    print(f"{'='*80}")
    
    for combo in results_df['combo'].unique():
        combo_results = results_df[results_df['combo'] == combo]
        print(f"\n{combo[:70]}...")
        print(f"  Original Sharpe: {combo_results['original_sharpe'].iloc[0]:.3f}")
        for _, r in combo_results.iterrows():
            status = "✅" if abs(r['sharpe_diff']) < 0.2 else "⚠️"
            print(f"  {status} Step={r['step_size']}d: Sharpe={r['sharpe']:.3f} "
                  f"(Δ={r['sharpe_diff']:+.3f}), MaxDD={r['max_dd']:.3f}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    
    # Flag risky combos (large Sharpe variation)
    combo_variance = results_df.groupby('combo')['sharpe'].std()
    risky = combo_variance[combo_variance > 0.15].index.tolist()
    stable = combo_variance[combo_variance <= 0.15].index.tolist()
    
    print(f"\nStable combos (Sharpe std <= 0.15): {len(stable)}")
    for c in stable:
        avg_sharpe = results_df[results_df['combo'] == c]['sharpe'].mean()
        print(f"  ✅ {c[:65]}... (avg Sharpe={avg_sharpe:.3f})")
    
    print(f"\nRisky combos (Sharpe std > 0.15): {len(risky)}")
    for c in risky:
        std_sharpe = combo_variance[c]
        print(f"  ⚠️ {c[:65]}... (Sharpe std={std_sharpe:.3f})")
    
    return results_df


if __name__ == "__main__":
    results = quick_robustness_check(
        candidates_csv='production/strategy_candidates_selected.csv',
        step_sizes=[80, 100],
        is_period=252,
        oos_period=60,
        position_size=5,
    )
