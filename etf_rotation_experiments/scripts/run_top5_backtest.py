#!/usr/bin/env python3
"""
Top-5 模型对比回测脚本
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from strategies.backtest.production_backtest import backtest_no_lookahead
from real_backtest.run_profit_backtest import apply_slippage_to_nav

# Configuration
TOP_5_COMBOS = [
    ["ADX_14D", "CMF_20D", "CORRELATION_TO_MARKET_20D", "RET_VOL_20D", "RSI_14"],
    ["CMF_20D", "MAX_DD_60D", "RSI_14", "SHARPE_RATIO_20D"],
    ["ADX_14D", "CMF_20D", "RET_VOL_20D", "RSI_14", "SHARPE_RATIO_20D"],
    ["ADX_14D", "CMF_20D", "PRICE_POSITION_20D", "RELATIVE_STRENGTH_VS_MARKET_20D", "RSI_14"],
    ["ADX_14D", "CMF_20D", "RELATIVE_STRENGTH_VS_MARKET_20D", "RET_VOL_20D", "RSI_14"]
]
REBALANCE_FREQ = 8
SLIPPAGE_BPS = 5.0
INITIAL_CAPITAL = 100_000.0

def main():
    print("=" * 80)
    print("🚀 Top-5 模型对比回测 (Top-5 Models Comparative Backtest)")
    print("=" * 80)
    
    # 1. Load Data
    print("1. 加载数据 (Loading Data)...")
    repo_root = PROJECT_ROOT.parent
    loader = DataLoader(
        data_dir=repo_root / "raw" / "ETF" / "daily",
        cache_dir=repo_root / "raw" / "cache"
    )
    
    # Try to load ETF list from config or directory
    import yaml
    pool_config = repo_root / "configs" / "etf_pools.yaml"
    if pool_config.exists():
        with open(pool_config) as f:
            conf = yaml.safe_load(f)
            etf_codes = conf.get("etf_pool", [])
    else:
        etf_codes = [f.stem for f in (repo_root / "raw" / "ETF" / "daily").glob("*.csv")]
        
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes)
    print(f"   数据覆盖: {len(ohlcv['close'].columns)} 只ETF, {len(ohlcv['close'])} 个交易日")

    # 2. Compute Factors
    print("2. 计算因子 (Computing Factors)...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # 3. Cross-Section Processing
    print("3. 横截面处理 (Cross-Section Processing)...")
    processor = CrossSectionProcessor(lower_percentile=2.5, upper_percentile=97.5)
    factors_dict = {name: factors_df[name] for name in lib.list_factors()}
    standardized = processor.process_all_factors(factors_dict)
    
    # 4. Run Backtests
    print("4. 执行回测 (Running Backtests)...")
    T, N = ohlcv['close'].shape
    returns = ohlcv['close'].pct_change(fill_method=None).values
    etf_names = list(ohlcv['close'].columns)
    
    results = []
    
    for i, combo in enumerate(TOP_5_COMBOS):
        print(f"   Running Rank {i+1}: {' + '.join(combo)}")
        F = len(combo)
        factors_data = np.zeros((T, N, F))
        for j, fname in enumerate(combo):
            if fname in standardized:
                factors_data[:, :, j] = standardized[fname].values
            else:
                print(f"❌ 错误: 因子 {fname} 未找到!")
                continue
        
        base_result = backtest_no_lookahead(
            factors_data=factors_data,
            returns=returns,
            etf_names=etf_names,
            rebalance_freq=REBALANCE_FREQ,
            lookback_window=252,
            position_size=5,
            initial_capital=INITIAL_CAPITAL,
            commission_rate=0.00005,
            factors_data_full=factors_data
        )
        
        final_result = apply_slippage_to_nav(
            base_result, 
            slippage_rate=SLIPPAGE_BPS/10000.0, 
            freq=REBALANCE_FREQ
        )
        
        results.append({
            "Rank": i + 1,
            "Factors": len(combo),
            "Ann. Ret": final_result['annual_ret_net'],
            "Max DD": final_result['max_dd_net'],
            "Sharpe": final_result['sharpe_net'],
            "Total Ret": final_result['total_ret_net']
        })

    # 5. Report Results
    print("\n" + "=" * 80)
    print("📊 Top-5 组合表现对比 (Performance Comparison)")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Factors':<7} | {'Ann. Ret':<10} | {'Max DD':<10} | {'Sharpe':<8} | {'Total Ret':<10}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['Rank']:<5} | {res['Factors']:<7} | {res['Ann. Ret']*100:9.2f}% | {res['Max DD']*100:9.2f}% | {res['Sharpe']:8.3f} | {res['Total Ret']*100:9.2f}%")
    print("-" * 80)
    
    # Recommendation
    best_sharpe = max(results, key=lambda x: x['Sharpe'])
    print(f"🏆 最佳 Sharpe: Rank {best_sharpe['Rank']} (Sharpe: {best_sharpe['Sharpe']:.3f})")
    
    best_ret = max(results, key=lambda x: x['Ann. Ret'])
    print(f"💰 最佳 收益: Rank {best_ret['Rank']} (Ann. Ret: {best_ret['Ann. Ret']*100:.2f}%)")
    
    best_dd = max(results, key=lambda x: x['Max DD']) # Max DD is negative, so max is closest to 0
    print(f"🛡️ 最佳 风控: Rank {best_dd['Rank']} (Max DD: {best_dd['Max DD']*100:.2f}%)")

if __name__ == "__main__":
    main()
