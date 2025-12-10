import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import date, datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor
from etf_strategy.core.market_timing import LightTimingModule
from etf_strategy.core.utils.rebalance import shift_timing_signal, generate_rebalance_schedule
from etf_strategy.auditor.core.engine import GenericStrategy, PandasData

def main():
    print("="*80)
    print("🔍 RECENT PERFORMANCE ANALYSIS (2025-10-01 to 2025-12-08)")
    print("="*80)

    # 1. Configuration
    DATA_DIR = ROOT / 'raw/ETF/daily_qmt_aligned'
    CACHE_DIR = ROOT / '.cache'
    
    # Top 1 Strategy from Selection
    COMBO = ['MAX_DD_60D', 'MOM_20D', 'PRICE_POSITION_120D', 'PRICE_POSITION_20D', 'SHARPE_RATIO_20D']
    FREQ = 3
    POS_SIZE = 2
    
    print(f"Strategy: {COMBO}")
    print(f"Freq: {FREQ}, Pos: {POS_SIZE}")

    # 2. Load Data
    print("\nLoading Data...")
    loader = DataLoader(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    
    # Load all available ETFs in the directory
    etf_files = list(DATA_DIR.glob("*.parquet"))
    # Extract clean codes (e.g. 159801.SZ from 159801.SZ_daily_...)
    # DataLoader strips suffixes, so we should too for consistency with ohlcv keys
    etf_codes_with_suffix = [f.stem.split('_')[0] for f in etf_files]
    etf_codes_clean = [c.split('.')[0] for c in etf_codes_with_suffix]
    
    # Load using clean codes (DataLoader handles matching)
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes_clean)
    
    # DEBUG: Check price of 518850
    if '518850' in ohlcv['close'].columns:
        print(f"DEBUG: 518850 close on 2025-12-08: {ohlcv['close']['518850'].get('2025-12-08')}")
        print(f"DEBUG: 518850 open on 2025-12-08: {ohlcv['open']['518850'].get('2025-12-08')}")
    
    # 3. Compute Factors
    print("Computing Factors...")
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    
    # Extract relevant factors
    raw_factors = {f: raw_factors_df[f] for f in COMBO}
    
    # Process Factors (Standardization)
    processor = CrossSectionProcessor(verbose=False)
    std_factors = processor.process_all_factors(raw_factors)
    
    # Combine Factors (Sum)
    combined_score = pd.DataFrame(0.0, index=std_factors[COMBO[0]].index, columns=std_factors[COMBO[0]].columns)
    for f in COMBO:
        combined_score += std_factors[f].fillna(0.0)
        
    # 4. Timing Signal
    print("Computing Timing Signal...")
    timing_module = LightTimingModule(extreme_threshold=-0.1, extreme_position=0.1)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    
    # Shift signals (No Lookahead)
    # shift_timing_signal returns numpy array, we need to preserve index for GenericStrategy
    timing_values = shift_timing_signal(timing_series.values)
    timing_series = pd.Series(timing_values, index=timing_series.index)
    
    print(f"DEBUG: Timing Series Tail:\n{timing_series.tail()}")
    
    combined_score = combined_score.shift(1) 
    
    # 5. Run Backtrader
    print("Running Backtest...")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000.0)
    cerebro.broker.setcommission(commission=0.0002)
    cerebro.broker.set_coc(True) # Cheat-On-Close
    
    # Add Data Feeds
    data_feeds = {}
    # Use clean codes for iteration
    for code in etf_codes_clean:
        if code not in ohlcv['open'].columns:
            continue
            
        df = pd.DataFrame({
            'open': ohlcv['open'][code],
            'high': ohlcv['high'][code],
            'low': ohlcv['low'][code],
            'close': ohlcv['close'][code],
            'volume': ohlcv['volume'][code],
        }).dropna()
        
        if not df.empty:
            if code == '518850':
                print(f"DEBUG: DataFrame for 518850 tail:\n{df.tail()}")
            data = PandasData(dataname=df, name=code)
            cerebro.adddata(data)
            data_feeds[code] = df

    # Generate Rebalance Schedule
    rebalance_schedule = generate_rebalance_schedule(
        len(combined_score.index), 
        lookback_window=252,
        freq=FREQ
    )

    # Add Strategy
    cerebro.addstrategy(
        GenericStrategy,
        scores=combined_score,
        timing=timing_series,
        etf_codes=etf_codes_clean,
        freq=FREQ,
        pos_size=POS_SIZE,
        rebalance_schedule=rebalance_schedule,
        dynamic_leverage_enabled=False # Disable for simplicity/robustness as per v3.1
    )
    
    # Add Analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    
    results = cerebro.run()
    strat = results[0]
    
    # 6. Analyze Recent Performance
    print("\n" + "="*80)
    print("📅 RECENT PERFORMANCE REPORT (2025-10-01 to 2025-12-08)")
    print("="*80)
    
    # Get Daily Returns
    timereturns = strat.analyzers.timereturn.get_analysis()
    
    # Filter for recent period
    # Handle datetime vs date comparison
    start_date = date(2025, 10, 1)
    recent_returns = {}
    for k, v in timereturns.items():
        k_date = k.date() if isinstance(k, datetime) else k
        if k_date >= start_date:
            recent_returns[k_date] = v
            
    sorted_dates = sorted(recent_returns.keys())
    
    cum_ret = 1.0
    max_dd = 0.0
    peak = 1.0
    
    print(f"{'Date':<12} | {'Daily Ret':<10} | {'Cum Ret':<10} | {'Drawdown':<10}")
    print("-" * 50)
    
    for d in sorted_dates:
        r = recent_returns[d]
        cum_ret *= (1 + r)
        if cum_ret > peak:
            peak = cum_ret
        dd = (peak - cum_ret) / peak
        if dd > max_dd:
            max_dd = dd
            
        print(f"{d} | {r:>9.2%} | {cum_ret-1:>9.2%} | {dd:>9.2%}")
        
    print("-" * 50)
    print(f"Period Return: {cum_ret - 1:.2%}")
    print(f"Max Drawdown:  {max_dd:.2%}")
    
    # Print Recent Trades
    print("\n📋 RECENT TRANSACTIONS")
    print("-" * 80)
    print(f"{'Date':<12} | {'Type':<6} | {'Ticker':<10} | {'Price':<10} | {'Value':<12}")
    print("-" * 80)
    
    # strat.orders contains all orders. Filter for recent ones.
    # Note: GenericStrategy stores orders in self.orders list as dicts
    recent_orders = [o for o in strat.orders if o['date'] >= date(2025, 10, 1)]
    
    for o in recent_orders:
        print(f"{o['date']} | {o['type']:<6} | {o['ticker']:<10} | {o['price']:>10.3f} | {o['value']:>12.2f}")

if __name__ == "__main__":
    main()
