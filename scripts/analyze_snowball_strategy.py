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
    sys.path.append(str(Path(os.getcwd()).parent))
    from etf_rotation_optimized.core.data_loader import DataLoader
    from etf_rotation_optimized.core.precise_factor_library_v2 import PreciseFactorLibrary
    from etf_rotation_optimized.core.cross_section_processor import CrossSectionProcessor
    from etf_rotation_optimized.core.market_timing import LightTimingModule

# Snowball Strategy
COMBO = "PV_CORR_20D + SHARPE_RATIO_20D + VORTEX_14D"
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002 # 2 bps

def analyze_snowball_strategy():
    print("="*80)
    print(f"❄️  深度审计「滚雪球」策略 (Snowball Strategy Audit)")
    print(f"🎯 策略: {COMBO}")
    print(f"⚙️  参数: Freq={FREQ}, Pos={POS_SIZE}, Comm={COMMISSION_RATE:.4%}")
    print("="*80)
    
    # 1. Load Data
    print("📥 Loading Data...")
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
    
    # 2. Factors
    print("🧮 Computing Factors...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    factors_dict = {}
    available_factors = factors_df.columns.get_level_values(0).unique()
    for f in available_factors:
        factors_dict[f] = factors_df[f]
        
    proc = CrossSectionProcessor(
        lower_percentile=config["cross_section"]["winsorize_lower"] * 100,
        upper_percentile=config["cross_section"]["winsorize_upper"] * 100,
        verbose=False
    )
    std_factors = proc.process_all_factors(factors_dict)
    
    factor_names = sorted(std_factors.keys())
    combo_factors = [f.strip() for f in COMBO.split(" + ")]
    factor_indices = [factor_names.index(f) for f in combo_factors]
    
    F_data = np.stack([std_factors[n].values for n in factor_names], axis=-1)
    F_sel = F_data[:, :, factor_indices]
    
    # Data Arrays
    close_prices = ohlcv["close"].values
    dates = ohlcv["close"].index
    etf_codes = ohlcv["close"].columns.tolist()
    
    # Timing
    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_series = timing_series.shift(1).fillna(1.0)
    timing_arr = timing_series.values
    
    # 3. Simulation with Trade Tracking
    T, N, F = F_sel.shape
    lookback = 252
    
    # State
    cash = INITIAL_CAPITAL
    holdings = {} # {etf_idx: {'shares': s, 'entry_price': p, 'entry_date': d}}
    trade_log = []
    
    print("🚀 Running Simulation (Strict T+1)...")
    
    for t in range(lookback, T):
        current_date = dates[t]
        
        # Mark to Market
        current_value = cash
        for idx, info in holdings.items():
            current_value += info['shares'] * close_prices[t, idx]
            
        if t % FREQ == 0:
            # Signal T-1
            combined_score = np.sum(F_sel[t-1], axis=1)
            valid_mask = ~np.isnan(combined_score)
            
            if np.sum(valid_mask) >= POS_SIZE:
                sorted_indices = np.argsort(combined_score[valid_mask])
                top_k_local = sorted_indices[-POS_SIZE:]
                valid_indices = np.where(valid_mask)[0]
                target_indices = valid_indices[top_k_local].tolist()
            else:
                target_indices = []
                
            timing_ratio = timing_arr[t]
            
            # Sell Logic
            current_indices = list(holdings.keys())
            for idx in current_indices:
                if idx not in target_indices:
                    # Sell
                    info = holdings[idx]
                    shares = info['shares']
                    price = close_prices[t, idx]
                    proceeds = shares * price * (1 - COMMISSION_RATE)
                    cash += proceeds
                    
                    # Log
                    pnl = (price - info['entry_price']) / info['entry_price']
                    pnl_amount = (price - info['entry_price']) * shares
                    hold_days = (current_date - info['entry_date']).days
                    
                    trade_log.append({
                        'ticker': etf_codes[idx],
                        'entry_date': info['entry_date'],
                        'exit_date': current_date,
                        'entry_price': info['entry_price'],
                        'exit_price': price,
                        'shares': shares,
                        'pnl_pct': pnl,
                        'pnl_amount': pnl_amount,
                        'hold_days': hold_days,
                        'exit_reason': 'Rebalance'
                    })
                    
                    del holdings[idx]
            
            # Buy Logic
            target_pos_value = (current_value * timing_ratio) / POS_SIZE
            
            for idx in target_indices:
                price = close_prices[t, idx]
                if np.isnan(price): continue
                
                if idx in holdings:
                    # Hold
                    pass
                else:
                    # Buy
                    shares_to_buy = target_pos_value / price
                    cost = shares_to_buy * price * (1 + COMMISSION_RATE)
                    
                    if cash >= cost:
                        cash -= cost
                        holdings[idx] = {
                            'shares': shares_to_buy,
                            'entry_price': price,
                            'entry_date': current_date
                        }
                    else:
                        if cash > 0:
                            shares_to_buy = cash / (price * (1 + COMMISSION_RATE))
                            cash = 0
                            holdings[idx] = {
                                'shares': shares_to_buy,
                                'entry_price': price,
                                'entry_date': current_date
                            }

    # Close all at end
    final_date = dates[-1]
    for idx, info in holdings.items():
        price = close_prices[-1, idx]
        pnl = (price - info['entry_price']) / info['entry_price']
        pnl_amount = (price - info['entry_price']) * info['shares']
        hold_days = (final_date - info['entry_date']).days
        
        trade_log.append({
            'ticker': etf_codes[idx],
            'entry_date': info['entry_date'],
            'exit_date': final_date,
            'entry_price': info['entry_price'],
            'exit_price': price,
            'shares': info['shares'],
            'pnl_pct': pnl,
            'pnl_amount': pnl_amount,
            'hold_days': hold_days,
            'exit_reason': 'End of Backtest'
        })

    # Analysis
    df_trades = pd.DataFrame(trade_log)
    if df_trades.empty:
        print("No trades generated.")
        return

    print(f"\n📊 交易统计 (Trade Stats):")
    print(f"   总交易数: {len(df_trades)}")
    print(f"   胜率 (Win Rate): {len(df_trades[df_trades['pnl_pct']>0]) / len(df_trades):.2%}")
    
    wins = df_trades[df_trades['pnl_pct'] > 0]['pnl_amount'].sum()
    losses = abs(df_trades[df_trades['pnl_pct'] <= 0]['pnl_amount'].sum())
    pf = wins / losses if losses > 0 else float('inf')
    
    print(f"   盈亏比 (Profit Factor): {pf:.2f}")
    print(f"   平均盈利: {df_trades[df_trades['pnl_pct']>0]['pnl_pct'].mean():.2%}")
    print(f"   平均亏损: {df_trades[df_trades['pnl_pct']<=0]['pnl_pct'].mean():.2%}")
    print(f"   最大单笔盈利: {df_trades['pnl_pct'].max():.2%}")
    print(f"   最大单笔亏损: {df_trades['pnl_pct'].min():.2%}")
    
    # Save
    df_trades.to_csv("results/snowball_strategy_trades.csv", index=False)
    print(f"\n💾 交易记录已保存: results/snowball_strategy_trades.csv")
    
    # Show recent
    print("\n📅 最近 10 笔交易:")
    print(df_trades.tail(10)[['entry_date', 'exit_date', 'ticker', 'pnl_pct', 'hold_days']].to_string())

if __name__ == "__main__":
    analyze_snowball_strategy()
