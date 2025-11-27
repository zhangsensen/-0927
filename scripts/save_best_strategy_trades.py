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

# Rank 8983
COMBO = "SHARPE_RATIO_20D + VORTEX_14D"
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002 # 2 bps

def save_best_strategy_trades():
    print("="*80)
    print(f"🌟 生成最佳策略交易明细 (Best Strategy Trade Log)")
    print(f"🎯 策略: {COMBO}")
    print("="*80)
    
    # 1. Load Data
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
    
    # 3. Simulation
    T, N, F = F_sel.shape
    lookback = 252
    cash = INITIAL_CAPITAL
    holdings = {} 
    trade_log = []
    
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
            
            # Fix: Allow selecting fewer than POS_SIZE if not enough valid data
            # Original strict audit was: if np.sum(valid_mask) >= POS_SIZE: ... else: []
            # This caused excessive cash holding when only 1 or 2 ETFs had valid data.
            
            if np.sum(valid_mask) > 0:
                sorted_indices = np.argsort(combined_score[valid_mask])
                # Take up to POS_SIZE
                top_k_local = sorted_indices[-min(POS_SIZE, np.sum(valid_mask)):]
                valid_indices = np.where(valid_mask)[0]
                target_indices = valid_indices[top_k_local].tolist()
            else:
                target_indices = []
                
            timing_ratio = timing_arr[t]
            
            # Sell
            current_indices = list(holdings.keys())
            for idx in current_indices:
                if idx not in target_indices:
                    info = holdings[idx]
                    shares = info['shares']
                    price = close_prices[t, idx]
                    proceeds = shares * price * (1 - COMMISSION_RATE)
                    cash += proceeds
                    
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
            
            # Buy & Rebalance Logic
            # Active Rebalancing: Adjust all target positions to target weight
            
            # 1. Calculate target value per position
            # Note: We use (current_value * timing_ratio) / len(target_indices) if we want full investment?
            # Or fixed POS_SIZE?
            # Standard is fixed POS_SIZE (1/3 of capital).
            # If only 2 targets, we invest 2/3, hold 1/3 cash.
            
            target_pos_value = (current_value * timing_ratio) / POS_SIZE
            
            for idx in target_indices:
                price = close_prices[t, idx]
                if np.isnan(price): continue
                
                if idx in holdings:
                    # Rebalance
                    current_shares = holdings[idx]['shares']
                    current_pos_val = current_shares * price
                    diff_val = target_pos_value - current_pos_val
                    
                    # Threshold to avoid tiny trades (e.g. 1%)
                    if abs(diff_val) > target_pos_value * 0.05:
                        if diff_val > 0:
                            # Buy more
                            cost = diff_val * (1 + COMMISSION_RATE)
                            if cash >= cost:
                                shares_add = diff_val / price
                                cash -= cost
                                holdings[idx]['shares'] += shares_add
                                # Update entry price? Weighted average?
                                # For PnL tracking, weighted average is best.
                                old_shares = current_shares
                                old_price = holdings[idx]['entry_price']
                                new_avg = (old_shares * old_price + shares_add * price) / (old_shares + shares_add)
                                holdings[idx]['entry_price'] = new_avg
                        else:
                            # Sell some
                            sell_val = -diff_val
                            shares_sell = sell_val / price
                            proceeds = sell_val * (1 - COMMISSION_RATE)
                            cash += proceeds
                            holdings[idx]['shares'] -= shares_sell
                            # Entry price doesn't change on sell
                else:
                    # New Buy
                    shares_to_buy = target_pos_value / price
                    cost = shares_to_buy * price * (1 + COMMISSION_RATE)
                    if cash >= cost:
                        cash -= cost
                        holdings[idx] = {
                            'shares': shares_to_buy,
                            'entry_price': price,
                            'entry_date': current_date
                        }
                    elif cash > 0:
                        # Buy with remaining cash
                        shares_to_buy = cash / (price * (1 + COMMISSION_RATE))
                        cash = 0
                        holdings[idx] = {
                            'shares': shares_to_buy,
                            'entry_price': price,
                            'entry_date': current_date
                        }

    # Close all
    final_date = dates[-1]
    for idx, info in holdings.items():
        price = close_prices[-1, idx]
        if np.isnan(price): price = info['entry_price']
        pnl = (price - info['entry_price']) / info['entry_price']
        pnl_amount = (price - info['entry_price']) * info['shares']
        hold_days = (final_date - info['entry_date']).days
        
        # Sell to update cash
        proceeds = info['shares'] * price * (1 - COMMISSION_RATE)
        cash += proceeds
        
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

    df_trades = pd.DataFrame(trade_log)
    output_file = "results/best_snowball_trades_8983.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"💾 交易记录已保存: {output_file}")
    
    # Print Stats
    print(f"\n📊 策略 8983 最终统计 (修正后):")
    print(f"   交易次数: {len(df_trades)}")
    print(f"   胜率: {len(df_trades[df_trades['pnl_pct']>0])/len(df_trades):.2%}")
    wins = df_trades[df_trades['pnl_pct'] > 0]['pnl_amount'].sum()
    losses = abs(df_trades[df_trades['pnl_pct'] <= 0]['pnl_amount'].sum())
    print(f"   盈亏比: {wins/losses:.2f}")
    
    # Calculate Total Return
    final_value = cash
    # (Already closed all positions)
    total_ret = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    print(f"   总收益率: {total_ret:.2%}")
    
    # Annualized
    days = (dates[-1] - dates[lookback]).days
    years = days / 365.25
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    print(f"   年化收益: {ann_ret:.2%}")

if __name__ == "__main__":
    save_best_strategy_trades()
