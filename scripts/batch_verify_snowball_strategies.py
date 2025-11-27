import os
import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
import time

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

# Constants
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002 # 2 bps

def run_strict_backtest(combo_name, ohlcv, factors_dict, timing_arr):
    """
    Runs a strict T+1 backtest for a specific combo.
    Returns metrics and trade log.
    """
    # Prepare Factors
    factor_names = sorted(factors_dict.keys())
    combo_factors = [f.strip() for f in combo_name.split(" + ")]
    
    # Check if factors exist
    for f in combo_factors:
        if f not in factor_names:
            return None, None
            
    factor_indices = [factor_names.index(f) for f in combo_factors]
    
    # Stack Data
    # We assume factors_dict values are already aligned DataFrames
    # But we need to be careful about alignment if we just take .values
    # The caller should ensure factors_dict contains aligned DFs
    
    # To be safe, let's stack using the first factor's index/columns
    first_factor = factors_dict[factor_names[0]]
    T, N = first_factor.shape
    
    # Create 3D array: (T, N, F_total)
    # This might be memory intensive if we do it for ALL factors every time.
    # Optimization: Only stack the factors we need for this combo.
    
    F_sel_list = []
    for f in combo_factors:
        F_sel_list.append(factors_dict[f].values)
    
    F_sel = np.stack(F_sel_list, axis=-1) # (T, N, F_subset)
    
    # Data Arrays
    close_prices = ohlcv["close"].values
    dates = ohlcv["close"].index
    etf_codes = ohlcv["close"].columns.tolist()
    
    lookback = 252
    
    # State
    cash = INITIAL_CAPITAL
    holdings = {} # {etf_idx: {'shares': s, 'entry_price': p, 'entry_date': d}}
    trade_log = []
    
    # Simulation
    for t in range(lookback, T):
        current_date = dates[t]
        
        # Mark to Market
        current_value = cash
        for idx, info in holdings.items():
            current_value += info['shares'] * close_prices[t, idx]
            
        if t % FREQ == 0:
            # Signal T-1
            # Sum of factors (equal weight)
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
        if np.isnan(price): price = info['entry_price'] # Fallback
        
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
    
    # Clear holdings
    holdings = {}
        
    # Calculate Metrics
    df_trades = pd.DataFrame(trade_log)
    if df_trades.empty:
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'avg_pnl': 0.0,
            'max_dd': 0.0, # Need equity curve for this, simplified here
            'final_value': INITIAL_CAPITAL
        }, df_trades
        
    wins = df_trades[df_trades['pnl_pct'] > 0]['pnl_amount'].sum()
    losses = abs(df_trades[df_trades['pnl_pct'] <= 0]['pnl_amount'].sum())
    pf = wins / losses if losses > 0 else float('inf')
    win_rate = len(df_trades[df_trades['pnl_pct'] > 0]) / len(df_trades)
    
    # Calculate Max Drawdown from Portfolio Values?
    # We didn't store portfolio values in this simplified function to save memory/time
    # But we can approximate final value
    final_value = cash
    # (Already closed all positions)
    
    metrics = {
        'win_rate': win_rate,
        'profit_factor': pf,
        'total_trades': len(df_trades),
        'avg_pnl': df_trades['pnl_pct'].mean(),
        'final_value': final_value,
        'total_ret': (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    }
    
    return metrics, df_trades

def batch_verify():
    print("="*80)
    print("🕵️  批量严格审计「滚雪球」策略 (Batch Strict Audit)")
    print("🎯 目标: 验证 Top 20 策略在严格 T+1 条件下的真实表现")
    print("="*80)
    
    # 1. Load Candidates
    candidates_file = "results/snowball_strategies.csv"
    if not os.path.exists(candidates_file):
        print(f"❌ 找不到候选文件: {candidates_file}")
        return
        
    df_candidates = pd.read_csv(candidates_file)
    top_candidates = df_candidates.head(20) # Check Top 20
    
    print(f"📥 加载了 {len(top_candidates)} 个候选策略")
    
    # 2. Load Data & Precompute Factors (Once)
    print("📥 Loading Market Data...")
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
    
    print("🧮 Computing All Factors (This may take a moment)...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # Process Factors
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
    
    # Timing
    timing_module = LightTimingModule(extreme_threshold=-0.3, extreme_position=0.3)
    timing_series = timing_module.compute_position_ratios(ohlcv['close'])
    timing_series = timing_series.shift(1).fillna(1.0)
    timing_arr = timing_series.values
    
    print("✅ 数据准备就绪，开始批量回测...")
    print("-" * 100)
    
    results = []
    
    for i, row in top_candidates.iterrows():
        combo = row['combo']
        print(f"[{i+1}/{len(top_candidates)}] Testing: {combo[:50]}...")
        
        metrics, trade_log = run_strict_backtest(combo, ohlcv, std_factors, timing_arr)
        
        if metrics:
            # Compare with expected
            expected_wr = row['win_rate']
            if expected_wr < 1.0: expected_wr *= 100
            
            real_wr = metrics['win_rate'] * 100
            
            results.append({
                'rank': row['rank'],
                'combo': combo,
                'exp_win_rate': expected_wr,
                'real_win_rate': real_wr,
                'exp_pf': row['profit_factor'],
                'real_pf': metrics['profit_factor'],
                'real_ret': metrics['total_ret'],
                'trades': metrics['total_trades']
            })
            
            # Save trade log for the top 1 specifically
            if i == 0:
                trade_log.to_csv("results/audit_top1_trades.csv", index=False)
    
    # 3. Report
    df_res = pd.DataFrame(results)
    df_res['wr_diff'] = df_res['real_win_rate'] - df_res['exp_win_rate']
    df_res['pf_diff'] = df_res['real_pf'] - df_res['exp_pf']
    
    print("\n📊 审计报告 (Audit Report):")
    print("-" * 120)
    headers = ["Rank", "Combo (Partial)", "Real WR", "Exp WR", "Diff", "Real PF", "Exp PF", "Real Ret", "Trades"]
    print(f"{headers[0]:<5} | {headers[1]:<40} | {headers[2]:<7} | {headers[3]:<7} | {headers[4]:<5} | {headers[5]:<7} | {headers[6]:<7} | {headers[7]:<8} | {headers[8]}")
    print("-" * 120)
    
    for _, row in df_res.iterrows():
        combo_short = row['combo'][:40]
        print(f"{row['rank']:<5} | {combo_short:<40} | {row['real_win_rate']:6.2f}% | {row['exp_win_rate']:6.2f}% | {row['wr_diff']:+5.1f} | {row['real_pf']:6.2f}  | {row['exp_pf']:6.2f}  | {row['real_ret']*100:6.1f}% | {row['trades']}")
        
    # Save
    df_res.to_csv("results/snowball_verification_report.csv", index=False)
    print(f"\n💾 完整审计报告已保存: results/snowball_verification_report.csv")
    print(f"💾 Top 1 策略交易明细: results/audit_top1_trades.csv")

if __name__ == "__main__":
    batch_verify()
