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

# Top 1 Strategy from previous analysis
COMBO = "MAX_DD_60D + MOM_20D + RSI_14 + VOL_RATIO_20D + VOL_RATIO_60D"
FREQ = 8
POS_SIZE = 3
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0002 # 2 bps (more realistic than 0.5 bps for conservative check)

def analyze_trades():
    print("="*80)
    print(f"🕵️  深度交易分析 (Deep Trade Analysis)")
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
    holdings = {} # {etf_idx: shares}
    
    # Trade History
    trades = [] # {entry_date, exit_date, code, entry_price, exit_price, shares, pnl, ret, reason}
    
    # Equity Curve
    equity_curve = []
    
    print(f"\n🚀 Running Simulation ({T} days)...")
    
    # Strict T+1 Logic:
    # Signal at T-1 (using data up to T-1)
    # Execution at T Close
    
    pending_target_weights = None # Weights determined at T-1, to be executed at T
    
    for t in range(lookback + 1, T):
        date = dates[t]
        price_today = close_prices[t] # Array of N prices
        
        # 1. Calculate Portfolio Value (Mark to Market)
        port_value = cash
        for idx, shares in holdings.items():
            port_value += shares * price_today[idx]
        equity_curve.append({"date": date, "equity": port_value})
        
        # 2. Execute Pending Rebalance (from T-1 signal)
        if pending_target_weights is not None:
            # Target Value per ETF
            # Note: We use current portfolio value to determine target amounts
            # This implies we reinvest profits
            
            current_holdings_set = set(holdings.keys())
            target_indices = np.where(pending_target_weights > 0)[0]
            target_holdings_set = set(target_indices)
            
            # Sell Logic (First sell to raise cash)
            to_sell = current_holdings_set - target_holdings_set
            for idx in to_sell:
                shares = holdings.get(idx, 0.0)
                price = price_today[idx]
                proceeds = shares * price * (1 - COMMISSION_RATE)
                cash += proceeds
                
                if idx in holdings:
                    del holdings[idx]
            
            # Buy/Adjust Logic
            # For simplicity in this verification:
            # We clear ALL positions and re-buy target positions to match weights exactly
            # This is slightly inefficient (turnover) but cleaner for verification logic
            # Optimization: Only adjust diffs.
            
            # Let's implement "Adjust Diffs"
            # Target Value for each ETF
            target_val_per_etf = port_value * pending_target_weights # Vector
            
            # Sell first
            for idx in current_holdings_set:
                if idx not in holdings: continue # Already sold above
                current_val = holdings[idx] * price_today[idx]
                target_val = target_val_per_etf[idx]
                
                if current_val > target_val:
                    # Sell diff
                    sell_val = current_val - target_val
                    shares_to_sell = sell_val / price_today[idx]
                    proceeds = shares_to_sell * price_today[idx] * (1 - COMMISSION_RATE)
                    cash += proceeds
                    holdings[idx] -= shares_to_sell
                    
                    # If holding becomes negligible, remove
                    if holdings[idx] * price_today[idx] < 100: # Dust
                        cash += holdings[idx] * price_today[idx] * (1 - COMMISSION_RATE)
                        del holdings[idx]

            # Buy second
            for idx in target_indices:
                current_shares = holdings.get(idx, 0.0)
                current_val = current_shares * price_today[idx]
                target_val = target_val_per_etf[idx]
                
                if target_val > current_val:
                    buy_val = target_val - current_val
                    # Check cash
                    if cash < buy_val:
                        buy_val = cash # Cap at available cash
                    
                    if buy_val > 0:
                        cost = buy_val * (1 + COMMISSION_RATE)
                        if cash >= cost:
                            shares_to_buy = buy_val / price_today[idx]
                            cash -= cost
                            holdings[idx] = holdings.get(idx, 0.0) + shares_to_buy
                        else:
                            # Buy max possible
                            if cash > 10: # Min cash to trade
                                actual_buy_val = cash / (1 + COMMISSION_RATE)
                                shares_to_buy = actual_buy_val / price_today[idx]
                                cash = 0
                                holdings[idx] = holdings.get(idx, 0.0) + shares_to_buy

            pending_target_weights = None # Reset
            
        # 3. Generate Signal (at T, for execution at T+1)
        # Check if T is a rebalance day (based on schedule)
        # Note: The loop index 't' here represents "Today". 
        # If today is rebalance day, we generate signal using data up to Today.
        # And execute tomorrow.
        
        is_rebalance_day = (t - (lookback + 1)) % FREQ == 0
        
        if is_rebalance_day:
            # Use data up to T (Today)
            # Factors are already aligned such that F_sel[t] contains data known at T close
            f_day = F_sel[t]
            
            # Equal Weight Factors
            score = np.nanmean(f_day, axis=1)
            
            # Top N
            valid_mask = ~np.isnan(score)
            target_w = np.zeros(N)
            
            if valid_mask.sum() >= POS_SIZE:
                score_filled = score.copy()
                score_filled[~valid_mask] = -np.inf
                top_idx = np.argsort(score_filled)[-POS_SIZE:]
                
                # Apply Timing
                pos_ratio = timing_arr[t] # Signal for T (known at T close? timing_arr is shifted 1, so timing_arr[t] is signal from T-1 applied to T?)
                # Wait, timing_arr was shifted: timing_series.shift(1).
                # So timing_arr[t] is the signal generated at T-1, to be used for T.
                # But here we are generating signal at T for T+1.
                # So we should use timing_arr[t+1] if available, or re-compute timing at T.
                # Let's re-compute timing signal at T.
                # Actually, timing_arr is aligned to "Trading Day".
                # If we trade tomorrow (T+1), we need timing signal known at T.
                # timing_module.compute... returns signal aligned to Close.
                # So timing_series[t] is signal at T close.
                
                # Let's look at timing_arr again.
                # timing_series = compute... (Index=Date)
                # timing_series = timing_series.shift(1)
                # timing_arr[t] corresponds to Date[t]. It is the signal from Date[t-1].
                # If we trade at T+1, we want signal from T.
                # That would be timing_arr[t+1].
                
                # For simplicity, let's assume we use the timing signal available for "Tomorrow"
                # which is based on "Today's" close.
                # In the array, timing_arr[t+1] is based on t.
                
                next_t = t + 1
                if next_t < len(timing_arr):
                    current_timing = timing_arr[next_t]
                else:
                    current_timing = 1.0
                
                # Allocation
                # If timing < 1, we hold cash.
                # Weights sum to current_timing.
                # e.g. 0.3 -> 3 ETFs get 0.1 each.
                
                weight_per_asset = current_timing / POS_SIZE
                target_w[top_idx] = weight_per_asset
                
            pending_target_weights = target_w

    # 4. Process Trade History for Analysis
    # Since we didn't log individual trades in the loop (it's complex with partial fills/adjustments),
    # let's reconstruct "Virtual Trades" from the Weight Changes or just analyze the Daily Returns.
    # User wants "Trade Records".
    # Let's do a second pass or simplified "Trade Object" tracking.
    
    # Re-run with simplified Trade Object tracking
    print("🔄 Generating Trade Logs...")
    trades_log = []
    
    # Simulation State for Logging
    # Position: {code: {entry_date, entry_price, shares}}
    positions = {} 
    
    # We need to re-simulate or just parse the logic. 
    # Let's just do a discrete simulation for logging.
    
    curr_holdings = [] # List of codes
    
    # Re-simulate logic for logging
    # We will track "Signal Changes" as trades.
    
    # ... (Simplified logic for logging) ...
    # Actually, let's just use the equity curve to calc stats, 
    # and generate a "Signal Log" which is cleaner.
    
    signal_log = []
    
    for t in range(lookback + 1, T):
        if (t - (lookback + 1)) % FREQ == 0:
            date = dates[t]
            f_day = F_sel[t]
            score = np.nanmean(f_day, axis=1)
            valid_mask = ~np.isnan(score)
            if valid_mask.sum() >= POS_SIZE:
                score_filled = score.copy()
                score_filled[~valid_mask] = -np.inf
                top_idx = np.argsort(score_filled)[-POS_SIZE:]
                codes = [etf_codes[i] for i in top_idx]
                
                # Timing
                next_t = t + 1
                timing = timing_arr[next_t] if next_t < len(timing_arr) else 1.0
                
                signal_log.append({
                    "date": date,
                    "holdings": codes,
                    "timing": timing
                })

    # 5. Calculate Trade Stats from Signal Log
    # We can infer trades: If code is in New but not in Old -> Buy. If in Old but not New -> Sell.
    # We can look up prices at T+1 Close.
    
    real_trades = []
    prev_set = set()
    
    # Price lookup helper
    def get_price(date_obj, code):
        try:
            idx = dates.get_loc(date_obj)
            col_idx = etf_codes.index(code)
            return close_prices[idx, col_idx]
        except:
            return None

    for i in range(len(signal_log)):
        curr = signal_log[i]
        curr_date = curr["date"]
        curr_set = set(curr["holdings"]) if curr["timing"] > 0.1 else set() # If timing is low, we might clear pos?
        # Assuming timing scales position, but if timing < threshold we might clear?
        # LightTimingModule usually just reduces exposure.
        # Let's assume we hold the ETFs but with reduced weight.
        # So the "Set" of holdings is still valid.
        curr_set = set(curr["holdings"])
        
        # Execution Date is Next Day (Strict T+1)
        # Find next trading day index
        try:
            curr_idx = dates.get_loc(curr_date)
            exec_idx = curr_idx + 1
            if exec_idx >= len(dates): break
            exec_date = dates[exec_idx]
        except:
            continue
            
        # Sells
        to_sell = prev_set - curr_set
        for code in to_sell:
            # Find when we bought this? 
            # We need to track open positions.
            pass
            
        prev_set = curr_set

    # Let's use a proper Position Tracker
    open_positions = {} # code -> {entry_date, entry_price}
    closed_trades = []
    
    for i in range(len(signal_log)):
        sig = signal_log[i]
        sig_date = sig["date"]
        
        # Execution Date (T+1)
        try:
            sig_idx = dates.get_loc(sig_date)
            exec_idx = sig_idx + 1
            if exec_idx >= len(dates): break
            exec_date = dates[exec_idx]
        except:
            continue
            
        target_codes = set(sig["holdings"])
        
        # Check Sells
        current_codes = list(open_positions.keys())
        for code in current_codes:
            if code not in target_codes:
                # Sell
                entry = open_positions.pop(code)
                exit_price = get_price(exec_date, code)
                if exit_price:
                    ret = (exit_price - entry["entry_price"]) / entry["entry_price"] - COMMISSION_RATE * 2
                    closed_trades.append({
                        "code": code,
                        "entry_date": entry["entry_date"],
                        "exit_date": exec_date,
                        "entry_price": entry["entry_price"],
                        "exit_price": exit_price,
                        "return": ret
                    })
        
        # Check Buys
        for code in target_codes:
            if code not in open_positions:
                # Buy
                entry_price = get_price(exec_date, code)
                if entry_price:
                    open_positions[code] = {
                        "entry_date": exec_date,
                        "entry_price": entry_price
                    }
                    
    # Close remaining at end
    last_date = dates[-1]
    for code, entry in open_positions.items():
        exit_price = get_price(last_date, code)
        if exit_price:
            ret = (exit_price - entry["entry_price"]) / entry["entry_price"]
            closed_trades.append({
                "code": code,
                "entry_date": entry["entry_date"],
                "exit_date": last_date,
                "entry_price": entry["entry_price"],
                "exit_price": exit_price,
                "return": ret,
                "status": "Open"
            })

    # 6. Analysis Output
    df_trades = pd.DataFrame(closed_trades)
    if len(df_trades) == 0:
        print("No trades generated.")
        return

    print(f"\n📊 交易统计 (Trade Statistics):")
    print(f"  总交易次数: {len(df_trades)}")
    print(f"  胜率 (Win Rate): {(df_trades['return'] > 0).mean():.2%}")
    print(f"  平均收益 (Avg Return): {df_trades['return'].mean():.2%}")
    print(f"  盈亏比 (Profit Factor): {df_trades[df_trades['return']>0]['return'].sum() / -df_trades[df_trades['return']<0]['return'].sum():.2f}")
    print(f"  最大单笔盈利: {df_trades['return'].max():.2%}")
    print(f"  最大单笔亏损: {df_trades['return'].min():.2%}")
    
    print(f"\n📝 最近 10 笔交易记录:")
    print(df_trades.tail(10)[['code', 'entry_date', 'exit_date', 'return']].to_string(index=False))
    
    print(f"\n🏆 最佳 5 笔交易:")
    print(df_trades.nlargest(5, 'return')[['code', 'entry_date', 'exit_date', 'return']].to_string(index=False))
    
    print(f"\n💀 最差 5 笔交易:")
    print(df_trades.nsmallest(5, 'return')[['code', 'entry_date', 'exit_date', 'return']].to_string(index=False))

    # Save to CSV
    df_trades.to_csv("results/top_strategy_trades.csv", index=False)
    print(f"\n💾 完整交易记录已保存至: results/top_strategy_trades.csv")

if __name__ == "__main__":
    analyze_trades()
