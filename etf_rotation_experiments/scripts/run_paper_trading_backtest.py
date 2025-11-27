#!/usr/bin/env python3
"""
Paper Trading Backtest Runner
Runs the Platinum Strategy over the last 6 years using the SimpleTrader framework.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import DataLoader
from core.precise_factor_library_v2 import PreciseFactorLibrary
from core.cross_section_processor import CrossSectionProcessor
from core.simple_trader import SimpleTrader
from strategies.backtest.production_backtest import compute_spearman_ic_numba

# Configuration
TARGET_COMBO = [
    "OBV_SLOPE_10D",
    "PRICE_POSITION_20D",
    "RSI_14",
    "SLOPE_20D",
    "VORTEX_14D"
]
LOOKBACK_WINDOW = 252
TOP_N = 5
REBALANCE_FREQ = 1
START_DATE = "2020-01-01" # Approx 6 years
INITIAL_CAPITAL = 100000.0
BACKTEST_DIR = PROJECT_ROOT / "_backtest_data"

def main():
    print(f"🚀 Starting Paper Trading Backtest from {START_DATE}...")
    
    # 1. Setup Environment
    if BACKTEST_DIR.exists():
        shutil.rmtree(BACKTEST_DIR)
    BACKTEST_DIR.mkdir()
    
    trader = SimpleTrader(data_dir=BACKTEST_DIR, initial_capital=INITIAL_CAPITAL)
    
    # 2. Load Data
    print("Loading Data...")
    REPO_ROOT = PROJECT_ROOT.parent
    loader = DataLoader(
        data_dir=REPO_ROOT / "raw" / "ETF" / "daily",
        cache_dir=REPO_ROOT / "raw" / "cache"
    )
    # Load all ETFs
    etf_codes = [f.stem for f in (REPO_ROOT / "raw" / "ETF" / "daily").glob("*.csv")]
    ohlcv = loader.load_ohlcv(etf_codes=etf_codes)
    
    # Align dates
    dates = ohlcv['close'].index
    start_idx = dates.searchsorted(pd.Timestamp(START_DATE))
    if start_idx < LOOKBACK_WINDOW:
        print(f"⚠️ Warning: Start date {START_DATE} is too early for lookback window. Adjusting...")
        start_idx = LOOKBACK_WINDOW + 1
        
    print(f"Backtest Range: {dates[start_idx]} to {dates[-1]} ({len(dates) - start_idx} days)")

    # 3. Compute Factors (Pre-compute all)
    print("Computing Factors...")
    lib = PreciseFactorLibrary()
    factors_df = lib.compute_all_factors(ohlcv)
    
    # 4. Standardize (Pre-compute all)
    print("Standardizing...")
    processor = CrossSectionProcessor(lower_percentile=2.5, upper_percentile=97.5)
    factors_dict = {name: factors_df[name] for name in lib.list_factors()}
    standardized = processor.process_all_factors(factors_dict)
    
    # Prepare Data Cubes
    T, N = ohlcv['close'].shape
    F = len(TARGET_COMBO)
    factors_data = np.zeros((T, N, F))
    
    for i, fname in enumerate(TARGET_COMBO):
        if fname in standardized:
            factors_data[:, :, i] = standardized[fname].values
        else:
            raise ValueError(f"Factor {fname} not found")
            
    returns = ohlcv['close'].pct_change().values
    close_prices = ohlcv['close'].values
    tickers = ohlcv['close'].columns.tolist()
    
    # 5. Event Loop
    print(f"Running Event Loop (Rebalance Freq: {REBALANCE_FREQ} days)...")
    
    # Initialize target weights
    target_weights = {}
    
    for t in range(start_idx, T):
        current_date = dates[t]
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Check if rebalance day
        # We align rebalance days from start_idx
        days_since_start = t - start_idx
        is_rebalance_day = (days_since_start % REBALANCE_FREQ == 0)
        
        if is_rebalance_day:
            # --- Signal Generation Logic ---
            # Use factors from t-1 (Yesterday) to trade at t (Today)
            # This matches production_backtest.py logic (Signal T-1 -> Trade T)
            
            # Training Data:
            # We need to calculate weights based on history available at t-1.
            # History: Factors up to t-2, Returns up to t-1.
            # Wait, if we are at t, and we use factors[t-1], we can use weights calculated at t-1.
            # Weights at t-1 use history [t-1-LOOKBACK, t-1].
            
            # Let's align strictly with production_backtest:
            # factors_yesterday = factors_data[day_idx - 1]
            # factor_weights = ic_weights_matrix[rebalance_counter]
            
            # ic_weights_matrix is pre-calculated.
            # In production_backtest, it uses history [day_idx - lookback, day_idx].
            # Wait, rebalance_indices[i] is day_idx.
            # hist_start = max(0, day_idx - lookback_window)
            # hist_end = day_idx
            # factors_hist = factors_data[hist_start:hist_end] (0 to -1 relative to day_idx)
            # returns_hist = returns[hist_start:hist_end]
            # So it uses factors[day_idx-LOOKBACK : day_idx] and returns[day_idx-LOOKBACK : day_idx].
            # returns[k] is return at k (Close[k]/Close[k-1]-1).
            # factors[k] is factor at k.
            # So it correlates Factor[k] with Return[k].
            # Wait, compute_spearman_ic_numba(factors, returns).
            # If factors[k] predicts returns[k], that is "Concurrent IC".
            # Usually we want "Forward IC": Factor[k] predicts Return[k+1].
            # If production_backtest uses Concurrent IC, then it assumes Factor[k] is available before Return[k] is realized?
            # No, Factor[k] uses Close[k]. Return[k] uses Close[k].
            # So it's finding factors that explain *current* return.
            # Then it uses `factors_yesterday` (Factor[t-1]) to predict... what?
            # It calculates `signal_yesterday` using `factors_yesterday` and `weights`.
            # Then it ranks `signal_yesterday`.
            # And buys based on that.
            # So it buys at T based on Signal at T-1.
            # And Signal at T-1 is based on "Concurrent Correlation" of past 252 days.
            # i.e. "Factors that correlated with returns *on the same day* in the past".
            # And we apply that relationship to Yesterday's factor.
            # Does Yesterday's factor predict Today's return?
            # Only if Factor[t-1] ~ Return[t-1] implies Factor[t-1] ~ Return[t].
            # This seems like a Momentum strategy on the Factor itself?
            
            # Regardless of theory, I must replicate the code.
            # Training Data:
            # production_backtest uses factors[t-LOOKBACK : t] and returns[t-LOOKBACK : t]
            # This is Concurrent IC (Factor[k] vs Return[k])
            
            train_factors = factors_data[t-LOOKBACK_WINDOW : t]
            train_returns = returns[t-LOOKBACK_WINDOW : t]
            
            if len(train_factors) < LOOKBACK_WINDOW:
                continue

            # Calculate Weights
            ics = np.zeros(F)
            for f in range(F):
                ics[f] = compute_spearman_ic_numba(train_factors[:, :, f], train_returns)
                
            abs_ics = np.abs(ics)
            if abs_ics.sum() > 0:
                weights = abs_ics / abs_ics.sum()
            else:
                weights = np.ones(F) / F
                
            # Calculate Scores using Today's Factors (t)
            # We trade at Close[t], so we capture Return[t+1].
            # This matches production_backtest (Factor[t-1] -> Return[t])
            
            current_factors = factors_data[t] # (N, F)
            final_scores = np.zeros(N)
            valid_mask = np.ones(N, dtype=bool)
            
            for n in range(N):
                score = 0.0
                w_sum = 0.0
                for f in range(F):
                    val = current_factors[n, f]
                    if not np.isnan(val):
                        score += val * weights[f]
                        w_sum += weights[f]
                
                if w_sum > 0:
                    final_scores[n] = score / w_sum
                else:
                    final_scores[n] = -999
                    valid_mask[n] = False
                    
            # Rank
            day_scores = pd.DataFrame({'score': final_scores, 'idx': range(N)})
            day_scores = day_scores[valid_mask].sort_values('score', ascending=False).head(TOP_N)
            
            target_weights = {}
            for _, row in day_scores.iterrows():
                ticker_idx = int(row['idx'])
                ticker = tickers[ticker_idx]
                target_weights[ticker] = 1.0 / TOP_N
            
        # --- Execution ---
        # Get current prices (Today t)
        current_prices = {tickers[i]: close_prices[t, i] for i in range(N) if not np.isnan(close_prices[t, i])}
        
        orders = []
        if is_rebalance_day:
            orders = trader.generate_rebalance_orders(target_weights, current_prices)
        
        # Execute
        for order in orders:
            # Custom execute logic to support backtest date
            action = order["action"]
            ticker = order["ticker"]
            qty = order["quantity"]
            price = order["price"]
            amount = qty * price
            
            # Fee logic (Match production_backtest: 0.5bps, no tax)
            fee_rate = 0.0 # 0.0 bps to test Gross Return match
            min_fee = 0.0      # No min fee
            commission = amount * fee_rate
            stamp_duty = 0.0   # No tax
            total_fee = commission + stamp_duty
            
            success = False
            
        # --- Execution ---
        # Get current prices
        current_prices = {tickers[i]: close_prices[t, i] for i in range(N) if not np.isnan(close_prices[t, i])}
        
        # Generate Orders (Only if rebalance day OR if we want to enforce weights daily?)
        # If we only generate orders on rebalance day, we hold for 8 days.
        # But SimpleTrader.generate_rebalance_orders will try to match target_weights.
        # If prices move, weights drift.
        # If we call it every day with SAME target_weights, it will rebalance every day to restore 20% weights.
        # This is "Daily Rebalancing to Fixed Weights".
        # Usually "Rebalance Freq 8" means we only touch the portfolio every 8 days.
        # So we should only call generate_rebalance_orders on rebalance day.
        
        orders = []
        if is_rebalance_day:
            orders = trader.generate_rebalance_orders(target_weights, current_prices)
        
        # Execute
        for order in orders:
            # Override date in trader for backtest accuracy
            # SimpleTrader uses datetime.now(), we need to patch or just accept the log has wrong timestamp?
            # Better to patch execute_order or just let it log and we fix the CSV later?
            # Actually SimpleTrader writes 'date' column. We can overwrite it in the CSV or modify SimpleTrader.
            # For this script, let's just execute. The 'timestamp' will be now, but we can add a 'backtest_date' column or just rely on the sequence.
            # Wait, SimpleTrader.log_daily_nav uses datetime.now().strftime("%Y-%m-%d").
            # This will result in all rows having TODAY's date. This is bad for analysis.
            
            # Quick Hack: Modify SimpleTrader instance's internal clock or pass date?
            # SimpleTrader doesn't accept date.
            # Let's subclass or monkeypatch for the backtest.
            pass
            
        # Monkeypatching execution for correct dates
        # We will manually write to ledger/nav to ensure correct dates, 
        # OR we modify SimpleTrader to accept date.
        # Modifying SimpleTrader is cleaner (Linus Principle 2: API stability? Maybe add optional date arg)
        
        # Let's do the monkeypatch here for the script to avoid changing core for a test script if possible,
        # BUT Linus says "No special cases". So updating SimpleTrader to accept date is better.
        # However, I cannot edit SimpleTrader in the middle of this script generation.
        # I will use a subclass here.
        
        # Execute Orders
        for order in orders:
            # We need to inject the date into the order or the execution
            # Let's just call execute and then fix the ledger last line? No, race condition if real.
            # Here it's serial.
            
            # Custom execute logic to support backtest date
            action = order["action"]
            ticker = order["ticker"]
            qty = order["quantity"]
            price = order["price"]
            amount = qty * price
            
            # Fee logic
            fee_rate = 0.0003
            min_fee = 5.0
            commission = max(amount * fee_rate, min_fee)
            stamp_duty = amount * 0.0005 if action == "SELL" else 0.0
            total_fee = commission + stamp_duty
            
            success = False
            if action == "BUY":
                cost = amount + total_fee
                if trader.account["cash"] >= cost:
                    trader.account["cash"] -= cost
                    trader.account["holdings"][ticker] = trader.account["holdings"].get(ticker, 0) + qty
                    success = True
            elif action == "SELL":
                if trader.account["holdings"].get(ticker, 0) >= qty:
                    proceeds = amount - total_fee
                    trader.account["cash"] += proceeds
                    trader.account["holdings"][ticker] -= qty
                    if trader.account["holdings"][ticker] == 0:
                        del trader.account["holdings"][ticker]
                    success = True
            
            if success:
                # Log with correct date
                record = {
                    "timestamp": f"{date_str} 15:00:00",
                    "date": date_str,
                    "ticker": ticker,
                    "action": action,
                    "price": price,
                    "quantity": qty,
                    "amount": amount,
                    "fee": total_fee,
                    "cash_balance": trader.account["cash"]
                }
                pd.DataFrame([record]).to_csv(trader.ledger_file, mode='a', header=False, index=False)

        # Log NAV with correct date
        nav = trader.calculate_nav(current_prices)
        holdings_val = nav - trader.account["cash"]
        ret_pct = (nav - trader.initial_capital) / trader.initial_capital * 100
        
        nav_record = {
            "date": date_str,
            "total_value": nav,
            "cash": trader.account["cash"],
            "holdings_value": holdings_val,
            "return_pct": ret_pct
        }
        pd.DataFrame([nav_record]).to_csv(trader.nav_file, mode='a', header=False, index=False)
        
        if t % 100 == 0:
            print(f"Processed {date_str}: NAV {nav:.2f}")

    print("Backtest Complete.")
    print(f"Final NAV: {nav:.2f}")
    print(f"Results saved to {BACKTEST_DIR}")

if __name__ == "__main__":
    main()
