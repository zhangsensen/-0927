import pandas as pd
import numpy as np
from pathlib import Path

BACKTEST_DIR = Path("/home/sensen/dev/projects/-0927/etf_rotation_experiments/_backtest_data")
nav_file = BACKTEST_DIR / "nav.csv"
ledger_file = BACKTEST_DIR / "ledger.csv"

print(f"Analyzing Backtest Results from {BACKTEST_DIR}...")

# 1. NAV Analysis
if nav_file.exists():
    nav_df = pd.read_csv(nav_file)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    
    # Calculate Metrics
    total_ret = (nav_df['total_value'].iloc[-1] / nav_df['total_value'].iloc[0]) - 1
    days = (nav_df.index[-1] - nav_df.index[0]).days
    years = days / 365.25
    annual_ret = (1 + total_ret) ** (1 / years) - 1
    
    # Daily Returns for Sharpe
    nav_df['daily_ret'] = nav_df['total_value'].pct_change()
    sharpe = nav_df['daily_ret'].mean() / nav_df['daily_ret'].std() * np.sqrt(252)
    
    # Max Drawdown
    nav_df['cummax'] = nav_df['total_value'].cummax()
    nav_df['drawdown'] = (nav_df['total_value'] - nav_df['cummax']) / nav_df['cummax']
    max_dd = nav_df['drawdown'].min()
    
    print("-" * 40)
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Annual Return: {annual_ret*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print("-" * 40)

# 2. Ledger Analysis
if ledger_file.exists():
    ledger_df = pd.read_csv(ledger_file)
    print(f"Total Trades: {len(ledger_df)}")
    print(f"Total Fees Paid: {ledger_df['fee'].sum():.2f}")
    
    # Turnover
    # Estimate turnover?
    # Just count trades.
    
    # Win Rate (Trade level)
    # Hard to calculate trade-level win rate from ledger without matching buy/sell.
    # But we can check if we sold for profit?
    # We need to track cost basis.
    pass
