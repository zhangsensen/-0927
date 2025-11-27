import pandas as pd
import sys

file_path = "/home/sensen/dev/projects/-0927/etf_rotation_experiments/results_combo_wfo/20251116_151853_20251124_183126/top100_profit_backtest_slip5bps_20251116_151853_20251124_183126.csv"

df = pd.read_csv(file_path)

print(f"Loaded {len(df)} strategies.")

# Filter for balanced strategies
# Criteria 1: Max DD > -0.20 (less than 20% drawdown)
balanced_df = df[df['max_dd_net'] > -0.20].copy()

print(f"Strategies with Max DD < 20%: {len(balanced_df)}")

if len(balanced_df) > 0:
    # Sort by Sharpe Ratio
    balanced_df = balanced_df.sort_values('sharpe_net', ascending=False)
    
    print("\nTop 5 Balanced Strategies (Sorted by Sharpe):")
    cols = ['rank', 'combo', 'annual_ret_net', 'max_dd_net', 'sharpe_net', 'calmar_ratio']
    print(balanced_df[cols].head(5).to_string(index=False))
    
    # Also check for very low drawdown (< 15%)
    very_safe = df[df['max_dd_net'] > -0.15].copy()
    print(f"\nStrategies with Max DD < 15%: {len(very_safe)}")
    if len(very_safe) > 0:
        very_safe = very_safe.sort_values('sharpe_net', ascending=False)
        print(very_safe[cols].head(5).to_string(index=False))

# Also show the absolute top performer in this batch
print("\nTop Performer in this batch (by Sharpe):")
print(df.sort_values('sharpe_net', ascending=False)[cols].head(1).to_string(index=False))
