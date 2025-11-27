import pandas as pd
import sys

# Path to the results file
file_path = "/home/sensen/dev/projects/-0927/etf_rotation_experiments/results_combo_wfo/20251116_151853_20251124_184248/top12597_profit_backtest_slip5bps_20251116_151853_20251124_184248.csv"

df = pd.read_csv(file_path)
print(f"Loaded {len(df)} strategies.")

# Define criteria for "Balanced"
# 1. Max Drawdown > -20% (Strict risk control)
# 2. Annual Return > 15% (Decent profit)
# 3. Sharpe Ratio > 0.9 (Good risk-adjusted return)

balanced_df = df[
    (df['max_dd_net'] > -0.20) & 
    (df['annual_ret_net'] > 0.15)
].copy()

print(f"Strategies meeting criteria (DD > -20%, AnnRet > 15%): {len(balanced_df)}")

if len(balanced_df) > 0:
    # Sort by Sharpe Ratio
    balanced_df = balanced_df.sort_values('sharpe_net', ascending=False)
    
    cols = ['rank', 'combo', 'annual_ret_net', 'max_dd_net', 'sharpe_net', 'calmar_ratio', 'total_ret_net']
    print("\nTop 10 Balanced Strategies (Sorted by Sharpe):")
    print(balanced_df[cols].head(10).to_string(index=False))
    
    # Find the "Golden" one: Best combination of Return/Risk
    # Let's define a custom score: Sharpe * (1 + AnnualRet) / abs(MaxDD)
    balanced_df['custom_score'] = balanced_df['sharpe_net'] * (1 + balanced_df['annual_ret_net']) / abs(balanced_df['max_dd_net'])
    best_custom = balanced_df.sort_values('custom_score', ascending=False).head(5)
    
    print("\n🏆 Top 5 'Golden' Strategies (Custom Score):")
    print(best_custom[cols].to_string(index=False))

else:
    print("No strategies met the strict criteria. Relaxing constraints...")
    relaxed_df = df[df['max_dd_net'] > -0.25].sort_values('sharpe_net', ascending=False)
    cols = ['rank', 'combo', 'annual_ret_net', 'max_dd_net', 'sharpe_net', 'calmar_ratio']
    print("\nTop 5 Strategies with DD > -25%:")
    print(relaxed_df[cols].head(5).to_string(index=False))
