import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compare VEC and BT metrics")
    parser.add_argument("--input", type=str, required=True, help="Path to BT results CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    print("=" * 80)
    print("📊 VEC vs BT Metrics Comparison (Top 100)")
    print("=" * 80)
    
    # 1. Return Comparison
    df['return_diff'] = df['bt_return'] - df['vec_return']
    print(f"\n1. Return (Total)")
    print(f"   Mean VEC: {df['vec_return'].mean()*100:.2f}%")
    print(f"   Mean BT:  {df['bt_return'].mean()*100:.2f}%")
    print(f"   Mean Diff: {df['return_diff'].mean()*100:.4f} pp")
    print(f"   Max Diff:  {df['return_diff'].abs().max()*100:.4f} pp")
    print(f"   Correlation: {df['vec_return'].corr(df['bt_return']):.6f}")

    # 2. Win Rate Comparison
    # VEC win rate might be 0-1 or 0-100, let's check. Usually 0-1.
    # BT win rate is 0-1.
    print(f"\n2. Win Rate")
    print(f"   Mean VEC: {df['vec_win_rate'].mean()*100:.2f}%")
    print(f"   Mean BT:  {df['bt_win_rate'].mean()*100:.2f}%")
    df['wr_diff'] = df['bt_win_rate'] - df['vec_win_rate']
    print(f"   Mean Diff: {df['wr_diff'].mean()*100:.2f} pp")
    print(f"   Max Diff:  {df['wr_diff'].abs().max()*100:.2f} pp")
    print(f"   Correlation: {df['vec_win_rate'].corr(df['bt_win_rate']):.6f}")

    # 3. Trade Counts Comparison
    print(f"\n3. Trade Counts")
    print(f"   Mean VEC: {df['vec_trades'].mean():.1f}")
    print(f"   Mean BT:  {df['bt_trades'].mean():.1f}")
    df['trades_diff'] = df['bt_trades'] - df['vec_trades']
    print(f"   Mean Diff: {df['trades_diff'].mean():.1f}")
    print(f"   Max Diff:  {df['trades_diff'].abs().max():.1f}")
    print(f"   Correlation: {df['vec_trades'].corr(df['bt_trades']):.6f}")

    # 4. Detailed Top 10 Comparison Table
    print(f"\n{'='*100}")
    print("🏆 Top 10 Strategies Comparison")
    print(f"{'Rank':<4} | {'Return (V/B)':<15} | {'WinRate (V/B)':<15} | {'Trades (V/B)':<12} | {'Diff (Ret)':<10}")
    print("-" * 100)
    
    # Sort by score if available, else by bt_return
    if 'score_balanced' in df.columns:
        df_top = df.nlargest(10, 'score_balanced')
    else:
        df_top = df.nlargest(10, 'bt_return')

    for i, row in enumerate(df_top.itertuples(), 1):
        ret_str = f"{row.vec_return*100:.1f}% / {row.bt_return*100:.1f}%"
        wr_str = f"{row.vec_win_rate*100:.1f}% / {row.bt_win_rate*100:.1f}%"
        trades_str = f"{row.vec_trades:.0f} / {row.bt_trades:.0f}"
        diff_str = f"{(row.bt_return - row.vec_return)*100:+.2f}pp"
        print(f"{i:<4} | {ret_str:<15} | {wr_str:<15} | {trades_str:<12} | {diff_str:<10}")

if __name__ == "__main__":
    main()
