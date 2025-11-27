import pandas as pd
import numpy as np
import sys
import os

def find_snowball_strategies():
    print("="*80)
    print("❄️  寻找「高胜率滚雪球」策略 (High Win-Rate Snowball Strategies)")
    print("🎯 目标: 胜率 52%-60% | 盈亏比 > 1.3 | Sharpe > 1.0")
    print("="*80)

    # 1. Load Results
    result_file = 'results_combo_wfo/20251126_190236_20251127_125624/top12597_backtest_by_ic_20251126_190236_20251127_125624_full.csv'
    if not os.path.exists(result_file):
        print(f"❌ 结果文件不存在: {result_file}")
        return

    df = pd.read_csv(result_file)
    total = len(df)
    print(f"📚 总策略池: {total} 个")

    # 2. Filter
    # Criteria 1: Win Rate 52% - 60%
    # Note: win_rate in CSV might be 0.52 or 52.0? Usually 0.xx based on previous outputs.
    # Let's check data range.
    if df['win_rate'].mean() > 1.0:
        # It's percentage (e.g. 52.0)
        wr_min, wr_max = 52.0, 60.0
    else:
        # It's ratio (e.g. 0.52)
        wr_min, wr_max = 0.52, 0.60

    # Criteria 2: Profit Factor > 1.3 (Healthy)
    pf_min = 1.3

    # Criteria 3: Sharpe > 0.8 (Baseline quality)
    sharpe_min = 0.8

    candidates = df[
        (df['win_rate'] >= wr_min) & 
        (df['win_rate'] <= wr_max) & 
        (df['profit_factor'] >= pf_min) &
        (df['sharpe'] >= sharpe_min)
    ].copy()

    print(f"\n🔍 筛选标准:")
    print(f"  1. 胜率 (Win Rate): {wr_min:.1%} - {wr_max:.1%}")
    print(f"  2. 盈亏比 (Profit Factor): >= {pf_min}")
    print(f"  3. 夏普比率 (Sharpe): >= {sharpe_min}")
    
    print(f"\n✅ 符合条件的策略: {len(candidates)} 个 (占比 {len(candidates)/total:.1%})")

    if len(candidates) == 0:
        print("⚠️  没有找到完全符合条件的策略。尝试放宽条件...")
        return

    # 3. Rank
    # User wants to "Snowball" -> Consistent compounding.
    # Sort by Sharpe (Risk-adjusted return) is best for consistency.
    # Or Calmar (Return / MaxDD) for safety.
    
    candidates['score'] = candidates['sharpe'] * 0.7 + candidates['calmar_ratio'] * 0.3
    top_candidates = candidates.sort_values('score', ascending=False).head(20)

    # 4. Display
    print(f"\n🏆 Top 10 「滚雪球」候选策略:")
    print("-" * 120)
    headers = ["Rank", "WinRate", "P.Factor", "Sharpe", "Ann.Ret", "MaxDD", "Combo"]
    print(f"{headers[0]:<5} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<6} | {headers[4]:<7} | {headers[5]:<7} | {headers[6]}")
    print("-" * 120)

    for i, row in top_candidates.head(10).iterrows():
        wr = row['win_rate']
        if wr < 1.0: wr *= 100
        
        print(f"{row['rank']:<5} | {wr:6.1f}% | {row['profit_factor']:6.2f}   | {row['sharpe']:5.2f}  | {row['annual_ret']*100:5.1f}% | {row['max_dd']*100:5.1f}% | {row['combo']}")

    # 5. Deep Dive into Top 1
    best = top_candidates.iloc[0]
    print(f"\n🌟 最佳推荐: {best['combo']}")
    print(f"   该策略在保持 {best['win_rate']*100:.1f}% 高胜率的同时，拥有 {best['profit_factor']:.2f} 的盈亏比。")
    print(f"   年化收益 {best['annual_ret']*100:.1f}%，最大回撤仅 {best['max_dd']*100:.1f}%。")
    print(f"   非常适合 100万 资金追求稳健复利。")

    # Save
    top_candidates.to_csv("results/snowball_strategies.csv", index=False)
    print(f"\n💾 候选列表已保存至: results/snowball_strategies.csv")

if __name__ == "__main__":
    find_snowball_strategies()
