#!/usr/bin/env python3
"""
验证 production_backtest.py 的时间逻辑
打印关键时间点，确认是否存在前视偏差
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_loader import DataLoader

# Load minimal data
REPO_ROOT = PROJECT_ROOT.parent
loader = DataLoader(
    data_dir=REPO_ROOT / "raw" / "ETF" / "daily",
    cache_dir=REPO_ROOT / "raw" / "cache"
)

etf_codes = ['510300', '159928']  # Just 2 ETFs for clarity
ohlcv = loader.load_ohlcv(etf_codes=etf_codes)

close = ohlcv['close'].values
returns = ohlcv['close'].pct_change().values
dates = ohlcv['close'].index

T = len(dates)
start_idx = 260  # After warmup

print("=" * 80)
print("production_backtest.py 时间逻辑验证")
print("=" * 80)

# Simulate key dates
for offset in range(5):  # First 5 days
    day_idx = start_idx + offset
    date_t = dates[day_idx]
    date_t_minus_1 = dates[day_idx - 1]
    
    print(f"\n{'='*80}")
    print(f"Offset: {offset}, day_idx: {day_idx}")
    print(f"Date T (Today):     {date_t.strftime('%Y-%m-%d')}")
    print(f"Date T-1 (Yest):    {date_t_minus_1.strftime('%Y-%m-%d')}")
    print(f"{'-'*80}")
    
    # Simulate production_backtest logic
    print("Step 1: 在 T 日调仓 (is_rebalance_day)")
    print(f"  - 使用 factors_yesterday = factors_data[{day_idx - 1}]  # T-1 日因子")
    print(f"  - 计算 signal_yesterday")
    print(f"  - 决定 target_weights (持仓配置)")
    
    print("\nStep 2: 计算 T 日收益")
    print(f"  - close_to_close_ret = returns[{day_idx}]  # (Close[T] / Close[T-1]) - 1")
    print(f"  - daily_ret = sum(current_weights * close_to_close_ret)")
    
    print("\nStep 3: 更新净值")
    print(f"  - portfolio_values[{offset + 1}] = portfolio_values[{offset}] * (1 + daily_ret)")
    
    print(f"\n关键问题: current_weights 在何时被更新?")
    if offset == 0:
        print(f"  - 在 Step 1 末尾: current_weights = target_weights")
        print(f"  - 因此，Step 2 使用的 current_weights 是刚刚在 T 日调整后的权重")
    
    print(f"\n时间线分析:")
    print(f"  T-1 日 收盘: Close[{day_idx - 1}] = {close[day_idx - 1, 0]:.3f}")
    print(f"  T 日 收盘:   Close[{day_idx}] = {close[day_idx, 0]:.3f}")
    print(f"  T 日 收益率: (Close[{day_idx}] / Close[{day_idx - 1}]) - 1 = {returns[day_idx, 0]:.4f}")
    
    print(f"\n逻辑推演:")
    print(f"  1. 在 T-1 日收盘后，我们计算因子 factors[T-1]")
    print(f"  2. 在 T 日（循环中 day_idx={day_idx}），我们:")
    print(f"     - 读取 factors[T-1]（昨天的因子）")
    print(f"     - 决定新持仓 target_weights")
    print(f"     - 更新 current_weights = target_weights")
    print(f"     - 立即用 current_weights 计算今日收益 returns[T]")
    print(f"  3. 问题: 我们能在 T 日用 T-1 的因子捕获 T 日的收益吗?")
    
    print(f"\n实盘逻辑:")
    print(f"  - 如果我们在 T-1 日收盘时决定持仓，理论上可以在 T 日开盘买入")
    print(f"  - 但代码中，调仓发生在 'T 日的循环迭代中'")
    print(f"  - 并且立即结算 'T 日的收益 returns[T]'")
    print(f"  - returns[T] = Close[T] / Close[T-1] - 1")
    print(f"  - 问题: 我们需要在 T-1 收盘时就持有头寸，才能吃到 T 日的收益")
    print(f"  - 但代码逻辑是: 在 T 日才决定持仓")

print("\n" + "=" * 80)
print("结论:")
print("=" * 80)
print("production_backtest.py 的逻辑等价于:")
print("  - 在 T 日，根据 T-1 的因子，瞬间完成调仓")
print("  - 并立即捕获 T 日的收益 (Close[T] / Close[T-1])")
print("")
print("这在数学上等价于:")
print("  - 在 T-1 日收盘时刻，已经持有了 T 日的目标头寸")
print("  - 即: 用 Factor[T-1] 预测 Return[T]，这是 Lag-1 IC")
print("")
print("实盘中:")
print("  - 如果在 T 日才能获取信号（基于 Factor[T]）")
print("  - 最快也只能在 T 日收盘或 T+1 开盘执行")
print("  - 捕获的是 Return[T+1]（或 Open[T+1] 到 Close[T+1]）")
print("  - 即: 用 Factor[T] 预测 Return[T+1]，这是 Lag-2 IC")
print("")
print("差异:")
print("  Lag-1 IC (理论) vs Lag-2 IC (实盘)")
print("  对于短周期因子(RSI, SLOPE, VORTEX)，这个差异是致命的")
print("=" * 80)
