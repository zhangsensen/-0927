#!/usr/bin/env python3
"""
止损后悔分析：如果不止损会怎样？
分析每次止损后，价格在接下来 N 天的表现
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
import numpy as np

from etf_strategy.core.data_loader import DataLoader

def main():
    print("=" * 80)
    print("止损后悔分析：如果不止损会怎样？")
    print("=" * 80)

    # 加载配置
    config_path = ROOT / "configs/combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 加载数据
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    
    etf_codes = config["data"]["symbols"]
    close_prices = ohlcv["close"][etf_codes].ffill().bfill()
    
    # 从诊断结果读取止损事件（简化：手动输入几个典型案例）
    # 格式: (day, etf_idx, entry_price, exec_price)
    stop_cases = [
        (262, 9, 1.425, 1.292),
        (275, 27, 2.430, 2.385),
        (275, 34, 2.000, 1.800),
        (284, 33, 1.498, 1.404),
        (350, 36, 1.158, 1.147),
        (373, 31, 2.462, 2.232),
        (379, 37, 1.262, 1.205),
        (417, 33, 2.155, 2.164),
        (421, 32, 1.458, 1.394),
        (488, 36, 1.537, 1.467),
    ]
    
    print("\n分析止损后 1, 2, 4, 8 周（调仓周期）的价格表现:")
    print(f"{'案例':>4s} {'买入价':>8s} {'止损价':>8s} {'止损亏损':>10s} | {'1周后':>8s} {'2周后':>8s} {'4周后':>8s} {'8周后':>8s}")
    print("-" * 100)
    
    regret_1w = []
    regret_2w = []
    regret_4w = []
    regret_8w = []
    
    for i, (day, etf_idx, entry, exec_price) in enumerate(stop_cases, 1):
        stop_loss = (exec_price - entry) / entry * 100
        
        # 获取该 ETF 的价格序列
        etf_code = etf_codes[etf_idx]
        prices = close_prices[etf_code].values
        
        # 如果不止损，持有到未来的收益
        future_days = [8, 16, 32, 64]  # 1, 2, 4, 8 周
        future_returns = []
        
        for days in future_days:
            future_idx = day + days
            if future_idx < len(prices):
                future_price = prices[future_idx]
                future_return = (future_price - entry) / entry * 100
                future_returns.append(future_return)
            else:
                future_returns.append(np.nan)
        
        print(f"{i:4d} {entry:8.3f} {exec_price:8.3f} {stop_loss:9.2f}% |", end="")
        for ret in future_returns:
            if np.isnan(ret):
                print(f"{'N/A':>8s}", end="")
            else:
                color = "+" if ret > stop_loss else ""
                print(f"{color}{ret:7.2f}%", end="")
                
                # 记录后悔值
                regret = ret - stop_loss
                if days == 8:
                    regret_1w.append(regret)
                elif days == 16:
                    regret_2w.append(regret)
                elif days == 32:
                    regret_4w.append(regret)
                elif days == 64:
                    regret_8w.append(regret)
        print()
    
    # 统计
    print("\n" + "=" * 80)
    print("后悔值统计（如果不止损会多赚/少亏多少）:")
    print("=" * 80)
    print(f"{'周期':>10s} {'平均后悔值':>15s} {'后悔>0':>15s} {'后悔<0':>15s}")
    print("-" * 80)
    
    for period, regrets in [("1周", regret_1w), ("2周", regret_2w), ("4周", regret_4w), ("8周", regret_8w)]:
        valid_regrets = [r for r in regrets if not np.isnan(r)]
        if valid_regrets:
            avg_regret = np.mean(valid_regrets)
            positive_pct = sum(1 for r in valid_regrets if r > 0) / len(valid_regrets) * 100
            negative_pct = sum(1 for r in valid_regrets if r < 0) / len(valid_regrets) * 100
            print(f"{period:>10s} {avg_regret:14.2f}% {positive_pct:14.1f}% {negative_pct:14.1f}%")
    
    print("\n结论:")
    print(f"  如果平均后悔值 > 0，说明止损是错误的（不止损会更好）")
    print(f"  如果后悔>0 的比例高，说明大部分止损是提前的")


if __name__ == "__main__":
    main()
