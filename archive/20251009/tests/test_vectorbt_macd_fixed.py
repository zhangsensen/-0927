#!/usr/bin/env python3
"""
测试修复后的VectorBT MACD
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

def test_vectorbt_macd_fixed():
    """测试修复后的VectorBT MACD"""
    print("🧪 测试修复后的VectorBT MACD...")

    # 创建测试数据
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    price = pd.Series(np.random.uniform(100, 200, 100), index=dates)

    print(f"VectorBT版本: {vbt.__version__}")

    # 测试修复后的参数
    test_cases = [
        {"fast_window": 12, "slow_window": 26, "signal_window": 9},
        {"fast_window": 12, "slow_window": 26, "signal_window": 9, "macd_ewm": False, "signal_ewm": False},
    ]

    for i, params in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {params}")
        try:
            result = vbt.MACD.run(price, **params)
            print(f"✅ 成功: MACD={result.macd.shape}, Signal={result.signal.shape}, Hist={result.hist.shape}")
            print(f"  非空值: MACD={result.macd.notna().sum()}, Signal={result.signal.notna().sum()}, Hist={result.hist.notna().sum()}")
        except Exception as e:
            print(f"❌ 失败: {e}")

if __name__ == "__main__":
    test_vectorbt_macd_fixed()