#!/usr/bin/env python3
"""
测试修复后的VectorBT适配器
"""

import os
import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


def test_vectorbt_adapter_fixed():
    """测试修复后的VectorBT适配器"""
    print("🧪 测试修复后的VectorBT适配器...")

    # 创建测试数据
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    test_data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(100, 200, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    # 确保high >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["low"])

    price = test_data["close"]
    high = test_data["high"]
    low = test_data["low"]

    adapter = get_vectorbt_adapter()

    # 测试RSI
    try:
        rsi = adapter.calculate_rsi(price, timeperiod=14)
        print(f"✅ RSI: {rsi.shape}, 非空值: {rsi.notna().sum()}")
    except Exception as e:
        print(f"❌ RSI失败: {e}")

    # 测试MACD
    try:
        macd = adapter.calculate_macd(
            price, fast_period=12, slow_period=26, signal_period=9
        )
        print(f"✅ MACD: {macd.shape}, 非空值: {macd.notna().sum()}")
    except Exception as e:
        print(f"❌ MACD失败: {e}")

    # 测试ATR
    try:
        atr = adapter.calculate_atr(high, low, price, timeperiod=14)
        print(f"✅ ATR: {atr.shape}, 非空值: {atr.notna().sum()}")
    except Exception as e:
        print(f"❌ ATR失败: {e}")

    # 测试STOCH
    try:
        stoch = adapter.calculate_stoch(
            high, low, price, fastk_period=5, slowk_period=3, slowd_period=3
        )
        print(f"✅ STOCH: {stoch.shape}, 非空值: {stoch.notna().sum()}")
    except Exception as e:
        print(f"❌ STOCH失败: {e}")

    # 测试WILLR
    try:
        willr = adapter.calculate_willr(high, low, price, timeperiod=14)
        print(f"✅ WILLR: {willr.shape}, 非空值: {willr.notna().sum()}")
    except Exception as e:
        print(f"❌ WILLR失败: {e}")


if __name__ == "__main__":
    test_vectorbt_adapter_fixed()
