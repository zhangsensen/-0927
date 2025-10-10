#!/usr/bin/env python3
"""
共享计算器一致性测试
"""

import numpy as np
import pandas as pd

from factor_system.shared.factor_calculators import SHARED_CALCULATORS


def test_shared_calculators():
    """测试共享计算器的所有功能"""
    print("🧪 测试共享计算器...")

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
    volume = test_data["volume"]

    # 测试RSI
    try:
        rsi_result = SHARED_CALCULATORS.calculate_rsi(price, period=14)
        print(f"✅ RSI: {rsi_result.shape}, 非空值: {rsi_result.notna().sum()}")
    except Exception as e:
        print(f"❌ RSI失败: {e}")

    # 测试MACD
    try:
        macd_result = SHARED_CALCULATORS.calculate_macd(
            price, fastperiod=12, slowperiod=26, signalperiod=9
        )
        print(f"✅ MACD: {len(macd_result)} 个组件")
        for key, value in macd_result.items():
            print(f"  - {key}: {value.shape}, 非空值: {value.notna().sum()}")
    except Exception as e:
        print(f"❌ MACD失败: {e}")

    # 测试ATR
    try:
        atr_result = SHARED_CALCULATORS.calculate_atr(high, low, price, timeperiod=14)
        print(f"✅ ATR: {atr_result.shape}, 非空值: {atr_result.notna().sum()}")
    except Exception as e:
        print(f"❌ ATR失败: {e}")

    # 测试STOCH
    try:
        stoch_result = SHARED_CALCULATORS.calculate_stoch(
            high, low, price, fastk_period=5, slowk_period=3, slowd_period=3
        )
        print(f"✅ STOCH: {len(stoch_result)} 个组件")
        for key, value in stoch_result.items():
            print(f"  - {key}: {value.shape}, 非空值: {value.notna().sum()}")
    except Exception as e:
        print(f"❌ STOCH失败: {e}")

    # 测试WILLR
    try:
        willr_result = SHARED_CALCULATORS.calculate_willr(
            high, low, price, timeperiod=14
        )
        print(f"✅ WILLR: {willr_result.shape}, 非空值: {willr_result.notna().sum()}")
    except Exception as e:
        print(f"❌ WILLR失败: {e}")

    # 测试SMA
    try:
        sma_result = SHARED_CALCULATORS.calculate_sma(price, timeperiod=20)
        print(f"✅ SMA: {sma_result.shape}, 非空值: {sma_result.notna().sum()}")
    except Exception as e:
        print(f"❌ SMA失败: {e}")

    # 测试EMA
    try:
        ema_result = SHARED_CALCULATORS.calculate_ema(price, timeperiod=20)
        print(f"✅ EMA: {ema_result.shape}, 非空值: {ema_result.notna().sum()}")
    except Exception as e:
        print(f"❌ EMA失败: {e}")


if __name__ == "__main__":
    test_shared_calculators()
