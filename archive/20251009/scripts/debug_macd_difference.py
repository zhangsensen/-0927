#!/usr/bin/env python3
"""
调试MACD计算差异
"""

import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd
import talib

from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


def debug_macd_difference():
    """调试MACD计算差异"""
    print("🔍 MACD计算差异调试...")

    # 创建简单的测试数据
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    price = pd.Series(np.random.uniform(100, 200, 50), index=dates)

    # 方法1: TA-Lib直接计算
    talib_macd = talib.MACD(price, fastperiod=12, slowperiod=26, signalperiod=9)
    print(
        f"TA-Lib MACD: 非空值={talib_macd[0].notna().sum()}, 前5个值={talib_macd[0].dropna().head().values}"
    )

    # 方法2: 共享计算器
    shared_macd = SHARED_CALCULATORS.calculate_macd(
        price, fastperiod=12, slowperiod=26, signalperiod=9
    )
    print(
        f"共享计算器MACD: 非空值={shared_macd['macd'].notna().sum()}, 前5个值={shared_macd['macd'].dropna().head().values}"
    )

    # 方法3: VectorBT
    adapter = get_vectorbt_adapter()
    vbt_macd = adapter.calculate_macd(
        price, fast_period=12, slow_period=26, signal_period=9
    )
    print(
        f"VectorBT MACD: 非空值={vbt_macd.notna().sum()}, 前5个值={vbt_macd.dropna().head().values}"
    )

    # 计算差异
    shared_vs_talib = np.abs(shared_macd["macd"] - talib_macd[0]).max()
    vbt_vs_talib = np.abs(vbt_macd - talib_macd[0]).max()
    shared_vs_vbt = np.abs(shared_macd["macd"] - vbt_macd).max()

    print(f"\n📊 差异分析:")
    print(f"  共享计算器 vs TA-Lib: {shared_vs_talib:.6f}")
    print(f"  VectorBT vs TA-Lib: {vbt_vs_talib:.6f}")
    print(f"  共享计算器 vs VectorBT: {shared_vs_vbt:.6f}")

    # 检查VectorBT MACD参数
    import inspect

    import vectorbt as vbt

    sig = inspect.signature(vbt.MACD.run)
    print(f"\n🔍 VectorBT MACD参数:")
    for name, param in sig.parameters.items():
        if name != "self":
            print(
                f"  - {name}: {param.default if param.default != inspect.Parameter.empty else 'required'}"
            )


if __name__ == "__main__":
    debug_macd_difference()
