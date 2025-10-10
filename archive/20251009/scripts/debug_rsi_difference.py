#!/usr/bin/env python3
"""
调试RSI计算差异
"""

import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd
import talib

from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


def debug_rsi_difference():
    """调试RSI计算差异"""
    print("🔍 RSI计算差异调试...")

    # 创建简单的测试数据
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    price = pd.Series(np.random.uniform(100, 200, 50), index=dates)

    # 方法1: TA-Lib直接计算
    talib_rsi = talib.RSI(price, timeperiod=14)
    print(
        f"TA-Lib RSI: 非空值={talib_rsi.notna().sum()}, 前5个值={talib_rsi.dropna().head().values}"
    )

    # 方法2: 共享计算器
    shared_rsi = SHARED_CALCULATORS.calculate_rsi(price, period=14)
    print(
        f"共享计算器RSI: 非空值={shared_rsi.notna().sum()}, 前5个值={shared_rsi.dropna().head().values}"
    )

    # 方法3: VectorBT
    adapter = get_vectorbt_adapter()
    vbt_rsi = adapter.calculate_rsi(price, timeperiod=14)
    print(
        f"VectorBT RSI: 非空值={vbt_rsi.notna().sum()}, 前5个值={vbt_rsi.dropna().head().values}"
    )

    # 计算差异
    shared_vs_talib = np.abs(shared_rsi - talib_rsi).max()
    vbt_vs_talib = np.abs(vbt_rsi - talib_rsi).max()
    shared_vs_vbt = np.abs(shared_rsi - vbt_rsi).max()

    print(f"\n📊 差异分析:")
    print(f"  共享计算器 vs TA-Lib: {shared_vs_talib:.6f}")
    print(f"  VectorBT vs TA-Lib: {vbt_vs_talib:.6f}")
    print(f"  共享计算器 vs VectorBT: {shared_vs_vbt:.6f}")

    # 检查是否都是NaN
    shared_notna = shared_rsi.notna()
    vbt_notna = vbt_rsi.notna()
    both_notna = shared_notna & vbt_notna

    print(f"\n🔍 NaN分析:")
    print(f"  共享计算器非NaN: {shared_notna.sum()}")
    print(f"  VectorBT非NaN: {vbt_notna.sum()}")
    print(f"  两者都非NaN: {both_notna.sum()}")

    if both_notna.sum() > 0:
        shared_valid = shared_rsi[both_notna]
        vbt_valid = vbt_rsi[both_notna]
        diff = np.abs(shared_valid - vbt_valid)
        print(f"  有效数据差异: 最大={diff.max():.6f}, 平均={diff.mean():.6f}")
        print(f"  共享计算器样本: {shared_valid.head().values}")
        print(f"  VectorBT样本: {vbt_valid.head().values}")


if __name__ == "__main__":
    debug_rsi_difference()
