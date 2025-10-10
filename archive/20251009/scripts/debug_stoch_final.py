#!/usr/bin/env python3
"""
调试STOCH最终差异
"""

import sys
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927')

import pandas as pd
import numpy as np
import talib
from factor_system.shared.factor_calculators import SHARED_CALCULATORS
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter

def debug_stoch_final():
    """调试STOCH最终差异"""
    print("🔍 STOCH最终差异调试...")

    # 创建简单的测试数据
    dates = pd.date_range('2025-01-01', periods=50, freq='D')
    test_data = pd.DataFrame({
        'high': np.random.uniform(100, 200, 50),
        'low': np.random.uniform(100, 200, 50),
        'close': np.random.uniform(100, 200, 50),
    }, index=dates)

    test_data['high'] = np.maximum(test_data['high'], test_data['low'])

    high = test_data['high']
    low = test_data['low']
    close = test_data['close']

    # 方法1: TA-Lib直接计算
    talib_stoch = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    print(f"TA-Lib STOCH: {len(talib_stoch)} 个组件")
    for i, component in enumerate(talib_stoch):
        if component is not None:
            print(f"  组件{i}: 非空值={component.notna().sum()}, 前5个值={component.dropna().head().values}")

    # 方法2: 共享计算器
    shared_stoch = SHARED_CALCULATORS.calculate_stoch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    print(f"共享计算器STOCH: {len(shared_stoch)} 个组件")
    for key, value in shared_stoch.items():
        print(f"  {key}: 非空值={value.notna().sum()}, 前5个值={value.dropna().head().values}")

    # 方法3: VectorBT
    adapter = get_vectorbt_adapter()
    vbt_stoch = adapter.calculate_stoch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    print(f"VectorBT STOCH: 非空值={vbt_stoch.notna().sum()}, 前5个值={vbt_stoch.dropna().head().values}")

    # 分析差异
    if 'slowk' in shared_stoch:
        shared_slowk = shared_stoch['slowk']
        # 找到共同非NaN的索引
        both_notna = shared_slowk.notna() & vbt_stoch.notna()
        if both_notna.sum() > 0:
            max_diff = np.abs(shared_slowk[both_notna] - vbt_stoch[both_notna]).max()
            print(f"\n📊 STOCH差异分析:")
            print(f"  共享计算器 vs VectorBT: {max_diff:.6f}")

            # 比较与TA-Lib的差异
            if talib_stoch[0] is not None:
                talib_slowk = talib_stoch[0]
                talib_vs_shared = np.abs(talib_slowk - shared_slowk).max()
                talib_vs_vbt = np.abs(talib_slowk - vbt_stoch).max()
                print(f"  TA-Lib vs 共享计算器: {talib_vs_shared:.6f}")
                print(f"  TA-Lib vs VectorBT: {talib_vs_vbt:.6f}")

if __name__ == "__main__":
    debug_stoch_final()