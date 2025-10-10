#!/usr/bin/env python3
"""
验证清理后的适配器
"""

import sys
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927')

import pandas as pd
import numpy as np

def verify_cleaned_adapter():
    """验证清理后的适配器"""
    print("🔍 验证清理后的适配器...")

    from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter

    # 创建测试数据
    dates = pd.date_range('2025-01-01', periods=50, freq='D')
    test_data = pd.DataFrame({
        'high': np.random.uniform(100, 200, 50),
        'low': np.random.uniform(100, 200, 50),
        'close': np.random.uniform(100, 200, 50),
    }, index=dates)

    test_data['high'] = np.maximum(test_data['high'], test_data['low'])

    adapter = get_vectorbt_adapter()

    # 测试支持的方法
    supported_methods = [
        'calculate_rsi',
        'calculate_stoch',
        'calculate_macd',
        'calculate_macd_signal',
        'calculate_macd_histogram'
    ]

    for method in supported_methods:
        if hasattr(adapter, method):
            print(f"  ✅ 支持方法: {method}")
        else:
            print(f"  ❌ 缺失方法: {method}")

    # 检查是否有不必要的方法
    all_methods = [method for method in dir(adapter) if method.startswith('calculate_')]
    unsupported_methods = set(all_methods) - set(supported_methods)

    if unsupported_methods:
        print(f"  ⚠️  不应该支持的方法: {unsupported_methods}")
    else:
        print(f"  ✅ 只支持必要的方法")

    # 测试基本功能
    try:
        rsi = adapter.calculate_rsi(test_data['close'], timeperiod=14)
        print(f"  ✅ RSI计算测试通过: {rsi.shape}")

        macd = adapter.calculate_macd(test_data['close'], fast_period=12, slow_period=26, signal_period=9)
        print(f"  ✅ MACD计算测试通过: {macd.shape}")

        stoch = adapter.calculate_stoch(test_data['high'], test_data['low'], test_data['close'])
        print(f"  ✅ STOCH计算测试通过: {stoch.shape}")

        return True

    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
        return False

if __name__ == "__main__":
    success = verify_cleaned_adapter()

    if success:
        print(f"\n🎉 VectorBT适配器验证通过！")
    else:
        print(f"\n⚠️ 验证失败")