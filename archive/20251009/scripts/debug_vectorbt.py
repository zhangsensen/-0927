#!/usr/bin/env python3
"""
调试VectorBT指标可用性问题
"""

import vectorbt as vbt
import pandas as pd
import numpy as np

def check_vectorbt_indicators():
    """检查VectorBT中可用的指标"""
    print("🔍 检查VectorBT指标可用性...")

    # 创建测试数据
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100),
    }, index=dates)

    price = test_data['close']
    high = test_data['high']
    low = test_data['low']
    volume = test_data['volume']

    # 检查VectorBT版本和支持的指标
    print(f"VectorBT版本: {vbt.__version__}")

    # 检查TA-Lib支持
    if hasattr(vbt, 'talib'):
        print("✅ TA-Lib支持可用")

        # 测试一些常见的TA-Lib指标
        talib_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'STOCH', 'ATR', 'WILLR', 'BBANDS']
        for indicator in talib_indicators:
            try:
                talib_func = vbt.talib(indicator)
                print(f"  ✅ TA_{indicator}: 可用")
            except Exception as e:
                print(f"  ❌ TA_{indicator}: 不可用 - {e}")
    else:
        print("❌ TA-Lib支持不可用")

    # 检查VectorBT内置指标
    vbt_indicators = ['RSI', 'MACD', 'STOCH', 'ATR']
    print("\nVectorBT内置指标:")
    for indicator in vbt_indicators:
        if hasattr(vbt, indicator):
            print(f"  ✅ {indicator}: 可用")
        else:
            print(f"  ❌ {indicator}: 不可用")

    # 测试实际计算
    print("\n🧪 测试实际计算...")

    # 测试RSI
    try:
        if hasattr(vbt, 'RSI'):
            result = vbt.RSI.run(price, window=14)
            print(f"✅ RSI计算成功: {result.rsi.shape}")
        else:
            print("❌ RSI不可用")
    except Exception as e:
        print(f"❌ RSI计算失败: {e}")

    # 测试MACD
    try:
        if hasattr(vbt, 'MACD'):
            result = vbt.MACD.run(price, fast=12, slow=26, signal=9)
            print(f"✅ MACD计算成功: MACD={result.macd.shape}, Signal={result.signal.shape}")
        else:
            print("❌ MACD不可用")
    except Exception as e:
        print(f"❌ MACD计算失败: {e}")

    # 测试ATR
    try:
        if hasattr(vbt, 'ATR'):
            result = vbt.ATR.run(high, low, price, window=14)
            print(f"✅ ATR计算成功: {result.atr.shape}")
        else:
            print("❌ ATR不可用")
    except Exception as e:
        print(f"❌ ATR计算失败: {e}")

    # 测试TA-Lib SMA
    try:
        if hasattr(vbt, 'talib'):
            talib_sma = vbt.talib('SMA')
            result = talib_sma.run(price, timeperiod=20)
            print(f"✅ TA-Lib SMA计算成功: {result.real.shape}")
        else:
            print("❌ TA-Lib不可用")
    except Exception as e:
        print(f"❌ TA-Lib SMA计算失败: {e}")

    # 测试TA-Lib EMA
    try:
        if hasattr(vbt, 'talib'):
            talib_ema = vbt.talib('EMA')
            result = talib_ema.run(price, timeperiod=20)
            print(f"✅ TA-Lib EMA计算成功: {result.real.shape}")
        else:
            print("❌ TA-Lib不可用")
    except Exception as e:
        print(f"❌ TA-Lib EMA计算失败: {e}")

    # 列出所有VectorBT属性
    print("\n📋 VectorBT所有属性:")
    all_attrs = [attr for attr in dir(vbt) if not attr.startswith('_')]
    indicator_attrs = [attr for attr in all_attrs if attr.isupper()]

    print("  技术指标类:")
    for attr in sorted(indicator_attrs):
        print(f"    - {attr}")

def check_shared_calculators():
    """检查共享计算器"""
    print("\n🔍 检查共享计算器...")

    try:
        from factor_system.shared.factor_calculators import SHARED_CALCULATORS
        print("✅ 共享计算器导入成功")

        # 测试RSI计算
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        price = pd.Series(np.random.uniform(100, 200, 100), index=dates)

        rsi_result = SHARED_CALCULATORS.calculate_rsi(price, period=14)
        print(f"✅ 共享计算器RSI: {rsi_result.shape}, 非空值: {rsi_result.notna().sum()}")

        # 测试MACD计算
        macd_result = SHARED_CALCULATORS.calculate_macd(price, fastperiod=12, slowperiod=26, signalperiod=9)
        print(f"✅ 共享计算器MACD: {len(macd_result)} 个组件")
        for key, value in macd_result.items():
            print(f"  - {key}: {value.shape}, 非空值: {value.notna().sum()}")

    except Exception as e:
        print(f"❌ 共享计算器检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_vectorbt_indicators()
    check_shared_calculators()