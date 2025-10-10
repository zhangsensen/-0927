#!/usr/bin/env python3
"""
调试VectorBT STOCH结果
"""

import pandas as pd
import numpy as np
import vectorbt as vbt

def debug_stoch_result():
    """调试STOCH结果属性"""
    print("🔍 VectorBT STOCH结果调试...")

    # 创建测试数据
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    test_data = pd.DataFrame({
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
    }, index=dates)

    # 确保high >= low
    test_data['high'] = np.maximum(test_data['high'], test_data['low'])

    high = test_data['high']
    low = test_data['low']
    close = test_data['close']

    try:
        result = vbt.STOCH.run(high, low, close, k_window=14, d_window=3)
        print(f"✅ STOCH计算成功")
        print(f"STOCH结果类型: {type(result)}")
        print(f"STOCH结果属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        # 检查各个属性
        for attr in ['stoch_k', 'stoch_d', 'slowk', 'slowd', 'k', 'd']:
            if hasattr(result, attr):
                data = getattr(result, attr)
                print(f"  - {attr}: {type(data)}, shape={getattr(data, 'shape', 'N/A')}, 非空值={data.notna().sum() if hasattr(data, 'notna') else 'N/A'}")

    except Exception as e:
        print(f"❌ STOCH计算失败: {e}")

if __name__ == "__main__":
    debug_stoch_result()