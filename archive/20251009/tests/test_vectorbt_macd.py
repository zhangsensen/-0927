#!/usr/bin/env python3
"""
专门测试VectorBT MACD修复
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


def test_vectorbt_macd():
    """测试VectorBT MACD参数"""
    print("🧪 测试VectorBT MACD...")

    # 创建测试数据
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    price = pd.Series(np.random.uniform(100, 200, 100), index=dates)

    print(f"VectorBT版本: {vbt.__version__}")

    # 测试不同参数组合
    test_cases = [
        {"fast": 12, "slow": 26, "signal": 9},
        {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    ]

    for i, params in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: {params}")
        try:
            result = vbt.MACD.run(price, **params)
            print(
                f"✅ 成功: MACD={result.macd.shape}, Signal={result.signal.shape}, Hist={result.hist.shape}"
            )
        except Exception as e:
            print(f"❌ 失败: {e}")

    # 测试MACD类的属性
    print(f"\n🔍 MACD.run参数:")
    import inspect

    sig = inspect.signature(vbt.MACD.run)
    for name, param in sig.parameters.items():
        if name != "self":
            print(
                f"  - {name}: {param.default if param.default != inspect.Parameter.empty else 'required'}"
            )


if __name__ == "__main__":
    test_vectorbt_macd()
