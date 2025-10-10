#!/usr/bin/env python3
"""
测试因子注册
"""

import sys
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927')

def test_factor_registration():
    """测试因子注册"""
    print("🧪 测试因子注册...")

    from factor_system.factor_engine.core.registry import get_global_registry
    from factor_system.factor_engine.factors.technical import RSI, MACD, ATR, STOCH, WILLR

    registry = get_global_registry()

    # 手动注册一些因子
    registry.register(RSI)
    registry.register(MACD)
    registry.register(ATR)
    registry.register(STOCH)
    registry.register(WILLR)

    all_factors = registry.list_factors()
    print(f"已注册因子: {sorted(all_factors)}")

    return len(all_factors) > 0

if __name__ == "__main__":
    success = test_factor_registration()
    print(f"因子注册: {'✅ 成功' if success else '❌ 失败'}")