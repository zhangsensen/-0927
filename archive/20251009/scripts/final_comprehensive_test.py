#!/usr/bin/env python3
"""
最终综合测试
"""

import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd


def test_complete_workflow():
    """完整工作流测试"""
    print("🧪 完整工作流测试...")

    # 1. 因子注册
    from factor_system.factor_engine.core.registry import get_global_registry
    from factor_system.factor_engine.factors.technical import ATR, MACD, RSI, STOCH

    registry = get_global_registry()
    registry.register(RSI)
    registry.register(MACD)
    registry.register(ATR)
    registry.register(STOCH)

    # 2. 创建测试数据
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

    test_data["high"] = np.maximum(test_data["high"], test_data["low"])

    # 3. 使用FactorEngine计算因子
    print("\n📊 FactorEngine计算...")
    factors = {}
    for factor_id in ["RSI", "MACD", "ATR", "STOCH"]:
        try:
            factor_instance = registry.get_factor(factor_id)
            result = factor_instance.calculate(test_data)
            factors[factor_id] = result
            print(f"  ✅ {factor_id}: {result.shape}, 非空值={result.notna().sum()}")
        except Exception as e:
            print(f"  ❌ {factor_id}: {e}")
            factors[factor_id] = None

    # 4. 一致性验证
    from factor_system.factor_engine.core.consistency_validator import (
        get_consistency_validator,
    )

    validator = get_consistency_validator()
    engine_factors = list(factors.keys())
    result = validator.validate_consistency(engine_factors)

    print(f"\n📋 一致性验证结果:")
    print(f"  ✅ 有效因子: {len(result.valid_factors)}")
    print(f"  ❌ 无效因子: {len(result.invalid_factors)}")
    print(f"  ⚠️  缺失因子: {len(result.missing_factors)}")
    print(f"  📈 总体状态: {'✅ 通过' if result.is_valid else '❌ 失败'}")

    # 5. 共享计算器一致性检查
    from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter
    from factor_system.shared.factor_calculators import SHARED_CALCULATORS

    adapter = get_vectorbt_adapter()
    high = test_data["high"]
    low = test_data["low"]
    close = test_data["close"]

    print(f"\n🔍 计算器一致性检查...")
    tolerance = 1e-6

    # RSI
    shared_rsi = SHARED_CALCULATORS.calculate_rsi(close, period=14)
    vbt_rsi = adapter.calculate_rsi(close, timeperiod=14)
    rsi_diff = np.abs(shared_rsi - vbt_rsi).max()
    rsi_ok = rsi_diff < tolerance
    print(f"  RSI: {'✅' if rsi_ok else '❌'} (差异: {rsi_diff:.6f})")

    # MACD
    shared_macd = SHARED_CALCULATORS.calculate_macd(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )["macd"]
    vbt_macd = adapter.calculate_macd(
        close, fast_period=12, slow_period=26, signal_period=9
    )
    macd_diff = np.abs(shared_macd - vbt_macd).max()
    macd_ok = macd_diff < tolerance
    print(f"  MACD: {'✅' if macd_ok else '❌'} (差异: {macd_diff:.6f})")

    # ATR
    shared_atr = SHARED_CALCULATORS.calculate_atr(high, low, close, timeperiod=14)
    vbt_atr = adapter.calculate_atr(high, low, close, timeperiod=14)
    atr_diff = np.abs(shared_atr - vbt_atr).max()
    atr_ok = atr_diff < tolerance
    print(f"  ATR: {'✅' if atr_ok else '❌'} (差异: {atr_diff:.6f})")

    # 6. 综合评估
    all_consistency_ok = rsi_ok and macd_ok and atr_ok
    engine_validation_ok = result.is_valid

    print(f"\n🎯 综合评估:")
    print(f"  FactorEngine验证: {'✅ 通过' if engine_validation_ok else '❌ 失败'}")
    print(f"  计算器一致性: {'✅ 通过' if all_consistency_ok else '❌ 失败'}")
    print(
        f"  整体状态: {'✅ 全部通过' if engine_validation_ok and all_consistency_ok else '❌ 存在问题'}"
    )

    return engine_validation_ok and all_consistency_ok


if __name__ == "__main__":
    success = test_complete_workflow()
    if success:
        print("\n🎉 所有一致性测试通过！FactorEngine可以作为统一服务层使用。")
    else:
        print("\n⚠️ 仍存在问题，需要进一步修复。")
