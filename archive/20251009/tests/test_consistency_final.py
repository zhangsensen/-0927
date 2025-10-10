#!/usr/bin/env python3
"""
最终一致性测试
"""

import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.consistency_validator import (
    get_consistency_validator,
)
from factor_system.factor_engine.core.registry import get_global_registry


def test_final_consistency():
    """最终一致性测试"""
    print("🧪 最终一致性测试...")

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

    # 获取所有已注册因子
    registry = get_global_registry()
    all_factors = registry.list_factors()

    print(f"已注册因子总数: {len(all_factors)}")
    print(f"因子列表: {sorted(all_factors)}")

    # 验证一致性
    validator = get_consistency_validator()
    result = validator.validate_consistency(all_factors)

    print(f"\n📊 一致性验证结果:")
    print(f"  ✅ 有效因子: {len(result.valid_factors)}")
    print(f"  ❌ 无效因子: {len(result.invalid_factors)}")
    print(f"  ⚠️  缺失因子: {len(result.missing_factors)}")
    print(f"  📈 总体状态: {'✅ 通过' if result.is_valid else '❌ 失败'}")

    if result.invalid_factors:
        print(f"\n❌ 无效因子:")
        for factor in result.invalid_factors:
            print(f"  - {factor}")

    if result.warnings:
        print(f"\n⚠️ 警告:")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print(f"\n❌ 错误:")
        for error in result.errors:
            print(f"  - {error}")

    return result.is_valid


def test_shared_vs_vectorbt_consistency():
    """测试共享计算器与VectorBT适配器的一致性"""
    print("\n🔍 共享计算器 vs VectorBT适配器一致性测试...")

    from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter
    from factor_system.shared.factor_calculators import SHARED_CALCULATORS

    # 创建测试数据
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    test_data = pd.DataFrame(
        {
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(100, 200, 100),
            "close": np.random.uniform(100, 200, 100),
        },
        index=dates,
    )

    test_data["high"] = np.maximum(test_data["high"], test_data["low"])

    high = test_data["high"]
    low = test_data["low"]
    close = test_data["close"]

    adapter = get_vectorbt_adapter()

    # 测试RSI一致性
    shared_rsi = SHARED_CALCULATORS.calculate_rsi(close, period=14)
    vbt_rsi = adapter.calculate_rsi(close, timeperiod=14)

    rsi_diff = np.abs(shared_rsi - vbt_rsi).max()
    print(f"RSI最大差异: {rsi_diff:.6f}")

    # 测试ATR一致性
    shared_atr = SHARED_CALCULATORS.calculate_atr(high, low, close, timeperiod=14)
    vbt_atr = adapter.calculate_atr(high, low, close, timeperiod=14)

    atr_diff = np.abs(shared_atr - vbt_atr).max()
    print(f"ATR最大差异: {atr_diff:.6f}")

    # 测试WILLR一致性
    shared_willr = SHARED_CALCULATORS.calculate_willr(high, low, close, timeperiod=14)
    vbt_willr = adapter.calculate_willr(high, low, close, timeperiod=14)

    willr_diff = np.abs(shared_willr - vbt_willr).max()
    print(f"WILLR最大差异: {willr_diff:.6f}")

    # 判断一致性
    tolerance = 1e-6
    rsi_ok = rsi_diff < tolerance
    atr_ok = atr_diff < tolerance
    willr_ok = willr_diff < tolerance

    print(f"\n📊 一致性结果:")
    print(f"  RSI: {'✅' if rsi_ok else '❌'} (差异: {rsi_diff:.6f})")
    print(f"  ATR: {'✅' if atr_ok else '❌'} (差异: {atr_diff:.6f})")
    print(f"  WILLR: {'✅' if willr_ok else '❌'} (差异: {willr_diff:.6f})")

    return rsi_ok and atr_ok and willr_ok


if __name__ == "__main__":
    consistency_ok = test_final_consistency()
    shared_vs_vbt_ok = test_shared_vs_vectorbt_consistency()

    print(f"\n🎯 最终结果:")
    print(f"  因子一致性验证: {'✅ 通过' if consistency_ok else '❌ 失败'}")
    print(f"  计算器一致性: {'✅ 通过' if shared_vs_vbt_ok else '❌ 失败'}")
    print(
        f"  整体状态: {'✅ 全部通过' if consistency_ok and shared_vs_vbt_ok else '❌ 存在问题'}"
    )
