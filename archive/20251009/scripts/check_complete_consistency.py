#!/usr/bin/env python3
"""
检查完整一致性 - factor_generation vs FactorEngine
"""

import sys

sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

import numpy as np
import pandas as pd


def check_factor_generation_factors():
    """检查factor_generation中实际支持的因子"""
    print("🔍 检查factor_generation中实际支持的因子...")

    # 读取enhanced_factor_calculator.py文件
    with open(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py",
        "r",
    ) as f:
        content = f.read()

    # 查找calculate_comprehensive_factors函数中的因子
    import re

    # 查找所有calculate_开头的函数调用
    factor_calls = re.findall(r"SHARED_CALCULATORS\.calculate_[a-zA-Z_]+\(", content)
    factor_names = set()

    for call in factor_calls:
        factor_name = call.replace("SHARED_CALCULATORS.calculate_", "").replace("(", "")
        factor_names.add(factor_name)

    print(f"factor_generation中发现的因子函数: {sorted(factor_names)}")

    # 映射到标准因子名
    factor_mapping = {
        "rsi": "RSI",
        "macd": "MACD",
        "atr": "ATR",
        "stoch": "STOCH",
        "willr": "WILLR",
        "bbands": "BBANDS",
        "cci": "CCI",
        "mfi": "MFI",
        "obv": "OBV",
        "adx": "ADX",
        "sma": "SMA",
        "ema": "EMA",
    }

    actual_factors = set()
    for factor in factor_names:
        if factor in factor_mapping:
            actual_factors.add(factor_mapping[factor])
        else:
            actual_factors.add(factor.upper())

    print(f"映射后的标准因子名: {sorted(actual_factors)}")
    return actual_factors


def check_factor_engine_factors():
    """检查FactorEngine中支持的因子"""
    print("\n🔍 检查FactorEngine中支持的因子...")

    from factor_system.factor_engine.core.registry import get_global_registry
    from factor_system.factor_engine.factors import technical

    registry = get_global_registry()

    # 手动注册所有可用的技术因子
    available_factors = []

    # 检查所有已实现的因子类
    for attr_name in dir(technical):
        attr = getattr(technical, attr_name)
        if (
            isinstance(attr, type)
            and hasattr(attr, "__bases__")
            and any("BaseFactor" in str(base) for base in attr.__bases__)
            and attr_name not in ["BaseFactor"]
        ):
            try:
                registry.register(attr)
                available_factors.append(attr_name)
                print(f"  ✅ 注册因子: {attr_name}")
            except Exception as e:
                print(f"  ❌ 注册失败 {attr_name}: {e}")

    all_factors = registry.list_factors()
    print(f"FactorEngine中的因子: {sorted(all_factors)}")
    return set(all_factors)


def analyze_consistency():
    """分析一致性"""
    print("\n📊 分析一致性...")

    gen_factors = check_factor_generation_factors()
    engine_factors = check_factor_engine_factors()

    print(f"\n📈 一致性分析结果:")
    print(f"  factor_generation中的因子: {len(gen_factors)} 个")
    print(f"  FactorEngine中的因子: {len(engine_factors)} 个")

    # 找出共同的因子
    common_factors = gen_factors & engine_factors
    print(f"  共同因子: {len(common_factors)} 个")
    if common_factors:
        print(f"    {sorted(common_factors)}")

    # 找出factor_generation中有但FactorEngine中没有的因子
    missing_in_engine = gen_factors - engine_factors
    print(f"  FactorEngine缺失的因子: {len(missing_in_engine)} 个")
    if missing_in_engine:
        print(f"    {sorted(missing_in_engine)}")

    # 找出FactorEngine中有但factor_generation中没有的因子
    extra_in_engine = engine_factors - gen_factors
    print(f"  FactorEngine多余的因子: {len(extra_in_engine)} 个")
    if extra_in_engine:
        print(f"    {sorted(extra_in_engine)}")

    # 计算一致性比例
    if len(gen_factors) > 0:
        consistency_ratio = len(common_factors) / len(gen_factors) * 100
        print(f"\n🎯 一致性比例: {consistency_ratio:.1f}%")

        if consistency_ratio == 100:
            print(
                "✅ 完全一致！factor_generation中的所有因子都在FactorEngine中得到支持"
            )
        elif consistency_ratio >= 80:
            print("⚠️ 高度一致，但仍有少数因子缺失")
        else:
            print("❌ 一致性不足，需要进一步修复")

    return common_factors, missing_in_engine, extra_in_engine


if __name__ == "__main__":
    common, missing, extra = analyze_consistency()

    print(f"\n🔧 需要采取的行动:")
    if missing:
        print(f"  - 在FactorEngine中实现缺失的因子: {sorted(missing)}")
    if extra:
        print(f"  - 移除或验证多余的因子: {sorted(extra)}")
    if not missing and not extra:
        print("  - ✅ 无需进一步操作，已完全一致")
