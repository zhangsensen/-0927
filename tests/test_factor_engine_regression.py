"""
FactorEngine API 回归测试

验证关键修复：
1. 包导入一致性
2. calculate_factors API 返回结果过滤
3. 缓存一致性
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# 测试导入是否正常
def test_import_consistency():
    """测试所有关键模块可以正常导入"""
    try:
        from factor_system.factor_engine import api
        from factor_system.factor_engine.core.cache import CacheManager
        from factor_system.factor_engine.core.engine import FactorEngine
        from factor_system.factor_engine.core.registry import get_global_registry
        from factor_system.factor_engine.providers.csv_provider import CSVDataProvider
        from factor_system.shared.factor_calculators import SHARED_CALCULATORS

        print("✅ 所有关键模块导入成功")
        return True
    except ImportError as e:
        pytest.fail(f"模块导入失败: {e}")


def test_calculate_factors_api_filtering():
    """测试 calculate_factors API 只返回请求的因子"""
    try:
        from factor_system.factor_engine import api

        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 50)
        close_series = pd.Series(base_price * (1 + returns).cumprod())

        high = close_series * (1 + np.abs(np.random.normal(0, 0.01, 50)))
        low = close_series * (1 - np.abs(np.random.normal(0, 0.01, 50)))
        open_price = close_series.shift(1).fillna(base_price)

        test_data = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_series,
                "volume": np.random.randint(100000, 1000000, 50),
            },
            index=dates,
        )

        # 只请求 RSI 因子
        result = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0700.HK"],
            timeframe="daily",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 20),
            use_cache=False,
        )

        # 验证结果只包含请求的因子
        assert not result.empty, "计算结果不应为空"
        assert "RSI" in result.columns, "结果应包含RSI因子"

        # 检查是否只返回了请求的因子（不包括依赖因子）
        unexpected_columns = [col for col in result.columns if col != "RSI"]
        assert len(unexpected_columns) == 0, f"结果包含意外的列: {unexpected_columns}"

        print("✅ calculate_factors API 过滤测试通过")
        return True

    except Exception as e:
        pytest.fail(f"calculate_factors API 过滤测试失败: {e}")


def test_multiple_factors_api_filtering():
    """测试多因子请求时的API过滤"""
    try:
        from factor_system.factor_engine import api

        # 请求多个因子
        requested_factors = ["RSI", "MACD", "STOCH"]

        result = api.calculate_factors(
            factor_ids=requested_factors,
            symbols=["0700.HK"],
            timeframe="daily",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            use_cache=False,
        )

        # 验证结果包含所有请求的因子
        assert not result.empty, "计算结果不应为空"

        for factor in requested_factors:
            assert factor in result.columns, f"结果应包含{factor}因子"

        # 验证结果不包含意外的因子
        # 注意：MACD 可能返回多个相关列，如 MACD, MACD_SIGNAL, MACD_HIST
        allowed_columns = set(
            requested_factors
            + ["MACD_SIGNAL", "MACD_HIST", "STOCH_SLOWK", "STOCH_SLOWD"]
        )
        unexpected_columns = [
            col for col in result.columns if col not in allowed_columns
        ]
        assert len(unexpected_columns) == 0, f"结果包含意外的列: {unexpected_columns}"

        print("✅ 多因子 API 过滤测试通过")
        return True

    except Exception as e:
        pytest.fail(f"多因子 API 过滤测试失败: {e}")


def test_cache_consistency():
    """测试缓存一致性"""
    try:
        from factor_system.factor_engine import api

        # 第一次计算（无缓存）
        result1 = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0700.HK"],
            timeframe="daily",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            use_cache=True,
        )

        # 第二次计算（使用缓存）
        result2 = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0700.HK"],
            timeframe="daily",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            use_cache=True,
        )

        # 验证两次结果一致
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)

        # 验证结果只包含请求的因子
        assert "RSI" in result2.columns, "缓存结果应包含RSI因子"
        unexpected_columns = [col for col in result2.columns if col != "RSI"]
        assert (
            len(unexpected_columns) == 0
        ), f"缓存结果包含意外的列: {unexpected_columns}"

        print("✅ 缓存一致性测试通过")
        return True

    except Exception as e:
        pytest.fail(f"缓存一致性测试失败: {e}")


def test_shared_calculators_integration():
    """测试共享计算器集成"""
    try:
        from factor_system.shared.factor_calculators import SHARED_CALCULATORS

        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        close = pd.Series(100 + np.random.normal(0, 1, 100).cumsum(), index=dates)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, 100)))

        # 测试 RSI 计算
        rsi = SHARED_CALCULATORS.calculate_rsi(close, period=14)
        assert not rsi.empty, "RSI计算结果不应为空"

        # 测试 MACD 计算
        macd = SHARED_CALCULATORS.calculate_macd(close)
        assert isinstance(macd, dict), "MACD应返回字典"
        assert "macd" in macd, "MACD结果应包含macd列"

        # 测试 STOCH 计算
        stoch = SHARED_CALCULATORS.calculate_stoch(high, low, close)
        assert isinstance(stoch, dict), "STOCH应返回字典"
        assert "slowk" in stoch, "STOCH结果应包含slowk列"

        print("✅ 共享计算器集成测试通过")
        return True

    except Exception as e:
        pytest.fail(f"共享计算器集成测试失败: {e}")


def test_package_structure():
    """测试包结构完整性"""
    try:
        # 测试关键模块是否可以正常导入
        from factor_system.factor_engine.core import cache, engine, registry
        from factor_system.factor_engine.factors import technical
        from factor_system.factor_engine.providers import csv_provider
        from factor_system.shared import factor_calculators

        print("✅ 包结构完整性测试通过")
        return True

    except ImportError as e:
        pytest.fail(f"包结构完整性测试失败: {e}")


if __name__ == "__main__":
    """运行所有回归测试"""
    print("🔧 开始运行 FactorEngine 回归测试...")

    tests = [
        ("导入一致性", test_import_consistency),
        ("API过滤", test_calculate_factors_api_filtering),
        ("多因子API过滤", test_multiple_factors_api_filtering),
        ("缓存一致性", test_cache_consistency),
        ("共享计算器集成", test_shared_calculators_integration),
        ("包结构完整性", test_package_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试失败: {e}")
            failed += 1

    print(f"\n📊 测试结果: {passed}个通过, {failed}个失败")

    if failed == 0:
        print("🎉 所有回归测试通过！修复效果验证成功。")
    else:
        print("⚠️ 部分测试失败，需要进一步修复。")
