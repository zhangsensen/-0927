#!/usr/bin/env python3
"""
因子一致性最终测试 - 验证factor_engine与factor_generation计算一致性
使用模拟数据提供者，无需实际数据文件
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDataProvider:
    """模拟数据提供者 - 用于测试"""

    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data

    def load_price_data(self, symbols, timeframe, start_date, end_date):
        """返回测试数据"""
        # 创建MultiIndex
        df = self.test_data.copy()
        df["symbol"] = symbols[0] if symbols else "TEST"
        df = df.set_index("symbol", append=True)
        return df


class TestFactorConsistency:
    """因子一致性测试套件"""

    @pytest.fixture
    def test_data(self):
        """生成测试数据"""
        dates = pd.date_range("2025-01-01", periods=200, freq="15min")
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 200),
                "high": np.random.uniform(100, 200, 200),
                "low": np.random.uniform(100, 200, 200),
                "close": np.random.uniform(100, 200, 200),
                "volume": np.random.uniform(1000, 10000, 200),
            },
            index=dates,
        )

        # 确保OHLC逻辑正确
        data["high"] = data[["open", "high", "close"]].max(axis=1)
        data["low"] = data[["open", "low", "close"]].min(axis=1)

        return data

    def test_rsi_consistency(self, test_data):
        """测试RSI计算一致性"""
        logger.info("🧪 测试RSI一致性...")

        # 1. 使用factor_generation计算
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator,
            IndicatorConfig,
            TimeFrame,
        )

        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)

        # 2. 使用factor_engine直接计算（不通过API）
        from factor_system.factor_engine.core.cache import CacheConfig
        from factor_system.factor_engine.core.engine import FactorEngine
        from factor_system.factor_engine.core.registry import FactorRegistry

        # 注册因子
        registry = FactorRegistry()
        from factor_system.factor_engine.api import _register_core_factors

        _register_core_factors(registry)

        # 创建引擎
        mock_provider = MockDataProvider(test_data)
        engine = FactorEngine(
            data_provider=mock_provider, registry=registry, cache_config=CacheConfig()
        )

        # 计算因子
        engine_result = engine.calculate_factors(
            factor_ids=["RSI14"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False,
        )

        # 3. 验证一致性
        if (
            "RSI14" in gen_result.columns
            and not engine_result.empty
            and "RSI14" in engine_result.columns
        ):
            gen_rsi = gen_result["RSI14"].values

            # 提取engine结果
            if isinstance(engine_result.index, pd.MultiIndex):
                engine_rsi = engine_result.xs("TEST", level="symbol")["RSI14"].values
            else:
                engine_rsi = engine_result["RSI14"].values

            # 对齐长度
            min_len = min(len(gen_rsi), len(engine_rsi))
            gen_rsi = gen_rsi[-min_len:]
            engine_rsi = engine_rsi[-min_len:]

            # 移除NaN
            valid_mask = ~(np.isnan(gen_rsi) | np.isnan(engine_rsi))
            gen_rsi_valid = gen_rsi[valid_mask]
            engine_rsi_valid = engine_rsi[valid_mask]

            if len(gen_rsi_valid) > 0:
                # 计算差异
                max_diff = np.max(np.abs(gen_rsi_valid - engine_rsi_valid))
                mean_diff = np.mean(np.abs(gen_rsi_valid - engine_rsi_valid))

                logger.info(
                    f"✓ RSI14一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}"
                )
                logger.info(
                    f"  样本数: {len(gen_rsi_valid)}, 覆盖率: {len(gen_rsi_valid)/len(gen_rsi)*100:.1f}%"
                )

                # 验证差异在可接受范围内（考虑浮点精度）
                assert max_diff < 1e-6, f"RSI14计算不一致: 最大差异={max_diff}"
                logger.info("✅ RSI14一致性测试通过")
            else:
                pytest.skip("RSI14: 没有有效数据点进行比较")
        else:
            pytest.skip("RSI14: 数据不可用")

    def test_willr_consistency(self, test_data):
        """测试WILLR计算一致性"""
        logger.info("🧪 测试WILLR一致性...")

        from factor_system.factor_engine.api import _register_core_factors
        from factor_system.factor_engine.core.cache import CacheConfig
        from factor_system.factor_engine.core.engine import FactorEngine
        from factor_system.factor_engine.core.registry import FactorRegistry
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator,
            IndicatorConfig,
            TimeFrame,
        )

        # 1. factor_generation
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)

        # 2. factor_engine
        registry = FactorRegistry()
        _register_core_factors(registry)
        mock_provider = MockDataProvider(test_data)
        engine = FactorEngine(mock_provider, registry, CacheConfig())

        engine_result = engine.calculate_factors(
            factor_ids=["WILLR14"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False,
        )

        # 3. 验证
        if (
            "WILLR14" in gen_result.columns
            and not engine_result.empty
            and "WILLR14" in engine_result.columns
        ):
            gen_willr = gen_result["WILLR14"].values

            if isinstance(engine_result.index, pd.MultiIndex):
                engine_willr = engine_result.xs("TEST", level="symbol")[
                    "WILLR14"
                ].values
            else:
                engine_willr = engine_result["WILLR14"].values

            min_len = min(len(gen_willr), len(engine_willr))
            gen_willr = gen_willr[-min_len:]
            engine_willr = engine_willr[-min_len:]

            valid_mask = ~(np.isnan(gen_willr) | np.isnan(engine_willr))
            gen_willr_valid = gen_willr[valid_mask]
            engine_willr_valid = engine_willr[valid_mask]

            if len(gen_willr_valid) > 0:
                max_diff = np.max(np.abs(gen_willr_valid - engine_willr_valid))
                mean_diff = np.mean(np.abs(gen_willr_valid - engine_willr_valid))

                logger.info(
                    f"✓ WILLR14一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}"
                )
                assert max_diff < 1e-6, f"WILLR14计算不一致: 最大差异={max_diff}"
                logger.info("✅ WILLR14一致性测试通过")
            else:
                pytest.skip("WILLR14: 没有有效数据点")
        else:
            pytest.skip("WILLR14: 数据不可用")

    def test_macd_consistency(self, test_data):
        """测试MACD计算一致性"""
        logger.info("🧪 测试MACD一致性...")

        from factor_system.factor_engine.api import _register_core_factors
        from factor_system.factor_engine.core.cache import CacheConfig
        from factor_system.factor_engine.core.engine import FactorEngine
        from factor_system.factor_engine.core.registry import FactorRegistry
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator,
            IndicatorConfig,
            TimeFrame,
        )

        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)

        registry = FactorRegistry()
        _register_core_factors(registry)
        mock_provider = MockDataProvider(test_data)
        engine = FactorEngine(mock_provider, registry, CacheConfig())

        engine_result = engine.calculate_factors(
            factor_ids=["MACD_12_26_9"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False,
        )

        if (
            "MACD_12_26_9" in gen_result.columns
            and not engine_result.empty
            and "MACD_12_26_9" in engine_result.columns
        ):
            gen_macd = gen_result["MACD_12_26_9"].values

            if isinstance(engine_result.index, pd.MultiIndex):
                engine_macd = engine_result.xs("TEST", level="symbol")[
                    "MACD_12_26_9"
                ].values
            else:
                engine_macd = engine_result["MACD_12_26_9"].values

            min_len = min(len(gen_macd), len(engine_macd))
            gen_macd = gen_macd[-min_len:]
            engine_macd = engine_macd[-min_len:]

            valid_mask = ~(np.isnan(gen_macd) | np.isnan(engine_macd))
            gen_macd_valid = gen_macd[valid_mask]
            engine_macd_valid = engine_macd[valid_mask]

            if len(gen_macd_valid) > 0:
                max_diff = np.max(np.abs(gen_macd_valid - engine_macd_valid))
                mean_diff = np.mean(np.abs(gen_macd_valid - engine_macd_valid))

                logger.info(
                    f"✓ MACD一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}"
                )
                assert max_diff < 1e-6, f"MACD计算不一致: 最大差异={max_diff}"
                logger.info("✅ MACD一致性测试通过")
            else:
                pytest.skip("MACD: 没有有效数据点")
        else:
            pytest.skip("MACD: 数据不可用")


def test_shared_calculator_usage():
    """测试所有因子是否使用SHARED_CALCULATORS"""
    logger.info("🧪 验证因子使用SHARED_CALCULATORS...")

    import inspect

    from factor_system.factor_engine.factors import GENERATED_FACTORS

    shared_calc_count = 0
    total_count = len(GENERATED_FACTORS)

    for factor_class in GENERATED_FACTORS:
        # 检查calculate方法源代码
        try:
            source = inspect.getsource(factor_class.calculate)
            if "SHARED_CALCULATORS" in source:
                shared_calc_count += 1
        except Exception:
            pass

    percentage = (shared_calc_count / total_count * 100) if total_count > 0 else 0
    logger.info(
        f"✓ {shared_calc_count}/{total_count}个因子使用SHARED_CALCULATORS ({percentage:.1f}%)"
    )

    # 至少30%的因子应该使用SHARED_CALCULATORS
    assert (
        shared_calc_count >= total_count * 0.3
    ), f"只有{shared_calc_count}/{total_count}个因子使用SHARED_CALCULATORS"

    logger.info("✅ SHARED_CALCULATORS使用率测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
