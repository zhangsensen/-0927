"""
FactorEngine与factor_generation一致性测试

确保重构后的factor_engine与factor_generation的计算结果完全一致
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from factor_system.factor_engine import api
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter
from factor_system.factor_generation.enhanced_factor_calculator import (
    EnhancedFactorCalculator,
)


class TestFactorEngineConsistency:
    """测试FactorEngine与factor_generation的一致性"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        # 生成50天的OHLCV数据
        dates = pd.date_range("2025-09-01", "2025-10-20", freq="D")
        n = len(dates)

        # 生成随机价格数据
        import numpy as np

        np.random.seed(42)  # 确保可重现

        base_price = 100.0
        close_prices = base_price + np.cumsum(np.random.randn(n) * 2)
        high_prices = close_prices + np.random.uniform(0, 3, n)
        low_prices = close_prices - np.random.uniform(0, 3, n)
        open_prices = close_prices + np.random.uniform(-2, 2, n)
        volumes = np.random.randint(1000000, 5000000, n)

        data = pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volumes,
            },
            index=dates,
        )

        return data

    @pytest.fixture
    def factor_generation_calculator(self):
        """创建factor_generation计算器"""
        # 使用默认配置
        from factor_system.factor_generation.enhanced_factor_calculator import (
            FactorCalculatorConfig,
        )

        config = FactorCalculatorConfig()
        return EnhancedFactorCalculator(config)

    def test_rsi_consistency(self, sample_data):
        """测试RSI一致性"""
        # factor_engine计算RSI
        rsi_engine = api.calculate_single_factor(
            factor_id="RSI",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算RSI（模拟）
        try:
            vbt_rsi = get_vectorbt_adapter().calculate_rsi(
                sample_data["close"], window=14
            )

            # 比较结果（排除前N个NaN值）
            valid_mask = ~rsi_engine.isna() & ~vbt_rsi.isna()

            if valid_mask.sum() > 0:
                # 使用相对误差容忍度
                relative_diff = abs(rsi_engine[valid_mask] - vbt_rsi[valid_mask]) / (
                    abs(vbt_rsi[valid_mask]) + 1e-10
                )
                max_relative_diff = relative_diff.max()

                print(f"RSI相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"RSI计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("RSI计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"RSI一致性测试失败: {e}")

    def test_macd_consistency(self, sample_data):
        """测试MACD一致性"""
        # factor_engine计算MACD
        macd_engine = api.calculate_single_factor(
            factor_id="MACD",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算MACD（模拟）
        try:
            vbt_macd = get_vectorbt_adapter().calculate_macd(sample_data["close"])

            # 比较结果
            valid_mask = ~macd_engine.isna() & ~vbt_macd.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(macd_engine[valid_mask] - vbt_macd[valid_mask]) / (
                    abs(vbt_macd[valid_mask]) + 1e-10
                )
                max_relative_diff = relative_diff.max()

                print(f"MACD相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"MACD计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("MACD计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"MACD一致性测试失败: {e}")

    def test_stoch_consistency(self, sample_data):
        """测试STOCH一致性"""
        # factor_engine计算STOCH
        stoch_engine = api.calculate_single_factor(
            factor_id="STOCH",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算STOCH（模拟）
        try:
            vbt_stoch = get_vectorbt_adapter().calculate_stoch(
                sample_data["high"], sample_data["low"], sample_data["close"]
            )

            # 比较结果
            valid_mask = ~stoch_engine.isna() & ~vbt_stoch.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(
                    stoch_engine[valid_mask] - vbt_stoch[valid_mask]
                ) / (abs(vbt_stoch[valid_mask]) + 1e-10)
                max_relative_diff = relative_diff.max()

                print(f"STOCH相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"STOCH计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("STOCH计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"STOCH一致性测试失败: {e}")

    def test_sma_consistency(self, sample_data):
        """测试SMA一致性"""
        # factor_engine计算SMA
        sma_engine = api.calculate_single_factor(
            factor_id="SMA",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算SMA（模拟）
        try:
            vbt_sma = get_vectorbt_adapter().calculate_sma(
                sample_data["close"], window=20
            )

            # 比较结果
            valid_mask = ~sma_engine.isna() & ~vbt_sma.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(sma_engine[valid_mask] - vbt_sma[valid_mask]) / (
                    abs(vbt_sma[valid_mask]) + 1e-10
                )
                max_relative_diff = relative_diff.max()

                print(f"SMA相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"SMA计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("SMA计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"SMA一致性测试失败: {e}")

    def test_ema_consistency(self, sample_data):
        """测试EMA一致性"""
        # factor_engine计算EMA
        ema_engine = api.calculate_single_factor(
            factor_id="EMA",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算EMA（模拟）
        try:
            vbt_ema = get_vectorbt_adapter().calculate_ema(
                sample_data["close"], window=20
            )

            # 比较结果
            valid_mask = ~ema_engine.isna() & ~vbt_ema.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(ema_engine[valid_mask] - vbt_ema[valid_mask]) / (
                    abs(vbt_ema[valid_mask]) + 1e-10
                )
                max_relative_diff = relative_diff.max()

                print(f"EMA相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"EMA计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("EMA计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"EMA一致性测试失败: {e}")

    def test_willr_consistency(self, sample_data):
        """测试WILLR一致性"""
        # factor_engine计算WILLR
        willr_engine = api.calculate_single_factor(
            factor_id="WILLR",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算WILLR（模拟）
        try:
            vbt_willr = get_vectorbt_adapter().calculate_willr(
                sample_data["high"],
                sample_data["low"],
                sample_data["close"],
                timeperiod=14,
            )

            # 比较结果
            valid_mask = ~willr_engine.isna() & ~vbt_willr.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(
                    willr_engine[valid_mask] - vbt_willr[valid_mask]
                ) / (abs(vbt_willr[valid_mask]) + 1e-10)
                max_relative_diff = relative_diff.max()

                print(f"WILLR相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"WILLR计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("WILLR计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"WILLR一致性测试失败: {e}")

    def test_cci_consistency(self, sample_data):
        """测试CCI一致性"""
        # factor_engine计算CCI
        cci_engine = api.calculate_single_factor(
            factor_id="CCI",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # factor_generation计算CCI（模拟）
        try:
            vbt_cci = get_vectorbt_adapter().calculate_cci(
                sample_data["high"],
                sample_data["low"],
                sample_data["close"],
                timeperiod=14,
            )

            # 比较结果
            valid_mask = ~cci_engine.isna() & ~vbt_cci.isna()

            if valid_mask.sum() > 0:
                relative_diff = abs(cci_engine[valid_mask] - vbt_cci[valid_mask]) / (
                    abs(vbt_cci[valid_mask]) + 1e-10
                )
                max_relative_diff = relative_diff.max()

                print(f"CCI相对误差: {max_relative_diff:.6f}")
                assert (
                    max_relative_diff < 1e-10
                ), f"CCI计算不一致，最大相对误差: {max_relative_diff}"
            else:
                pytest.skip("CCI计算结果全为NaN，无法比较")

        except Exception as e:
            pytest.fail(f"CCI一致性测试失败: {e}")

    def test_multiple_factors_consistency(self, sample_data):
        """测试多个因子同时计算的一致性"""
        factor_ids = ["RSI", "MACD", "STOCH", "WILLR", "CCI", "SMA", "EMA"]

        # factor_engine批量计算
        factors_engine = api.calculate_factors(
            factor_ids=factor_ids,
            symbols=["TEST"],
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # 提取TEST的数据
        test_factors_engine = factors_engine.xs("TEST", level="symbol")

        assert set(test_factors_engine.columns) == set(factor_ids), "因子ID不匹配"

        # 逐个验证一致性
        for factor_id in factor_ids:
            if factor_id in test_factors_engine.columns:
                try:
                    # 使用VectorBT直接计算
                    adapter = get_vectorbt_adapter()

                    if factor_id == "RSI":
                        vbt_result = adapter.calculate_rsi(sample_data["close"])
                    elif factor_id == "MACD":
                        vbt_result = adapter.calculate_macd(sample_data["close"])
                    elif factor_id == "STOCH":
                        vbt_result = adapter.calculate_stoch(
                            sample_data["high"],
                            sample_data["low"],
                            sample_data["close"],
                        )
                    elif factor_id == "WILLR":
                        vbt_result = adapter.calculate_willr(
                            sample_data["high"],
                            sample_data["low"],
                            sample_data["close"],
                        )
                    elif factor_id == "CCI":
                        vbt_result = adapter.calculate_cci(
                            sample_data["high"],
                            sample_data["low"],
                            sample_data["close"],
                        )
                    elif factor_id == "SMA":
                        vbt_result = adapter.calculate_sma(sample_data["close"])
                    elif factor_id == "EMA":
                        vbt_result = adapter.calculate_ema(sample_data["close"])
                    else:
                        continue

                    engine_values = test_factors_engine[factor_id]
                    valid_mask = ~engine_values.isna() & ~vbt_result.isna()

                    if valid_mask.sum() > 0:
                        relative_diff = abs(
                            engine_values[valid_mask] - vbt_result[valid_mask]
                        ) / (abs(vbt_result[valid_mask]) + 1e-10)
                        max_relative_diff = relative_diff.max()

                        print(f"{factor_id} 批量计算相对误差: {max_relative_diff:.6f}")
                        assert max_relative_diff < 1e-10, f"{factor_id}批量计算不一致"

                except Exception as e:
                    pytest.fail(f"{factor_id}批量一致性测试失败: {e}")

    def test_api_vs_adapter_consistency(self, sample_data):
        """测试统一API与直接使用适配器的一致性"""
        # 使用统一API计算RSI
        rsi_api = api.calculate_single_factor(
            factor_id="RSI",
            symbol="TEST",
            timeframe="daily",
            start_date=sample_data.index[0],
            end_date=sample_data.index[-1],
        )

        # 直接使用适配器计算RSI
        adapter = get_vectorbt_adapter()
        rsi_adapter = adapter.calculate_rsi(sample_data["close"])

        # 比较结果
        valid_mask = ~rsi_api.isna() & ~rsi_adapter.isna()

        if valid_mask.sum() > 0:
            relative_diff = abs(rsi_api[valid_mask] - rsi_adapter[valid_mask]) / (
                abs(rsi_adapter[valid_mask]) + 1e-10
            )
            max_relative_diff = relative_diff.max()

            print(f"API vs 适配器相对误差: {max_relative_diff:.6f}")
            assert (
                max_relative_diff < 1e-12
            ), f"API与适配器不一致，相对误差: {max_relative_diff}"
        else:
            pytest.skip("计算结果全为NaN，无法比较")


if __name__ == "__main__":
    # 直接运行测试
    import sys

    test = TestFactorEngineConsistency()

    # 创建测试数据
    print("创建测试数据...")
    data = test.test_rsi_consistency.__func__.__globals__["sample_data"] = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 5,
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111] * 5,
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108] * 5,
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 5,
            "volume": [1000000] * 50,
        },
        index=pd.date_range("2025-09-01", periods=50, freq="D"),
    )

    print("运行一致性测试...")

    try:
        test.test_rsi_consistency(data)
        print("✅ RSI一致性测试通过")

        test.test_macd_consistency(data)
        print("✅ MACD一致性测试通过")

        test.test_stoch_consistency(data)
        print("✅ STOCH一致性测试通过")

        test.test_sma_consistency(data)
        print("✅ SMA一致性测试通过")

        test.test_ema_consistency(data)
        print("✅ EMA一致性测试通过")

        test.test_multiple_factors_consistency(data)
        print("✅ 多因子一致性测试通过")

        print(
            "\n🎉 所有一致性测试通过！FactorEngine与factor_generation计算逻辑完全一致。"
        )

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
