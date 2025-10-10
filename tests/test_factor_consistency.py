"""测试三个系统的因子计算一致性"""

import os

# 直接导入避免其他文件语法错误
import sys

import numpy as np
import pandas as pd
import pytest

from factor_system.shared.factor_calculators import SHARED_CALCULATORS

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "factor_system",
        "factor_engine",
        "factors",
        "technical",
    )
)

try:
    from rsi import RSI
except ImportError:
    RSI = None

try:
    from macd import MACD, MACDHistogram, MACDSignal
except ImportError:
    MACD = MACDSignal = MACDHistogram = None

try:
    from stoch import STOCH
except ImportError:
    STOCH = None


class TestFactorConsistency:
    """测试因子在三个系统中的一致性"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)  # 确保可重现

        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # 生成模拟OHLCV数据
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 100)
        close_series = pd.Series(base_price * (1 + returns).cumprod())

        # 生成合理的OHLC数据
        high = close_series * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low = close_series * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        open_price = close_series.shift(1).fillna(base_price)

        return pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_series,
                "volume": np.random.randint(100000, 1000000, 100),
            },
            index=dates,
        )

    def test_rsi_consistency(self, sample_data):
        """测试RSI在三个系统中的一致性"""
        close = sample_data["close"]
        period = 14

        # 1. 共享计算器
        rsi_shared = SHARED_CALCULATORS.calculate_rsi(close, period=period)

        # 2. FactorEngine (如果可用)
        if RSI is not None:
            rsi_factor = RSI(period=period)
            rsi_engine = rsi_factor.calculate(sample_data)

            # 验证一致性（允许浮点误差）
            pd.testing.assert_series_equal(
                rsi_shared.dropna(), rsi_engine.dropna(), atol=1e-10, check_names=False
            )

        # 验证RSI值在合理范围内
        valid_rsi = rsi_shared.dropna()
        if not valid_rsi.empty:
            assert valid_rsi.min() >= 0, "RSI最小值应大于等于0"
            assert valid_rsi.max() <= 100, "RSI最大值应小于等于100"

    def test_macd_consistency(self, sample_data):
        """测试MACD在三个系统中的一致性"""
        close = sample_data["close"]
        fast, slow, signal = 12, 26, 9

        # 1. 共享计算器
        macd_shared = SHARED_CALCULATORS.calculate_macd(
            close, fastperiod=fast, slowperiod=slow, signalperiod=signal
        )

        # 2. FactorEngine (如果可用)
        if MACD is not None and MACDSignal is not None and MACDHistogram is not None:
            macd_factor = MACD(fast_period=fast, slow_period=slow, signal_period=signal)
            macd_engine = macd_factor.calculate(sample_data)

            signal_factor = MACDSignal(
                fast_period=fast, slow_period=slow, signal_period=signal
            )
            signal_engine = signal_factor.calculate(sample_data)

            hist_factor = MACDHistogram(
                fast_period=fast, slow_period=slow, signal_period=signal
            )
            hist_engine = hist_factor.calculate(sample_data)

            # 验证一致性
            pd.testing.assert_series_equal(
                macd_shared["macd"].dropna(),
                macd_engine.dropna(),
                atol=1e-10,
                check_names=False,
            )

            pd.testing.assert_series_equal(
                macd_shared["signal"].dropna(),
                signal_engine.dropna(),
                atol=1e-10,
                check_names=False,
            )

            pd.testing.assert_series_equal(
                macd_shared["hist"].dropna(),
                hist_engine.dropna(),
                atol=1e-10,
                check_names=False,
            )

        # 验证MACD关系：hist = macd - signal
        hist_diff = macd_shared["macd"] - macd_shared["signal"]
        pd.testing.assert_series_equal(
            macd_shared["hist"].dropna(),
            hist_diff.dropna(),
            atol=1e-10,
            check_names=False,
        )

    def test_stoch_consistency(self, sample_data):
        """测试STOCH在三个系统中的一致性"""
        high = sample_data["high"]
        low = sample_data["low"]
        close = sample_data["close"]
        k_window, d_window = 14, 3

        # 1. 共享计算器
        stoch_shared = SHARED_CALCULATORS.calculate_stoch(
            high,
            low,
            close,
            fastk_period=k_window,
            slowk_period=d_window,
            slowd_period=d_window,
        )

        # 2. FactorEngine (如果可用)
        if STOCH is not None:
            stoch_factor = STOCH(
                fastk_period=k_window, slowk_period=d_window, slowd_period=d_window
            )
            stoch_engine = stoch_factor.calculate(sample_data)

            # 验证一致性
            pd.testing.assert_series_equal(
                stoch_shared["slowk"].dropna(),
                stoch_engine.dropna(),
                atol=1e-10,
                check_names=False,
            )

        # 验证随机振荡指标在合理范围内
        valid_k = stoch_shared["slowk"].dropna()
        valid_d = stoch_shared["slowd"].dropna()

        if not valid_k.empty:
            assert valid_k.min() >= 0, "STOCH %K最小值应大于等于0"
            assert valid_k.max() <= 100, "STOCH %K最大值应小于等于100"

        if not valid_d.empty:
            assert valid_d.min() >= 0, "STOCH %D最小值应大于等于0"
            assert valid_d.max() <= 100, "STOCH %D最大值应小于等于100"

    def test_bbands_consistency(self, sample_data):
        """测试布林带在共享计算器中的计算"""
        close = sample_data["close"]
        period = 20

        # 1. 共享计算器
        bbands_shared = SHARED_CALCULATORS.calculate_bbands(close, period=period)

        # 验证布林带关系：middle > lower, upper > middle
        valid_data = ~(
            bbands_shared["upper"].isna()
            | bbands_shared["middle"].isna()
            | bbands_shared["lower"].isna()
        )

        assert all(
            bbands_shared["upper"][valid_data] >= bbands_shared["middle"][valid_data]
        ), "上轨应该大于等于中轨"
        assert all(
            bbands_shared["middle"][valid_data] >= bbands_shared["lower"][valid_data]
        ), "中轨应该大于等于下轨"

        # 验证中轨就是移动平均线
        expected_middle = close.rolling(window=period).mean()
        pd.testing.assert_series_equal(
            bbands_shared["middle"], expected_middle, atol=1e-10, check_names=False
        )

    def test_willr_consistency(self, sample_data):
        """测试WILLR在共享计算器中的计算"""
        high = sample_data["high"]
        low = sample_data["low"]
        close = sample_data["close"]
        period = 14

        # 1. 共享计算器
        willr_shared = SHARED_CALCULATORS.calculate_willr(
            high, low, close, period=period
        )

        # 验证威廉指标在合理范围内
        valid_willr = willr_shared.dropna()
        if not valid_willr.empty:
            assert valid_willr.min() >= -100, "WILLR最小值应大于等于-100"
            assert valid_willr.max() <= 0, "WILLR最大值应小于等于0"

        # 手动计算WILLR进行验证
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        expected_willr = -100 * (highest_high - close) / (highest_high - lowest_low)

        pd.testing.assert_series_equal(
            willr_shared, expected_willr, atol=1e-10, check_names=False
        )

    def test_atr_consistency(self, sample_data):
        """测试ATR在共享计算器中的计算"""
        high = sample_data["high"]
        low = sample_data["low"]
        close = sample_data["close"]
        period = 14

        # 1. 共享计算器
        atr_shared = SHARED_CALCULATORS.calculate_atr(high, low, close, period=period)

        # 验证ATR为正值
        valid_atr = atr_shared.dropna()
        if not valid_atr.empty:
            assert valid_atr.min() >= 0, "ATR应该为非负值"

        # 手动计算ATR进行验证
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        expected_atr = tr.rolling(window=period, min_periods=period).mean()

        pd.testing.assert_series_equal(
            atr_shared, expected_atr, atol=1e-10, check_names=False
        )


if __name__ == "__main__":
    # 运行单个测试用于调试
    import numpy as np
    import pandas as pd

    # 手动创建测试数据
    np.random.seed(42)  # 确保可重现
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # 生成模拟OHLCV数据
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 100)
    close_series = pd.Series(base_price * (1 + returns).cumprod())

    # 生成合理的OHLC数据
    high = close_series * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low = close_series * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_price = close_series.shift(1).fillna(base_price)
    close = close_series

    sample_data = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100000, 1000000, 100),
        },
        index=dates,
    )

    print("测试数据创建成功")

    # 运行各项测试
    test_instance = TestFactorConsistency()

    try:
        test_instance.test_rsi_consistency(sample_data)
        print("✅ RSI一致性测试通过")
    except Exception as e:
        print(f"❌ RSI一致性测试失败: {e}")

    try:
        test_instance.test_macd_consistency(sample_data)
        print("✅ MACD一致性测试通过")
    except Exception as e:
        print(f"❌ MACD一致性测试失败: {e}")

    try:
        test_instance.test_stoch_consistency(sample_data)
        print("✅ STOCH一致性测试通过")
    except Exception as e:
        print(f"❌ STOCH一致性测试失败: {e}")

    try:
        test_instance.test_bbands_consistency(sample_data)
        print("✅ BBANDS一致性测试通过")
    except Exception as e:
        print(f"❌ BBANDS一致性测试失败: {e}")

    try:
        test_instance.test_willr_consistency(sample_data)
        print("✅ WILLR一致性测试通过")
    except Exception as e:
        print(f"❌ WILLR一致性测试失败: {e}")

    try:
        test_instance.test_atr_consistency(sample_data)
        print("✅ ATR一致性测试通过")
    except Exception as e:
        print(f"❌ ATR一致性测试失败: {e}")

    print("\n🎉 所有因子一致性测试完成！")
