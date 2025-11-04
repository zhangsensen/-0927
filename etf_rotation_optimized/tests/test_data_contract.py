"""
数据契约单元测试

测试核心功能:
1. T-1对齐逻辑
2. OHLCV验证
3. 因子验证

作者: Linus Test
日期: 2025-10-28
"""

import numpy as np
import pandas as pd
import pytest
from core.data_contract import DataContract, align_factor_to_return


class TestAlignFactorToReturn:
    """测试T-1对齐函数"""

    def test_basic_alignment(self):
        """测试基本对齐逻辑"""
        factors = np.random.randn(100, 43, 18)
        returns = np.random.randn(100, 43)

        factors_aligned, returns_aligned = align_factor_to_return(factors, returns)

        # 验证形状
        assert factors_aligned.shape == (99, 43, 18)
        assert returns_aligned.shape == (99, 43)

        # 验证对齐：因子[0] 对应 收益[1]
        assert np.array_equal(factors_aligned[0], factors[0])
        assert np.array_equal(returns_aligned[0], returns[1])

    def test_length_mismatch(self):
        """测试长度不匹配"""
        factors = np.random.randn(100, 43, 18)
        returns = np.random.randn(90, 43)

        with pytest.raises(ValueError, match="因子时间维度"):
            align_factor_to_return(factors, returns)

    def test_too_short(self):
        """测试数据太短"""
        factors = np.random.randn(1, 43, 18)
        returns = np.random.randn(1, 43)

        with pytest.raises(ValueError, match="数据长度.*< 2"):
            align_factor_to_return(factors, returns)

    def test_dimension_mismatch(self):
        """测试维度不匹配"""
        factors = np.random.randn(100, 43, 18)
        returns = np.random.randn(100, 50)

        with pytest.raises(ValueError, match="标的维度"):
            align_factor_to_return(factors, returns)


class TestDataContractOHLCV:
    """测试OHLCV验证"""

    def test_valid_ohlcv(self):
        """测试有效OHLCV数据"""
        ohlcv = {
            "open": pd.DataFrame(np.random.randn(100, 43)),
            "high": pd.DataFrame(np.random.randn(100, 43)),
            "low": pd.DataFrame(np.random.randn(100, 43)),
            "close": pd.DataFrame(np.random.randn(100, 43)),
            "volume": pd.DataFrame(np.abs(np.random.randn(100, 43))),
        }

        # 不应抛出异常
        DataContract.validate_ohlcv(ohlcv)

    def test_missing_column(self):
        """测试缺失列"""
        ohlcv = {
            "open": pd.DataFrame(np.random.randn(100, 43)),
            "high": pd.DataFrame(np.random.randn(100, 43)),
            "low": pd.DataFrame(np.random.randn(100, 43)),
            # 缺少 close
            "volume": pd.DataFrame(np.abs(np.random.randn(100, 43))),
        }

        with pytest.raises(ValueError, match="缺失必需列: close"):
            DataContract.validate_ohlcv(ohlcv)

    def test_too_many_nans(self):
        """测试NaN过多"""
        ohlcv = {
            "open": pd.DataFrame(np.random.randn(100, 43)),
            "high": pd.DataFrame(np.random.randn(100, 43)),
            "low": pd.DataFrame(np.random.randn(100, 43)),
            "close": pd.DataFrame(np.full((100, 43), np.nan)),  # 全NaN
            "volume": pd.DataFrame(np.abs(np.random.randn(100, 43))),
        }

        with pytest.raises(ValueError, match="close NaN率"):
            DataContract.validate_ohlcv(ohlcv)

    def test_too_few_days(self):
        """测试交易日数太少"""
        ohlcv = {
            "open": pd.DataFrame(np.random.randn(50, 43)),  # 只有50天
            "high": pd.DataFrame(np.random.randn(50, 43)),
            "low": pd.DataFrame(np.random.randn(50, 43)),
            "close": pd.DataFrame(np.random.randn(50, 43)),
            "volume": pd.DataFrame(np.abs(np.random.randn(50, 43))),
        }

        with pytest.raises(ValueError, match="交易日数"):
            DataContract.validate_ohlcv(ohlcv)


class TestDataContractFactor:
    """测试因子验证"""

    def test_valid_factor(self):
        """测试有效因子"""
        factor = pd.DataFrame(np.random.randn(100, 43))

        # 不应抛出异常
        DataContract.validate_factor(factor, "TEST_FACTOR")

    def test_factor_too_many_nans(self):
        """测试因子NaN过多"""
        factor = pd.DataFrame(np.full((100, 43), np.nan))

        with pytest.raises(ValueError, match="TEST_FACTOR NaN率"):
            DataContract.validate_factor(factor, "TEST_FACTOR")

    def test_custom_nan_threshold(self):
        """测试自定义NaN阈值"""
        # 50% NaN
        data = np.random.randn(100, 43)
        data[:50, :] = np.nan
        factor = pd.DataFrame(data)

        # 默认阈值30%应该失败
        with pytest.raises(ValueError):
            DataContract.validate_factor(factor, "TEST_FACTOR")

        # 自定义阈值60%应该通过
        DataContract.validate_factor(factor, "TEST_FACTOR", max_nan_ratio=0.6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
