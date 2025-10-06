#!/usr/bin/env python3
"""
未来函数防护测试套件
作者：量化首席工程师
版本：1.0.0
日期：2025-10-02

功能：
- 测试静态分析工具的检测能力
- 验证运行时验证机制
- 确保架构层防护有效
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# 添加项目路径
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "utils"))
sys.path.insert(0, os.path.join(parent_dir, "scripts"))

from scripts.check_future_functions import check_file_for_future_functions  # noqa: E402
from utils.temporal_validator import (  # noqa: E402
    TemporalValidator,
    validate_factor_return_alignment,
)
from utils.time_series_protocols import (  # noqa: E402
    SafeTimeSeriesProcessor,
    safe_ic_calculation,
)


class TestStaticAnalysis:
    """静态分析工具测试"""

    def test_check_future_functions_with_negative_shift(self):
        """测试检测负数shift"""
        test_code = """
import pandas as pd

def bad_function(data):
    return data.shift(-1)  # 未来函数
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            issues = check_file_for_future_functions(temp_file)

            assert len(issues) > 0
            assert any("shift(-1)" in issue.get("code", "") for issue in issues)

        finally:
            os.unlink(temp_file)

    def test_check_future_functions_with_future_variables(self):
        """测试检测future变量"""
        test_code = """
def calculate_signals(data):
    future_price = data['close'].shift(-5)  # 未来变量
    current_signal = data['volume'] > 1000
    return current_signal
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            issues = check_file_for_future_functions(temp_file)

            assert len(issues) > 0
            assert any("future_" in issue.get("code", "") for issue in issues)

        finally:
            os.unlink(temp_file)


class TestTemporalValidator:
    """时间验证器测试"""

    def setup_method(self):
        """设置测试数据"""
        self.validator = TemporalValidator(strict_mode=True)

        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.factor_data = pd.Series(np.random.randn(100), index=dates)
        self.return_data = pd.Series(np.random.randn(100) * 0.01, index=dates)

    def test_validate_time_alignment_success(self):
        """测试时间对齐验证成功"""
        is_valid, message = self.validator.validate_time_alignment(
            self.factor_data, self.return_data, horizon=5, context="test"
        )

        assert is_valid
        assert "验证通过" in message

    def test_validate_time_alignment_insufficient_data(self):
        """测试数据不足的情况"""
        short_factor = self.factor_data.iloc[:10]  # 只有10个数据点

        is_valid, message = self.validator.validate_time_alignment(
            short_factor, self.return_data, horizon=10, context="test"
        )

        assert not is_valid
        assert "对齐数据不足" in message

    def test_validate_ic_calculation(self):
        """测试IC计算验证"""
        horizons = [1, 3, 5, 10]
        results = self.validator.validate_ic_calculation(
            self.factor_data, self.return_data, horizons, context="test"
        )

        assert len(results) == len(horizons)
        for horizon in horizons:
            assert horizon in results
            assert "ic" in results[horizon]
            assert "sample_size" in results[horizon]
            assert "is_valid" in results[horizon]

    def test_validate_no_future_data_success(self):
        """测试无未来数据验证成功"""
        data = pd.DataFrame(
            {
                "price": np.random.randn(100) + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "momentum": np.random.randn(100),
            },
            index=pd.date_range("2023-01-01", periods=100),
        )

        result = self.validator.validate_no_future_data(data, context="test")
        assert result


class TestSafeTimeSeriesProcessor:
    """安全时间序列处理器测试"""

    def setup_method(self):
        """设置测试数据"""
        self.processor = SafeTimeSeriesProcessor(strict_mode=True)

        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.price_data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        self.factor_data = pd.Series(np.random.randn(100), index=dates)

    def test_calculate_ic_safe(self):
        """测试安全IC计算"""
        # 创建收益数据
        returns = self.price_data.pct_change().fillna(0)

        ic = self.processor.calculate_ic_safe(self.factor_data, returns, horizon=5)

        assert isinstance(ic, float)
        assert -1 <= ic <= 1

    def test_shift_forward_allowed(self):
        """测试允许向前shift"""
        shifted = self.processor.shift_forward(self.factor_data, periods=5)

        assert len(shifted) == len(self.factor_data)
        assert not shifted.equals(self.factor_data)

    def test_shift_backward_forbidden(self):
        """测试禁止向后shift"""
        with pytest.raises(NotImplementedError, match="向后shift.*被禁止"):
            self.processor.shift_backward(self.factor_data, periods=-5)

    def test_calculate_forward_returns(self):
        """测试计算前向收益"""
        horizons = [1, 5, 10]
        returns_df = self.processor.calculate_forward_returns(self.price_data, horizons)

        assert len(returns_df.columns) == len(horizons)
        for horizon in horizons:
            assert f"return_{horizon}d" in returns_df.columns

    def test_validate_no_future_leakage_success(self):
        """测试无未来泄露验证成功"""
        data = pd.DataFrame(
            {
                "close": self.price_data,
                "volume": np.random.randint(1000, 10000, 100),
                "rsi": np.random.random(100),
            },
            index=self.price_data.index,
        )

        result = self.processor.validate_no_future_leakage(data)
        assert result

    def test_validate_no_future_leakage_detection(self):
        """测试检测未来泄露"""
        data = pd.DataFrame(
            {
                "close": self.price_data,
                "future_return": np.random.randn(100),  # 包含future关键词
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=self.price_data.index,
        )

        with pytest.raises(ValueError, match="数据验证发现问题"):
            self.processor.validate_no_future_leakage(data)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def setup_method(self):
        """设置测试数据"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        self.factor_data = pd.Series(np.random.randn(50), index=dates)
        self.return_data = pd.Series(np.random.randn(50) * 0.01, index=dates)

    def test_validate_factor_return_alignment(self):
        """测试便捷对齐验证函数"""
        result = validate_factor_return_alignment(
            self.factor_data, self.return_data, horizon=3, context="test"
        )
        assert result

    def test_safe_ic_calculation(self):
        """测试便捷IC计算函数"""
        ic = safe_ic_calculation(self.factor_data, self.return_data, horizon=5)

        assert isinstance(ic, float)
        assert -1 <= ic <= 1


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_data_handling(self):
        """测试空数据处理"""
        processor = SafeTimeSeriesProcessor(strict_mode=False)

        empty_series = pd.Series(dtype=float)

        # 宽松模式下应该能处理空数据
        ic = processor.calculate_ic_safe(empty_series, empty_series, horizon=1)
        assert ic == 0.0

    def test_single_point_data(self):
        """测试单点数据处理"""
        processor = SafeTimeSeriesProcessor(strict_mode=False)

        single_point = pd.Series([1.0], index=[pd.Timestamp("2023-01-01")])

        # 单点数据应该返回0 IC
        ic = processor.calculate_ic_safe(single_point, single_point, horizon=1)
        assert ic == 0.0

    def test_all_nan_data(self):
        """测试全NaN数据处理"""
        processor = SafeTimeSeriesProcessor(strict_mode=False)

        nan_series = pd.Series(
            [np.nan] * 10, index=pd.date_range("2023-01-01", periods=10)
        )

        ic = processor.calculate_ic_safe(nan_series, nan_series, horizon=1)
        assert ic == 0.0


class TestPerformance:
    """性能测试"""

    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        import time

        # 创建大数据集
        dates = pd.date_range("2020-01-01", periods=10000, freq="D")
        factor_data = pd.Series(np.random.randn(10000), index=dates)
        return_data = pd.Series(np.random.randn(10000) * 0.01, index=dates)

        processor = SafeTimeSeriesProcessor(strict_mode=True)

        start_time = time.time()

        # 执行IC计算
        ic = processor.calculate_ic_safe(factor_data, return_data, horizon=5)

        end_time = time.time()
        execution_time = end_time - start_time

        # 验证性能（应该在合理时间内完成）
        assert execution_time < 1.0  # 应该在1秒内完成
        assert isinstance(ic, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
