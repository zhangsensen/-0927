"""
数据验证器单元测试
================================================================================
测试场景：
1. 满窗检查（正常/异常）
2. 覆盖率验证
3. 日期验证
4. 成交量异常检测
5. Amount列添加
================================================================================
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):
    """数据验证器测试套件"""

    def setUp(self):
        """测试前设置"""
        self.validator = DataValidator(verbose=False)
        self.dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # 创建测试数据
        self.valid_prices = {
            "close": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 100,
                    "SH600001": np.random.randn(100).cumsum() + 100,
                },
                index=self.dates,
            ),
            "high": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 102,
                    "SH600001": np.random.randn(100).cumsum() + 102,
                },
                index=self.dates,
            ),
            "low": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 98,
                    "SH600001": np.random.randn(100).cumsum() + 98,
                },
                index=self.dates,
            ),
            "volume": pd.DataFrame(
                {
                    "SH600000": np.random.randint(100000, 500000, 100),
                    "SH600001": np.random.randint(100000, 500000, 100),
                },
                index=self.dates,
            ),
        }

    def test_check_full_window_pass(self):
        """测试：满窗检查通过"""
        result = self.validator.check_full_window(self.valid_prices, window_days=20)
        self.assertTrue(result)

    def test_check_full_window_pass_with_occasional_nan(self):
        """测试：满窗检查通过（偶发NaN<window_days）- 新规则"""
        # 根据精确定义规则：偶发缺失应该被允许
        prices = self.valid_prices.copy()
        # 添加5个非连续的NaN（不连续，或少于window_days）
        prices["close"].iloc[10] = np.nan
        prices["close"].iloc[20] = np.nan
        prices["close"].iloc[30] = np.nan
        prices["close"].iloc[40] = np.nan
        prices["close"].iloc[50] = np.nan

        # 应该通过（因为没有连续的NaN≥20）
        result = self.validator.check_full_window(prices, window_days=20)
        self.assertTrue(result)

    def test_check_full_window_fail(self):
        """测试：满窗检查失败（连续NaN≥window_days）"""
        # 根据新规则：只有连续NaN≥window_days时才失败
        prices = self.valid_prices.copy()
        # 添加连续20个NaN（等于window_days）
        prices["close"].iloc[10:30] = np.nan

        result = self.validator.check_full_window(prices, window_days=20)
        self.assertFalse(result)

    def test_check_coverage_high(self):
        """测试：覆盖率检查（高覆盖）"""
        result = self.validator.check_coverage(self.valid_prices, threshold=0.90)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["valid_symbols"]), 2)

    def test_check_coverage_low(self):
        """测试：覆盖率检查（低覆盖）"""
        prices = self.valid_prices.copy()
        # 添加大量NaN使覆盖率降低
        prices["close"].iloc[:50] = np.nan

        result = self.validator.check_coverage(prices, threshold=0.90)
        self.assertFalse(result["passed"])

    def test_coverage_stats(self):
        """测试：覆盖率统计"""
        result = self.validator.check_coverage(self.valid_prices)

        # 检查覆盖率字典
        self.assertIn("coverage_stats", result)
        self.assertIn("SH600000", result["coverage_stats"])

        # 检查覆盖率值
        for symbol, coverage in result["coverage_stats"].items():
            self.assertGreaterEqual(coverage, 0)
            self.assertLessEqual(coverage, 1)

    def test_add_amount_column_typical_price(self):
        """测试：添加Amount列（typical_price方法）"""
        result = self.validator.add_amount_column(
            self.valid_prices, fill_method="typical_price"
        )

        # 检查amount列是否存在
        self.assertIn("amount", result)

        # 检查金额计算是否合理
        typical_price = (result["high"] + result["low"] + result["close"]) / 3
        expected_amount = typical_price * result["volume"]

        pd.testing.assert_frame_equal(result["amount"], expected_amount)

    def test_add_amount_column_close(self):
        """测试：添加Amount列（close方法）"""
        result = self.validator.add_amount_column(
            self.valid_prices, fill_method="close"
        )

        self.assertIn("amount", result)
        expected_amount = self.valid_prices["close"] * self.valid_prices["volume"]

        pd.testing.assert_frame_equal(result["amount"], expected_amount)

    def test_add_amount_existing(self):
        """测试：amount列已存在"""
        prices = self.valid_prices.copy()
        prices["amount"] = prices["close"] * prices["volume"]

        result = self.validator.add_amount_column(prices)

        # 应该返回原始amount列
        pd.testing.assert_frame_equal(result["amount"], prices["amount"])

    def test_detect_volume_anomalies_none(self):
        """测试：成交量异常检测（无异常）"""
        result = self.validator.detect_volume_anomalies(
            self.valid_prices, window=20, z_score_threshold=3.0
        )

        self.assertIn("anomalies", result)
        # 正常随机数据不应该有异常
        self.assertLessEqual(result["anomaly_count"], 2)

    def test_detect_volume_anomalies_with_spike(self):
        """测试：成交量异常检测（有尖峰）"""
        prices = self.valid_prices.copy()
        # 添加成交量尖峰
        prices["volume"].iloc[50, 0] = 10000000  # 异常高值

        result = self.validator.detect_volume_anomalies(
            prices, window=20, z_score_threshold=2.0
        )

        # 应该检测到异常
        self.assertGreater(result["anomaly_count"], 0)

    def test_validate_dates(self):
        """测试：日期验证"""
        result = self.validator.validate_dates(self.valid_prices)

        self.assertTrue(result["passed"])
        self.assertEqual(result["total_days"], 100)
        self.assertEqual(result["min_date"], self.dates[0])
        self.assertEqual(result["max_date"], self.dates[-1])

    def test_full_validation_pass(self):
        """测试：完整验证通过"""
        result = self.validator.full_validation(
            self.valid_prices, window_days=20, coverage_threshold=0.90
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["full_window"])
        self.assertTrue(result["checks"]["coverage"])
        self.assertTrue(result["checks"]["dates"])

    def test_full_validation_fail_window(self):
        """测试：完整验证失败（连续NaN≥window_days）"""
        prices = self.valid_prices.copy()
        # 根据新规则，需要连续20个或以上的NaN才会失败
        prices["close"].iloc[10:30] = np.nan

        result = self.validator.full_validation(
            prices, window_days=20, coverage_threshold=0.90
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["full_window"])

    def test_full_validation_fail_coverage(self):
        """测试：完整验证失败（覆盖率）"""
        prices = self.valid_prices.copy()
        prices["close"].iloc[:50] = np.nan

        result = self.validator.full_validation(
            prices, window_days=20, coverage_threshold=0.90
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["coverage"])

    def test_validator_with_missing_columns(self):
        """测试：缺失必要列"""
        incomplete_prices = {
            "close": self.valid_prices["close"]
            # 缺少其他列
        }

        result = self.validator.add_amount_column(incomplete_prices)
        self.assertNotIn("amount", result)

    def test_coverage_threshold_customizable(self):
        """测试：覆盖率阈值可自定义"""
        validator1 = DataValidator(coverage_threshold=0.95)
        validator2 = DataValidator(coverage_threshold=0.80)

        result1 = validator1.check_coverage(self.valid_prices)
        result2 = validator2.check_coverage(self.valid_prices)

        # 不同的阈值应该得到不同的结果
        self.assertEqual(result1["threshold"], 0.95)
        self.assertEqual(result2["threshold"], 0.80)

    def test_symbol_specific_validation(self):
        """测试：特定标的的验证"""
        result = self.validator.check_full_window(
            self.valid_prices, window_days=20, symbol="SH600000"
        )

        # 应该检查特定标的
        self.assertTrue(result)


class TestDataValidatorEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def setUp(self):
        self.validator = DataValidator(verbose=False)

    def test_empty_prices(self):
        """测试：空价格数据"""
        empty_prices = {"close": pd.DataFrame()}
        result = self.validator.check_coverage(empty_prices)
        # 空DataFrame应该处理优雅 - 没有列意味着没有有效标的
        self.assertEqual(len(result.get("valid_symbols", [])), 0)

    def test_single_symbol(self):
        """测试：单个标的"""
        dates = pd.date_range("2023-01-01", periods=50)
        single_prices = {
            "close": pd.DataFrame(
                {"SH600000": np.random.randn(50).cumsum() + 100}, index=dates
            ),
            "volume": pd.DataFrame(
                {"SH600000": np.random.randint(100000, 500000, 50)}, index=dates
            ),
        }

        result = self.validator.check_coverage(single_prices, threshold=0.95)
        self.assertTrue(result["passed"])

    def test_all_nan_series(self):
        """测试：全NaN序列"""
        dates = pd.date_range("2023-01-01", periods=50)
        nan_prices = {
            "close": pd.DataFrame(
                {
                    "SH600000": np.full(50, np.nan),
                },
                index=dates,
            ),
            "volume": pd.DataFrame(
                {
                    "SH600000": np.full(50, np.nan),
                },
                index=dates,
            ),
        }

        result = self.validator.check_coverage(nan_prices, threshold=0.95)
        self.assertFalse(result["passed"])


class TestDataValidatorIntegration(unittest.TestCase):
    """集成测试"""

    def test_complete_pipeline(self):
        """测试：完整数据处理管道"""
        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=100)
        prices = {
            "close": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 100,
                    "SH600001": np.random.randn(100).cumsum() + 100,
                },
                index=dates,
            ),
            "high": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 102,
                    "SH600001": np.random.randn(100).cumsum() + 102,
                },
                index=dates,
            ),
            "low": pd.DataFrame(
                {
                    "SH600000": np.random.randn(100).cumsum() + 98,
                    "SH600001": np.random.randn(100).cumsum() + 98,
                },
                index=dates,
            ),
            "volume": pd.DataFrame(
                {
                    "SH600000": np.random.randint(100000, 500000, 100),
                    "SH600001": np.random.randint(100000, 500000, 100),
                },
                index=dates,
            ),
        }

        validator = DataValidator(verbose=False)

        # Step 1: 验证数据质量
        validation_result = validator.full_validation(prices, window_days=20)
        self.assertTrue(validation_result["passed"])

        # Step 2: 添加amount列
        prices_with_amount = validator.add_amount_column(prices)
        self.assertIn("amount", prices_with_amount)

        # Step 3: 再次验证（应该仍然通过）
        validation_result2 = validator.full_validation(
            prices_with_amount, window_days=20
        )
        self.assertTrue(validation_result2["passed"])


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidatorEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidatorIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    # 打印总结
    print("\n" + "=" * 70)
    print(f"✅ 测试完成: {result.testsRun}个测试")
    print(f"✓ 通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ 失败: {len(result.failures)}")
    print(f"⚠️ 错误: {len(result.errors)}")
    print("=" * 70)
