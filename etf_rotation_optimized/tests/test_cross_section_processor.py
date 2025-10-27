"""
跨截面处理器测试套件 | CrossSectionProcessor Test Suite

测试覆盖:
  1. NaN 传播验证
  2. 有界因子不截断
  3. 无界因子标准化正确性
  4. 极值截断阈值准确性
  5. 批量处理完整性
  6. 元数据完整性

作者: Step 3 Testing
日期: 2025-10-26
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from core.cross_section_processor import CrossSectionProcessor, FactorMetadata


class TestCrossSectionProcessor(unittest.TestCase):
    """跨截面处理器测试"""

    def setUp(self):
        """测试前准备"""
        self.processor = CrossSectionProcessor(
            lower_percentile=2.5, upper_percentile=97.5, verbose=False
        )

        # 创建测试数据
        np.random.seed(42)
        self.dates = pd.date_range("2025-01-01", periods=20)
        self.symbols = [f"ETF{i:02d}" for i in range(30)]

    def test_nan_propagation_unbounded(self):
        """测试无界因子的 NaN 传播"""
        # 创建包含 NaN 的因子
        data = np.random.randn(len(self.dates), len(self.symbols))
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        # 随机插入 NaN
        nan_indices = [(5, 10), (8, 15), (12, 20), (18, 5)]
        for date_idx, symbol_idx in nan_indices:
            df.iloc[date_idx, symbol_idx] = np.nan

        factors = {"MOM_20D": df}

        # 处理
        processed = self.processor.process_all_factors(factors)

        # 验证：NaN 个数应保持不变
        original_nan = df.isna().sum().sum()
        processed_nan = processed["MOM_20D"].isna().sum().sum()

        self.assertEqual(
            original_nan,
            processed_nan,
            f"NaN 计数不匹配: {original_nan} vs {processed_nan}",
        )

    def test_nan_propagation_bounded(self):
        """测试有界因子的 NaN 传播"""
        # 创建包含 NaN 的有界因子
        data = np.random.uniform(0, 1, (len(self.dates), len(self.symbols)))
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        # 插入 NaN
        df.iloc[5, 10] = np.nan
        df.iloc[8, 15] = np.nan

        factors = {"PRICE_POSITION_20D": df}

        # 处理
        processed = self.processor.process_all_factors(factors)

        # 验证：NaN 个数应保持不变
        original_nan = df.isna().sum().sum()
        processed_nan = processed["PRICE_POSITION_20D"].isna().sum().sum()

        self.assertEqual(original_nan, processed_nan)

    def test_bounded_factor_no_clipping(self):
        """测试有界因子不被极值截断"""
        # 创建有界因子 [0, 1]
        data = np.random.uniform(0, 1, (len(self.dates), len(self.symbols)))
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        original_min = df.min().min()
        original_max = df.max().max()

        factors = {"PRICE_POSITION_20D": df}
        processed = self.processor.process_all_factors(factors)

        processed_min = processed["PRICE_POSITION_20D"].min().min()
        processed_max = processed["PRICE_POSITION_20D"].max().max()

        # 有界因子不应被改变
        self.assertAlmostEqual(original_min, processed_min, places=5)
        self.assertAlmostEqual(original_max, processed_max, places=5)

    def test_bounded_factor_range_preserved(self):
        """测试有界因子的值域被保留"""
        # 测试四个有界因子
        bounded_factors = {
            "PRICE_POSITION_20D": (0.0, 1.0),
            "PRICE_POSITION_120D": (0.0, 1.0),
            "PV_CORR_20D": (-1.0, 1.0),
            "RSI_14": (0.0, 100.0),
        }

        factors = {}
        for factor_name, (lower, upper) in bounded_factors.items():
            data = np.random.uniform(lower, upper, (len(self.dates), len(self.symbols)))
            factors[factor_name] = pd.DataFrame(
                data, index=self.dates, columns=self.symbols
            )

        # 处理
        processed = self.processor.process_all_factors(factors)

        # 验证每个有界因子的值域
        for factor_name, (expected_lower, expected_upper) in bounded_factors.items():
            result_min = processed[factor_name].min().min()
            result_max = processed[factor_name].max().max()

            self.assertGreaterEqual(
                result_min, expected_lower - 1e-6, f"{factor_name} min 超出下界"
            )
            self.assertLessEqual(
                result_max, expected_upper + 1e-6, f"{factor_name} max 超出上界"
            )

    def test_standardization_z_score(self):
        """测试 Z-score 标准化的正确性"""
        # 创建无界因子，确保每个日期有足够的数据
        np.random.seed(42)
        data = np.random.randn(len(self.dates), len(self.symbols)) * 10 + 50
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        factors = {"MOM_20D": df}
        processed = self.processor.process_all_factors(factors)

        # 每个日期的数据应该有 mean ≈ 0, std ≈ 1 (在 NaN 之外)
        passed = 0
        for date in processed["MOM_20D"].index:
            series = processed["MOM_20D"].loc[date]
            valid_series = series.dropna()

            if len(valid_series) > 1:
                mean = valid_series.mean()
                std = valid_series.std()

                # 检查 mean 和 std 是否接近目标值（允许 sample std 有一定波动）
                if abs(mean) < 0.2 and 0.8 < std < 1.2:
                    passed += 1

        # 至少大部分日期应该符合标准
        self.assertGreater(
            passed,
            len(self.dates) * 0.8,
            f"只有 {passed}/{len(self.dates)} 个日期通过标准化检查",
        )

    def test_winsorization_threshold(self):
        """测试 Winsorize 极值截断的阈值准确性"""
        # 创建已知分布的数据
        np.random.seed(123)
        data = np.random.randn(len(self.dates), len(self.symbols))
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        # 先标准化
        standardized, _ = self.processor.standardize_factor(df.iloc[0])

        # 再截断
        winsorized, win_stats = self.processor.winsorize_factor(standardized)

        # 验证截断后的值在下上界之间
        lower_bound = win_stats["lower_bound"]
        upper_bound = win_stats["upper_bound"]

        valid_winsorized = winsorized.dropna()

        self.assertTrue(
            (valid_winsorized >= lower_bound - 1e-6).all(), "存在小于下界的值"
        )
        self.assertTrue(
            (valid_winsorized <= upper_bound + 1e-6).all(), "存在大于上界的值"
        )

    def test_extreme_values_clipping(self):
        """测试极端值的正确截断"""
        # 创建包含极端值的数据
        # 注意：shape 需要匹配 (symbols, dates)
        data = {
            "S1": [-1000.0, -500.0],
            "S2": [-100.0, -50.0],
            "S3": [-1.0, -0.5],
            "S4": [0.0, 0.5],
            "S5": [1.0, 50.0],
            "S6": [100.0, 500.0],
            "S7": [1000.0, np.nan],
        }

        df = pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=2))

        # 标准化
        series = df.iloc[0]
        standardized, _ = self.processor.standardize_factor(series)

        # 截断
        winsorized, stats = self.processor.winsorize_factor(standardized)

        # 验证极端值被截断
        self.assertGreaterEqual(winsorized.min(), stats["lower_bound"] - 1e-6)
        self.assertLessEqual(winsorized.max(), stats["upper_bound"] + 1e-6)

    def test_all_nan_factor(self):
        """测试全 NaN 因子的处理"""
        # 创建全 NaN 因子
        df = pd.DataFrame(
            np.full((len(self.dates), len(self.symbols)), np.nan),
            index=self.dates,
            columns=self.symbols,
        )

        factors = {"MOM_20D": df}
        processed = self.processor.process_all_factors(factors)

        # 验证全 NaN 被保留
        self.assertTrue(processed["MOM_20D"].isna().all().all())

    def test_single_value_factor(self):
        """测试所有值相同的因子"""
        # 创建所有值为 5.0 的因子
        df = pd.DataFrame(
            np.full((len(self.dates), len(self.symbols)), 5.0),
            index=self.dates,
            columns=self.symbols,
        )

        factors = {"MOM_20D": df}
        processed = self.processor.process_all_factors(factors)

        # 标准差为 0，标准化应该返回原值（或 NaN）
        # 这是边界情况，应该有警告
        self.assertIn("方差为0", str(self.processor.processing_report["warnings"]))

    def test_batch_processing_completion(self):
        """测试批量处理的完整性"""
        # 创建包含所有 10 个因子的字典
        factors = {}

        # 无界因子
        for factor_name in [
            "MOM_20D",
            "SLOPE_20D",
            "RET_VOL_20D",
            "MAX_DD_60D",
            "VOL_RATIO_20D",
            "VOL_RATIO_60D",
        ]:
            data = np.random.randn(len(self.dates), len(self.symbols)) * 5 + 10
            factors[factor_name] = pd.DataFrame(
                data, index=self.dates, columns=self.symbols
            )

        # 有界因子
        factors["PRICE_POSITION_20D"] = pd.DataFrame(
            np.random.rand(len(self.dates), len(self.symbols)),
            index=self.dates,
            columns=self.symbols,
        )
        factors["PRICE_POSITION_120D"] = pd.DataFrame(
            np.random.rand(len(self.dates), len(self.symbols)),
            index=self.dates,
            columns=self.symbols,
        )
        factors["PV_CORR_20D"] = pd.DataFrame(
            np.random.uniform(-1, 1, (len(self.dates), len(self.symbols))),
            index=self.dates,
            columns=self.symbols,
        )
        factors["RSI_14"] = pd.DataFrame(
            np.random.uniform(0, 100, (len(self.dates), len(self.symbols))),
            index=self.dates,
            columns=self.symbols,
        )

        # 处理
        processed = self.processor.process_all_factors(factors)

        # 验证所有因子都被处理
        self.assertEqual(len(processed), 10, "未能处理所有 10 个因子")

        for factor_name in factors.keys():
            self.assertIn(factor_name, processed, f"缺少因子 {factor_name}")
            self.assertEqual(
                processed[factor_name].shape,
                factors[factor_name].shape,
                f"因子 {factor_name} 形状改变",
            )

    def test_metadata_correctness(self):
        """测试元数据完整性"""
        # 验证有界因子名单
        bounded = self.processor.list_bounded_factors()
        self.assertEqual(len(bounded), 4)

        for factor in bounded:
            self.assertIn(factor, self.processor.BOUNDED_FACTORS)
            bounds = self.processor.get_factor_bounds(factor)
            self.assertIsNotNone(bounds)

        # 验证无界因子名单
        unbounded = self.processor.list_unbounded_factors()
        self.assertEqual(len(unbounded), 6)

        for factor in unbounded:
            self.assertNotIn(factor, self.processor.BOUNDED_FACTORS)

    def test_inf_handling(self):
        """测试 Inf 值的处理"""
        # 创建包含 Inf 的因子
        data = {
            "S1": [1.0, 7.0],
            "S2": [2.0, 8.0],
            "S3": [np.inf, 9.0],
            "S4": [4.0, 10.0],
            "S5": [-np.inf, 11.0],
            "S6": [6.0, 12.0],
        }

        df = pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=2))

        # 标准化
        series = df.iloc[0]
        standardized, stats = self.processor.standardize_factor(series)

        # 验证 Inf 被转换为 NaN
        self.assertTrue(pd.isna(standardized["S3"]))
        self.assertTrue(pd.isna(standardized["S5"]))

    def test_mixed_nan_inf(self):
        """测试混合 NaN 和 Inf 的处理"""
        data = {
            "S1": [1.0, 7.0],
            "S2": [np.nan, 8.0],
            "S3": [np.inf, 9.0],
            "S4": [4.0, np.nan],
            "S5": [-np.inf, 11.0],
            "S6": [np.nan, 12.0],
        }

        df = pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=2))

        series = df.iloc[0]
        standardized, stats = self.processor.standardize_factor(series)

        # 应该有 4 个 NaN (原本2个 + Inf转换的2个)
        nan_count = standardized.isna().sum()
        self.assertGreaterEqual(nan_count, 4)

    def test_empty_dataframe(self):
        """测试空 DataFrame 的处理"""
        df = pd.DataFrame(
            np.full((len(self.dates), len(self.symbols)), np.nan),
            index=self.dates,
            columns=self.symbols,
        )

        factors = {"MOM_20D": df}
        processed = self.processor.process_all_factors(factors)

        # 应该返回全 NaN
        result = processed["MOM_20D"]
        self.assertTrue(result.isna().all().all())

    def test_consistency_across_dates(self):
        """测试日期间的一致性"""
        # 创建所有日期 + 因子都是独立的正态分布
        data = np.random.randn(len(self.dates), len(self.symbols))
        df = pd.DataFrame(data, index=self.dates, columns=self.symbols)

        factors = {"SLOPE_20D": df}
        processed = self.processor.process_all_factors(factors)

        # 每个日期独立标准化，应该都是 mean≈0, std≈1
        for date in processed["SLOPE_20D"].index:
            series = processed["SLOPE_20D"].loc[date]
            valid = series.dropna()

            if len(valid) > 2:
                mean = valid.mean()
                std = valid.std()

                # 允许一些数值误差
                self.assertLess(abs(mean), 0.15, f"Date {date}: mean 偏离0太大")
                self.assertLess(abs(std - 1.0), 0.25, f"Date {date}: std 偏离1太大")

    def test_processing_report(self):
        """测试处理报告的完整性"""
        factors = {
            "MOM_20D": pd.DataFrame(
                np.random.randn(len(self.dates), len(self.symbols)),
                index=self.dates,
                columns=self.symbols,
            )
        }

        processed = self.processor.process_all_factors(factors)
        report = self.processor.get_report()

        # 验证报告字段
        self.assertIsNotNone(report["timestamp"])
        self.assertIn("MOM_20D", report["factors_processed"])
        self.assertIn("MOM_20D", report["nan_stats"])

    def test_percentile_parameters(self):
        """测试不同分位数参数的影响"""
        data = np.random.randn(100, 50)
        df = pd.DataFrame(data)

        # 测试不同的截断参数
        for lower, upper in [(1.0, 99.0), (2.5, 97.5), (5.0, 95.0)]:
            processor = CrossSectionProcessor(
                lower_percentile=lower, upper_percentile=upper, verbose=False
            )

            series = df.iloc[0]
            standardized, _ = processor.standardize_factor(series)
            winsorized, stats = processor.winsorize_factor(standardized)

            # 验证参数被正确应用
            self.assertEqual(stats["lower_percentile"], lower)
            self.assertEqual(stats["upper_percentile"], upper)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_end_to_end_processing(self):
        """端到端处理测试"""
        np.random.seed(42)

        # 创建 10 个因子的完整矩阵
        dates = pd.date_range("2025-01-01", periods=20)
        symbols = [f"ETF{i:02d}" for i in range(30)]

        factors = {}

        # 无界因子
        for factor_name in [
            "MOM_20D",
            "SLOPE_20D",
            "RET_VOL_20D",
            "MAX_DD_60D",
            "VOL_RATIO_20D",
            "VOL_RATIO_60D",
        ]:
            data = np.random.randn(len(dates), len(symbols)) * 10
            factors[factor_name] = pd.DataFrame(data, index=dates, columns=symbols)

            # 插入一些 NaN
            mask = np.random.random((len(dates), len(symbols))) < 0.05
            factors[factor_name][mask] = np.nan

        # 有界因子
        factors["PRICE_POSITION_20D"] = pd.DataFrame(
            np.random.rand(len(dates), len(symbols)), index=dates, columns=symbols
        )
        factors["PRICE_POSITION_120D"] = pd.DataFrame(
            np.random.rand(len(dates), len(symbols)), index=dates, columns=symbols
        )
        factors["PV_CORR_20D"] = pd.DataFrame(
            np.random.uniform(-1, 1, (len(dates), len(symbols))),
            index=dates,
            columns=symbols,
        )
        factors["RSI_14"] = pd.DataFrame(
            np.random.uniform(0, 100, (len(dates), len(symbols))),
            index=dates,
            columns=symbols,
        )

        # 处理
        processor = CrossSectionProcessor(verbose=False)
        processed = processor.process_all_factors(factors)

        # 全面验证
        self.assertEqual(len(processed), 10)

        # 验证无界因子
        for factor_name in ["MOM_20D", "SLOPE_20D", "RET_VOL_20D"]:
            result = processed[factor_name]

            # 检查形状
            self.assertEqual(result.shape, factors[factor_name].shape)

            # 检查 NaN 保留
            original_nan = factors[factor_name].isna().sum().sum()
            processed_nan = result.isna().sum().sum()
            self.assertEqual(original_nan, processed_nan)

        # 验证有界因子
        for factor_name in ["PRICE_POSITION_20D", "PRICE_POSITION_120D"]:
            result = processed[factor_name]
            self.assertTrue((result.min().min() >= 0.0 - 1e-6))
            self.assertTrue((result.max().max() <= 1.0 + 1e-6))


if __name__ == "__main__":
    # 运行测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestCrossSectionProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结 | Test Summary")
    print("=" * 70)
    print(f"运行测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)
