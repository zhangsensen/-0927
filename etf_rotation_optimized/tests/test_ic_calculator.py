"""
IC 计算器测试套件 | IC Calculator Test Suite

测试覆盖:
  1. Pearson IC 计算正确性
  2. Spearman RankIC 计算正确性
  3. Kendall τ 相关系数计算
  4. NaN 处理 (移除无效数据对)
  5. IC 统计量计算
  6. 显著性检验 (t-统计量、p-值)
  7. 多因子 IC 计算
  8. 边界情况处理
  9. 样本不足处理
  10. 报告生成

作者: Step 4 IC Calculator Tests
日期: 2025-10-26
"""

import numpy as np
import pandas as pd
import pytest
from core.ic_calculator import ICCalculator, ICStats
from scipy import stats as scipy_stats


class TestPearsonIC:
    """Pearson IC 计算测试"""

    def test_perfect_positive_correlation(self):
        """完全正相关"""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([2, 4, 6, 8, 10])

        ic = ICCalculator.compute_pearson_ic(factor, returns)
        assert np.isclose(ic, 1.0), f"期望 IC=1.0，得到 {ic}"

    def test_perfect_negative_correlation(self):
        """完全负相关"""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([5, 4, 3, 2, 1])

        ic = ICCalculator.compute_pearson_ic(factor, returns)
        assert np.isclose(ic, -1.0), f"期望 IC=-1.0，得到 {ic}"

    def test_zero_correlation(self):
        """无相关性"""
        np.random.seed(42)
        factor = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))  # 独立随机

        ic = ICCalculator.compute_pearson_ic(factor, returns)
        assert abs(ic) < 0.3, f"期望 IC ≈ 0，得到 {ic}"

    def test_nan_handling(self):
        """NaN 处理"""
        factor = pd.Series([1, 2, np.nan, 4, 5])
        returns = pd.Series([2, 4, 6, np.nan, 10])

        ic = ICCalculator.compute_pearson_ic(factor, returns)

        # 只应该使用有效数据对 (1,2), (2,4), (5,10)
        valid_factor = pd.Series([1, 2, 5])
        valid_returns = pd.Series([2, 4, 10])
        expected_ic = valid_factor.corr(valid_returns)

        assert np.isclose(ic, expected_ic), f"期望 IC={expected_ic:.4f}，得到 {ic:.4f}"

    def test_insufficient_data(self):
        """数据不足 (< 2个有效点)"""
        factor = pd.Series([1, np.nan])
        returns = pd.Series([2, 3])

        ic = ICCalculator.compute_pearson_ic(factor, returns)
        assert np.isnan(ic), "期望返回 NaN"


class TestSpearmanIC:
    """Spearman RankIC 测试"""

    def test_rank_correlation_monotonic(self):
        """单调关系"""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([10, 20, 30, 40, 50])

        ic = ICCalculator.compute_spearman_ic(factor, returns)
        assert np.isclose(ic, 1.0), f"期望 RankIC=1.0，得到 {ic}"

    def test_rank_robustness_to_outliers(self):
        """对离群值鲁棒"""
        factor = pd.Series([1, 2, 3, 4, 1000])  # 最后一个离群值
        returns = pd.Series([1, 2, 3, 4, 5])

        rank_ic = ICCalculator.compute_spearman_ic(factor, returns)

        # Pearson IC 会受离群值影响较大
        pearson_ic = ICCalculator.compute_pearson_ic(factor, returns)

        # RankIC 应该仍然 ≈ 1.0，而 PearsonIC 会较小
        assert rank_ic > pearson_ic, "RankIC 应该比 PearsonIC 更鲁棒"

    def test_nan_handling(self):
        """NaN 处理"""
        factor = pd.Series([1, 2, np.nan, 4, 5])
        returns = pd.Series([1, 2, 3, 4, 5])

        ic = ICCalculator.compute_spearman_ic(factor, returns)

        # 只使用 4 个有效数据对
        assert not np.isnan(ic), "应该能计算 RankIC"


class TestKendallIC:
    """Kendall τ 测试"""

    def test_kendall_correlation(self):
        """Kendall τ 计算"""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([1, 2, 3, 4, 5])

        ic = ICCalculator.compute_kendall_ic(factor, returns)

        # 完全一致排序应该给出接近 1 的 τ
        assert ic > 0.9, f"期望 τ > 0.9，得到 {ic}"

    def test_kendall_nan_handling(self):
        """NaN 处理"""
        factor = pd.Series([1, 2, np.nan, 4, 5])
        returns = pd.Series([1, 2, 3, 4, 5])

        ic = ICCalculator.compute_kendall_ic(factor, returns)
        assert not np.isnan(ic), "应该能计算 Kendall τ"


class TestICComputation:
    """IC 时间序列计算测试"""

    def setup_method(self):
        """测试前准备"""
        np.random.seed(42)

        # 创建模拟数据
        self.dates = pd.date_range("2025-01-01", periods=50)
        self.symbols = [f"ETF{i:02d}" for i in range(20)]

        # 标准化因子：完全随机
        factor_data = np.random.randn(len(self.dates), len(self.symbols))
        self.factor = pd.DataFrame(factor_data, index=self.dates, columns=self.symbols)

        # 收益率与因子相关
        noise = np.random.randn(len(self.dates), len(self.symbols)) * 0.5
        self.returns = pd.DataFrame(
            factor_data * 0.1 + noise, index=self.dates, columns=self.symbols
        )

    def test_ic_series_computation(self):
        """IC 时间序列计算"""
        calc = ICCalculator(verbose=False)
        factors_dict = {"TEST": self.factor}

        ic_dict = calc.compute_ic(factors_dict, self.returns, method="pearson")

        # 检查输出
        assert "TEST" in ic_dict, "因子未计算"
        ic_series = ic_dict["TEST"]

        # IC 值应该在 [-1, 1] 之间
        assert ic_series.min() >= -1.0, "IC 最小值异常"
        assert ic_series.max() <= 1.0, "IC 最大值异常"

        # 应该有多个 IC 值
        assert len(ic_series) > 0, "IC 序列为空"

    def test_spearman_ic_computation(self):
        """Spearman IC 计算"""
        calc = ICCalculator(verbose=False)
        factors_dict = {"TEST": self.factor}

        ic_dict = calc.compute_ic(factors_dict, self.returns, method="spearman")

        ic_series = ic_dict["TEST"]
        assert all(
            -1 <= ic <= 1 for ic in ic_series if not np.isnan(ic)
        ), "RankIC 值超出范围"

    def test_multiple_factors_ic(self):
        """多因子 IC 计算"""
        calc = ICCalculator(verbose=False)

        factors_dict = {
            "FACTOR_A": self.factor,
            "FACTOR_B": self.factor * 2,
            "FACTOR_C": -self.factor,
        }

        ic_dict = calc.compute_ic(factors_dict, self.returns, method="pearson")

        assert len(ic_dict) == 3, "应该有 3 个因子的 IC"
        assert all(len(ic) > 0 for ic in ic_dict.values()), "所有因子都应该有 IC 值"


class TestICStats:
    """IC 统计量计算测试"""

    def test_ic_stats_basic(self):
        """基本统计量"""
        ic_series = pd.Series([0.1, 0.15, 0.05, 0.2, 0.08])

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        assert stats_obj is not None, "统计量计算失败"
        assert np.isclose(stats_obj.mean, ic_series.mean()), "平均值错误"
        assert np.isclose(stats_obj.std, ic_series.std()), "标准差错误"

    def test_ic_stats_ir_calculation(self):
        """IR 计算"""
        ic_series = pd.Series([0.1, 0.12, 0.08, 0.11, 0.09])

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        expected_ir = stats_obj.mean / stats_obj.std
        assert np.isclose(stats_obj.ir, expected_ir), "IR 计算错误"

    def test_ic_stats_t_statistic(self):
        """t-统计量"""
        # 创建有明显信号的 IC 序列
        ic_series = pd.Series(np.random.randn(100) * 0.02 + 0.05)

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        # t-统计量应该 > 0
        assert stats_obj.t_stat > 0, "t-统计量应该 > 0"

        # 手动验证
        se = stats_obj.std / np.sqrt(len(ic_series))
        expected_t = stats_obj.mean / se
        assert np.isclose(stats_obj.t_stat, expected_t), "t-统计量计算错误"

    def test_ic_stats_p_value(self):
        """p 值计算"""
        ic_series = pd.Series(np.random.randn(100) * 0.01 + 0.05)

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        # p 值应该在 [0, 1]
        assert 0 <= stats_obj.p_value <= 1, "p 值超出范围"

    def test_insufficient_ic_data(self):
        """数据不足处理"""
        ic_series = pd.Series([0.1])  # 只有 1 个值

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        assert stats_obj is None, "数据不足应该返回 None"

    def test_all_nan_ic(self):
        """全 NaN 处理"""
        ic_series = pd.Series([np.nan, np.nan, np.nan])

        calc = ICCalculator(verbose=False)
        stats_obj = calc.compute_ic_stats(ic_series)

        assert stats_obj is None, "全 NaN 应该返回 None"


class TestEdgeCases:
    """边界情况测试"""

    def test_single_symbol(self):
        """单只标的"""
        dates = pd.date_range("2025-01-01", periods=20)

        factor = pd.DataFrame({"ETF01": np.random.randn(20)}, index=dates)
        returns = pd.DataFrame({"ETF01": np.random.randn(20)}, index=dates)

        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic({"SINGLE": factor}, returns)

        assert "SINGLE" in ic_dict, "单只标的计算失败"

    def test_all_nan_factor(self):
        """全 NaN 因子"""
        dates = pd.date_range("2025-01-01", periods=20)
        symbols = ["ETF01", "ETF02"]

        factor = pd.DataFrame(np.full((20, 2), np.nan), index=dates, columns=symbols)
        returns = pd.DataFrame(np.random.randn(20, 2), index=dates, columns=symbols)

        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic({"NAN_FACTOR": factor}, returns)

        # 应该返回空或全 NaN 的 IC
        ic_series = ic_dict["NAN_FACTOR"]
        assert len(ic_series) == 0 or all(
            pd.isna(ic_series)
        ), "全 NaN 因子应该得到全 NaN IC"

    def test_missing_dates(self):
        """缺失日期处理"""
        factor_dates = pd.date_range("2025-01-01", periods=20)
        return_dates = pd.date_range("2025-01-01", periods=30)

        factor = pd.DataFrame(np.random.randn(20, 3), index=factor_dates)
        returns = pd.DataFrame(np.random.randn(30, 3), index=return_dates)

        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic({"MISMATCHED": factor}, returns)

        # 应该处理日期不匹配
        ic_series = ic_dict["MISMATCHED"]
        assert len(ic_series) > 0, "应该有 IC 值"


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """完整工作流"""
        np.random.seed(42)

        # 准备数据
        dates = pd.date_range("2025-01-01", periods=100)
        symbols = [f"ETF{i:02d}" for i in range(30)]

        # 创建有预测能力的因子
        base_factor = np.random.randn(len(dates), len(symbols))
        noise = np.random.randn(len(dates), len(symbols)) * 0.3

        factor_df = pd.DataFrame(base_factor, index=dates, columns=symbols)
        returns_df = pd.DataFrame(
            base_factor * 0.08 + noise, index=dates, columns=symbols  # 因子有 8% 解释力
        )

        # 计算 IC
        calc = ICCalculator(verbose=False)
        ic_dict = calc.compute_ic(
            {"PREDICTOR": factor_df}, returns_df, method="pearson"
        )

        # 计算统计量
        ic_stats = calc.compute_all_ic_stats()

        assert "PREDICTOR" in ic_stats, "统计量计算失败"
        stats_obj = ic_stats["PREDICTOR"]

        # 验证统计量
        assert stats_obj.mean > 0, "有预测能力的因子应该有正平均 IC"
        assert stats_obj.std > 0, "标准差应该 > 0"
        assert stats_obj.ir > 0, "IR 应该 > 0"

    def test_comparison_multiple_methods(self):
        """多方法对比"""
        np.random.seed(42)

        dates = pd.date_range("2025-01-01", periods=50)
        symbols = [f"ETF{i:02d}" for i in range(20)]

        factor = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols
        )
        returns = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols
        )

        calc = ICCalculator(verbose=False)

        # 计算不同方法的 IC
        ic_pearson = calc.compute_ic({"TEST": factor}, returns, method="pearson")
        ic_spearman = calc.compute_ic({"TEST": factor}, returns, method="spearman")
        ic_kendall = calc.compute_ic({"TEST": factor}, returns, method="kendall")

        # 所有方法都应该产生结果
        assert len(ic_pearson["TEST"]) > 0
        assert len(ic_spearman["TEST"]) > 0
        assert len(ic_kendall["TEST"]) > 0

        # IC 值应该都在 [-1, 1]
        for ic_series in [ic_pearson["TEST"], ic_spearman["TEST"], ic_kendall["TEST"]]:
            valid_ic = ic_series.dropna()
            assert (valid_ic.abs() <= 1.0).all(), "IC 超出范围"


class TestReporting:
    """报告生成测试"""

    def test_summary_stats_generation(self):
        """总结统计表生成"""
        np.random.seed(42)

        dates = pd.date_range("2025-01-01", periods=50)
        symbols = [f"ETF{i:02d}" for i in range(20)]

        factors_dict = {}
        for i in range(3):
            factor_data = np.random.randn(len(dates), len(symbols))
            factors_dict[f"FACTOR_{i}"] = pd.DataFrame(
                factor_data, index=dates, columns=symbols
            )

        returns = pd.DataFrame(
            np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols
        )

        calc = ICCalculator(verbose=False)
        calc.compute_ic(factors_dict, returns)
        calc.compute_all_ic_stats()

        summary = calc.get_summary_stats()

        # 检查总结表
        assert len(summary) == 3, "应该有 3 个因子的统计"
        assert "因子" in summary.columns, "应该有因子列"
        assert "平均IC" in summary.columns, "应该有平均IC列"
        assert "显著" in summary.columns, "应该有显著列"


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
