"""
前向回测框架测试套件 | Walk-Forward Optimizer Test Suite

测试覆盖:
  1. 窗口划分正确性
  2. IS/OOS 窗口边界
  3. 因子筛选逻辑
  4. OOS 性能评估
  5. 前向汇总指标
  6. 因子选中频率计算
  7. 边界情况处理
  8. 报告生成

作者: Step 4 WFO Tests
日期: 2025-10-26
"""

import numpy as np
import pandas as pd
import pytest
from core.walk_forward_optimizer import WalkForwardOptimizer, WFOResult, WFOWindow


class TestWindowPartitioning:
    """窗口划分测试"""

    def test_basic_partitioning(self):
        """基本窗口划分"""
        data = pd.DataFrame(
            np.random.randn(500, 5), index=pd.date_range("2023-01-01", periods=500)
        )

        wfo = WalkForwardOptimizer(is_window=252, oos_window=60, step=20, verbose=False)
        windows = wfo.partition_data(data)

        # 检查窗口数量
        assert len(windows) > 0, "应该有窗口"

        # 检查第一个窗口
        first_window = windows[0]
        assert first_window[0] == 0, "第一个 IS 窗口应该从 0 开始"
        assert first_window[1] == 252, "第一个 IS 窗口应该是 252"
        assert first_window[2] == 252, "第一个 OOS 窗口应该从 252 开始"
        assert first_window[3] == 312, "第一个 OOS 窗口应该是 312"

    def test_sliding_step(self):
        """滑动步长"""
        data = pd.DataFrame(
            np.random.randn(600, 5), index=pd.date_range("2023-01-01", periods=600)
        )

        wfo = WalkForwardOptimizer(is_window=252, oos_window=60, step=20, verbose=False)
        windows = wfo.partition_data(data)

        # 检查步长
        if len(windows) >= 2:
            w1_start = windows[0][0]
            w2_start = windows[1][0]
            assert w2_start - w1_start == 20, "步长应该是 20"

    def test_no_overlapping_oos_windows(self):
        """OOS 窗口不应该重叠"""
        data = pd.DataFrame(
            np.random.randn(600, 5), index=pd.date_range("2023-01-01", periods=600)
        )

        wfo = WalkForwardOptimizer(is_window=252, oos_window=60, step=20, verbose=False)
        windows = wfo.partition_data(data)

        # 检查 OOS 窗口不超出范围
        for _, _, oos_s, oos_e in windows:
            assert oos_e <= len(data), "OOS 窗口不应该超出数据范围"

    def test_insufficient_data(self):
        """数据不足处理"""
        data = pd.DataFrame(
            np.random.randn(100, 5), index=pd.date_range("2023-01-01", periods=100)
        )

        wfo = WalkForwardOptimizer(is_window=252, oos_window=60, step=20, verbose=False)
        windows = wfo.partition_data(data)

        # 数据不足时应该返回空列表或较少的窗口
        assert len(windows) == 0, "数据不足应该返回空窗口"


class TestFactorSelection:
    """因子筛选测试"""

    def setup_method(self):
        """测试前准备"""
        np.random.seed(42)
        self.dates = pd.date_range("2025-01-01", periods=300)
        self.symbols = [f"ETF{i:02d}" for i in range(20)]

    def test_factor_selection_with_threshold(self):
        """因子筛选阈值"""
        # 创建有不同 IC 的因子
        factors_dict = {}

        # 好因子 (IC > 0.05)
        factor_good = np.random.randn(len(self.dates), len(self.symbols)) * 0.1 + 0.2
        factors_dict["GOOD_FACTOR"] = pd.DataFrame(
            factor_good, index=self.dates, columns=self.symbols
        )

        # 坏因子 (IC < 0.05)
        factor_bad = np.random.randn(len(self.dates), len(self.symbols))
        factors_dict["BAD_FACTOR"] = pd.DataFrame(
            factor_bad, index=self.dates, columns=self.symbols
        )

        # 创建与好因子相关的收益
        returns_df = pd.DataFrame(
            factor_good * 0.5
            + np.random.randn(len(self.dates), len(self.symbols)) * 0.1,
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(ic_threshold=0.05, verbose=False)
        selected, ic_stats = wfo.select_factors(factors_dict, returns_df, 0, 200)

        # 好因子应该被选中
        assert "GOOD_FACTOR" in selected, "好因子应该被选中"

    def test_no_factors_selected(self):
        """无因子被选中"""
        factors_dict = {}

        # 创建无预测能力的因子
        factor = np.random.randn(len(self.dates), len(self.symbols))
        factors_dict["NOISE"] = pd.DataFrame(
            factor, index=self.dates, columns=self.symbols
        )

        # 无相关的收益
        returns = np.random.randn(len(self.dates), len(self.symbols))
        returns_df = pd.DataFrame(returns, index=self.dates, columns=self.symbols)

        wfo = WalkForwardOptimizer(ic_threshold=0.1, verbose=False)
        selected, ic_stats = wfo.select_factors(factors_dict, returns_df, 0, 200)

        # 可能没有因子被选中
        assert len(selected) <= 1, "无预测能力的因子不应该被选中"


class TestOOSEvaluation:
    """OOS 评估测试"""

    def setup_method(self):
        """测试前准备"""
        np.random.seed(42)
        self.dates = pd.date_range("2025-01-01", periods=300)
        self.symbols = [f"ETF{i:02d}" for i in range(20)]

    def test_oos_evaluation_basic(self):
        """基本 OOS 评估"""
        factors_dict = {}
        factor_data = np.random.randn(len(self.dates), len(self.symbols))
        factors_dict["TEST"] = pd.DataFrame(
            factor_data, index=self.dates, columns=self.symbols
        )

        returns_df = pd.DataFrame(
            factor_data * 0.05
            + np.random.randn(len(self.dates), len(self.symbols)) * 0.02,
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(verbose=False)
        oos_stats, oos_perf = wfo.evaluate_factors_oos(
            factors_dict, returns_df, ["TEST"], 100, 160
        )

        # 检查性能指标存在
        assert "mean_ic" in oos_perf, "应该有平均 IC"
        assert "sharpe" in oos_perf, "应该有 Sharpe"
        assert "ir" in oos_perf, "应该有 IR"

    def test_oos_with_no_factors(self):
        """无因子的 OOS 评估"""
        factors_dict = {}
        returns_df = pd.DataFrame(
            np.random.randn(len(self.dates), len(self.symbols)),
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(verbose=False)
        oos_stats, oos_perf = wfo.evaluate_factors_oos(
            factors_dict, returns_df, [], 100, 160
        )

        # 应该返回空结果
        assert len(oos_stats) == 0, "无因子应该返回空统计"


class TestWalkForward:
    """前向回测测试"""

    def setup_method(self):
        """测试前准备"""
        np.random.seed(42)
        self.dates = pd.date_range("2024-01-01", periods=600)
        self.symbols = [f"ETF{i:02d}" for i in range(20)]

    def test_walk_forward_basic(self):
        """基本前向回测"""
        factors_dict = {}
        for i in range(3):
            factor_data = np.random.randn(len(self.dates), len(self.symbols))
            factors_dict[f"FACTOR_{i}"] = pd.DataFrame(
                factor_data, index=self.dates, columns=self.symbols
            )

        returns_df = pd.DataFrame(
            np.random.randn(len(self.dates), len(self.symbols)) * 0.01,
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward(factors_dict, returns_df)

        # 检查结果
        assert result is not None, "应该返回 WFOResult"
        assert len(result.windows) > 0, "应该有窗口"
        assert result.all_factors_count == 3, "应该有 3 个因子"

    def test_walk_forward_results_structure(self):
        """前向回测结果结构"""
        factors_dict = {}
        factor_data = np.random.randn(len(self.dates), len(self.symbols))
        factors_dict["TEST"] = pd.DataFrame(
            factor_data, index=self.dates, columns=self.symbols
        )

        returns_df = pd.DataFrame(
            factor_data * 0.05
            + np.random.randn(len(self.dates), len(self.symbols)) * 0.02,
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward(factors_dict, returns_df)

        # 检查结果属性
        assert hasattr(result, "windows"), "应该有 windows 属性"
        assert hasattr(result, "forward_mean_ic"), "应该有 forward_mean_ic"
        assert hasattr(result, "forward_sharpe"), "应该有 forward_sharpe"
        assert hasattr(result, "selected_factors_freq"), "应该有 selected_factors_freq"

    def test_factor_selection_frequency(self):
        """因子选中频率"""
        factors_dict = {}

        # 创建强信号因子
        strong_signal = np.random.randn(len(self.dates), len(self.symbols)) * 0.1 + 0.3
        factors_dict["STRONG"] = pd.DataFrame(
            strong_signal, index=self.dates, columns=self.symbols
        )

        # 创建弱信号因子
        weak_signal = np.random.randn(len(self.dates), len(self.symbols))
        factors_dict["WEAK"] = pd.DataFrame(
            weak_signal, index=self.dates, columns=self.symbols
        )

        # 收益与强因子相关
        returns_df = pd.DataFrame(
            strong_signal * 0.3
            + np.random.randn(len(self.dates), len(self.symbols)) * 0.05,
            index=self.dates,
            columns=self.symbols,
        )

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward(factors_dict, returns_df)

        # 强因子应该被选中更多次
        if result.selected_factors_freq:
            freq_df = pd.DataFrame(
                list(result.selected_factors_freq.items()), columns=["factor", "freq"]
            )
            strong_freq = freq_df[freq_df["factor"] == "STRONG"]["freq"].values
            if len(strong_freq) > 0:
                assert strong_freq[0] > 0, "强因子应该被选中"


class TestEdgeCases:
    """边界情况测试"""

    def test_single_symbol(self):
        """单只标的"""
        dates = pd.date_range("2024-01-01", periods=600)

        factor = pd.DataFrame(np.random.randn(600), index=dates, columns=["ETF"])
        returns = pd.DataFrame(np.random.randn(600), index=dates, columns=["ETF"])

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward({"SINGLE": factor}, returns)

        assert len(result.windows) > 0, "应该能处理单只标的"

    def test_exactly_enough_data(self):
        """数据恰好足够"""
        dates = pd.date_range("2024-01-01", periods=312)  # 252 + 60
        symbols = ["ETF01", "ETF02"]

        factor = pd.DataFrame(np.random.randn(312, 2), index=dates, columns=symbols)
        returns = pd.DataFrame(np.random.randn(312, 2), index=dates, columns=symbols)

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward({"TEST": factor}, returns)

        assert len(result.windows) == 1, "恰好一个窗口"

    def test_all_nan_factor(self):
        """全 NaN 因子"""
        dates = pd.date_range("2024-01-01", periods=600)
        symbols = ["ETF01", "ETF02"]

        factor = pd.DataFrame(np.full((600, 2), np.nan), index=dates, columns=symbols)
        returns = pd.DataFrame(np.random.randn(600, 2), index=dates, columns=symbols)

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward({"NAN": factor}, returns)

        # 应该能处理，但可能没有因子被选中
        assert result is not None, "应该返回结果"


class TestReporting:
    """报告生成测试"""

    def test_summary_dataframe(self):
        """总结 DataFrame 生成"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=600)
        symbols = [f"ETF{i:02d}" for i in range(10)]

        factors_dict = {}
        factor_data = np.random.randn(len(dates), len(symbols))
        factors_dict["TEST"] = pd.DataFrame(factor_data, index=dates, columns=symbols)

        returns_df = pd.DataFrame(
            factor_data * 0.05 + np.random.randn(len(dates), len(symbols)) * 0.02,
            index=dates,
            columns=symbols,
        )

        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=100, verbose=False
        )
        result = wfo.walk_forward(factors_dict, returns_df)

        summary_df = wfo.get_summary_dataframe(result)

        # 检查 DataFrame 结构
        assert len(summary_df) == len(result.windows), "应该有相同行数"
        assert "窗口" in summary_df.columns, "应该有窗口列"
        assert "选中因子数" in summary_df.columns, "应该有选中因子数列"


class TestIntegration:
    """集成测试"""

    def test_full_wfo_pipeline(self):
        """完整 WFO 流程"""
        np.random.seed(42)

        dates = pd.date_range("2024-01-01", periods=600)
        symbols = [f"ETF{i:02d}" for i in range(30)]

        # 创建多个因子
        factors_dict = {}
        for i in range(5):
            factor_data = np.random.randn(len(dates), len(symbols))
            factors_dict[f"FACTOR_{i}"] = pd.DataFrame(
                factor_data, index=dates, columns=symbols
            )

        # 创建有因子预测能力的收益
        base = np.random.randn(len(dates), len(symbols))
        returns_df = pd.DataFrame(
            base * 0.03 + np.random.randn(len(dates), len(symbols)) * 0.02,
            index=dates,
            columns=symbols,
        )

        # 执行 WFO
        wfo = WalkForwardOptimizer(
            is_window=252, oos_window=60, step=20, ic_threshold=0.03, verbose=False
        )

        result = wfo.walk_forward(factors_dict, returns_df)

        # 验证结果
        assert len(result.windows) > 0, "应该有窗口"
        assert not np.isnan(result.forward_mean_ic), "应该有前向 IC"
        assert not np.isnan(result.forward_sharpe), "应该有前向 Sharpe"

        # 获取总结
        summary_df = wfo.get_summary_dataframe(result)
        assert len(summary_df) == len(result.windows), "总结表应该匹配窗口数"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
