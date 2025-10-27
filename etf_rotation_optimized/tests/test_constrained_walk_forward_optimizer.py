"""
约束WFO优化器测试套件 | Constrained Walk-Forward Optimizer Tests

测试覆盖:
  1. 窗口划分
  2. IC计算整合
  3. 约束应用整合
  4. 前向性能评估
  5. 报告生成
  6. 集成场景

作者: Step 5 Constrained WFO Tests
日期: 2025-10-26
"""

import numpy as np
import pandas as pd
import pytest
from core.constrained_walk_forward_optimizer import (
    ConstrainedWalkForwardOptimizer,
    ConstraintApplicationReport,
)
from core.factor_selector import create_default_selector


class TestWindowPartitioning:
    """窗口划分测试"""

    def test_window_partition_basic(self):
        """基本窗口划分"""
        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        windows = optimizer._partition_windows(
            num_time_steps=500, is_period=100, oos_period=20, step_size=20
        )

        assert len(windows) > 0

        # 检查第一个窗口
        is_start, is_end, oos_start, oos_end = windows[0]
        assert is_start == 0
        assert is_end == 100
        assert oos_start == 100
        assert oos_end == 120

    def test_window_continuity(self):
        """窗口连续性"""
        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        windows = optimizer._partition_windows(
            num_time_steps=500, is_period=100, oos_period=20, step_size=20
        )

        for i in range(len(windows) - 1):
            is_start1, is_end1, oos_start1, oos_end1 = windows[i]
            is_start2, is_end2, _, _ = windows[i + 1]

            # 前一窗口的OOS结束 <= 后一窗口的IS开始
            assert oos_end1 <= is_start2 or is_end2 == is_end1 + 20

    def test_window_boundaries(self):
        """窗口边界检查"""
        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        windows = optimizer._partition_windows(
            num_time_steps=500, is_period=100, oos_period=20, step_size=20
        )

        # 所有窗口都应该在时间范围内
        for is_start, is_end, oos_start, oos_end in windows:
            assert is_start >= 0
            assert oos_end <= 500
            assert is_start < is_end
            assert oos_start < oos_end


class TestICComputation:
    """IC计算集成测试"""

    def test_ic_computation_window(self):
        """窗口IC计算"""
        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        np.random.seed(42)
        factors = np.random.randn(100, 30, 5)
        returns = np.random.randn(100, 30)
        factor_names = [f"FACTOR_{i}" for i in range(5)]

        ic_scores = optimizer._compute_window_ic(factors, returns, factor_names)

        assert len(ic_scores) == 5
        for name, ic in ic_scores.items():
            assert -1 <= ic <= 1

    def test_ic_computation_correlation(self):
        """IC计算相关性"""
        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        np.random.seed(42)

        # 创建与收益相关的因子
        returns = np.random.randn(100, 30)
        factors1 = returns[:, :, np.newaxis] * 0.8 + np.random.randn(100, 30, 1) * 0.2
        factors2 = np.random.randn(100, 30, 1)  # 无关因子

        factors = np.concatenate([factors1, factors2], axis=2)
        factor_names = ["HIGH_IC_FACTOR", "LOW_IC_FACTOR"]

        ic_scores = optimizer._compute_window_ic(factors, returns, factor_names)

        # 第一个因子IC应该更高
        assert ic_scores["HIGH_IC_FACTOR"] > ic_scores["LOW_IC_FACTOR"]


class TestConstraintIntegration:
    """约束集成测试"""

    def test_constraint_application_in_wfo(self):
        """WFO中的约束应用"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        # 生成测试数据
        factors = np.random.randn(200, 30, 10) * 0.1
        returns = np.random.randn(200, 30)
        factor_names = [f"FACTOR_{i}" for i in range(10)]

        # 运行约束WFO
        forward_df, window_reports = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=50,
            oos_period=20,
            step_size=20,
            target_factor_count=5,
        )

        # 检查结果
        assert len(forward_df) > 0
        assert "selected_factor_count" in forward_df.columns
        assert forward_df["selected_factor_count"].max() <= 5


class TestPerformanceMetrics:
    """性能指标测试"""

    def test_ic_drop_computation(self):
        """IC衰减计算"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        factors = np.random.randn(300, 30, 8) * 0.1
        returns = np.random.randn(300, 30)
        factor_names = [f"FACTOR_{i}" for i in range(8)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=80,
            oos_period=20,
            step_size=30,
            target_factor_count=4,
        )

        # IC衰减应该是可计算的
        assert "ic_drop" in forward_df.columns
        assert len(forward_df) > 0

    def test_forward_performance_structure(self):
        """前向性能结构"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        factors = np.random.randn(250, 25, 6)
        returns = np.random.randn(250, 25)
        factor_names = [f"FACTOR_{i}" for i in range(6)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=60,
            oos_period=15,
            step_size=20,
            target_factor_count=3,
        )

        required_columns = [
            "window",
            "is_start",
            "is_end",
            "oos_start",
            "oos_end",
            "is_ic_mean",
            "selected_factor_count",
            "selection_ic_mean",
            "oos_ic_mean",
        ]

        for col in required_columns:
            assert col in forward_df.columns


class TestWindowReports:
    """窗口报告测试"""

    def test_report_generation(self):
        """报告生成"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        factors = np.random.randn(250, 20, 5)
        returns = np.random.randn(250, 20)
        factor_names = [f"FACTOR_{i}" for i in range(5)]

        _, window_reports = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=60,
            oos_period=20,
            step_size=30,
            target_factor_count=3,
        )

        assert len(window_reports) > 0

        for report in window_reports:
            assert isinstance(report, ConstraintApplicationReport)
            assert hasattr(report, "window_index")
            assert hasattr(report, "selected_factors")
            assert hasattr(report, "oos_ic_mean")

    def test_report_structure(self):
        """报告结构验证"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        factors = np.random.randn(200, 15, 4)
        returns = np.random.randn(200, 15)
        factor_names = [f"FACTOR_{i}" for i in range(4)]

        _, window_reports = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=50,
            oos_period=15,
            step_size=25,
            target_factor_count=2,
        )

        report = window_reports[0]

        # 检查IC统计
        assert "mean" in report.is_ic_stats
        assert "median" in report.is_ic_stats
        assert "std" in report.is_ic_stats


class TestConstraintSelector:
    """约束选择器集成测试"""

    def test_custom_selector(self):
        """自定义选择器集成"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.01

        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        np.random.seed(42)
        factors = np.random.randn(200, 20, 6)
        returns = np.random.randn(200, 20)
        factor_names = [f"FACTOR_{i}" for i in range(6)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=50,
            oos_period=20,
            step_size=25,
            target_factor_count=4,
        )

        assert len(forward_df) > 0


class TestScalability:
    """可扩展性测试"""

    def test_large_dataset(self):
        """大规模数据集"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        # 500日期 × 50资产 × 15因子
        factors = np.random.randn(500, 50, 15) * 0.1
        returns = np.random.randn(500, 50)
        factor_names = [f"FACTOR_{i}" for i in range(15)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=30,
            target_factor_count=5,
        )

        assert len(forward_df) > 0
        assert forward_df["selected_factor_count"].max() <= 5

    def test_many_factors(self):
        """多因子场景"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        # 300日期 × 30资产 × 30因子
        factors = np.random.randn(300, 30, 30) * 0.1
        returns = np.random.randn(300, 30)
        factor_names = [f"FACTOR_{i}" for i in range(30)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=80,
            oos_period=20,
            step_size=25,
            target_factor_count=8,
        )

        assert len(forward_df) > 0
        assert forward_df["selected_factor_count"].max() <= 8


class TestEdgeCases:
    """边界情况测试"""

    def test_single_window(self):
        """单窗口场景"""
        np.random.seed(42)

        optimizer = ConstrainedWalkForwardOptimizer(verbose=False)

        factors = np.random.randn(150, 20, 5)
        returns = np.random.randn(150, 20)
        factor_names = [f"FACTOR_{i}" for i in range(5)]

        forward_df, window_reports = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=80,
            oos_period=50,
            step_size=100,
            target_factor_count=3,
        )

        # 应该有至少1个窗口
        assert len(forward_df) >= 1

    def test_all_factors_filtered(self):
        """所有因子被过滤"""
        np.random.seed(42)

        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.5  # 很高的阈值

        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        factors = np.random.randn(200, 20, 5)
        returns = np.random.randn(200, 20)
        factor_names = [f"FACTOR_{i}" for i in range(5)]

        forward_df, _ = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=50,
            oos_period=20,
            step_size=25,
            target_factor_count=3,
        )

        # 可能有窗口选中0个因子
        assert forward_df["selected_factor_count"].min() >= 0


class TestConsistency:
    """一致性测试"""

    def test_deterministic_results(self):
        """确定性结果"""
        np.random.seed(42)

        factors = np.random.randn(250, 25, 6)
        returns = np.random.randn(250, 25)
        factor_names = [f"FACTOR_{i}" for i in range(6)]

        selector1 = create_default_selector()
        optimizer1 = ConstrainedWalkForwardOptimizer(selector=selector1, verbose=False)

        df1, _ = optimizer1.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=60,
            oos_period=20,
            step_size=30,
            target_factor_count=4,
        )

        selector2 = create_default_selector()
        optimizer2 = ConstrainedWalkForwardOptimizer(selector=selector2, verbose=False)

        df2, _ = optimizer2.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=60,
            oos_period=20,
            step_size=30,
            target_factor_count=4,
        )

        # 结果应该相同
        assert len(df1) == len(df2)
        assert (df1["selected_factor_count"] == df2["selected_factor_count"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
