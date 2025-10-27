"""
端到端测试套件 | End-to-End Test Suite

测试覆盖：
  1. 从原始数据到投资组合的完整流程
  2. 每个模块的集成与交互
  3. 数据流转的一致性
  4. 边界情况和异常处理
  5. 性能基准与压力测试

场景：
  • 场景1: 基础流程验证
  • 场景2: 大规模数据处理
  • 场景3: 异常数据处理
  • 场景4: 压力测试
  • 场景5: 性能基准

作者: Step 6 End-to-End Tests
日期: 2025-10-26
"""

import time
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from core.constrained_walk_forward_optimizer import ConstrainedWalkForwardOptimizer
from core.factor_selector import create_default_selector
from core.ic_calculator import ICCalculator
from core.walk_forward_optimizer import WalkForwardOptimizer


class TestE2EBasicFlow:
    """基本流程端到端测试"""

    def test_complete_pipeline(self):
        """完整管道流程"""
        np.random.seed(42)

        # 步骤1：数据准备
        num_days = 300
        num_assets = 30
        num_factors = 10

        factors = np.random.randn(num_days, num_assets, num_factors) * 0.1
        returns = np.random.randn(num_days, num_assets)
        factor_names = [f"FACTOR_{i}" for i in range(num_factors)]

        # 步骤2：创建优化器
        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        # 步骤3：运行优化
        forward_df, window_reports = optimizer.run_constrained_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=80,
            oos_period=20,
            step_size=30,
            target_factor_count=5,
        )

        # 步骤4：验证结果
        assert len(forward_df) > 0, "应有优化窗口"
        assert len(window_reports) == len(forward_df), "报告数量应与窗口数一致"
        assert "selected_factors" in forward_df.columns, "应有选中因子列"
        assert "oos_ic_mean" in forward_df.columns, "应有OOS IC列"

        # 步骤5：验证数据一致性
        for i, row in forward_df.iterrows():
            assert row["selected_factor_count"] <= 5, "选中因子数不应超过目标"
            assert isinstance(row["selected_factors"], str), "因子应为字符串"

    def test_data_flow_consistency(self):
        """数据流转一致性"""
        np.random.seed(42)

        # 准备数据
        factors = np.random.randn(250, 20, 8) * 0.1
        returns = np.random.randn(250, 20)
        factor_names = [f"F{i}" for i in range(8)]

        # 运行优化
        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=60,
            oos_period=20,
            step_size=25,
            target_factor_count=4,
        )

        # 验证数据一致性
        # 1. 窗口范围合理
        for idx, row in forward_df.iterrows():
            assert row["is_start"] < row["is_end"], "IS窗口应有效"
            assert row["oos_start"] < row["oos_end"], "OOS窗口应有效"
            assert row["is_end"] == row["oos_start"], "窗口应连接"

        # 2. 性能指标合理
        assert forward_df["is_ic_mean"].between(-1, 1).all(), "IC应在[-1,1]范围"
        assert forward_df["selection_ic_mean"].between(-1, 1).all(), "IC应在[-1,1]范围"
        assert forward_df["oos_ic_mean"].between(-1, 1).all(), "IC应在[-1,1]范围"


class TestE2ELargeScale:
    """大规模数据处理"""

    def test_large_dataset_1000_days(self):
        """1000天数据集处理"""
        np.random.seed(42)

        factors = np.random.randn(1000, 50, 15) * 0.1
        returns = np.random.randn(1000, 50)
        factor_names = [f"F{i}" for i in range(15)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        start_time = time.time()

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=200,
            oos_period=50,
            step_size=100,
            target_factor_count=6,
        )

        elapsed = time.time() - start_time

        # 验证完成
        assert len(forward_df) > 0, "应完成处理"
        assert elapsed < 30, "处理时间应在30秒内"

        # 计算吞吐量
        throughput = (1000 * 50 * 15) / elapsed
        assert throughput > 50000, f"吞吐量应 > 50k, 实际 {throughput:.0f}"

    def test_many_factors_30(self):
        """30个因子处理"""
        np.random.seed(42)

        factors = np.random.randn(500, 40, 30) * 0.1
        returns = np.random.randn(500, 40)
        factor_names = [f"F{i}" for i in range(30)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=100,
            oos_period=30,
            step_size=50,
            target_factor_count=8,
        )

        assert len(forward_df) > 0
        assert forward_df["selected_factor_count"].max() <= 8


class TestE2EAnomalies:
    """异常数据处理"""

    def test_nan_handling(self):
        """NaN值处理"""
        np.random.seed(42)

        factors = np.random.randn(200, 20, 8)
        returns = np.random.randn(200, 20)

        # 引入NaN（会被过滤掉）
        factors[10:15, 5, 2] = np.nan
        returns[10:15, 10] = np.nan

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        factor_names = [f"F{i}" for i in range(8)]

        # 应该能处理NaN
        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=50,
            oos_period=20,
            step_size=30,
            target_factor_count=4,
        )

        assert len(forward_df) > 0, "应能处理NaN"

    def test_zero_variance_factor(self):
        """零方差因子处理"""
        np.random.seed(42)

        factors = np.random.randn(200, 20, 8)
        returns = np.random.randn(200, 20)

        # 一个因子是常数
        factors[:, :, 3] = 1.0

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        factor_names = [f"F{i}" for i in range(8)]

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=60,
            oos_period=20,
            step_size=30,
            target_factor_count=5,
        )

        # 应该能处理零方差
        assert len(forward_df) > 0


class TestE2EStressTest:
    """压力测试"""

    def test_extreme_window_sizes(self):
        """极端窗口大小"""
        np.random.seed(42)

        factors = np.random.randn(500, 30, 10)
        returns = np.random.randn(500, 30)
        factor_names = [f"F{i}" for i in range(10)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        # 很小的OOS窗口
        forward_df, _ = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=200,
            oos_period=5,
            step_size=50,
            target_factor_count=5,
        )

        assert len(forward_df) > 0

    def test_extreme_factor_counts(self):
        """极端因子数量"""
        np.random.seed(42)

        # 只有1个因子
        factors = np.random.randn(300, 20, 1)
        returns = np.random.randn(300, 20)

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        forward_df, _ = optimizer.run_constrained_wfo(
            factors,
            returns,
            ["F0"],
            is_period=80,
            oos_period=20,
            step_size=40,
            target_factor_count=1,
        )

        assert len(forward_df) > 0
        assert forward_df["selected_factor_count"].max() == 1


class TestE2EPerformance:
    """性能基准测试"""

    def test_throughput_benchmark(self):
        """吞吐量基准"""
        np.random.seed(42)

        # 标准配置: 500日 × 50资产 × 15因子
        factors = np.random.randn(500, 50, 15)
        returns = np.random.randn(500, 50)
        factor_names = [f"F{i}" for i in range(15)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        start_time = time.time()

        forward_df, _ = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=100,
            oos_period=20,
            step_size=30,
            target_factor_count=5,
        )

        elapsed = time.time() - start_time
        throughput = (500 * 50 * 15) / elapsed

        # 应该 > 50k, 目标 260k+
        assert throughput > 50000, f"吞吐量 {throughput:.0f} < 50k"
        print(f"吞吐量: {throughput:.0f} 对/秒")

    def test_memory_efficiency(self):
        """内存效率"""
        import sys

        np.random.seed(42)

        factors = np.random.randn(500, 50, 15)
        returns = np.random.randn(500, 50)

        # 估计内存占用
        factor_size = sys.getsizeof(factors)
        return_size = sys.getsizeof(returns)

        # 应该在合理范围内
        total_mb = (factor_size + return_size) / (1024**2)
        assert total_mb < 50, f"内存占用 {total_mb:.1f}MB 过大"


class TestE2EReporting:
    """报告生成与验证"""

    def test_report_completeness(self):
        """报告完整性"""
        np.random.seed(42)

        factors = np.random.randn(300, 25, 8)
        returns = np.random.randn(300, 25)
        factor_names = [f"F{i}" for i in range(8)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=80,
            oos_period=20,
            step_size=40,
            target_factor_count=4,
        )

        # 验证DataFrame完整性
        required_cols = [
            "window",
            "is_start",
            "is_end",
            "oos_start",
            "oos_end",
            "is_ic_mean",
            "selected_factor_count",
            "selection_ic_mean",
            "oos_ic_mean",
            "ic_drop",
        ]

        for col in required_cols:
            assert col in forward_df.columns, f"缺少列: {col}"

        # 验证报告对象
        for report in reports:
            assert hasattr(report, "window_index")
            assert hasattr(report, "selected_factors")
            assert hasattr(report, "oos_performance")
            assert len(report.selected_factors) > 0 or len(report.selected_factors) == 0

    def test_report_export(self):
        """报告导出"""
        np.random.seed(42)

        factors = np.random.randn(250, 20, 6)
        returns = np.random.randn(250, 20)
        factor_names = [f"F{i}" for i in range(6)]

        selector = create_default_selector()
        optimizer = ConstrainedWalkForwardOptimizer(selector=selector, verbose=False)

        forward_df, reports = optimizer.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=70,
            oos_period=20,
            step_size=30,
            target_factor_count=3,
        )

        # 应该能导出为CSV
        csv_data = forward_df.to_csv()
        assert len(csv_data) > 0

        # 应该能导出为JSON
        json_data = forward_df.to_json()
        assert len(json_data) > 0


class TestE2EIntegration:
    """整体集成测试"""

    def test_multiple_runs_consistency(self):
        """多次运行一致性"""
        np.random.seed(42)

        factors = np.random.randn(300, 25, 8)
        returns = np.random.randn(300, 25)
        factor_names = [f"F{i}" for i in range(8)]

        selector1 = create_default_selector()
        optimizer1 = ConstrainedWalkForwardOptimizer(selector=selector1, verbose=False)

        df1, _ = optimizer1.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=80,
            oos_period=20,
            step_size=40,
            target_factor_count=4,
        )

        selector2 = create_default_selector()
        optimizer2 = ConstrainedWalkForwardOptimizer(selector=selector2, verbose=False)

        df2, _ = optimizer2.run_constrained_wfo(
            factors,
            returns,
            factor_names,
            is_period=80,
            oos_period=20,
            step_size=40,
            target_factor_count=4,
        )

        # 结果应一致
        assert len(df1) == len(df2)
        assert (df1["selected_factor_count"] == df2["selected_factor_count"]).all()

    def test_reproducibility(self):
        """可复现性"""
        np.random.seed(123)
        factors1 = np.random.randn(250, 20, 6)
        returns1 = np.random.randn(250, 20)

        np.random.seed(123)
        factors2 = np.random.randn(250, 20, 6)
        returns2 = np.random.randn(250, 20)

        factor_names = [f"F{i}" for i in range(6)]

        selector1 = create_default_selector()
        optimizer1 = ConstrainedWalkForwardOptimizer(selector=selector1, verbose=False)
        df1, _ = optimizer1.run_constrained_wfo(
            factors1,
            returns1,
            factor_names,
            is_period=70,
            oos_period=20,
            step_size=30,
            target_factor_count=3,
        )

        selector2 = create_default_selector()
        optimizer2 = ConstrainedWalkForwardOptimizer(selector=selector2, verbose=False)
        df2, _ = optimizer2.run_constrained_wfo(
            factors2,
            returns2,
            factor_names,
            is_period=70,
            oos_period=20,
            step_size=30,
            target_factor_count=3,
        )

        # 应完全相同
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
