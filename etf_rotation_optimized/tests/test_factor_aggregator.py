"""
因子汇总器单元测试

测试核心功能:
1. 从WFO结果提取稳定因子
2. 统计因子出现频率
3. 计算因子平均性能

作者: Linus Test
日期: 2025-10-28
"""

import pandas as pd
import pytest
from core.factor_aggregator import FactorAggregator


class TestFactorAggregator:
    """测试因子汇总器"""

    @pytest.fixture
    def mock_wfo_results(self):
        """模拟WFO结果"""
        data = {
            "window_index": [0, 1, 2],
            "oos_ensemble_ic": [0.05, 0.03, 0.04],
            "top10_combos": [
                str(
                    [
                        ("MOM_20D", "VOL_RATIO_20D", "RSI_14"),
                        ("MOM_20D", "SLOPE_20D", "RSI_14"),
                        ("VOL_RATIO_20D", "RSI_14", "ADX_14D"),
                    ]
                    * 3
                    + [("PRICE_POSITION_20D", "RSI_14", "ADX_14D")]
                ),
                str(
                    [
                        ("MOM_20D", "VOL_RATIO_20D", "RSI_14"),
                        ("MOM_20D", "SLOPE_20D", "PV_CORR_20D"),
                        ("VOL_RATIO_20D", "RSI_14", "ADX_14D"),
                    ]
                    * 3
                    + [("PRICE_POSITION_20D", "RSI_14", "CMF_20D")]
                ),
                str(
                    [
                        ("MOM_20D", "VOL_RATIO_20D", "RSI_14"),
                        ("MOM_20D", "SLOPE_20D", "RSI_14"),
                        ("VOL_RATIO_20D", "RSI_14", "ADX_14D"),
                    ]
                    * 3
                    + [("PRICE_POSITION_20D", "RSI_14", "ADX_14D")]
                ),
            ],
        }
        return pd.DataFrame(data)

    def test_aggregate_from_wfo(self, mock_wfo_results):
        """测试基本汇总功能"""
        results = FactorAggregator.aggregate_from_wfo(
            mock_wfo_results, top_n=5, min_frequency=0.1
        )

        # 验证返回结构
        assert "top_factors" in results
        assert "factor_stats" in results
        assert "top_combinations" in results

        # 验证top_factors
        assert isinstance(results["top_factors"], list)
        assert len(results["top_factors"]) > 0

        # 验证factor_stats
        assert isinstance(results["factor_stats"], pd.DataFrame)
        assert "factor" in results["factor_stats"].columns
        assert "frequency" in results["factor_stats"].columns
        assert "avg_oos_ic" in results["factor_stats"].columns

    def test_factor_frequency_calculation(self, mock_wfo_results):
        """测试因子频率计算"""
        results = FactorAggregator.aggregate_from_wfo(
            mock_wfo_results, top_n=10, min_frequency=0.1
        )

        factor_stats = results["factor_stats"]

        # MOM_20D 应该出现频率最高（每个窗口都有）
        mom_stats = factor_stats[factor_stats["factor"] == "MOM_20D"]
        assert len(mom_stats) > 0
        assert mom_stats.iloc[0]["frequency"] > 0.5

        # RSI_14 也应该频繁出现
        rsi_stats = factor_stats[factor_stats["factor"] == "RSI_14"]
        assert len(rsi_stats) > 0
        assert rsi_stats.iloc[0]["frequency"] > 0.5

    def test_min_frequency_filter(self, mock_wfo_results):
        """测试最小频率过滤"""
        # 高频率阈值
        results_high = FactorAggregator.aggregate_from_wfo(
            mock_wfo_results, top_n=10, min_frequency=0.8
        )

        # 低频率阈值
        results_low = FactorAggregator.aggregate_from_wfo(
            mock_wfo_results, top_n=10, min_frequency=0.1
        )

        # 低阈值应该返回更多因子
        assert len(results_low["top_factors"]) >= len(results_high["top_factors"])

    def test_top_combinations_extraction(self, mock_wfo_results):
        """测试最优组合提取"""
        results = FactorAggregator.aggregate_from_wfo(
            mock_wfo_results, top_n=10, min_frequency=0.1
        )

        top_combos = results["top_combinations"]

        # 验证返回组合
        assert isinstance(top_combos, list)
        assert len(top_combos) > 0

        # 每个组合应该是tuple
        for combo in top_combos:
            assert isinstance(combo, tuple)
            assert len(combo) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
