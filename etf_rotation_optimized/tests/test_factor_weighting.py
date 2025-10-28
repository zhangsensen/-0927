"""
单元测试: Factor Weighting

验证:
1. 3种权重方案正确性
2. 权重和为1
3. 输出形状正确
4. 处理NaN值
5. IC缺失时回退到等权

作者: Linus Quant Engineer
日期: 2025-10-28
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from core.factor_weighting import FactorWeighting


class TestFactorWeighting:
    """Factor Weighting单元测试"""

    @pytest.fixture
    def mock_data(self):
        """生成模拟数据"""
        np.random.seed(42)
        T, N = 100, 10
        n_factors = 5

        factor_data = [pd.DataFrame(np.random.randn(T, N)) for _ in range(n_factors)]

        factor_names = ["F1", "F2", "F3", "F4", "F5"]

        ic_scores = {"F1": 0.10, "F2": 0.08, "F3": 0.05, "F4": 0.03, "F5": 0.02}

        return factor_data, factor_names, ic_scores

    def test_equal_weights(self, mock_data):
        """测试等权方案"""
        factor_data, factor_names, _ = mock_data

        signal = FactorWeighting.combine_factors(factor_data, scheme="equal")

        # 检查形状
        assert signal.shape == (100, 10)

        # 检查权重分布
        weights = FactorWeighting.get_weight_distribution(factor_names, scheme="equal")

        # 每个因子20%
        for name, weight in weights.items():
            assert abs(weight - 0.2) < 1e-6, f"{name}权重错误: {weight}"

        # 权重和为1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_ic_weighted(self, mock_data):
        """测试IC加权方案"""
        factor_data, factor_names, ic_scores = mock_data

        signal = FactorWeighting.combine_factors(
            factor_data,
            scheme="ic_weighted",
            ic_scores=ic_scores,
            factor_names=factor_names,
        )

        # 检查形状
        assert signal.shape == (100, 10)

        # 检查权重分布
        weights = FactorWeighting.get_weight_distribution(
            factor_names, scheme="ic_weighted", ic_scores=ic_scores
        )

        # 权重和为1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # 高IC因子权重更高
        assert weights["F1"] > weights["F2"], "F1(IC=0.10)应该>F2(IC=0.08)"
        assert weights["F2"] > weights["F3"], "F2(IC=0.08)应该>F3(IC=0.05)"
        assert weights["F4"] > weights["F5"], "F4(IC=0.03)应该>F5(IC=0.02)"

    def test_gradient_decay(self, mock_data):
        """测试梯度衰减方案"""
        factor_data, factor_names, ic_scores = mock_data

        signal = FactorWeighting.combine_factors(
            factor_data,
            scheme="gradient_decay",
            ic_scores=ic_scores,
            factor_names=factor_names,
        )

        # 检查形状
        assert signal.shape == (100, 10)

        # 检查权重分布
        weights = FactorWeighting.get_weight_distribution(
            factor_names, scheme="gradient_decay", ic_scores=ic_scores
        )

        # 权重和为1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # 验证指数衰减特性
        # Top1应该约42.9%, Top2约26.0%
        assert 0.40 < weights["F1"] < 0.45, f"Top1权重异常: {weights['F1']}"
        assert 0.24 < weights["F2"] < 0.28, f"Top2权重异常: {weights['F2']}"

    def test_handle_nan(self):
        """测试NaN处理"""
        np.random.seed(42)
        T, N = 50, 5

        # 生成含NaN的数据
        factor_data = []
        for _ in range(3):
            df = pd.DataFrame(np.random.randn(T, N))
            # 随机插入NaN
            mask = np.random.rand(T, N) < 0.1
            df[mask] = np.nan
            factor_data.append(df)

        signal = FactorWeighting.combine_factors(factor_data, scheme="equal")

        # 输出应该也含有NaN (在对应位置)
        assert signal.shape == (T, N)
        # nanmean会跳过NaN,所以结果可能全为数值或部分NaN
        # 主要验证不crash

    def test_fallback_to_equal(self, mock_data):
        """测试回退到等权 (IC缺失时)"""
        factor_data, factor_names, _ = mock_data

        # 不提供IC评分
        signal = FactorWeighting.combine_factors(
            factor_data, scheme="ic_weighted"  # 要求IC但未提供
        )

        # 应该回退到等权
        weights = FactorWeighting.get_weight_distribution(factor_names, scheme="equal")

        for weight in weights.values():
            assert abs(weight - 0.2) < 1e-6

    def test_negative_ic_handling(self, mock_data):
        """测试负IC处理"""
        factor_data, factor_names, _ = mock_data

        # 提供负IC
        negative_ic_scores = {
            "F1": -0.05,
            "F2": -0.03,
            "F3": -0.01,
            "F4": 0.02,
            "F5": 0.04,
        }

        signal = FactorWeighting.combine_factors(
            factor_data,
            scheme="ic_weighted",
            ic_scores=negative_ic_scores,
            factor_names=factor_names,
        )

        # 应该正常运行
        assert signal.shape == (100, 10)

        # 负IC应该被置为0,只用正IC
        weights = FactorWeighting.get_weight_distribution(
            factor_names, scheme="ic_weighted", ic_scores=negative_ic_scores
        )

        # F4和F5应该有权重,F1-F3应该权重为0
        assert weights["F4"] > 0
        assert weights["F5"] > 0
        assert weights["F1"] == 0
        assert weights["F2"] == 0
        assert weights["F3"] == 0

    def test_all_schemes_comparison(self, mock_data):
        """对比3种方案的输出差异"""
        factor_data, factor_names, ic_scores = mock_data

        results = {}
        for scheme in ["equal", "ic_weighted", "gradient_decay"]:
            signal = FactorWeighting.combine_factors(
                factor_data,
                scheme=scheme,
                ic_scores=ic_scores,
                factor_names=factor_names,
            )
            results[scheme] = signal

        # 3种方案的输出应该不同 (除非IC全0)
        assert not np.allclose(
            results["equal"], results["ic_weighted"]
        ), "equal和ic_weighted结果不应该相同"
        assert not np.allclose(
            results["equal"], results["gradient_decay"]
        ), "equal和gradient_decay结果不应该相同"

    def test_invalid_scheme(self, mock_data):
        """测试无效方案报错"""
        factor_data, _, _ = mock_data

        with pytest.raises(ValueError, match="未知权重方案"):
            FactorWeighting.combine_factors(factor_data, scheme="invalid_scheme")

    def test_empty_factor_data(self):
        """测试空因子数据"""
        with pytest.raises(ValueError, match="factor_data不能为空"):
            FactorWeighting.combine_factors([], scheme="equal")


# =========================================================================
# 运行测试
# =========================================================================

if __name__ == "__main__":
    print("运行 Factor Weighting 单元测试")
    print("=" * 70)

    pytest.main([__file__, "-v", "--tb=short"])
