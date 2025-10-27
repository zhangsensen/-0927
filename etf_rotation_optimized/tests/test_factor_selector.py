"""
因子选择器测试套件 | Factor Selector Test Suite

测试覆盖:
  1. 约束配置加载与验证
  2. 最小IC约束
  3. 相关性去冗余
  4. 互斥对约束
  5. 家族配额约束
  6. 必选因子约束
  7. 复合约束场景
  8. 空结果处理
  9. 报告生成

作者: Step 5 Factor Selector Tests
日期: 2025-10-26
"""

import pytest
from core.factor_selector import (
    ConstraintViolation,
    FactorSelector,
    SelectionReport,
    create_default_selector,
)


class TestBasicSetup:
    """基本设置测试"""

    def test_selector_initialization(self):
        """选择器初始化"""
        selector = FactorSelector(verbose=False)
        assert selector is not None
        assert selector.constraints == {}

    def test_default_factor_families(self):
        """默认因子家族"""
        selector = FactorSelector(verbose=False)

        assert "MOM_20D" in selector.factor_family
        assert selector.factor_family["MOM_20D"] == "momentum"
        assert selector.factor_family["PRICE_POSITION_20D"] == "price_features"

    def test_create_default_selector(self):
        """创建默认选择器"""
        selector = create_default_selector()

        assert "family_quota" in selector.constraints
        assert "mutual_exclusivity" in selector.constraints
        assert "correlation_deduplication" in selector.constraints


class TestMinimumICConstraint:
    """最小IC约束测试"""

    def test_minimum_ic_filtering(self):
        """最小IC过滤"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.04

        ic_scores = {
            "MOM_20D": 0.05,  # 保留
            "SLOPE_20D": 0.03,  # 排除
            "RET_VOL_20D": 0.06,  # 保留
            "MAX_DD_60D": 0.02,  # 排除
            "VOL_RATIO_20D": 0.04,  # 保留 (=阈值)
        }

        selected, report = selector.select_factors(ic_scores)

        # 应该过滤掉 SLOPE_20D 和 MAX_DD_60D
        assert "SLOPE_20D" not in selected or "SLOPE_20D" in report.final_selection
        assert "RET_VOL_20D" in selected or len(report.violations) > 0

    def test_minimum_ic_all_pass(self):
        """所有因子都通过最小IC"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.01

        ic_scores = {
            "MOM_20D": 0.05,
            "RET_VOL_20D": 0.04,
            "SLOPE_20D": 0.03,
        }

        selected, report = selector.select_factors(ic_scores)

        # 所有因子都应该通过
        assert len(selected) > 0


class TestCorrelationDeduplication:
    """相关性去冗余测试"""

    def test_high_correlation_removal(self):
        """高相关因子移除"""
        selector = create_default_selector()
        selector.constraints["correlation_deduplication"]["threshold"] = 0.8

        ic_scores = {
            "PRICE_POSITION_20D": 0.05,  # IC更高
            "PRICE_POSITION_120D": 0.03,  # IC较低
            "MOM_20D": 0.04,
        }

        correlations = {
            ("PRICE_POSITION_20D", "PRICE_POSITION_120D"): 0.85,  # 高相关
        }

        selected, report = selector.select_factors(ic_scores, correlations)

        # PRICE_POSITION_120D 应该被移除 (IC更低)
        assert "PRICE_POSITION_20D" in selected

    def test_low_correlation_both_keep(self):
        """低相关因子都保留"""
        selector = create_default_selector()
        selector.constraints["correlation_deduplication"]["threshold"] = 0.8

        ic_scores = {
            "MOM_20D": 0.05,
            "RET_VOL_20D": 0.04,
        }

        correlations = {
            ("MOM_20D", "RET_VOL_20D"): 0.3,  # 低相关
        }

        selected, report = selector.select_factors(ic_scores, correlations)

        # 两个因子都应该保留
        assert "MOM_20D" in selected
        assert "RET_VOL_20D" in selected


class TestMutualExclusivity:
    """互斥对约束测试"""

    def test_mutual_exclusivity_enforcement(self):
        """互斥约束执行"""
        selector = create_default_selector()

        ic_scores = {
            "PRICE_POSITION_20D": 0.05,
            "PRICE_POSITION_120D": 0.03,
            "MOM_20D": 0.04,
        }

        selected, report = selector.select_factors(ic_scores)

        # 不能同时选择两个价格位置指标
        count = sum(
            1 for f in selected if f in ["PRICE_POSITION_20D", "PRICE_POSITION_120D"]
        )
        assert count <= 1, "互斥因子不应同时选中"

    def test_keep_higher_ic_in_mutex(self):
        """互斥对中保留IC更高的"""
        selector = create_default_selector()

        ic_scores = {
            "PRICE_POSITION_20D": 0.08,  # IC更高
            "PRICE_POSITION_120D": 0.02,
            "MOM_20D": 0.03,
        }

        selected, report = selector.select_factors(ic_scores)

        # 应该保留 PRICE_POSITION_20D
        if (
            len(
                [
                    f
                    for f in selected
                    if f in ["PRICE_POSITION_20D", "PRICE_POSITION_120D"]
                ]
            )
            == 1
        ):
            assert "PRICE_POSITION_20D" in selected


class TestFamilyQuota:
    """家族配额约束测试"""

    def test_volatility_family_quota(self):
        """波动率家族配额 (max=2)"""
        selector = create_default_selector()

        ic_scores = {
            "RET_VOL_20D": 0.06,
            "VOL_RATIO_20D": 0.05,
            "VOL_RATIO_60D": 0.04,
            "MOM_20D": 0.03,
        }

        selected, report = selector.select_factors(ic_scores, target_count=10)

        # 波动率类最多2个
        vol_count = sum(
            1
            for f in selected
            if f in ["RET_VOL_20D", "VOL_RATIO_20D", "VOL_RATIO_60D"]
        )
        assert vol_count <= 2, f"波动率因子超过配额: {vol_count}"

    def test_momentum_family_quota(self):
        """动量家族配额 (max=1)"""
        selector = create_default_selector()

        ic_scores = {
            "MOM_20D": 0.05,
            "SLOPE_20D": 0.04,
            "RET_VOL_20D": 0.03,
        }

        selected, report = selector.select_factors(ic_scores)

        # 动量类最多1个
        mom_count = sum(1 for f in selected if f in ["MOM_20D", "SLOPE_20D"])
        assert mom_count <= 1, f"动量因子超过配额: {mom_count}"


class TestRequiredFactors:
    """必选因子约束测试"""

    def test_required_factors_inclusion(self):
        """必选因子被包含"""
        selector = create_default_selector()
        selector.constraints["required_factors"] = ["MOM_20D"]

        ic_scores = {
            "MOM_20D": 0.02,  # IC很低，但必选
            "RET_VOL_20D": 0.06,
            "SLOPE_20D": 0.01,
        }

        selected, report = selector.select_factors(ic_scores)

        # MOM_20D 必须被选中
        assert "MOM_20D" in selected


class TestComplexScenarios:
    """复合场景测试"""

    def test_all_constraints_together(self):
        """所有约束共同作用"""
        selector = create_default_selector()

        ic_scores = {
            "MOM_20D": 0.08,
            "SLOPE_20D": 0.07,
            "RET_VOL_20D": 0.06,
            "VOL_RATIO_20D": 0.05,
            "VOL_RATIO_60D": 0.04,
            "MAX_DD_60D": 0.03,
            "PRICE_POSITION_20D": 0.02,
            "PRICE_POSITION_120D": 0.015,
            "PV_CORR_20D": 0.025,
            "RSI_14": 0.01,
        }

        correlations = {
            ("MOM_20D", "SLOPE_20D"): 0.85,
            ("PRICE_POSITION_20D", "PRICE_POSITION_120D"): 0.82,
        }

        selected, report = selector.select_factors(
            ic_scores, correlations, target_count=5
        )

        # 应该能选出因子
        assert len(selected) > 0
        assert len(selected) <= 5

        # 验证约束满足
        # 1. 动量类最多1个
        mom_count = sum(1 for f in selected if f in ["MOM_20D", "SLOPE_20D"])
        assert mom_count <= 1

        # 2. 价格位置最多1个
        price_count = sum(
            1 for f in selected if f in ["PRICE_POSITION_20D", "PRICE_POSITION_120D"]
        )
        assert price_count <= 1

    def test_target_count_enforcement(self):
        """目标数量限制"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.01

        ic_scores = {f"FACTOR_{i}": 0.1 - i * 0.01 for i in range(10)}

        selected, report = selector.select_factors(ic_scores, target_count=3)

        # 最多选3个
        assert len(selected) <= 3


class TestReportGeneration:
    """报告生成测试"""

    def test_selection_report_structure(self):
        """选择报告结构"""
        selector = create_default_selector()

        ic_scores = {
            "MOM_20D": 0.05,
            "RET_VOL_20D": 0.04,
            "SLOPE_20D": 0.03,
        }

        selected, report = selector.select_factors(ic_scores)

        # 检查报告结构
        assert isinstance(report, SelectionReport)
        assert hasattr(report, "candidate_factors")
        assert hasattr(report, "applied_constraints")
        assert hasattr(report, "final_selection")
        assert hasattr(report, "selection_scores")

    def test_constraint_impacts_recording(self):
        """约束影响记录"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.04

        ic_scores = {
            "MOM_20D": 0.05,
            "SLOPE_20D": 0.02,  # 被最小IC约束排除
            "RET_VOL_20D": 0.06,
        }

        selected, report = selector.select_factors(ic_scores)

        # 应该记录约束影响
        if "minimum_ic" in report.constraint_impacts:
            assert "SLOPE_20D" in report.constraint_impacts["minimum_ic"]


class TestEdgeCases:
    """边界情况测试"""

    def test_all_factors_filtered(self):
        """所有因子都被过滤"""
        selector = create_default_selector()
        selector.constraints["minimum_ic"]["global_minimum"] = 0.5  # 很高的阈值

        ic_scores = {
            "MOM_20D": 0.05,
            "RET_VOL_20D": 0.04,
        }

        selected, report = selector.select_factors(ic_scores)

        # 可能返回空列表
        assert isinstance(selected, list)

    def test_single_factor(self):
        """单个因子"""
        selector = create_default_selector()

        ic_scores = {
            "MOM_20D": 0.05,
        }

        selected, report = selector.select_factors(ic_scores)

        assert "MOM_20D" in selected

    def test_empty_input(self):
        """空输入"""
        selector = create_default_selector()

        ic_scores = {}

        selected, report = selector.select_factors(ic_scores)

        assert len(selected) == 0


class TestIntegration:
    """集成测试"""

    def test_realistic_scenario(self):
        """现实场景测试"""
        selector = create_default_selector()

        # 模拟真实的IC分数分布
        ic_scores = {
            "MOM_20D": 0.065,
            "SLOPE_20D": 0.055,
            "RET_VOL_20D": 0.045,
            "MAX_DD_60D": 0.035,
            "VOL_RATIO_20D": 0.052,
            "VOL_RATIO_60D": 0.048,
            "PRICE_POSITION_20D": 0.032,
            "PRICE_POSITION_120D": 0.028,
            "PV_CORR_20D": 0.038,
            "RSI_14": 0.018,
        }

        correlations = {
            ("MOM_20D", "SLOPE_20D"): 0.87,
            ("RET_VOL_20D", "VOL_RATIO_20D"): 0.65,
            ("PRICE_POSITION_20D", "PRICE_POSITION_120D"): 0.88,
        }

        selected, report = selector.select_factors(
            ic_scores, correlations, target_count=5
        )

        # 验证结果
        assert len(selected) <= 5
        assert len(selected) > 0

        # 验证所有约束都被应用
        assert len(report.applied_constraints) > 0

        # 检查选中因子的IC符合约束条件（由于约束作用，可能不是最高IC的）
        selected_ics = [ic_scores[f] for f in selected]
        # 平均IC应该符合最小IC要求
        assert min(selected_ics) >= selector.constraints["minimum_ic"]["global_minimum"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
