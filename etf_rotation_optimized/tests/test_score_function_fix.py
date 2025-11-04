"""
测试score函数修复

验证：
1. 覆盖率惩罚生效
2. 低覆盖率策略得分下降
3. 高覆盖率策略得分正常
"""

import pytest
from core.wfo_multi_strategy_selector import WFOMultiStrategySelector


def test_score_coverage_penalty():
    """测试覆盖率惩罚"""
    selector = WFOMultiStrategySelector(turnover_penalty=0.1)

    kpi = {
        "sharpe_ratio": 1.0,
        "annual_return": 0.2,
        "calmar_ratio": 1.0,
    }

    # 测试不同覆盖率
    score_100 = selector._score(kpi, avg_turnover=0.2, coverage=1.0)
    score_70 = selector._score(kpi, avg_turnover=0.2, coverage=0.7)
    score_50 = selector._score(kpi, avg_turnover=0.2, coverage=0.5)
    score_30 = selector._score(kpi, avg_turnover=0.2, coverage=0.3)
    score_10 = selector._score(kpi, avg_turnover=0.2, coverage=0.1)

    print(f"\n覆盖率vs得分:")
    print(f"coverage=1.0: score={score_100:.3f}")
    print(f"coverage=0.7: score={score_70:.3f} (惩罚={(score_100-score_70):.3f})")
    print(f"coverage=0.5: score={score_50:.3f} (惩罚={(score_100-score_50):.3f})")
    print(f"coverage=0.3: score={score_30:.3f} (惩罚={(score_100-score_30):.3f})")
    print(f"coverage=0.1: score={score_10:.3f} (惩罚={(score_100-score_10):.3f})")

    # 验证：覆盖率越低，得分越低
    assert score_100 > score_70 > score_50 > score_30 > score_10

    # 验证：低覆盖率惩罚显著
    assert (score_100 - score_10) > 0.3  # 覆盖率10%时惩罚>0.3


def test_score_vs_old_implementation():
    """对比新旧实现"""
    selector = WFOMultiStrategySelector(turnover_penalty=0.1)

    kpi_high_sharpe = {
        "sharpe_ratio": 1.5,
        "annual_return": 0.3,
        "calmar_ratio": 1.5,
    }

    kpi_normal_sharpe = {
        "sharpe_ratio": 0.8,
        "annual_return": 0.15,
        "calmar_ratio": 0.8,
    }

    # 场景1: 高Sharpe但低覆盖率 vs 正常Sharpe但高覆盖率
    score_high_low_cov = selector._score(
        kpi_high_sharpe, avg_turnover=0.2, coverage=0.1
    )
    score_normal_high_cov = selector._score(
        kpi_normal_sharpe, avg_turnover=0.2, coverage=0.7
    )

    print(f"\n场景对比:")
    print(f"高Sharpe(1.5)+低覆盖率(10%): score={score_high_low_cov:.3f}")
    print(f"正常Sharpe(0.8)+高覆盖率(70%): score={score_normal_high_cov:.3f}")

    # 验证：高覆盖率的正常策略应该得分更高
    assert score_normal_high_cov > score_high_low_cov, "高覆盖率策略应该得分更高"


def test_coverage_penalty_formula():
    """测试覆盖率惩罚公式"""
    selector = WFOMultiStrategySelector(turnover_penalty=0.0)

    kpi = {"sharpe_ratio": 1.0, "annual_return": 0.2, "calmar_ratio": 1.0}

    # 计算基准分（无惩罚）
    base_score = 0.5 * 1.0 + 0.35 * 0.2 + 0.15 * 1.0

    # 测试覆盖率惩罚
    for cov in [1.0, 0.7, 0.5, 0.3, 0.1]:
        score = selector._score(kpi, avg_turnover=0.0, coverage=cov)
        expected_penalty = 0.5 * (1.0 - cov) ** 2
        expected_score = base_score - expected_penalty

        print(
            f"coverage={cov:.1f}: penalty={expected_penalty:.3f}, score={score:.3f}, expected={expected_score:.3f}"
        )

        assert abs(score - expected_score) < 1e-6, f"覆盖率{cov}的得分计算错误"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
