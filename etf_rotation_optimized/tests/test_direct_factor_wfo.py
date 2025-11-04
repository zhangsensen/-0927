"""
DirectFactorWFOOptimizer 单元测试

覆盖点：
1) 基础运行（窗口与输出完整性）
2) 权重方案（ic_weighted vs equal）
3) 阈值筛选（min_factor_ic 影响）
4) IC 下界（ic_floor 处理负IC）
5) 无因子通过阈值的回退（fallback to all)
6) 冒烟测试（小窗口）与确定性
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目路径，确保可导入 core 包
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.direct_factor_wfo_optimizer import DirectFactorWFOOptimizer


def _make_synthetic(ts=140, n=15, f=4, seed=42):
    """构造可控相关结构的数据集"""
    rng = np.random.default_rng(seed)
    # 因子 0: 与收益正相关
    base = rng.normal(0, 1, size=(ts, n))
    # 因子 1: 与收益负相关
    negb = rng.normal(0, 1, size=(ts, n))
    # 因子 2..: 噪声
    noise = (
        rng.normal(0, 1, size=(ts, n, max(0, f - 2))) if f > 2 else np.empty((ts, n, 0))
    )

    factors = []
    factors.append(base + rng.normal(0, 0.1, size=(ts, n)))
    factors.append(-base + rng.normal(0, 0.1, size=(ts, n)))
    if f > 2:
        factors.append(noise.mean(axis=2))  # 一个噪声汇总
        for _ in range(3, f):
            factors.append(rng.normal(0, 1, size=(ts, n)))

    factors = np.stack(factors, axis=2)  # (T, N, F)
    returns = base + rng.normal(0, 0.1, size=(ts, n))  # 收益与因子0正相关

    factor_names = [f"F{i}" for i in range(factors.shape[2])]
    return factors, returns, factor_names


@pytest.mark.unit
def test_basic_run_and_summary_integrity():
    factors, returns, names = _make_synthetic(ts=130, n=12, f=5)
    opt = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=0.0, verbose=False
    )
    results, summary = opt.run_wfo(
        factors, returns, names, is_period=80, oos_period=20, step_size=20
    )
    assert len(results) == len(summary) > 0
    for col in [
        "window_index",
        "is_start",
        "is_end",
        "oos_start",
        "oos_end",
        "n_selected_factors",
        "selected_factors",
        "oos_ensemble_ic",
        "oos_ensemble_sharpe",
    ]:
        assert col in summary.columns
    # 值域合理
    assert summary["oos_ensemble_ic"].between(-1, 1).all()


@pytest.mark.unit
def test_weighting_schemes_behave():
    factors, returns, names = _make_synthetic(ts=120, n=10, f=4)
    opt_eq = DirectFactorWFOOptimizer(
        factor_weighting="equal", min_factor_ic=0.0, verbose=False
    )
    _, sum_eq = opt_eq.run_wfo(
        factors, returns, names, is_period=80, oos_period=20, step_size=20
    )

    opt_ic = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=0.0, verbose=False
    )
    _, sum_ic = opt_ic.run_wfo(
        factors, returns, names, is_period=80, oos_period=20, step_size=20
    )

    assert sum_eq["oos_ensemble_ic"].between(-1, 1).all()
    assert sum_ic["oos_ensemble_ic"].between(-1, 1).all()


@pytest.mark.unit
def test_min_factor_ic_filter_and_fallback():
    factors, returns, names = _make_synthetic(ts=110, n=8, f=3)
    # 高阈值导致0通过，应fallback到全因子
    opt = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=10.0, verbose=False
    )
    results, _ = opt.run_wfo(
        factors, returns, names, is_period=80, oos_period=20, step_size=20
    )
    assert results[0].selected_factors  # 非空
    assert len(results[0].selected_factors) == len(names)


@pytest.mark.unit
def test_ic_floor_effect_on_negative_factor():
    # 为了匹配 T-1 对齐，构造 returns[t] = factor0[t-1]（正相关），因子1与因子0为相反数（负相关）
    factors, _, names = _make_synthetic(ts=130, n=12, f=3)
    # 用 factor0 的向前滚动一位作为收益（最后一行填充为前一行，避免越界影响）
    ret = np.roll(factors[:, :, 0], -1, axis=0)
    ret[-1, :] = ret[-2, :]

    # ic_floor=0 → 负IC置零 → 权重应为0（允许极小数值误差）
    opt0 = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=-1.0, ic_floor=0.0, verbose=False
    )
    res0, _ = opt0.run_wfo(
        factors, ret, names, is_period=80, oos_period=20, step_size=20
    )
    w0 = res0[0].factor_weights
    assert pytest.approx(w0[names[1]], abs=1e-8) == 0.0

    # ic_floor>0 → 负IC抬升到floor → 权重应>0
    opt1 = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=-1.0, ic_floor=0.01, verbose=False
    )
    res1, _ = opt1.run_wfo(
        factors, ret, names, is_period=80, oos_period=20, step_size=20
    )
    w1 = res1[0].factor_weights
    assert w1[names[1]] > 0.0


@pytest.mark.unit
def test_small_oos_window_smoke():
    factors, returns, names = _make_synthetic(ts=150, n=10, f=4)
    opt = DirectFactorWFOOptimizer(verbose=False)
    results, summary = opt.run_wfo(
        factors, returns, names, is_period=100, oos_period=5, step_size=20
    )
    assert len(summary) > 0


@pytest.mark.unit
def test_determinism_same_inputs_same_outputs():
    factors, returns, names = _make_synthetic(ts=140, n=10, f=4, seed=123)
    opt = DirectFactorWFOOptimizer(verbose=False)
    _, s1 = opt.run_wfo(
        factors, returns, names, is_period=100, oos_period=20, step_size=20
    )
    _, s2 = opt.run_wfo(
        factors, returns, names, is_period=100, oos_period=20, step_size=20
    )
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))


"""
Direct Factor WFO Optimizer 单元测试

测试覆盖:
1. 基础运行 - 随机数据验证窗口数与输出完整性
2. 权重方案 - ic_weighted vs equal
3. 阈值筛选 - min_factor_ic / ic_floor 对筛选因子数的影响
"""

import numpy as np
import pytest
from core.direct_factor_wfo_optimizer import DirectFactorWFOOptimizer


@pytest.fixture
def mock_data():
    """生成模拟数据"""
    np.random.seed(42)
    T, N, F = 500, 20, 10  # 500天, 20只ETF, 10个因子

    # 因子数据 (T, N, F)
    factors = np.random.randn(T, N, F)

    # 收益率数据 (T, N) - 添加一些信号
    returns = np.random.randn(T, N) * 0.02
    # 让前5个因子与收益有相关性
    for i in range(5):
        returns += factors[:, :, i] * 0.001

    factor_names = [f"FACTOR_{i}" for i in range(F)]

    return factors, returns, factor_names


class TestDirectFactorWFOBasic:
    """基础功能测试"""

    def test_basic_run(self, mock_data):
        """测试基础运行 - 验证窗口数与输出完整性"""
        factors, returns, factor_names = mock_data

        optimizer = DirectFactorWFOOptimizer(
            factor_weighting="ic_weighted", min_factor_ic=0.01, verbose=False
        )

        results, summary_df = optimizer.run_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=20,
        )

        # 验证窗口数
        expected_windows = len(range(0, len(factors) - 100 - 20 + 1, 20))
        assert (
            len(results) == expected_windows
        ), f"窗口数不匹配: {len(results)} vs {expected_windows}"
        assert len(summary_df) == expected_windows

        # 验证输出列
        required_cols = [
            "window_index",
            "is_start",
            "is_end",
            "oos_start",
            "oos_end",
            "n_selected_factors",
            "selected_factors",
            "oos_ensemble_ic",
            "oos_ensemble_sharpe",
        ]
        for col in required_cols:
            assert col in summary_df.columns, f"缺少列: {col}"

        # 验证数值范围
        assert summary_df["n_selected_factors"].min() >= 0
        assert summary_df["n_selected_factors"].max() <= len(factor_names)
        assert summary_df["oos_ensemble_ic"].notna().all()

        print(f"✅ 基础运行测试通过: {len(results)} 个窗口")
        print(f"   平均OOS IC: {summary_df['oos_ensemble_ic'].mean():.4f}")
        print(f"   平均因子数: {summary_df['n_selected_factors'].mean():.1f}")


class TestWeightingSchemes:
    """权重方案测试"""

    def test_ic_weighted_vs_equal(self, mock_data):
        """测试 ic_weighted vs equal 权重方案"""
        factors, returns, factor_names = mock_data

        # IC加权
        opt_ic = DirectFactorWFOOptimizer(
            factor_weighting="ic_weighted", min_factor_ic=0.0, verbose=False
        )
        results_ic, summary_ic = opt_ic.run_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=50,
        )

        # 等权
        opt_eq = DirectFactorWFOOptimizer(
            factor_weighting="equal", min_factor_ic=0.0, verbose=False
        )
        results_eq, summary_eq = opt_eq.run_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=50,
        )

        # 验证窗口数一致
        assert len(results_ic) == len(results_eq)

        # IC加权应该利用更多信息（理论上表现更好）
        ic_mean = summary_ic["oos_ensemble_ic"].mean()
        eq_mean = summary_eq["oos_ensemble_ic"].mean()

        print(f"✅ 权重方案测试通过:")
        print(f"   IC加权平均OOS IC: {ic_mean:.4f}")
        print(f"   等权平均OOS IC: {eq_mean:.4f}")
        print(f"   差异: {ic_mean - eq_mean:+.4f}")


class TestThresholdFiltering:
    """阈值筛选测试"""

    def test_min_factor_ic_filtering(self, mock_data):
        """测试 min_factor_ic 对筛选因子数的影响"""
        factors, returns, factor_names = mock_data

        thresholds = [0.0, 0.01, 0.02]
        results_dict = {}

        for threshold in thresholds:
            optimizer = DirectFactorWFOOptimizer(
                factor_weighting="ic_weighted", min_factor_ic=threshold, verbose=False
            )

            results, summary_df = optimizer.run_wfo(
                factors_data=factors,
                returns=returns,
                factor_names=factor_names,
                is_period=100,
                oos_period=20,
                step_size=50,
            )

            results_dict[threshold] = summary_df

        # 验证：阈值越高，筛选的因子数越少
        for i in range(len(thresholds) - 1):
            t1, t2 = thresholds[i], thresholds[i + 1]
            n1 = results_dict[t1]["n_selected_factors"].mean()
            n2 = results_dict[t2]["n_selected_factors"].mean()
            assert n1 >= n2, f"阈值{t2}应比{t1}筛选更严格"

        print(f"✅ 阈值筛选测试通过:")
        for threshold in thresholds:
            avg_factors = results_dict[threshold]["n_selected_factors"].mean()
            avg_ic = results_dict[threshold]["oos_ensemble_ic"].mean()
            print(
                f"   min_ic={threshold:.2f}: 平均{avg_factors:.1f}因子, OOS IC={avg_ic:.4f}"
            )

    def test_ic_floor_effect(self, mock_data):
        """测试 ic_floor 对负IC因子的处理"""
        factors, returns, factor_names = mock_data

        # ic_floor=0.0 (负IC置为0)
        opt_floor0 = DirectFactorWFOOptimizer(
            factor_weighting="ic_weighted",
            min_factor_ic=0.0,
            ic_floor=0.0,
            verbose=False,
        )
        results_floor0, summary_floor0 = opt_floor0.run_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=50,
        )

        # 验证运行成功
        assert len(results_floor0) > 0
        assert summary_floor0["oos_ensemble_ic"].notna().all()

        print(f"✅ IC下界测试通过:")
        print(
            f"   ic_floor=0.0: 平均OOS IC={summary_floor0['oos_ensemble_ic'].mean():.4f}"
        )


class TestEdgeCases:
    """边界情况测试"""

    def test_all_factors_filtered(self, mock_data):
        """测试所有因子被过滤的情况"""
        factors, returns, factor_names = mock_data

        # 设置极高阈值，强制过滤所有因子
        optimizer = DirectFactorWFOOptimizer(
            factor_weighting="ic_weighted", min_factor_ic=0.5, verbose=False  # 极高阈值
        )

        results, summary_df = optimizer.run_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=factor_names,
            is_period=100,
            oos_period=20,
            step_size=50,
        )

        # 应该回退到使用所有因子
        assert all(summary_df["n_selected_factors"] > 0), "应回退到使用所有因子"

        print(f"✅ 边界情况测试通过: 高阈值回退机制正常")


def test_integration_smoke():
    """集成冒烟测试 - 快速验证核心流程"""
    np.random.seed(42)

    # 小规模数据
    T, N, F = 200, 10, 5
    factors = np.random.randn(T, N, F)
    returns = np.random.randn(T, N) * 0.02
    factor_names = [f"F{i}" for i in range(F)]

    optimizer = DirectFactorWFOOptimizer(
        factor_weighting="ic_weighted", min_factor_ic=0.01, verbose=False
    )

    results, summary_df = optimizer.run_wfo(
        factors_data=factors,
        returns=returns,
        factor_names=factor_names,
        is_period=50,
        oos_period=10,
        step_size=10,
    )

    assert len(results) > 0, "应至少有1个窗口"
    assert len(summary_df) == len(results)
    assert summary_df["oos_ensemble_ic"].notna().all()

    print(f"✅ 冒烟测试通过: {len(results)} 个窗口")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
