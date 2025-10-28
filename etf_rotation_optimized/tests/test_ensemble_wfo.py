"""
Ensemble WFO Optimizer 单元测试

测试集成优化器的核心功能:
1. 单窗口优化流程
2. 批量组合评估
3. Top10选择逻辑
4. OOS集成预测
5. 完整WFO流程
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from core.ensemble_wfo_optimizer import EnsembleWFOOptimizer


# 测试配置
MOCK_CONSTRAINTS = {
    "family_quotas": {
        "momentum_trend": {
            "max_count": 2,
            "candidates": [
                "MOM_20D",
                "SLOPE_20D",
                "MOM_63D",
                "MOM_120D",
                "SLOPE_60D",
                "ADX_14",
            ],
        },
        "volatility_risk": {
            "max_count": 2,
            "candidates": [
                "RET_VOL_20D",
                "MAX_DD_60D",
                "VOL_RATIO_20D",
                "MAX_DD_120D",
                "ATR_14",
                "DOWNSIDE_VOL_20D",
            ],
        },
        "volume_liquidity": {"max_count": 1, "candidates": ["CMF_20D", "OBV_SLOPE_10D"]},
    },
    "mutually_exclusive_pairs": [
        {"pair": ["MOM_20D", "MOM_63D"]},
        {"pair": ["RET_VOL_20D", "VOL_RATIO_20D"]},
        {"pair": ["MOM_63D", "MOM_120D"]},
    ],
}

FACTOR_NAMES = [
    # 动量+趋势 (6个)
    "MOM_20D",
    "SLOPE_20D",
    "MOM_63D",
    "MOM_120D",
    "SLOPE_60D",
    "ADX_14",
    # 波动+风险 (6个)
    "RET_VOL_20D",
    "MAX_DD_60D",
    "VOL_RATIO_20D",
    "MAX_DD_120D",
    "ATR_14",
    "DOWNSIDE_VOL_20D",
    # 成交量 (2个)
    "CMF_20D",
    "OBV_SLOPE_10D",
    # 质量因子 (4个)
    "RSI_14",
    "SHARPE_RATIO_20D",
    "SHARPE_RATIO_60D",
    "CALMAR_RATIO",
]


class TestEnsembleWFOOptimizer:
    """Ensemble WFO Optimizer 单元测试"""

    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        np.random.seed(42)

        # 小规模数据: 150天 × 10资产 × 18因子
        T, N, K = 150, 10, 18

        # 生成因子数据 (加入一些真实相关性)
        factors = np.random.randn(T, N, K) * 0.05

        # 生成收益数据 (与前5个因子有正相关)
        base_return = np.mean(factors[:, :, :5], axis=2) * 2
        noise = np.random.randn(T, N) * 0.02
        returns = base_return + noise

        return factors, returns

    def test_initialization(self):
        """测试初始化"""
        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=100,
            combo_size=5,
            top_k=10,
            random_seed=42,
            verbose=False,
        )

        assert optimizer.n_samples == 100
        assert optimizer.combo_size == 5
        assert optimizer.top_k == 10
        assert optimizer.random_seed == 42
        assert len(optimizer.window_results) == 0

    def test_batch_evaluate_combos(self, mock_data):
        """测试批量组合评估"""
        factors, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS, verbose=False
        )

        # 创建测试组合
        test_combos = [
            ("MOM_20D", "SLOPE_20D", "MAX_DD_60D", "CMF_20D", "RSI_14"),
            ("MOM_63D", "SLOPE_60D", "RET_VOL_20D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D"),
            ("MOM_120D", "ADX_14", "VOL_RATIO_20D", "CMF_20D", "CALMAR_RATIO"),
        ]

        # 使用前100天作为IS数据
        is_factors = factors[:100]
        is_returns = returns[:100]

        # 批量评估
        combo_ics = optimizer._batch_evaluate_combos(
            test_combos, is_factors, is_returns, FACTOR_NAMES
        )

        # 验证
        assert len(combo_ics) == 3
        assert all(isinstance(ic, float) for ic in combo_ics)
        assert all(-1 <= ic <= 1 for ic in combo_ics)  # IC范围合理

        print(f"✓ 批量评估3个组合: IC = {combo_ics}")

    def test_compute_signal_ic(self, mock_data):
        """测试信号IC计算"""
        _, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS, verbose=False
        )

        # 创建测试信号 (与returns正相关)
        signal = returns + np.random.randn(*returns.shape) * 0.01

        ic = optimizer._compute_signal_ic(signal, returns)

        # 验证
        assert isinstance(ic, float)
        assert 0.5 <= ic <= 1.0  # 应该有较高正相关

        print(f"✓ 信号IC计算: {ic:.4f}")

    def test_single_window_optimization(self, mock_data):
        """测试单窗口优化流程"""
        factors, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=50,  # 小样本快速测试
            combo_size=5,
            top_k=5,
            verbose=False,
        )

        # 运行单窗口
        window_result = optimizer._run_single_window(
            factors_data=factors,
            returns=returns,
            factor_names=FACTOR_NAMES,
            is_start=10,
            is_end=100,
            oos_start=100,
            oos_end=120,
            window_idx=0,
        )

        # 验证结果
        assert window_result.window_index == 0
        assert window_result.n_sampled_combos >= 30  # 至少30个有效组合
        assert len(window_result.top10_combos) == 5  # Top5
        assert len(window_result.top10_is_ics) == 5
        assert isinstance(window_result.oos_ensemble_ic, float)
        assert isinstance(window_result.oos_ensemble_sharpe, float)

        # IC范围合理
        assert -1 <= window_result.oos_ensemble_ic <= 1

        print(f"✓ 单窗口优化完成:")
        print(f"  - 采样组合数: {window_result.n_sampled_combos}")
        print(f"  - Top5 IS IC: {np.mean(window_result.top10_is_ics):.4f}")
        print(f"  - OOS集成IC: {window_result.oos_ensemble_ic:.4f}")
        print(f"  - OOS Sharpe: {window_result.oos_ensemble_sharpe:.2f}")

    def test_complete_wfo_run(self, mock_data):
        """测试完整WFO运行"""
        factors, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=50,
            combo_size=5,
            top_k=5,
            weighting_scheme="gradient_decay",
            random_seed=42,
            verbose=False,
        )

        # 运行WFO (小窗口快速测试)
        summary_df = optimizer.run_ensemble_wfo(
            factors_data=factors,
            returns=returns,
            factor_names=FACTOR_NAMES,
            is_period=60,
            oos_period=20,
            step_size=20,
        )

        # 验证汇总结果
        assert len(summary_df) > 0  # 至少1个窗口
        assert "oos_ensemble_ic" in summary_df.columns
        assert "oos_ensemble_sharpe" in summary_df.columns
        assert "top10_mean_is_ic" in summary_df.columns

        # 验证窗口结果
        assert len(optimizer.window_results) == len(summary_df)

        # OOS IC应该合理
        mean_oos_ic = summary_df["oos_ensemble_ic"].mean()
        assert -1 <= mean_oos_ic <= 1

        print(f"✓ 完整WFO运行完成:")
        print(f"  - 窗口数: {len(summary_df)}")
        print(f"  - 平均OOS IC: {mean_oos_ic:.4f}")
        print(f"  - 平均OOS Sharpe: {summary_df['oos_ensemble_sharpe'].mean():.2f}")
        print(f"\n汇总报告:\n{summary_df.head()}")

    def test_top10_selection_logic(self, mock_data):
        """测试Top10选择逻辑"""
        factors, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=100,
            top_k=10,
            verbose=False,
        )

        # 运行单窗口
        window_result = optimizer._run_single_window(
            factors_data=factors,
            returns=returns,
            factor_names=FACTOR_NAMES,
            is_start=10,
            is_end=100,
            oos_start=100,
            oos_end=120,
            window_idx=0,
        )

        # 验证Top10排序
        top10_ics = window_result.top10_is_ics

        # Top10 IC应该递减
        for i in range(len(top10_ics) - 1):
            assert (
                top10_ics[i] >= top10_ics[i + 1]
            ), f"Top{i+1} IC应该 >= Top{i+2} IC"

        print(f"✓ Top10选择逻辑正确:")
        print(f"  - Top10 IS IC: {top10_ics}")

    def test_weighting_schemes(self, mock_data):
        """测试不同加权方案"""
        factors, returns = mock_data

        schemes = ["equal", "ic_weighted", "gradient_decay"]
        results = {}

        for scheme in schemes:
            optimizer = EnsembleWFOOptimizer(
                constraints_config=MOCK_CONSTRAINTS,
                n_samples=50,
                top_k=5,
                weighting_scheme=scheme,
                random_seed=42,
                verbose=False,
            )

            window_result = optimizer._run_single_window(
                factors_data=factors,
                returns=returns,
                factor_names=FACTOR_NAMES,
                is_start=10,
                is_end=100,
                oos_start=100,
                oos_end=120,
                window_idx=0,
            )

            results[scheme] = window_result.oos_ensemble_ic

        # 验证不同方案产生不同结果
        print(f"✓ 不同加权方案OOS IC:")
        for scheme, ic in results.items():
            print(f"  - {scheme}: {ic:.4f}")

        # gradient_decay通常表现更稳定 (但不是绝对的)
        assert all(-1 <= ic <= 1 for ic in results.values())

    def test_determinism(self, mock_data):
        """测试确定性 (固定seed)"""
        factors, returns = mock_data

        # 第1次运行
        optimizer1 = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=50,
            random_seed=123,
            verbose=False,
        )

        result1 = optimizer1._run_single_window(
            factors, returns, FACTOR_NAMES, 10, 100, 100, 120, 0
        )

        # 第2次运行 (相同seed)
        optimizer2 = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=50,
            random_seed=123,
            verbose=False,
        )

        result2 = optimizer2._run_single_window(
            factors, returns, FACTOR_NAMES, 10, 100, 100, 120, 0
        )

        # 验证完全一致
        assert result1.top10_combos == result2.top10_combos
        assert result1.top10_is_ics == result2.top10_is_ics
        assert result1.oos_ensemble_ic == result2.oos_ensemble_ic

        print("✓ 确定性验证通过: 相同seed结果完全一致")

    def test_save_results(self, mock_data, tmp_path):
        """测试结果保存"""
        factors, returns = mock_data

        optimizer = EnsembleWFOOptimizer(
            constraints_config=MOCK_CONSTRAINTS,
            n_samples=30,
            verbose=False,
        )

        # 运行WFO
        optimizer.run_ensemble_wfo(
            factors, returns, FACTOR_NAMES, is_period=60, oos_period=20, step_size=20
        )

        # 保存结果
        optimizer.save_results(tmp_path)

        # 验证文件存在
        assert (tmp_path / "ensemble_wfo_summary.csv").exists()
        assert (tmp_path / "ensemble_wfo_detailed.json").exists()

        # 验证CSV可读
        import pandas as pd

        summary_df = pd.read_csv(tmp_path / "ensemble_wfo_summary.csv")
        assert len(summary_df) > 0
        assert "oos_ensemble_ic" in summary_df.columns

        # 验证JSON可读
        import json

        with open(tmp_path / "ensemble_wfo_detailed.json") as f:
            detailed = json.load(f)
        assert len(detailed) > 0
        assert "top10_combos" in detailed[0]

        print(f"✓ 结果保存验证通过:")
        print(f"  - CSV: {tmp_path / 'ensemble_wfo_summary.csv'}")
        print(f"  - JSON: {tmp_path / 'ensemble_wfo_detailed.json'}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
