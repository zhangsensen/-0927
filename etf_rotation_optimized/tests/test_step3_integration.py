"""
Step3 Integration Test | 集成测试

快速验证step3_ensemble_wfo的完整流程
使用小数据集测试，确保没有bug
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from scripts.step3_ensemble_wfo import (
    load_constraints_config,
    prepare_wfo_data,
    run_ensemble_wfo,
)


class TestStep3Integration:
    """Step3集成测试"""

    @pytest.fixture
    def mock_factor_data(self, tmp_path):
        """创建模拟因子数据"""
        np.random.seed(42)

        # 创建OHLCV数据
        T, N = 150, 10  # 150天, 10个ETF
        dates = pd.date_range("2024-01-01", periods=T, freq="D")
        etf_codes = [f"ETF_{i:02d}" for i in range(N)]

        # OHLCV - 逐列创建DataFrame
        ohlcv_dict = {}
        for col in ["Open", "High", "Low", "Close"]:
            ohlcv_dict[col] = pd.DataFrame(
                np.random.randn(T, N) * 0.01 + 100, index=dates, columns=etf_codes
            )
        
        ohlcv_dict["Volume"] = pd.DataFrame(
            np.random.randint(1e6, 1e7, (T, N)), index=dates, columns=etf_codes
        )
        
        ohlcv_dict["RET_1D"] = pd.DataFrame(
            np.random.randn(T, N) * 0.02, index=dates, columns=etf_codes
        )

        # 合并为MultiIndex DataFrame
        ohlcv_data = pd.concat(ohlcv_dict, axis=1)

        # 创建18个因子
        factor_names = [
            "MOM_20D",
            "MOM_63D",
            "SLOPE_20D",
            "SLOPE_60D",
            "RSI_14",
            "MAX_DD_60D",
            "VOL_RATIO_20D",
            "RET_VOL_20D",
            "SHARPE_RATIO_20D",
            "SHARPE_RATIO_60D",
            "OBV_SLOPE_10D",
            "CMF_20D",
            "ADX_14",
            "ATR_14",
            "MOM_120D",
            "MAX_DD_120D",
            "DOWNSIDE_VOL_20D",
            "CALMAR_RATIO",
        ]

        factors_dict = {}
        for factor_name in factor_names:
            # 因子与收益有一些相关性
            factor_data = np.random.randn(T, N) * 0.05
            if "MOM" in factor_name or "SLOPE" in factor_name:
                # 动量因子与未来收益正相关
                factor_data += ohlcv_data["RET_1D"].values * 0.3

            factors_dict[factor_name] = pd.DataFrame(
                factor_data, index=dates, columns=etf_codes
            )

        # 保存到临时目录
        factor_dir = tmp_path / "factor_selection" / "20250128" / "20250128_120000"
        standardized_dir = factor_dir / "standardized"
        standardized_dir.mkdir(parents=True, exist_ok=True)

        # 保存OHLCV
        ohlcv_data.to_parquet(standardized_dir / "OHLCV.parquet")

        # 保存因子
        for factor_name, factor_df in factors_dict.items():
            factor_df.to_parquet(standardized_dir / f"{factor_name}.parquet")

        # 保存元数据
        import json

        metadata = {
            "n_factors": len(factor_names),
            "n_dates": T,
            "n_etfs": N,
            "factor_names": factor_names,
        }
        with open(factor_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return factor_dir, factors_dict

    def test_load_constraints_config(self):
        """测试约束配置加载"""
        import logging

        logger = logging.getLogger(__name__)

        config = load_constraints_config(logger)

        # 验证配置结构
        assert "family_quotas" in config
        assert "mutually_exclusive_pairs" in config

        print(f"✓ 约束配置加载成功:")
        print(f"  - 家族配额: {len(config['family_quotas'])} 个")
        print(f"  - 互斥对: {len(config['mutually_exclusive_pairs'])} 对")

    def test_prepare_wfo_data(self, mock_factor_data):
        """测试WFO数据准备"""
        import logging

        logger = logging.getLogger(__name__)

        factor_dir, factors_dict = mock_factor_data

        # 加载OHLCV
        ohlcv_data = pd.read_parquet(factor_dir / "standardized" / "OHLCV.parquet")

        # 准备WFO数据
        factors_array, returns_array, factor_names = prepare_wfo_data(
            ohlcv_data, factors_dict, logger
        )

        # 验证形状
        T, N, K = factors_array.shape
        assert T == 150  # 时间步
        assert N == 10  # 资产数
        assert K == 18  # 因子数

        assert returns_array.shape == (T, N)
        assert len(factor_names) == K

        print(f"✓ WFO数据准备成功:")
        print(f"  - 因子数组: {factors_array.shape}")
        print(f"  - 收益数组: {returns_array.shape}")
        print(f"  - 因子名称: {len(factor_names)} 个")

    def test_run_ensemble_wfo_quick(self, mock_factor_data, tmp_path):
        """快速测试Ensemble WFO运行"""
        import logging

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        factor_dir, factors_dict = mock_factor_data

        # 加载数据
        ohlcv_data = pd.read_parquet(factor_dir / "standardized" / "OHLCV.parquet")
        factors_array, returns_array, factor_names = prepare_wfo_data(
            ohlcv_data, factors_dict, logger
        )

        # 加载约束
        constraints_config = load_constraints_config(logger)

        # 运行WFO (小参数快速测试)
        output_dir = tmp_path / "ensemble_wfo_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df = run_ensemble_wfo(
            factors_array=factors_array,
            returns_array=returns_array,
            factor_names=factor_names,
            constraints_config=constraints_config,
            output_dir=output_dir,
            logger=logger,
            n_samples=50,  # 小样本快速测试
            combo_size=5,
            top_k=5,
            weighting_scheme="gradient_decay",
            is_period=60,  # 小窗口
            oos_period=20,
            step_size=20,
        )

        # 验证结果
        assert len(summary_df) > 0  # 至少1个窗口
        assert "oos_ensemble_ic" in summary_df.columns
        assert "oos_ensemble_sharpe" in summary_df.columns

        # 验证文件保存
        assert (output_dir / "ensemble_wfo_summary.csv").exists()
        assert (output_dir / "ensemble_wfo_detailed.json").exists()

        print(f"✓ Ensemble WFO快速测试完成:")
        print(f"  - 窗口数: {len(summary_df)}")
        print(f"  - 平均OOS IC: {summary_df['oos_ensemble_ic'].mean():.4f}")
        print(f"  - 平均OOS Sharpe: {summary_df['oos_ensemble_sharpe'].mean():.2f}")
        print(f"\n汇总:\n{summary_df}")

    def test_full_step3_pipeline(self, mock_factor_data, tmp_path):
        """完整pipeline测试 (模拟真实运行)"""
        import logging

        factor_dir, _ = mock_factor_data

        # 设置日志
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # 模拟step3_ensemble_wfo.main()的流程
        from scripts.step3_ensemble_wfo import (
            load_factor_selection_data,
            generate_summary_report,
        )

        # 1. 加载数据
        ohlcv_data, factors_dict, metadata = load_factor_selection_data(
            factor_dir, logger
        )

        assert len(factors_dict) == 18
        assert ohlcv_data.shape[0] == 150

        # 2. 准备WFO数据
        factors_array, returns_array, factor_names = prepare_wfo_data(
            ohlcv_data, factors_dict, logger
        )

        # 3. 加载约束
        constraints_config = load_constraints_config(logger)

        # 4. 运行WFO
        output_dir = tmp_path / "full_pipeline_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df = run_ensemble_wfo(
            factors_array=factors_array,
            returns_array=returns_array,
            factor_names=factor_names,
            constraints_config=constraints_config,
            output_dir=output_dir,
            logger=logger,
            n_samples=50,
            combo_size=5,
            top_k=5,
        )

        # 5. 生成报告
        generate_summary_report(summary_df, output_dir, logger)

        # 验证所有输出文件
        assert (output_dir / "ensemble_wfo_summary.csv").exists()
        assert (output_dir / "ensemble_wfo_detailed.json").exists()
        assert (output_dir / "performance_stats.json").exists()

        # 验证性能统计
        import json

        with open(output_dir / "performance_stats.json") as f:
            stats = json.load(f)

        assert "total_windows" in stats
        assert "mean_oos_ic" in stats
        assert "positive_ic_ratio" in stats

        print(f"✓ 完整Pipeline测试通过:")
        print(f"  - 总窗口数: {stats['total_windows']}")
        print(f"  - 平均OOS IC: {stats['mean_oos_ic']:.4f}")
        print(f"  - 正IC比率: {stats['positive_ic_ratio']:.1%}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
