"""
WFO并行枚举器测试

测试：
- 并行计算正确性
- 增量计算功能
- Parquet存储
- 中断恢复
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from core.wfo_multi_strategy_selector import StrategySpec
from core.wfo_parallel_enumerator import WFOParallelEnumerator


@pytest.fixture
def mock_data():
    """模拟数据"""
    T, N, K = 100, 10, 5
    factors = np.random.randn(T, N, K)
    returns = np.random.randn(T, N) * 0.01
    factor_names = [f"F{i}" for i in range(K)]
    dates = pd.date_range("2020-01-01", periods=T, freq="D")

    # 模拟WFO结果
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        selected_factors: list
        factor_weights: dict
        oos_start: int
        oos_end: int

    results_list = [
        MockResult(
            selected_factors=["F0", "F1", "F2"],
            factor_weights={"F0": 0.5, "F1": 0.3, "F2": 0.2},
            oos_start=50,
            oos_end=100,
        )
    ]

    return factors, returns, factor_names, dates, results_list


def test_parallel_enumeration(mock_data):
    """测试并行枚举"""
    factors, returns, factor_names, dates, results_list = mock_data

    specs = [
        StrategySpec(factors=("F0", "F1"), tau=1.0, top_n=3),
        StrategySpec(factors=("F1", "F2"), tau=1.0, top_n=3),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        enumerator = WFOParallelEnumerator(
            n_workers=2,
            chunk_size=1,
            use_parquet=True,
            enable_incremental=False,
        )

        df = enumerator.enumerate_strategies(
            specs=specs,
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir,
            dates=dates,
        )

        # 验证结果
        assert len(df) == 2
        assert "factors" in df.columns
        assert "sharpe_ratio" in df.columns

        # 验证Parquet文件
        parquet_file = out_dir / "strategies_ranked.parquet"
        assert parquet_file.exists()


def test_incremental_computation(mock_data):
    """测试增量计算"""
    factors, returns, factor_names, dates, results_list = mock_data

    specs = [
        StrategySpec(factors=("F0", "F1"), tau=1.0, top_n=3),
        StrategySpec(factors=("F1", "F2"), tau=1.0, top_n=3),
        StrategySpec(factors=("F2", "F3"), tau=1.0, top_n=3),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        enumerator = WFOParallelEnumerator(
            n_workers=1,
            chunk_size=1,
            use_parquet=True,
            enable_incremental=True,
        )

        # 第一次运行：计算前2个
        df1 = enumerator.enumerate_strategies(
            specs=specs[:2],
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir,
            dates=dates,
        )
        assert len(df1) == 2

        # 第二次运行：增量计算第3个
        df2 = enumerator.enumerate_strategies(
            specs=specs,
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir,
            dates=dates,
        )
        assert len(df2) == 3  # 应该包含全部3个


def test_parquet_compression(mock_data):
    """测试Parquet压缩"""
    factors, returns, factor_names, dates, results_list = mock_data

    specs = [StrategySpec(factors=("F0", "F1"), tau=1.0, top_n=3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Parquet模式
        enumerator_parquet = WFOParallelEnumerator(
            n_workers=1,
            use_parquet=True,
            enable_incremental=False,
        )
        enumerator_parquet.enumerate_strategies(
            specs=specs,
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir,
            dates=dates,
        )

        parquet_file = out_dir / "strategies_ranked.parquet"
        parquet_size = parquet_file.stat().st_size

        # CSV模式
        out_dir2 = Path(tmpdir) / "csv"
        out_dir2.mkdir()
        enumerator_csv = WFOParallelEnumerator(
            n_workers=1,
            use_parquet=False,
            enable_incremental=False,
        )
        enumerator_csv.enumerate_strategies(
            specs=specs,
            results_list=results_list,
            factors=factors,
            returns=returns,
            factor_names=factor_names,
            out_dir=out_dir2,
            dates=dates,
        )

        csv_file = out_dir2 / "strategies_ranked.csv"
        csv_size = csv_file.stat().st_size

        # Parquet应该更小
        assert parquet_size < csv_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
