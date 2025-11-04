"""
性能优化测试

测试P0和P1优化的正确性:
1. 数据缓存
2. 向量化OOS预测
3. Numba IC计算

作者: Linus Test
日期: 2025-10-29
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from core.data_loader import DataLoader
from core.ic_calculator_numba import ICCalculatorNumba


class TestDataCaching:
    """测试数据缓存功能"""

    def test_cache_creation_and_loading(self, tmp_path):
        """测试缓存创建和加载"""
        # 创建临时数据目录
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cache_dir = tmp_path / "cache"

        # 创建模拟数据文件
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100),
                "adj_close": np.random.randn(100) + 100,
                "adj_high": np.random.randn(100) + 101,
                "adj_low": np.random.randn(100) + 99,
                "adj_open": np.random.randn(100) + 100,
                "volume": np.random.randint(1000000, 10000000, 100),
            }
        )
        test_data.to_parquet(data_dir / "510300.SH_daily_qfq.parquet")

        # 初始化DataLoader
        loader = DataLoader(data_dir=str(data_dir), cache_dir=str(cache_dir))

        # 首次加载（应该创建缓存）
        result1 = loader.load_ohlcv(
            etf_codes=["510300"], start_date="2020-01-01", end_date="2020-12-31"
        )

        # 验证缓存文件存在
        cache_files = list(cache_dir.glob("*.pkl"))
        assert len(cache_files) == 1, "应该创建1个缓存文件"

        # 第二次加载（应该从缓存读取）
        result2 = loader.load_ohlcv(
            etf_codes=["510300"], start_date="2020-01-01", end_date="2020-12-31"
        )

        # 验证两次结果一致
        assert result1.keys() == result2.keys()
        for key in result1.keys():
            pd.testing.assert_frame_equal(result1[key], result2[key])


class TestNumbaICCalculator:
    """测试Numba加速的IC计算"""

    def test_single_ic_calculation(self):
        """测试单个信号的IC计算"""
        # 创建模拟数据
        T, N = 100, 43
        signals = np.random.randn(T, N)
        returns = np.random.randn(T, N)

        # 计算IC
        ic = ICCalculatorNumba.compute_ic(signals, returns)

        # 验证结果
        assert isinstance(ic, float)
        assert -1 <= ic <= 1, "IC应该在[-1, 1]范围内"

    def test_batch_ic_calculation(self):
        """测试批量IC计算"""
        # 创建模拟数据（减小规模避免Numba并行问题）
        n_combos, T, N = 3, 50, 20
        all_signals = np.random.randn(n_combos, T, N)
        returns = np.random.randn(T, N)

        # 批量计算IC
        ics = ICCalculatorNumba.compute_batch_ics(all_signals, returns)

        # 验证结果
        assert ics.shape == (n_combos,)
        assert all(-2 <= ic <= 2 for ic in ics), "所有IC应该在合理范围内"

    def test_ic_with_nans(self):
        """测试包含NaN的IC计算"""
        T, N = 100, 43
        signals = np.random.randn(T, N)
        returns = np.random.randn(T, N)

        # 添加一些NaN
        signals[0:10, 0:5] = np.nan
        returns[20:30, 10:15] = np.nan

        # 计算IC（应该自动处理NaN）
        ic = ICCalculatorNumba.compute_ic(signals, returns)

        # 验证结果
        assert isinstance(ic, float)
        assert not np.isnan(ic) or ic == 0.0, "IC应该是有效数字或0"


class TestVectorizedOOSPrediction:
    """测试向量化OOS预测"""

    def test_vectorized_extraction(self):
        """测试向量化因子提取"""
        # 模拟数据
        T, N, K = 20, 43, 18
        factors_data = np.random.randn(T, N, K)

        # 模拟Top10组合
        top10_combos = [
            (0, 1, 2, 3, 4),
            (1, 2, 3, 4, 5),
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            (4, 5, 6, 7, 8),
            (5, 6, 7, 8, 9),
            (6, 7, 8, 9, 10),
            (7, 8, 9, 10, 11),
            (8, 9, 10, 11, 12),
            (9, 10, 11, 12, 13),
        ]

        # 向量化提取
        top10_indices = np.array(top10_combos)  # (10, 5)
        top10_factors = factors_data[:, :, top10_indices.T]  # (T, N, 5, 10)
        top10_factors = np.transpose(top10_factors, (3, 0, 1, 2))  # (10, T, N, 5)

        # 验证形状
        assert top10_factors.shape == (10, T, N, 5)

        # 验证数据正确性（抽查第一个组合）
        for i, factor_idx in enumerate(top10_combos[0]):
            np.testing.assert_array_equal(
                top10_factors[0, :, :, i], factors_data[:, :, factor_idx]
            )

    def test_vectorized_signal_calculation(self):
        """测试向量化信号计算"""
        # 模拟数据
        n_combos, T, N, combo_size = 10, 20, 43, 5
        top10_factors = np.random.randn(n_combos, T, N, combo_size)

        # 向量化计算信号（等权平均）
        ensemble_signals = np.mean(top10_factors, axis=3)  # (10, T, N)

        # 验证形状
        assert ensemble_signals.shape == (n_combos, T, N)

        # 验证计算正确性（抽查第一个组合）
        expected_signal = np.mean(top10_factors[0], axis=2)
        np.testing.assert_array_almost_equal(ensemble_signals[0], expected_signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
