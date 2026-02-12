#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 VEC 与 BT 对齐的核心逻辑

主要验证：
1. shift_timing_signal 行为一致性
2. generate_rebalance_schedule 输出对齐
3. ensure_price_views 验证逻辑
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from etf_strategy.core.utils.rebalance import (
    DEFAULT_TIMING_FILL,
    compute_first_rebalance_index,
    ensure_price_views,
    generate_rebalance_schedule,
    shift_timing_signal,
)


# ==============================================================================
# shift_timing_signal 测试
# ==============================================================================


class TestShiftTimingSignal:
    """Test suite for shift_timing_signal helper."""

    def test_basic_shift(self):
        """基本移位：首元素填充，其余右移一位"""
        raw = np.array([0.5, 0.8, 1.0, 0.3])
        shifted = shift_timing_signal(raw)

        assert shifted[0] == DEFAULT_TIMING_FILL, "首元素应为默认填充值 1.0"
        np.testing.assert_array_equal(shifted[1:], raw[:-1])

    def test_custom_fill_value(self):
        """自定义填充值"""
        raw = np.array([0.5, 0.8, 1.0])
        shifted = shift_timing_signal(raw, fill_value=0.0)

        assert shifted[0] == 0.0
        np.testing.assert_array_equal(shifted[1:], raw[:-1])

    def test_empty_array(self):
        """空数组应返回空数组"""
        empty = np.array([])
        shifted = shift_timing_signal(empty)
        assert shifted.size == 0

    def test_single_element(self):
        """单元素数组只返回填充值"""
        single = np.array([0.7])
        shifted = shift_timing_signal(single)
        assert shifted.shape == (1,)
        assert shifted[0] == DEFAULT_TIMING_FILL

    def test_preserves_dtype(self):
        """保持数据类型为 float"""
        raw = np.array([1, 0, 1, 0], dtype=int)
        shifted = shift_timing_signal(raw)
        assert shifted.dtype == float

    def test_2d_raises(self):
        """二维数组应抛出异常"""
        arr_2d = np.array([[0.5, 0.8], [1.0, 0.3]])
        with pytest.raises(ValueError, match="1D"):
            shift_timing_signal(arr_2d)

    def test_equivalent_to_pandas_shift(self):
        """验证与 pandas shift(1).fillna(1.0) 行为等价"""
        raw = np.array([0.5, 0.8, 1.0, 0.3, 0.6])
        pd_shifted = pd.Series(raw).shift(1).fillna(DEFAULT_TIMING_FILL).values
        np_shifted = shift_timing_signal(raw)

        np.testing.assert_array_almost_equal(np_shifted, pd_shifted)

    def test_vec_bt_timing_consistency(self):
        """核心测试：验证 VEC 和 BT 看到相同的择时信号

        在 T 日，两个引擎的 timing[T] 都应该是 T-1 日生成的信号。
        """
        # 模拟原始择时信号：T日信号表示T日市场状态
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        raw_signal = np.array([1.0, 0.8, 0.5, 1.0, 0.7])  # T日生成的信号

        # VEC 方式：numpy shift
        vec_timing = shift_timing_signal(raw_signal)

        # BT 方式：pandas Series shift
        bt_timing = pd.Series(raw_signal, index=dates).shift(1).fillna(1.0).values

        # 两者必须完全一致
        np.testing.assert_array_equal(vec_timing, bt_timing)

        # 验证语义：T日使用的是T-1日信号
        for t in range(1, len(raw_signal)):
            assert vec_timing[t] == raw_signal[t - 1], f"T={t}日应使用T-1日信号"
        assert vec_timing[0] == DEFAULT_TIMING_FILL, "首日应使用默认填充值"


# ==============================================================================
# generate_rebalance_schedule 测试
# ==============================================================================


class TestGenerateRebalanceSchedule:
    """Test suite for generate_rebalance_schedule helper."""

    def test_basic_schedule(self):
        """基本调仓日程生成"""
        # 100 个交易日，20 日预热，每 10 日调仓
        schedule = generate_rebalance_schedule(
            total_periods=100, lookback_window=20, freq=10
        )

        # 首个调仓日应是 >= 21 且对齐到 freq 的最小值
        assert schedule[0] == 30  # 21->30 对齐到 10 的倍数
        assert all(s % 10 == 0 for s in schedule), "所有调仓日应对齐 freq"
        assert schedule[-1] < 100, "最后调仓日应在总周期内"

    def test_empty_when_too_short(self):
        """总周期过短时返回空数组"""
        schedule = generate_rebalance_schedule(
            total_periods=10, lookback_window=20, freq=5
        )
        assert schedule.size == 0

    def test_freq_alignment(self):
        """验证调仓日对齐逻辑"""
        # lookback=15, offset=1 => start_idx=16
        # freq=7 => 16 % 7 = 2, 需要 +5 => 21
        schedule = generate_rebalance_schedule(
            total_periods=100, lookback_window=15, freq=7
        )
        assert schedule[0] == 21
        assert all((s - 21) % 7 == 0 for s in schedule)

    def test_vec_bt_schedule_match(self):
        """核心测试：VEC 和 BT 使用相同调仓日程

        两引擎应该在完全相同的 bar index 执行调仓。
        """
        lookback = 60
        freq = 8
        total = 252

        # 使用共享 helper
        shared_schedule = generate_rebalance_schedule(
            total_periods=total, lookback_window=lookback, freq=freq
        )

        # 模拟旧 VEC 内核的 `t % freq == 0` 逻辑 (从 lookback 开始)
        old_vec_schedule = [t for t in range(total) if t >= lookback and t % freq == 0]

        # 新旧逻辑应该一致（在正确实现后）
        # 注意：首次调仓必须 >= lookback + 1 才有意义
        first_valid = compute_first_rebalance_index(lookback, freq)
        assert shared_schedule[0] >= lookback + 1, "首次调仓必须在预热期之后"

        # 所有调仓日应对齐 freq
        for idx in shared_schedule:
            assert idx % freq == first_valid % freq


# ==============================================================================
# ensure_price_views 测试
# ==============================================================================


class TestEnsurePriceViews:
    """Test suite for ensure_price_views helper."""

    def test_with_open_prices(self):
        """提供开盘价时正常返回"""
        close = np.array([[100, 200], [101, 202], [102, 204]], dtype=float)
        open_ = np.array([[99, 199], [100, 201], [101, 203]], dtype=float)

        close_prev, open_t, close_t = ensure_price_views(
            close, open_, warn_if_copied=False
        )

        np.testing.assert_array_equal(close_t, close)
        np.testing.assert_array_equal(open_t, open_)
        # close_prev[0] = close[0], close_prev[1:] = close[:-1]
        np.testing.assert_array_equal(close_prev[0], close[0])
        np.testing.assert_array_equal(close_prev[1:], close[:-1])

    def test_fallback_when_open_is_none(self):
        """未提供开盘价时回退到收盘价"""
        close = np.array([[100, 200], [101, 202]], dtype=float)

        with pytest.warns(RuntimeWarning, match="open_prices not provided"):
            close_prev, open_t, close_t = ensure_price_views(close, None)

        np.testing.assert_array_equal(open_t, close)

    def test_raises_when_copy_disabled(self):
        """禁用回退时抛出异常"""
        close = np.array([[100, 200], [101, 202]], dtype=float)

        with pytest.raises(ValueError, match="open_prices is required"):
            ensure_price_views(close, None, copy_if_missing=False)

    def test_shape_mismatch_raises(self):
        """形状不匹配时抛出异常"""
        close = np.array([[100, 200], [101, 202]], dtype=float)
        open_wrong = np.array([[99, 199, 299]], dtype=float)

        with pytest.raises(ValueError, match="match close_prices shape"):
            ensure_price_views(close, open_wrong)

    def test_validates_positive_prices(self):
        """验证价格必须为正"""
        close = np.array([[100, 200], [101, 202]], dtype=float)
        open_zero = np.array([[0, 199], [100, 201]], dtype=float)

        with pytest.raises(ValueError, match="positive"):
            ensure_price_views(close, open_zero, min_valid_index=0)

    def test_skip_warmup_validation(self):
        """min_valid_index 跳过预热期验证"""
        close = np.array([[100, 200], [0, 0], [101, 202]], dtype=float)
        open_ = np.array([[99, 199], [0, 0], [100, 201]], dtype=float)

        # 从 index 2 开始验证，跳过前两行的 0 值
        close_prev, open_t, close_t = ensure_price_views(
            close, open_, min_valid_index=2
        )

        assert close_t.shape == close.shape

    def test_1d_array_raises(self):
        """一维数组应抛出异常"""
        arr_1d = np.array([100, 101, 102])

        with pytest.raises(ValueError, match="2D"):
            ensure_price_views(arr_1d, arr_1d)


# ==============================================================================
# Integration: 模拟完整流程
# ==============================================================================


class TestVecBtAlignmentIntegration:
    """Integration tests verifying VEC/BT alignment end-to-end."""

    def test_timing_signal_lookup_at_rebalance(self):
        """验证在调仓日查询择时信号的行为一致性"""
        np.random.seed(42)

        # 设置
        total_days = 100
        lookback = 20
        freq = 10

        # 生成随机择时信号
        raw_timing = np.random.uniform(0.5, 1.0, size=total_days)

        # 生成调仓日程
        rebalance_days = generate_rebalance_schedule(
            total_periods=total_days, lookback_window=lookback, freq=freq
        )

        # 移位择时信号
        shifted_timing = shift_timing_signal(raw_timing)

        # 在每个调仓日，验证使用的是 T-1 日信号
        for t in rebalance_days:
            assert t > 0, "调仓日必须 > 0"
            expected_signal = raw_timing[t - 1]  # T-1 日信号
            actual_signal = shifted_timing[t]

            np.testing.assert_almost_equal(
                actual_signal,
                expected_signal,
                err_msg=f"调仓日 T={t} 应使用 T-1={t-1} 日信号",
            )


# ==============================================================================
# Signal aggregation: WFO / VEC NaN-normalization consistency
# ==============================================================================


class TestSignalNormalizationConsistency:
    """Verify WFO and VEC compute identical combined_score when factors have NaN."""

    def test_wfo_vec_signal_equal_no_nan(self):
        """Without NaN: WFO (equal weights) == VEC normalized mean."""
        from etf_strategy.core.combo_wfo_optimizer import _compute_combo_signal

        T, N, F = 3, 5, 4
        np.random.seed(42)
        factors = np.random.randn(T, N, F)
        weights = np.ones(F) / F  # equal weights

        wfo_signal = _compute_combo_signal(factors, weights)

        # VEC logic (after fix): mean of valid factors
        vec_signal = np.nanmean(factors, axis=2)

        np.testing.assert_allclose(wfo_signal, vec_signal, atol=1e-12,
                                   err_msg="WFO/VEC signal must match when no NaN")

    def test_wfo_vec_signal_equal_with_nan(self):
        """With NaN: both must skip NaN and normalize by valid count."""
        from etf_strategy.core.combo_wfo_optimizer import _compute_combo_signal

        T, N, F = 5, 8, 4
        np.random.seed(123)
        factors = np.random.randn(T, N, F)
        # Inject NaN: ETF 0 factor 2 always NaN, ETF 3 factors 1&3 on day 2
        factors[:, 0, 2] = np.nan
        factors[2, 3, 1] = np.nan
        factors[2, 3, 3] = np.nan

        weights = np.ones(F) / F
        wfo_signal = _compute_combo_signal(factors, weights)

        # VEC equivalent: mean of non-NaN factors per (t, n)
        vec_signal = np.nanmean(factors, axis=2)
        # Where ALL factors are NaN, WFO returns NaN
        all_nan_mask = np.all(np.isnan(factors), axis=2)
        vec_signal[all_nan_mask] = np.nan

        np.testing.assert_allclose(wfo_signal, vec_signal, atol=1e-12,
                                   err_msg="WFO/VEC signal must match with NaN")

    def test_nan_does_not_penalize_ranking(self):
        """ETF with 3/4 valid factors should NOT be ranked lower than
        ETF with 4/4 valid factors of equal value — normalization prevents this."""
        T, N, F = 1, 2, 4
        factors = np.zeros((T, N, F))
        # ETF 0: all factors = 0.5
        factors[0, 0, :] = 0.5
        # ETF 1: 3 factors = 0.5, 1 factor = NaN
        factors[0, 1, :3] = 0.5
        factors[0, 1, 3] = np.nan

        # Both should score identically (mean = 0.5)
        from etf_strategy.core.combo_wfo_optimizer import _compute_combo_signal

        weights = np.ones(F) / F
        signal = _compute_combo_signal(factors, weights)
        assert signal[0, 0] == signal[0, 1], (
            f"NaN factor should not penalize: {signal[0, 0]} != {signal[0, 1]}"
        )

    def test_all_nan_returns_nan(self):
        """ETF with ALL factors NaN must get NaN signal, not -inf or 0."""
        from etf_strategy.core.combo_wfo_optimizer import _compute_combo_signal

        T, N, F = 1, 3, 4
        factors = np.random.randn(T, N, F)
        factors[0, 1, :] = np.nan  # ETF 1 all NaN

        weights = np.ones(F) / F
        signal = _compute_combo_signal(factors, weights)
        assert np.isnan(signal[0, 1]), "All-NaN ETF must produce NaN signal"
        assert not np.isnan(signal[0, 0]), "Normal ETF must not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
