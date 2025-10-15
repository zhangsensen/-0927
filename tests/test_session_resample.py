"""
A股会话感知重采样 - 回归测试

验证每日K线数是否符合预期，防止未来改动导致回归
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider


@pytest.fixture
def provider():
    """创建数据提供者"""
    raw_dir = Path(__file__).parent.parent / "raw"
    if not raw_dir.exists():
        pytest.skip("raw数据目录不存在")
    return ParquetDataProvider(raw_data_dir=raw_dir)


@pytest.fixture
def test_symbol():
    """测试用股票代码"""
    return "000600.SZ"


@pytest.fixture
def date_range():
    """测试日期范围"""
    return datetime(2024, 8, 23), datetime(2025, 8, 22)


class TestSessionResample:
    """会话感知重采样测试"""

    @pytest.mark.parametrize(
        "timeframe,expected_bars",
        [
            ("60min", 4),
            ("30min", 8),
            ("15min", 16),
            ("5min", 48),
        ],
    )
    def test_daily_bar_count(
        self, provider, test_symbol, date_range, timeframe, expected_bars
    ):
        """测试每日K线数是否符合预期"""
        start_date, end_date = date_range

        # 加载1min基础数据
        market = provider._detect_market(test_symbol)
        market_dir = provider.market_dirs[market]
        base_data = provider._load_timeframe_file(
            test_symbol, market_dir, "1min", start_date, end_date
        )

        if base_data.empty:
            pytest.skip(f"无{test_symbol}的1min数据")

        # 重采样
        resampled = provider._resample_to_timeframe(base_data, timeframe, test_symbol)

        # 验证每日K线数
        daily_counts = resampled.groupby(resampled.index.date).size()
        avg_bars = daily_counts.mean()

        assert (
            abs(avg_bars - expected_bars) < 0.1
        ), f"{timeframe}: 平均每日{avg_bars:.1f}根，期望{expected_bars}根"

    @pytest.mark.parametrize("timeframe", ["60min", "30min", "15min", "5min"])
    def test_no_invalid_timestamps(self, provider, test_symbol, date_range, timeframe):
        """测试是否存在非法时间戳（午休、盘后）"""
        start_date, end_date = date_range

        market = provider._detect_market(test_symbol)
        market_dir = provider.market_dirs[market]
        base_data = provider._load_timeframe_file(
            test_symbol, market_dir, "1min", start_date, end_date
        )

        if base_data.empty:
            pytest.skip(f"无{test_symbol}的1min数据")

        resampled = provider._resample_to_timeframe(base_data, timeframe, test_symbol)

        # 检查非法时间戳
        def is_invalid(ts):
            h, m = ts.hour, ts.minute
            # 午休（不含11:30和13:00）
            if (h == 11 and m > 30) or (h == 12):
                return True
            # 盘后
            if h == 15 and m > 0:
                return True
            if h > 15:
                return True
            return False

        invalid_stamps = [ts for ts in resampled.index if is_invalid(ts)]

        assert (
            len(invalid_stamps) == 0
        ), f"{timeframe}: 发现{len(invalid_stamps)}个非法时间戳: {invalid_stamps[:5]}"

    def test_60min_specific_times(self, provider, test_symbol, date_range):
        """测试60min的具体时间点（10:30, 11:30, 14:00, 15:00）"""
        start_date, end_date = date_range

        market = provider._detect_market(test_symbol)
        market_dir = provider.market_dirs[market]
        base_data = provider._load_timeframe_file(
            test_symbol, market_dir, "1min", start_date, end_date
        )

        if base_data.empty:
            pytest.skip(f"无{test_symbol}的1min数据")

        resampled = provider._resample_to_timeframe(base_data, "60min", test_symbol)

        # 检查第一天的时间点
        first_day = resampled.index.min().date()
        first_day_data = resampled[resampled.index.date == first_day]

        times = [(ts.hour, ts.minute) for ts in first_day_data.index]
        expected_times = [(10, 30), (11, 30), (14, 0), (15, 0)]

        assert (
            times == expected_times
        ), f"60min第一天时间点: {times}, 期望: {expected_times}"

    def test_no_cross_lunch_bars(self, provider, test_symbol, date_range):
        """测试是否存在跨午休的K线"""
        start_date, end_date = date_range

        market = provider._detect_market(test_symbol)
        market_dir = provider.market_dirs[market]
        base_data = provider._load_timeframe_file(
            test_symbol, market_dir, "1min", start_date, end_date
        )

        if base_data.empty:
            pytest.skip(f"无{test_symbol}的1min数据")

        # 60min最容易出现跨午休问题
        resampled = provider._resample_to_timeframe(base_data, "60min", test_symbol)

        # 检查是否有12:00-13:00之间的时间戳
        lunch_stamps = [
            ts
            for ts in resampled.index
            if ts.hour == 12 or (ts.hour == 11 and ts.minute > 30)
        ]

        assert len(lunch_stamps) == 0, f"发现跨午休时间戳: {lunch_stamps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
