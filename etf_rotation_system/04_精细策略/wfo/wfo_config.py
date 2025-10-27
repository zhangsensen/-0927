#!/usr/bin/env python3
"""
WFO配置模块
定义WFO的时间窗口、参数等配置
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class WFOPeriod:
    """单个WFO周期"""

    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # 结果存储
    is_results: Optional[dict] = None  # In-Sample结果
    oos_results: Optional[dict] = None  # Out-of-Sample结果
    best_strategy: Optional[dict] = None  # IS期最优策略

    def __post_init__(self):
        """验证时间窗口合法性"""
        if self.train_end >= self.test_start:
            raise ValueError(f"训练窗口结束时间必须早于测试窗口开始时间")

        if self.train_start >= self.train_end:
            raise ValueError(f"训练窗口开始时间必须早于结束时间")

        if self.test_start >= self.test_end:
            raise ValueError(f"测试窗口开始时间必须早于结束时间")

    @property
    def train_days(self) -> int:
        """训练窗口天数"""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """测试窗口天数"""
        return (self.test_end - self.test_start).days

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "period_id": self.period_id,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "train_days": self.train_days,
            "test_days": self.test_days,
        }


@dataclass
class WFOConfig:
    """WFO配置"""

    # 时间窗口配置
    train_window_months: int = 12  # 训练窗口：12个月
    test_window_months: int = 3  # 测试窗口：3个月
    step_months: int = 3  # 步进：3个月

    # 数据范围
    data_start_date: Optional[datetime] = None
    data_end_date: Optional[datetime] = None

    # 优化配置
    top_n_strategies: int = 10  # IS期选择Top-N策略进入OOS
    min_is_sharpe: float = 0.4  # IS期最小Sharpe要求

    # 过拟合检测阈值
    max_overfit_ratio: float = 1.5  # 最大IS/OOS Sharpe比
    max_decay_rate: float = 0.25  # 最大性能衰减率
    min_oos_sharpe: float = 0.3  # OOS最小Sharpe

    # 分析配置
    enable_detailed_analysis: bool = True
    save_period_results: bool = True

    # 并行配置
    n_workers: int = 8
    chunk_size: int = 10

    # 随机种子
    random_seed: Optional[int] = 42

    def __post_init__(self):
        """验证配置合法性"""
        if self.train_window_months <= 0:
            raise ValueError("训练窗口必须大于0")

        if self.test_window_months <= 0:
            raise ValueError("测试窗口必须大于0")

        if self.step_months <= 0:
            raise ValueError("步进必须大于0")

        if self.top_n_strategies <= 0:
            raise ValueError("top_n_strategies必须大于0")

    def estimate_periods(self) -> int:
        """估算WFO周期数"""
        if self.data_start_date is None or self.data_end_date is None:
            return 0

        total_months = (self.data_end_date.year - self.data_start_date.year) * 12 + (
            self.data_end_date.month - self.data_start_date.month
        )

        required_months = self.train_window_months + self.test_window_months

        if total_months < required_months:
            return 0

        # 计算可以滚动的次数
        available_months = total_months - required_months
        n_periods = available_months // self.step_months + 1

        return max(0, n_periods)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "train_window_months": self.train_window_months,
            "test_window_months": self.test_window_months,
            "step_months": self.step_months,
            "data_start_date": (
                self.data_start_date.strftime("%Y-%m-%d")
                if self.data_start_date
                else None
            ),
            "data_end_date": (
                self.data_end_date.strftime("%Y-%m-%d") if self.data_end_date else None
            ),
            "top_n_strategies": self.top_n_strategies,
            "min_is_sharpe": self.min_is_sharpe,
            "max_overfit_ratio": self.max_overfit_ratio,
            "max_decay_rate": self.max_decay_rate,
            "min_oos_sharpe": self.min_oos_sharpe,
            "estimated_periods": self.estimate_periods(),
        }


def add_months(date: datetime, months: int) -> datetime:
    """增加月份（处理跨年）"""
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(
        date.day,
        [
            31,
            29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ][month - 1],
    )
    return datetime(year, month, day)
