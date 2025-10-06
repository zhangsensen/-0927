#!/usr/bin/env python3
"""
时间序列安全处理协议 - 架构层防护
作者：量化首席工程师
版本：1.0.0
日期：2025-10-02

功能：
- 定义时间序列处理的安全接口
- 强制防止未来函数使用
- 提供类型安全的时间序列操作
"""

from abc import abstractmethod
from typing import Generic, List, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)
NumericType = TypeVar("NumericType", bound=np.generic)


@runtime_checkable
class TimeSeriesProcessor(Protocol[T]):
    """时间序列处理器协议 - 强制类型安全"""

    @abstractmethod
    def validate_temporal_alignment(
        self, factor_data: pd.Series, return_data: pd.Series, horizon: int
    ) -> bool:
        """验证时间序列对齐"""
        ...

    @abstractmethod
    def calculate_ic_safe(
        self, factor_data: pd.Series, return_data: pd.Series, horizon: int
    ) -> float:
        """安全的IC计算 - 禁止未来函数"""
        ...

    @abstractmethod
    def shift_forward(self, data: T, periods: int) -> T:
        """仅允许向前shift（历史数据）"""
        ...

    def shift_backward(self, data: T, periods: int) -> T:
        """禁止向后shift（未来数据）"""
        raise NotImplementedError("向后shift（未来函数）被禁止使用")


class SafeTimeSeriesProcessor(Generic[T]):
    """安全时间序列处理器基类 - 架构层强制防护"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.operation_log = []

    def shift_backward(self, data: T, periods: int) -> T:
        """显式禁止向后 shift，保持与协议一致"""
        raise NotImplementedError("向后shift（未来函数）被禁止使用")

    def validate_temporal_alignment(
        self, factor_data: pd.Series, return_data: pd.Series, horizon: int
    ) -> bool:
        """验证时间序列对齐"""
        try:
            # 确保时间索引对齐
            if not isinstance(factor_data.index, pd.DatetimeIndex):
                factor_data.index = pd.to_datetime(factor_data.index)
            if not isinstance(return_data.index, pd.DatetimeIndex):
                return_data.index = pd.to_datetime(return_data.index)

            # 检查时间序列完整性
            common_index = factor_data.index.intersection(return_data.index)
            if len(common_index) < max(30, horizon * 2):
                return False

            # 正确的时间对齐：当前因子预测未来收益
            aligned_return = return_data.loc[common_index].shift(horizon)
            aligned_factor = factor_data.loc[common_index]

            # 计算IC
            ic = aligned_factor.corr(aligned_return)
            self.operation_log.append(f"IC计算: horizon={horizon}, ic={ic:.4f}")

            return True

        except Exception as e:
            if self.strict_mode:
                raise ValueError(f"时间序列验证失败: {e}")
            return False

    def calculate_ic_safe(
        self, factor_data: pd.Series, return_data: pd.Series, horizon: int
    ) -> float:
        """安全的IC计算 - 强制防止未来函数"""
        if not self.validate_temporal_alignment(factor_data, return_data, horizon):
            if self.strict_mode:
                raise ValueError("时间序列对齐验证失败")
            return 0.0

        try:
            # 获取对齐的数据
            common_index = factor_data.index.intersection(return_data.index)
            aligned_factor = factor_data.loc[common_index]
            aligned_return = return_data.loc[common_index].shift(
                horizon
            )  # 注意：这里是正数horizon

            # 计算IC
            ic = aligned_factor.corr(aligned_return)

            if pd.isna(ic):
                return 0.0

            return float(ic)

        except Exception as e:
            if self.strict_mode:
                raise ValueError(f"IC计算失败: {e}")
            return 0.0

    def shift_forward(self, data: T, periods: int) -> T:
        """安全的向前shift - 使用历史数据"""
        if periods < 0:
            raise ValueError("不允许负数shift（未来函数）")

        result = data.shift(periods)
        self.operation_log.append(f"安全向前shift: {periods}期")
        return result

    def calculate_forward_returns(
        self, price_data: pd.Series, horizons: List[int]
    ) -> pd.DataFrame:
        """计算多周期未来收益率 - 安全方法"""
        returns_df = pd.DataFrame(index=price_data.index)

        for horizon in horizons:
            if horizon < 0:
                raise ValueError(f"不允许负数horizon: {horizon}")

            # 计算收益率：未来价格 / 当前价格 - 1
            future_price = price_data.shift(-horizon)  # 向前查找未来价格
            forward_return = future_price / price_data - 1

            returns_df[f"return_{horizon}d"] = forward_return
            self.operation_log.append(f"计算{horizon}期前向收益率")

        return returns_df

    def validate_no_future_leakage(self, data: pd.DataFrame) -> bool:
        """验证数据中无未来信息泄露"""
        issues = []

        # 检查列名
        forbidden_patterns = ["future", "lead", "ahead"]
        for pattern in forbidden_patterns:
            cols = [col for col in data.columns if pattern in col.lower()]
            if cols:
                issues.append(f"发现禁用列名模式 '{pattern}': {cols}")

        # 检查数据值
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isna().sum() > len(data) * 0.9:
                issues.append(f"列 {col} 缺失值过多，可能存在问题")

        if issues:
            error_msg = f"数据验证发现问题: {'; '.join(issues)}"
            if self.strict_mode:
                raise ValueError(error_msg)
            self.operation_log.append(f"警告: {error_msg}")
            return False

        self.operation_log.append("数据验证通过：无未来信息泄露")
        return True

    def get_operation_summary(self) -> str:
        """获取操作摘要"""
        if not self.operation_log:
            return "无操作记录"

        summary = ["时间序列处理操作摘要:"]
        summary.extend(f"  {i+1}. {log}" for i, log in enumerate(self.operation_log))
        return "\n".join(summary)


# 类型安全的工厂函数
def create_safe_processor(strict_mode: bool = True) -> SafeTimeSeriesProcessor:
    """创建安全的时间序列处理器"""
    return SafeTimeSeriesProcessor(strict_mode=strict_mode)


# 运行时类型检查装饰器
def validate_time_series_operation(func):
    """时间序列操作验证装饰器"""

    def wrapper(*args, **kwargs):
        # 检查参数中是否有时间序列数据
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                if hasattr(arg, "index") and len(arg) > 0:
                    if not isinstance(arg.index, pd.DatetimeIndex):
                        raise TypeError("时间序列数据必须使用DatetimeIndex")

        result = func(*args, **kwargs)

        # 检查结果
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if hasattr(result, "index") and len(result) > 0:
                if not isinstance(result.index, pd.DatetimeIndex):
                    raise TypeError("返回结果必须使用DatetimeIndex")

        return result

    return wrapper


# 全局安全处理器实例
safe_processor = create_safe_processor(strict_mode=True)


# 便捷函数
@validate_time_series_operation
def safe_ic_calculation(
    factor_data: pd.Series, return_data: pd.Series, horizon: int
) -> float:
    """便捷的IC计算函数"""
    return safe_processor.calculate_ic_safe(factor_data, return_data, horizon)


@validate_time_series_operation
def safe_forward_shift(data: T, periods: int) -> T:
    """便捷的前向shift函数"""
    return safe_processor.shift_forward(data, periods)
