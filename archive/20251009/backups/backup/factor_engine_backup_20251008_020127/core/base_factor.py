"""因子基类定义"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FactorMetadata:
    """因子元数据"""

    factor_id: str
    version: str
    category: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]


class BaseFactor(ABC):
    """
    因子基类

    所有因子实现必须继承此类并实现calculate方法
    """

    # 子类必须定义这些属性
    factor_id: str = ""
    version: str = "v1.0"
    category: str = ""
    description: str = ""

    def __init__(self, **params):
        """
        初始化因子

        Args:
            **params: 因子参数
        """
        self.params = params
        self.dependencies: List[str] = []

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子值

        Args:
            data: 包含OHLCV的DataFrame
                  columns: ['open', 'high', 'low', 'close', 'volume']
                  index: DatetimeIndex or MultiIndex(timestamp, symbol)

        Returns:
            Series with factor values, same index as input
        """
        pass

    def get_metadata(self) -> FactorMetadata:
        """获取因子元数据"""
        return FactorMetadata(
            factor_id=self.factor_id,
            version=self.version,
            category=self.category,
            description=self.description,
            parameters=self.params,
            dependencies=self.dependencies,
        )

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据

        Args:
            data: 待验证的DataFrame

        Returns:
            是否通过验证
        """
        # 检查数据是否为空
        if data.empty:
            return False

        # 检查必需列
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(data.columns):
            return False

        # 检查必需列的数据类型
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False

        # 检查NaN比例（允许最多30%的NaN）
        for col in required_columns:
            nan_ratio = data[col].isna().sum() / len(data)
            if nan_ratio > 0.3:
                return False

        # 检查数据长度（至少需要10个有效数据点，适合短期指标计算）
        valid_data_count = data[list(required_columns)].dropna().shape[0]
        if valid_data_count < 10:
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.factor_id}, v{self.version})"
