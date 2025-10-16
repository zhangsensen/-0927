#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子注册表
支持动态因子注册和管理，保持现有架构不变
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
from functools import wraps

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


class FactorCategory(Enum):
    """因子类别枚举"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    TREND = "trend"
    OVERLAP = "overlap"
    CANDLESTICK = "candlestick"


@dataclass
class FactorMetadata:
    """因子元数据"""
    factor_id: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: FactorCategory = FactorCategory.OVERLAP
    description: str = ""
    is_dynamic: bool = False
    base_factor_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())


class ETFFactorRegistry:
    """ETF因子注册表"""

    def __init__(self):
        """初始化因子注册表"""
        self._factors: Dict[str, FactorMetadata] = {}
        self._categories: Dict[FactorCategory, List[str]] = {
            category: [] for category in FactorCategory
        }
        self._dynamic_factors: Dict[str, FactorMetadata] = {}

        logger.info("ETF因子注册表初始化完成")

    def register_factor(self,
                       factor_id: str,
                       function: Callable,
                       parameters: Optional[Dict[str, Any]] = None,
                       category: FactorCategory = FactorCategory.OVERLAP,
                       description: str = "",
                       is_dynamic: bool = False,
                       base_factor_id: Optional[str] = None) -> bool:
        """
        注册单个因子

        Args:
            factor_id: 因子ID
            function: 因子计算函数
            parameters: 参数字典
            category: 因子类别
            description: 描述
            is_dynamic: 是否为动态生成的因子
            base_factor_id: 基础因子ID（用于动态因子）

        Returns:
            注册是否成功
        """
        try:
            # 验证函数签名
            self._validate_factor_function(function)

            # 创建元数据
            metadata = FactorMetadata(
                factor_id=factor_id,
                function=function,
                parameters=parameters or {},
                category=category,
                description=description,
                is_dynamic=is_dynamic,
                base_factor_id=base_factor_id
            )

            # 注册因子
            self._factors[factor_id] = metadata

            # 添加到类别索引
            if factor_id not in self._categories[category]:
                self._categories[category].append(factor_id)

            # 如果是动态因子，添加到动态因子索引
            if is_dynamic:
                self._dynamic_factors[factor_id] = metadata

            logger.debug(f"因子注册成功: {factor_id} ({category.value})")
            return True

        except Exception as e:
            logger.error(f"因子注册失败 {factor_id}: {str(e)}")
            return False

    def register_batch_factors(self, factor_specs: List[Dict[str, Any]]) -> int:
        """
        批量注册因子

        Args:
            factor_specs: 因子规格列表

        Returns:
            成功注册的因子数量
        """
        success_count = 0

        for spec in factor_specs:
            try:
                success = self.register_factor(**spec)
                if success:
                    success_count += 1
            except Exception as e:
                logger.error(f"批量注册失败: {str(e)}")
                continue

        logger.info(f"批量注册完成: {success_count}/{len(factor_specs)} 个因子成功")
        return success_count

    def unregister_factor(self, factor_id: str) -> bool:
        """
        注销因子

        Args:
            factor_id: 因子ID

        Returns:
            注销是否成功
        """
        try:
            if factor_id not in self._factors:
                logger.warning(f"因子不存在: {factor_id}")
                return False

            metadata = self._factors[factor_id]

            # 从主注册表移除
            del self._factors[factor_id]

            # 从类别索引移除
            if factor_id in self._categories[metadata.category]:
                self._categories[metadata.category].remove(factor_id)

            # 从动态因子索引移除
            if factor_id in self._dynamic_factors:
                del self._dynamic_factors[factor_id]

            logger.debug(f"因子注销成功: {factor_id}")
            return True

        except Exception as e:
            logger.error(f"因子注销失败 {factor_id}: {str(e)}")
            return False

    def get_factor(self, factor_id: str) -> Optional[FactorMetadata]:
        """
        获取因子元数据

        Args:
            factor_id: 因子ID

        Returns:
            因子元数据或None
        """
        return self._factors.get(factor_id)

    def list_factors(self, category: Optional[FactorCategory] = None,
                    is_dynamic: Optional[bool] = None) -> List[str]:
        """
        列出因子ID

        Args:
            category: 因子类别过滤
            is_dynamic: 是否动态因子过滤

        Returns:
            因子ID列表
        """
        factors = []

        if category is not None:
            factors = self._categories[category].copy()
        else:
            factors = list(self._factors.keys())

        if is_dynamic is not None:
            if is_dynamic:
                factors = [f for f in factors if f in self._dynamic_factors]
            else:
                factors = [f for f in factors if f not in self._dynamic_factors]

        return sorted(factors)

    def get_factors_by_category(self, category: FactorCategory) -> Dict[str, FactorMetadata]:
        """
        按类别获取因子

        Args:
            category: 因子类别

        Returns:
            因子字典
        """
        factor_ids = self._categories[category]
        return {fid: self._factors[fid] for fid in factor_ids}

    def get_dynamic_factors(self) -> Dict[str, FactorMetadata]:
        """获取所有动态因子"""
        return self._dynamic_factors.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取注册表统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_factors": len(self._factors),
            "dynamic_factors": len(self._dynamic_factors),
            "static_factors": len(self._factors) - len(self._dynamic_factors),
            "categories": {}
        }

        for category in FactorCategory:
            factor_ids = self._categories[category]
            stats["categories"][category.value] = {
                "total": len(factor_ids),
                "dynamic": len([f for f in factor_ids if f in self._dynamic_factors]),
                "static": len([f for f in factor_ids if f not in self._dynamic_factors])
            }

        return stats

    def _validate_factor_function(self, function: Callable) -> None:
        """
        验证因子计算函数

        Args:
            function: 要验证的函数

        Raises:
            FactorSystemError: 函数验证失败
        """
        if not callable(function):
            raise FactorSystemError("因子函数必须是可调用的")

        try:
            # 获取函数签名
            sig = inspect.signature(function)

            # 检查参数（至少应该接受data参数）
            params = list(sig.parameters.keys())
            if not params:
                raise FactorSystemError("因子函数必须至少有一个参数")

        except Exception as e:
            raise FactorSystemError(f"函数签名验证失败: {str(e)}")

    def clear_dynamic_factors(self) -> int:
        """
        清除所有动态因子

        Returns:
            清除的因子数量
        """
        dynamic_factor_ids = list(self._dynamic_factors.keys())
        cleared_count = 0

        for factor_id in dynamic_factor_ids:
            if self.unregister_factor(factor_id):
                cleared_count += 1

        logger.info(f"清除动态因子: {cleared_count} 个")
        return cleared_count

    def export_registry(self) -> Dict[str, Any]:
        """
        导出注册表数据

        Returns:
            注册表数据
        """
        export_data = {
            "factors": {},
            "categories": {cat.value: factors for cat, factors in self._categories.items()},
            "statistics": self.get_statistics(),
            "export_time": pd.Timestamp.now().isoformat()
        }

        for factor_id, metadata in self._factors.items():
            export_data["factors"][factor_id] = {
                "factor_id": metadata.factor_id,
                "parameters": metadata.parameters,
                "category": metadata.category.value,
                "description": metadata.description,
                "is_dynamic": metadata.is_dynamic,
                "base_factor_id": metadata.base_factor_id,
                "created_at": metadata.created_at
                # 注意：不导出函数对象，因为不可序列化
            }

        return export_data


# 全局因子注册表实例
_global_registry = ETFFactorRegistry()


def get_factor_registry() -> ETFFactorRegistry:
    """获取全局因子注册表实例"""
    return _global_registry


def register_etf_factor(factor_id: str,
                       function: Callable,
                       parameters: Optional[Dict[str, Any]] = None,
                       category: FactorCategory = FactorCategory.OVERLAP,
                       description: str = "",
                       is_dynamic: bool = False,
                       base_factor_id: Optional[str] = None) -> Callable:
    """
    装饰器：注册ETF因子

    Args:
        factor_id: 因子ID
        function: 因子计算函数
        parameters: 参数字典
        category: 因子类别
        description: 描述
        is_dynamic: 是否为动态生成的因子
        base_factor_id: 基础因子ID

    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        registry = get_factor_registry()
        success = registry.register_factor(
            factor_id=factor_id,
            function=func,
            parameters=parameters,
            category=category,
            description=description,
            is_dynamic=is_dynamic,
            base_factor_id=base_factor_id
        )

        if not success:
            logger.warning(f"因子注册失败: {factor_id}")

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.factor_id = factor_id
        wrapper.is_registered = success

        return wrapper

    return decorator


@safe_operation
def main():
    """主函数 - 测试因子注册表"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 获取注册表
    registry = get_factor_registry()

    # 测试注册函数
    def test_factor_rsi(data, period=14):
        """测试RSI因子"""
        return data['close'].rolling(period).apply(lambda x: 100 * (x.diff().gt(0).sum() / (x.diff().abs().sum())))

    # 注册因子
    success = registry.register_factor(
        factor_id="TEST_RSI_14",
        function=test_factor_rsi,
        parameters={"period": 14},
        category=FactorCategory.MOMENTUM,
        description="测试RSI因子"
    )

    print(f"因子注册结果: {success}")

    # 查看统计信息
    stats = registry.get_statistics()
    print(f"注册表统计: {stats}")

    # 列出因子
    factors = registry.list_factors()
    print(f"已注册因子: {factors}")


if __name__ == "__main__":
    main()