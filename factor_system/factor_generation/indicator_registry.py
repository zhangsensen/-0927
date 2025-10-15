#!/usr/bin/env python3
"""
指标注册中心 - 配置驱动的指标管理
消除闭包堆积，统一参数管理
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSpec:
    """指标规格定义"""

    name: str  # 指标名称（如 MA, RSI）
    indicator_type: str  # 类型：vbt/talib/manual
    param_grid: Dict[str, List[Any]]  # 参数网格
    requires_entries: bool = False  # 是否需要entries信号
    batch_capable: bool = True  # 是否支持批量计算
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他指标
    postprocess: Optional[Callable] = None  # 后处理函数
    enabled: bool = True  # 是否启用


class IndicatorRegistry:
    """指标注册中心 - 管理所有指标配置"""

    def __init__(self):
        self.specs: Dict[str, IndicatorSpec] = {}
        logger.info("指标注册中心初始化")

    def register(self, spec: IndicatorSpec):
        """注册指标"""
        if spec.name in self.specs:
            logger.warning(f"指标 {spec.name} 已存在，将被覆盖")
        self.specs[spec.name] = spec
        logger.debug(
            f"注册指标: {spec.name} (type={spec.indicator_type}, batch={spec.batch_capable})"
        )

    def get(self, name: str) -> Optional[IndicatorSpec]:
        """获取指标规格"""
        return self.specs.get(name)

    def list_enabled(self) -> List[IndicatorSpec]:
        """列出所有启用的指标"""
        return [spec for spec in self.specs.values() if spec.enabled]

    def list_batch_capable(self) -> List[IndicatorSpec]:
        """列出支持批量计算的指标"""
        return [
            spec for spec in self.specs.values() if spec.enabled and spec.batch_capable
        ]

    def get_by_type(self, indicator_type: str) -> List[IndicatorSpec]:
        """按类型获取指标"""
        return [
            spec
            for spec in self.specs.values()
            if spec.enabled and spec.indicator_type == indicator_type
        ]


def create_default_registry(timeframe: str = "5min") -> IndicatorRegistry:
    """创建默认指标注册表"""
    registry = IndicatorRegistry()

    # 根据时间框架调整参数
    if timeframe == "1min":
        ma_windows = [3, 5, 10, 15, 20]
        rsi_windows = [7, 14, 21]
        mstd_windows = [5, 10, 15]
    elif timeframe == "5min":
        ma_windows = [3, 5, 8, 10, 15, 20]
        rsi_windows = [7, 10, 14]
        mstd_windows = [5, 10, 15]
    elif timeframe == "1day":
        ma_windows = [5, 10, 20, 30, 60, 120, 250]
        rsi_windows = [6, 12, 24]
        mstd_windows = [10, 20, 30, 60]
    else:
        ma_windows = [5, 10, 20, 30]
        rsi_windows = [14, 20]
        mstd_windows = [10, 20]

    # 注册批量化指标
    registry.register(
        IndicatorSpec(
            name="MA",
            indicator_type="vbt",
            param_grid={"window": ma_windows},
            batch_capable=True,
        )
    )

    registry.register(
        IndicatorSpec(
            name="EMA",
            indicator_type="manual",
            param_grid={"span": ma_windows},
            batch_capable=True,
        )
    )

    registry.register(
        IndicatorSpec(
            name="MSTD",
            indicator_type="vbt",
            param_grid={"window": mstd_windows},
            batch_capable=True,
        )
    )

    registry.register(
        IndicatorSpec(
            name="RSI",
            indicator_type="shared",
            param_grid={"period": rsi_windows},
            batch_capable=True,
        )
    )

    registry.register(
        IndicatorSpec(
            name="ATR",
            indicator_type="vbt",
            param_grid={"window": [7, 10, 14]},
            batch_capable=True,
        )
    )

    # 不支持批量的指标
    registry.register(
        IndicatorSpec(
            name="OBV", indicator_type="vbt", param_grid={}, batch_capable=False
        )
    )

    # 需要entries的指标（默认关闭）
    registry.register(
        IndicatorSpec(
            name="OHLCSTX",
            indicator_type="vbt",
            param_grid={},
            requires_entries=True,
            enabled=False,
        )
    )

    logger.info(f"默认注册表创建完成: {len(registry.specs)} 个指标")
    return registry
