#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置类定义 - 类型安全的配置管理"""
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = getLogger(__name__)


@dataclass
class TradingConfig:
    """交易参数配置"""

    days_per_year: int = 252
    epsilon_small: float = 1e-10
    min_periods: int = 1

    def __post_init__(self):
        """验证配置参数"""
        if self.days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        if self.epsilon_small <= 0:
            raise ValueError("epsilon_small must be positive")
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1")


@dataclass
class FactorWindowsConfig:
    """因子时间窗口配置"""

    momentum: List[int] = field(default_factory=lambda: [20, 63, 126, 252])
    volatility: List[int] = field(default_factory=lambda: [20, 60, 120])
    drawdown: List[int] = field(default_factory=lambda: [63, 126])
    rsi: List[int] = field(default_factory=lambda: [6, 14, 24])
    price_position: List[int] = field(default_factory=lambda: [20, 60, 120])
    volume_ratio: List[int] = field(default_factory=lambda: [5, 20, 60])

    # 技术指标特定窗口
    atr_period: int = 14
    vpt_trend_window: int = 20
    vol_volatility_window: int = 20
    amount_surge_short: int = 5
    amount_surge_long: int = 20
    vol_ratio_window: int = 20
    intraday_position_window: int = 5
    price_volume_div_window: int = 5

    def __post_init__(self):
        """验证窗口参数"""
        # 验证所有窗口都是正整数
        all_windows = (
            self.momentum
            + self.volatility
            + self.drawdown
            + self.rsi
            + self.price_position
            + self.volume_ratio
        )

        for window in all_windows:
            if window <= 0:
                raise ValueError(f"Window size must be positive: {window}")

        # 验证特定窗口
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.amount_surge_short >= self.amount_surge_long:
            raise ValueError("amount_surge_short must be less than amount_surge_long")


@dataclass
class ThresholdsConfig:
    """阈值参数配置"""

    large_order_volume_ratio: float = 1.2
    doji_body_threshold: Optional[float] = None
    hammer_lower_shadow_ratio: float = 2.0
    hammer_upper_shadow_ratio: float = 1.0

    def __post_init__(self):
        """验证阈值参数"""
        if self.large_order_volume_ratio <= 0:
            raise ValueError("large_order_volume_ratio must be positive")
        if self.hammer_lower_shadow_ratio <= 0:
            raise ValueError("hammer_lower_shadow_ratio must be positive")
        if self.hammer_upper_shadow_ratio < 0:
            raise ValueError("hammer_upper_shadow_ratio must be non-negative")


@dataclass
class PathsConfig:
    """路径配置"""

    data_dir: str = "raw/ETF/daily"
    output_dir: str = "etf_rotation_system/data/results/panels"
    config_file: str = "config/etf_config.yaml"

    def __post_init__(self):
        """验证路径配置"""
        # 路径将在使用时进行具体验证
        pass


@dataclass
class ProcessingConfig:
    """并行处理配置"""

    max_workers: int = 4
    continue_on_symbol_error: bool = True
    max_failure_rate: float = 0.1

    def __post_init__(self):
        """验证处理配置"""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0 <= self.max_failure_rate <= 1:
            raise ValueError("max_failure_rate must be between 0 and 1")


@dataclass
class FactorEnableConfig:
    """因子开关配置"""

    # 原有18个因子
    momentum: bool = True
    volatility: bool = True
    drawdown: bool = True
    momentum_acceleration: bool = True
    rsi: bool = True
    price_position: bool = True
    volume_ratio: bool = True

    # 新增技术因子
    overnight_return: bool = True
    atr: bool = True
    doji_pattern: bool = True
    intraday_range: bool = True
    bullish_engulfing: bool = True
    hammer_pattern: bool = True
    price_impact: bool = True
    volume_price_trend: bool = True
    vol_ma_ratio_5: bool = True
    vol_volatility_20: bool = True
    true_range: bool = True
    buy_pressure: bool = True

    # 资金流因子
    vwap_deviation: bool = True
    amount_surge_5d: bool = True
    price_volume_div: bool = True
    intraday_position: bool = True
    large_order_signal: bool = True


@dataclass
class DataProcessingConfig:
    """数据处理配置"""

    required_columns: List[str] = field(
        default_factory=lambda: [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
    )
    optional_columns: List[str] = field(default_factory=lambda: ["amount"])
    volume_column_alias: str = "vol"
    fallback_estimation: bool = True

    def __post_init__(self):
        """验证数据处理配置"""
        required_set = set(self.required_columns)
        optional_set = set(self.optional_columns)

        if required_set & optional_set:
            raise ValueError("Columns cannot be both required and optional")


@dataclass
class OutputConfig:
    """输出配置"""

    save_execution_log: bool = True
    save_metadata: bool = True
    timestamp_subdirectory: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self):
        """验证日志配置"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")


@dataclass
class DisplayConfig:
    """显示配置"""

    log_separator_length: int = 80
    progress_bar_desc: str = "计算因子"
    factor_list_start: int = 1
    json_indent: int = 2
    coverage_format: str = ".2%"

    def __post_init__(self):
        """验证显示配置"""
        if self.log_separator_length <= 0:
            raise ValueError("log_separator_length must be positive")
        if self.json_indent < 0:
            raise ValueError("json_indent must be non-negative")
        if self.factor_list_start < 1:
            raise ValueError("factor_list_start must be >= 1")


@dataclass
class ConstantsConfig:
    """计算常量配置"""

    rsi_multiplier: int = 100
    concat_axis: int = 1
    astype_float: str = "float"
    shift_days: int = 1
    concat_max_axis: int = 1
    sign_multiplier: int = 1

    def __post_init__(self):
        """验证常量配置"""
        if self.rsi_multiplier <= 0:
            raise ValueError("rsi_multiplier must be positive")
        if self.concat_axis not in [0, 1]:
            raise ValueError("concat_axis must be 0 or 1")
        if self.shift_days <= 0:
            raise ValueError("shift_days must be positive")
        if self.concat_max_axis not in [0, 1]:
            raise ValueError("concat_max_axis must be 0 or 1")
        if self.astype_float not in ["float", "float32", "float64"]:
            raise ValueError("astype_float must be a valid float type")


@dataclass
class FactorPanelConfig:
    """因子面板生成完整配置"""

    trading: TradingConfig = field(default_factory=TradingConfig)
    factor_windows: FactorWindowsConfig = field(default_factory=FactorWindowsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    factor_enable: FactorEnableConfig = field(default_factory=FactorEnableConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    constants: ConstantsConfig = field(default_factory=ConstantsConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FactorPanelConfig":
        """从YAML文件加载配置"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            return cls.from_dict(config_dict)

        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return cls()

        except Exception as e:
            logger.error(f"配置加载失败: {e}，使用默认配置")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FactorPanelConfig":
        """从字典创建配置"""
        # 递归创建嵌套配置对象
        trading_config = TradingConfig(**config_dict.get("trading", {}))
        factor_windows_config = FactorWindowsConfig(
            **config_dict.get("factor_windows", {})
        )
        thresholds_config = ThresholdsConfig(**config_dict.get("thresholds", {}))
        paths_config = PathsConfig(**config_dict.get("paths", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        factor_enable_config = FactorEnableConfig(
            **config_dict.get("factor_enable", {})
        )
        data_processing_config = DataProcessingConfig(
            **config_dict.get("data_processing", {})
        )
        output_config = OutputConfig(**config_dict.get("output", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        display_config = DisplayConfig(**config_dict.get("display", {}))
        constants_config = ConstantsConfig(**config_dict.get("constants", {}))

        return cls(
            trading=trading_config,
            factor_windows=factor_windows_config,
            thresholds=thresholds_config,
            paths=paths_config,
            processing=processing_config,
            factor_enable=factor_enable_config,
            data_processing=data_processing_config,
            output=output_config,
            logging=logging_config,
            display=display_config,
            constants=constants_config,
        )

    def to_legacy_format(self) -> Dict:
        """转换为原有格式，保持向后兼容"""
        return {
            "factor_generation": {
                "momentum_periods": self.factor_windows.momentum,
                "volatility_windows": self.factor_windows.volatility,
                "rsi_windows": self.factor_windows.rsi,
                "price_position_windows": self.factor_windows.price_position,
                "volume_ratio_windows": self.factor_windows.volume_ratio,
            }
        }

    def validate(self) -> bool:
        """验证配置完整性"""
        try:
            # 各个配置类的__post_init__方法已经进行了验证
            # 这里可以添加额外的跨配置验证

            # 检查因子数量合理性
            enabled_factors = sum(
                [
                    self.factor_enable.momentum,
                    self.factor_enable.volatility,
                    self.factor_enable.drawdown,
                    self.factor_enable.momentum_acceleration,
                    self.factor_enable.rsi,
                    self.factor_enable.price_position,
                    self.factor_enable.volume_ratio,
                    self.factor_enable.overnight_return,
                    self.factor_enable.atr,
                    self.factor_enable.doji_pattern,
                    self.factor_enable.intraday_range,
                    self.factor_enable.bullish_engulfing,
                    self.factor_enable.hammer_pattern,
                    self.factor_enable.price_impact,
                    self.factor_enable.volume_price_trend,
                    self.factor_enable.vol_ma_ratio_5,
                    self.factor_enable.vol_volatility_20,
                    self.factor_enable.true_range,
                    self.factor_enable.buy_pressure,
                    self.factor_enable.vwap_deviation,
                    self.factor_enable.amount_surge_5d,
                    self.factor_enable.price_volume_div,
                    self.factor_enable.intraday_position,
                    self.factor_enable.large_order_signal,
                ]
            )

            if enabled_factors == 0:
                raise ValueError("至少需要启用一个因子")

            logger.info(f"配置验证通过，启用了 {enabled_factors} 个因子")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
