#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 配置管理模块
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 管理未来函数防护组件的配置
- 提供预设环境和自定义配置
- 支持配置验证和热更新
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import ConfigurationError


class StrictMode(Enum):
    """严格模式枚举"""

    DISABLED = "disabled"  # 禁用严格模式
    WARN_ONLY = "warn_only"  # 仅警告
    ENFORCED = "enforced"  # 强制执行


class MonitoringLevel(Enum):
    """监控级别枚举"""

    NONE = "none"  # 无监控
    BASIC = "basic"  # 基础监控
    COMPREHENSIVE = "comprehensive"  # 全面监控
    REAL_TIME = "real_time"  # 实时监控


class AlertThreshold(Enum):
    """报警阈值枚举"""

    LIBERAL = "liberal"  # 宽松阈值
    MODERATE = "moderate"  # 适中阈值
    CONSERVATIVE = "conservative"  # 保守阈值


@dataclass
class StaticCheckConfig:
    """静态检查配置"""

    enabled: bool = True
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    check_patterns: List[str] = field(
        default_factory=lambda: [
            r"\.shift\(-\d+\)",  # .shift(-n)
            r"future_\w+",  # future_变量
            r"lead_\w+",  # lead_变量
            r"\.shift\(-",  # .shift(- 开头
            r"shift\(-\d",  # shift(-d 开头
        ]
    )
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            r"_test\.py$",  # 测试文件
            r"__pycache__/",  # 缓存目录
            r"\.git/",  # Git目录
        ]
    )
    max_file_size_mb: int = 10  # 最大文件大小限制


@dataclass
class RuntimeValidationConfig:
    """运行时验证配置"""

    enabled: bool = True
    strict_mode: StrictMode = StrictMode.WARN_ONLY
    time_series_safety: bool = True
    data_integrity: bool = True
    statistical_checks: bool = True
    correlation_threshold: float = 0.95
    coverage_threshold: float = 0.9
    min_history_map: Dict[str, int] = field(
        default_factory=lambda: {
            "1min": 60,
            "5min": 48,
            "15min": 16,
            "30min": 8,
            "60min": 24,
            "120min": 12,
            "240min": 6,
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
        }
    )
    price_consistency: bool = True
    error_tolerance: float = 0.01  # 错误容忍度


@dataclass
class HealthMonitorConfig:
    """健康监控配置"""

    enabled: bool = True
    monitoring_level: MonitoringLevel = MonitoringLevel.BASIC
    alert_threshold: AlertThreshold = AlertThreshold.MODERATE
    coverage_threshold: float = 0.9
    variance_threshold: float = 1e-10
    extreme_value_threshold: float = 0.01  # 1%异常值阈值
    correlation_threshold: float = 0.95
    real_time_alerts: bool = False
    export_reports: bool = True
    report_retention_days: int = 30


@dataclass
class CacheConfig:
    """缓存配置"""

    enabled: bool = True
    cache_dir: Optional[str] = None
    max_cache_size_mb: int = 100
    ttl_hours: int = 24
    compression_enabled: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True


@dataclass
class GuardConfig:
    """未来函数防护组件总配置"""

    # 基础配置
    mode: str = "auto"  # auto, development, research, production
    strict_mode: StrictMode = StrictMode.WARN_ONLY

    # 子模块配置
    static_check: StaticCheckConfig = field(default_factory=StaticCheckConfig)
    runtime_validation: RuntimeValidationConfig = field(
        default_factory=RuntimeValidationConfig
    )
    health_monitor: HealthMonitorConfig = field(default_factory=HealthMonitorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # 全局设置
    enabled: bool = True
    auto_fix: bool = True  # 自动修复可修复的问题
    report_generation: bool = True
    performance_monitoring: bool = False

    def __post_init__(self):
        """初始化后处理"""
        # 设置默认缓存目录
        if self.cache.cache_dir is None:
            self.cache.cache_dir = str(Path.home() / ".future_function_guard_cache")

        # 验证配置
        self.validate()

    def validate(self) -> None:
        """验证配置的有效性"""
        errors = []

        # 验证模式
        valid_modes = ["auto", "development", "research", "production", "custom"]
        if self.mode not in valid_modes:
            errors.append(f"Invalid mode: {self.mode}")

        # 验证阈值
        if not 0.0 <= self.runtime_validation.correlation_threshold <= 1.0:
            errors.append("correlation_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.runtime_validation.coverage_threshold <= 1.0:
            errors.append("coverage_threshold must be between 0.0 and 1.0")

        # 验证缓存配置
        if self.cache.max_cache_size_mb <= 0:
            errors.append("max_cache_size_mb must be positive")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}",
                context={"validation_errors": errors},
            )

    @classmethod
    def preset(cls, environment: str) -> GuardConfig:
        """
        创建预设环境的配置

        Args:
            environment: 环境名称 (auto, development, research, production)

        Returns:
            对应环境的配置
        """
        if environment == "auto":
            # auto模式使用环境变量或默认为research模式
            import os

            env_mode = os.getenv("FUTURE_GUARD_MODE", "research")
            return cls.preset(env_mode)

        elif environment == "development":
            return cls(
                mode="development",
                strict_mode=StrictMode.DISABLED,
                static_check=StaticCheckConfig(
                    enabled=True, cache_enabled=False  # 开发环境不使用缓存
                ),
                runtime_validation=RuntimeValidationConfig(
                    enabled=True,
                    strict_mode=StrictMode.WARN_ONLY,
                    correlation_threshold=0.99,  # 更宽松的阈值
                ),
                health_monitor=HealthMonitorConfig(
                    enabled=True,
                    monitoring_level=MonitoringLevel.BASIC,
                    alert_threshold=AlertThreshold.LIBERAL,
                ),
                logging=LoggingConfig(level="DEBUG"),
            )

        elif environment == "research":
            return cls(
                mode="research",
                strict_mode=StrictMode.WARN_ONLY,
                static_check=StaticCheckConfig(enabled=True),
                runtime_validation=RuntimeValidationConfig(
                    enabled=True,
                    strict_mode=StrictMode.WARN_ONLY,
                    correlation_threshold=0.95,
                ),
                health_monitor=HealthMonitorConfig(
                    enabled=True,
                    monitoring_level=MonitoringLevel.COMPREHENSIVE,
                    alert_threshold=AlertThreshold.MODERATE,
                ),
                logging=LoggingConfig(level="INFO"),
            )

        elif environment == "production":
            return cls(
                mode="production",
                strict_mode=StrictMode.ENFORCED,
                static_check=StaticCheckConfig(enabled=True),
                runtime_validation=RuntimeValidationConfig(
                    enabled=True,
                    strict_mode=StrictMode.ENFORCED,
                    correlation_threshold=0.90,  # 更严格的阈值
                    coverage_threshold=0.95,
                ),
                health_monitor=HealthMonitorConfig(
                    enabled=True,
                    monitoring_level=MonitoringLevel.REAL_TIME,
                    alert_threshold=AlertThreshold.CONSERVATIVE,
                    real_time_alerts=True,
                ),
                logging=LoggingConfig(level="WARNING"),
            )

        else:
            raise ConfigurationError(f"Unknown environment: {environment}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> GuardConfig:
        """
        从字典创建配置

        Args:
            config_dict: 配置字典

        Returns:
            配置实例
        """
        try:
            # 处理枚举类型
            if "strict_mode" in config_dict and isinstance(
                config_dict["strict_mode"], str
            ):
                # 尝试通过值查找枚举成员
                strict_mode_value = config_dict["strict_mode"]
                for mode in StrictMode:
                    if mode.value == strict_mode_value:
                        config_dict["strict_mode"] = mode
                        break
                else:
                    # 如果找不到，使用默认值
                    config_dict["strict_mode"] = StrictMode.WARN_ONLY

            # 处理嵌套配置中的枚举
            def convert_enums_in_dict(
                config_data: Dict[str, Any], config_class: type
            ) -> Dict[str, Any]:
                """转换字典中的枚举值"""
                result = config_data.copy()

                # 定义枚举映射
                enum_mappings = {
                    "StrictMode": StrictMode,
                    "MonitoringLevel": MonitoringLevel,
                    "AlertThreshold": AlertThreshold,
                }

                # 获取配置类的字段信息
                if hasattr(config_class, "__dataclass_fields__"):
                    for (
                        field_name,
                        field_info,
                    ) in config_class.__dataclass_fields__.items():
                        if field_name in result:
                            field_value = result[field_name]

                            # 如果值已经是枚举类型，跳过
                            if hasattr(field_value, "value"):
                                continue

                            # 尝试通过字段名和类型推断枚举类型
                            field_type_str = str(field_info.type)

                            # 检查字段类型是否包含枚举名称
                            for enum_name, enum_class in enum_mappings.items():
                                if enum_name in field_type_str and isinstance(
                                    field_value, str
                                ):
                                    # 尝试通过值查找枚举成员
                                    for enum_member in enum_class:
                                        if enum_member.value == field_value:
                                            result[field_name] = enum_member
                                            break
                                    break

                return result

            # 处理嵌套配置
            if "static_check" in config_dict:
                static_check_dict = convert_enums_in_dict(
                    config_dict["static_check"], StaticCheckConfig
                )
                config_dict["static_check"] = StaticCheckConfig(**static_check_dict)

            if "runtime_validation" in config_dict:
                runtime_validation_dict = convert_enums_in_dict(
                    config_dict["runtime_validation"], RuntimeValidationConfig
                )
                config_dict["runtime_validation"] = RuntimeValidationConfig(
                    **runtime_validation_dict
                )

            if "health_monitor" in config_dict:
                health_monitor_dict = convert_enums_in_dict(
                    config_dict["health_monitor"], HealthMonitorConfig
                )
                config_dict["health_monitor"] = HealthMonitorConfig(
                    **health_monitor_dict
                )

            if "cache" in config_dict:
                cache_dict = convert_enums_in_dict(config_dict["cache"], CacheConfig)
                config_dict["cache"] = CacheConfig(**cache_dict)

            if "logging" in config_dict:
                logging_dict = convert_enums_in_dict(
                    config_dict["logging"], LoggingConfig
                )
                config_dict["logging"] = LoggingConfig(**logging_dict)

            return cls(**config_dict)

        except Exception as e:
            raise ConfigurationError(f"Failed to create config from dict: {e}")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> GuardConfig:
        """
        从文件加载配置

        Args:
            file_path: 配置文件路径

        Returns:
            配置实例
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigurationError(f"Config file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {file_path.suffix}"
                    )

            return cls.from_dict(config_dict)

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {e}")

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> GuardConfig:
        """
        从文件加载配置 (别名方法)

        Args:
            file_path: 配置文件路径

        Returns:
            配置实例
        """
        return cls.load_from_file(file_path)

    @classmethod
    def from_environment(cls) -> GuardConfig:
        """
        从环境变量创建配置

        Returns:
            配置实例
        """
        # 获取环境配置
        env_mode = os.getenv("FUTURE_GUARD_MODE", "auto")
        env_strict = os.getenv("FUTURE_GUARD_STRICT", "warn_only")

        # 基础配置
        config = cls(mode=env_mode)

        # 设置严格模式
        try:
            config.strict_mode = StrictMode(env_strict)
        except ValueError:
            config.strict_mode = StrictMode.WARN_ONLY

        # 其他环境变量
        if os.getenv("FUTURE_GUARD_CACHE_DIR"):
            config.cache.cache_dir = os.getenv("FUTURE_GUARD_CACHE_DIR")

        if os.getenv("FUTURE_GUARD_LOG_LEVEL"):
            config.logging.level = os.getenv("FUTURE_GUARD_LOG_LEVEL")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)

        # 转换枚举为字符串
        if isinstance(result["strict_mode"], StrictMode):
            result["strict_mode"] = result["strict_mode"].value

        # 转换嵌套配置中的枚举
        def convert_enums(config_dict: Dict[str, Any]) -> Dict[str, Any]:
            """递归转换配置字典中的枚举"""
            result = {}
            for key, value in config_dict.items():
                if isinstance(value, (StrictMode, MonitoringLevel, AlertThreshold)):
                    result[key] = value.value
                elif isinstance(value, dict):
                    result[key] = convert_enums(value)
                else:
                    result[key] = value
            return result

        # 转换嵌套配置
        result["static_check"] = convert_enums(asdict(self.static_check))
        result["runtime_validation"] = convert_enums(asdict(self.runtime_validation))
        result["health_monitor"] = convert_enums(asdict(self.health_monitor))
        result["cache"] = convert_enums(asdict(self.cache))
        result["logging"] = convert_enums(asdict(self.logging))

        return result

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        保存配置到文件

        Args:
            file_path: 配置文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save config file: {e}")

    def merge(self, other: GuardConfig) -> GuardConfig:
        """
        合并两个配置，other的设置会覆盖当前设置

        Args:
            other: 要合并的配置

        Returns:
            合并后的新配置
        """
        current_dict = self.to_dict()
        other_dict = other.to_dict()

        # 深度合并
        def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
            result = dict1.copy()
            for key, value in dict2.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_dict = deep_merge(current_dict, other_dict)
        return GuardConfig.from_dict(merged_dict)

    def copy(self) -> GuardConfig:
        """创建配置的副本"""
        return GuardConfig.from_dict(self.to_dict())

    def __str__(self) -> str:
        """字符串表示"""
        return f"GuardConfig(mode={self.mode}, strict_mode={self.strict_mode.value})"


# 全局默认配置
DEFAULT_CONFIG = GuardConfig.preset("auto")
