#!/usr/bin/env python3
"""
配置加载器 - 统一配置管理入口
不允许任何硬编码或默认值
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .enhanced_factor_calculator import IndicatorConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器 - 严格的配置管理"""

    @staticmethod
    def load_config(config_path: str = None) -> IndicatorConfig:
        """
        从配置文件加载配置

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径

        Returns:
            IndicatorConfig: 配置对象

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置项缺失或无效
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")

        # 验证必需的配置项
        required_fields = [
            "indicators.enable_ma",
            "indicators.enable_ema",
            "indicators.enable_macd",
            "indicators.enable_rsi",
            "indicators.enable_bbands",
            "indicators.enable_stoch",
            "indicators.enable_atr",
            "indicators.enable_obv",
            "indicators.enable_mstd",
            "indicators.enable_manual_indicators",
            "indicators.enable_all_periods",
            "indicators.memory_efficient",
        ]

        missing_fields = []
        for field in required_fields:
            keys = field.split(".")
            value = config_data
            try:
                for key in keys:
                    value = value[key]
                if value is None:
                    missing_fields.append(field)
            except (KeyError, TypeError):
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"配置文件中缺少必需字段: {missing_fields}")

        # 创建配置对象
        indicators_config = config_data.get("indicators", {})

        try:
            config = IndicatorConfig(
                enable_ma=indicators_config["enable_ma"],
                enable_ema=indicators_config["enable_ema"],
                enable_macd=indicators_config["enable_macd"],
                enable_rsi=indicators_config["enable_rsi"],
                enable_bbands=indicators_config["enable_bbands"],
                enable_stoch=indicators_config["enable_stoch"],
                enable_atr=indicators_config["enable_atr"],
                enable_obv=indicators_config["enable_obv"],
                enable_mstd=indicators_config["enable_mstd"],
                enable_manual_indicators=indicators_config["enable_manual_indicators"],
                enable_all_periods=indicators_config["enable_all_periods"],
                memory_efficient=indicators_config["memory_efficient"],
            )

            logger.info(f"配置加载成功: {config_path}")
            logger.info(
                f"配置: enable_all_periods={config.enable_all_periods}, memory_efficient={config.memory_efficient}"
            )

            return config

        except Exception as e:
            raise ValueError(f"创建配置对象失败: {e}")

    @staticmethod
    def create_full_config() -> IndicatorConfig:
        """
        创建完整功能的配置 - 启用所有指标和周期

        Returns:
            IndicatorConfig: 完整配置
        """
        return IndicatorConfig(
            enable_ma=True,
            enable_ema=True,
            enable_macd=True,
            enable_rsi=True,
            enable_bbands=True,
            enable_stoch=True,
            enable_atr=True,
            enable_obv=True,
            enable_mstd=True,
            enable_manual_indicators=True,
            enable_all_periods=True,  # 启用所有周期
            memory_efficient=False,  # 使用完整功能模式
        )

    @staticmethod
    def create_basic_config() -> IndicatorConfig:
        """
        创建基础配置 - 仅启用核心指标

        Returns:
            IndicatorConfig: 基础配置
        """
        return IndicatorConfig(
            enable_ma=True,
            enable_ema=True,
            enable_macd=True,
            enable_rsi=True,
            enable_bbands=True,
            enable_stoch=True,
            enable_atr=True,
            enable_obv=True,
            enable_mstd=True,
            enable_manual_indicators=True,  # 使用手动指标
            enable_all_periods=True,  # 使用所有周期
            memory_efficient=False,  # 完整计算模式
        )

    @staticmethod
    def validate_config(config: IndicatorConfig) -> bool:
        """
        验证配置的有效性

        Args:
            config: 要验证的配置

        Returns:
            bool: 配置是否有效

        Raises:
            ValueError: 配置无效
        """
        if not isinstance(config, IndicatorConfig):
            raise ValueError("配置必须是 IndicatorConfig 类型")

        # 验证布尔类型
        bool_fields = [
            "enable_ma",
            "enable_ema",
            "enable_macd",
            "enable_rsi",
            "enable_bbands",
            "enable_stoch",
            "enable_atr",
            "enable_obv",
            "enable_mstd",
            "enable_manual_indicators",
            "enable_all_periods",
            "memory_efficient",
        ]

        for field in bool_fields:
            value = getattr(config, field)
            if not isinstance(value, bool):
                raise ValueError(f"配置项 {field} 必须是布尔类型，当前值: {value}")

        logger.info("配置验证通过")
        return True
