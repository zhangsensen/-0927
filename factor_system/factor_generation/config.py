#!/usr/bin/env python3
"""
增强配置管理模块 - Linus风格设计
支持多时间框架154指标因子引擎的完整配置
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml


class SimpleConfig:
    """简化配置类"""

    def __init__(self, config_file: str = None):
        """初始化配置"""
        if config_file is None:
            # 默认使用同目录下的config.yaml
            config_file = Path(__file__).parent / "config.yaml"
        self.config_file = Path(config_file)
        self.config_data = None

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f)
            return self.config_data
        except Exception as e:
            raise RuntimeError(f"加载配置失败: {e}")

    def get(self, key: str, default=None):
        """获取配置值"""
        if self.config_data is None:
            self.load_config()

        # 支持嵌套键，如 "data.root_dir"
        keys = key.split(".")
        value = self.config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_root(self) -> str:
        """获取数据根目录"""
        return self.get("data.root_dir", "../raw/HK")

    def get_output_dir(self) -> str:
        """获取输出目录"""
        return self.get("output.directory", "./output")

    def get_log_file(self, timestamp: str = None) -> str:
        """获取日志文件路径"""
        if timestamp:
            # 如果有时间戳，创建带时间戳的日志文件
            base_name = Path(self.get("logging.file", "multi_tf_detector.log")).stem
            extension = Path(self.get("logging.file", "multi_tf_detector.log")).suffix
            log_dir = Path(self.get("logging.file", "multi_tf_detector.log")).parent
            return str(log_dir / f"{base_name}_{timestamp}{extension}")
        else:
            # 默认日志文件（向后兼容）
            return self.get("logging.file", "multi_tf_detector.log")

    def get_log_level(self) -> str:
        """获取日志级别"""
        return self.get("logging.level", "INFO")

    # 简化的配置方法 - 直接从配置文件读取
    def get_indicator_config(self) -> Dict[str, Any]:
        """获取指标配置"""
        return self.get("indicators", {})

    def get_timeframe_config(self) -> Dict[str, Any]:
        """获取时间框架配置"""
        return self.get("timeframes", {})

    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self.get("performance", {})

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.get("output", {})

    def get_enabled_indicators(self) -> List[str]:
        """获取启用的指标列表"""
        indicator_config = self.get_indicator_config()
        enabled = []

        if indicator_config.get("enable_ma", True):
            enabled.extend(["MA", "EMA"])
        if indicator_config.get("enable_macd", True):
            enabled.append("MACD")
        if indicator_config.get("enable_rsi", True):
            enabled.append("RSI")
        if indicator_config.get("enable_bbands", True):
            enabled.append("BBANDS")
        if indicator_config.get("enable_stoch", True):
            enabled.append("STOCH")
        if indicator_config.get("enable_atr", True):
            enabled.append("ATR")
        if indicator_config.get("enable_obv", True):
            enabled.append("OBV")
        if indicator_config.get("enable_mstd", True):
            enabled.append("MSTD")
        if indicator_config.get("enable_manual_indicators", True):
            enabled.extend(["WILLR", "CCI", "Momentum", "Position", "Trend", "Volume"])

        return enabled

    def get_enabled_timeframes(self) -> List[str]:
        """获取启用的时间框架列表"""
        return self.get_timeframe_config().get(
            "enabled", ["5min", "15min", "30min", "60min", "daily"]
        )


# 全局配置实例
_config = None


def get_config() -> SimpleConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = SimpleConfig()
    return _config


def setup_logging(timestamp: str = None):
    """设置日志"""
    config = get_config()

    # 清除所有现有的日志处理器，确保配置生效
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)

    # 创建日志目录
    log_file = Path(config.get_log_file(timestamp))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, config.get_log_level().upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,  # 强制重新配置
    )

    return str(log_file)
