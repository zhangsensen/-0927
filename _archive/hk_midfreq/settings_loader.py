# -*- coding: utf-8 -*-
"""
设置加载器

从 settings.yaml 加载配置，禁止硬编码
"""
import logging
from pathlib import Path
from typing import Any

import yaml


class Settings:
    """全局设置单例"""

    _instance: "Settings" = None
    _config: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_settings()
        return cls._instance

    def _load_settings(self) -> None:
        """加载配置文件"""
        settings_path = Path(__file__).parent / "settings.yaml"
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        with open(settings_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套访问 (如: "log.level")
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @property
    def log_level(self) -> str:
        """日志级别"""
        level_str = self.get("log.level", "INFO")
        return level_str.upper()

    @property
    def log_format(self) -> str:
        """日志格式"""
        return self.get(
            "log.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    @property
    def log_rotating_max_bytes(self) -> int:
        """日志文件最大字节数"""
        return self.get("log.rotating.max_bytes", 10485760)

    @property
    def log_rotating_backup_count(self) -> int:
        """日志文件备份数量"""
        return self.get("log.rotating.backup_count", 5)

    @property
    def session_isolated_logging(self) -> bool:
        """是否启用会话级日志隔离"""
        return self.get("log.session_isolated", True)

    @property
    def chart_min_file_size_kb(self) -> int:
        """图表最小文件大小 (kB)"""
        return self.get("charts.min_file_size_kb", 3)

    @property
    def chart_raise_on_empty(self) -> bool:
        """空图表时是否抛异常"""
        return self.get("charts.raise_on_empty", True)

    @property
    def factor_clean_columns(self) -> bool:
        """是否清洗因子列名"""
        return self.get("factors.clean_column_names", True)

    @property
    def factor_allowed_chars(self) -> str:
        """因子列名允许的字符正则"""
        return self.get("factors.allowed_chars", "[^0-9A-Za-z_]")

    @property
    def env_snapshot_enabled(self) -> bool:
        """是否启用环境快照"""
        return self.get("environment.snapshot_enabled", True)


def get_settings() -> Settings:
    """获取全局设置实例"""
    return Settings()


def get_log_level() -> int:
    """获取日志级别 (logging 常量)"""
    settings = get_settings()
    level_str = settings.log_level
    return getattr(logging, level_str, logging.INFO)
