#!/usr/bin/env python3
"""
配置加载器
统一加载和管理精细策略系统的所有配置
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "config" / "fine_strategy_config.yaml"
            )

        self.config_path = Path(config_path)
        self.config = None

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            logger.info(f"成功加载配置文件: {self.config_path}")
            return self.config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key_path: 配置键路径，用点分隔，如 "data_paths.vbt_results_base"
            default: 默认值

        Returns:
            配置值
        """
        if self.config is None:
            self.load_config()

        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_paths(self) -> Dict[str, str]:
        """获取数据路径配置"""
        return self.get("data_paths", {})

    def get_analysis_config(self) -> Dict[str, Any]:
        """获取分析配置"""
        return self.get("analysis_config", {})

    def get_screening_config(self) -> Dict[str, Any]:
        """获取筛选配置"""
        return self.get("screening_config", {})

    def get_optimization_config(self) -> Dict[str, Any]:
        """获取优化配置"""
        return self.get("optimization_config", {})

    def get_strategy_templates(self) -> Dict[str, Any]:
        """获取策略模板配置"""
        return self.get("strategy_templates", {})

    def get_factors_config(self) -> Dict[str, Any]:
        """获取因子配置"""
        return self.get("factors", {})

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.get("output_config", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get("logging_config", {})

    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return self.get("cache_config", {})

    def get_validation_config(self) -> Dict[str, Any]:
        """获取验证配置"""
        return self.get("validation_config", {})

    def get_advanced_config(self) -> Dict[str, Any]:
        """获取高级配置"""
        return self.get("advanced_config", {})

    def get_debug_config(self) -> Dict[str, Any]:
        """获取调试配置"""
        return self.get("debug_config", {})

    def get_execution_config(self) -> Dict[str, Any]:
        """获取执行配置"""
        return self.get("execution_config", {})

    def setup_logging(self):
        """设置日志配置"""
        import logging.config

        logging_config = self.get_logging_config()

        # 简单的日志设置
        level = getattr(logging, logging_config.get("level", "INFO").upper())
        format_str = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 配置根日志器
        logging.basicConfig(level=level, format=format_str, force=True)

        # 如果启用了文件日志
        if logging_config.get("file_logging", {}).get("enabled", False):
            self._setup_file_logging(logging_config)

    def _setup_file_logging(self, logging_config: Dict[str, Any]):
        """设置文件日志"""
        try:
            from logging.handlers import RotatingFileHandler

            file_config = logging_config["file_logging"]
            log_dir = Path(self.get("data_paths.log_dir", "./logs"))
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / "fine_strategy.log"
            max_bytes = self._parse_size(file_config.get("max_file_size", "10MB"))
            backup_count = file_config.get("backup_count", 5)

            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )

            formatter = logging.Formatter(
                logging_config.get(
                    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            handler.setFormatter(formatter)

            # 添加到根日志器
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)

        except Exception as e:
            logger.warning(f"设置文件日志失败: {e}")

    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串"""
        size_str = size_str.upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def validate_config(self) -> bool:
        """验证配置完整性"""
        required_sections = [
            "data_paths",
            "analysis_config",
            "screening_config",
            "optimization_config",
        ]

        missing_sections = []
        for section in required_sections:
            if not self.get(section):
                missing_sections.append(section)

        if missing_sections:
            logger.error(f"配置文件缺少必要部分: {missing_sections}")
            return False

        # 验证路径配置
        data_paths = self.get_data_paths()
        required_paths = ["vbt_results_base", "analysis_output"]
        for path_key in required_paths:
            if not data_paths.get(path_key):
                logger.error(f"缺少必要的路径配置: {path_key}")
                return False

        logger.info("配置验证通过")
        return True

    def save_config(self, output_path: Optional[str] = None) -> bool:
        """保存当前配置"""
        try:
            if self.config is None:
                logger.error("没有可保存的配置")
                return False

            if output_path is None:
                output_path = self.config_path

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )

            logger.info(f"配置已保存至: {output_file}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            if self.config is None:
                self.load_config()

            self._deep_update(self.config, updates)
            logger.info("配置更新成功")
            return True

        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def create_directories(self) -> bool:
        """创建必要的目录"""
        try:
            directories = [
                self.get("data_paths.analysis_output"),
                self.get("data_paths.cache_dir"),
                self.get("data_paths.log_dir"),
            ]

            for directory in directories:
                if directory:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                    logger.debug(f"创建目录: {directory}")

            logger.info("目录创建完成")
            return True

        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            return False

    def get_factor_list(self, factor_type: str = "all") -> list:
        """获取因子列表"""
        factors_config = self.get_factors_config()
        all_factors = []

        if factor_type in ["all", "core"]:
            core_factors = factors_config.get("core_factors", [])
            all_factors.extend([f["name"] for f in core_factors])

        if factor_type in ["all", "supplementary"]:
            supp_factors = factors_config.get("supplementary_factors", [])
            all_factors.extend([f["name"] for f in supp_factors])

        return list(set(all_factors))

    def get_factor_importance(self, factor_name: str) -> float:
        """获取因子重要性"""
        factors_config = self.get_factors_config()

        # 搜索核心因子
        for factor in factors_config.get("core_factors", []):
            if factor["name"] == factor_name:
                return factor.get("importance", 0.5)

        # 搜索补充因子
        for factor in factors_config.get("supplementary_factors", []):
            if factor["name"] == factor_name:
                return factor.get("importance", 0.3)

        return 0.1  # 默认重要性


# 全局配置加载器实例
_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
        _config_loader.load_config()
        _config_loader.setup_logging()
        _config_loader.create_directories()
    return _config_loader


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    return get_config_loader(config_path).config
