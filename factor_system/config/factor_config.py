"""
因子配置管理器

功能：
- 加载和管理因子参数配置
- 支持YAML配置文件
- 提供配置验证和默认值
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


class FactorConfig:
    """因子配置管理器"""

    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "factor_config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "money_flow": {
                "core_factors": {
                    "MainNetInflow_Rate": {"window": 5, "enabled": True},
                    "LargeOrder_Ratio": {"window": 10, "enabled": True},
                    "SuperLargeOrder_Ratio": {"window": 20, "enabled": True},
                    "OrderConcentration": {"enabled": True},
                    "MoneyFlow_Hierarchy": {"enabled": True},
                    "MoneyFlow_Consensus": {"window": 5, "enabled": True},
                    "MainFlow_Momentum": {
                        "short_window": 5,
                        "long_window": 10,
                        "enabled": True,
                    },
                    "Flow_Price_Divergence": {"window": 5, "enabled": True},
                },
                "enhanced_factors": {
                    "Institutional_Absorption": {"enabled": True},
                    "Flow_Tier_Ratio_Delta": {"window": 5, "enabled": True},
                    "Flow_Reversal_Ratio": {"enabled": True},
                },
                "constraints": {
                    "gap_signal": {"threshold_sigma": 1.8, "window": 20},
                    "tail30_ratio": {
                        "include_auction": False,
                        "zscore_window": 60,
                    },
                },
            }
        }

    def get_factor_params(
        self, factor_name: str, category: str = "core_factors"
    ) -> Dict[str, Any]:
        """
        获取因子参数

        Args:
            factor_name: 因子名称
            category: 因子类别 (core_factors, enhanced_factors, constraints)

        Returns:
            因子参数字典
        """
        money_flow_config = self._config.get("money_flow", {})
        category_config = money_flow_config.get(category, {})
        return category_config.get(factor_name, {})

    def get_enabled_factors(self, category: str = "core_factors") -> List[str]:
        """
        获取启用的因子列表

        Args:
            category: 因子类别

        Returns:
            启用的因子名称列表
        """
        money_flow_config = self._config.get("money_flow", {})
        category_config = money_flow_config.get(category, {})

        enabled_factors = []
        for name, config in category_config.items():
            if isinstance(config, dict) and config.get("enabled", True):
                enabled_factors.append(name)

        return enabled_factors

    def is_factor_enabled(self, factor_name: str, category: str = "core_factors") -> bool:
        """
        检查因子是否启用

        Args:
            factor_name: 因子名称
            category: 因子类别

        Returns:
            是否启用
        """
        params = self.get_factor_params(factor_name, category)
        return params.get("enabled", True)

    def get_all_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config

    def save_config(self, config_path: Path = None) -> None:
        """
        保存配置到文件

        Args:
            config_path: 保存路径，默认为当前配置路径
        """
        save_path = config_path or self.config_path

        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

        print(f"Config saved to {save_path}")
