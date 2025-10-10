#!/usr/bin/env python3
"""
官方因子配置文件 / Official Factor Configuration

该文件定义了系统中所有官方因子的配置。
FactorEngine 必须严格遵循此配置，不得添加或修改任何因子。

This file defines the configuration for all official factors in the system.
FactorEngine must strictly follow this configuration and cannot add or modify any factors.
"""

from typing import Any, Dict

# 官方因子配置 / Official Factor Configuration
# 严格遵循 FACTOR_REGISTRY.md 中定义的因子清单
FACTOR_CONFIG: Dict[str, Dict[str, Any]] = {
    "MACD": {
        "function": "talib.MACD",
        "parameters": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "output_columns": ["MACD", "MACD_signal", "MACD_hist"],
        "description": "移动平均收敛散度 / Moving Average Convergence Divergence",
        "category": "趋势指标 / Trend Indicator",
    },
    "RSI": {
        "function": "talib.RSI",
        "parameters": {"timeperiod": 14},
        "output_columns": ["RSI"],
        "description": "相对强弱指数 / Relative Strength Index",
        "category": "动量指标 / Momentum Indicator",
    },
    "STOCH": {
        "function": "talib.STOCH",
        "parameters": {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3},
        "output_columns": ["STOCH_slowk", "STOCH_slowd"],
        "description": "随机指标 / Stochastic Oscillator",
        "category": "动量指标 / Momentum Indicator",
    },
}

# 因子元数据 / Factor Metadata
FACTOR_METADATA: Dict[str, Dict[str, Any]] = {
    "MACD": {
        "id": "MACD",
        "status": "🟢 ACTIVE",
        "created_date": "2025-10-08",
        "version": "1.0",
        "dependencies": ["EMA"],
        "computational_complexity": "O(n)",
        "memory_usage": "Low",
    },
    "RSI": {
        "id": "RSI",
        "status": "🟢 ACTIVE",
        "created_date": "2025-10-08",
        "version": "1.0",
        "dependencies": [],
        "computational_complexity": "O(n)",
        "memory_usage": "Low",
    },
    "STOCH": {
        "id": "STOCH",
        "status": "🟢 ACTIVE",
        "created_date": "2025-10-08",
        "version": "1.0",
        "dependencies": [],
        "computational_complexity": "O(n)",
        "memory_usage": "Low",
    },
}

# 系统配置 / System Configuration
SYSTEM_CONFIG: Dict[str, Any] = {
    "strict_mode": True,  # 严格模式：只允许配置中的因子
    "version": "1.0",
    "last_updated": "2025-10-08",
    "registry_file": "factor_system/FACTOR_REGISTRY.md",
    "validation_enabled": True,
}


# 辅助函数 / Helper Functions
def get_factor_ids() -> list:
    """获取所有因子ID / Get all factor IDs"""
    return list(FACTOR_CONFIG.keys())


def get_factor_config(factor_id: str) -> Dict[str, Any]:
    """获取指定因子的配置 / Get configuration for specific factor"""
    if factor_id not in FACTOR_CONFIG:
        raise ValueError(
            f"因子 '{factor_id}' 不在官方清单中 / Factor '{factor_id}' not in official registry"
        )
    return FACTOR_CONFIG[factor_id]


def get_factor_metadata(factor_id: str) -> Dict[str, Any]:
    """获取指定因子的元数据 / Get metadata for specific factor"""
    if factor_id not in FACTOR_METADATA:
        raise ValueError(
            f"因子 '{factor_id}' 不在官方清单中 / Factor '{factor_id}' not in official registry"
        )
    return FACTOR_METADATA[factor_id]


def validate_factor_ids(factor_ids: list) -> bool:
    """验证因子ID是否都在官方清单中 / Validate if all factor IDs are in official registry"""
    registry_ids = set(get_factor_ids())
    input_ids = set(factor_ids)

    unauthorized = input_ids - registry_ids
    if unauthorized:
        print(f"❌ 发现未授权因子: {unauthorized}")
        return False

    print(f"✅ 所有因子都在官方清单中: {factor_ids}")
    return True


def is_strict_mode() -> bool:
    """检查是否启用严格模式 / Check if strict mode is enabled"""
    return SYSTEM_CONFIG.get("strict_mode", True)


def get_system_version() -> str:
    """获取系统版本 / Get system version"""
    return SYSTEM_CONFIG.get("version", "1.0")


# 导出的公共接口 / Public API
__all__ = [
    "FACTOR_CONFIG",
    "FACTOR_METADATA",
    "SYSTEM_CONFIG",
    "get_factor_ids",
    "get_factor_config",
    "get_factor_metadata",
    "validate_factor_ids",
    "is_strict_mode",
    "get_system_version",
]


# 模块初始化验证 / Module Initialization Validation
def _validate_configuration():
    """初始化时验证配置的完整性 / Validate configuration integrity on initialization"""
    # 检查所有因子都有完整的配置
    for factor_id in get_factor_ids():
        if factor_id not in FACTOR_METADATA:
            raise ValueError(f"因子 '{factor_id}' 缺少元数据配置")

        config = FACTOR_CONFIG[factor_id]
        required_keys = ["function", "parameters", "output_columns"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"因子 '{factor_id}' 缺少必需的配置项: {key}")

    print(f"✅ 因子配置验证通过 - {len(get_factor_ids())} 个官方因子")


# 模块加载时自动验证 / Auto-validate on module load
_validate_configuration()
