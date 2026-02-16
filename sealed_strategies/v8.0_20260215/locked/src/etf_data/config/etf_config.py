#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器配置管理模块
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .etf_config_standalone import ETFConfig


def get_config_dir() -> Path:
    """获取配置目录"""
    # Return the project root configs directory
    # src/etf_data/config/etf_config.py -> .../configs
    return Path(__file__).resolve().parents[3] / "configs"


def load_config(config_name: str = "etf_config") -> ETFConfig:
    """
    加载配置文件

    Args:
        config_name: 配置文件名（不含扩展名）

    Returns:
        ETF配置对象
    """
    config_file = get_config_dir() / f"{config_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    return ETFConfig.from_yaml(str(config_file))


def load_custom_config(config_path: str) -> ETFConfig:
    """
    加载自定义配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        ETF配置对象
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    return ETFConfig.from_yaml(config_path)


def save_config(config: ETFConfig, config_name: str = "custom_config") -> str:
    """
    保存配置文件

    Args:
        config: ETF配置对象
        config_name: 配置文件名（不含扩展名）

    Returns:
        保存的配置文件路径
    """
    config_dir = get_config_dir()
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / f"{config_name}.yaml"
    config.save_yaml(str(config_file))

    return str(config_file)


def get_default_configs() -> Dict[str, ETFConfig]:
    """
    获取所有默认配置

    Returns:
        配置名字典
    """
    configs = {}

    # 标准配置
    try:
        configs["default"] = load_config("etf_config")
    except FileNotFoundError:
        # 如果文件不存在，创建默认配置
        configs["default"] = ETFConfig()

    # 快速配置
    try:
        configs["quick"] = load_config("quick_config")
    except FileNotFoundError:
        # 快速下载配置
        configs["quick"] = ETFConfig(
            years_back=1,
            max_retries=2,
            retry_delay=0.5,
            request_delay=0.1,
            batch_size=20,
        )

    # 完整配置
    try:
        configs["full"] = load_config("full_config")
    except FileNotFoundError:
        # 完整下载配置
        configs["full"] = ETFConfig(
            years_back=3,
            max_retries=5,
            retry_delay=2.0,
            request_delay=0.3,
            batch_size=20,
            timeout=45,
        )

    return configs


def list_available_configs() -> Dict[str, str]:
    """
    列出可用的配置文件

    Returns:
        配置名字典 {配置名: 配置文件路径}
    """
    configs = {}
    config_dir = get_config_dir()

    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = str(config_file)

    return configs


def create_user_config(user_name: str, **kwargs) -> str:
    """
    创建用户自定义配置

    Args:
        user_name: 用户名
        **kwargs: 配置参数

    Returns:
        创建的配置文件路径
    """
    # 从默认配置开始
    base_config = ETFConfig()

    # 应用用户参数
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    # 保存用户配置
    config_name = f"user_{user_name}_config"
    config_path = save_config(base_config, config_name)

    return config_path


def validate_config(config: ETFConfig) -> Dict[str, Any]:
    """
    验证配置的有效性

    Args:
        config: ETF配置对象

    Returns:
        验证结果字典
    """
    result = {"valid": True, "errors": [], "warnings": []}

    # 检查Token
    if config.source.value == "tushare" and not config.tushare_token:
        result["errors"].append("Tushare Token未设置")
        result["valid"] = False

    # 检查数据目录
    try:
        data_dir = Path(config.base_dir)
        if not data_dir.parent.exists():
            result["warnings"].append(f"数据目录的父目录不存在: {data_dir.parent}")
    except Exception as e:
        result["errors"].append(f"数据目录路径无效: {e}")
        result["valid"] = False

    # 检查日期范围
    try:
        if config.start_date and len(config.start_date) != 8:
            result["errors"].append("开始日期格式无效，应为YYYYMMDD")
            result["valid"] = False

        if config.end_date and len(config.end_date) != 8:
            result["errors"].append("结束日期格式无效，应为YYYYMMDD")
            result["valid"] = False
    except Exception as e:
        result["errors"].append(f"日期验证失败: {e}")
        result["valid"] = False

    # 检查下载类型
    if not config.download_types:
        result["errors"].append("至少需要指定一种下载类型")
        result["valid"] = False

    # 检查保存格式
    if config.save_format not in ["parquet", "csv"]:
        result["errors"].append(f"不支持的保存格式: {config.save_format}")
        result["valid"] = False

    # 检查API参数
    if config.max_retries < 0:
        result["errors"].append("最大重试次数不能为负数")
        result["valid"] = False

    if config.retry_delay < 0:
        result["errors"].append("重试延迟不能为负数")
        result["valid"] = False

    if config.request_delay < 0:
        result["errors"].append("请求延迟不能为负数")
        result["valid"] = False

    if config.batch_size <= 0:
        result["errors"].append("批处理大小必须为正数")
        result["valid"] = False

    return result


def print_config_summary(config: ETFConfig, config_name: str = "配置"):
    """打印配置摘要"""
    print(f"=== {config_name}摘要 ===")
    print(f"数据源: {config.source.value}")
    print(f"数据目录: {config.base_dir}")
    print(f"时间范围: {config.start_date} ~ {config.end_date}")
    print(f"下载类型: {[dt.value for dt in config.download_types]}")
    print(f"保存格式: {config.save_format}")
    print(f"批处理大小: {config.batch_size}")
    print(f"最大重试: {config.max_retries}")
    print(f"请求延迟: {config.request_delay}s")
    print(f"详细输出: {config.verbose}")


def setup_environment():
    """设置环境变量和依赖检查"""
    issues = []

    # 检查Tushare Token
    if not os.getenv("TUSHARE_TOKEN"):
        issues.append("环境变量 TUSHARE_TOKEN 未设置")

    # 检查Python包
    try:
        import tushare
    except ImportError:
        issues.append("未安装tushare包，请运行: pip install tushare")

    try:
        import pandas
    except ImportError:
        issues.append("未安装pandas包，请运行: pip install pandas")

    try:
        import pyarrow
    except ImportError:
        issues.append("未安装pyarrow包，请运行: pip install pyarrow")

    # 检查数据目录权限
    try:
        data_dir = Path("raw/ETF")
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        issues.append(f"数据目录写入权限问题: {e}")

    if issues:
        print("=== 环境检查发现问题 ===")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        print("\n请解决这些问题后重试")
        return False

    print("✅ 环境检查通过")
    return True
