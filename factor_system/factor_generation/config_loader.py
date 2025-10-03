#!/usr/bin/env python3
"""
配置加载器 - 支持YAML配置文件和动态配置管理
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from professional_factor_screener import ScreeningConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_from_yaml(config_path: Union[str, Path]) -> ScreeningConfig:
        """从YAML文件加载配置"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            logger.info(f"从YAML文件加载配置: {config_path}")

            # 转换配置数据
            screening_config = ConfigLoader._convert_yaml_to_config(config_data)

            return screening_config

        except Exception as e:
            logger.error(f"加载YAML配置失败: {str(e)}")
            raise

    @staticmethod
    def _convert_yaml_to_config(config_data: Dict[str, Any]) -> ScreeningConfig:
        """将YAML数据转换为ScreeningConfig"""

        # 提取各部分配置
        multi_horizon = config_data.get("multi_horizon_ic", {})
        statistical = config_data.get("statistical_testing", {})
        independence = config_data.get("independence_analysis", {})
        trading_costs = config_data.get("trading_costs", {})
        thresholds = config_data.get("screening_thresholds", {})
        performance = config_data.get("performance", {})
        weights = config_data.get("scoring_weights", {})

        # 创建ScreeningConfig对象
        config = ScreeningConfig(
            # 多周期IC参数
            ic_horizons=multi_horizon.get("horizons", [1, 3, 5, 10, 20]),
            min_sample_size=multi_horizon.get("min_sample_size", 100),
            rolling_window=multi_horizon.get("rolling_window", 60),
            # 统计显著性参数
            alpha_level=statistical.get("alpha_level", 0.05),
            fdr_method=statistical.get("fdr_method", "benjamini_hochberg"),
            # 独立性分析参数
            vif_threshold=independence.get("vif_threshold", 5.0),
            correlation_threshold=independence.get("correlation_threshold", 0.8),
            base_factors=independence.get(
                "base_factors", ["MA5", "MA10", "RSI14", "MACD_12_26_9"]
            ),
            # 交易成本参数
            commission_rate=trading_costs.get("commission_rate", 0.002),
            slippage_bps=trading_costs.get("slippage_bps", 5.0),
            market_impact_coeff=trading_costs.get("market_impact_coeff", 0.1),
            # 筛选阈值
            min_ic_threshold=thresholds.get("min_ic_threshold", 0.02),
            min_ir_threshold=thresholds.get("min_ir_threshold", 0.5),
            min_stability_threshold=thresholds.get("min_stability_threshold", 0.6),
            max_vif_threshold=thresholds.get("max_vif_threshold", 10.0),
            max_cost_threshold=thresholds.get("max_cost_threshold", 0.01),
            # 性能参数
            max_workers=performance.get("max_workers", 4),
            cache_enabled=performance.get("cache_enabled", True),
            memory_limit_mb=performance.get("memory_limit_mb", 2048),
            # 评分权重
            weight_predictive=weights.get("predictive_power", 0.35),
            weight_stability=weights.get("stability", 0.25),
            weight_independence=weights.get("independence", 0.20),
            weight_practicality=weights.get("practicality", 0.15),
            weight_adaptability=weights.get("adaptability", 0.05),
        )

        return config

    @staticmethod
    def save_to_yaml(config: ScreeningConfig, output_path: Union[str, Path]) -> None:
        """将配置保存为YAML文件"""
        output_path = Path(output_path)

        # 转换为字典格式
        config_dict = ConfigLoader._convert_config_to_yaml(config)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

            logger.info(f"配置已保存到YAML文件: {output_path}")

        except Exception as e:
            logger.error(f"保存YAML配置失败: {str(e)}")
            raise

    @staticmethod
    def _convert_config_to_yaml(config: ScreeningConfig) -> Dict[str, Any]:
        """将ScreeningConfig转换为YAML格式"""

        config_dict = {
            "multi_horizon_ic": {
                "horizons": config.ic_horizons,
                "min_sample_size": config.min_sample_size,
                "rolling_window": config.rolling_window,
            },
            "statistical_testing": {
                "alpha_level": config.alpha_level,
                "fdr_method": config.fdr_method,
                "multiple_testing_correction": True,
            },
            "independence_analysis": {
                "vif_threshold": config.vif_threshold,
                "correlation_threshold": config.correlation_threshold,
                "base_factors": config.base_factors,
            },
            "trading_costs": {
                "commission_rate": config.commission_rate,
                "slippage_bps": config.slippage_bps,
                "market_impact_coeff": config.market_impact_coeff,
            },
            "screening_thresholds": {
                "min_ic_threshold": config.min_ic_threshold,
                "min_ir_threshold": config.min_ir_threshold,
                "min_stability_threshold": config.min_stability_threshold,
                "max_vif_threshold": config.max_vif_threshold,
                "max_cost_threshold": config.max_cost_threshold,
            },
            "performance": {
                "max_workers": config.max_workers,
                "cache_enabled": config.cache_enabled,
                "memory_limit_mb": config.memory_limit_mb,
                "timeout_seconds": 300,
            },
            "scoring_weights": {
                "predictive_power": config.weight_predictive,
                "stability": config.weight_stability,
                "independence": config.weight_independence,
                "practicality": config.weight_practicality,
                "adaptability": config.weight_adaptability,
            },
            "logging": {"level": "INFO", "file_enabled": True, "console_enabled": True},
            "output": {
                "save_detailed_report": True,
                "save_summary_report": True,
                "export_formats": ["csv", "json"],
                "include_metadata": True,
            },
        }

        return config_dict

    @staticmethod
    def load_from_json(config_path: Union[str, Path]) -> ScreeningConfig:
        """从JSON文件加载配置"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            logger.info(f"从JSON文件加载配置: {config_path}")

            # 转换配置数据
            screening_config = ConfigLoader._convert_yaml_to_config(config_data)

            return screening_config

        except Exception as e:
            logger.error(f"加载JSON配置失败: {str(e)}")
            raise

    @staticmethod
    def save_to_json(config: ScreeningConfig, output_path: Union[str, Path]) -> None:
        """将配置保存为JSON文件"""
        output_path = Path(output_path)

        # 转换为字典格式
        config_dict = ConfigLoader._convert_config_to_yaml(config)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"配置已保存到JSON文件: {output_path}")

        except Exception as e:
            logger.error(f"保存JSON配置失败: {str(e)}")
            raise

    @staticmethod
    def validate_config(config: ScreeningConfig) -> Dict[str, Any]:
        """验证配置的有效性"""
        validation_result = {"is_valid": True, "warnings": [], "errors": []}

        # 检查IC周期
        if not config.ic_horizons or len(config.ic_horizons) == 0:
            validation_result["errors"].append("IC周期不能为空")
            validation_result["is_valid"] = False

        if any(h <= 0 for h in config.ic_horizons):
            validation_result["errors"].append("IC周期必须为正数")
            validation_result["is_valid"] = False

        # 检查样本量
        if config.min_sample_size < 30:
            validation_result["warnings"].append("最小样本量过小，建议至少50")

        # 检查显著性水平
        if not (0 < config.alpha_level < 1):
            validation_result["errors"].append("显著性水平必须在0和1之间")
            validation_result["is_valid"] = False

        # 检查权重
        total_weight = (
            config.weight_predictive
            + config.weight_stability
            + config.weight_independence
            + config.weight_practicality
            + config.weight_adaptability
        )

        if abs(total_weight - 1.0) > 0.01:
            validation_result["errors"].append(
                f"权重总和必须为1.0，当前为{total_weight:.3f}"
            )
            validation_result["is_valid"] = False

        # 检查阈值
        if config.min_ic_threshold < 0:
            validation_result["errors"].append("IC阈值不能为负数")
            validation_result["is_valid"] = False

        if config.min_ir_threshold < 0:
            validation_result["errors"].append("IR阈值不能为负数")
            validation_result["is_valid"] = False

        # 检查交易成本
        if config.commission_rate < 0:
            validation_result["errors"].append("佣金率不能为负数")
            validation_result["is_valid"] = False

        if config.commission_rate > 0.01:
            validation_result["warnings"].append("佣金率过高，可能影响策略收益")

        # 检查性能参数
        if config.max_workers < 1:
            validation_result["errors"].append("工作线程数必须至少为1")
            validation_result["is_valid"] = False

        if config.memory_limit_mb < 512:
            validation_result["warnings"].append("内存限制过低，可能影响性能")

        return validation_result

    @staticmethod
    def create_preset_configs() -> Dict[str, ScreeningConfig]:
        """创建预设配置"""

        presets = {}

        # 默认配置
        presets["default"] = ScreeningConfig()

        # 保守型配置
        presets["conservative"] = ScreeningConfig(
            ic_horizons=[5, 10, 20],
            min_sample_size=200,
            alpha_level=0.01,
            fdr_method="bonferroni",
            min_ic_threshold=0.03,
            min_ir_threshold=0.8,
            weight_stability=0.40,
            weight_predictive=0.30,
            weight_independence=0.20,
            weight_practicality=0.10,
            weight_adaptability=0.00,
        )

        # 激进型配置
        presets["aggressive"] = ScreeningConfig(
            ic_horizons=[1, 2, 3],
            min_sample_size=50,
            alpha_level=0.10,
            fdr_method="benjamini_hochberg",
            min_ic_threshold=0.015,
            min_ir_threshold=0.3,
            weight_predictive=0.50,
            weight_adaptability=0.20,
            weight_stability=0.15,
            weight_independence=0.10,
            weight_practicality=0.05,
        )

        # 高频交易配置
        presets["high_frequency"] = ScreeningConfig(
            ic_horizons=[1, 2],
            min_sample_size=30,
            rolling_window=20,
            alpha_level=0.15,
            min_ic_threshold=0.01,
            min_ir_threshold=0.2,
            commission_rate=0.001,
            slippage_bps=2.0,
            weight_predictive=0.40,
            weight_adaptability=0.30,
            weight_practicality=0.20,
            weight_stability=0.05,
            weight_independence=0.05,
        )

        # 长期投资配置
        presets["long_term"] = ScreeningConfig(
            ic_horizons=[10, 20, 30, 60],
            min_sample_size=300,
            rolling_window=120,
            alpha_level=0.01,
            fdr_method="bonferroni",
            min_ic_threshold=0.05,
            min_ir_threshold=1.0,
            weight_stability=0.50,
            weight_predictive=0.25,
            weight_independence=0.15,
            weight_practicality=0.05,
            weight_adaptability=0.05,
        )

        return presets


def demo_config_loader():
    """演示配置加载器功能"""
    print("=" * 80)
    print("配置加载器演示")
    print("=" * 80)

    # 1. 加载YAML配置
    print("\n1. 从YAML文件加载配置:")
    try:
        yaml_config_path = Path(__file__).parent / "config" / "screening_config.yaml"
        if yaml_config_path.exists():
            config = ConfigLoader.load_from_yaml(yaml_config_path)
            print(f"✅ YAML配置加载成功")
            print(f"   - IC周期: {config.ic_horizons}")
            print(f"   - 显著性水平: {config.alpha_level}")
            print(f"   - 权重分配: 预测{config.weight_predictive:.0%}")
        else:
            print(f"❌ YAML配置文件不存在: {yaml_config_path}")
    except Exception as e:
        print(f"❌ YAML配置加载失败: {str(e)}")

    # 2. 配置验证
    print(f"\n2. 配置验证:")
    default_config = ScreeningConfig()
    validation = ConfigLoader.validate_config(default_config)

    if validation["is_valid"]:
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")
        for error in validation["errors"]:
            print(f"   错误: {error}")

    for warning in validation["warnings"]:
        print(f"   警告: {warning}")

    # 3. 预设配置
    print(f"\n3. 预设配置:")
    presets = ConfigLoader.create_preset_configs()

    for name, config in presets.items():
        print(f"   📋 {name}: IC周期{config.ic_horizons}, α={config.alpha_level}")

    # 4. 保存配置示例
    print(f"\n4. 保存配置示例:")
    try:
        output_dir = Path(__file__).parent / "config"
        output_dir.mkdir(exist_ok=True)

        # 保存预设配置
        for name, config in presets.items():
            if name != "default":
                output_path = output_dir / f"{name}_config.yaml"
                ConfigLoader.save_to_yaml(config, output_path)
                print(f"✅ {name}配置已保存: {output_path.name}")

    except Exception as e:
        print(f"❌ 配置保存失败: {str(e)}")


if __name__ == "__main__":
    demo_config_loader()
