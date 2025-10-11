#!/usr/bin/env python3
"""
快速启动配置管理器
作者：量化首席工程师
版本：1.0.0
日期：2025-09-30

功能：
1. 统一配置管理（YAML/JSON）
2. 预设配置模板
3. 参数验证和默认值
4. 动态配置生成
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

# 导入路径管理工具
try:
    from ..utils import get_factor_output_dir, get_screening_results_dir
except ImportError:
    # 回退到相对路径（保持向后兼容）
    def get_factor_output_dir():
        return Path("../factor_output")

    def get_screening_results_dir():
        return Path("./screening_results")


logger = logging.getLogger(__name__)


@dataclass
class ScreeningConfig:
    """筛选配置类"""

    # 基础配置
    name: str = "default"
    description: str = "默认筛选配置"

    # 🔧 核心路径三元组（Linus式简化）
    data_root: str = str(get_factor_output_dir())  # 数据输入根目录
    output_root: str = str(get_screening_results_dir())  # 结果输出根目录
    log_root: str = ""  # 日志根目录（留空则与output_root一致）

    # 废弃字段（向后兼容，请使用data_root）
    raw_data_root: str = ""  # @deprecated 使用data_root替代

    # 股票配置
    symbols: List[str] = field(default_factory=lambda: ["0700.HK"])
    timeframes: List[str] = field(default_factory=lambda: ["60min"])

    # IC分析配置
    ic_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    min_sample_size: int = 100
    alpha_level: float = 0.05  # 修正参数名
    fdr_method: str = "benjamini_hochberg"

    # 稳定性配置
    rolling_window: int = 60  # 修正参数名
    min_ic_threshold: float = 0.015
    min_ir_threshold: float = 0.35
    min_robustness_score: float = 0.6

    # 独立性配置
    vif_threshold: float = 5.0
    correlation_threshold: float = 0.8
    base_factors: List[str] = field(
        default_factory=lambda: ["MA5", "MA10", "RSI14", "MACD_12_26_9"]
    )

    # 交易成本参数
    commission_rate: float = 0.002  # 0.2%佣金
    slippage_bps: float = 5.0  # 5bp滑点
    market_impact_coeff: float = 0.1

    # 筛选阈值
    min_stability_threshold: float = 0.6
    max_vif_threshold: float = 10.0
    max_cost_threshold: float = 0.01  # 1%最大交易成本

    # 数据质量参数
    max_missing_ratio: float = 0.8  # 最大缺失比例
    min_data_points: int = 50  # 最小数据点数
    min_momentum_samples: int = 120  # 动量分析最小样本数
    factor_change_threshold: float = 0.05  # 因子变化阈值
    high_rank_threshold: float = 0.8  # 高排名阈值
    progress_report_interval: int = 50  # 进度报告间隔

    # 综合评分权重
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "predictive_power": 0.35,
            "stability": 0.25,
            "independence": 0.20,
            "practicality": 0.10,
            "short_term_fitness": 0.10,
        }
    )

    # 🚀 时间框架自适应配置（P0修复：不同时间框架分层标准）
    enable_timeframe_adaptive: bool = True  # 启用时间框架自适应

    # 🔧 统一时间框架自适应配置（Linus式消灭冗余）
    timeframe_adaptive_config: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "1min": {"alpha": 0.05, "tier1": 0.85, "tier2": 0.70, "tier3": 0.50},
            "2min": {"alpha": 0.05, "tier1": 0.85, "tier2": 0.70, "tier3": 0.50},
            "3min": {"alpha": 0.05, "tier1": 0.85, "tier2": 0.70, "tier3": 0.50},
            "5min": {"alpha": 0.05, "tier1": 0.82, "tier2": 0.68, "tier3": 0.48},
            "15min": {"alpha": 0.07, "tier1": 0.80, "tier2": 0.65, "tier3": 0.45},
            "30min": {"alpha": 0.07, "tier1": 0.78, "tier2": 0.62, "tier3": 0.42},
            "60min": {"alpha": 0.08, "tier1": 0.75, "tier2": 0.58, "tier3": 0.40},
            "120min": {"alpha": 0.08, "tier1": 0.75, "tier2": 0.58, "tier3": 0.40},
            "2h": {"alpha": 0.10, "tier1": 0.72, "tier2": 0.55, "tier3": 0.38},
            "240min": {"alpha": 0.10, "tier1": 0.70, "tier2": 0.52, "tier3": 0.35},
            "4h": {"alpha": 0.10, "tier1": 0.70, "tier2": 0.52, "tier3": 0.35},
            "1day": {"alpha": 0.10, "tier1": 0.68, "tier2": 0.50, "tier3": 0.32},
        }
    )

    # @deprecated 废弃字段（向后兼容）
    timeframe_alpha_map: Dict[str, float] = field(
        default_factory=dict
    )  # 使用timeframe_adaptive_config替代

    # 样本量权重配置（Phase 2: 分段化折扣，抑制小样本虚高）
    sample_weight_params: Dict[str, Union[int, float, Dict]] = field(
        default_factory=lambda: {
            "enable": True,  # 启用样本量权重
            "min_full_weight_samples": 1000,  # 降低阈值，减少折扣
            "weight_power": 0.35,  # Phase 2: 从 0.2 提升至 0.35
            "predictive_weight_power": 0.20,  # Phase 2: 从 0.15 提升至 0.20
            "affected_dimensions": [
                "stability",
                "independence",
                "practicality",
            ],  # 移除predictive和adaptability
            # Phase 2: 新增分段阈值
            "thresholds": {
                "high_confidence": 2000,  # 样本>2000，无额外折扣
                "medium_confidence": 500,  # 500-2000，轻度折扣
                "low_confidence": 100,  # 100-500，重度折扣
            },
        }
    )

    # 时间框架分层高分阈值（避免系统性劣势）
    timeframe_high_score_map: Dict[str, float] = field(
        default_factory=lambda: {
            # 高频：标准阈值
            "1min": 0.60,
            "2min": 0.60,
            "3min": 0.60,
            "5min": 0.60,
            # 中频：略微降低
            "15min": 0.58,
            "30min": 0.57,
            "60min": 0.56,
            "120min": 0.55,
            "2h": 0.55,
            "240min": 0.54,
            "4h": 0.54,
            "1day": 0.53,
        }
    )

    # 时间框架自适应Tier阈值（5个阈值：tier1/tier2/tier3/upgrade_tier2/upgrade_tier1）
    timeframe_tier_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # 高频：严格阈值（提高筛选标准）
            "1min": {
                "tier1": 0.85,
                "tier2": 0.70,
                "tier3": 0.50,
                "upgrade_tier2": 0.65,
                "upgrade_tier1": 0.80,
            },
            "2min": {
                "tier1": 0.85,
                "tier2": 0.70,
                "tier3": 0.50,
                "upgrade_tier2": 0.65,
                "upgrade_tier1": 0.80,
            },
            "3min": {
                "tier1": 0.85,
                "tier2": 0.70,
                "tier3": 0.50,
                "upgrade_tier2": 0.65,
                "upgrade_tier1": 0.80,
            },
            "5min": {
                "tier1": 0.82,
                "tier2": 0.68,
                "tier3": 0.48,
                "upgrade_tier2": 0.62,
                "upgrade_tier1": 0.78,
            },
            "15min": {
                "tier1": 0.80,
                "tier2": 0.65,
                "tier3": 0.45,
                "upgrade_tier2": 0.58,
                "upgrade_tier1": 0.75,
            },
            "30min": {
                "tier1": 0.78,
                "tier2": 0.62,
                "tier3": 0.42,
                "upgrade_tier2": 0.55,
                "upgrade_tier1": 0.72,
            },
            # 低频：适中阈值（提高标准但保持合理性）
            "60min": {
                "tier1": 0.75,
                "tier2": 0.58,
                "tier3": 0.40,
                "upgrade_tier2": 0.52,
                "upgrade_tier1": 0.70,
            },
            "120min": {
                "tier1": 0.72,
                "tier2": 0.55,
                "tier3": 0.38,
                "upgrade_tier2": 0.48,
                "upgrade_tier1": 0.68,
            },
            "2h": {
                "tier1": 0.72,
                "tier2": 0.55,
                "tier3": 0.38,
                "upgrade_tier2": 0.48,
                "upgrade_tier1": 0.68,
            },
            "240min": {
                "tier1": 0.70,
                "tier2": 0.52,
                "tier3": 0.35,
                "upgrade_tier2": 0.45,
                "upgrade_tier1": 0.65,
            },
            "4h": {
                "tier1": 0.70,
                "tier2": 0.52,
                "tier3": 0.35,
                "upgrade_tier2": 0.45,
                "upgrade_tier1": 0.65,
            },
            "1day": {
                "tier1": 0.68,
                "tier2": 0.50,
                "tier3": 0.32,
                "upgrade_tier2": 0.42,
                "upgrade_tier1": 0.62,
            },
        }
    )

    # 对齐失败策略
    alignment_failure_strategy: str = "warn"  # "warn" | "fail_fast" | "fallback"

    # 🎯 最优解配置：预测能力核心化评分
    use_optimal_fair_scoring: bool = True  # 启用最优解公平评分
    optimal_scoring_config_path: str = (
        "./configs/optimal_fair_scoring_config.yaml"  # 最优解配置路径
    )

    # 🔧 样本量自适应Tier阈值（从最优解配置中加载）
    adaptive_tier_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            # 默认值（会被配置文件覆盖）
            "1min": {
                "tier1": 0.80,
                "tier2": 0.70,
                "tier3": 0.55,
                "upgrade_tier2": 0.65,
                "upgrade_tier1": 0.77,
            },
            "2min": {
                "tier1": 0.78,
                "tier2": 0.62,
                "tier3": 0.48,
                "upgrade_tier2": 0.58,
                "upgrade_tier1": 0.75,
            },
            "3min": {
                "tier1": 0.77,
                "tier2": 0.60,
                "tier3": 0.46,
                "upgrade_tier2": 0.56,
                "upgrade_tier1": 0.73,
            },
            "5min": {
                "tier1": 0.75,
                "tier2": 0.58,
                "tier3": 0.44,
                "upgrade_tier2": 0.54,
                "upgrade_tier1": 0.71,
            },
            "15min": {
                "tier1": 0.72,
                "tier2": 0.55,
                "tier3": 0.42,
                "upgrade_tier2": 0.51,
                "upgrade_tier1": 0.68,
            },
            "30min": {
                "tier1": 0.70,
                "tier2": 0.52,
                "tier3": 0.40,
                "upgrade_tier2": 0.48,
                "upgrade_tier1": 0.65,
            },
            "60min": {
                "tier1": 0.68,
                "tier2": 0.50,
                "tier3": 0.38,
                "upgrade_tier2": 0.45,
                "upgrade_tier1": 0.62,
            },
            "120min": {
                "tier1": 0.66,
                "tier2": 0.48,
                "tier3": 0.36,
                "upgrade_tier2": 0.43,
                "upgrade_tier1": 0.60,
            },
            "2h": {
                "tier1": 0.66,
                "tier2": 0.48,
                "tier3": 0.36,
                "upgrade_tier2": 0.43,
                "upgrade_tier1": 0.60,
            },
            "240min": {
                "tier1": 0.65,
                "tier2": 0.47,
                "tier3": 0.35,
                "upgrade_tier2": 0.42,
                "upgrade_tier1": 0.58,
            },
            "4h": {
                "tier1": 0.65,
                "tier2": 0.47,
                "tier3": 0.35,
                "upgrade_tier2": 0.42,
                "upgrade_tier1": 0.58,
            },
            "1day": {
                "tier1": 0.64,
                "tier2": 0.46,
                "tier3": 0.34,
                "upgrade_tier2": 0.41,
                "upgrade_tier1": 0.57,
            },
        }
    )

    # @deprecated 废弃路径字段（向后兼容，请使用核心路径三元组）
    factor_data_root: str = ""  # 使用data_root替代
    price_data_root: str = ""  # 使用data_root替代
    cache_root: str = ""  # 默认为output_root/cache

    # 并行处理配置
    max_workers: int = 4
    enable_parallel: bool = True

    # @deprecated 输出配置（向后兼容）
    output_dir: str = ""  # 废弃，使用output_root
    save_reports: bool = True
    save_detailed_metrics: bool = True
    log_level: str = "INFO"

    # P2-1修复：传统格式保存选项
    enable_legacy_format: bool = False  # 是否启用传统格式保存
    legacy_format_priority: bool = False  # 传统格式是否优先于增强格式

    # 性能配置
    memory_limit_gb: float = 8.0
    timeout_minutes: int = 60


@dataclass
class BatchConfig:
    """批量处理配置类"""

    # 批量任务配置
    batch_name: str = "batch_screening"
    description: str = "批量因子筛选任务"

    # 任务列表
    screening_configs: List[ScreeningConfig] = field(default_factory=list)

    # 全局设置
    global_data_root: Optional[str] = None
    global_output_dir: Optional[str] = None

    # 并行配置
    max_concurrent_tasks: int = 2
    enable_task_parallel: bool = True

    # 报告配置
    generate_summary_report: bool = True
    compare_results: bool = True

    # 错误处理
    continue_on_error: bool = True
    max_retries: int = 2


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()

        # 预设配置模板
        self.presets = self._create_presets()

    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"{__name__}.ConfigManager")
        return logger

    def _create_presets(self) -> Dict[str, ScreeningConfig]:
        """创建预设配置模板"""
        presets = {}

        # 1. 默认配置
        presets["default"] = ScreeningConfig(
            name="default", description="默认筛选配置 - 平衡的参数设置"
        )

        # 2. 快速配置（适合测试）
        presets["quick"] = ScreeningConfig(
            name="quick",
            description="快速筛选配置 - 适合测试和快速验证",
            ic_horizons=[1, 3, 5],
            rolling_window=30,
            min_sample_size=50,
            max_workers=2,
        )

        # 3. 深度配置（全面分析）
        presets["deep"] = ScreeningConfig(
            name="deep",
            description="深度筛选配置 - 全面的因子分析",
            ic_horizons=[1, 3, 5, 10, 20, 30],
            rolling_window=120,
            min_sample_size=200,
            min_ic_threshold=0.01,
            min_ir_threshold=0.25,
            vif_threshold=3.0,
            correlation_threshold=0.7,
            max_workers=6,
        )

        # 4. 高频配置（短周期优化）
        presets["high_freq"] = ScreeningConfig(
            name="high_freq",
            description="高频筛选配置 - 优化短周期因子",
            timeframes=["1min", "5min", "15min"],
            ic_horizons=[1, 2, 3, 5],
            rolling_window=30,
            weights={
                "predictive_power": 0.25,
                "stability": 0.20,
                "independence": 0.15,
                "practicality": 0.15,
                "short_term_fitness": 0.25,
            },
        )

        # 5. 多时间框架配置
        presets["multi_timeframe"] = ScreeningConfig(
            name="multi_timeframe",
            description="多时间框架筛选配置 - 预设路径",
            data_root=str(get_factor_output_dir()),
            raw_data_root="",  # 废弃字段，使用data_root
            output_dir="./因子筛选",
            timeframes=["5min", "15min", "30min", "60min", "daily"],
            ic_horizons=[1, 3, 5, 10],
            max_workers=8,
            enable_parallel=True,
        )

        return presets

    def get_preset(self, preset_name: str) -> ScreeningConfig:
        """获取预设配置"""
        if preset_name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"未知的预设配置: {preset_name}. 可用配置: {available}")

        return self.presets[preset_name]

    def list_presets(self) -> Dict[str, str]:
        """列出所有预设配置"""
        return {name: config.description for name, config in self.presets.items()}

    def save_config(
        self,
        config: Union[ScreeningConfig, BatchConfig],
        filename: str,
        format: str = "yaml",
    ) -> Path:
        """保存配置到文件"""
        file_path = self.config_dir / f"{filename}.{format}"

        config_dict = asdict(config)

        if format.lower() == "yaml":
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
        elif format.lower() == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的格式: {format}")

        self.logger.info(f"配置已保存: {file_path}")
        return file_path

    def load_config(
        self, file_path: Union[str, Path], config_type: str = "screening"
    ) -> Union[ScreeningConfig, BatchConfig]:
        """从文件加载配置"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        # 根据文件扩展名确定格式
        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        # 创建配置对象
        if config_type.lower() == "screening":
            return ScreeningConfig(**config_dict)
        elif config_type.lower() == "batch":
            # 处理嵌套的screening_configs
            if "screening_configs" in config_dict:
                screening_configs = []
                for sc_dict in config_dict["screening_configs"]:
                    screening_configs.append(ScreeningConfig(**sc_dict))
                config_dict["screening_configs"] = screening_configs
            return BatchConfig(**config_dict)
        else:
            raise ValueError(f"不支持的配置类型: {config_type}")

    def create_batch_config(
        self,
        batch_name: str,
        symbols: List[str],
        timeframes: List[str],
        preset: str = "default",
    ) -> BatchConfig:
        """创建批量配置"""
        base_config = self.get_preset(preset)

        screening_configs = []

        # 为每个股票和时间框架组合创建配置
        for symbol in symbols:
            for timeframe in timeframes:
                config = ScreeningConfig(
                    name=f"{symbol}_{timeframe}",
                    description=f"{symbol} {timeframe} 筛选配置",
                    symbols=[symbol],
                    timeframes=[timeframe],
                    # 继承基础配置的其他参数
                    ic_horizons=base_config.ic_horizons,
                    min_sample_size=base_config.min_sample_size,
                    alpha_level=base_config.alpha_level,
                    fdr_method=base_config.fdr_method,
                    rolling_window=base_config.rolling_window,
                    min_ic_threshold=base_config.min_ic_threshold,
                    min_ir_threshold=base_config.min_ir_threshold,
                    min_robustness_score=base_config.min_robustness_score,
                    vif_threshold=base_config.vif_threshold,
                    correlation_threshold=base_config.correlation_threshold,
                    base_factors=base_config.base_factors.copy(),
                    commission_rate=base_config.commission_rate,
                    slippage_bps=base_config.slippage_bps,
                    market_impact_coeff=base_config.market_impact_coeff,
                    min_stability_threshold=base_config.min_stability_threshold,
                    max_vif_threshold=base_config.max_vif_threshold,
                    max_cost_threshold=base_config.max_cost_threshold,
                    weights=base_config.weights.copy(),
                    max_workers=base_config.max_workers,
                    enable_parallel=base_config.enable_parallel,
                    output_dir=base_config.output_dir,
                    save_reports=base_config.save_reports,
                    save_detailed_metrics=base_config.save_detailed_metrics,
                    log_level=base_config.log_level,
                    memory_limit_gb=base_config.memory_limit_gb,
                    timeout_minutes=base_config.timeout_minutes,
                )
                screening_configs.append(config)

        batch_config = BatchConfig(
            batch_name=batch_name,
            description=f"批量筛选任务: {len(symbols)}个股票 x {len(timeframes)}个时间框架",
            screening_configs=screening_configs,
        )

        return batch_config

    def validate_config(self, config: Union[ScreeningConfig, BatchConfig]) -> List[str]:
        """验证配置"""
        errors = []

        if isinstance(config, ScreeningConfig):
            # 验证筛选配置
            if not config.symbols:
                errors.append("symbols不能为空")

            if not config.timeframes:
                errors.append("timeframes不能为空")

            if not config.ic_horizons:
                errors.append("ic_horizons不能为空")

            if config.min_sample_size < 10:
                errors.append("min_sample_size不能小于10")

            if not (0 < config.alpha_level < 1):
                errors.append("alpha_level必须在0和1之间")

            if config.vif_threshold < 1:
                errors.append("vif_threshold不能小于1")

            if not (0 < config.correlation_threshold < 1):
                errors.append("correlation_threshold必须在0和1之间")

            # 验证权重（Linus式严格验证）
            weight_sum = sum(config.weights.values())
            if abs(weight_sum - 1.0) > 1e-8:
                errors.append(f"权重总和必须精确等于1.0，当前为{weight_sum:.10f}")

        elif isinstance(config, BatchConfig):
            # 验证批量配置
            if not config.screening_configs:
                errors.append("screening_configs不能为空")

            # 验证每个子配置
            for i, sc in enumerate(config.screening_configs):
                sub_errors = self.validate_config(sc)
                for error in sub_errors:
                    errors.append(f"screening_configs[{i}]: {error}")

        return errors

    def create_config_templates(self) -> None:
        """创建配置文件模板"""
        templates_dir = self.config_dir / "templates"
        templates_dir.mkdir(exist_ok=True)

        # 1. 创建单个筛选配置模板
        single_config = self.get_preset("default")
        self.save_config(single_config, "templates/single_screening_template", "yaml")

        # 2. 创建批量配置模板
        batch_config = self.create_batch_config(
            batch_name="example_batch",
            symbols=["0700.HK", "0005.HK"],
            timeframes=["30min", "60min"],
            preset="default",
        )
        self.save_config(batch_config, "templates/batch_screening_template", "yaml")

        # 3. 创建高频配置模板
        hf_config = self.get_preset("high_freq")
        self.save_config(hf_config, "templates/high_freq_template", "yaml")

        # 4. 创建深度分析配置模板
        deep_config = self.get_preset("deep")
        self.save_config(deep_config, "templates/deep_analysis_template", "yaml")

        self.logger.info(f"配置模板已创建在: {templates_dir}")

        # 创建说明文档
        readme_content = """# 配置文件模板说明

## 模板文件

1. **single_screening_template.yaml** - 单个筛选任务配置模板
2. **batch_screening_template.yaml** - 批量筛选任务配置模板
3. **high_freq_template.yaml** - 高频交易优化配置模板
4. **deep_analysis_template.yaml** - 深度分析配置模板

## 使用方法

1. 复制模板文件
2. 修改相关参数（股票代码、时间框架等）
3. 使用batch_screener.py加载配置运行

## 配置参数说明

### 基础参数
- `symbols`: 股票代码列表
- `timeframes`: 时间框架列表
- `ic_horizons`: IC计算周期

### 筛选参数
- `min_ic_threshold`: 最小IC阈值
- `vif_threshold`: VIF阈值
- `correlation_threshold`: 相关性阈值

### 权重配置
- `weights`: 5维度评分权重分配

详细参数说明请参考主文档。
"""

        with open(templates_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

    def get_config_summary(self, config: Union[ScreeningConfig, BatchConfig]) -> str:
        """获取配置摘要"""
        if isinstance(config, ScreeningConfig):
            return """
筛选配置摘要:
- 名称: {config.name}
- 描述: {config.description}
- 股票: {config.symbols}
- 时间框架: {config.timeframes}
- IC周期: {config.ic_horizons}
- 最小样本: {config.min_sample_size}
- 并行工作数: {config.max_workers}
"""
        elif isinstance(config, BatchConfig):
            symbols = set()
            timeframes = set()
            for sc in config.screening_configs:
                symbols.update(sc.symbols)
                timeframes.update(sc.timeframes)

            return """
批量配置摘要:
- 任务名称: {config.batch_name}
- 描述: {config.description}
- 总任务数: {total_tasks}
- 涉及股票: {sorted(symbols)}
- 涉及时间框架: {sorted(timeframes)}
- 最大并发: {config.max_concurrent_tasks}
"""


if __name__ == "__main__":
    # 示例用法
    manager = ConfigManager()

    # 创建配置模板
    manager.create_config_templates()

    # 列出预设配置
    print("可用预设配置:")
    for name, desc in manager.list_presets().items():
        print(f"  {name}: {desc}")

    # 创建批量配置示例
    batch_config = manager.create_batch_config(
        batch_name="test_batch",
        symbols=["0700.HK", "0005.HK"],
        timeframes=["30min", "60min"],
        preset="quick",
    )

    print(manager.get_config_summary(batch_config))
