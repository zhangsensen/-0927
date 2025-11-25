#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF横截面因子筛选配置系统 - Linus工程风格"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def get_project_root() -> Path:
    """获取项目根目录（深度量化0927）"""
    current = Path(__file__).resolve().parent
    # 从 02_因子筛选 向上找到包含 raw/ 的目录
    while current.parent != current:
        if (current / "raw").exists() or (current.parent / "raw").exists():
            if (current / "raw").exists():
                return current
            else:
                return current.parent
        current = current.parent
    # 兜底：使用环境变量或当前工作目录
    return Path(os.getenv("PROJECT_ROOT", Path.cwd()))


@dataclass
class DataSourceConfig:
    """数据源配置"""

    price_dir: Path
    panel_file: Path
    price_columns: List[str] = field(default_factory=lambda: ["trade_date", "close"])
    file_pattern: str = "*.parquet"
    symbol_extract_method: str = "stem_split"  # stem_split, regex, custom

    def __post_init__(self):
        # 转换为Path对象
        if isinstance(self.price_dir, str):
            self.price_dir = Path(self.price_dir)
        if isinstance(self.panel_file, str):
            self.panel_file = Path(self.panel_file)

        # 若为相对路径，解析为项目根的绝对路径
        project_root = get_project_root()
        if not self.price_dir.is_absolute():
            self.price_dir = (project_root / self.price_dir).resolve()
        if not self.panel_file.is_absolute():
            self.panel_file = (project_root / self.panel_file).resolve()


@dataclass
class AnalysisConfig:
    """分析参数配置"""

    # IC分析周期
    ic_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # 样本量阈值
    min_observations: int = 30  # ETF小样本标准
    min_ranking_samples: int = 5  # 横截面排名最小样本
    min_ic_observations: int = 20  # IC计算最小观测值

    # 相关性分析
    correlation_method: str = "spearman"
    correlation_min_periods: int = 30

    # 数值计算
    epsilon_small: float = 1e-8  # IR计算小值
    stability_split_ratio: float = 0.5  # 稳定性分析分割比例


@dataclass
class ScreeningConfig:
    """筛选标准配置 - Linus优化"""

    # 基础筛选标准
    min_ic: float = 0.01  # 最小IC (1%)
    min_ir: float = 0.08  # 最小IR
    max_pvalue: float = 0.05  # 最大p值
    min_coverage: float = 0.75  # 最小覆盖率

    # 去重标准
    max_correlation: float = 0.75  # 从0.65放宽到0.75,避免误杀

    # FDR校正
    use_fdr: bool = True
    fdr_alpha: float = 0.10  # 10% FDR显著性水平

    # 因子数量控制
    force_include_factors: List[str] = field(default_factory=list)
    max_factors: int = 20  # 从15提升到20
    priority_metric: str = "ic_ir"

    # Linus新增: 混合策略配置
    use_period_specific_ic: bool = True  # 启用分周期IC筛选
    target_rebalance_period: int = 5  # 目标换仓周期(日)
    ic_period_for_screening: str = "ic_5d"  # 筛选用的IC列名

    # 分层评级阈值
    tier_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "core": {"ic": 0.025, "ir": 0.15},
            "supplement": {"ic": 0.015, "ir": 0.10},
            "research": {"ic": 0.0, "ir": 0.0},
        }
    )

    # 分层评级标签
    tier_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "core": "🟢 核心",
            "supplement": "🟡 补充",
            "research": "🔵 研究",
        }
    )


@dataclass
class OutputConfig:
    """输出配置"""

    output_dir: Path
    use_timestamp_subdir: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    subdir_prefix: str = "screening_"

    # 输出文件名
    files: Dict[str, str] = field(
        default_factory=lambda: {
            "ic_analysis": "ic_analysis.csv",
            "passed_factors": "passed_factors.csv",
            "screening_report": "screening_report.txt",
        }
    )

    # 报告格式
    include_factor_details: bool = True
    include_summary_statistics: bool = True
    encoding: str = "utf-8"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # 若为相对路径，解析为项目根的绝对路径
        project_root = get_project_root()
        if not self.output_dir.is_absolute():
            self.output_dir = (project_root / self.output_dir).resolve()


@dataclass
class ETFCrossSectionConfig:
    """ETF横截面因子筛选完整配置"""

    data_source: DataSourceConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(
            output_dir=Path("etf_rotation_system/data/results/screening")
        )
    )

    # 系统级控制
    debug_mode: bool = False
    progress_reporting: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ETFCrossSectionConfig":
        """从字典创建配置对象"""
        return cls(
            data_source=DataSourceConfig(**config_dict["data_source"]),
            analysis=AnalysisConfig(**config_dict.get("analysis", {})),
            screening=ScreeningConfig(**config_dict.get("screening", {})),
            output=OutputConfig(**config_dict.get("output", {})),
            debug_mode=config_dict.get("debug_mode", False),
            progress_reporting=config_dict.get("progress_reporting", True),
        )

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "ETFCrossSectionConfig":
        """从YAML文件加载配置"""
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_file: Path) -> None:
        """保存配置到YAML文件"""
        config_dict = {
            "data_source": {
                "price_dir": str(self.data_source.price_dir),
                "panel_file": str(self.data_source.panel_file),
                "price_columns": self.data_source.price_columns,
                "file_pattern": self.data_source.file_pattern,
                "symbol_extract_method": self.data_source.symbol_extract_method,
            },
            "analysis": {
                "ic_periods": self.analysis.ic_periods,
                "min_observations": self.analysis.min_observations,
                "min_ranking_samples": self.analysis.min_ranking_samples,
                "min_ic_observations": self.analysis.min_ic_observations,
                "correlation_method": self.analysis.correlation_method,
                "correlation_min_periods": self.analysis.correlation_min_periods,
                "epsilon_small": self.analysis.epsilon_small,
                "stability_split_ratio": self.analysis.stability_split_ratio,
            },
            "screening": {
                "min_ic": self.screening.min_ic,
                "min_ir": self.screening.min_ir,
                "max_pvalue": self.screening.max_pvalue,
                "min_coverage": self.screening.min_coverage,
                "max_correlation": self.screening.max_correlation,
                "use_fdr": self.screening.use_fdr,
                "fdr_alpha": self.screening.fdr_alpha,
                "tier_thresholds": self.screening.tier_thresholds,
                "tier_labels": self.screening.tier_labels,
            },
            "output": {
                "output_dir": str(self.output.output_dir),
                "use_timestamp_subdir": self.output.use_timestamp_subdir,
                "timestamp_format": self.output.timestamp_format,
                "subdir_prefix": self.output.subdir_prefix,
                "files": self.output.files,
                "include_factor_details": self.output.include_factor_details,
                "include_summary_statistics": self.output.include_summary_statistics,
                "encoding": self.output.encoding,
            },
            "debug_mode": self.debug_mode,
            "progress_reporting": self.progress_reporting,
        }

        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, allow_unicode=True, indent=2
            )


def get_latest_panel_file() -> Path:
    """获取最新生成的panel文件，若不存在则返回默认路径"""
    project_root = get_project_root()
    panels_dir = project_root / "etf_rotation_system/data/results/panels"

    if panels_dir.exists():
        # 找到最新的 panel 目录
        panel_dirs = sorted(
            [d for d in panels_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if panel_dirs:
            latest_panel = panel_dirs[0] / "panel.parquet"
            if latest_panel.exists():
                return latest_panel

    # 兜底：返回默认位置
    return project_root / "etf_rotation_system/data/factor_panel.parquet"


# 预定义配置模板
ETF_STANDARD_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=get_latest_panel_file(),
    )
)

ETF_STRICT_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=get_latest_panel_file(),
    ),
    screening=ScreeningConfig(
        min_ic=0.008,  # 更严格的IC要求
        min_ir=0.08,  # 更严格的IR要求
        max_pvalue=0.1,  # 更严格的显著性
        use_fdr=True,
        fdr_alpha=0.1,  # 更严格的FDR
    ),
)

ETF_RELAXED_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=get_latest_panel_file(),
    ),
    screening=ScreeningConfig(
        min_ic=0.003,  # 更宽松的IC要求
        min_ir=0.03,  # 更宽松的IR要求
        max_pvalue=0.3,  # 更宽松的显著性
        use_fdr=False,  # 可选关闭FDR
    ),
)


def create_default_config_file(output_path: Path) -> None:
    """创建默认配置文件"""
    ETF_STANDARD_CONFIG.to_yaml(output_path)


if __name__ == "__main__":
    # 示例：创建默认配置文件
    config_file = Path("etf_cross_section_config.yaml")
    create_default_config_file(config_file)
    print(f"✅ 默认配置文件已创建: {config_file}")
