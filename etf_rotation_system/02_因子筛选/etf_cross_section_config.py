#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF横截面因子筛选配置系统 - Linus工程风格"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class DataSourceConfig:
    """数据源配置"""
    price_dir: Path
    panel_file: Path
    price_columns: List[str] = field(default_factory=lambda: ['trade_date', 'close'])
    file_pattern: str = "*.parquet"
    symbol_extract_method: str = "stem_split"  # stem_split, regex, custom

    def __post_init__(self):
        # 确保路径存在
        if isinstance(self.price_dir, str):
            self.price_dir = Path(self.price_dir)
        if isinstance(self.panel_file, str):
            self.panel_file = Path(self.panel_file)


@dataclass
class AnalysisConfig:
    """分析参数配置"""
    # IC分析周期
    ic_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # 样本量阈值
    min_observations: int = 30          # ETF小样本标准
    min_ranking_samples: int = 5        # 横截面排名最小样本
    min_ic_observations: int = 20       # IC计算最小观测值

    # 相关性分析
    correlation_method: str = "spearman"
    correlation_min_periods: int = 30

    # 数值计算
    epsilon_small: float = 1e-8         # IR计算小值
    stability_split_ratio: float = 0.5   # 稳定性分析分割比例


@dataclass
class ScreeningConfig:
    """筛选标准配置"""
    # 基础筛选标准
    min_ic: float = 0.005              # 最小IC (0.5%)
    min_ir: float = 0.05               # 最小IR (实用标准)
    max_pvalue: float = 0.2            # 最大p值
    min_coverage: float = 0.7          # 最小覆盖率

    # 去重标准
    max_correlation: float = 0.7       # 最大因子间相关性

    # FDR校正
    use_fdr: bool = True               # 是否启用FDR
    fdr_alpha: float = 0.2             # FDR显著性水平

    # 分层评级阈值
    tier_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "core": {"ic": 0.02, "ir": 0.1},
        "supplement": {"ic": 0.01, "ir": 0.07},
        "research": {"ic": 0.0, "ir": 0.0}
    })

    # 分层评级标签
    tier_labels: Dict[str, str] = field(default_factory=lambda: {
        "core": "🟢 核心",
        "supplement": "🟡 补充",
        "research": "🔵 研究"
    })


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: Path
    use_timestamp_subdir: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    subdir_prefix: str = "screening_"

    # 输出文件名
    files: Dict[str, str] = field(default_factory=lambda: {
        "ic_analysis": "ic_analysis.csv",
        "passed_factors": "passed_factors.csv",
        "screening_report": "screening_report.txt"
    })

    # 报告格式
    include_factor_details: bool = True
    include_summary_statistics: bool = True
    encoding: str = "utf-8"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ETFCrossSectionConfig:
    """ETF横截面因子筛选完整配置"""
    data_source: DataSourceConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        output_dir=Path("etf_rotation_system/data/results/screening")
    ))

    # 系统级控制
    debug_mode: bool = False
    progress_reporting: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ETFCrossSectionConfig':
        """从字典创建配置对象"""
        return cls(
            data_source=DataSourceConfig(**config_dict['data_source']),
            analysis=AnalysisConfig(**config_dict.get('analysis', {})),
            screening=ScreeningConfig(**config_dict.get('screening', {})),
            output=OutputConfig(**config_dict.get('output', {})),
            debug_mode=config_dict.get('debug_mode', False),
            progress_reporting=config_dict.get('progress_reporting', True)
        )

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> 'ETFCrossSectionConfig':
        """从YAML文件加载配置"""
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_file: Path) -> None:
        """保存配置到YAML文件"""
        config_dict = {
            'data_source': {
                'price_dir': str(self.data_source.price_dir),
                'panel_file': str(self.data_source.panel_file),
                'price_columns': self.data_source.price_columns,
                'file_pattern': self.data_source.file_pattern,
                'symbol_extract_method': self.data_source.symbol_extract_method
            },
            'analysis': {
                'ic_periods': self.analysis.ic_periods,
                'min_observations': self.analysis.min_observations,
                'min_ranking_samples': self.analysis.min_ranking_samples,
                'min_ic_observations': self.analysis.min_ic_observations,
                'correlation_method': self.analysis.correlation_method,
                'correlation_min_periods': self.analysis.correlation_min_periods,
                'epsilon_small': self.analysis.epsilon_small,
                'stability_split_ratio': self.analysis.stability_split_ratio
            },
            'screening': {
                'min_ic': self.screening.min_ic,
                'min_ir': self.screening.min_ir,
                'max_pvalue': self.screening.max_pvalue,
                'min_coverage': self.screening.min_coverage,
                'max_correlation': self.screening.max_correlation,
                'use_fdr': self.screening.use_fdr,
                'fdr_alpha': self.screening.fdr_alpha,
                'tier_thresholds': self.screening.tier_thresholds,
                'tier_labels': self.screening.tier_labels
            },
            'output': {
                'output_dir': str(self.output.output_dir),
                'use_timestamp_subdir': self.output.use_timestamp_subdir,
                'timestamp_format': self.output.timestamp_format,
                'subdir_prefix': self.output.subdir_prefix,
                'files': self.output.files,
                'include_factor_details': self.output.include_factor_details,
                'include_summary_statistics': self.output.include_summary_statistics,
                'encoding': self.output.encoding
            },
            'debug_mode': self.debug_mode,
            'progress_reporting': self.progress_reporting
        }

        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)


# 预定义配置模板
ETF_STANDARD_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=Path("etf_rotation_system/data/factor_panel.parquet")
    )
)

ETF_STRICT_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=Path("etf_rotation_system/data/factor_panel.parquet")
    ),
    screening=ScreeningConfig(
        min_ic=0.008,      # 更严格的IC要求
        min_ir=0.08,       # 更严格的IR要求
        max_pvalue=0.1,    # 更严格的显著性
        use_fdr=True,
        fdr_alpha=0.1      # 更严格的FDR
    )
)

ETF_RELAXED_CONFIG = ETFCrossSectionConfig(
    data_source=DataSourceConfig(
        price_dir=Path("raw/ETF/daily"),
        panel_file=Path("etf_rotation_system/data/factor_panel.parquet")
    ),
    screening=ScreeningConfig(
        min_ic=0.003,      # 更宽松的IC要求
        min_ir=0.03,       # 更宽松的IR要求
        max_pvalue=0.3,    # 更宽松的显著性
        use_fdr=False      # 可选关闭FDR
    )
)


def create_default_config_file(output_path: Path) -> None:
    """创建默认配置文件"""
    ETF_STANDARD_CONFIG.to_yaml(output_path)


if __name__ == "__main__":
    # 示例：创建默认配置文件
    config_file = Path("etf_cross_section_config.yaml")
    create_default_config_file(config_file)
    print(f"✅ 默认配置文件已创建: {config_file}")