#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置类定义 - 类型安全的配置管理"""
import os
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = getLogger(__name__)


def get_project_root() -> Path:
    """获取项目根目录（深度量化0927）"""
    current = Path(__file__).resolve().parent.parent  # 从 config/ 向上
    # 向上找到包含 raw/ 的目录
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
class TradingConfig:
    """交易参数配置"""

    days_per_year: int = 252
    epsilon_small: float = 1e-10
    min_periods: int = 1

    def __post_init__(self):
        """验证配置参数"""
        if self.days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        if self.epsilon_small <= 0:
            raise ValueError("epsilon_small must be positive")
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1")


@dataclass
class FactorWindowsConfig:
    """因子时间窗口配置"""

    # 🔧 修复:扩展窗口参数以包含短中长期窗口,提升因子质量
    # 动量类窗口:扩展为20D(短期),63D(季度),126D(半年),252D(年度)
    momentum: List[int] = field(default_factory=lambda: [20, 63, 126, 252])
    # 波动率:扩展为20D(短期),60D(季度),120D(半年)
    volatility: List[int] = field(default_factory=lambda: [20, 60, 120])
    # 回撤:扩展为63D(季度),126D(半年)
    drawdown: List[int] = field(default_factory=lambda: [63, 126])
    # RSI:保持14D(经典周期)
    rsi: List[int] = field(default_factory=lambda: [14])
    # 价格位置:扩展为20D(月度),60D(季度),120D(半年)
    price_position: List[int] = field(default_factory=lambda: [20, 60, 120])
    # 成交量比率:扩展为5D(周度),20D(月度),60D(季度)
    volume_ratio: List[int] = field(default_factory=lambda: [5, 20, 60])

    # 技术指标特定窗口
    atr_period: int = 14
    vpt_trend_window: int = 20
    vol_volatility_window: int = 20
    amount_surge_short: int = 5
    amount_surge_long: int = 20
    vol_ratio_window: int = 20
    intraday_position_window: int = 5
    price_volume_div_window: int = 5

    # 保留的因子窗口参数
    amplitude_window: int = 20
    up_down_days_window: int = 20
    linear_slope_window: int = 20
    distance_to_high_window: int = 252
    relative_strength_window: int = 20
    turnover_ma_window: int = 60
    drawdown_recovery_window: int = 120

    # 经典技术指标窗口参数
    macd_fast: int = 12  # MACD快线
    macd_slow: int = 26  # MACD慢线
    macd_signal: int = 9  # MACD信号线
    kdj_n: int = 9  # KDJ的N值(RSV周期)
    kdj_m1: int = 3  # KDJ的M1值(K值平滑)
    kdj_m2: int = 3  # KDJ的M2值(D值平滑)
    boll_window: int = 20  # 布林带周期
    boll_std: float = 2.0  # 布林带标准差倍数
    wr_window: int = 14  # 威廉指标周期
    obv_ma_window: int = 20  # OBV移动平均周期

    # 新增流动性和质量因子窗口
    illiquidity_window: int = 20  # Amihud非流动性指标窗口
    amount_change_window: int = 20  # 成交额变化率窗口
    return_quality_window: int = 60  # 收益质量窗口
    sharpe_ratio_window: int = 60  # 夏普比率窗口

    # 新增5个简单ETF因子窗口参数
    trend_consistency_window: int = 20  # 趋势一致性窗口
    extreme_return_window: int = 60  # 极端收益统计窗口
    extreme_return_threshold: float = 2.0  # 极端收益阈值（几倍标准差）
    volume_price_corr_window: int = 20  # 量价相关性窗口
    volatility_short_window: int = 20  # 短期波动率窗口
    volatility_long_window: int = 60  # 长期波动率窗口

    # 乖离率窗口
    bias_windows: List[int] = field(default_factory=lambda: [5, 20, 60])  # 乖离率周期

    def __post_init__(self):
        """验证窗口参数"""
        # 验证所有窗口都是正整数
        all_windows = (
            self.momentum
            + self.volatility
            + self.drawdown
            + self.rsi
            + self.price_position
            + self.volume_ratio
            + self.bias_windows
        )

        for window in all_windows:
            if window <= 0:
                raise ValueError(f"Window size must be positive: {window}")

        # 验证特定窗口
        if self.atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if self.amount_surge_short >= self.amount_surge_long:
            raise ValueError("amount_surge_short must be less than amount_surge_long")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be less than macd_slow")


@dataclass
class ThresholdsConfig:
    """阈值参数配置"""

    large_order_volume_ratio: float = 1.2
    doji_body_threshold: Optional[float] = None
    hammer_lower_shadow_ratio: float = 2.0
    hammer_upper_shadow_ratio: float = 1.0

    def __post_init__(self):
        """验证阈值参数"""
        if self.large_order_volume_ratio <= 0:
            raise ValueError("large_order_volume_ratio must be positive")
        if self.hammer_lower_shadow_ratio <= 0:
            raise ValueError("hammer_lower_shadow_ratio must be positive")
        if self.hammer_upper_shadow_ratio < 0:
            raise ValueError("hammer_upper_shadow_ratio must be non-negative")


@dataclass
class PathsConfig:
    """路径配置"""

    data_dir: str = "raw/ETF/daily"
    output_dir: str = "etf_rotation_system/data/results/panels"
    config_file: str = "config/etf_config.yaml"

    def __post_init__(self):
        """验证并解析路径为绝对路径"""
        project_root = get_project_root()

        # 将相对路径解析为绝对路径
        data_path = Path(self.data_dir)
        if not data_path.is_absolute():
            self.data_dir = str((project_root / data_path).resolve())

        output_path = Path(self.output_dir)
        if not output_path.is_absolute():
            self.output_dir = str((project_root / output_path).resolve())


@dataclass
class ProcessingConfig:
    """并行处理配置"""

    max_workers: int = 4
    continue_on_symbol_error: bool = True
    max_failure_rate: float = 0.1

    def __post_init__(self):
        """验证处理配置"""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0 <= self.max_failure_rate <= 1:
            raise ValueError("max_failure_rate must be between 0 and 1")


@dataclass
class FactorEnableConfig:
    """因子开关配置"""

    # 原有18个因子
    momentum: bool = True
    volatility: bool = True
    drawdown: bool = True
    momentum_acceleration: bool = True
    rsi: bool = True
    price_position: bool = True
    volume_ratio: bool = True

    # 新增技术因子
    overnight_return: bool = True
    atr: bool = True
    doji_pattern: bool = True
    intraday_range: bool = True
    bullish_engulfing: bool = True
    hammer_pattern: bool = True
    price_impact: bool = True
    volume_price_trend: bool = True
    vol_ma_ratio_5: bool = True
    vol_volatility_20: bool = True
    true_range: bool = True
    buy_pressure: bool = True

    # 资金流因子
    vwap_deviation: bool = True
    amount_surge_5d: bool = True
    price_volume_div: bool = True
    intraday_position: bool = True
    large_order_signal: bool = True

    # 流动性因子
    illiquidity: bool = True
    turnover_ratio: bool = True
    amount_change_rate: bool = True

    # 微观结构因子
    amplitude: bool = True
    shadow_ratio: bool = True
    up_down_days_ratio: bool = True

    # 趋势强度因子
    linear_slope: bool = True
    distance_to_high: bool = True

    # 相对强弱因子
    relative_strength_vs_index: bool = True
    relative_amplitude: bool = True

    # 质量因子
    return_quality: bool = True
    sharpe_ratio: bool = True
    drawdown_recovery_speed: bool = True

    # 经典技术指标（学术验证）
    macd: bool = True  # MACD指标
    kdj: bool = True  # KDJ随机指标
    bollinger_bands: bool = True  # 布林带
    bias: bool = True  # 乖离率
    williams_r: bool = True  # 威廉指标
    obv: bool = True  # 能量潮

    # 新增5个简单ETF因子
    trend_consistency: bool = True  # 趋势一致性
    extreme_return_freq: bool = True  # 极端收益频率
    consecutive_up_days: bool = True  # 连续上涨天数
    volume_price_divergence: bool = True  # 量价背离强度
    volatility_regime_shift: bool = True  # 波动率突变


@dataclass
class DataProcessingConfig:
    """数据处理配置"""

    required_columns: List[str] = field(
        default_factory=lambda: [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
        ]
    )
    optional_columns: List[str] = field(default_factory=lambda: ["amount"])
    volume_column_alias: str = "vol"
    fallback_estimation: bool = True

    def __post_init__(self):
        """验证数据处理配置"""
        required_set = set(self.required_columns)
        optional_set = set(self.optional_columns)

        if required_set & optional_set:
            raise ValueError("Columns cannot be both required and optional")


@dataclass
class OutputConfig:
    """输出配置"""

    save_execution_log: bool = True
    save_metadata: bool = True
    timestamp_subdirectory: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    def __post_init__(self):
        """验证日志配置"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")


@dataclass
class DisplayConfig:
    """显示配置"""

    log_separator_length: int = 80
    progress_bar_desc: str = "计算因子"
    factor_list_start: int = 1
    json_indent: int = 2
    coverage_format: str = ".2%"

    def __post_init__(self):
        """验证显示配置"""
        if self.log_separator_length <= 0:
            raise ValueError("log_separator_length must be positive")
        if self.json_indent < 0:
            raise ValueError("json_indent must be non-negative")
        if self.factor_list_start < 1:
            raise ValueError("factor_list_start must be >= 1")


@dataclass
class ConstantsConfig:
    """计算常量配置"""

    rsi_multiplier: int = 100
    concat_axis: int = 1
    astype_float: str = "float"
    shift_days: int = 1
    concat_max_axis: int = 1
    sign_multiplier: int = 1

    def __post_init__(self):
        """验证常量配置"""
        if self.rsi_multiplier <= 0:
            raise ValueError("rsi_multiplier must be positive")
        if self.concat_axis not in [0, 1]:
            raise ValueError("concat_axis must be 0 or 1")
        if self.shift_days <= 0:
            raise ValueError("shift_days must be positive")
        if self.concat_max_axis not in [0, 1]:
            raise ValueError("concat_max_axis must be 0 or 1")
        if self.astype_float not in ["float", "float32", "float64"]:
            raise ValueError("astype_float must be a valid float type")


@dataclass
class FactorPanelConfig:
    """因子面板生成完整配置"""

    trading: TradingConfig = field(default_factory=TradingConfig)
    factor_windows: FactorWindowsConfig = field(default_factory=FactorWindowsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    factor_enable: FactorEnableConfig = field(default_factory=FactorEnableConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    constants: ConstantsConfig = field(default_factory=ConstantsConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FactorPanelConfig":
        """从YAML文件加载配置"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            return cls.from_dict(config_dict)

        except FileNotFoundError:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return cls()

        except Exception as e:
            logger.error(f"配置加载失败: {e}，使用默认配置")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FactorPanelConfig":
        """从字典创建配置"""
        # 递归创建嵌套配置对象
        trading_config = TradingConfig(**config_dict.get("trading", {}))
        factor_windows_config = FactorWindowsConfig(
            **config_dict.get("factor_windows", {})
        )
        thresholds_config = ThresholdsConfig(**config_dict.get("thresholds", {}))
        paths_config = PathsConfig(**config_dict.get("paths", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        factor_enable_config = FactorEnableConfig(
            **config_dict.get("factor_enable", {})
        )
        data_processing_config = DataProcessingConfig(
            **config_dict.get("data_processing", {})
        )
        output_config = OutputConfig(**config_dict.get("output", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        display_config = DisplayConfig(**config_dict.get("display", {}))
        constants_config = ConstantsConfig(**config_dict.get("constants", {}))

        return cls(
            trading=trading_config,
            factor_windows=factor_windows_config,
            thresholds=thresholds_config,
            paths=paths_config,
            processing=processing_config,
            factor_enable=factor_enable_config,
            data_processing=data_processing_config,
            output=output_config,
            logging=logging_config,
            display=display_config,
            constants=constants_config,
        )

    def to_legacy_format(self) -> Dict:
        """转换为原有格式，保持向后兼容"""
        return {
            "factor_generation": {
                "momentum_periods": self.factor_windows.momentum,
                "volatility_windows": self.factor_windows.volatility,
                "rsi_windows": self.factor_windows.rsi,
                "price_position_windows": self.factor_windows.price_position,
                "volume_ratio_windows": self.factor_windows.volume_ratio,
            }
        }

    def validate(self) -> bool:
        """验证配置完整性"""
        try:
            # 各个配置类的__post_init__方法已经进行了验证
            # 这里可以添加额外的跨配置验证

            # 检查因子数量合理性
            enabled_factors = sum(
                [
                    self.factor_enable.momentum,
                    self.factor_enable.volatility,
                    self.factor_enable.drawdown,
                    self.factor_enable.momentum_acceleration,
                    self.factor_enable.rsi,
                    self.factor_enable.price_position,
                    self.factor_enable.volume_ratio,
                    self.factor_enable.overnight_return,
                    self.factor_enable.atr,
                    self.factor_enable.doji_pattern,
                    self.factor_enable.intraday_range,
                    self.factor_enable.bullish_engulfing,
                    self.factor_enable.hammer_pattern,
                    self.factor_enable.price_impact,
                    self.factor_enable.volume_price_trend,
                    self.factor_enable.vol_ma_ratio_5,
                    self.factor_enable.vol_volatility_20,
                    self.factor_enable.true_range,
                    self.factor_enable.buy_pressure,
                    self.factor_enable.vwap_deviation,
                    self.factor_enable.amount_surge_5d,
                    self.factor_enable.price_volume_div,
                    self.factor_enable.intraday_position,
                    self.factor_enable.large_order_signal,
                ]
            )

            if enabled_factors == 0:
                raise ValueError("至少需要启用一个因子")

            logger.info(f"配置验证通过，启用了 {enabled_factors} 个因子")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
