"""ETF轮动系统 - 统一配置管理器

职责:
1. 加载所有YAML配置
2. 提供类型安全的配置访问
3. 支持配置覆盖和验证

遵循原则:
- 单一真理源: 所有配置来自 etf_rotation_system/config/
- 显式优于隐式: 参数必须明确声明
- Fail Fast: 配置错误立即报错
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""

    # 成本模型 (港股)
    commission_rate: float = 0.002  # 佣金 0.2%
    stamp_duty: float = 0.001  # 印花税 0.1%
    slippage_hkd: float = 0.05  # 滑点 0.05 HKD

    # 组合参数
    n_jobs: int = -1  # 并行核心数
    init_cash: float = 1_000_000  # 初始资金

    # 轮动参数
    top_n: int = 5  # 选股数量
    rebalance_freq: str = "1W"  # 调仓频率

    # 权重方案
    weight_schemes: List[str] = field(
        default_factory=lambda: ["equal", "rank", "score"]
    )

    # 惩罚系数
    turnover_penalties: List[float] = field(
        default_factory=lambda: [0.0, 0.001, 0.002, 0.003, 0.005]
    )


@dataclass
class ScreeningConfig:
    """因子筛选配置"""

    # IC 计算
    forward_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    min_ic_threshold: float = 0.05
    min_ir_threshold: float = 0.5

    # FDR 校验
    use_fdr_correction: bool = True
    fdr_alpha: float = 0.05

    # Newey-West 标准误
    use_newey_west: bool = True
    nw_lags: int = 5

    # 平行计算
    n_jobs: int = -1


@dataclass
class PathsConfig:
    """路径配置（兼容旧代码）"""

    data_dir: str = "raw/ETF/daily"
    output_dir: str = "etf_rotation_system/data/results/panels"


@dataclass
class ProcessingConfig:
    """处理配置（兼容旧代码）"""

    max_workers: int = 4
    continue_on_symbol_error: bool = True
    max_failure_rate: float = 0.1


@dataclass
class DataProcessingConfig:
    """数据处理配置（兼容旧代码）"""

    volume_column_alias: str = "vol"
    required_columns: List[str] = field(
        default_factory=lambda: ["date", "open", "high", "low", "close", "volume"]
    )
    optional_columns: List[str] = field(default_factory=lambda: ["amount", "turnover"])
    fallback_estimation: bool = True


@dataclass
class FactorPanelConfig:
    """因子面板配置"""

    # 嵌套配置（兼容旧代码）
    paths: PathsConfig = field(default_factory=PathsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)

    # 因子组
    factor_groups: List[str] = field(
        default_factory=lambda: ["technical", "volume", "momentum", "volatility"]
    )

    # 时间参数
    lookback_days: int = 252
    min_obs: int = 60


class ConfigManager:
    """统一配置管理器

    用法:
        cfg = ConfigManager()
        backtest_cfg = cfg.get_backtest_config()
        screening_cfg = cfg.get_screening_config()
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """初始化配置管理器

        Args:
            config_dir: 配置目录路径，默认为 etf_rotation_system/config/
        """
        if config_dir is None:
            # 自动检测项目根目录
            current = Path(__file__).resolve()
            project_root = current.parent.parent.parent
            config_dir = project_root / "etf_rotation_system" / "config"

        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.config_dir}")

        logger.info(f"✅ ConfigManager 初始化: {self.config_dir}")

        # 加载所有配置
        self._backtest_config: Optional[BacktestConfig] = None
        self._screening_config: Optional[ScreeningConfig] = None
        self._factor_panel_config: Optional[FactorPanelConfig] = None

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.debug(f"加载配置: {filename}")
        return data or {}

    def get_backtest_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> BacktestConfig:
        """获取回测配置

        Args:
            overrides: 覆盖参数（用于实验）

        Returns:
            BacktestConfig 实例
        """
        if self._backtest_config is None:
            data = self._load_yaml("backtest_config.yaml")
            cfg_data = data.get("backtest", {})

            # 解析配置
            self._backtest_config = BacktestConfig(
                commission_rate=cfg_data.get("commission_rate", 0.002),
                stamp_duty=cfg_data.get("stamp_duty", 0.001),
                slippage_hkd=cfg_data.get("slippage_hkd", 0.05),
                n_jobs=cfg_data.get("n_jobs", -1),
                init_cash=cfg_data.get("init_cash", 1_000_000),
                top_n=cfg_data.get("top_n", 5),
                rebalance_freq=cfg_data.get("rebalance_freq", "1W"),
                weight_schemes=cfg_data.get("weight_schemes", ["equal"]),
                turnover_penalties=cfg_data.get("turnover_penalties", [0.0]),
            )

        # 应用覆盖
        if overrides:
            cfg_dict = self._backtest_config.__dict__.copy()
            cfg_dict.update(overrides)
            return BacktestConfig(**cfg_dict)

        return self._backtest_config

    def get_screening_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> ScreeningConfig:
        """获取因子筛选配置

        Args:
            overrides: 覆盖参数

        Returns:
            ScreeningConfig 实例
        """
        if self._screening_config is None:
            data = self._load_yaml("screening_config.yaml")

            # 处理旧格式配置
            if "screening" in data:
                cfg_data = data["screening"]
            else:
                cfg_data = data

            self._screening_config = ScreeningConfig(
                forward_periods=cfg_data.get("forward_periods", [5, 10, 20]),
                min_ic_threshold=cfg_data.get("min_ic_threshold", 0.05),
                min_ir_threshold=cfg_data.get("min_ir_threshold", 0.5),
                use_fdr_correction=cfg_data.get("use_fdr_correction", True),
                fdr_alpha=cfg_data.get("fdr_alpha", 0.05),
                use_newey_west=cfg_data.get("use_newey_west", True),
                nw_lags=cfg_data.get("nw_lags", 5),
                n_jobs=cfg_data.get("n_jobs", -1),
            )

        if overrides:
            cfg_dict = self._screening_config.__dict__.copy()
            cfg_dict.update(overrides)
            return ScreeningConfig(**cfg_dict)

        return self._screening_config

    def get_factor_panel_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> FactorPanelConfig:
        """获取因子面板配置

        Args:
            overrides: 覆盖参数

        Returns:
            FactorPanelConfig 实例
        """
        if self._factor_panel_config is None:
            data = self._load_yaml("factor_panel_config.yaml")

            # 解析嵌套配置
            paths_data = data.get("paths", {})
            processing_data = data.get("processing", {})
            data_processing_data = data.get("data_processing", {})

            self._factor_panel_config = FactorPanelConfig(
                paths=PathsConfig(
                    data_dir=paths_data.get("data_dir", "raw/ETF/daily"),
                    output_dir=paths_data.get(
                        "output_dir", "etf_rotation_system/data/results/panels"
                    ),
                ),
                processing=ProcessingConfig(
                    max_workers=processing_data.get("max_workers", 4),
                    continue_on_symbol_error=processing_data.get(
                        "continue_on_symbol_error", True
                    ),
                    max_failure_rate=processing_data.get("max_failure_rate", 0.1),
                ),
                data_processing=DataProcessingConfig(
                    volume_column_alias=data_processing_data.get(
                        "volume_column_alias", "vol"
                    ),
                    required_columns=data_processing_data.get(
                        "required_columns",
                        ["date", "open", "high", "low", "close", "volume"],
                    ),
                    optional_columns=data_processing_data.get(
                        "optional_columns", ["amount", "turnover"]
                    ),
                    fallback_estimation=data_processing_data.get(
                        "fallback_estimation", True
                    ),
                ),
                factor_groups=data.get(
                    "factor_groups", ["technical", "volume", "momentum"]
                ),
                lookback_days=data.get("lookback_days", 252),
                min_obs=data.get("min_obs", 60),
            )

        if overrides:
            cfg_dict = self._factor_panel_config.__dict__.copy()
            cfg_dict.update(overrides)
            return FactorPanelConfig(**cfg_dict)

        return self._factor_panel_config

    def reload(self):
        """重新加载所有配置（用于热更新）"""
        self._backtest_config = None
        self._screening_config = None
        self._factor_panel_config = None
        logger.info("🔄 配置重新加载")

    def validate(self) -> bool:
        """验证所有配置的完整性

        Returns:
            True if all configs are valid
        """
        try:
            self.get_backtest_config()
            self.get_screening_config()
            self.get_factor_panel_config()
            logger.info("✅ 配置验证通过")
            return True
        except Exception as e:
            logger.error(f"❌ 配置验证失败: {e}")
            return False


# 全局单例（可选）
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器单例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager
