"""Configuration objects for the HK mid-frequency strategy stack."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, Tuple

DEFAULT_TAKE_PROFITS = (0.006, 0.01, 0.014, 0.018)


def _find_project_root() -> Path:
    """自动发现项目根目录（包含 factor_system 和 raw 目录）"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "factor_system").exists() and (current / "raw").exists():
            return current
        current = current.parent
    # 如果找不到，使用环境变量或默认路径
    if "PROJECT_ROOT" in os.environ:
        return Path(os.environ["PROJECT_ROOT"])
    # 默认回退到当前文件的祖父目录
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    """统一路径配置 - P0 级优化：消除硬编码路径"""

    project_root: Path = field(default_factory=_find_project_root)

    @property
    def raw_data_dir(self) -> Path:
        """原始数据层路径"""
        return self.project_root / "raw"

    @property
    def hk_raw_dir(self) -> Path:
        """港股原始数据路径"""
        return self.raw_data_dir / "HK"

    @property
    def factor_system_dir(self) -> Path:
        """因子系统根目录"""
        return self.project_root / "factor_system"

    @property
    def factor_output_dir(self) -> Path:
        """因子输出层路径 (Factor Output Layer)"""
        return self.factor_system_dir / "factor_output"

    @property
    def factor_screening_dir(self) -> Path:
        """因子筛选层路径 (Factor Screening Results)"""
        return self.factor_system_dir / "factor_screening" / "screening_results"

    @property
    def factor_ready_dir(self) -> Path:
        """优秀因子存储路径"""
        return self.factor_system_dir / "factor_ready"
    
    @property
    def factor_registry_path(self) -> Path:
        """因子注册表路径"""
        return self.factor_system_dir / "research" / "metadata" / "factor_registry.json"

    @property
    def backtest_output_dir(self) -> Path:
        """回测输出目录 - 带时间戳的会话管理"""
        return self.project_root / "hk_midfreq" / "backtest_results"

    def validate_paths(self) -> bool:
        """验证关键路径是否存在"""
        required_paths = [
            self.project_root,
            self.raw_data_dir,
            self.factor_system_dir,
        ]
        return all(p.exists() for p in required_paths)

    def __repr__(self) -> str:
        return (
            f"PathConfig(project_root={self.project_root}, "
            f"hk_raw={self.hk_raw_dir.exists()}, "
            f"factor_output={self.factor_output_dir.exists()})"
        )


@dataclass(frozen=True)
class FusionConfig:
    """Parameters governing how multi-timeframe factor scores are fused."""

    factor_weighting: str = "ic_weighted"
    timeframe_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "daily": 0.5,
            "60min": 0.3,
            "15min": 0.2,
        }
    )
    trend_timeframe: str = "daily"
    confirmation_timeframe: str = "60min"
    intraday_timeframes: Tuple[str, ...] = ("15min", "5min")
    trend_threshold: float = 0.0
    confirmation_threshold: float = 0.0

    def ordered_timeframes(self) -> Tuple[str, ...]:
        """Return the unique timeframes required by the fusion rules."""

        ordered: list[str] = []
        for tf in (
            self.trend_timeframe,
            self.confirmation_timeframe,
            *self.intraday_timeframes,
        ):
            if tf and tf not in ordered:
                ordered.append(tf)
        # Include explicitly weighted timeframes for the loader to fetch
        for tf in self.timeframe_weights.keys():
            if tf and tf not in ordered:
                ordered.append(tf)
        return tuple(ordered)


@dataclass(frozen=True)
class TradingConfig:
    """Portfolio sizing and holding horizon parameters."""

    capital: float = 1_000_000.0
    position_size: float = 100_000.0
    max_positions: int = 8
    hold_days: int = 4

    def allocation_per_position(self) -> float:
        """Return the maximum cash allocated to a single position."""

        return min(self.position_size, self.capital / max(self.max_positions, 1))


@dataclass(frozen=True)
class ExecutionConfig:
    """Trading frictions and stop management settings."""

    transaction_cost: float = 0.0038
    slippage: float = 0.001
    stop_loss: float = 0.018
    take_profit_levels: Sequence[float] = field(
        default_factory=lambda: DEFAULT_TAKE_PROFITS
    )

    def primary_take_profit(self) -> float:
        """Pick the first take-profit level for engines that only support one value."""

        return self.take_profit_levels[0] if self.take_profit_levels else 0.0


@dataclass(frozen=True)
class StrategyRuntimeConfig:
    """High-level toggles for the HK mid-frequency workflow."""

    strategy_name: str = "HK_MIDFREQ_REVERSAL"
    paths: PathConfig = field(default_factory=PathConfig)
    default_timeframe: str = "60min"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    trend_ma_window: int = 20
    confirmation_ma_window: int = 10

    @property
    def base_output_dir(self) -> Path:
        """向后兼容：保持原有 API"""
        return self.paths.factor_screening_dir


# 全局默认配置实例
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_EXECUTION_CONFIG = ExecutionConfig()
DEFAULT_RUNTIME_CONFIG = StrategyRuntimeConfig()
