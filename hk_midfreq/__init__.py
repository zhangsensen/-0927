"""极简HK中频交易策略包 - Linus风格

P0 优化实现：
- 统一配置管理 (PathConfig)
- 路径解耦（移除硬编码）
- 错误处理标准化
"""

from hk_midfreq.backtest_engine import run_single_asset_backtest
from hk_midfreq.config import (
    ExecutionConfig,
    PathConfig,
    StrategyRuntimeConfig,
    TradingConfig,
)
from hk_midfreq.factor_interface import FactorLoadError, FactorScoreLoader
from hk_midfreq.price_loader import DataLoadError, PriceDataLoader
from hk_midfreq.result_manager import BacktestResultManager
from hk_midfreq.strategy_core import StrategyCore, StrategySignals

__all__ = [
    "run_single_asset_backtest",
    "TradingConfig",
    "ExecutionConfig",
    "PathConfig",
    "StrategyRuntimeConfig",
    "StrategyCore",
    "StrategySignals",
    "PriceDataLoader",
    "FactorScoreLoader",
    "BacktestResultManager",
    "DataLoadError",
    "FactorLoadError",
]
