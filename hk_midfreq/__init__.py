"""Public exports for the HK mid-frequency strategy package."""

from hk_midfreq import backtest_engine, factor_interface, review_tools, strategy_core
from hk_midfreq.backtest_engine import (
    BacktestArtifacts,
    run_portfolio_backtest,
    run_single_asset_backtest,
)
from hk_midfreq.config import (
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_TRADING_CONFIG,
    ExecutionConfig,
    StrategyRuntimeConfig,
    TradingConfig,
)
from hk_midfreq.factor_interface import (
    FactorScoreLoader,
    SymbolScore,
    load_factor_scores,
)
from hk_midfreq.strategy_core import StrategyCore, StrategySignals, hk_reversal_logic

__all__ = [
    "backtest_engine",
    "factor_interface",
    "review_tools",
    "strategy_core",
    "BacktestArtifacts",
    "run_portfolio_backtest",
    "run_single_asset_backtest",
    "DEFAULT_EXECUTION_CONFIG",
    "DEFAULT_RUNTIME_CONFIG",
    "DEFAULT_TRADING_CONFIG",
    "ExecutionConfig",
    "StrategyRuntimeConfig",
    "TradingConfig",
    "FactorScoreLoader",
    "SymbolScore",
    "load_factor_scores",
    "StrategyCore",
    "StrategySignals",
    "hk_reversal_logic",
]
