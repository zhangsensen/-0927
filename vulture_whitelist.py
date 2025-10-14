# Vulture whitelist for 深度量化0927 project
# This file contains false positives that Vulture identifies as unused code
# but are actually required for the quantitative trading system

# FactorEngine API exports - used by external modules
__all__ = [
    "api",
    "calculate_factors",
    "list_available_factors",
    "get_cache_stats",
    "prewarm_cache",
]

# Dynamic imports used in factor generation
from factor_system.factor_engine.factors import *  # noqa: F403,F401

# Test fixtures and helpers (used by pytest)
test_helpers = [
    "create_test_data",
    "mock_factor_provider",
    "assert_factor_consistency",
]

# CLI entry points (used by console scripts)
cli_functions = [
    "main",
    "cli_entry_point",
    "run_analysis",
]

# Configuration constants (used dynamically by YAML loader)
config_keys = [
    "LOW_COMPLEXITY_THRESHOLD",
    "MEDIUM_COMPLEXITY_THRESHOLD",
    "MAX_COMPLEXITY",
    "MIN_SIMILARITY_THRESHOLD",
]

# Factor registry entries (registered dynamically)
registered_factors = [
    "RSI",
    "MACD",
    "STOCH",
    "BOLLINGER_BANDS",
    "ATR",
    "OBV",
    "MONEY_FLOW_INDEX",
    "WILLIAMS_R",
]

# Money flow factors (A-share specific)
money_flow_factors = [
    "MainNetInflow_Rate",
    "LargeOrder_Ratio",
    "Flow_Price_Divergence",
    "Institutional_Absorption",
    "Flow_Tier_Strength",
    "Gap_Up_Signal",
    "Tradability_Mask",
]

# Abstract base classes (implemented by subclasses)
abstract_methods = [
    "AbstractFactorProvider",
    "AbstractDataLoader",
    "AbstractFactorCalculator",
]

# Plugin interfaces (implemented by external modules)
plugin_interfaces = [
    "DataPlugin",
    "FactorPlugin",
    "ScreeningPlugin",
]

# Performance optimization utilities (used by vectorized operations)
vectorized_utils = [
    "batch_calculate",
    "parallel_process",
    "cache_result",
    "validate_inputs",
]

# Logging and monitoring (used by decorators)
logging_utils = [
    "log_performance",
    "track_execution_time",
    "monitor_memory_usage",
]

# Error handling utilities (used by exception decorators)
error_utils = [
    "safe_operation",
    "handle_factor_error",
    "validate_configuration",
]

# Data loader utilities (used dynamically by data loading system)
data_loader_utils = [
    "construct_price_file_path",
]
