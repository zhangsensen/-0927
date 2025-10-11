# -*- coding: utf-8 -*-
"""
工具函数模块

 Linus式重构：统一路径管理与异常处理
"""

from .dependency_checker import DependencyChecker
from .error_utils import (
    CalculationError,
    ConfigurationError,
    DataValidationError,
    FactorSystemError,
    PathError,
    safe_compute_operation,
    safe_io_operation,
    safe_operation,
    validate_in_range,
    validate_not_none,
    validate_positive,
)
from .path_utils import PathStandardizer
from .project_paths import (
    ProjectPaths,
    get_cache_dir,
    get_config_dir,
    get_factor_output_dir,
    get_logs_dir,
    get_project_root,
    get_raw_data_dir,
    get_screening_results_dir,
    validate_project_structure,
)

__all__ = [
    # 依赖检查
    "DependencyChecker",
    # 路径工具
    "PathStandardizer",
    "ProjectPaths",
    "get_project_root",
    "get_raw_data_dir",
    "get_factor_output_dir",
    "get_screening_results_dir",
    "get_logs_dir",
    "get_cache_dir",
    "get_config_dir",
    "validate_project_structure",
    # 异常处理
    "FactorSystemError",
    "ConfigurationError",
    "DataValidationError",
    "CalculationError",
    "PathError",
    "safe_operation",
    "safe_io_operation",
    "safe_compute_operation",
    "validate_not_none",
    "validate_positive",
    "validate_in_range",
]
