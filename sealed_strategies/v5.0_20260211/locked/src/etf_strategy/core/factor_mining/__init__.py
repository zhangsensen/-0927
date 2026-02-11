"""
因子挖掘体系 | Factor Mining System
================================================================================
5 层架构:
  Layer 1: FactorZoo (注册中心)        — registry.py
  Layer 2: FactorQualityAnalyzer (质检) — quality.py
  Layer 3: FactorDiscoveryPipeline (挖掘) — discovery.py
  Layer 4: FactorSelector (筛选)        — selection.py
  Layer 5: 全流程脚本                    — scripts/run_factor_mining.py
"""

from .discovery import (
    AlgebraicSearch,
    FactorDiscoveryPipeline,
    TransformSearch,
    WindowOptimizer,
)
from .quality import (
    FactorQualityAnalyzer,
    FactorQualityReport,
    compute_forward_returns,
    spearman_ic_series,
)
from .registry import FactorEntry, FactorZoo
from .selection import FactorSelector

__all__ = [
    "FactorEntry",
    "FactorZoo",
    "FactorQualityAnalyzer",
    "FactorQualityReport",
    "FactorDiscoveryPipeline",
    "AlgebraicSearch",
    "WindowOptimizer",
    "TransformSearch",
    "FactorSelector",
    "spearman_ic_series",
    "compute_forward_returns",
]
