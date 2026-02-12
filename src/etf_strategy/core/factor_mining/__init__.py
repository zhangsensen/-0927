"""
因子挖掘体系 | Factor Mining System
================================================================================
6 层架构:
  Layer 1: FactorZoo (注册中心)          — registry.py
  Layer 2: FactorQualityAnalyzer (质检)  — quality.py
  Layer 3: FactorDiscoveryPipeline (挖掘) — discovery.py
  Layer 4: FactorPrefilter (可组合性预筛) — prefilter.py
  Layer 5: FactorSelector (去冗余)       — selection.py
  Layer 6: 全流程脚本                     — scripts/run_factor_mining.py
"""

from .discovery import (
    AlgebraicSearch,
    FactorDiscoveryPipeline,
    TransformSearch,
    WindowOptimizer,
)
from .prefilter import FactorPrefilter, PrefilterConfig, PrefilterResult
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
    "FactorPrefilter",
    "PrefilterConfig",
    "PrefilterResult",
    "FactorSelector",
    "spearman_ic_series",
    "compute_forward_returns",
]
