"""
ML Ranker: Learning-to-Rank模型用于WFO策略排序校准

核心功能:
- 从WFO指标学习到真实回测表现的映射
- 使用LightGBM LambdaRank优化排序质量
- 提供训练、推理和评估完整流程
"""

__version__ = "0.1.0"
__author__ = "ETF Rotation Team"

from .data_loader import load_wfo_features, load_real_backtest_results, build_training_dataset
from .feature_engineer import build_feature_matrix
from .ltr_model import LTRRanker
from .evaluator import compute_ranking_metrics, generate_evaluation_report

__all__ = [
    "load_wfo_features",
    "load_real_backtest_results", 
    "build_training_dataset",
    "build_feature_matrix",
    "LTRRanker",
    "compute_ranking_metrics",
    "generate_evaluation_report"
]
