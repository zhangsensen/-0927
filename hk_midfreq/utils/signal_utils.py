"""信号处理共享工具函数 - 避免代码重复"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def standardize_factor_data(
    factor_df: pd.DataFrame, method: str = "zscore"
) -> pd.DataFrame:
    """
    标准化因子数据

    Args:
        factor_df: 因子数据框 (行=时间, 列=因子)
        method: 标准化方法 ("zscore" 或 "minmax")

    Returns:
        标准化后的因子数据框
    """
    if method == "zscore":
        # Z-score标准化
        mean = factor_df.mean()
        std = factor_df.std(ddof=0)
        normalized = (factor_df - mean) / std.replace(0, 1)  # 避免除零
    elif method == "minmax":
        # Min-Max标准化到[0,1]
        min_val = factor_df.min()
        max_val = factor_df.max()
        range_val = (max_val - min_val).replace(0, 1)
        normalized = (factor_df - min_val) / range_val
    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    # 处理无穷值和NaN
    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return normalized


def align_time_indices(
    price_series: pd.Series,
    factor_df: pd.DataFrame,
    method: str = "intersection",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    对齐价格和因子数据的时间索引

    Args:
        price_series: 价格序列
        factor_df: 因子数据框
        method: 对齐方法
            - "intersection": 仅保留重叠时间点
            - "reindex_ffill": 重索引并前向填充

    Returns:
        (对齐后的价格序列, 对齐后的因子数据框)
    """
    # 确保索引为DatetimeIndex
    if not isinstance(price_series.index, pd.DatetimeIndex):
        price_series.index = pd.to_datetime(price_series.index)
    if not isinstance(factor_df.index, pd.DatetimeIndex):
        factor_df.index = pd.to_datetime(factor_df.index)

    if method == "intersection":
        # 取交集
        common_index = price_series.index.intersection(factor_df.index)

        if len(common_index) == 0:
            raise ValueError("价格和因子数据时间索引完全不重叠")

        aligned_price = price_series.loc[common_index]
        aligned_factor = factor_df.loc[common_index]

        logger.debug(f"时间对齐完成 (intersection): {len(common_index)}个数据点")

    elif method == "reindex_ffill":
        # 重索引到价格时间，前向填充因子
        aligned_factor = factor_df.reindex(price_series.index, method="ffill")

        # 移除NaN行
        valid_mask = aligned_factor.notna().any(axis=1)
        aligned_factor = aligned_factor[valid_mask]
        aligned_price = price_series[valid_mask]

        if len(aligned_price) == 0:
            raise ValueError("对齐后无有效数据点")

        logger.debug(f"时间对齐完成 (reindex_ffill): {len(aligned_price)}个数据点")

    else:
        raise ValueError(f"不支持的对齐方法: {method}")

    return aligned_price, aligned_factor


def calculate_composite_score(
    factor_scores: pd.DataFrame,
    weights: pd.Series | None = None,
    method: str = "mean",
) -> pd.Series:
    """
    计算因子复合得分

    Args:
        factor_scores: 标准化后的因子得分 (行=时间, 列=因子)
        weights: 因子权重 (索引=因子名)，None表示等权
        method: 聚合方法 ("mean", "median", "weighted_mean")

    Returns:
        复合得分序列
    """
    if method == "mean":
        composite = factor_scores.mean(axis=1)

    elif method == "median":
        composite = factor_scores.median(axis=1)

    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("加权平均需要提供权重")

        # 确保权重对齐
        aligned_weights = weights.reindex(factor_scores.columns, fill_value=0)
        aligned_weights = aligned_weights / aligned_weights.sum()  # 归一化

        # 加权求和
        composite = (factor_scores * aligned_weights.values).sum(axis=1)

    else:
        raise ValueError(f"不支持的聚合方法: {method}")

    return composite


def generate_quantile_signals(
    composite_score: pd.Series,
    entry_quantile: float = 0.75,
    exit_quantile: float = 0.25,
) -> Tuple[pd.Series, pd.Series]:
    """
    基于分位数生成入场/出场信号

    Args:
        composite_score: 复合得分序列
        entry_quantile: 入场分位数阈值（例如0.75表示超过75%分位数时入场）
        exit_quantile: 出场分位数阈值（例如0.25表示低于25%分位数时出场）

    Returns:
        (入场信号序列, 出场信号序列)，均为布尔型
    """
    upper_threshold = composite_score.quantile(entry_quantile)
    lower_threshold = composite_score.quantile(exit_quantile)

    entries = (composite_score > upper_threshold).astype(bool)
    exits = (composite_score < lower_threshold).astype(bool)

    logger.debug(
        f"信号生成完成 - 入场阈值: {upper_threshold:.4f}, "
        f"出场阈值: {lower_threshold:.4f}, "
        f"入场信号: {entries.sum()}, 出场信号: {exits.sum()}"
    )

    return entries, exits


def calculate_correlation_matrix(
    factor_df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """
    计算因子相关性矩阵

    Args:
        factor_df: 因子数据框
        method: 相关性计算方法 ("pearson", "spearman")

    Returns:
        相关性矩阵
    """
    corr = factor_df.corr(method=method)

    # 处理无穷值和NaN
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return corr

