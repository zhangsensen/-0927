"""Helper utilities to fuse factor scores across multiple timeframes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from hk_midfreq.config import DEFAULT_RUNTIME_CONFIG, StrategyRuntimeConfig

# 设置日志
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusedScores:
    """Container describing the composite factor view for a symbol."""

    symbol: str
    timeframe_scores: pd.Series
    composite_score: float


class FactorFusionEngine:
    """Fuse per-factor panels into composite scores for each symbol."""

    def __init__(
        self,
        runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
    ) -> None:
        self._config = runtime_config

    def _select_score_column(self, frame: pd.DataFrame) -> str:
        if "comprehensive_score" in frame.columns:
            return "comprehensive_score"
        if "predictive_score" in frame.columns:
            return "predictive_score"
        raise KeyError("No supported score column present in factor frame")

    def _factor_weights(self, frame: pd.DataFrame) -> pd.Series:
        method = self._config.fusion.factor_weighting.lower()
        logger.debug(f"计算因子权重 - 方法: {method}, 因子数量: {len(frame)}")

        if method == "ic_weighted" and "mean_ic" in frame.columns:
            weights = frame["mean_ic"].abs()
            logger.debug(
                f"使用IC加权 - IC统计: 最大={weights.max():.4f}, 最小={weights.min():.4f}, 均值={weights.mean():.4f}"
            )
        else:
            weights = pd.Series(1.0, index=frame.index)
            logger.debug(f"使用等权重 - 每个因子权重: {1.0/len(frame):.4f}")

        if weights.sum() == 0:
            logger.warning("权重总和为0，回退到等权重")
            weights = pd.Series(1.0, index=frame.index)

        normalized_weights = weights / weights.sum()

        # 记录权重分配详情
        for factor_name, weight in normalized_weights.items():
            logger.debug(f"因子权重: {factor_name} = {weight:.4f}")

        return normalized_weights

    def _aggregate_timeframe(self, frame: pd.DataFrame) -> float:
        if frame.empty:
            logger.debug("时间框架数据为空，返回NaN")
            return np.nan

        try:
            score_column = self._select_score_column(frame)
            logger.debug(f"选择评分列: {score_column}")
        except KeyError as e:
            logger.warning(f"未找到支持的评分列: {e}")
            return np.nan

        weights = self._factor_weights(frame)
        aligned_scores = frame[score_column].reindex(weights.index).astype(float)

        valid_scores = aligned_scores.dropna()
        if valid_scores.empty:
            logger.debug("有效评分为空，返回NaN")
            return np.nan

        logger.debug(
            f"有效评分数量: {len(valid_scores)}, 评分范围: [{valid_scores.min():.4f}, {valid_scores.max():.4f}]"
        )

        weighted_score = float((aligned_scores * weights).sum())
        logger.debug(f"时间框架聚合结果: {weighted_score:.4f}")

        return weighted_score

    def _combine_timeframes(self, scores: pd.Series) -> float:
        fusion_cfg = self._config.fusion
        logger.debug(f"开始跨时间框架融合 - 输入时间框架: {list(scores.index)}")

        # 趋势过滤
        trend_tf = fusion_cfg.trend_timeframe
        if trend_tf:
            trend_score = scores.get(trend_tf)
            logger.debug(
                f"趋势过滤 - 时间框架: {trend_tf}, 评分: {trend_score}, 阈值: {fusion_cfg.trend_threshold}"
            )
            if pd.notna(trend_score) and trend_score < fusion_cfg.trend_threshold:
                logger.info(
                    f"趋势过滤失败: {trend_score:.4f} < {fusion_cfg.trend_threshold:.4f}"
                )
                return np.nan

        # 确认过滤
        confirmation_tf = fusion_cfg.confirmation_timeframe
        if confirmation_tf:
            confirmation_score = scores.get(confirmation_tf)
            logger.debug(
                f"确认过滤 - 时间框架: {confirmation_tf}, 评分: {confirmation_score}, 阈值: {fusion_cfg.confirmation_threshold}"
            )
            if (
                pd.notna(confirmation_score)
                and confirmation_score < fusion_cfg.confirmation_threshold
            ):
                logger.info(
                    f"确认过滤失败: {confirmation_score:.4f} < {fusion_cfg.confirmation_threshold:.4f}"
                )
                return np.nan

        # 时间框架权重分配
        weights = pd.Series(fusion_cfg.timeframe_weights, dtype=float)
        weights = weights[weights.index.isin(scores.index)]
        available_scores = scores[weights.index].dropna()

        logger.debug(f"原始时间框架权重: {dict(fusion_cfg.timeframe_weights)}")
        logger.debug(f"可用评分数量: {len(available_scores)}")

        if available_scores.empty:
            logger.warning("无可用评分，返回NaN")
            return np.nan

        weights = weights[available_scores.index]
        total = float(weights.sum())

        if total == 0.0:
            logger.warning("权重总和为0，使用等权重平均")
            composite_score = float(available_scores.mean())
        else:
            normalized = weights / total
            # 记录最终权重分配
            for tf, weight in normalized.items():
                score = available_scores.get(tf, np.nan)
                logger.debug(
                    f"时间框架融合: {tf} = 评分{score:.4f} × 权重{weight:.4f} = 贡献{score*weight:.4f}"
                )
            composite_score = float((available_scores * normalized).sum())

        logger.info(f"跨时间框架融合完成 - 复合评分: {composite_score:.4f}")
        return composite_score

    def fuse(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a factor panel into per-symbol composite scores."""

        logger.info(f"开始因子融合 - 输入面板形状: {panel.shape}")

        if panel.empty:
            logger.warning("因子面板为空，返回空DataFrame")
            return pd.DataFrame()

        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 3:
            raise ValueError(
                "Factor panel must be indexed by (symbol, timeframe, factor_name)."
            )

        # 获取基本统计信息
        symbols = panel.index.get_level_values("symbol").unique()
        timeframes = panel.index.get_level_values("timeframe").unique()
        factors = panel.index.get_level_values("factor_name").unique()

        logger.info(
            f"融合统计 - 股票: {len(symbols)}个, 时间框架: {len(timeframes)}个, 因子: {len(factors)}个"
        )
        logger.debug(f"股票列表: {list(symbols)}")
        logger.debug(f"时间框架列表: {list(timeframes)}")
        logger.debug(f"因子列表: {list(factors)}")

        # 按(股票, 时间框架)分组聚合
        logger.debug("开始按时间框架聚合因子...")
        grouped = panel.groupby(level=["symbol", "timeframe"], sort=False)
        timeframe_scores = grouped.apply(self._aggregate_timeframe)
        timeframe_scores.name = "score"

        logger.info(
            f"时间框架聚合完成 - 有效评分: {timeframe_scores.notna().sum()}/{len(timeframe_scores)}"
        )

        # 转换为透视表
        timeframe_matrix = timeframe_scores.reset_index().pivot_table(
            index="symbol", columns="timeframe", values="score"
        )

        logger.debug(f"透视表形状: {timeframe_matrix.shape}")

        # 跨时间框架融合
        logger.debug("开始跨时间框架融合...")
        composite = timeframe_matrix.apply(self._combine_timeframes, axis=1)
        timeframe_matrix["composite_score"] = composite

        # 统计最终结果
        valid_composite = composite.notna().sum()
        logger.info(f"因子融合完成 - 有效复合评分: {valid_composite}/{len(composite)}")

        if valid_composite > 0:
            logger.debug(
                f"复合评分统计: 最高={composite.max():.4f}, 最低={composite.min():.4f}, 均值={composite.mean():.4f}"
            )

        result = timeframe_matrix.sort_values(by="composite_score", ascending=False)
        logger.info(f"因子融合结果排序完成 - 返回形状: {result.shape}")

        return result

    def fuse_symbol(self, panel: pd.DataFrame, symbol: str) -> Optional[FusedScores]:
        """Convenience accessor returning ``FusedScores`` for a single symbol."""

        if panel.empty:
            return None
        try:
            symbol_frame = panel.xs(symbol, level="symbol")
        except KeyError:
            return None

        grouped = symbol_frame.groupby(level="timeframe", sort=False)
        timeframe_scores = grouped.apply(self._aggregate_timeframe)
        composite = self._combine_timeframes(timeframe_scores)
        return FusedScores(
            symbol=symbol,
            timeframe_scores=timeframe_scores,
            composite_score=composite,
        )


__all__ = ["FactorFusionEngine", "FusedScores"]
