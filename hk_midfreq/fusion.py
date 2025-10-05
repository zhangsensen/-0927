"""Helper utilities to fuse factor scores across multiple timeframes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from hk_midfreq.config import DEFAULT_RUNTIME_CONFIG, StrategyRuntimeConfig


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
        if method == "ic_weighted" and "mean_ic" in frame.columns:
            weights = frame["mean_ic"].abs()
        else:
            weights = pd.Series(1.0, index=frame.index)

        if weights.sum() == 0:
            weights = pd.Series(1.0, index=frame.index)
        return weights / weights.sum()

    def _aggregate_timeframe(self, frame: pd.DataFrame) -> float:
        if frame.empty:
            return np.nan

        score_column = self._select_score_column(frame)
        weights = self._factor_weights(frame)
        aligned_scores = frame[score_column].reindex(weights.index).astype(float)
        return float((aligned_scores * weights).sum())

    def _combine_timeframes(self, scores: pd.Series) -> float:
        fusion_cfg = self._config.fusion

        trend_tf = fusion_cfg.trend_timeframe
        if trend_tf:
            trend_score = scores.get(trend_tf)
            if pd.notna(trend_score) and trend_score < fusion_cfg.trend_threshold:
                return np.nan

        confirmation_tf = fusion_cfg.confirmation_timeframe
        if confirmation_tf:
            confirmation_score = scores.get(confirmation_tf)
            if (
                pd.notna(confirmation_score)
                and confirmation_score < fusion_cfg.confirmation_threshold
            ):
                return np.nan

        weights = pd.Series(fusion_cfg.timeframe_weights, dtype=float)
        weights = weights[weights.index.isin(scores.index)]
        available_scores = scores[weights.index].dropna()
        if available_scores.empty:
            return np.nan
        weights = weights[available_scores.index]
        total = float(weights.sum())
        if total == 0.0:
            return float(available_scores.mean())
        normalized = weights / total
        return float((available_scores * normalized).sum())

    def fuse(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a factor panel into per-symbol composite scores."""

        if panel.empty:
            return pd.DataFrame()
        if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 3:
            raise ValueError(
                "Factor panel must be indexed by (symbol, timeframe, factor_name)."
            )

        grouped = panel.groupby(level=["symbol", "timeframe"], sort=False)
        timeframe_scores = grouped.apply(self._aggregate_timeframe)
        timeframe_scores.name = "score"
        timeframe_matrix = timeframe_scores.pivot_table(
            index="symbol", columns="timeframe", values="score"
        )

        composite = timeframe_matrix.apply(self._combine_timeframes, axis=1)
        timeframe_matrix["composite_score"] = composite
        return timeframe_matrix.sort_values(by="composite_score", ascending=False)

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
