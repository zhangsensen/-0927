"""Interfaces to consume screened factor scores for strategy selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from hk_midfreq.config import StrategyRuntimeConfig, DEFAULT_RUNTIME_CONFIG


@dataclass(frozen=True)
class SymbolScore:
    """Aggregated factor score for a single symbol."""

    symbol: str
    timeframe: str
    score: float
    source_session: str


class FactorScoreLoader:
    """Load screened factor scores from enhanced screener outputs."""

    def __init__(
        self,
        runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
        session_id: Optional[str] = None,
    ) -> None:
        self._config = runtime_config
        self._session_id = session_id

    def list_sessions(self) -> List[Path]:
        """Return all available screening sessions sorted by recency."""

        if not self._config.base_output_dir.exists():
            return []

        sessions = [p for p in self._config.base_output_dir.iterdir() if p.is_dir()]
        sessions.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return sessions

    def resolve_session(self) -> Optional[Path]:
        """Resolve the directory containing screening artifacts."""

        if self._session_id is not None:
            target = self._config.base_output_dir / self._session_id
            return target if target.exists() else None

        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def load_factor_table(
        self, symbol: str, timeframe: Optional[str] = None, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """Load raw factor rows for ``symbol`` from the selected session."""

        session_dir = self.resolve_session()
        if session_dir is None:
            raise FileNotFoundError("No screening session directory found.")

        timeframe_dir = session_dir / "timeframes"
        if not timeframe_dir.exists():
            raise FileNotFoundError(
                f"`timeframes` directory missing under {session_dir}"
            )

        frames: List[pd.DataFrame] = []
        for candidate_dir in timeframe_dir.iterdir():
            if not candidate_dir.is_dir():
                continue
            if not candidate_dir.name.startswith(f"{symbol}_"):
                continue
            tf = candidate_dir.name.split("_")[1]
            if timeframe is not None and tf != timeframe:
                continue
            top_factors_file = candidate_dir / "top_factors_detailed.json"
            if not top_factors_file.exists():
                continue
            with top_factors_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not data:
                continue
            df = pd.DataFrame(data)
            df["symbol"] = symbol
            df["timeframe"] = tf
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                f"No factor data found for symbol={symbol} timeframe={timeframe}"
            )

        table = pd.concat(frames, ignore_index=True)
        table.sort_values(by="rank", inplace=True)
        if top_n is not None:
            table = table.head(top_n)
        return table

    def load_symbol_scores(
        self,
        symbols: Iterable[str],
        timeframe: Optional[str] = None,
        top_n: int = 5,
        agg: str = "mean",
    ) -> List[SymbolScore]:
        """Aggregate factor scores for a list of symbols."""

        results: List[SymbolScore] = []
        session_dir = self.resolve_session()
        session_name = session_dir.name if session_dir is not None else ""
        for symbol in symbols:
            try:
                table = self.load_factor_table(symbol, timeframe=timeframe, top_n=top_n)
            except FileNotFoundError:
                continue
            if table.empty or "comprehensive_score" not in table.columns:
                continue
            if agg == "mean":
                score_value = float(table["comprehensive_score"].mean())
            elif agg == "max":
                score_value = float(table["comprehensive_score"].max())
            else:
                raise ValueError(f"Unsupported aggregation method: {agg}")
            results.append(
                SymbolScore(
                    symbol=symbol,
                    timeframe=str(table["timeframe"].iloc[0]),
                    score=score_value,
                    source_session=session_name,
                )
            )
        return results

    def scores_to_series(self, scores: Iterable[SymbolScore]) -> pd.Series:
        """Convert ``SymbolScore`` objects to a descending ``Series``."""

        mapping: Dict[str, float] = {score.symbol: score.score for score in scores}
        series = pd.Series(mapping, name="factor_score")
        series.sort_values(ascending=False, inplace=True)
        return series

    def load_scores_as_series(
        self,
        symbols: Iterable[str],
        timeframe: Optional[str] = None,
        top_n: int = 5,
        agg: str = "mean",
    ) -> pd.Series:
        """Convenience wrapper to fetch aggregated scores as ``Series``."""

        scores = self.load_symbol_scores(
            symbols, timeframe=timeframe, top_n=top_n, agg=agg
        )
        if not scores:
            return pd.Series(dtype=float, name="factor_score")
        return self.scores_to_series(scores)

    def load_all_symbols(
        self, timeframe: Optional[str] = None, top_n: int = 5, agg: str = "mean"
    ) -> pd.Series:
        """Inspect the session folder to load scores for every tracked symbol."""

        session_dir = self.resolve_session()
        if session_dir is None:
            return pd.Series(dtype=float, name="factor_score")
        timeframe_dir = session_dir / "timeframes"
        if not timeframe_dir.exists():
            return pd.Series(dtype=float, name="factor_score")

        symbols = {
            path.name.split("_")[0]
            for path in timeframe_dir.iterdir()
            if path.is_dir() and "_" in path.name
        }
        return self.load_scores_as_series(
            symbols, timeframe=timeframe, top_n=top_n, agg=agg
        )


def load_factor_scores(
    symbols: Iterable[str],
    timeframe: Optional[str] = None,
    loader: Optional[FactorScoreLoader] = None,
    top_n: int = 5,
    agg: str = "mean",
) -> pd.Series:
    """Module-level helper mirroring the original scaffold signature."""

    score_loader = loader or FactorScoreLoader()
    return score_loader.load_scores_as_series(
        symbols, timeframe=timeframe, top_n=top_n, agg=agg
    )
