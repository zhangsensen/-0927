"""因子接口 - P0 优化：路径解耦，错误处理标准化"""

from __future__ import annotations

import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from hk_midfreq.config import DEFAULT_RUNTIME_CONFIG, PathConfig, StrategyRuntimeConfig

# 设置日志
logger = logging.getLogger(__name__)


class FactorLoadError(Exception):
    """因子数据加载异常 - P0 优化：标准化错误处理"""

    pass


@dataclass(frozen=True)
class SymbolScore:
    """Aggregated factor score for a single symbol."""

    symbol: str
    timeframe: str
    score: float
    source_session: str


class FactorScoreLoader:
    """Load screened factor scores from enhanced screener outputs.

    P0 优化：
    - 移除硬编码路径，使用 PathConfig
    - 标准化错误处理
    - 保持向后兼容
    """

    def __init__(
        self,
        runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
        session_id: Optional[str] = None,
        path_config: Optional[PathConfig] = None,
    ) -> None:
        self._config = runtime_config or DEFAULT_RUNTIME_CONFIG
        self._session_id = session_id
        self._path_config = path_config or (
            self._config.paths if hasattr(self._config, "paths") else self._config
        )

    def list_sessions(self) -> List[Path]:
        """Return all available screening sessions sorted by recency."""
        # 直接使用factor_ready目录
        output_dir = self._path_config.factor_ready_dir
        if not output_dir.exists():
            return []

        # factor_ready目录包含parquet文件，不是目录结构
        if output_dir.is_dir():
            return [output_dir]
        return []

    def resolve_session(self) -> Optional[Path]:
        """Resolve the directory containing screening artifacts."""

        if self._session_id is not None:
            target = self._config.base_output_dir / self._session_id
            return target if target.exists() else None

        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def _iter_factor_frames(
        self,
        symbols: Optional[Sequence[str]] = None,
        timeframes: Optional[Sequence[str]] = None,
    ) -> Iterator[Tuple[str, str, pd.DataFrame]]:
        """Yield normalized factor tables for each symbol/timeframe combination."""

        session_dir = self.resolve_session()
        if session_dir is None:
            logger.error("未找到factor_ready目录")
            raise FileNotFoundError("No factor_ready directory found.")

        logger.debug(f"使用factor_ready目录: {session_dir}")

        # 直接加载parquet文件
        factor_file = session_dir / "0700_HK_best_factors.parquet"
        if not factor_file.exists():
            raise FileNotFoundError(f"Factor file not found: {factor_file}")

        import pandas as pd

        df = pd.read_parquet(factor_file)
        logger.info(f"加载因子数据: {len(df)}个因子")

        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None

        if symbol_filter and "0700.HK" not in symbol_filter:
            return

        # 按时间框架分组
        for tf in df["timeframe"].unique():
            if timeframe_filter and tf not in timeframe_filter:
                continue

            tf_data = df[df["timeframe"] == tf].copy()

            # 转换列名以匹配系统期望
            tf_data = tf_data.rename(columns={"ic_mean": "mean_ic", "ic_ir": "ic_ir"})

            # 确保必需列存在
            if "mean_ic" not in tf_data.columns:
                tf_data["mean_ic"] = 0.0
            if "ic_ir" not in tf_data.columns:
                tf_data["ic_ir"] = 0.0

            yield "0700.HK", tf, tf_data

    def load_factor_table(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load raw factor rows for ``symbol`` from the selected session."""

        frames = [
            frame
            for _, _, frame in self._iter_factor_frames(
                symbols=[symbol],
                timeframes=[timeframe] if timeframe is not None else None,
            )
        ]

        if not frames:
            raise FileNotFoundError(
                f"No factor data found for symbol={symbol} timeframe={timeframe}"
            )

        table = pd.concat(frames, ignore_index=True)
        table = table.sort_values(by="rank")
        if top_n is not None:
            table = table.head(top_n)
        return table

    def load_factor_panels(
        self,
        symbols: Optional[Sequence[str]] = None,
        timeframes: Optional[Sequence[str]] = None,
        max_factors: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return a MultiIndex panel of factors keyed by (symbol, timeframe)."""

        logger.info(
            f"开始加载因子面板 - 股票: {symbols}, 时间框架: {timeframes}, 最大因子数: {max_factors}"
        )

        frames: List[pd.DataFrame] = []
        total_factors_loaded = 0

        for symbol, timeframe, frame in self._iter_factor_frames(
            symbols=symbols, timeframes=timeframes
        ):
            original_count = len(frame)

            if max_factors is not None and len(frame) > max_factors:
                frame = frame.nsmallest(max_factors, columns="rank")
                logger.debug(
                    f"{symbol}-{timeframe}: 因子筛选 {original_count} -> {len(frame)} (top {max_factors})"
                )
            else:
                logger.debug(f"{symbol}-{timeframe}: 加载 {original_count} 个因子")

            frames.append(frame)
            total_factors_loaded += len(frame)

        if not frames:
            logger.warning("未找到任何因子数据，返回空面板")
            empty = pd.DataFrame(
                columns=[
                    "rank",
                    "factor_name",
                    "comprehensive_score",
                    "predictive_score",
                    "stability_score",
                    "independence_score",
                    "practicality_score",
                    "adaptability_score",
                    "is_significant",
                    "tier",
                    "type",
                    "description",
                    "mean_ic",
                    "ic_ir",
                    "rolling_ic_mean",
                    "vif",
                    "turnover_rate",
                    "transaction_cost",
                    "symbol",
                    "timeframe",
                ]
            )
            empty = empty.set_index(["symbol", "timeframe", "factor_name"])
            return empty

        logger.info(f"因子数据合并中 - 总计 {total_factors_loaded} 个因子记录")
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(by="rank")
        panel = panel.set_index(["symbol", "timeframe", "factor_name"])

        # 统计最终面板信息
        symbols_count = len(panel.index.get_level_values("symbol").unique())
        timeframes_count = len(panel.index.get_level_values("timeframe").unique())
        factors_count = len(panel.index.get_level_values("factor_name").unique())

        logger.info(f"因子面板加载完成 - 形状: {panel.shape}")
        logger.info(
            f"面板统计: {symbols_count}个股票, {timeframes_count}个时间框架, {factors_count}个因子"
        )

        return panel

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

    def load_factor_time_series(
        self,
        symbol: str,
        timeframe: str,
        factor_names: List[str],
        factor_data_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Load factor time series data for a symbol and timeframe.

        P0 优化：使用 PathConfig 自动发现因子输出目录
        """

        if factor_data_dir is None:
            factor_data_dir = self._path_config.factor_output_dir

        # Construct filename pattern
        timeframe_path = factor_data_dir / timeframe
        if not timeframe_path.exists():
            raise FactorLoadError(f"Timeframe directory not found: {timeframe_path}")

        # Find the most recent factor file for this symbol and timeframe
        pattern = f"{symbol}_{timeframe}_factors_*.parquet"
        files = glob.glob(str(timeframe_path / pattern))

        if not files:
            raise FactorLoadError(
                f"No factor data file found for {symbol} {timeframe} "
                f"in {timeframe_path}"
            )

        # Use the most recent file
        latest_file = max(
            files,
            key=lambda x: x.split("_")[-2] + x.split("_")[-1].replace(".parquet", ""),
        )

        # Load the factor data
        factor_data = pd.read_parquet(latest_file)

        # Filter to only include requested factors
        available_factors = [f for f in factor_names if f in factor_data.columns]
        missing_factors = set(factor_names) - set(available_factors)

        if missing_factors:
            print(f"Warning: Missing factors {missing_factors} in {latest_file}")

        if not available_factors:
            raise FactorLoadError(f"No requested factors found in {latest_file}")

        # Return only the requested factor columns
        factor_df = factor_data[available_factors].copy()

        return factor_df

    def scores_to_series(self, scores: Iterable[SymbolScore]) -> pd.Series:
        """Convert ``SymbolScore`` objects to a descending ``Series``."""

        mapping: Dict[str, float] = {score.symbol: score.score for score in scores}
        series = pd.Series(mapping, name="factor_score")
        series = series.sort_values(ascending=False)
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
        self,
        timeframe: Optional[str] = None,
        top_n: int = 5,
        agg: str = "mean",
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
