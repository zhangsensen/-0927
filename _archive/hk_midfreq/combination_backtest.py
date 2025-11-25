"""组合回测核心组件 - 严格遵循VECTORBT_COMBINATION_BACKTEST_DESIGN.md"""

from __future__ import annotations

import itertools
import json
import logging
import math
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# 配置matplotlib以支持中文并抑制字体警告
import matplotlib
import numpy as np
import pandas as pd
import vectorbt as vbt
from joblib import Parallel, delayed

matplotlib.use("Agg")  # 使用非交互式后端
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "DejaVu Sans",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib.font_manager"
)

from hk_midfreq.config import (
    DEFAULT_RUNTIME_CONFIG,
    PathConfig,
    StrategyRuntimeConfig,
)
from hk_midfreq.factor_interface import FactorScoreLoader
from hk_midfreq.price_loader import PriceDataLoader
from hk_midfreq.result_manager import BacktestResultManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CombinationBacktestConfig:
    """组合回测配置 - 含性能与指标权重"""

    combination_sizes: Tuple[int, ...] = (3, 5, 8, 10, 12)
    max_combinations: int = 10_000
    correlation_threshold: float = 0.7
    random_state: int = 42
    chunk_size: int = 1_000
    parallel_jobs: int = -1
    entry_quantile: float = 0.75
    exit_quantile: float = 0.25
    hold_days: int = 5
    max_factor_pool_size: int = 50
    metrics_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "sharpe_ratio": 0.4,
            "max_drawdown": 0.3,
            "total_return": 0.2,
            "win_rate": 0.1,
        }
    )


class FactorCombinationEngine:
    """因子组合生成引擎 - 支持相关性过滤与智能采样"""

    def __init__(
        self,
        factor_pool: Sequence[str],
        correlation_matrix: pd.DataFrame,
        config: CombinationBacktestConfig,
    ) -> None:
        self.factor_pool: List[str] = list(dict.fromkeys(factor_pool))
        self.correlation_matrix = correlation_matrix.reindex(
            index=self.factor_pool, columns=self.factor_pool
        )
        self.config = config
        self._random = random.Random(config.random_state)

    def generate_combinations(
        self,
        sizes: Sequence[int] | None = None,
        max_combinations: Optional[int] = None,
    ) -> List[Tuple[str, ...]]:
        """生成通过相关性过滤的因子组合列表"""

        target_sizes = tuple(sizes or self.config.combination_sizes)
        max_total = max_combinations or self.config.max_combinations
        if max_total <= 0:
            raise ValueError("max_combinations 必须大于0")

        per_size_quota = math.ceil(max_total / len(target_sizes))
        results: List[Tuple[str, ...]] = []

        for size in target_sizes:
            quota = min(per_size_quota, max_total - len(results))
            if quota <= 0:
                break
            combos = self._generate_for_size(size=size, quota=quota)
            results.extend(combos)
            if len(results) >= max_total:
                break

        return results[:max_total]

    def correlation_filter(
        self,
        factors: Sequence[str],
        max_correlation: Optional[float] = None,
    ) -> bool:
        """判断因子组合是否满足相关性阈值"""

        threshold = max_correlation or self.config.correlation_threshold
        if len(factors) <= 1:
            return True

        for idx, left in enumerate(factors[:-1]):
            right = list(factors[idx + 1 :])
            if left not in self.correlation_matrix.index:
                continue
            corr_row = self.correlation_matrix.loc[left, right]
            corr_values = np.abs(np.asarray(corr_row, dtype=float))
            if corr_values.size == 0:
                continue
            if np.isnan(corr_values).all():
                continue
            if np.nanmax(corr_values) > threshold:
                return False
        return True

    def smart_sampling(
        self,
        factor_pool: Sequence[str],
        target_size: int,
        samples_needed: int,
    ) -> List[Tuple[str, ...]]:
        """智能随机采样组合 - 避免组合爆炸"""

        pool = list(factor_pool)
        if len(pool) < target_size:
            return []

        max_attempts = samples_needed * 20
        seen: set[Tuple[str, ...]] = set()
        samples: List[Tuple[str, ...]] = []
        attempts = 0

        while len(samples) < samples_needed and attempts < max_attempts:
            attempts += 1
            candidate = tuple(sorted(self._random.sample(pool, target_size)))
            if candidate in seen:
                continue
            seen.add(candidate)
            if self.correlation_filter(candidate):
                samples.append(candidate)

        return samples

    def _generate_for_size(self, size: int, quota: int) -> List[Tuple[str, ...]]:
        total = math.comb(len(self.factor_pool), size)
        combos: List[Tuple[str, ...]] = []

        if total <= quota * 2:
            for candidate in itertools.combinations(self.factor_pool, size):
                if self.correlation_filter(candidate):
                    combos.append(candidate)
                if len(combos) >= quota:
                    break
            return combos

        return self.smart_sampling(
            factor_pool=self.factor_pool, target_size=size, samples_needed=quota
        )


class VectorBTBatchBacktester:
    """VectorBT批量回测器 - 支持向量化组合回测"""

    def __init__(
        self,
        price_data: pd.Series,
        factor_frames: Mapping[str, pd.DataFrame],
        config: CombinationBacktestConfig,
    ) -> None:
        # 确保索引为DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        self.price_data = price_data.sort_index()

        self.factor_frames = {}
        for tf, frame in factor_frames.items():
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame.index = pd.to_datetime(frame.index)
            self.factor_frames[tf] = frame.sort_index()

        self.config = config
        self._normalized_cache: Dict[str, pd.DataFrame] = {}

    def build_signal_matrix(
        self,
        combinations: Sequence[Tuple[str, ...]],
        timeframe: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构建组合入场与出场信号矩阵"""

        if timeframe not in self.factor_frames:
            raise KeyError(f"未发现时间框架数据: {timeframe}")

        factor_df = self._zscore_frame(timeframe)
        index = factor_df.index
        factor_names = factor_df.columns.tolist()
        factor_index: Dict[str, int] = {name: i for i, name in enumerate(factor_names)}
        grouped: Dict[int, List[Tuple[str, ...]]] = {}
        for combo in combinations:
            grouped.setdefault(len(combo), []).append(combo)

        composite_list: List[np.ndarray] = []
        column_labels: List[str] = []

        factor_values = factor_df.values  # (T, F)

        for size, combos in grouped.items():
            if not combos:
                continue
            indices = np.array(
                [[factor_index[name] for name in combo] for combo in combos],
                dtype=np.int32,
            )
            # shape -> (T, N, size)
            composite = factor_values[:, indices]
            composite_mean = composite.mean(axis=2)
            composite_list.append(composite_mean)
            column_labels.extend([",".join(combo) for combo in combos])

        if not composite_list:
            return (
                pd.DataFrame(index=index, columns=[]),
                pd.DataFrame(index=index, columns=[]),
            )

        composite_matrix = np.concatenate(composite_list, axis=1)
        q_high = np.nanquantile(composite_matrix, self.config.entry_quantile, axis=0)
        q_low = np.nanquantile(composite_matrix, self.config.exit_quantile, axis=0)

        entries = composite_matrix > q_high[np.newaxis, :]
        exits = composite_matrix < q_low[np.newaxis, :]

        entries_df = pd.DataFrame(entries, index=index, columns=column_labels)
        exits_df = pd.DataFrame(exits, index=index, columns=column_labels)
        entries_df = entries_df.astype(bool)
        exits_df = exits_df.astype(bool)
        mask = entries_df.any(axis=0)
        entries_df = entries_df.loc[:, mask]
        exits_df = exits_df.loc[:, mask]
        return entries_df.astype(bool), exits_df.astype(bool)

    def batch_backtest(
        self,
        combinations: Sequence[Tuple[str, ...]],
        timeframe: str,
    ) -> pd.DataFrame:
        """并行执行批量回测并返回指标结果"""

        if not combinations:
            return pd.DataFrame()

        chunk_size = self.config.chunk_size
        chunks: List[List[Tuple[str, ...]]] = [
            combinations[i : i + chunk_size]
            for i in range(0, len(combinations), chunk_size)
        ]

        logger.info(
            "启动批量回测 - 组合总数: %s, 分块: %s, 时间框架: %s",
            len(combinations),
            len(chunks),
            timeframe,
        )

        results = Parallel(n_jobs=self.config.parallel_jobs, backend="loky")(
            delayed(self.chunked_backtest)(chunk, timeframe) for chunk in chunks
        )

        if not results:
            return pd.DataFrame()

        merged = pd.concat(results, axis=0)
        merged = merged[~merged.index.duplicated(keep="first")]
        return merged

    def chunked_backtest(
        self,
        combinations: Sequence[Tuple[str, ...]],
        timeframe: str,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """对组合分块执行回测，返回单块回测结果"""

        if not combinations:
            return pd.DataFrame()

        entries, exits = self.build_signal_matrix(combinations, timeframe)
        if entries.empty:
            return pd.DataFrame()

        portfolio = vbt.Portfolio.from_signals(  # type: ignore[call-arg]
            close=self.price_data.reindex(entries.index),
            entries=entries,
            exits=exits,
            freq="1D",
            group_by=False,
        )

        metrics_df = pd.DataFrame(
            {
                "total_return": portfolio.total_return(),
                "sharpe_ratio": portfolio.sharpe_ratio(),
                "max_drawdown": portfolio.max_drawdown(),
                "win_rate": portfolio.trades.win_rate(),
                "profit_factor": portfolio.trades.profit_factor(),
            }
        )
        metrics_df.index.name = "combination"
        metrics_df = metrics_df.reset_index()
        if metrics_df.empty:
            return metrics_df

        metrics_df["num_factors"] = metrics_df["combination"].map(
            lambda x: len(str(x).split(","))
        )
        metrics_df = metrics_df.set_index("combination")
        return metrics_df

    def _zscore_frame(self, timeframe: str) -> pd.DataFrame:
        if timeframe in self._normalized_cache:
            return self._normalized_cache[timeframe]

        frame = self.factor_frames[timeframe]
        normalized = (frame - frame.mean()) / frame.std(ddof=0)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._normalized_cache[timeframe] = normalized
        return normalized


class CombinationOptimizer:
    """组合性能优化器 - 负责评分与帕累托分析"""

    def __init__(self, metrics_weights: Mapping[str, float]) -> None:
        self.metrics_weights = dict(metrics_weights)

    def evaluate_combinations(self, backtest_results: pd.DataFrame) -> pd.DataFrame:
        if backtest_results.empty:
            return backtest_results

        scores = []
        for _, row in backtest_results.iterrows():
            metrics = row.to_dict()
            scores.append(self.calculate_composite_score(metrics))
        backtest_results = backtest_results.copy()
        backtest_results["composite_score"] = scores
        return backtest_results.sort_values(by="composite_score", ascending=False)

    def calculate_composite_score(self, metrics: Mapping[str, float]) -> float:
        score = 0.0
        for metric, weight in self.metrics_weights.items():
            value = metrics.get(metric, 0.0) or 0.0
            if metric == "max_drawdown":
                value = -value
            score += weight * float(value)
        return score

    def pareto_optimal_frontier(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results

        metrics = ["total_return", "max_drawdown", "sharpe_ratio"]
        subset = results[metrics].fillna(0.0).to_numpy()
        dominated = np.zeros(len(subset), dtype=bool)

        for i, point in enumerate(subset):
            if dominated[i]:
                continue
            dom_mask = np.all(subset >= point, axis=1)
            better_mask = np.any(subset > point, axis=1)
            dominated |= dom_mask & better_mask
            dominated[i] = False

        frontier = results.loc[~dominated]
        return frontier.sort_values(by="composite_score", ascending=False)


class PerformanceAnalyzer:
    """绩效分析器 - 负责可视化与因子重要性分析"""

    def factor_importance_analysis(
        self, top_combinations: Sequence[Tuple[str, ...]]
    ) -> pd.Series:
        counter: Dict[str, int] = {}
        for combo in top_combinations:
            for factor in combo:
                counter[factor] = counter.get(factor, 0) + 1
        if not counter:
            return pd.Series(dtype=int)
        importance = pd.Series(counter).sort_values(ascending=False)
        importance.name = "count"
        return importance

    def generate_factor_importance_plot(
        self, importance: pd.Series, output_path: str
    ) -> None:
        if importance.empty:
            return
        ax = importance.plot(kind="bar", figsize=(12, 6), title="Factor Importance")
        ax.set_ylabel("使用次数")
        ax.figure.tight_layout()
        ax.figure.savefig(output_path)
        ax.figure.clf()

    def generate_performance_heatmap(
        self, results: pd.DataFrame, output_path: str
    ) -> None:
        if results.empty:
            return
        heatmap = results.groupby("num_factors")["composite_score"].mean()
        ax = heatmap.plot(
            kind="bar", figsize=(10, 5), title="Composite Score by Combination Size"
        )
        ax.set_ylabel("平均综合得分")
        ax.figure.tight_layout()
        ax.figure.savefig(output_path)
        ax.figure.clf()


def load_factor_pool(
    loader: FactorScoreLoader,
    symbol: str,
    timeframes: Sequence[str],
    max_factors: int,
) -> List[str]:
    panel = loader.load_factor_panels(
        symbols=[symbol], timeframes=timeframes, max_factors=max_factors
    )
    if panel.empty:
        raise ValueError("因子面板为空，无法进行组合回测")
    index = panel.index
    if isinstance(index, pd.MultiIndex):
        level_names = list(index.names)
        if "factor_name" in level_names:
            names = index.get_level_values("factor_name")
        else:
            names = index.get_level_values(-1)
        return pd.Index(names).dropna().unique().astype(str).tolist()

    columns = panel.columns
    if "factor_name" in columns:
        names = panel["factor_name"].astype(str)
    elif "name" in columns:
        names = panel["name"].astype(str)
    else:
        raise KeyError(
            f"因子面板缺少factor_name信息，索引: {index.names}, 列: {list(columns)}"
        )
    return names.dropna().unique().tolist()


def load_factor_time_series(
    loader: FactorScoreLoader,
    symbol: str,
    timeframes: Sequence[str],
    factor_names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        frame = loader.load_factor_time_series(
            symbol=symbol,
            timeframe=timeframe,
            factor_names=list(factor_names),
        )
        frames[timeframe] = frame
    return frames


def prepare_correlation_matrix(
    factor_frame: pd.DataFrame,
) -> pd.DataFrame:
    corr = factor_frame.corr()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return corr


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_combination_backtest(
    symbol: str,
    timeframe: str,
    runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
    combination_config: Optional[CombinationBacktestConfig] = None,
    session_id: Optional[str] = None,
) -> Dict[str, object]:
    """执行组合回测，生成指定产出并返回结果摘要"""
    import os

    import psutil

    # 性能监控初始化
    process = psutil.Process(os.getpid())
    start_time = datetime.now()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu_times = process.cpu_times()

    combo_config = combination_config or CombinationBacktestConfig()

    path_config: PathConfig
    if hasattr(runtime_config, "paths"):
        path_config = runtime_config.paths  # type: ignore[assignment]
    else:
        path_config = PathConfig()

    # 创建会话目录（简化版，暂不使用result_manager以避免复杂度）
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = (
        session_id
        or f"{symbol.replace('.', '_')}_{timeframe}_combo_{session_timestamp}"
    )
    session_dir = path_config.backtest_output_dir / session_id
    ensure_directory(session_dir)

    # 创建logs和charts目录
    logs_dir = session_dir / "logs"
    ensure_directory(logs_dir)
    charts_dir = session_dir / "charts"
    ensure_directory(charts_dir)

    # 配置会话级日志
    log_file = logs_dir / "combination_backtest.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"组合回测会话创建: {session_id}")
    logger.info(f"标的: {symbol}, 时间框架: {timeframe}")
    logger.info(
        f"配置: max_combinations={combo_config.max_combinations}, correlation_threshold={combo_config.correlation_threshold}"
    )

    # 注册会话到索引
    from hk_midfreq.session_index_manager import SessionIndexManager

    index_manager = SessionIndexManager(path_config.backtest_output_dir)
    index_manager.register_session(
        session_id=session_id,
        metadata={
            "type": "combination",
            "symbol": symbol,
            "timeframe": timeframe,
            "started_at": datetime.now().isoformat(),
        },
    )

    loader = FactorScoreLoader(runtime_config=runtime_config)
    price_loader = PriceDataLoader(path_config=path_config)

    fusion_timeframes: Sequence[str]
    if hasattr(runtime_config, "fusion"):
        fusion_timeframes = runtime_config.fusion.ordered_timeframes()  # type: ignore[attr-defined]
    else:
        fusion_timeframes = [timeframe]

    factor_pool = load_factor_pool(
        loader=loader,
        symbol=symbol,
        timeframes=fusion_timeframes,
        max_factors=combo_config.max_factor_pool_size,
    )

    primary_timeframe = timeframe
    factor_frames = load_factor_time_series(
        loader=loader,
        symbol=symbol,
        timeframes=[primary_timeframe],
        factor_names=factor_pool,
    )
    primary_frame = factor_frames[primary_timeframe]
    available_factors = [name for name in factor_pool if name in primary_frame.columns]
    if not available_factors:
        raise RuntimeError("在因子输出层中未找到符合要求的因子")
    primary_frame = primary_frame[available_factors]
    correlation_matrix = prepare_correlation_matrix(primary_frame)

    engine = FactorCombinationEngine(
        factor_pool=available_factors,
        correlation_matrix=correlation_matrix,
        config=combo_config,
    )
    combinations = engine.generate_combinations()
    if not combinations:
        raise RuntimeError("组合生成失败：未找到有效的因子组合")

    price_df = price_loader.load_price(symbol=symbol, timeframe=primary_timeframe)
    price_series = price_df["close"].sort_index()

    backtester = VectorBTBatchBacktester(
        price_data=price_series,
        factor_frames=factor_frames,
        config=combo_config,
    )
    backtest_results = backtester.batch_backtest(
        combinations=combinations,
        timeframe=primary_timeframe,
    )
    if backtest_results.empty:
        raise RuntimeError("回测结果为空，无法继续")

    optimizer = CombinationOptimizer(combo_config.metrics_weights)
    evaluated = optimizer.evaluate_combinations(backtest_results)
    frontier = optimizer.pareto_optimal_frontier(evaluated)

    analyzer = PerformanceAnalyzer()
    top_combos = [tuple(combo.split(",")) for combo in evaluated.head(100).index]
    importance = analyzer.factor_importance_analysis(top_combos)

    results_path = session_dir / "combination_backtest_results.csv"
    evaluated.to_csv(results_path)

    top_100_json = session_dir / "top_100_combinations.json"
    top_100_payload = [
        {
            "combination": combo,
            "num_factors": int(row["num_factors"]),
            "total_return": float(row.get("total_return", 0.0) or 0.0),
            "sharpe_ratio": float(row.get("sharpe_ratio", 0.0) or 0.0),
            "max_drawdown": float(row.get("max_drawdown", 0.0) or 0.0),
            "win_rate": float(row.get("win_rate", 0.0) or 0.0),
            "profit_factor": float(row.get("profit_factor", 0.0) or 0.0),
            "composite_score": float(row.get("composite_score", 0.0) or 0.0),
        }
        for combo, row in evaluated.head(100).iterrows()
    ]
    top_100_json.write_text(
        json.dumps(top_100_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    importance_path = charts_dir / "factor_importance_analysis.png"
    analyzer.generate_factor_importance_plot(importance, str(importance_path))

    heatmap_path = charts_dir / "performance_heatmap.png"
    analyzer.generate_performance_heatmap(evaluated, str(heatmap_path))

    # 性能监控收尾
    end_time = datetime.now()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu_times = process.cpu_times()
    elapsed_seconds = (end_time - start_time).total_seconds()

    # 组合规模分布统计
    combo_size_dist = evaluated["num_factors"].value_counts().sort_index().to_dict()

    performance_metrics = {
        "elapsed_seconds": elapsed_seconds,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "memory_start_mb": start_memory,
        "memory_end_mb": end_memory,
        "memory_peak_mb": end_memory,  # 简化版，实际峰值需要持续监控
        "memory_delta_mb": end_memory - start_memory,
        "cpu_user_seconds": end_cpu_times.user - start_cpu_times.user,
        "cpu_system_seconds": end_cpu_times.system - start_cpu_times.system,
        "combinations_per_second": (
            len(combinations) / elapsed_seconds if elapsed_seconds > 0 else 0
        ),
        "combination_size_distribution": combo_size_dist,
    }

    logger.info(
        f"组合回测完成 - 耗时: {elapsed_seconds:.2f}秒, 速度: {performance_metrics['combinations_per_second']:.2f} 组合/秒"
    )
    logger.info(
        f"内存使用: 开始={start_memory:.1f}MB, 结束={end_memory:.1f}MB, 增量={performance_metrics['memory_delta_mb']:.1f}MB"
    )
    logger.info(
        f"CPU时间: 用户={performance_metrics['cpu_user_seconds']:.2f}秒, 系统={performance_metrics['cpu_system_seconds']:.2f}秒"
    )
    logger.info(f"组合规模分布: {combo_size_dist}")

    summary = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "symbol": symbol,
        "timeframe": timeframe,
        "total_combinations": len(combinations),
        "evaluated_combinations": int(evaluated.shape[0]),
        "best_combination": evaluated.index[0] if not evaluated.empty else None,
        "best_score": (
            float(evaluated.iloc[0]["composite_score"]) if not evaluated.empty else None
        ),
        "pareto_count": int(frontier.shape[0]),
        "results_csv": str(results_path),
        "top100_json": str(top_100_json),
        "factor_importance_png": str(importance_path),
        "performance_heatmap_png": str(heatmap_path),
        "performance_metrics": performance_metrics,
    }

    summary_path = session_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 保存独立的性能监控报告
    perf_report_path = session_dir / "performance_metrics.json"
    perf_report_path.write_text(
        json.dumps(performance_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 移除文件handler以避免日志泄漏
    logger.removeHandler(file_handler)
    file_handler.close()

    return summary
