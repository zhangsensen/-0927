"""Strategy logic and candidate selection for the HK mid-frequency stack."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import vectorbt as vbt

from hk_midfreq.config import (
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_TRADING_CONFIG,
    ExecutionConfig,
    StrategyRuntimeConfig,
    TradingConfig,
)
from hk_midfreq.factor_interface import FactorLoadError, FactorScoreLoader
from hk_midfreq.fusion import FactorFusionEngine
from factor_system.factor_engine import api
from factor_system.factor_engine.core.registry import get_global_registry
from factor_system.shared.factor_calculators import SHARED_CALCULATORS

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class StrategySignals:
    """Bundle of entry/exit signals and risk settings for one symbol."""

    symbol: str
    timeframe: str
    entries: pd.Series
    exits: pd.Series
    stop_loss: float
    take_profit: float

    def as_dict(self) -> Dict[str, pd.Series | float | str]:
        """Serialize the signal bundle into a mapping."""

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entries": self.entries,
            "exits": self.exits,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


def _compute_time_based_exits(entries: pd.Series, hold_days: int) -> pd.Series:
    """Generate exit signals ``hold_days`` bars after each entry."""

    cleaned_entries = entries.fillna(False).astype(bool)
    exits = pd.Series(False, index=cleaned_entries.index)
    if hold_days <= 0:
        return exits

    entry_positions = np.flatnonzero(cleaned_entries.to_numpy())
    if entry_positions.size == 0:
        return exits

    exit_positions = entry_positions + hold_days
    exit_positions = exit_positions[exit_positions < len(cleaned_entries.index)]
    if exit_positions.size == 0:
        return exits

    exits.iloc[exit_positions] = True
    return exits


def hk_reversal_logic(
    close: pd.Series,
    volume: pd.Series,
    hold_days: int,
    rsi_window: int = 14,
    bb_window: int = 20,
    volume_window: int = 5,
    rsi_threshold: float = 30.0,
    volume_multiplier: float = 1.2,
) -> StrategySignals:
    """Generate reversal signals using RSI, Bollinger Bands, and volume confirmation."""

    if close.empty:
        raise ValueError("Close series cannot be empty")

    aligned_volume = volume.reindex(close.index).ffill()

    # 使用共享计算器确保与factor_engine、factor_generation完全一致
    rsi = SHARED_CALCULATORS.calculate_rsi(close, period=rsi_window)
    bb = SHARED_CALCULATORS.calculate_bbands(close, period=bb_window)
    rolling_volume = aligned_volume.rolling(
        window=volume_window, min_periods=volume_window
    ).mean()

    cond_rsi = rsi < rsi_threshold
    cond_bb = close <= bb['lower']
    cond_vol = aligned_volume >= (rolling_volume * volume_multiplier)

    entries = (cond_rsi & cond_bb & cond_vol).fillna(False)
    exits = _compute_time_based_exits(entries, hold_days)

    return StrategySignals(
        symbol="",
        timeframe="",
        entries=entries.astype(bool),
        exits=exits,
        stop_loss=DEFAULT_EXECUTION_CONFIG.stop_loss,
        take_profit=DEFAULT_EXECUTION_CONFIG.primary_take_profit(),
    )


@dataclass(frozen=True)
class FactorDescriptor:
    """Minimal payload describing a selected factor for signal generation."""

    name: str
    timeframe: str
    metadata: Mapping[str, object] | None = None


def _normalize_timeframe_label(label: str) -> str:
    """Normalize timeframe aliases (e.g. ``60min`` -> ``60m``)."""

    lowered = label.lower()
    if lowered in {"60m", "60min", "1h"}:
        return "60m"
    if lowered in {"30m", "30min"}:
        return "30m"
    if lowered in {"15m", "15min"}:
        return "15m"
    if lowered in {"5m", "5min"}:
        return "5m"
    if lowered in {"1d", "1day", "daily", "day"}:
        return "1day"
    return lowered


def _compute_stochrsi(
    close: pd.Series,
    timeperiod: int,
    fastk_period: int,
    fastd_period: int,
) -> tuple[pd.Series, pd.Series]:
    """Return the %K and %D series for a classic StochRSI calculation."""

    if close.empty:
        raise ValueError("Close price series cannot be empty when computing StochRSI")

    # 使用共享计算器确保与factor_engine、factor_generation完全一致
    rsi = SHARED_CALCULATORS.calculate_rsi(close, period=timeperiod)
    lowest = rsi.rolling(window=timeperiod, min_periods=timeperiod).min()
    highest = rsi.rolling(window=timeperiod, min_periods=timeperiod).max()
    denominator = (highest - lowest).replace(0.0, np.nan)
    stoch_rsi = (rsi - lowest) / denominator
    stoch_rsi = stoch_rsi.clip(lower=0.0, upper=1.0).ffill().fillna(0.0)

    fastk = (
        stoch_rsi.rolling(window=fastk_period, min_periods=fastk_period)
        .mean()
        .fillna(0.0)
    )
    fastd = (
        fastk.rolling(window=fastd_period, min_periods=fastd_period).mean().fillna(0.0)
    )
    return fastk * 100.0, fastd * 100.0


def _parse_stochrsi_params(name: str) -> tuple[int, int, int, str]:
    """Extract timeperiod, fastk, fastd settings and target line from name."""

    tokens = name.split("_")
    timeperiod = 14
    fastk = 5
    fastd = 3
    line = "k"

    for token in tokens:
        match = re.search(r"timeperiod(\d+)", token)
        if match:
            timeperiod = int(match.group(1))
        match = re.search(r"fastk(?:_period)?(\d+)", token)
        if match:
            fastk = int(match.group(1))
        match = re.search(r"fastd(?:_period)?(\d+)", token)
        if match:
            fastd = int(match.group(1))
        if token.lower() in {"k", "d"}:
            line = token.lower()

    if name.endswith("_K"):
        line = "k"
    elif name.endswith("_D"):
        line = "d"

    return timeperiod, fastk, fastd, line


def generate_multi_factor_signals(
    symbol: str,
    timeframe: str,
    close: pd.Series,
    volume: Optional[pd.Series],
    factor_names: List[str],
    factor_loader: FactorScoreLoader,
    hold_days: int,
    stop_loss: float,
    take_profit: float,
) -> StrategySignals:
    """
    使用多因子融合算法生成交易信号

    Args:
        symbol: 股票代码
        timeframe: 时间框架
        close: 收盘价序列
        volume: 成交量序列
        factor_names: 因子名称列表
        factor_loader: 因子加载器
        hold_days: 持仓天数
        stop_loss: 止损比例
        take_profit: 止盈比例

    Returns:
        StrategySignals: 交易信号
    """
    logger.info(f"{symbol} 开始多因子融合信号生成 - 因子数量: {len(factor_names)}")

    try:
        # 1. 从factor_output加载因子时间序列数据
        logger.debug(
            f"{symbol} 从factor_output加载时间序列: {timeframe} - {len(factor_names)}个因子"
        )
        factor_data = factor_loader.load_factor_time_series(
            symbol=symbol, timeframe=timeframe, factor_names=factor_names
        )

        logger.info(
            f"{symbol} 因子时间序列加载成功 - 形状: {factor_data.shape}, 时间范围: {factor_data.index[0]} 到 {factor_data.index[-1]}"
        )
        logger.debug(
            f"{symbol} 价格数据时间范围: {close.index[0]} 到 {close.index[-1]}"
        )

        # 2. 对齐因子数据和价格数据的时间索引
        # 确保索引都是DatetimeIndex
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            factor_data.index = pd.to_datetime(factor_data.index)
        if not isinstance(close.index, pd.DatetimeIndex):
            close.index = pd.to_datetime(close.index)

        # 尝试对齐
        common_index = close.index.intersection(factor_data.index)
        logger.debug(f"{symbol} 初次对齐结果 - 重叠数据点: {len(common_index)}")

        # 如果没有重叠，尝试使用reindex进行前向填充
        if len(common_index) < 20:
            logger.warning(f"{symbol} 直接对齐失败，尝试使用reindex前向填充")
            factor_data_aligned = factor_data.reindex(close.index, method="ffill")
            # 移除NaN行
            valid_mask = factor_data_aligned.notna().any(axis=1)
            factor_data_aligned = factor_data_aligned[valid_mask]
            close_aligned = close[valid_mask]
            logger.debug(f"{symbol} reindex后有效数据点: {len(factor_data_aligned)}")

            if len(factor_data_aligned) < 20:
                raise ValueError(
                    f"对齐后的数据点仍然不足 ({len(factor_data_aligned)} < 20)"
                )
        else:
            factor_data_aligned = factor_data.loc[common_index]
            close_aligned = close.loc[common_index]

        logger.debug(f"{symbol} 数据对齐完成 - 有效数据点: {len(factor_data_aligned)}")

        # 3. 计算多因子复合得分
        # 向量化标准化：一次完成所有列的Z-score归一化
        factor_scores = factor_data_aligned.copy()
        
        # 向量化计算mean和std
        means = factor_scores.mean(axis=0)
        stds = factor_scores.std(axis=0, ddof=1)
        
        # 避免除零：std=0的列设为0
        stds = stds.replace(0, 1e-10)
        
        # 向量化标准化（广播）
        factor_scores = (factor_scores - means) / stds
        
        # 处理NaN（如果std=0导致的）
        factor_scores = factor_scores.fillna(0.0)
        
        # 计算复合得分（等权平均）
        composite_score = factor_scores.mean(axis=1)

        logger.debug(
            f"{symbol} 复合得分统计 - 最高: {composite_score.max():.4f}, 最低: {composite_score.min():.4f}, 均值: {composite_score.mean():.4f}"
        )

        # 4. 基于复合得分生成信号
        # 入场：复合得分 > 上四分位数
        # 出场：复合得分 < 下四分位数 或 时间止盈
        upper_threshold = composite_score.quantile(0.75)
        lower_threshold = composite_score.quantile(0.25)

        logger.debug(
            f"{symbol} 信号阈值 - 入场: {upper_threshold:.4f}, 出场: {lower_threshold:.4f}"
        )

        entries = (composite_score > upper_threshold).astype(bool)
        exits_score = (composite_score < lower_threshold).astype(bool)

        # 时间止盈
        time_exits = _compute_time_based_exits(entries, hold_days)
        exits = exits_score | time_exits

        # 对齐到原始close索引
        entries_full = pd.Series(False, index=close.index)
        exits_full = pd.Series(False, index=close.index)
        entries_full.loc[common_index] = entries
        exits_full.loc[common_index] = exits

        entry_count = entries_full.sum()
        exit_count = exits_full.sum()

        logger.info(
            f"{symbol} 多因子融合信号生成完成 - 入场信号: {entry_count}, 出场信号: {exit_count}"
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=_normalize_timeframe_label(timeframe),
            entries=entries_full,
            exits=exits_full,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    except FactorLoadError as e:
        logger.error(f"{symbol} 因子时间序列加载失败: {e}")
        raise
    except Exception as e:
        logger.error(f"{symbol} 多因子融合信号生成失败: {e}")
        raise


def generate_factor_signals(
    symbol: str,
    timeframe: str,
    close: pd.Series,
    volume: Optional[pd.Series],
    descriptor: FactorDescriptor,
    hold_days: int,
    stop_loss: float,
    take_profit: float,
) -> StrategySignals:
    """Translate a factor descriptor into deterministic entry/exit signals.

    为保持研究与回测一致性，所有指标值一律来自统一的FactorEngine，
    此处仅负责根据因子序列生成交易信号，不再重复实现指标计算。
    """

    normalized_tf = _normalize_timeframe_label(timeframe)
    factor_id = descriptor.name

    # 验证因子ID格式
    if not factor_id.replace('_', '').isalnum():
        raise ValueError(f"因子ID格式无效: {factor_id}")

    engine = api.get_engine()

    # 验证因子是否已注册
    available_factors = engine.registry.list_factors()
    if factor_id not in available_factors:
        error_msg = (
            f"❌ 未注册的因子: '{factor_id}'\n\n"
            f"为确保回测与研究一致性，禁止使用回退策略。\n\n"
            f"📋 可用因子列表 ({len(available_factors)}个):\n"
            f"   {', '.join(sorted(available_factors))}\n\n"
            f"🔧 解决方案:\n"
            f"   1. 使用上述已注册的标准因子名\n"
            f"   2. 或在FactorEngine中实现并注册该因子\n"
            f"   3. 检查因子配置是否使用了正确的标准格式"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    factors = engine.calculate_factors(
        factor_ids=[factor_id],
        symbols=[symbol],
        timeframe=normalized_tf,
        start_date=close.index.min(),
        end_date=close.index.max(),
        use_cache=True,
    )

    if factors.empty:
        raise ValueError(f"因子 {factor_id} 计算结果为空，无法生成信号")

    if isinstance(factors.index, pd.MultiIndex):
        factor_df = factors.xs(symbol, level="symbol")
    else:
        factor_df = factors

    # 严格验证列名 - 必须完全匹配
    if factor_id not in factor_df.columns:
        available_columns = list(factor_df.columns)
        raise KeyError(
            f"因子列 '{factor_id}' 不存在。\n"
            f"可用列: {available_columns}\n"
            f"请确保因子计算返回的列名与请求的factor_id完全一致"
        )

    factor_series = factor_df[factor_id].reindex(close.index).ffill()

    # 检查因子数据有效性
    if factor_series.isna().all():
        raise ValueError(f"因子 {factor_id} 全部为NaN，无法生成信号")

    # 检查有效数据点数量
    valid_values = factor_series.dropna()
    if len(valid_values) < 10:  # 至少需要10个有效值点
        raise ValueError(f"因子 {factor_id} 有效数据点不足 ({len(valid_values)})")

    # 因子标准化后基于分位数生成信号
    normalized = (factor_series - factor_series.mean()) / factor_series.std(ddof=0)
    normalized = normalized.fillna(0.0)

    upper_threshold = normalized.quantile(0.75)
    lower_threshold = normalized.quantile(0.25)

    entries = (normalized > upper_threshold).fillna(False)
    exits_score = (normalized < lower_threshold).fillna(False)
    exits_time = _compute_time_based_exits(entries, hold_days)
    exits = exits_score | exits_time

    logger.info(
        f"生成信号完成 - {symbol} {factor_id}: "
        f"入场{entries.sum()}次, 出场{exits.sum()}次"
    )

    return StrategySignals(
        symbol=symbol,
        timeframe=normalized_tf,
        entries=entries.astype(bool),
        exits=exits.astype(bool),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


class StrategyCore:
    """High-level orchestrator for candidate selection and signal generation."""

    def __init__(
        self,
        trading_config: TradingConfig = DEFAULT_TRADING_CONFIG,
        execution_config: ExecutionConfig = DEFAULT_EXECUTION_CONFIG,
        runtime_config: StrategyRuntimeConfig = DEFAULT_RUNTIME_CONFIG,
        factor_loader: Optional[FactorScoreLoader] = None,
        fusion_engine: Optional[FactorFusionEngine] = None,
    ) -> None:
        self.trading_config = trading_config
        self.execution_config = execution_config
        self.runtime_config = runtime_config
        self.factor_loader = factor_loader or FactorScoreLoader(runtime_config)
        self.fusion_engine = fusion_engine or FactorFusionEngine(runtime_config)
        self._last_factor_panel: Optional[pd.DataFrame] = None
        self._last_fused_scores: Optional[pd.DataFrame] = None

    def select_candidates(
        self,
        universe: Iterable[str],
        timeframe: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Select candidate symbols based on aggregated factor scores."""

        symbols = list(universe)
        logger.info(f"开始候选股票筛选 - 输入股票池: {len(symbols)}个标的")

        if not symbols:
            logger.warning("股票池为空，无法进行候选筛选")
            return []

        timeframes: Sequence[str] | None
        if timeframe is not None:
            timeframes = [timeframe]
        else:
            timeframes = self.runtime_config.fusion.ordered_timeframes()

        logger.info(f"使用时间框架: {timeframes}")
        max_factors = self.trading_config.max_positions * 2
        logger.debug(f"最大因子数量: {max_factors}")

        try:
            logger.debug("开始加载因子面板数据...")
            panel = self.factor_loader.load_factor_panels(
                symbols=symbols,
                timeframes=timeframes,
                max_factors=max_factors,
            )
            logger.info(f"因子面板加载完成 - 形状: {panel.shape}")
        except FileNotFoundError as e:
            logger.error(f"因子数据文件未找到: {e}")
            return []
        except Exception as e:
            logger.error(f"因子面板加载失败: {e}")
            return []

        self._last_factor_panel = panel
        if panel.empty:
            logger.warning("因子面板为空，无法进行筛选")
            return []

        logger.debug("开始因子融合...")
        fused = self.fusion_engine.fuse(panel)
        self._last_fused_scores = fused

        if fused.empty or "composite_score" not in fused.columns:
            logger.warning("因子融合失败或缺少composite_score列")
            return []

        composite = fused["composite_score"].dropna()
        if composite.empty:
            logger.warning("复合评分为空，无有效候选股票")
            return []

        logger.info(f"有效复合评分数量: {len(composite)}")
        logger.debug(
            f"复合评分统计: 最高={composite.max():.4f}, 最低={composite.min():.4f}, 均值={composite.mean():.4f}"
        )

        limit = top_n or self.trading_config.max_positions
        selected = composite.sort_values(ascending=False).head(limit).index.tolist()

        logger.info(f"候选筛选完成 - 选中 {len(selected)} 个标的: {selected}")

        # 记录每个选中标的的评分
        for i, symbol in enumerate(selected):
            score = composite.loc[symbol]
            logger.debug(f"排名 {i+1}: {symbol} = {score:.4f}")

        return selected

    def _passes_trend_filter(self, frames: Mapping[str, pd.DataFrame]) -> bool:
        trend_tf = self.runtime_config.fusion.trend_timeframe
        logger.debug(f"趋势过滤器检查 - 时间框架: {trend_tf}")

        if not trend_tf or trend_tf not in frames:
            logger.debug("趋势时间框架未配置或数据缺失，跳过趋势过滤")
            return True

        data = frames[trend_tf]
        if "close" not in data:
            logger.debug("趋势数据缺少close列，跳过趋势过滤")
            return True

        close = data["close"].dropna()
        if close.empty:
            logger.debug("趋势数据close列为空，跳过趋势过滤")
            return True

        window = max(self.runtime_config.trend_ma_window, 1)
        if len(close) < window:
            logger.debug(f"趋势数据长度不足 ({len(close)} < {window})，跳过趋势过滤")
            return True

        moving_average = close.rolling(window=window).mean()
        current_price = close.iloc[-1]
        current_ma = moving_average.iloc[-1]

        trend_up = current_price >= current_ma
        logger.debug(
            f"趋势过滤结果: 当前价格={current_price:.4f}, MA({window})={current_ma:.4f}, 趋势向上={trend_up}"
        )

        return bool(trend_up)

    def _passes_confirmation_filter(self, frames: Mapping[str, pd.DataFrame]) -> bool:
        confirmation_tf = self.runtime_config.fusion.confirmation_timeframe
        logger.debug(f"确认过滤器检查 - 时间框架: {confirmation_tf}")

        if not confirmation_tf or confirmation_tf not in frames:
            logger.debug("确认时间框架未配置或数据缺失，跳过确认过滤")
            return True

        data = frames[confirmation_tf]
        if "close" not in data:
            logger.debug("确认数据缺少close列，跳过确认过滤")
            return True

        close = data["close"].dropna()
        if close.empty:
            logger.debug("确认数据close列为空，跳过确认过滤")
            return True

        window = max(self.runtime_config.confirmation_ma_window, 1)
        if len(close) < window:
            logger.debug(f"确认数据长度不足 ({len(close)} < {window})，跳过确认过滤")
            return True

        moving_average = close.rolling(window=window).mean()
        current_price = close.iloc[-1]
        current_ma = moving_average.iloc[-1]

        confirmation_up = current_price >= current_ma
        logger.debug(
            f"确认过滤结果: 当前价格={current_price:.4f}, MA({window})={current_ma:.4f}, 确认向上={confirmation_up}"
        )

        return bool(confirmation_up)

    def _select_entry_timeframe(
        self, frames: Mapping[str, pd.DataFrame]
    ) -> Optional[str]:
        for timeframe in self.runtime_config.fusion.intraday_timeframes:
            if timeframe in frames:
                return timeframe
        if self.runtime_config.default_timeframe in frames:
            return self.runtime_config.default_timeframe
        return next(iter(frames.keys()), None)

    def _resolve_top_factor(
        self, symbol: str, timeframe: str
    ) -> Optional[FactorDescriptor]:
        if self._last_factor_panel is None or self._last_factor_panel.empty:
            return None

        normalized_tf = _normalize_timeframe_label(timeframe)
        panel = self._last_factor_panel.reset_index()
        panel["normalized_timeframe"] = panel["timeframe"].map(
            lambda tf: _normalize_timeframe_label(str(tf))
        )

        subset = panel[
            (panel["symbol"] == symbol)
            & (panel["normalized_timeframe"] == normalized_tf)
        ]
        if subset.empty:
            return None

        ordered = subset.sort_values(by="rank")
        best = ordered.iloc[0]
        metadata = {
            key: best[key]
            for key in ordered.columns
            if key
            not in {
                "symbol",
                "timeframe",
                "factor_name",
                "normalized_timeframe",
            }
        }
        return FactorDescriptor(
            name=str(best["factor_name"]),
            timeframe=normalized_tf,
            metadata=metadata,
        )

    def generate_signals_for_symbol(
        self,
        symbol: str,
        frames: Mapping[str, pd.DataFrame],
    ) -> Optional[StrategySignals]:
        """Create strategy signals for a single symbol using multi-timeframe data."""

        logger.info(f"开始为 {symbol} 生成交易信号")
        logger.debug(f"可用时间框架: {list(frames.keys())}")

        # 趋势过滤
        if not self._passes_trend_filter(frames):
            logger.info(f"{symbol} 未通过趋势过滤器，跳过信号生成")
            return None

        # 确认过滤
        if not self._passes_confirmation_filter(frames):
            logger.info(f"{symbol} 未通过确认过滤器，跳过信号生成")
            return None

        # 选择入场时间框架
        entry_timeframe = self._select_entry_timeframe(frames)
        if entry_timeframe is None:
            logger.warning(f"{symbol} 无可用的入场时间框架")
            return None

        logger.info(f"{symbol} 选择入场时间框架: {entry_timeframe}")

        data = frames[entry_timeframe]
        if "close" not in data or "volume" not in data:
            logger.warning(f"{symbol} 入场时间框架数据缺少close或volume列")
            return None

        logger.debug(
            f"{symbol} 入场数据形状: {data.shape}, 时间范围: {data.index[0]} 到 {data.index[-1]}"
        )

        # 1. 从factor_ready获取优秀因子列表
        if self._last_factor_panel is None or self._last_factor_panel.empty:
            logger.warning(f"{symbol} 因子面板为空，无法使用多因子融合")
            return None

        # 获取该symbol和timeframe的所有优秀因子
        normalized_tf = _normalize_timeframe_label(entry_timeframe)
        panel = self._last_factor_panel.reset_index()
        panel["normalized_timeframe"] = panel["timeframe"].map(
            lambda tf: _normalize_timeframe_label(str(tf))
        )

        subset = panel[
            (panel["symbol"] == symbol)
            & (panel["normalized_timeframe"] == normalized_tf)
        ]

        if subset.empty:
            logger.warning(f"{symbol} 在 {entry_timeframe} 时间框架下没有优秀因子")
            return None

        # 按rank排序，选择top 50个因子
        top_factors = subset.sort_values(by="rank").head(50)
        factor_names = top_factors["factor_name"].tolist()

        logger.info(f"{symbol} 从factor_ready筛选出 {len(factor_names)} 个优秀因子")
        logger.debug(f"{symbol} 前10个因子: {factor_names[:10]}")

        # 2. 使用多因子融合算法生成信号
        try:
            logger.info(
                f"{symbol} 使用多因子融合算法生成信号 - 因子数量: {len(factor_names)}"
            )
            signals = generate_multi_factor_signals(
                    symbol=symbol,
                    timeframe=entry_timeframe,
                    close=data["close"],
                    volume=data.get("volume"),
                factor_names=factor_names,
                factor_loader=self.factor_loader,
                    hold_days=self.trading_config.hold_days,
                    stop_loss=self.execution_config.stop_loss,
                    take_profit=self.execution_config.primary_take_profit(),
                )
            logger.info(
                f"{symbol} 多因子融合信号生成成功 - 入场信号数: {signals.entries.sum()}"
            )
            return signals
        except FactorLoadError as e:
            logger.error(f"{symbol} 因子时间序列加载失败: {e}")
            logger.error(f"严格模式：禁止降级到传统逻辑")
            return None
        except Exception as e:
            logger.error(f"{symbol} 多因子融合信号生成失败: {e}")
            logger.error(f"严格模式：禁止降级到传统逻辑")
            return None

    def build_signal_universe(
        self,
        price_data: Mapping[str, Mapping[str, pd.DataFrame] | pd.DataFrame],
        timeframe: Optional[str] = None,
    ) -> Dict[str, StrategySignals]:
        """Generate signals for the selected candidate universe."""

        if not price_data:
            return {}

        expanded: Dict[str, Mapping[str, pd.DataFrame]] = {}
        for symbol, data in price_data.items():
            if isinstance(data, Mapping):
                expanded[symbol] = data
            else:
                expanded[symbol] = {self.runtime_config.default_timeframe: data}

        universe = list(expanded.keys())
        candidates = self.select_candidates(universe, timeframe=timeframe)
        if not candidates:
            fallback_signals: Dict[str, StrategySignals] = {}
            for symbol, frames in expanded.items():
                signal = self.generate_signals_for_symbol(symbol=symbol, frames=frames)
                if signal is not None:
                    fallback_signals[symbol] = signal
            return fallback_signals

        signals: Dict[str, StrategySignals] = {}
        for symbol in candidates:
            frames = expanded.get(symbol)
            if not frames:
                continue
            signal = self.generate_signals_for_symbol(symbol=symbol, frames=frames)
            if signal is not None:
                signals[symbol] = signal
        return signals


__all__ = [
    "FactorDescriptor",
    "StrategyCore",
    "StrategySignals",
    "generate_factor_signals",
    "hk_reversal_logic",
]
