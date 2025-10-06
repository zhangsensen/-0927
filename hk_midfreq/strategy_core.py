"""Strategy logic and candidate selection for the HK mid-frequency stack."""

from __future__ import annotations

import logging
import re
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
from hk_midfreq.factor_interface import FactorScoreLoader
from hk_midfreq.fusion import FactorFusionEngine

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

    rsi = vbt.RSI.run(close, window=rsi_window).rsi
    bb = vbt.BBANDS.run(close, window=bb_window)
    rolling_volume = aligned_volume.rolling(
        window=volume_window, min_periods=volume_window
    ).mean()

    cond_rsi = rsi < rsi_threshold
    cond_bb = close <= bb.lower
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

    rsi = vbt.RSI.run(close, window=timeperiod).rsi
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
    """Translate a factor descriptor into deterministic entry/exit signals."""

    normalized_tf = _normalize_timeframe_label(timeframe)
    factor_name = descriptor.name

    # StochRSI 因子
    if factor_name.startswith("TA_STOCHRSI"):
        timeperiod, fastk, fastd, line = _parse_stochrsi_params(factor_name)
        k_series, d_series = _compute_stochrsi(close, timeperiod, fastk, fastd)
        target = k_series if line == "k" else d_series

        oversold = 20.0
        overbought = 80.0
        crosses_up = (target.shift(1) <= oversold) & (target > oversold)
        crosses_down = (target.shift(1) >= overbought) & (target < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # RSI 因子
    elif factor_name.startswith("RSI") or factor_name.startswith("TA_RSI"):
        # 提取RSI周期参数
        period = 14  # 默认值
        if "RSI" in factor_name:
            import re

            match = re.search(r"RSI(\d+)", factor_name)
            if match:
                period = int(match.group(1))

        rsi = vbt.RSI.run(close, window=period).rsi
        oversold = 30.0
        overbought = 70.0

        crosses_up = (rsi.shift(1) <= oversold) & (rsi > oversold)
        crosses_down = (rsi.shift(1) >= overbought) & (rsi < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # STOCH 随机指标
    elif factor_name.startswith("STOCH"):
        # 提取参数
        k_period = 14
        d_period = 3
        import re

        # STOCH_14_5 格式
        match = re.search(r"STOCH_(\d+)_(\d+)", factor_name)
        if match:
            k_period = int(match.group(1))
            d_period = int(match.group(2))

        stoch = vbt.STOCH.run(close, close, close, k_period, d_period)
        target = stoch.percent_k  # 使用%K线

        oversold = 20.0
        overbought = 80.0
        crosses_up = (target.shift(1) <= oversold) & (target > oversold)
        crosses_down = (target.shift(1) >= overbought) & (target < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # Williams %R 指标
    elif factor_name.startswith("WILLR") or factor_name.startswith("TA_WILLR"):
        # 提取周期参数
        period = 14
        import re

        match = re.search(r"WILLR(\d+)", factor_name)
        if match:
            period = int(match.group(1))

        # 手动计算Williams %R
        high_roll = close.rolling(window=period).max()
        low_roll = close.rolling(window=period).min()
        willr = -100 * (high_roll - close) / (high_roll - low_roll)
        willr = willr.fillna(0)

        oversold = -80.0
        overbought = -20.0

        crosses_up = (willr.shift(1) <= oversold) & (willr > oversold)
        crosses_down = (willr.shift(1) >= overbought) & (willr < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # STOCHF 快速随机指标
    elif factor_name.startswith("TA_STOCHF"):
        # 使用标准STOCH计算，但参数更敏感
        k_period = 14
        d_period = 3

        stoch = vbt.STOCH.run(close, close, close, k_period, d_period)
        target = stoch.percent_k  # 使用%K线

        oversold = 20.0
        overbought = 80.0
        crosses_up = (target.shift(1) <= oversold) & (target > oversold)
        crosses_down = (target.shift(1) >= overbought) & (target < overbought)
        entries = crosses_up.fillna(False)
        exits = crosses_down.fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # 蜡烛图形态因子 - 使用简化的反转逻辑
    elif factor_name.startswith("TA_CDL") or factor_name.startswith("CDL"):
        # 对于蜡烛图形态，使用价格动量作为代理
        returns = close.pct_change()
        volatility = returns.rolling(20).std()

        # 寻找反转信号：大幅下跌后的反弹
        big_drop = returns < -2 * volatility
        recovery = returns.shift(-1) > 0  # 下一期反弹
        entries = (big_drop & recovery).fillna(False)
        exits = _compute_time_based_exits(entries, hold_days)

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # 均值回归因子 (MEANLB等)
    elif "MEAN" in factor_name or "LB" in factor_name:
        # 使用布林带均值回归策略
        bb = vbt.BBANDS.run(close, window=20, alpha=2.0)

        # 价格触及下轨时买入
        entries = (close <= bb.lower).fillna(False)
        # 价格回到中轨时卖出
        exits = (close >= bb.middle).fillna(False) | _compute_time_based_exits(
            entries, hold_days
        )

        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=entries.astype(bool),
            exits=exits.astype(bool),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # 通用回退策略 - 使用传统反转逻辑
    else:
        print(f"⚠️  因子 {factor_name} 暂不支持，使用通用反转策略")
        fallback_volume = (
            volume if volume is not None else pd.Series(1000000, index=close.index)
        )
        fallback_signals = hk_reversal_logic(
            close=close,
            volume=fallback_volume,
            hold_days=hold_days,
        )

        # 手动创建新的StrategySignals对象
        return StrategySignals(
            symbol=symbol,
            timeframe=normalized_tf,
            entries=fallback_signals.entries,
            exits=fallback_signals.exits,
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

        # 尝试基于因子的信号生成
        descriptor = self._resolve_top_factor(symbol, entry_timeframe)
        if descriptor is not None:
            factor_name = getattr(descriptor, "name", "unknown")
            logger.info(f"{symbol} 使用因子信号生成 - 因子: {factor_name}")
            try:
                signals = generate_factor_signals(
                    symbol=symbol,
                    timeframe=entry_timeframe,
                    close=data["close"],
                    volume=data.get("volume"),
                    descriptor=descriptor,
                    hold_days=self.trading_config.hold_days,
                    stop_loss=self.execution_config.stop_loss,
                    take_profit=self.execution_config.primary_take_profit(),
                )
                logger.info(
                    f"{symbol} 因子信号生成成功 - 入场信号数: {signals.entries.sum()}"
                )
                return signals
            except Exception as e:
                logger.warning(f"{symbol} 因子信号生成失败: {e}，回退到传统逻辑")

        # 回退到传统反转逻辑
        logger.info(f"{symbol} 使用传统反转逻辑生成信号")
        signal_bundle = hk_reversal_logic(
            close=data["close"],
            volume=data["volume"],
            hold_days=self.trading_config.hold_days,
        )

        signals = StrategySignals(
            symbol=symbol,
            timeframe=_normalize_timeframe_label(entry_timeframe),
            entries=signal_bundle.entries,
            exits=signal_bundle.exits,
            stop_loss=self.execution_config.stop_loss,
            take_profit=self.execution_config.primary_take_profit(),
        )

        logger.info(
            f"{symbol} 传统信号生成完成 - 入场信号数: {signals.entries.sum()}, 出场信号数: {signals.exits.sum()}"
        )
        logger.debug(
            f"{symbol} 风险参数 - 止损: {signals.stop_loss:.2%}, 止盈: {signals.take_profit:.2%}"
        )

        return signals

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
