"""
市场状态检测器 | Market Regime Detector

自动识别市场状态，为自适应WFO提供决策依据

市场状态定义：
- BULL: 牛市 (上涨趋势明显)
- BEAR: 熊市 (下跌趋势明显)
- SIDEWAYS: 震荡市 (无明显趋势)
- HIGH_VOL: 高波动 (异常波动)
- LOW_VOL: 低波动 (平静期)

作者: AI Agent
日期: 2025-10-29
"""

import logging
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""

    BULL = "牛市"
    BEAR = "熊市"
    SIDEWAYS = "震荡市"
    HIGH_VOL = "高波动"
    LOW_VOL = "低波动"


class RegimeSignal(NamedTuple):
    """市场状态信号"""

    regime: MarketRegime
    confidence: float  # 置信度 0-1
    duration: int  # 持续天数
    features: Dict[str, float]  # 特征值


class MarketRegimeDetector:
    """市场状态检测器"""

    def __init__(
        self,
        lookback_period: int = 60,  # 回看期
        volatility_window: int = 20,  # 波动率计算窗口
        trend_window: int = 30,  # 趋势计算窗口
        confidence_threshold: float = 0.6,  # 置信度门槛
    ):
        """
        初始化市场状态检测器

        Args:
            lookback_period: 回看期天数
            volatility_window: 波动率计算窗口
            trend_window: 趋势计算窗口
            confidence_threshold: 置信度门槛
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.confidence_threshold = confidence_threshold

        # 状态历史记录
        self.regime_history: List[RegimeSignal] = []

        logger.info(
            f"MarketRegimeDetector初始化: lookback={lookback_period}, "
            f"vol_window={volatility_window}, trend_window={trend_window}"
        )

    def detect_regime(
        self, returns: pd.DataFrame, prices: pd.DataFrame
    ) -> RegimeSignal:
        """
        检测当前市场状态

        Args:
            returns: 收益率数据 (日期 × 标的)
            prices: 价格数据 (日期 × 标的)

        Returns:
            RegimeSignal: 市场状态信号
        """
        # 确保有足够数据
        if len(returns) < self.lookback_period:
            logger.warning(f"数据不足: {len(returns)} < {self.lookback_period}")
            return self._create_default_signal()

        # 计算市场特征
        market_returns = returns.mean(axis=1)  # 市场平均收益
        features = self._calculate_features(market_returns, prices)

        # 基于特征判断市场状态
        regime, confidence = self._classify_regime(features)

        # 计算状态持续时间
        duration = self._calculate_regime_duration(regime)

        # 创建信号
        signal = RegimeSignal(
            regime=regime, confidence=confidence, duration=duration, features=features
        )

        # 记录历史
        self.regime_history.append(signal)
        if len(self.regime_history) > 252:  # 保留1年历史
            self.regime_history.pop(0)

        logger.info(
            f"市场状态检测: {regime.value}, 置信度={confidence:.2f}, "
            f"持续天数={duration}"
        )

        return signal

    def _calculate_features(
        self, market_returns: pd.Series, prices: pd.DataFrame
    ) -> Dict[str, float]:
        """计算市场特征"""
        # 使用最近的数据
        recent_returns = market_returns.tail(self.lookback_period)
        recent_prices = prices.tail(self.lookback_period)

        # 1. 波动率特征
        volatility = recent_returns.rolling(self.volatility_window).std().iloc[-1]
        volatility_percentile = self._calculate_volatility_percentile(volatility)

        # 2. 趋势特征
        trend_strength = self._calculate_trend_strength(recent_prices)
        return_trend = recent_returns.rolling(self.trend_window).mean().iloc[-1]

        # 3. 动量特征
        momentum_20d = (recent_prices.iloc[-1] / recent_prices.iloc[-20] - 1).mean()
        momentum_60d = (recent_prices.iloc[-1] / recent_prices.iloc[-60] - 1).mean()

        # 4. 回撤特征
        max_drawdown = self._calculate_max_drawdown(recent_prices)
        current_drawdown = self._calculate_current_drawdown(recent_prices)

        # 5. 连涨连跌特征
        consecutive_up = self._calculate_consecutive_days(recent_returns > 0)
        consecutive_down = self._calculate_consecutive_days(recent_returns < 0)

        features = {
            "volatility": float(volatility),
            "volatility_percentile": float(volatility_percentile),
            "trend_strength": float(trend_strength),
            "return_trend": float(return_trend),
            "momentum_20d": float(momentum_20d),
            "momentum_60d": float(momentum_60d),
            "max_drawdown": float(max_drawdown),
            "current_drawdown": float(current_drawdown),
            "consecutive_up": int(consecutive_up),
            "consecutive_down": int(consecutive_down),
        }

        return features

    def _classify_regime(
        self, features: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """基于特征分类市场状态"""
        vol = features["volatility_percentile"]
        trend = features["trend_strength"]
        return_trend = features["return_trend"]
        momentum = features["momentum_20d"]
        drawdown = features["current_drawdown"]

        # 规则引擎
        scores = {
            MarketRegime.HIGH_VOL: 0.0,
            MarketRegime.LOW_VOL: 0.0,
            MarketRegime.BULL: 0.0,
            MarketRegime.BEAR: 0.0,
            MarketRegime.SIDEWAYS: 0.0,
        }

        # 高波动判断
        if vol > 0.8:
            scores[MarketRegime.HIGH_VOL] += 0.4
        if vol > 0.6:
            scores[MarketRegime.HIGH_VOL] += 0.2

        # 低波动判断
        if vol < 0.2:
            scores[MarketRegime.LOW_VOL] += 0.4
        if vol < 0.3:
            scores[MarketRegime.LOW_VOL] += 0.2

        # 牛市判断
        if trend > 0.6:
            scores[MarketRegime.BULL] += 0.3
        if return_trend > 0.001:
            scores[MarketRegime.BULL] += 0.3
        if momentum > 0.02:
            scores[MarketRegime.BULL] += 0.2
        if drawdown > -0.05:
            scores[MarketRegime.BULL] += 0.2

        # 熊市判断
        if trend < -0.6:
            scores[MarketRegime.BEAR] += 0.3
        if return_trend < -0.001:
            scores[MarketRegime.BEAR] += 0.3
        if momentum < -0.02:
            scores[MarketRegime.BEAR] += 0.2
        if drawdown < -0.10:
            scores[MarketRegime.BEAR] += 0.2

        # 震荡市判断
        if abs(trend) < 0.3:
            scores[MarketRegime.SIDEWAYS] += 0.3
        if abs(return_trend) < 0.0005:
            scores[MarketRegime.SIDEWAYS] += 0.3
        if abs(momentum) < 0.01:
            scores[MarketRegime.SIDEWAYS] += 0.2
        if 0.2 < vol < 0.8:
            scores[MarketRegime.SIDEWAYS] += 0.2

        # 找到得分最高的状态
        best_regime = max(scores.items(), key=lambda x: x[1])
        regime = best_regime[0]
        confidence = best_regime[1]

        # 如果所有得分都很低，默认为震荡市
        if confidence < 0.3:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.5

        return regime, min(confidence, 1.0)

    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """计算当前波动率在历史中的百分位"""
        if len(self.regime_history) < 30:
            return 0.5  # 数据不足，返回中位数

        historical_vols = [
            sig.features["volatility"] for sig in self.regime_history[-100:]
        ]
        percentile = np.sum(np.array(historical_vols) <= current_vol) / len(
            historical_vols
        )
        return percentile

    def _calculate_trend_strength(self, prices: pd.DataFrame) -> float:
        """计算趋势强度"""
        # 使用R²衡量趋势强度
        market_index = prices.mean(axis=1)
        x = np.arange(len(market_index))

        # 计算线性回归
        coeffs = np.polyfit(x, market_index, 1)
        trend_line = np.polyval(coeffs, x)

        # 计算R²
        ss_res = np.sum((market_index - trend_line) ** 2)
        ss_tot = np.sum((market_index - np.mean(market_index)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 归一化到[-1, 1]，正值表示上升趋势
        trend_strength = np.sign(coeffs[0]) * r_squared
        return trend_strength

    def _calculate_max_drawdown(self, prices: pd.DataFrame) -> float:
        """计算最大回撤"""
        market_index = prices.mean(axis=1)
        cumulative = (1 + market_index).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_current_drawdown(self, prices: pd.DataFrame) -> float:
        """计算当前回撤"""
        market_index = prices.mean(axis=1)
        cumulative = (1 + market_index).cumprod()
        running_max = cumulative.expanding().max()
        current_dd = (cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        return current_dd

    def _calculate_consecutive_days(self, condition: pd.Series) -> int:
        """计算连续满足条件的天数"""
        if len(condition) == 0:
            return 0

        consecutive = 0
        for val in reversed(condition):
            if val:
                consecutive += 1
            else:
                break
        return consecutive

    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """计算当前状态的持续时间"""
        if not self.regime_history:
            return 0

        duration = 0
        for signal in reversed(self.regime_history):
            if signal.regime == current_regime:
                duration += 1
            else:
                break
        return duration

    def _create_default_signal(self) -> RegimeSignal:
        """创建默认信号（数据不足时使用）"""
        return RegimeSignal(
            regime=MarketRegime.SIDEWAYS, confidence=0.5, duration=0, features={}
        )

    def get_regime_statistics(
        self, days: int = 252
    ) -> Dict[MarketRegime, Dict[str, float]]:
        """获取市场状态统计"""
        recent_signals = (
            self.regime_history[-days:]
            if len(self.regime_history) >= days
            else self.regime_history
        )

        if not recent_signals:
            return {}

        stats = {}
        for regime in MarketRegime:
            regime_signals = [s for s in recent_signals if s.regime == regime]
            if regime_signals:
                count = len(regime_signals)
                avg_confidence = np.mean([s.confidence for s in regime_signals])
                avg_duration = np.mean([s.duration for s in regime_signals])

                stats[regime] = {
                    "count": count,
                    "frequency": count / len(recent_signals),
                    "avg_confidence": avg_confidence,
                    "avg_duration": avg_duration,
                }

        return stats

    def is_transition_period(self) -> bool:
        """判断是否处于状态转换期"""
        if len(self.regime_history) < 3:
            return False

        # 检查最近3次信号是否频繁变化
        recent_regimes = [s.regime for s in self.regime_history[-3:]]
        unique_regimes = len(set(recent_regimes))

        # 如果3天内状态变化超过2次，认为是转换期
        return unique_regimes > 2
