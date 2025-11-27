"""
精确因子库 v2 | Precise Factor Library v2
================================================================================
根据CANDIDATE_FACTORS_PRECISE_DEFINITION.md精确定义实现的因子库

核心设计原则：
1. 严格遵循精确定义：公式、缺失处理、极值规则
2. 缺失值处理：原始缺失→保留NaN；满窗不足→NaN（无向前填充）
3. 标准化位置：WFO内完成（不在生成阶段）
4. 极值截断：2.5%/97.5%分位（有界因子跳过）
5. 避免冗余：12-15个精选因子，遵循互斥规则

【首批精选因子】
维度 1 - 趋势/动量 (2个):
  ✓ MOM_20D          - 20日动量百分比
  ✓ SLOPE_20D        - 20日线性回归斜率

维度 2 - 价格位置 (2个):
  ✓ PRICE_POSITION_20D   - 20日价格位置（有界）
  ✓ PRICE_POSITION_120D  - 120日价格位置（有界）

维度 3 - 波动率 (2个):
  ✓ RET_VOL_20D      - 20日收益波动率
  ✓ MAX_DD_60D       - 60日最大回撤

维度 4 - 成交量 (2个):
  ✓ VOL_RATIO_20D    - 20日成交量比率
  ✓ VOL_RATIO_60D    - 60日成交量比率（中期）

维度 5 - 价量耦合 (1个):
  ✓ PV_CORR_20D      - 20日价量相关性

维度 6 - 反转 (1个):
  ✓ RSI_14           - 14日相对强度指数

=================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


# ============================================================================
# Numba加速函数（模块级定义）
# ============================================================================


@njit
def _rolling_max_dd_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滑窗最大回撤计算

    参数:
        prices: 1D价格序列
        window: 窗口长度

    返回:
        1D最大回撤序列（百分比，绝对值）
    """
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_prices = prices[i - window + 1 : i + 1]

        # 检查NaN
        if np.any(np.isnan(window_prices)):
            result[i] = np.nan
            continue

        # 计算最大回撤
        cummax = window_prices[0]
        max_dd = 0.0

        for j in range(1, window):
            if window_prices[j] > cummax:
                cummax = window_prices[j]
            dd = (window_prices[j] - cummax) / cummax
            if dd < max_dd:
                max_dd = dd

        result[i] = abs(max_dd) * 100.0  # 百分比

    return result


@njit
def _rolling_calmar_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滑窗卡玛比率计算

    参数:
        prices: 1D价格序列
        window: 窗口长度(60)

    返回:
        1D卡玛比率序列
    """
    n = len(prices)
    result = np.full(n, np.nan)
    eps = 1e-10

    for i in range(window - 1, n):
        window_prices = prices[i - window + 1 : i + 1]

        # 检查NaN
        if np.any(np.isnan(window_prices)):
            result[i] = np.nan
            continue

        # 累计收益
        cum_ret = (window_prices[-1] / window_prices[0]) - 1.0

        # 计算最大回撤
        cummax = window_prices[0]
        max_dd = 0.0

        for j in range(1, window):
            if window_prices[j] > cummax:
                cummax = window_prices[j]
            dd = (window_prices[j] - cummax) / cummax
            if dd < max_dd:
                max_dd = dd

        # 卡玛比率
        if abs(max_dd) < eps:
            result[i] = np.nan
        else:
            result[i] = cum_ret / abs(max_dd)

    return result


# ============================================================================
# 因子类定义
# ============================================================================


@dataclass
class FactorMetadata:
    """因子元数据"""

    name: str
    description: str
    dimension: str
    required_columns: list
    window: int
    bounded: bool  # 是否为有界因子（跳过极值截断）
    direction: str  # 'high_is_good', 'low_is_good', 'neutral'


class PreciseFactorLibrary:
    """
    精确因子库 v2

    12个精选因子的实现，严格按CANDIDATE_FACTORS_PRECISE_DEFINITION.md规范

    使用流程：
    1. 创建库实例
    2. 调用compute_all_factors()传入价格数据
    3. 返回所有因子的DataFrame
    4. 在WFO内进行标准化和极值截断
    """

    def __init__(self):
        self.factors_metadata = self._build_metadata()

    def _build_metadata(self) -> Dict[str, FactorMetadata]:
        """构建因子元数据"""
        return {
            "MOM_20D": FactorMetadata(
                name="MOM_20D",
                description="20日动量百分比",
                dimension="趋势/动量",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "SLOPE_20D": FactorMetadata(
                name="SLOPE_20D",
                description="20日线性回归斜率",
                dimension="趋势/动量",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "PRICE_POSITION_20D": FactorMetadata(
                name="PRICE_POSITION_20D",
                description="20日价格位置",
                dimension="价格位置",
                required_columns=["close", "high", "low"],
                window=20,
                bounded=True,  # [0,1]有界
                direction="neutral",
            ),
            "PRICE_POSITION_120D": FactorMetadata(
                name="PRICE_POSITION_120D",
                description="120日价格位置",
                dimension="价格位置",
                required_columns=["close", "high", "low"],
                window=120,
                bounded=True,  # [0,1]有界
                direction="neutral",
            ),
            "RET_VOL_20D": FactorMetadata(
                name="RET_VOL_20D",
                description="20日收益波动率（日收益标准差）",
                dimension="波动/风险",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
            ),
            "MAX_DD_60D": FactorMetadata(
                name="MAX_DD_60D",
                description="60日最大回撤（绝对值）",
                dimension="波动/风险",
                required_columns=["close"],
                window=60,
                bounded=False,
                direction="low_is_good",
            ),
            "VOL_RATIO_20D": FactorMetadata(
                name="VOL_RATIO_20D",
                description="20日成交量比率（近期vs历史）",
                dimension="量能/流动性",
                required_columns=["volume"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "VOL_RATIO_60D": FactorMetadata(
                name="VOL_RATIO_60D",
                description="60日成交量比率（近期vs历史）",
                dimension="量能/流动性",
                required_columns=["volume"],
                window=60,
                bounded=False,
                direction="high_is_good",
            ),
            "PV_CORR_20D": FactorMetadata(
                name="PV_CORR_20D",
                description="20日价量相关性",
                dimension="价量耦合",
                required_columns=["close", "volume"],
                window=20,
                bounded=True,  # [-1,1]有界
                direction="high_is_good",
            ),
            "RSI_14": FactorMetadata(
                name="RSI_14",
                description="14日相对强度指数",
                dimension="反转/过热",
                required_columns=["close"],
                window=14,
                bounded=True,  # [0,100]有界
                direction="neutral",
            ),
            # ============ 第1批新增：资金流因子 ============
            "OBV_SLOPE_10D": FactorMetadata(
                name="OBV_SLOPE_10D",
                description="10日OBV能量潮斜率",
                dimension="资金流",
                required_columns=["close", "volume"],
                window=10,
                bounded=False,
                direction="high_is_good",
            ),
            "CMF_20D": FactorMetadata(
                name="CMF_20D",
                description="20日蔡金资金流",
                dimension="资金流",
                required_columns=["high", "low", "close", "volume"],
                window=20,
                bounded=True,  # [-1,1]有界
                direction="high_is_good",
            ),
            # ============ 第2批新增：风险调整动量 ============
            "SHARPE_RATIO_20D": FactorMetadata(
                name="SHARPE_RATIO_20D",
                description="20日夏普比率",
                dimension="风险调整动量",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "CALMAR_RATIO_60D": FactorMetadata(
                name="CALMAR_RATIO_60D",
                description="60日卡玛比率",
                dimension="风险调整动量",
                required_columns=["close"],
                window=60,
                bounded=False,
                direction="high_is_good",
            ),
            # ============ 第3批新增：趋势强度 ============
            "ADX_14D": FactorMetadata(
                name="ADX_14D",
                description="14日平均趋向指数",
                dimension="趋势强度",
                required_columns=["high", "low", "close"],
                window=14,
                bounded=True,  # [0,100]有界
                direction="high_is_good",
            ),
            "VORTEX_14D": FactorMetadata(
                name="VORTEX_14D",
                description="14日螺旋指标",
                dimension="趋势强度",
                required_columns=["high", "low", "close"],
                window=14,
                bounded=False,
                direction="neutral",
            ),
            # ============ 第4批新增：相对强度 ============
            "RELATIVE_STRENGTH_VS_MARKET_20D": FactorMetadata(
                name="RELATIVE_STRENGTH_VS_MARKET_20D",
                description="20日相对市场强度",
                dimension="相对强度",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "CORRELATION_TO_MARKET_20D": FactorMetadata(
                name="CORRELATION_TO_MARKET_20D",
                description="20日与市场相关性",
                dimension="相对强度",
                required_columns=["close"],
                window=20,
                bounded=True,  # [-1,1]有界
                direction="low_is_good",
            ),
            # ============ [P0修复] 禁用新增7个因子，回滚到历史18个 ============
            # "TSMOM_60D": FactorMetadata(
            #     name="TSMOM_60D",
            #     description="60日时间序列动量",
            #     dimension="趋势/动量",
            #     required_columns=["close"],
            #     window=60,
            #     bounded=False,
            #     direction="high_is_good",
            # ),
            # "TSMOM_120D": FactorMetadata(
            #     name="TSMOM_120D",
            #     description="120日时间序列动量",
            #     dimension="趋势/动量",
            #     required_columns=["close"],
            #     window=120,
            #     bounded=False,
            #     direction="high_is_good",
            # ),
            # "BREAKOUT_20D": FactorMetadata(
            #     name="BREAKOUT_20D",
            #     description="20日突破信号",
            #     dimension="趋势/动量",
            #     required_columns=["high", "close"],
            #     window=20,
            #     bounded=False,
            #     direction="high_is_good",
            # ),
            # "TURNOVER_ACCEL_5_20": FactorMetadata(
            #     name="TURNOVER_ACCEL_5_20",
            #     description="5日vs20日换手率加速度",
            #     dimension="量能/流动性",
            #     required_columns=["volume"],
            #     window=20,
            #     bounded=False,
            #     direction="high_is_good",
            # ),
            # "REALIZED_VOL_20D": FactorMetadata(
            #     name="REALIZED_VOL_20D",
            #     description="20日实际波动率",
            #     dimension="波动/风险",
            #     required_columns=["close"],
            #     window=20,
            #     bounded=False,
            #     direction="low_is_good",
            # ),
            # "AMIHUD_ILLIQUIDITY": FactorMetadata(
            #     name="AMIHUD_ILLIQUIDITY",
            #     description="Amihud流动性指标（冲击成本代理）",
            #     dimension="流动性/成本",
            #     required_columns=["close", "volume"],
            #     window=20,
            #     bounded=False,
            #     direction="low_is_good",  # 值越低越好（低冲击）
            # ),
            # "SPREAD_PROXY": FactorMetadata(
            #     name="SPREAD_PROXY",
            #     description="日内价差代理（交易成本）",
            #     dimension="流动性/成本",
            #     required_columns=["high", "low", "close"],
            #     window=5,
            #     bounded=False,
            #     direction="low_is_good",  # 价差越低越好
            # ),
        }

    # =========================================================================
    # 批量处理方法（DataFrame输入，零循环）
    # =========================================================================

    def _slope_20d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 SLOPE_20D（所有列一次性处理）"""
        from scipy.signal import lfilter

        x = np.arange(1, 21, dtype=np.float64)
        x_dev = x - x.mean()
        weights = x_dev[::-1]
        denom = (x_dev**2).sum()

        # 对整个 DataFrame 应用 lfilter（逐列）
        result = np.apply_along_axis(
            lambda col: lfilter(weights, [1.0], col) / denom,
            axis=0,
            arr=close_df.values,
        )
        result[:19, :] = np.nan
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _max_dd_60d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 MAX_DD_60D（所有列一次性处理）"""
        result = np.apply_along_axis(
            lambda col: _rolling_max_dd_numba(col, window=60),
            axis=0,
            arr=close_df.values,
        )
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _calmar_60d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 CALMAR_60D（所有列一次性处理）"""
        result = np.apply_along_axis(
            lambda col: _rolling_calmar_numba(col, window=60),
            axis=0,
            arr=close_df.values,
        )
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _obv_slope_10d_batch(
        self, close_df: pd.DataFrame, volume_df: pd.DataFrame
    ) -> pd.DataFrame:
        """批量计算 OBV_SLOPE_10D（所有列一次性处理）"""
        from scipy.signal import lfilter

        # 计算 OBV
        price_change = close_df.diff()
        sign = np.sign(price_change.values)
        sign[np.isnan(sign)] = 0  # 第一天NaN改为0（无方向）
        obv_vals = np.cumsum(sign * volume_df.values, axis=0)

        # 预计算权重
        x = np.arange(1, 11, dtype=np.float64)
        x_dev = x - x.mean()
        weights = x_dev[::-1]
        denom = (x_dev**2).sum()

        # 逐列 lfilter
        result = np.apply_along_axis(
            lambda col: lfilter(weights, [1.0], col) / denom, axis=0, arr=obv_vals
        )
        result[:9, :] = np.nan
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _price_position_batch(
        self,
        close_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """批量计算 PRICE_POSITION（所有列一次性处理）"""
        high_max = high_df.rolling(window=window, min_periods=window).max()
        low_min = low_df.rolling(window=window, min_periods=window).min()
        range_val = high_max - low_min
        position = (close_df - low_min) / range_val
        position = position.where(range_val > 1e-10, 0.5)
        return position.clip(0, 1)

    def _cmf_20d_batch(
        self,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量计算 CMF_20D（所有列一次性处理）"""
        mfm = ((close_df - low_df) - (high_df - close_df)) / (high_df - low_df + 1e-10)
        mfm = mfm.where(high_df != low_df, np.nan)
        mfv = mfm * volume_df
        cmf = mfv.rolling(window=20, min_periods=20).sum() / (
            volume_df.rolling(window=20, min_periods=20).sum() + 1e-10
        )
        return cmf

    def _adx_14d_batch(
        self, high_df: pd.DataFrame, low_df: pd.DataFrame, close_df: pd.DataFrame
    ) -> pd.DataFrame:
        """批量计算 ADX_14D（所有列一次性处理）"""
        high_diff = high_df.diff()
        low_diff = -low_df.diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        prev_close = close_df.shift(1)
        tr1 = high_df - low_df
        tr2 = (high_df - prev_close).abs()
        tr3 = (low_df - prev_close).abs()
        
        # 修复：使用 np.maximum 逐元素比较，保持 DataFrame 结构
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
        plus_di = 100 * (
            plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
        )
        minus_di = 100 * (
            minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
        )

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
        return adx

    def _vortex_14d_batch(
        self, high_df: pd.DataFrame, low_df: pd.DataFrame, close_df: pd.DataFrame
    ) -> pd.DataFrame:
        """批量计算 VORTEX_14D（所有列一次性处理）
        
        修复：正确计算 TR（逐列取 max，而非全局 concat 后 max）
        """
        vm_plus = (high_df - low_df.shift(1)).abs()
        vm_minus = (low_df - high_df.shift(1)).abs()

        prev_close = close_df.shift(1)
        tr1 = high_df - low_df
        tr2 = (high_df - prev_close).abs()
        tr3 = (low_df - prev_close).abs()
        
        # 修复：使用 np.maximum 逐元素比较，保持 DataFrame 结构
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        vm_plus_sum = vm_plus.rolling(window=14, min_periods=14).sum()
        vm_minus_sum = vm_minus.rolling(window=14, min_periods=14).sum()
        tr_sum = tr.rolling(window=14, min_periods=14).sum()

        vi_plus = vm_plus_sum / (tr_sum + 1e-10)
        vi_minus = vm_minus_sum / (tr_sum + 1e-10)
        return vi_plus - vi_minus

    def _relative_strength_vs_market_20d_batch(
        self, close_df: pd.DataFrame
    ) -> pd.DataFrame:
        """批量计算 RELATIVE_STRENGTH_VS_MARKET_20D（所有列一次性处理）"""
        # 计算日收益率
        etf_returns = close_df.pct_change(fill_method=None)
        market_returns = etf_returns.mean(axis=1)  # 等权市场收益

        # 计算20日累计收益（使用 log return 近似）
        log_etf_ret = np.log1p(etf_returns)
        log_market_ret = np.log1p(market_returns)

        etf_cum = log_etf_ret.rolling(window=20, min_periods=20).sum()
        market_cum = log_market_ret.rolling(window=20, min_periods=20).sum()

        # 相对强度 = etf累计收益 - 市场累计收益
        relative_strength = etf_cum.sub(market_cum, axis=0)
        return relative_strength

    # =========================================================================
    # 维度 1：趋势/动量 (2个)
    # =========================================================================

    def mom_20d(self, close: pd.Series) -> pd.Series:
        """
        20日动量 | MOM_20D

        公式：(close[t] / close[t-20] - 1) * 100

        缺失处理：
        - 窗口内任一close缺失 → 该日MOM_20D = NaN（满窗原则）
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 动量序列（百分比形式）
        """
        mom = (close / close.shift(20) - 1) * 100
        return mom

    def slope_20d(self, close: pd.Series) -> pd.Series:
        """
        20日线性回归斜率 | SLOPE_20D (完全向量化 - 无.apply)

        公式：slope = Σ[(x - x̄)(y - ȳ)] / Σ[(x - x̄)²]

        实现：使用scipy.signal.lfilter一次性完成所有窗口计算
        性能：O(N)，比.apply快20-30倍
        """
        from scipy.signal import lfilter

        # 预计算固定X序列(1..20)的统计量
        x = np.arange(1, 21, dtype=np.float64)
        x_mean = x.mean()  # 10.5
        x_dev = x - x_mean
        denom = (x_dev**2).sum()  # 665.0

        # 计算滑动窗口的 Σ[(x-x̄)(y-ȳ)]
        # = Σ(x-x̄)·y - x̄·Σ(x-x̄)·窗口均值
        # 由于Σ(x-x̄)=0，简化为 Σ[(x-x̄)·y]
        y = close.values

        # 使用lfilter计算加权滑动和：Σ[w[i]·y[t-i]]
        # 权重为翻转的x_dev（因为lfilter是卷积）
        weights = x_dev[::-1]
        weighted_sum = lfilter(weights, [1.0], y)

        # 计算斜率
        slope_vals = weighted_sum / denom

        # 前19个值设为NaN（满窗要求）
        slope_vals[:19] = np.nan

        return pd.Series(slope_vals, index=close.index)

    # =========================================================================
    # 维度 2：价格位置 (2个，有界[0,1])
    # =========================================================================

    def price_position_20d(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> pd.Series:
        """
        20日价格位置 | PRICE_POSITION_20D

        公式：(close[t] - min(low[-20:])) / (max(high[-20:]) - min(low[-20:]))
              如果high==low（无波动），返回0.5

        缺失处理：
        - 窗口内任一close/high/low缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[0,1]）
        极值截断：无需（有界[0,1]）

        Returns:
            pd.Series: 价格位置 [0, 1]
        """
        # 向量化计算：滚动高点和低点
        high_max = high.rolling(window=20, min_periods=20).max()
        low_min = low.rolling(window=20, min_periods=20).min()

        # 计算位置
        range_val = high_max - low_min
        position = (close - low_min) / range_val

        # 无波动时（range=0）返回0.5
        position = position.where(range_val > 1e-10, 0.5)

        # 截断到[0,1]
        return position.clip(0, 1)

    def price_position_120d(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> pd.Series:
        """
        120日价格位置 | PRICE_POSITION_120D

        公式：(close[t] - min(low[-120:])) / (max(high[-120:]) - min(low[-120:]))

        缺失处理：
        - 窗口内任一close/high/low缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[0,1]）
        极值截断：无需（有界[0,1]）

        Returns:
            pd.Series: 价格位置 [0, 1]
        """
        # 向量化计算
        high_max = high.rolling(window=120, min_periods=120).max()
        low_min = low.rolling(window=120, min_periods=120).min()

        range_val = high_max - low_min
        position = (close - low_min) / range_val

        # 无波动时返回0.5
        position = position.where(range_val > 1e-10, 0.5)

        return position.clip(0, 1)

    # =========================================================================
    # 维度 3：波动/风险 (2个)
    # =========================================================================

    def ret_vol_20d(self, close: pd.Series) -> pd.Series:
        """
        20日收益波动率 | RET_VOL_20D

        公式：std(pct_change(close)[-20:])

        缺失处理：
        - 窗口内任一close缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 收益波动率（百分比）
        """
        ret = close.pct_change(fill_method=None) * 100  # 转为百分比
        vol = ret.rolling(window=20).std()
        return vol

    def max_dd_60d(self, close: pd.Series) -> pd.Series:
        """
        60日最大回撤 | MAX_DD_60D (Numba加速 - 无.apply)

        公式：
        cummax = cumulative_maximum(close[-60:])
        drawdown = (close - cummax) / cummax
        max_dd = abs(min(drawdown))

        实现：Numba JIT编译，O(60N)复杂度
        """
        result = _rolling_max_dd_numba(close.values, window=60)
        return pd.Series(result, index=close.index)

    # =========================================================================
    # 维度 4：成交量 (2个)
    # =========================================================================

    def vol_ratio_20d(self, volume: pd.Series) -> pd.Series:
        """
        20日成交量比率 | VOL_RATIO_20D (完全向量化 - 无.apply)

        公式：
        recent_vol = mean(volume[-20:])
        past_vol = mean(volume[-40:-20])
        vol_ratio = recent_vol / past_vol

        实现：使用rolling().mean()和shift()，O(N)复杂度
        """
        eps = 1e-10

        # 最近20日平均量
        recent = volume.rolling(window=20, min_periods=20).mean()

        # 前20日平均量（平移20天的20日均线）
        past = volume.rolling(window=20, min_periods=20).mean().shift(20)

        # 计算比率，避免除零
        ratio = recent / (past + eps)

        # 当past接近0时设为NaN
        ratio = ratio.where(past >= eps, np.nan)

        return ratio

    def vol_ratio_60d(self, volume: pd.Series) -> pd.Series:
        """
        60日成交量比率 | VOL_RATIO_60D (完全向量化 - 无.apply)

        公式：
        recent_vol = mean(volume[-60:])
        past_vol = mean(volume[-120:-60])
        vol_ratio = recent_vol / past_vol

        实现：使用rolling().mean()和shift()，O(N)复杂度
        """
        eps = 1e-10

        # 最近60日平均量
        recent = volume.rolling(window=60, min_periods=60).mean()

        # 前60日平均量（平移60天的60日均线）
        past = volume.rolling(window=60, min_periods=60).mean().shift(60)

        # 计算比率，避免除零
        ratio = recent / (past + eps)

        # 当past接近0时设为NaN
        ratio = ratio.where(past >= eps, np.nan)

        return ratio

    # =========================================================================
    # 维度 5：价量耦合 (1个，有界[-1,1])
    # =========================================================================

    def pv_corr_20d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        20日价量相关性 | PV_CORR_20D

        公式：correlation(pct_change(close), pct_change(volume))
              over 20-day window

        缺失处理：
        - 窗口内任一close/volume缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[-1,1]）
        极值截断：无需（有界[-1,1]）

        Returns:
            pd.Series: 相关系数 [-1, 1]
        """
        ret_price = close.pct_change(fill_method=None)
        ret_volume = volume.pct_change(fill_method=None)

        # 🔧 优化：使用pandas内置rolling corr代替手工循环
        # 满窗原则：窗口内任一NaN会导致结果为NaN
        corr_series = ret_price.rolling(window=20, min_periods=20).corr(ret_volume)

        return corr_series

    # =========================================================================
    # 维度 6：反转/过热 (1个，有界[0,100])
    # =========================================================================

    def rsi_14(self, close: pd.Series) -> pd.Series:
        """
        14日相对强度指数 | RSI_14

        公式：
        RS = avg_gain / avg_loss (14-day)
        RSI = 100 - (100 / (1 + RS))

        缺失处理：
        - 窗口内任一close缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[0,100]）
        极值截断：无需（有界[0,100]）

        Returns:
            pd.Series: RSI [0, 100]
        """
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # =========================================================================
    # 维度 7：资金流 (2个) - 第1批新增
    # =========================================================================

    def obv_slope_10d(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        10日OBV能量潮斜率 | OBV_SLOPE_10D (完全向量化 - 无.apply)

        公式：
        1. OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]
        2. SLOPE = linear_regression_slope(OBV, window=10)

        实现：使用scipy.signal.lfilter + cumsum，O(N)复杂度
        """
        from scipy.signal import lfilter

        # 计算OBV：累计 sign(price_change) * volume
        price_change = close.diff()
        sign = np.sign(price_change.values)
        obv_vals = np.cumsum(sign * volume.values)

        # 预计算10日窗口的回归权重
        x = np.arange(1, 11, dtype=np.float64)
        x_mean = x.mean()  # 5.5
        x_dev = x - x_mean
        denom = (x_dev**2).sum()  # 82.5

        # lfilter计算加权滑动和
        weights = x_dev[::-1]
        weighted_sum = lfilter(weights, [1.0], obv_vals)

        # 计算斜率
        slope_vals = weighted_sum / denom

        # 前9个值设为NaN（满窗要求）
        slope_vals[:9] = np.nan

        return pd.Series(slope_vals, index=close.index)

    def cmf_20d(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        20日蔡金资金流 | CMF_20D

        公式：
        1. MFM[t] = ((close - low) - (high - close)) / (high - low)
        2. MFV[t] = MFM[t] * volume[t]
        3. CMF = sum(MFV, 20) / sum(volume, 20)

        逻辑：
        - MFM衡量日内收盘价的位置（接近高点=1，接近低点=-1）
        - 乘以成交量得到资金流量
        - 20日累计反映资金流向

        缺失处理：
        - 窗口内任一high/low/close/volume缺失 → NaN
        - high=low时（无波动）→ NaN
        - 无任何向前填充

        标准化：无需（有界[-1,1]）
        极值截断：无需（有界[-1,1]）

        Returns:
            pd.Series: CMF [-1, 1]
        """
        # 计算MFM（Money Flow Multiplier）
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

        # 当high=low时，设为NaN
        mfm = mfm.where(high != low, np.nan)

        # 计算MFV（Money Flow Volume）
        mfv = mfm * volume

        # 计算20日CMF
        cmf = mfv.rolling(window=20, min_periods=20).sum() / (
            volume.rolling(window=20, min_periods=20).sum() + 1e-10
        )

        return cmf

    # =========================================================================
    # 维度 8：风险调整动量 (2个) - 第2批新增
    # =========================================================================

    def sharpe_ratio_20d(self, close: pd.Series) -> pd.Series:
        """
        20日夏普比率 | SHARPE_RATIO_20D (完全向量化 - 无.apply)

        公式：
        Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)

        实现：使用rolling().mean()/std()，O(N)复杂度
        """
        eps = 1e-10

        # 计算日收益率
        returns = close.pct_change(fill_method=None)

        # 20日均值和标准差
        mean_ret = returns.rolling(window=20, min_periods=20).mean()
        std_ret = returns.rolling(window=20, min_periods=20).std()

        # 年化夏普比率
        sharpe = (mean_ret / (std_ret + eps)) * np.sqrt(252)

        # 标准差接近0时设为NaN
        sharpe = sharpe.where(std_ret >= eps, np.nan)

        return sharpe

    def calmar_ratio_60d(self, close: pd.Series) -> pd.Series:
        """
        60日卡玛比率 | CALMAR_RATIO_60D (Numba加速 - 无.apply)

        公式：
        Calmar = cumulative_return / abs(max_drawdown)

        实现：Numba JIT编译，O(60N)复杂度
        """
        result = _rolling_calmar_numba(close.values, window=60)
        return pd.Series(result, index=close.index)

    # =========================================================================
    # 维度 9：趋势强度 (2个) - 第3批新增
    # =========================================================================

    def adx_14d(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        14日平均趋向指数 | ADX_14D

        公式：
        1. +DM = max(high[t] - high[t-1], 0)
        2. -DM = max(low[t-1] - low[t], 0)
        3. TR = max(high - low, abs(high - close.shift(1)), abs(low - close.shift(1)))
        4. +DI = 100 * EMA(+DM, 14) / EMA(TR, 14)
        5. -DI = 100 * EMA(-DM, 14) / EMA(TR, 14)
        6. DX = 100 * abs(+DI - -DI) / (+DI + -DI)
        7. ADX = EMA(DX, 14)

        逻辑：
        - ADX > 25：强趋势
        - ADX < 20：震荡市
        - 不指示方向，只指示强度

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[0,100]）
        极值截断：无需（有界[0,100]）

        Returns:
            pd.Series: ADX [0, 100]
        """
        # 计算+DM和-DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # 计算TR（真实波幅）
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算14日EMA
        atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
        plus_di = 100 * (
            plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
        )
        minus_di = 100 * (
            minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / (atr + 1e-10)
        )

        # 计算DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))

        # 计算ADX
        adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()

        return adx

    def vortex_14d(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        14日螺旋指标 | VORTEX_14D

        公式：
        1. VM+ = abs(high[t] - low[t-1])
        2. VM- = abs(low[t] - high[t-1])
        3. TR = max(high - low, abs(high - close[t-1]), abs(low - close[t-1]))
        4. VI+ = sum(VM+, 14) / sum(TR, 14)
        5. VI- = sum(VM-, 14) / sum(TR, 14)
        6. Vortex = VI+ - VI-

        逻辑：
        - Vortex > 0：上升趋势
        - Vortex < 0：下降趋势
        - 交叉点可能是趋势反转信号

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: Vortex差值
        """
        # 计算VM+和VM-
        vm_plus = (high - low.shift(1)).abs()
        vm_minus = (low - high.shift(1)).abs()

        # 计算TR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算14日求和
        vm_plus_sum = vm_plus.rolling(window=14, min_periods=14).sum()
        vm_minus_sum = vm_minus.rolling(window=14, min_periods=14).sum()
        tr_sum = tr.rolling(window=14, min_periods=14).sum()

        # 计算VI+和VI-
        vi_plus = vm_plus_sum / (tr_sum + 1e-10)
        vi_minus = vm_minus_sum / (tr_sum + 1e-10)

        # Vortex = VI+ - VI-
        vortex = vi_plus - vi_minus

        return vortex

    # =========================================================================
    # 维度 10：相对强度 (2个) - 第4批新增
    # =========================================================================

    def relative_strength_vs_market_20d(
        self, close: pd.Series, market_close: pd.DataFrame
    ) -> pd.Series:
        """
        20日相对市场强度 | RELATIVE_STRENGTH_VS_MARKET_20D

        公式：
        1. market_ret = mean(all_etf_returns)  # 等权市场组合
        2. etf_ret = individual_etf_return
        3. relative_strength = etf_ret - market_ret

        逻辑：
        - 正值：跑赢市场
        - 负值：跑输市场
        - 识别相对强势的ETF

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 相对强度
        """
        # 计算个股收益率
        etf_returns = close.pct_change(fill_method=None)

        # 计算市场收益率（所有ETF等权平均）
        market_returns = market_close.pct_change(fill_method=None).mean(axis=1)

        # 计算20日累计相对强度
        def calc_relative_strength(idx):
            if idx < 20:
                return np.nan

            window_etf_ret = etf_returns.iloc[idx - 19 : idx + 1]
            window_market_ret = market_returns.iloc[idx - 19 : idx + 1]

            if window_etf_ret.isna().any() or window_market_ret.isna().any():
                return np.nan

            # 累计收益差
            etf_cum = (1 + window_etf_ret).prod() - 1
            market_cum = (1 + window_market_ret).prod() - 1

            return etf_cum - market_cum

        relative_strength = pd.Series(
            [calc_relative_strength(i) for i in range(len(close))], index=close.index
        )

        return relative_strength

    def correlation_to_market_20d(
        self, close: pd.Series, market_close: pd.DataFrame
    ) -> pd.Series:
        """
        20日与市场相关性 | CORRELATION_TO_MARKET_20D

        公式：
        correlation(etf_returns, market_returns) over 20-day window

        逻辑：
        - 高相关（接近1）：跟随市场
        - 低相关（接近0）：独立行情
        - 负相关（<0）：对冲属性

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：无需（有界[-1,1]）
        极值截断：无需（有界[-1,1]）

        Returns:
            pd.Series: 相关系数 [-1, 1]
        """
        # 计算个股收益率
        etf_returns = close.pct_change(fill_method=None)

        # 计算市场收益率（所有ETF等权平均）
        market_returns = market_close.pct_change(fill_method=None).mean(axis=1)

        # 计算20日滚动相关系数
        corr = etf_returns.rolling(window=20, min_periods=20).corr(market_returns)

        return corr

    # =========================================================================
    # A方案优先因子 (4个核心增量因子)
    # =========================================================================

    def tsmom_60d(self, close: pd.Series) -> pd.Series:
        """
        60日时间序列动量 | TSMOM_60D

        公式：sign(close[t] / SMA(close, 60) - 1)
        或简化版：close[t] / SMA(close, 60) - 1（保留强度）

        逻辑：
        - 正值：价格在均线之上（上升趋势）
        - 负值：价格在均线之下（下降趋势）
        - 绝对值：偏离程度

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 时间序列动量（百分比形式）
        """
        sma_60 = close.rolling(window=60, min_periods=60).mean()
        tsmom = (close / sma_60 - 1) * 100  # 转为百分比
        return tsmom

    def tsmom_120d(self, close: pd.Series) -> pd.Series:
        """
        120日时间序列动量 | TSMOM_120D

        公式：close[t] / SMA(close, 120) - 1

        逻辑：
        - 长期趋势强度
        - 与TSMOM_60D互补（不同时间尺度）

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 长期时间序列动量（百分比形式）
        """
        sma_120 = close.rolling(window=120, min_periods=120).mean()
        tsmom = (close / sma_120 - 1) * 100  # 转为百分比
        return tsmom

    def breakout_20d(self, high: pd.Series, close: pd.Series) -> pd.Series:
        """
        20日突破信号 | BREAKOUT_20D

        公式：
        1. max_high_20 = max(high[-20:])
        2. breakout = (close[t] - max_high_20) / max_high_20

        逻辑：
        - 正值：突破前20日高点（强势信号）
        - 负值：未突破（弱势）
        - 绝对值：突破强度

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 突破强度（百分比形式）
        """
        # 计算前20日最高价（不包括当日）
        max_high = high.shift(1).rolling(window=20, min_periods=20).max()

        # 计算突破强度
        breakout = (close - max_high) / (max_high + 1e-10) * 100  # 转为百分比

        return breakout

    def turnover_accel_5_20(self, volume: pd.Series) -> pd.Series:
        """
        5日vs20日换手率加速度 | TURNOVER_ACCEL_5_20

        公式：
        1. avg_vol_5 = mean(volume[-5:])
        2. avg_vol_20 = mean(volume[-20:])
        3. accel = (avg_vol_5 / avg_vol_20) - 1

        逻辑：
        - 正值：近期成交量加速（资金热度上升）
        - 负值：成交量萎缩（资金退潮）
        - 识别资金流入/流出的变化

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 换手率加速度（百分比形式）
        """
        avg_vol_5 = volume.rolling(window=5, min_periods=5).mean()
        avg_vol_20 = volume.rolling(window=20, min_periods=20).mean()

        # 计算加速度
        accel = (avg_vol_5 / (avg_vol_20 + 1e-10) - 1) * 100  # 转为百分比

        return accel

    # =========================================================================
    # 辅助过滤因子（成本与容量约束，不作为选择因子）
    # =========================================================================

    def realized_vol_20d(self, close: pd.Series) -> pd.Series:
        """
        20日实际波动率 | REALIZED_VOL_20D

        公式：std(daily_returns) over 20-day window × sqrt(252)（年化）

        用途：
        - 风险过滤器：高波动期降权/减仓
        - 目标波动策略：动态调整仓位
        - 不作为因子打分，作为约束条件

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        Returns:
            pd.Series: 年化波动率（百分比形式）
        """
        returns = close.pct_change(fill_method=None)
        realized_vol = (
            returns.rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
        )
        return realized_vol

    def amihud_illiquidity(
        self, close: pd.Series, volume: pd.Series, amount: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Amihud流动性指标 | AMIHUD_ILLIQUIDITY

        公式：mean(|daily_return| / daily_amount) over 20-day window
        如果amount不可得，用 volume × close 近似

        用途：
        - 冲击成本代理：值越大→冲击成本越高→降权或不交易
        - 容量约束：Amihud > 阈值 → 排除
        - 100万资金体量：关键约束条件

        缺失处理：
        - 窗口内任一缺失 → NaN
        - 无任何向前填充

        Returns:
            pd.Series: Amihud流动性指标（×10^6，便于阅读）
        """
        returns = close.pct_change(fill_method=None).abs()

        # 计算成交额
        if amount is None:
            amount = volume * close  # 近似

        # 计算Amihud
        amihud = returns / (amount + 1e-10)

        # 20日滚动平均
        amihud_avg = amihud.rolling(window=20, min_periods=20).mean()

        # 放大为便于阅读的单位（×10^6）
        return amihud_avg * 1e6

    def spread_proxy(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """
        日内价差代理 | SPREAD_PROXY

        公式：(high - low) / close

        用途：
        - 交易成本代理：价差越大→成本越高
        - 流动性过滤器：极端价差→排除
        - 点差估计的简化版本

        缺失处理：
        - 任一缺失 → NaN
        - 无任何向前填充

        Returns:
            pd.Series: 价差比率（百分比形式）
        """
        spread = (high - low) / (close + 1e-10) * 100  # 转为百分比

        # 可选：20日平滑避免单日异常
        spread_smooth = spread.rolling(window=5, min_periods=5).mean()

        return spread_smooth

    # =========================================================================
    # 批量计算
    # =========================================================================

    def compute_all_factors(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算所有精选因子

        Args:
            prices: 价格数据字典
                {
                    'close': pd.DataFrame (index=date, columns=symbols),
                    'high': pd.DataFrame,
                    'low': pd.DataFrame,
                    'volume': pd.DataFrame
                }

        Returns:
            pd.DataFrame: 多层列索引 (因子名, 标的代码)
                          index=日期
                          如果某个标的某日数据缺失，对应因子=NaN

        Raises:
            ValueError: 如果缺少必要的OHLCV数据
        """
        required_cols = {"close", "high", "low", "volume"}
        if not required_cols.issubset(set(prices.keys())):
            raise ValueError(f"缺少必要列: {required_cols - set(prices.keys())}")

        close = prices["close"]
        high = prices["high"]
        low = prices["low"]
        volume = prices["volume"]

        symbols = close.columns

        # ========== 100%向量化：批量方法，零Python循环 ==========

        # 维度1：趋势/动量
        mom_20d = (close / close.shift(20) - 1) * 100
        slope_20d = self._slope_20d_batch(close)

        # 维度2：价格位置
        price_position_20d = self._price_position_batch(close, high, low, window=20)
        price_position_120d = self._price_position_batch(close, high, low, window=120)

        # 维度3：波动/风险
        ret = close.pct_change(fill_method=None) * 100
        ret_vol_20d = ret.rolling(window=20).std()
        max_dd_60d = self._max_dd_60d_batch(close)

        # 维度4：成交量
        eps = 1e-10
        recent_20 = volume.rolling(window=20, min_periods=20).mean()
        past_20 = volume.rolling(window=20, min_periods=20).mean().shift(20)
        vol_ratio_20d = (recent_20 / (past_20 + eps)).where(past_20 >= eps, np.nan)

        recent_60 = volume.rolling(window=60, min_periods=60).mean()
        past_60 = volume.rolling(window=60, min_periods=60).mean().shift(60)
        vol_ratio_60d = (recent_60 / (past_60 + eps)).where(past_60 >= eps, np.nan)

        # 维度5：价量耦合
        ret_price = close.pct_change(fill_method=None)
        ret_volume = volume.pct_change(fill_method=None)
        pv_corr_20d = ret_price.rolling(window=20, min_periods=20).corr(ret_volume)

        # 维度6：反转（RSI）
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_14 = 100 - (100 / (1 + rs))

        # 维度7：资金流
        obv_slope_10d = self._obv_slope_10d_batch(close, volume)
        cmf_20d = self._cmf_20d_batch(high, low, close, volume)

        # 维度8：风险调整
        returns = close.pct_change(fill_method=None)
        mean_ret = returns.rolling(window=20, min_periods=20).mean()
        std_ret = returns.rolling(window=20, min_periods=20).std()
        sharpe_ratio_20d = (mean_ret / (std_ret + eps)) * np.sqrt(252)
        sharpe_ratio_20d = sharpe_ratio_20d.where(std_ret >= eps, np.nan)

        calmar_ratio_60d = self._calmar_60d_batch(close)

        # 维度9：趋势强度
        adx_14d = self._adx_14d_batch(high, low, close)
        vortex_14d = self._vortex_14d_batch(high, low, close)

        # 维度10：相对强度
        relative_strength_vs_market_20d = self._relative_strength_vs_market_20d_batch(
            close
        )

        # correlation_to_market_20d
        etf_returns = close.pct_change(fill_method=None)
        market_returns = etf_returns.mean(axis=1)
        correlation_to_market_20d = etf_returns.rolling(window=20, min_periods=20).corr(
            market_returns
        )

        # ========== 使用pd.concat构建多层索引，一次性组装 ==========
        # 每个因子是一个(T, N)的DataFrame，keys为因子名
        factor_dfs = {
            "MOM_20D": mom_20d,
            "SLOPE_20D": slope_20d,
            "PRICE_POSITION_20D": price_position_20d,
            "PRICE_POSITION_120D": price_position_120d,
            "RET_VOL_20D": ret_vol_20d,
            "MAX_DD_60D": max_dd_60d,
            "VOL_RATIO_20D": vol_ratio_20d,
            "VOL_RATIO_60D": vol_ratio_60d,
            "PV_CORR_20D": pv_corr_20d,
            "RSI_14": rsi_14,
            "OBV_SLOPE_10D": obv_slope_10d,
            "CMF_20D": cmf_20d,
            "SHARPE_RATIO_20D": sharpe_ratio_20d,
            "CALMAR_RATIO_60D": calmar_ratio_60d,
            "ADX_14D": adx_14d,
            "VORTEX_14D": vortex_14d,
            "RELATIVE_STRENGTH_VS_MARKET_20D": relative_strength_vs_market_20d,
            "CORRELATION_TO_MARKET_20D": correlation_to_market_20d,
        }

        # 一次性拼接：columns=(factor, symbol)
        result = pd.concat(factor_dfs, axis=1, keys=factor_dfs.keys())
        result = result.sort_index(axis=1)

        logger.info(
            f"✅ 计算完成: {len(symbols)}个标的 × {len(self.factors_metadata)}个因子"
        )

        return result

    def get_metadata(self, factor_name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.factors_metadata.get(factor_name)

    def list_factors(self) -> Dict[str, FactorMetadata]:
        """列出所有因子及其元数据"""
        return self.factors_metadata


# =========================================================================
# 使用示例
# =========================================================================

if __name__ == "__main__":
    print("PreciseFactorLibrary v2 示例")
    print("=" * 70)

    # 创建库实例
    lib = PreciseFactorLibrary()

    # 列出所有因子
    print("\n【精选因子清单】")
    for factor_name, metadata in lib.list_factors().items():
        bounded = "有界" if metadata.bounded else "无界"
        print(f"  {factor_name:20} | {metadata.description:30} | {bounded}")

    print("\n【使用步骤】")
    print("  1. 准备prices数据: {'close': df, 'high': df, 'low': df, 'volume': df}")
    print("  2. 调用 lib.compute_all_factors(prices) 获取所有因子")
    print("  3. 在WFO内进行标准化和极值截断")
    print("  4. 提交给IC计算和约束筛选模块")

    print("\n✅ 因子库v2已准备就绪")
