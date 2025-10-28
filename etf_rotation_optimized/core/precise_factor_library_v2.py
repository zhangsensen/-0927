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

logger = logging.getLogger(__name__)


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
        20日线性回归斜率 | SLOPE_20D

        公式：np.polyfit(x=[1..20], y=close[-20:], 1)[0]

        缺失处理：
        - 窗口内任一close缺失 → 该日SLOPE = NaN（满窗原则）
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 斜率序列
        """

        def calc_slope(x):
            if x.isna().any() or len(x) < 20:
                return np.nan
            try:
                x_vals = np.arange(1, 21)
                y_vals = x.values
                slope = np.polyfit(x_vals, y_vals, 1)[0]
                return slope
            except:
                return np.nan

        slope = close.rolling(window=20).apply(calc_slope, raw=False)
        return slope

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
        60日最大回撤 | MAX_DD_60D

        公式：
        cummax = cumulative_maximum(close[-60:])
        drawdown = (close - cummax) / cummax
        max_dd = abs(min(drawdown))

        缺失处理：
        - 窗口内任一close缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 最大回撤（绝对值，百分比形式）
        """

        def calc_max_dd(x):
            if x.isna().any() or len(x) < 60:
                return np.nan
            try:
                cummax = x.cummax()
                drawdown = (x - cummax) / cummax
                max_dd = abs(drawdown.min())
                return max_dd * 100  # 转为百分比
            except:
                return np.nan

        max_dd = close.rolling(window=60).apply(calc_max_dd, raw=False)
        return max_dd

    # =========================================================================
    # 维度 4：成交量 (2个)
    # =========================================================================

    def vol_ratio_20d(self, volume: pd.Series) -> pd.Series:
        """
        20日成交量比率 | VOL_RATIO_20D

        公式：
        recent_vol = mean(volume[-20:])
        past_vol = mean(volume[-40:-20])  # 前20日平均
        vol_ratio = recent_vol / past_vol (避免除零)

        缺失处理：
        - 窗口内任一volume缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行（可选log变换）
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 成交量比率
        """

        def calc_vol_ratio(x):
            if x.isna().any() or len(x) < 40:
                return np.nan
            try:
                recent = x[-20:].mean()
                past = x[-40:-20].mean()
                if past < 1e-10:
                    return np.nan
                return recent / past
            except:
                return np.nan

        vol_ratio = volume.rolling(window=40).apply(calc_vol_ratio, raw=False)
        return vol_ratio

    def vol_ratio_60d(self, volume: pd.Series) -> pd.Series:
        """
        60日成交量比率 | VOL_RATIO_60D

        公式：
        recent_vol = mean(volume[-60:])
        past_vol = mean(volume[-120:-60])
        vol_ratio = recent_vol / past_vol

        缺失处理：
        - 窗口内任一volume缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 成交量比率
        """

        def calc_vol_ratio(x):
            if x.isna().any() or len(x) < 120:
                return np.nan
            try:
                recent = x[-60:].mean()
                past = x[-120:-60].mean()
                if past < 1e-10:
                    return np.nan
                return recent / past
            except:
                return np.nan

        vol_ratio = volume.rolling(window=120).apply(calc_vol_ratio, raw=False)
        return vol_ratio

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
        10日OBV能量潮斜率 | OBV_SLOPE_10D

        公式：
        1. OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]
        2. SLOPE = linear_regression_slope(OBV, window=10)

        逻辑：
        - OBV累计了资金流向（涨日volume为正，跌日为负）
        - 斜率反映资金流入/流出的趋势强度

        缺失处理：
        - 窗口内任一close/volume缺失 → NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: OBV斜率
        """
        # 计算价格变化的符号
        price_change = close.diff()
        sign = np.sign(price_change)

        # 计算OBV：累计 sign * volume
        obv = (sign * volume).cumsum()

        # 计算10日线性回归斜率
        def calc_slope(x):
            if x.isna().any() or len(x) < 10:
                return np.nan
            try:
                # 线性回归：y = ax + b，返回斜率a
                x_vals = np.arange(len(x))
                y_vals = x.values
                slope = np.polyfit(x_vals, y_vals, 1)[0]
                return slope
            except:
                return np.nan

        obv_slope = obv.rolling(window=10).apply(calc_slope, raw=False)
        return obv_slope

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
        20日夏普比率 | SHARPE_RATIO_20D

        公式：
        Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)

        逻辑：
        - 衡量单位风险的收益
        - 高夏普表示稳定上涨
        - 低夏普表示高波动或负收益

        缺失处理：
        - 窗口内任一close缺失 → NaN
        - 标准差=0（无波动）→ NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 夏普比率
        """
        returns = close.pct_change(fill_method=None)

        def calc_sharpe(x):
            if x.isna().any() or len(x) < 20:
                return np.nan
            try:
                mean_ret = x.mean()
                std_ret = x.std()
                if std_ret < 1e-10:
                    return np.nan
                # 年化：sqrt(252)
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
                return sharpe
            except:
                return np.nan

        sharpe = returns.rolling(window=20).apply(calc_sharpe, raw=False)
        return sharpe

    def calmar_ratio_60d(self, close: pd.Series) -> pd.Series:
        """
        60日卡玛比率 | CALMAR_RATIO_60D

        公式：
        Calmar = cumulative_return / abs(max_drawdown)

        逻辑：
        - 衡量收益与回撤的比率
        - 高卡玛表示高收益低回撤
        - 惩罚大幅回撤的策略

        缺失处理：
        - 窗口内任一close缺失 → NaN
        - 最大回撤=0（无回撤）→ NaN
        - 无任何向前填充

        标准化：WFO内执行
        极值截断：WFO内 2.5%/97.5%分位

        Returns:
            pd.Series: 卡玛比率
        """

        def calc_calmar(x):
            if x.isna().any() or len(x) < 60:
                return np.nan
            try:
                # 累计收益
                cum_ret = (x.iloc[-1] / x.iloc[0]) - 1

                # 计算最大回撤
                cum_prices = x / x.iloc[0]
                running_max = cum_prices.expanding().max()
                drawdown = (cum_prices - running_max) / running_max
                max_dd = drawdown.min()

                if abs(max_dd) < 1e-10:
                    return np.nan

                calmar = cum_ret / abs(max_dd)
                return calmar
            except:
                return np.nan

        calmar = close.rolling(window=60).apply(calc_calmar, raw=False)
        return calmar

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

        # 初始化多层结果DataFrame
        all_factors = {}

        # 遍历所有标的
        for symbol in symbols:
            symbol_factors = {}

            try:
                # 维度1：趋势/动量
                symbol_factors["MOM_20D"] = self.mom_20d(close[symbol])
                symbol_factors["SLOPE_20D"] = self.slope_20d(close[symbol])

                # 维度2：价格位置
                symbol_factors["PRICE_POSITION_20D"] = self.price_position_20d(
                    close[symbol], high[symbol], low[symbol]
                )
                symbol_factors["PRICE_POSITION_120D"] = self.price_position_120d(
                    close[symbol], high[symbol], low[symbol]
                )

                # 维度3：波动/风险
                symbol_factors["RET_VOL_20D"] = self.ret_vol_20d(close[symbol])
                symbol_factors["MAX_DD_60D"] = self.max_dd_60d(close[symbol])

                # 维度4：成交量
                symbol_factors["VOL_RATIO_20D"] = self.vol_ratio_20d(volume[symbol])
                symbol_factors["VOL_RATIO_60D"] = self.vol_ratio_60d(volume[symbol])

                # 维度5：价量耦合
                symbol_factors["PV_CORR_20D"] = self.pv_corr_20d(
                    close[symbol], volume[symbol]
                )

                # 维度6：反转
                symbol_factors["RSI_14"] = self.rsi_14(close[symbol])

                # 维度7：资金流（第1批新增）
                symbol_factors["OBV_SLOPE_10D"] = self.obv_slope_10d(
                    close[symbol], volume[symbol]
                )
                symbol_factors["CMF_20D"] = self.cmf_20d(
                    high[symbol], low[symbol], close[symbol], volume[symbol]
                )

                # 维度8：风险调整动量（第2批新增）
                symbol_factors["SHARPE_RATIO_20D"] = self.sharpe_ratio_20d(
                    close[symbol]
                )
                symbol_factors["CALMAR_RATIO_60D"] = self.calmar_ratio_60d(
                    close[symbol]
                )

                # 维度9：趋势强度（第3批新增）
                symbol_factors["ADX_14D"] = self.adx_14d(
                    high[symbol], low[symbol], close[symbol]
                )
                symbol_factors["VORTEX_14D"] = self.vortex_14d(
                    high[symbol], low[symbol], close[symbol]
                )

                # 维度10：相对强度（第4批新增）
                symbol_factors["RELATIVE_STRENGTH_VS_MARKET_20D"] = (
                    self.relative_strength_vs_market_20d(close[symbol], close)
                )
                symbol_factors["CORRELATION_TO_MARKET_20D"] = (
                    self.correlation_to_market_20d(close[symbol], close)
                )

                # ========== [P0修复] 禁用新增因子，回滚到历史18个 ==========
                # # 时间序列动量（2个）
                # symbol_factors["TSMOM_60D"] = self.tsmom_60d(close[symbol])
                # symbol_factors["TSMOM_120D"] = self.tsmom_120d(close[symbol])
                #
                # # 突破信号（1个）
                # symbol_factors["BREAKOUT_20D"] = self.breakout_20d(
                #     high[symbol], close[symbol]
                # )
                #
                # # 资金流加速度（1个）
                # symbol_factors["TURNOVER_ACCEL_5_20"] = self.turnover_accel_5_20(
                #     volume[symbol]
                # )
                #
                # # 辅助过滤因子（成本/容量约束）
                # symbol_factors["REALIZED_VOL_20D"] = self.realized_vol_20d(close[symbol])
                # symbol_factors["AMIHUD_ILLIQUIDITY"] = self.amihud_illiquidity(
                #     close[symbol], volume[symbol]
                # )
                # symbol_factors["SPREAD_PROXY"] = self.spread_proxy(
                #     high[symbol], low[symbol], close[symbol]
                # )

                all_factors[symbol] = pd.DataFrame(symbol_factors)

            except Exception as e:
                logger.error(f"计算标的 {symbol} 的因子失败: {e}")
                # 为该标的创建全NaN的因子表
                all_factors[symbol] = pd.DataFrame(
                    np.nan,
                    index=close.index,
                    columns=list(self.factors_metadata.keys()),
                )

        # 合并所有标的的因子
        result = pd.concat(all_factors, axis=1)  # 多层列索引
        result.columns = result.columns.swaplevel(
            0, 1
        )  # (symbol, factor) -> (factor, symbol)
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
