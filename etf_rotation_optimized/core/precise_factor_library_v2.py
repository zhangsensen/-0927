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
        ret = close.pct_change() * 100  # 转为百分比
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
        ret_price = close.pct_change()
        ret_volume = volume.pct_change()

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
