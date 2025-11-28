"""
精确因子库 | Precise Factor Library
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
    精确因子库

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
