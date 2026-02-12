"""
精确因子库 v2 | Precise Factor Library v2
================================================================================
ETF轮动策略多维因子库，覆盖OHLCV价量因子和非OHLCV另类因子。

核心设计原则：
1. 严格遵循精确定义：公式、缺失处理、极值规则
2. 缺失值处理：原始缺失→保留NaN；满窗不足→NaN（无向前填充）
3. 标准化位置：WFO内完成（不在生成阶段）
4. 极值截断：2.5%/97.5%分位（有界因子跳过rank标准化）
5. 非OHLCV因子通过外部预计算parquet加载，与OHLCV因子在WFO中统一标准化

因子总览: 40个注册因子, 24个生产活跃, 7个信息维度桶
========================================================================

OHLCV因子 (34个注册, 18个活跃) — 数据源: raw/ETF/daily/
------------------------------------------------------------------------
桶A 趋势/动量 (6):
  ★ MOM_20D              20日动量百分比
  ★ SLOPE_20D            20日线性回归斜率 [S1成员]
  ★ SHARPE_RATIO_20D     20日夏普比率 [S1成员]
  ★ BREAKOUT_20D         20日突破信号
  ★ VORTEX_14D           14日涡旋指标
    PRICE_POSITION_20D   20日价格位置 [有界0-1]

桶B 持续位置 (2):
    PRICE_POSITION_120D  120日价格位置 [有界0-1]
  ★ CALMAR_RATIO_60D     60日卡玛比率 [C2成员]

桶C 量能确认 (3):
  ★ OBV_SLOPE_10D        OBV斜率 [S1成员]
  ★ UP_DOWN_VOL_RATIO_20D 上涨量/下跌量比
    CMF_20D              Chaikin资金流 [有界-1~1]

桶D 微观结构 (3):
    PV_CORR_20D          价量相关性 [有界-1~1]
  ★ AMIHUD_ILLIQUIDITY   Amihud非流动性 [C2成员]
  ★ GK_VOL_RATIO_20D     GK波动率比

桶E 趋势强度/风险 (4):
  ★ ADX_14D              趋势强度 [S1成员, 有界0-100]
  ★ CORRELATION_TO_MARKET_20D 市场相关性 [C2成员, 有界-1~1]
  ★ MAX_DD_60D           60日最大回撤
  ★ VOL_RATIO_20D        波动率比率

未入桶 (16个注册未活跃):
    RET_VOL_20D, VOL_RATIO_60D, RSI_14, RELATIVE_STRENGTH_VS_MARKET_20D,
    TSMOM_60D, TSMOM_120D, TURNOVER_ACCEL_5_20, REALIZED_VOL_20D,
    SPREAD_PROXY, SKEW_20D, KURT_20D, INFO_DISCRETE_20D, IBS,
    MEAN_REV_RATIO_20D, ABNORMAL_VOLUME_20D, ULCER_INDEX_20D,
    DD_DURATION_60D, PERM_ENTROPY_20D, HURST_60D, DOWNSIDE_DEV_20D

非OHLCV因子 (6个注册, 6个活跃) — 数据源: raw/ETF/fund_share/, margin/
------------------------------------------------------------------------
桶F 资金流向 (4): IC来源=基金份额申赎行为
  ★ SHARE_CHG_5D         5日份额变化率    IC=-0.050 (反向)
  ★ SHARE_CHG_10D        10日份额变化率   IC=-0.056 (最强)
  ★ SHARE_CHG_20D        20日份额变化率   IC=-0.047 (反向)
  ★ SHARE_ACCEL          份额变化加速度   IC=+0.034 (正向, 拐点信号)

桶G 杠杆行为 (2): IC来源=融资融券数据
  ★ MARGIN_CHG_10D       10日融资余额变化 IC=-0.047 (反向)
  ★ MARGIN_BUY_RATIO     融资买入占比     IC=-0.031 (反向)

有界因子 (跳过Winsorize, 使用rank标准化):
  ADX_14D[0,100], CMF_20D[-1,1], CORRELATION_TO_MARKET_20D[-1,1],
  PRICE_POSITION_20D[0,1], PRICE_POSITION_120D[0,1], PV_CORR_20D[-1,1], RSI_14[0,100]

生产策略:
  S1 = ADX_14D + OBV_SLOPE_10D + SHARPE_RATIO_20D + SLOPE_20D (桶A+C+E)
  C2 = AMIHUD_ILLIQUIDITY + CALMAR_RATIO_60D + CORRELATION_TO_MARKET_20D (桶B+D+E)

API:
  compute_all_factors(prices)           → Dict[str, DataFrame]  # OHLCV因子
  compute_non_ohlcv_factors(prices, fund_share, margin) → Dict[str, DataFrame]  # 非OHLCV因子
  get_metadata(name)                    → FactorMetadata
  list_factors()                        → Dict[str, FactorMetadata]
========================================================================
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import numba
from numba import njit

logger = logging.getLogger(__name__)


# ============================================================================
# Numba加速函数（模块级定义）
# ============================================================================


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True, parallel=True)
def _rolling_max_dd_batch(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """批量滑窗最大回撤（prange 并行逐列）"""
    T, N = prices_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _rolling_max_dd_numba(prices_2d[:, j], window)
    return result


@njit(cache=True, parallel=True)
def _rolling_calmar_batch(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """批量滑窗卡玛比率（prange 并行逐列）"""
    T, N = prices_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _rolling_calmar_numba(prices_2d[:, j], window)
    return result


# ============================================================================
# v4.2 新增 Numba 加速函数
# ============================================================================


@njit(cache=True)
def _rolling_ulcer_index_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滑窗 Ulcer Index 计算

    参数:
        prices: 1D价格序列
        window: 窗口长度

    返回:
        1D Ulcer Index 序列
    """
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_prices = prices[i - window + 1 : i + 1]

        # 检查NaN
        if np.any(np.isnan(window_prices)):
            result[i] = np.nan
            continue

        # 计算窗口内回撤百分比的均方根
        running_max = window_prices[0]
        sum_sq = 0.0
        for j in range(window):
            if window_prices[j] > running_max:
                running_max = window_prices[j]
            dd_pct = (window_prices[j] / running_max - 1.0) * 100.0
            sum_sq += dd_pct * dd_pct

        result[i] = np.sqrt(sum_sq / window)

    return result


@njit(cache=True, parallel=True)
def _rolling_ulcer_index_batch(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """批量滑窗 Ulcer Index（prange 并行逐列）"""
    T, N = prices_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _rolling_ulcer_index_numba(prices_2d[:, j], window)
    return result


@njit(cache=True)
def _dd_duration_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的回撤持续天数计算

    从每个时点回溯，计算连续未创新高的天数 / window

    参数:
        prices: 1D价格序列
        window: 归一化窗口 (60)

    返回:
        1D DD_DURATION 序列 [0, 1]
    """
    n = len(prices)
    result = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(prices[i]):
            continue

        # 向前回溯找最近一次创新高的天数
        duration = 0
        peak = prices[i]
        for k in range(i - 1, -1, -1):
            if np.isnan(prices[k]):
                break
            if prices[k] >= peak:
                break
            duration += 1

        result[i] = min(duration / window, 1.0)

    return result


@njit(cache=True, parallel=True)
def _dd_duration_batch(prices_2d: np.ndarray, window: int) -> np.ndarray:
    """批量回撤持续天数（prange 并行逐列）"""
    T, N = prices_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _dd_duration_numba(prices_2d[:, j], window)
    return result


@njit(cache=True)
def _permutation_entropy_numba(series: np.ndarray, window: int, m: int, delay: int) -> np.ndarray:
    """
    Numba加速的滑窗排列熵计算

    参数:
        series: 1D序列 (收益率)
        window: 滑窗长度
        m: embedding dimension (3)
        delay: time delay (1)

    返回:
        1D 排列熵序列 [0, 1] (归一化)
    """
    n = len(series)
    result = np.full(n, np.nan)
    # m=3 时有 3!=6 种排列模式
    n_perms = 1
    for k in range(1, m + 1):
        n_perms *= k
    max_entropy = np.log(n_perms)

    if max_entropy < 1e-10:
        return result

    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]

        if np.any(np.isnan(seg)):
            result[i] = np.nan
            continue

        # 统计排列模式频率
        counts = np.zeros(n_perms, dtype=np.float64)
        n_patterns = 0

        for t in range(len(seg) - (m - 1) * delay):
            # 提取子序列并获得排列模式索引
            # m=3: 比较 3 个元素的相对大小
            vals = np.empty(m)
            for k in range(m):
                vals[k] = seg[t + k * delay]

            # 将排列模式编码为整数索引 (Lehmer code)
            idx = 0
            for a in range(m):
                rank = 0
                for b in range(a + 1, m):
                    if vals[b] < vals[a]:
                        rank += 1
                # 乘以阶乘
                fac = 1
                for f in range(1, m - a):
                    fac *= f
                idx += rank * fac

            counts[idx] += 1.0
            n_patterns += 1

        if n_patterns == 0:
            continue

        # 计算 Shannon 熵
        entropy = 0.0
        for k in range(n_perms):
            if counts[k] > 0:
                p = counts[k] / n_patterns
                entropy -= p * np.log(p)

        # 归一化到 [0, 1]
        result[i] = entropy / max_entropy

    return result


@njit(cache=True, parallel=True)
def _permutation_entropy_batch(series_2d: np.ndarray, window: int, m: int, delay: int) -> np.ndarray:
    """批量滑窗排列熵（prange 并行逐列）"""
    T, N = series_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _permutation_entropy_numba(series_2d[:, j], window, m, delay)
    return result


@njit(cache=True)
def _hurst_rs_numba(series: np.ndarray, window: int) -> np.ndarray:
    """
    Numba加速的滑窗 Hurst 指数 (R/S 分析法)

    参数:
        series: 1D收益率序列
        window: 滑窗长度 (60)

    返回:
        1D Hurst 指数序列 [0, 1]
    """
    n = len(series)
    result = np.full(n, np.nan)

    # 使用 2 个子区间长度进行 R/S 估计: window/4, window/2
    # 简化版: 直接用整个窗口的 R/S 比较理论值
    for i in range(window - 1, n):
        seg = series[i - window + 1 : i + 1]

        if np.any(np.isnan(seg)):
            result[i] = np.nan
            continue

        # 对 2-3 个不同子区间长度计算 R/S
        log_n_vals = np.empty(3)
        log_rs_vals = np.empty(3)
        valid_count = 0

        for div_idx, n_div in enumerate([2, 4, 8]):
            sub_len = window // n_div
            if sub_len < 4:
                continue

            rs_sum = 0.0
            rs_count = 0

            for d in range(n_div):
                start = d * sub_len
                end = start + sub_len
                if end > window:
                    break

                sub = seg[start:end]
                mean_val = 0.0
                for k in range(sub_len):
                    mean_val += sub[k]
                mean_val /= sub_len

                # 计算累计偏差
                cum_dev = 0.0
                max_dev = -1e30
                min_dev = 1e30
                for k in range(sub_len):
                    cum_dev += sub[k] - mean_val
                    if cum_dev > max_dev:
                        max_dev = cum_dev
                    if cum_dev < min_dev:
                        min_dev = cum_dev

                r = max_dev - min_dev

                # 标准差
                var_sum = 0.0
                for k in range(sub_len):
                    var_sum += (sub[k] - mean_val) ** 2
                s = np.sqrt(var_sum / sub_len)

                if s > 1e-10:
                    rs_sum += r / s
                    rs_count += 1

            if rs_count > 0:
                avg_rs = rs_sum / rs_count
                if avg_rs > 0:
                    log_n_vals[valid_count] = np.log(sub_len)
                    log_rs_vals[valid_count] = np.log(avg_rs)
                    valid_count += 1

        if valid_count >= 2:
            # 简单线性回归 log(R/S) = H * log(n) + c
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            for k in range(valid_count):
                sum_x += log_n_vals[k]
                sum_y += log_rs_vals[k]
                sum_xy += log_n_vals[k] * log_rs_vals[k]
                sum_xx += log_n_vals[k] * log_n_vals[k]

            denom = valid_count * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-10:
                h = (valid_count * sum_xy - sum_x * sum_y) / denom
                # 截断到 [0, 1]
                if h < 0.0:
                    h = 0.0
                elif h > 1.0:
                    h = 1.0
                result[i] = h

    return result


@njit(cache=True, parallel=True)
def _hurst_rs_batch(series_2d: np.ndarray, window: int) -> np.ndarray:
    """批量滑窗 Hurst 指数（prange 并行逐列）"""
    T, N = series_2d.shape
    result = np.empty((T, N))
    for j in numba.prange(N):
        result[:, j] = _hurst_rs_numba(series_2d[:, j], window)
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
    production_ready: bool = True  # 是否可用于生产策略 (默认 True)
    risk_note: str = ""  # 风险说明
    orthogonal_v1: bool = True  # 是否在正交因子集 v1 中 (15/25)


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
                orthogonal_v1=False,  # 被 SPREAD_PROXY(0.81)+MAX_DD(0.70) 覆盖
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
                orthogonal_v1=False,  # 已在 EXCLUDE_FACTORS，Top2 -32%
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
                orthogonal_v1=False,  # corr=0.93 VORTEX, IC 不显著
            ),
            # ============ 第1批新增：资金流因子 ============
            # ✅ OBV_SLOPE_10D: Currently used in v3.4 production strategies
            # Historical note: Initial BT audit showed 61pp drift, but diagnosis (2025-12-16)
            # confirmed timing alignment and correct batch computation. Drift likely from
            # normalization differences in early audit scripts (now fixed).
            "OBV_SLOPE_10D": FactorMetadata(
                name="OBV_SLOPE_10D",
                description="10日OBV能量潮斜率",
                dimension="资金流",
                required_columns=["close", "volume"],
                window=10,
                bounded=False,
                direction="high_is_good",
                production_ready=True,  # ✅ Used in v3.4 production (136.52% + 129.85% returns)
                risk_note="42.2% NaN rate due to early-period data; ranking divergence 5% vs non-OBV combos",
                orthogonal_v1=False,  # 42% NaN, IC 不显著, 已在 EXCLUDE_FACTORS
            ),
            # ⚠️ CMF_20D: 2025-12-01 BT审计发现 VEC/BT 差异达 35pp，不可用于生产
            "CMF_20D": FactorMetadata(
                name="CMF_20D",
                description="20日蔡金资金流",
                dimension="资金流",
                required_columns=["high", "low", "close", "volume"],
                window=20,
                bounded=True,  # [-1,1]有界
                direction="high_is_good",
                production_ready=False,  # ❌ BT审计不通过
                risk_note="VEC/BT差异35pp，疑似计算不一致",
                orthogonal_v1=False,  # production_ready=False, 35pp VEC/BT 漂移
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
                orthogonal_v1=False,  # ≡ MOM_20D (corr=1.000)
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
            # ============ [v4.0] 重新启用7个因子，扩展到25因子库 ============
            "TSMOM_60D": FactorMetadata(
                name="TSMOM_60D",
                description="60日时间序列动量",
                dimension="趋势/动量",
                required_columns=["close"],
                window=60,
                bounded=False,
                direction="high_is_good",
                orthogonal_v1=False,  # corr=0.87 CALMAR, IC 不显著
            ),
            "TSMOM_120D": FactorMetadata(
                name="TSMOM_120D",
                description="120日时间序列动量",
                dimension="趋势/动量",
                required_columns=["close"],
                window=120,
                bounded=False,
                direction="high_is_good",
                orthogonal_v1=False,  # IC 不显著, 0/4 候选出现
            ),
            "BREAKOUT_20D": FactorMetadata(
                name="BREAKOUT_20D",
                description="20日突破信号",
                dimension="趋势/动量",
                required_columns=["high", "close"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            "TURNOVER_ACCEL_5_20": FactorMetadata(
                name="TURNOVER_ACCEL_5_20",
                description="5日vs20日换手率加速度",
                dimension="量能/流动性",
                required_columns=["volume"],
                window=20,
                bounded=False,
                direction="high_is_good",
                production_ready=False,  # ❌ 无效因子: LS_Sharpe=-0.56, 评分0
                orthogonal_v1=False,  # IC 不显著, Top2 仅+1%
            ),
            "REALIZED_VOL_20D": FactorMetadata(
                name="REALIZED_VOL_20D",
                description="20日实际波动率",
                dimension="波动/风险",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                orthogonal_v1=False,  # ≡ RET_VOL_20D (corr=1.000, 仅差 √252 缩放)
            ),
            "AMIHUD_ILLIQUIDITY": FactorMetadata(
                name="AMIHUD_ILLIQUIDITY",
                description="Amihud流动性指标（冲击成本代理）",
                dimension="流动性/成本",
                required_columns=["close", "volume"],
                window=20,
                bounded=False,
                direction="low_is_good",
            ),
            "SPREAD_PROXY": FactorMetadata(
                name="SPREAD_PROXY",
                description="日内价差代理（交易成本）",
                dimension="流动性/成本",
                required_columns=["high", "low", "close"],
                window=5,
                bounded=False,
                direction="low_is_good",
            ),
            # ============ [v4.2] 因子扩展研究：15个新因子候选 ============
            # P0: 高阶矩 — 收益分布形状
            "SKEW_20D": FactorMetadata(
                name="SKEW_20D",
                description="20日收益偏度",
                dimension="高阶矩",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            "KURT_20D": FactorMetadata(
                name="KURT_20D",
                description="20日收益超额峰度",
                dimension="高阶矩",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            # P0: 动量质量 — Frog in the Pan
            "INFO_DISCRETE_20D": FactorMetadata(
                name="INFO_DISCRETE_20D",
                description="20日信息离散度（动量质量）",
                dimension="动量质量",
                required_columns=["close"],
                window=20,
                bounded=True,  # [-1, 1]
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            # P1: 均值回复
            "IBS": FactorMetadata(
                name="IBS",
                description="内部柱强度（单日均值回复）",
                dimension="均值回复",
                required_columns=["close", "high", "low"],
                window=1,
                bounded=True,  # [0, 1]
                direction="high_is_good",
                orthogonal_v1=False,
            ),
            "MEAN_REV_RATIO_20D": FactorMetadata(
                name="MEAN_REV_RATIO_20D",
                description="20日均值回复比率",
                dimension="均值回复",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                production_ready=False,  # ❌ 无效因子: LS_Sharpe=-0.29, 评分0
                orthogonal_v1=False,
            ),
            # P1: 量能方向性
            "UP_DOWN_VOL_RATIO_20D": FactorMetadata(
                name="UP_DOWN_VOL_RATIO_20D",
                description="20日上涨/下跌日成交量比率",
                dimension="量能方向性",
                required_columns=["close", "volume"],
                window=20,
                bounded=False,
                direction="high_is_good",
                orthogonal_v1=False,
            ),
            "ABNORMAL_VOLUME_20D": FactorMetadata(
                name="ABNORMAL_VOLUME_20D",
                description="异常成交量（5日/60日比率）",
                dimension="量能方向性",
                required_columns=["volume"],
                window=60,
                bounded=False,
                direction="neutral",
                production_ready=False,  # ❌ 无效因子: LS_Sharpe=-0.10, 评分0.5
                orthogonal_v1=False,
            ),
            # P1: 回撤恢复
            "ULCER_INDEX_20D": FactorMetadata(
                name="ULCER_INDEX_20D",
                description="20日 Ulcer Index（回撤深度×持续时间）",
                dimension="回撤恢复",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            "DD_DURATION_60D": FactorMetadata(
                name="DD_DURATION_60D",
                description="60日回撤持续天数比率",
                dimension="回撤恢复",
                required_columns=["close"],
                window=60,
                bounded=True,  # [0, 1]
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            # P2: 波动率微结构
            "GK_VOL_RATIO_20D": FactorMetadata(
                name="GK_VOL_RATIO_20D",
                description="20日 Garman-Klass/CC 波动率比率",
                dimension="波动率微结构",
                required_columns=["open", "high", "low", "close"],
                window=20,
                bounded=False,
                direction="neutral",
                orthogonal_v1=False,
            ),
            # P2: 复杂度/熵
            "PERM_ENTROPY_20D": FactorMetadata(
                name="PERM_ENTROPY_20D",
                description="20日排列熵（序列复杂度）",
                dimension="复杂度/熵",
                required_columns=["close"],
                window=20,
                bounded=True,  # [0, 1]
                direction="low_is_good",
                production_ready=False,  # ❌ 无效因子: LS_Sharpe=-0.19, 评分0
                orthogonal_v1=False,
            ),
            # P3: Hurst 指数
            "HURST_60D": FactorMetadata(
                name="HURST_60D",
                description="60日 Hurst 指数（趋势/均值回复体制）",
                dimension="长期记忆",
                required_columns=["close"],
                window=60,
                bounded=True,  # [0, 1]
                direction="neutral",
                production_ready=False,  # ❌ 无效因子: LS_Sharpe=-0.11, 评分0.5
                orthogonal_v1=False,
            ),
            # P3: 下行偏差
            "DOWNSIDE_DEV_20D": FactorMetadata(
                name="DOWNSIDE_DEV_20D",
                description="20日下行偏差（非对称风险）",
                dimension="非对称风险",
                required_columns=["close"],
                window=20,
                bounded=False,
                direction="low_is_good",
                orthogonal_v1=False,
            ),
            # ── 非OHLCV因子: 资金流向 (fund_share) ──
            "SHARE_CHG_5D": FactorMetadata(
                name="SHARE_CHG_5D",
                description="5日份额变化率（反向: 份额减少→收益高）",
                dimension="资金流向",
                required_columns=["fund_share"],
                window=5,
                bounded=False,
                direction="low_is_good",
            ),
            "SHARE_CHG_10D": FactorMetadata(
                name="SHARE_CHG_10D",
                description="10日份额变化率（反向: 份额减少→收益高）",
                dimension="资金流向",
                required_columns=["fund_share"],
                window=10,
                bounded=False,
                direction="low_is_good",
            ),
            "SHARE_CHG_20D": FactorMetadata(
                name="SHARE_CHG_20D",
                description="20日份额变化率（反向: 份额减少→收益高）",
                dimension="资金流向",
                required_columns=["fund_share"],
                window=20,
                bounded=False,
                direction="low_is_good",
            ),
            "SHARE_ACCEL": FactorMetadata(
                name="SHARE_ACCEL",
                description="份额变化加速度（短期变化-长期变化）",
                dimension="资金流向",
                required_columns=["fund_share"],
                window=20,
                bounded=False,
                direction="high_is_good",
            ),
            # ── 非OHLCV因子: 杠杆行为 (margin) ──
            "MARGIN_CHG_10D": FactorMetadata(
                name="MARGIN_CHG_10D",
                description="10日融资余额变化率（反向: 融资减少→收益高）",
                dimension="杠杆行为",
                required_columns=["margin_rzye"],
                window=10,
                bounded=False,
                direction="low_is_good",
            ),
            "MARGIN_BUY_RATIO": FactorMetadata(
                name="MARGIN_BUY_RATIO",
                description="融资买入占比（反向: 融资买入少→收益高）",
                dimension="杠杆行为",
                required_columns=["margin_rzmre", "close", "volume"],
                window=1,
                bounded=False,
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

        # 直接对 2D 数组沿 axis=0 应用 lfilter（避免 apply_along_axis 开销）
        result = lfilter(weights, [1.0], close_df.values, axis=0) / denom
        result[:19, :] = np.nan
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _max_dd_60d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 MAX_DD_60D（Numba prange 并行逐列）"""
        result = _rolling_max_dd_batch(close_df.values, window=60)
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _calmar_60d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 CALMAR_60D（Numba prange 并行逐列）"""
        result = _rolling_calmar_batch(close_df.values, window=60)
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

        # 直接对 2D 数组沿 axis=0 应用 lfilter（避免 apply_along_axis 开销）
        result = lfilter(weights, [1.0], obv_vals, axis=0) / denom
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
        etf_returns = close_df.pct_change()
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
    # v4.2 批量方法：15个新因子
    # =========================================================================

    def _ulcer_index_20d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 ULCER_INDEX_20D（Numba prange 并行逐列）"""
        result = _rolling_ulcer_index_batch(close_df.values, window=20)
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _dd_duration_60d_batch(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 DD_DURATION_60D（Numba prange 并行逐列）"""
        result = _dd_duration_batch(close_df.values, window=60)
        return pd.DataFrame(result, index=close_df.index, columns=close_df.columns)

    def _perm_entropy_20d_batch(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 PERM_ENTROPY_20D（Numba prange 并行逐列）"""
        result = _permutation_entropy_batch(returns_df.values, window=20, m=3, delay=1)
        return pd.DataFrame(result, index=returns_df.index, columns=returns_df.columns)

    def _hurst_60d_batch(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """批量计算 HURST_60D（Numba prange 并行逐列）"""
        result = _hurst_rs_batch(returns_df.values, window=60)
        return pd.DataFrame(result, index=returns_df.index, columns=returns_df.columns)

    def _gk_vol_ratio_20d_batch(
        self,
        open_df: pd.DataFrame,
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量计算 GK_VOL_RATIO_20D（Garman-Klass / Close-to-Close 波动率比率）"""
        eps = 1e-10

        # Garman-Klass 日内方差: 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2
        log_hl = np.log(high_df / (low_df + eps))
        log_co = np.log(close_df / (open_df + eps))
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

        # Close-to-Close 方差
        returns = close_df.pct_change()
        cc_var = returns**2

        # 20日滚动均值
        gk_mean = gk_var.rolling(window=20, min_periods=20).mean()
        cc_mean = cc_var.rolling(window=20, min_periods=20).mean()

        # 比率 (sqrt 使单位一致)
        ratio = np.sqrt(gk_mean.abs() / (cc_mean + eps))
        ratio = ratio.where(cc_mean >= eps, np.nan)

        return ratio

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
        ret = close.pct_change() * 100  # 转为百分比
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
        returns = close.pct_change()

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
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

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
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

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
        etf_returns = close.pct_change()

        # 计算市场收益率（所有ETF等权平均）
        market_returns = market_close.pct_change().mean(axis=1)

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
        etf_returns = close.pct_change()

        # 计算市场收益率（所有ETF等权平均）
        market_returns = market_close.pct_change().mean(axis=1)

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
        returns = close.pct_change()
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
        returns = close.pct_change().abs()

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
        ret = close.pct_change() * 100
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
        ret_price = close.pct_change()
        ret_volume = volume.pct_change()
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
        returns = close.pct_change()
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
        etf_returns = close.pct_change()
        market_returns = etf_returns.mean(axis=1)
        correlation_to_market_20d = etf_returns.rolling(window=20, min_periods=20).corr(
            market_returns
        )

        # ========== v4.0: 新增7个因子 ==========

        # 维度11：时间序列动量
        tsmom_60d = (close / close.shift(60) - 1) * 100
        tsmom_120d = (close / close.shift(120) - 1) * 100

        # 维度12：突破
        rolling_high_20 = high.rolling(window=20, min_periods=20).max()
        breakout_20d = (close / rolling_high_20 - 1) * 100

        # 维度13：换手率加速
        vol_ma5 = volume.rolling(window=5, min_periods=5).mean()
        vol_ma20 = volume.rolling(window=20, min_periods=20).mean()
        turnover_accel_5_20 = (vol_ma5 / (vol_ma20 + eps)).where(vol_ma20 >= eps, np.nan)

        # 维度14：实际波动率
        realized_vol_20d = returns.rolling(window=20, min_periods=20).std() * np.sqrt(252)

        # 维度15：Amihud 非流动性
        abs_ret = returns.abs()
        volume_yuan = volume * close  # 成交额近似
        amihud_daily = abs_ret / (volume_yuan + eps)
        amihud_illiquidity = amihud_daily.rolling(window=20, min_periods=20).mean() * 1e6

        # 维度16：日内价差代理
        spread_proxy = ((high - low) / (close + eps)).rolling(window=5, min_periods=5).mean() * 100

        # ========== v4.2: 15个新因子候选 ==========

        # 高阶矩
        skew_20d = returns.rolling(window=20, min_periods=20).skew()
        kurt_20d = returns.rolling(window=20, min_periods=20).kurt()

        # 动量质量 (Frog in the Pan)
        sign_mom = np.sign(mom_20d)
        pos_days = (returns > 0).rolling(window=20, min_periods=20).sum() / 20
        neg_days = (returns < 0).rolling(window=20, min_periods=20).sum() / 20
        info_discrete_20d = sign_mom * (neg_days - pos_days)

        # 均值回复
        ibs = (close - low) / (high - low + eps)
        ibs = ibs.where((high - low).abs() > eps, 0.5)
        ibs = ibs.clip(0, 1)

        mean_rev_ratio_20d = (close / close.rolling(window=20, min_periods=20).mean() - 1) * 100

        # 量能方向性
        up_vol = volume.where(returns > 0, 0.0).rolling(window=20, min_periods=20).sum()
        dn_vol = volume.where(returns <= 0, 0.0).rolling(window=20, min_periods=20).sum()
        up_down_vol_ratio_20d = up_vol / (dn_vol + eps)
        up_down_vol_ratio_20d = up_down_vol_ratio_20d.where(dn_vol >= eps, np.nan)

        vol_ma5_abn = volume.rolling(window=5, min_periods=5).mean()
        vol_ma60_abn = volume.rolling(window=60, min_periods=60).mean()
        abnormal_volume_20d = (vol_ma5_abn / (vol_ma60_abn + eps) - 1) * 100
        abnormal_volume_20d = abnormal_volume_20d.where(vol_ma60_abn >= eps, np.nan)

        # 回撤恢复
        ulcer_index_20d = self._ulcer_index_20d_batch(close)
        dd_duration_60d = self._dd_duration_60d_batch(close)

        # 波动率微结构
        open_df = prices.get("open", close)  # fallback to close if no open
        gk_vol_ratio_20d = self._gk_vol_ratio_20d_batch(open_df, high, low, close)

        # 复杂度/熵
        perm_entropy_20d = self._perm_entropy_20d_batch(returns)

        # Hurst 指数
        hurst_60d = self._hurst_60d_batch(returns)

        # 下行偏差
        neg_ret = returns.clip(upper=0)
        downside_dev_20d = np.sqrt((neg_ret**2).rolling(window=20, min_periods=20).mean()) * np.sqrt(252) * 100

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
            # v4.0: 新增7个因子
            "TSMOM_60D": tsmom_60d,
            "TSMOM_120D": tsmom_120d,
            "BREAKOUT_20D": breakout_20d,
            "TURNOVER_ACCEL_5_20": turnover_accel_5_20,
            "REALIZED_VOL_20D": realized_vol_20d,
            "AMIHUD_ILLIQUIDITY": amihud_illiquidity,
            "SPREAD_PROXY": spread_proxy,
            # v4.2: 15个新因子候选
            "SKEW_20D": skew_20d,
            "KURT_20D": kurt_20d,
            "INFO_DISCRETE_20D": info_discrete_20d,
            "IBS": ibs,
            "MEAN_REV_RATIO_20D": mean_rev_ratio_20d,
            "UP_DOWN_VOL_RATIO_20D": up_down_vol_ratio_20d,
            "ABNORMAL_VOLUME_20D": abnormal_volume_20d,
            "ULCER_INDEX_20D": ulcer_index_20d,
            "DD_DURATION_60D": dd_duration_60d,
            "GK_VOL_RATIO_20D": gk_vol_ratio_20d,
            "PERM_ENTROPY_20D": perm_entropy_20d,
            "HURST_60D": hurst_60d,
            "DOWNSIDE_DEV_20D": downside_dev_20d,
        }

        # 一次性拼接：columns=(factor, symbol)
        result = pd.concat(factor_dfs, axis=1, keys=factor_dfs.keys())
        result = result.sort_index(axis=1)

        logger.info(
            f"✅ 计算完成: {len(symbols)}个标的 × {len(self.factors_metadata)}个因子"
        )

        return result

    def compute_non_ohlcv_factors(
        self,
        prices: Dict[str, pd.DataFrame],
        fund_share_panel: Optional[pd.DataFrame] = None,
        margin_panels: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute non-OHLCV factors from fund_share and margin data.

        Args:
            prices: OHLCV price dict (needs 'close' and 'volume' for MARGIN_BUY_RATIO)
            fund_share_panel: DatetimeIndex x ETF codes, fund share data (亿份)
            margin_panels: {'rzye': DataFrame, 'rzmre': DataFrame} margin data

        Returns:
            Dict of {factor_name: pd.DataFrame} with same index/columns as inputs.
            Only includes factors whose input data is available.
        """
        result: Dict[str, pd.DataFrame] = {}
        eps = 1e-10

        # ── Fund share factors ──
        if fund_share_panel is not None and not fund_share_panel.empty:
            fd = fund_share_panel

            share_chg_5d = (fd - fd.shift(5)) / (fd.shift(5) + eps)
            share_chg_5d = share_chg_5d.where(fd.shift(5).abs() > eps, np.nan)
            result["SHARE_CHG_5D"] = share_chg_5d

            share_chg_10d = (fd - fd.shift(10)) / (fd.shift(10) + eps)
            share_chg_10d = share_chg_10d.where(fd.shift(10).abs() > eps, np.nan)
            result["SHARE_CHG_10D"] = share_chg_10d

            share_chg_20d = (fd - fd.shift(20)) / (fd.shift(20) + eps)
            share_chg_20d = share_chg_20d.where(fd.shift(20).abs() > eps, np.nan)
            result["SHARE_CHG_20D"] = share_chg_20d

            result["SHARE_ACCEL"] = share_chg_5d - share_chg_20d

            logger.info(
                f"Computed 4 fund_share factors: "
                f"{fd.shape[1]} ETFs, {fd.notna().any(axis=1).sum()} trading days"
            )

        # ── Margin factors ──
        if margin_panels is not None:
            rzye = margin_panels.get("rzye")
            rzmre = margin_panels.get("rzmre")

            if rzye is not None and not rzye.empty:
                margin_chg_10d = (rzye - rzye.shift(10)) / (rzye.shift(10) + eps)
                margin_chg_10d = margin_chg_10d.where(rzye.shift(10).abs() > eps, np.nan)
                result["MARGIN_CHG_10D"] = margin_chg_10d

                logger.info(
                    f"Computed MARGIN_CHG_10D: "
                    f"{rzye.shape[1]} ETFs, {rzye.notna().any(axis=1).sum()} trading days"
                )

            if rzmre is not None and not rzmre.empty:
                close = prices.get("close")
                volume = prices.get("volume")
                if close is not None and volume is not None:
                    turnover = close * volume
                    # Align rzmre to turnover index/columns
                    common_cols = sorted(set(rzmre.columns) & set(turnover.columns))
                    common_idx = rzmre.index.intersection(turnover.index)
                    if common_cols and len(common_idx) > 0:
                        rzmre_aligned = rzmre.loc[common_idx, common_cols]
                        turnover_aligned = turnover.loc[common_idx, common_cols]
                        margin_buy_ratio = rzmre_aligned / (turnover_aligned + eps)
                        margin_buy_ratio = margin_buy_ratio.where(
                            turnover_aligned.abs() > eps, np.nan
                        )
                        result["MARGIN_BUY_RATIO"] = margin_buy_ratio

                        logger.info(
                            f"Computed MARGIN_BUY_RATIO: "
                            f"{len(common_cols)} ETFs, {len(common_idx)} trading days"
                        )

        return result

    def get_metadata(self, factor_name: str) -> Optional[FactorMetadata]:
        """获取因子元数据"""
        return self.factors_metadata.get(factor_name)

    def list_factors(self) -> Dict[str, FactorMetadata]:
        """列出所有因子及其元数据"""
        return self.factors_metadata

    def list_production_factors(self) -> Dict[str, FactorMetadata]:
        """列出所有可用于生产的因子 (production_ready=True)"""
        return {
            name: meta
            for name, meta in self.factors_metadata.items()
            if meta.production_ready
        }

    def list_risky_factors(self) -> Dict[str, FactorMetadata]:
        """列出所有高风险因子 (production_ready=False)"""
        return {
            name: meta
            for name, meta in self.factors_metadata.items()
            if not meta.production_ready
        }

    def is_combo_production_ready(self, combo: str) -> tuple[bool, list[str]]:
        """检查因子组合是否可用于生产

        参数:
            combo: 因子组合字符串，如 "ADX_14D + PRICE_POSITION_20D"

        返回:
            (is_ready, risky_factors): 是否可用, 包含的高风险因子列表
        """
        factors = [f.strip() for f in combo.split(" + ")]
        risky = []
        for f in factors:
            meta = self.factors_metadata.get(f)
            if meta and not meta.production_ready:
                risky.append(f)
        return len(risky) == 0, risky


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
