"""
Factor Registry | 因子元数据注册中心

单一事实源 (Single Source of Truth) — 所有因子的元数据定义。
消除 CrossSectionProcessor / frozen_params / config YAML 三处同步问题。

用法:
    from etf_strategy.core.factor_registry import (
        FACTOR_SPECS,
        get_bounded_factors,
        get_factor_bounds,
        get_factor_direction,
    )

作者: Sensen
日期: 2026-02-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple


@dataclass(frozen=True)
class FactorSpec:
    """因子元数据规格"""

    name: str
    source: str  # 'ohlcv' | 'fund_share' | 'margin' | 'premium' | 'fx'
    is_bounded: bool = False
    bounds: Optional[Tuple[float, float]] = None  # 有界因子的值域
    description: str = ""
    direction: str = "high_is_good"  # 'high_is_good' | 'low_is_good' | 'neutral'


# ===========================================================================
# 单一事实源 — 所有因子元数据
# ===========================================================================
#
# 新增因子只需在此处注册，下游自动派生:
#   - CrossSectionProcessor.BOUNDED_FACTORS
#   - FrozenCrossSectionParams.bounded_factors
#   - FactorCache bounded_factors cache key
#
# 注意: 此处只定义元数据，不包含计算逻辑。
#   - OHLCV 因子由 PreciseFactorLibrary 批量计算
#   - 非 OHLCV 因子由 non_ohlcv_factors.py 计算

FACTOR_SPECS: Dict[str, FactorSpec] = {
    # =======================================================================
    # OHLCV-derived factors (34 registered, 18 active in production)
    # Data source: raw/ETF/daily/
    # =======================================================================
    #
    # --- Bucket A: 趋势/动量 (TREND_MOMENTUM) ---
    "MOM_20D": FactorSpec("MOM_20D", "ohlcv", description="20日动量百分比"),
    "SLOPE_20D": FactorSpec("SLOPE_20D", "ohlcv", description="20日线性回归斜率"),
    "SHARPE_RATIO_20D": FactorSpec(
        "SHARPE_RATIO_20D", "ohlcv", description="20日夏普比率"
    ),
    "BREAKOUT_20D": FactorSpec("BREAKOUT_20D", "ohlcv", description="20日突破信号"),
    "VORTEX_14D": FactorSpec("VORTEX_14D", "ohlcv", description="14日涡旋指标",
                             direction="neutral"),
    "PRICE_POSITION_20D": FactorSpec(
        "PRICE_POSITION_20D",
        "ohlcv",
        is_bounded=True,
        bounds=(0.0, 1.0),
        description="20日价格位置",
        direction="neutral",
    ),
    #
    # --- Bucket B: 持续位置 (SUSTAINED_POSITION) ---
    "PRICE_POSITION_120D": FactorSpec(
        "PRICE_POSITION_120D",
        "ohlcv",
        is_bounded=True,
        bounds=(0.0, 1.0),
        description="120日价格位置",
        direction="neutral",
    ),
    "CALMAR_RATIO_60D": FactorSpec(
        "CALMAR_RATIO_60D", "ohlcv", description="60日卡玛比率"
    ),
    #
    # --- Bucket C: 量能确认 (VOLUME_CONFIRMATION) ---
    "OBV_SLOPE_10D": FactorSpec("OBV_SLOPE_10D", "ohlcv", description="OBV斜率"),
    "UP_DOWN_VOL_RATIO_20D": FactorSpec(
        "UP_DOWN_VOL_RATIO_20D", "ohlcv", description="上涨量/下跌量比"
    ),
    "CMF_20D": FactorSpec(
        "CMF_20D",
        "ohlcv",
        is_bounded=True,
        bounds=(-1.0, 1.0),
        description="Chaikin资金流",
    ),
    #
    # --- Bucket D: 微观结构 (MICROSTRUCTURE) ---
    "PV_CORR_20D": FactorSpec(
        "PV_CORR_20D",
        "ohlcv",
        is_bounded=True,
        bounds=(-1.0, 1.0),
        description="价量相关性",
        direction="high_is_good",
    ),
    "AMIHUD_ILLIQUIDITY": FactorSpec(
        "AMIHUD_ILLIQUIDITY", "ohlcv", description="Amihud非流动性",
        direction="low_is_good",
    ),
    "GK_VOL_RATIO_20D": FactorSpec(
        "GK_VOL_RATIO_20D", "ohlcv", description="GK波动率比",
        direction="neutral",
    ),
    #
    # --- Bucket E: 趋势强度/风险 (TREND_STRENGTH_RISK) ---
    "ADX_14D": FactorSpec(
        "ADX_14D",
        "ohlcv",
        is_bounded=True,
        bounds=(0.0, 100.0),
        description="趋势强度",
    ),
    "CORRELATION_TO_MARKET_20D": FactorSpec(
        "CORRELATION_TO_MARKET_20D",
        "ohlcv",
        is_bounded=True,
        bounds=(-1.0, 1.0),
        description="市场相关性",
        direction="low_is_good",
    ),
    "MAX_DD_60D": FactorSpec("MAX_DD_60D", "ohlcv", description="60日最大回撤",
                             direction="low_is_good"),
    "VOL_RATIO_20D": FactorSpec("VOL_RATIO_20D", "ohlcv", description="波动率比率"),
    #
    # --- 未入桶 (registered but not active) ---
    "RET_VOL_20D": FactorSpec("RET_VOL_20D", "ohlcv", description="20日收益波动率",
                              direction="low_is_good"),
    "VOL_RATIO_60D": FactorSpec("VOL_RATIO_60D", "ohlcv", description="60日成交量比率"),
    "RSI_14": FactorSpec(
        "RSI_14",
        "ohlcv",
        is_bounded=True,
        bounds=(0.0, 100.0),
        description="14日相对强度",
        direction="neutral",
    ),
    "RELATIVE_STRENGTH_VS_MARKET_20D": FactorSpec(
        "RELATIVE_STRENGTH_VS_MARKET_20D", "ohlcv", description="相对市场强度"
    ),
    "TSMOM_60D": FactorSpec("TSMOM_60D", "ohlcv", description="60日时序动量"),
    "TSMOM_120D": FactorSpec("TSMOM_120D", "ohlcv", description="120日时序动量"),
    "TURNOVER_ACCEL_5_20": FactorSpec(
        "TURNOVER_ACCEL_5_20", "ohlcv", description="换手率加速度"
    ),
    "REALIZED_VOL_20D": FactorSpec(
        "REALIZED_VOL_20D", "ohlcv", description="20日已实现波动率",
        direction="low_is_good",
    ),
    "SPREAD_PROXY": FactorSpec("SPREAD_PROXY", "ohlcv", description="价差代理",
                               direction="low_is_good"),
    "SKEW_20D": FactorSpec("SKEW_20D", "ohlcv", description="20日偏度",
                           direction="low_is_good"),
    "KURT_20D": FactorSpec("KURT_20D", "ohlcv", description="20日峰度",
                           direction="low_is_good"),
    "INFO_DISCRETE_20D": FactorSpec(
        "INFO_DISCRETE_20D", "ohlcv", description="信息离散度",
        direction="low_is_good",
    ),
    "IBS": FactorSpec("IBS", "ohlcv", description="日内偏差指标"),
    "MEAN_REV_RATIO_20D": FactorSpec(
        "MEAN_REV_RATIO_20D", "ohlcv", description="均值回归比率",
        direction="low_is_good",
    ),
    "ABNORMAL_VOLUME_20D": FactorSpec(
        "ABNORMAL_VOLUME_20D", "ohlcv", description="异常成交量",
        direction="neutral",
    ),
    "ULCER_INDEX_20D": FactorSpec("ULCER_INDEX_20D", "ohlcv", description="溃疡指数",
                                  direction="low_is_good"),
    "DD_DURATION_60D": FactorSpec("DD_DURATION_60D", "ohlcv", description="回撤持续天数",
                                  direction="low_is_good"),
    "PERM_ENTROPY_20D": FactorSpec(
        "PERM_ENTROPY_20D", "ohlcv", description="排列熵",
        direction="low_is_good",
    ),
    "HURST_60D": FactorSpec("HURST_60D", "ohlcv", description="Hurst指数",
                            direction="neutral"),
    "DOWNSIDE_DEV_20D": FactorSpec(
        "DOWNSIDE_DEV_20D", "ohlcv", description="下行偏差",
        direction="low_is_good",
    ),
    #
    # =======================================================================
    # Non-OHLCV factors (6 registered, 6 active)
    # Data source: raw/ETF/fund_share/, raw/ETF/margin/
    # =======================================================================
    #
    # --- Bucket F: 资金流向 (fund_share) ---
    "SHARE_CHG_5D": FactorSpec(
        "SHARE_CHG_5D", "fund_share", description="5日份额变化率",
        direction="low_is_good",
    ),
    "SHARE_CHG_10D": FactorSpec(
        "SHARE_CHG_10D", "fund_share", description="10日份额变化率",
        direction="low_is_good",
    ),
    "SHARE_CHG_20D": FactorSpec(
        "SHARE_CHG_20D", "fund_share", description="20日份额变化率",
        direction="low_is_good",
    ),
    "SHARE_ACCEL": FactorSpec(
        "SHARE_ACCEL", "fund_share", description="份额变化加速度"
    ),
    #
    # --- Bucket G: 杠杆行为 (margin) ---
    "MARGIN_CHG_10D": FactorSpec(
        "MARGIN_CHG_10D", "margin", description="10日融资余额变化",
        direction="low_is_good",
    ),
    "MARGIN_BUY_RATIO": FactorSpec(
        "MARGIN_BUY_RATIO", "margin", description="融资买入占比",
        direction="low_is_good",
    ),
}


# ===========================================================================
# 派生函数 — 消除三处同步
# ===========================================================================


def get_bounded_factors() -> Set[str]:
    """返回所有有界因子名称集合"""
    return {name for name, spec in FACTOR_SPECS.items() if spec.is_bounded}


def get_factor_bounds() -> Dict[str, Tuple[float, float]]:
    """返回有界因子的值域映射"""
    return {
        name: spec.bounds
        for name, spec in FACTOR_SPECS.items()
        if spec.is_bounded and spec.bounds is not None
    }


def get_bounded_factors_tuple() -> Tuple[str, ...]:
    """返回有界因子名称排序元组 (用于 FrozenCrossSectionParams)"""
    return tuple(sorted(get_bounded_factors()))


def get_non_ohlcv_factor_names() -> list:
    """返回所有非 OHLCV 因子名称列表"""
    return [name for name, spec in FACTOR_SPECS.items() if spec.source != "ohlcv"]


def get_factor_source(factor_name: str) -> Optional[str]:
    """返回因子的数据源类型"""
    spec = FACTOR_SPECS.get(factor_name)
    return spec.source if spec else None


def get_factor_direction(factor_name: str) -> str:
    """返回因子方向: 'high_is_good' | 'low_is_good' | 'neutral'

    未注册因子默认返回 'high_is_good' (向后兼容)。
    """
    spec = FACTOR_SPECS.get(factor_name)
    return spec.direction if spec else "high_is_good"
