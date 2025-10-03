#!/usr/bin/env python3
"""
增强版多时间框架因子计算器 - 基于154个技术指标
整合vbt_professional_detector.py的成熟计算逻辑到多时间框架系统
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vectorbt as vbt

# 导入配置
from config import get_config, setup_logging

# 简化的时间框架定义


def extract_vbt_labels(vbt_result):
    """从VectorBT标签生成器对象中提取可序列化的标签数据"""
    try:
        if hasattr(vbt_result, "labels"):
            return vbt_result.labels.values
        elif hasattr(vbt_result, "values"):
            return vbt_result.values
        else:
            return pd.Series(vbt_result).values
    except Exception as e:
        logging.warning(f"提取VectorBT标签失败: {e}")
        return None


def extract_indicator_component(result, attr_candidates):
    """从指标结果对象中提取第一个可用的属性值"""
    for attr in attr_candidates:
        if hasattr(result, attr):
            return getattr(result, attr)

    if hasattr(result, "_asdict"):
        result_dict = result._asdict()
        for attr in attr_candidates:
            if attr in result_dict:
                return result_dict[attr]

    return None


# 统一的数据契约映射表 - Linus式设计
INDICATOR_DATA_REQUIREMENTS = {
    # 基础移动平均线类
    "SMA": ["close"],
    "EMA": ["close"],
    "WMA": ["close"],
    "DEMA": ["close"],
    "TEMA": ["close"],
    "TRIMA": ["close"],
    "KAMA": ["close"],
    "T3": ["close"],
    # 动量指标类
    "MOM": ["close"],
    "ROC": ["close"],
    "RSI": ["close"],
    "TRIX": ["close"],
    # 波动率指标类
    "STDDEV": ["close"],
    "VAR": ["close"],
    "ATR": ["high", "low", "close"],
    "NATR": ["high", "low", "close"],
    # 成交量指标类
    "AD": ["high", "low", "close", "volume"],
    "ADOSC": ["high", "low", "close", "volume"],
    "OBV": ["close", "volume"],
    # 趋势指标类
    "MACD": ["close"],
    "MACDEXT": ["close"],
    "MACDFIX": ["close"],
    "APO": ["close"],
    "PPO": ["close"],
    # 布林带类
    "BBANDS": ["close"],
    "MIDPOINT": ["high", "low"],
    "MIDPRICE": ["high", "low"],
    "SAR": ["high", "low"],
    "SAREXT": ["high", "low"],
    # 随机指标类
    "STOCH": ["high", "low", "close"],
    "STOCHF": ["high", "low", "close"],
    "STOCHRSI": ["close"],
    # 威廉指标类
    "WILLR": ["high", "low", "close"],
    # 商品通道指标类
    "CCI": ["high", "low", "close"],
    # 平均方向指数类
    "ADX": ["high", "low", "close"],
    "ADXR": ["high", "low", "close"],
    "DI": ["high", "low", "close"],
    "DX": ["high", "low", "close"],
    "MINUS_DI": ["high", "low", "close"],
    "PLUS_DI": ["high", "low", "close"],
    # 资金流量指标类
    "MFI": ["high", "low", "close", "volume"],
    # 阿隆指标类
    "AROON": ["high", "low"],
    "AROONOSC": ["high", "low"],
    # 蜡烛图模式类
    "CDL": ["open", "high", "low", "close"],  # 所有CDL指标
}


def validate_data_requirements(df, required_columns):
    """验证数据是否包含必需的列 - Linus式统一验证"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"缺少必需数据列: {missing_columns}"
    return True, "数据完整"


def prepare_indicator_data(df, required_columns):
    """准备指标计算所需的数据 - 统一数据提取接口"""
    # 验证数据完整性
    is_valid, message = validate_data_requirements(df, required_columns)
    if not is_valid:
        return None, message

    # 提取所需数据，统一转换为numpy数组
    data = {}
    for col in required_columns:
        if col in df.columns:
            data[col] = df[col].values

    return data, "数据准备完成"


def calculate_indicator_with_validation(
    talib_indicator, output_name, talib_name, df, *args, **kwargs
):
    """带数据验证的统一指标计算接口 - 消除特殊情况"""
    # 确定数据需求
    data_requirements = INDICATOR_DATA_REQUIREMENTS.get(talib_name, ["close"])

    # 准备数据
    data, message = prepare_indicator_data(df, data_requirements)
    if data is None:
        return None, message

    try:
        # 统一参数处理 - 消除特殊情况
        if talib_name in ["SAR", "SAREXT"]:
            # 抛物线SAR不需要timeperiod参数
            result = talib_indicator.run(data["high"], data["low"])
        elif talib_name in ["OBV"]:
            # OBV不需要timeperiod参数
            result = talib_indicator.run(data["close"], data["volume"])
        else:
            # 大部分指标都支持timeperiod参数
            result = talib_indicator.run(
                **kwargs,
                **{
                    "high": data["high"] if "high" in data else data["close"],
                    "low": data["low"] if "low" in data else data["close"],
                    "close": data["close"],
                    "open": data["open"] if "open" in data else data["close"],
                    "volume": data["volume"] if "volume" in data else None,
                },
            )

        return result, "计算成功"

    except Exception as e:
        return None, f"计算失败: {e}"


def ensure_series(values, index, name):
    """确保将输出转换为带名称的Series"""
    if isinstance(values, pd.Series):
        series = values.copy()
        series.name = name
        return series

    if hasattr(values, "to_pd"):
        converted = values.to_pd()
        if isinstance(converted, pd.Series):
            return converted.rename(name)
        if isinstance(converted, pd.DataFrame) and converted.shape[1] == 1:
            return converted.iloc[:, 0].rename(name)

    if hasattr(values, "to_series"):
        try:
            series = values.to_series()
            series.name = name
            return series
        except Exception:
            pass

    try:
        array = np.asarray(values)
    except Exception:
        array = np.asarray(pd.Series(values))

    if array.ndim > 1:
        array = array.reshape(array.shape[0])

    return pd.Series(array, index=index, name=name)


def extract_vbt_indicator(vbt_result):
    """从VectorBT指标对象中提取可序列化的数值数据"""
    try:
        if hasattr(vbt_result, "values"):
            return vbt_result.values
        if hasattr(vbt_result, "labels"):
            return vbt_result.labels.values
        if hasattr(vbt_result, "percent_k") and hasattr(vbt_result, "percent_d"):
            # 处理VectorBT原生STOCH指标
            return vbt_result.percent_k
        component = extract_indicator_component(vbt_result, ["k", "fastk", "slowk"])
        if component is not None:
            return component
        return vbt_result
    except Exception as e:
        logging.warning(f"提取VectorBT指标失败: {e}")
        return None


class TimeFrame(Enum):
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    MIN_60 = "60min"
    DAILY = "daily"


# 策略相关数据结构（从vbt_professional_detector.py迁移）
class ScreenOperator(Enum):
    """筛选操作符"""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    BETWEEN = "BETWEEN"
    TOP_N = "TOP_N"
    BOTTOM_N = "BOTTOM_N"


@dataclass
class ScreenCriteria:
    """筛选条件"""

    factor_name: str
    operator: ScreenOperator
    threshold: float
    weight: float = 1.0


@dataclass
class StrategyResult:
    """策略结果"""

    name: str
    selected_stocks: List[str]
    scores: pd.Series
    criteria_count: int
    backtest_result: Optional[Dict] = None


# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """指标配置类"""

    enable_ma: bool = True
    enable_ema: bool = True
    enable_macd: bool = True
    enable_rsi: bool = True
    enable_bbands: bool = True
    enable_stoch: bool = True
    enable_atr: bool = True
    enable_obv: bool = True
    enable_mstd: bool = True
    enable_manual_indicators: bool = True

    # 性能优化配置
    enable_all_periods: bool = False  # 是否启用所有周期（影响性能）
    memory_efficient: bool = True  # 内存高效模式


class EnhancedFactorCalculator:
    """
    增强版因子计算器 - 基于154个技术指标
    移植自vbt_professional_detector.py的成熟计算逻辑
    """

    def __init__(self, indicator_config: Optional[IndicatorConfig] = None):
        """初始化增强因子计算器"""
        self.config = indicator_config or IndicatorConfig()
        logger.info("增强版因子计算器初始化完成")

        logger.info(
            f"配置: MA={self.config.enable_ma}, MACD={self.config.enable_macd}, RSI={self.config.enable_rsi}"
        )

    def calculate_comprehensive_factors(
        self, df: pd.DataFrame, timeframe: TimeFrame
    ) -> Optional[pd.DataFrame]:
        """
        计算综合技术因子 - 基于专业探测器的154个指标
        这是核心方法，移植自vbt_professional_detector.py
        """
        logger.info(f"计算 {timeframe.value} 时间框架的综合技术因子...")

        try:
            start_time = time.time()
            logger.info(f"输入数据形状: {df.shape}")

            price = df["close"].astype("float64")
            high = df["high"].astype("float64")
            low = df["low"].astype("float64")
            volume = df["volume"].astype("float64")

            factor_data = {}

            # 检查VectorBT可用指标
            available_indicators = self._check_available_indicators()
            logger.info(f"可用VectorBT指标: {available_indicators}")

            # 计算所有可用的VectorBT指标
            factor_calculations = []

            # 1. 移动平均线系列 - 启用所有周期
            if self.config.enable_ma and "MA" in available_indicators:
                if self.config.enable_all_periods:
                    ma_windows = [
                        3,
                        5,
                        8,
                        10,
                        12,
                        15,
                        20,
                        25,
                        30,
                        40,
                        50,
                        60,
                        80,
                        100,
                        120,
                        150,
                        200,
                    ]
                else:
                    ma_windows = [5, 10, 20, 30, 60]  # 核心周期

                for window in ma_windows:
                    calc = lambda w=window: vbt.MA.run(price, window=w).ma.rename(
                        f"MA{w}"
                    )
                    factor_calculations.append((f"MA{window}", calc))

            # 2. 指数移动平均
            if self.config.enable_ema:
                if self.config.enable_all_periods:
                    ema_spans = [3, 5, 8, 12, 15, 20, 26, 30, 40, 50, 60]
                else:
                    ema_spans = [5, 10, 20, 30, 60]

                for span in ema_spans:
                    calc = (
                        lambda s=span: price.ewm(span=s, adjust=False)
                        .mean()
                        .rename(f"EMA{s}")
                    )
                    factor_calculations.append((f"EMA{span}", calc))

            # 3. MACD指标系列 - 多种参数组合
            if self.config.enable_macd and "MACD" in available_indicators:
                if self.config.enable_all_periods:
                    macd_params = [
                        (6, 13, 6),
                        (8, 17, 9),
                        (12, 26, 9),
                        (15, 30, 12),
                        (6, 19, 6),
                        (12, 30, 6),
                        (8, 21, 9),
                    ]
                else:
                    macd_params = [(12, 26, 9)]  # 标准参数

                for fast, slow, signal in macd_params:
                    calc = lambda f=fast, s=slow, sig=signal: (
                        vbt.MACD.run(
                            price, fast_window=f, slow_window=s, signal_window=sig
                        )
                    )
                    factor_calculations.append((f"MACD_{fast}_{slow}_{signal}", calc))

            # 4. RSI系列
            if self.config.enable_rsi and "RSI" in available_indicators:
                if self.config.enable_all_periods:
                    rsi_windows = [3, 6, 9, 12, 14, 18, 21, 25, 30]
                else:
                    rsi_windows = [14]  # 标准周期

                for window in rsi_windows:
                    calc = lambda w=window: vbt.RSI.run(price, window=w).rsi.rename(
                        f"RSI{w}"
                    )
                    factor_calculations.append((f"RSI{window}", calc))

            # 5. 布林带系列
            if self.config.enable_bbands and "BBANDS" in available_indicators:
                if self.config.enable_all_periods:
                    bb_params = [
                        (10, 1.5),
                        (12, 2.0),
                        (15, 2.0),
                        (20, 2.0),
                        (20, 2.5),
                        (25, 2.0),
                        (30, 2.0),
                    ]
                else:
                    bb_params = [(20, 2.0)]  # 标准参数

                for window, alpha in bb_params:
                    calc = lambda w=window, a=alpha: (
                        vbt.BBANDS.run(price, window=w, alpha=a)
                    )
                    factor_calculations.append((f"BB_{window}_{alpha}", calc))

            # 6. 随机指标
            if self.config.enable_stoch and "STOCH" in available_indicators:
                if self.config.enable_all_periods:
                    stoch_params = [(9, 3), (14, 3), (14, 5), (18, 6)]
                else:
                    stoch_params = [(14, 3)]  # 标准参数

                for k_window, d_window in stoch_params:
                    calc = lambda k=k_window, d=d_window: (
                        extract_vbt_indicator(
                            vbt.STOCH.run(high, low, price, k_window=k, d_window=d)
                        )
                    )
                    factor_calculations.append((f"STOCH_{k_window}_{d_window}", calc))

            # 7. 平均真实范围
            if self.config.enable_atr and "ATR" in available_indicators:
                if self.config.enable_all_periods:
                    atr_windows = [7, 14, 21, 28]
                else:
                    atr_windows = [14]  # 标准周期

                for window in atr_windows:
                    calc = lambda w=window: vbt.ATR.run(
                        high, low, price, window=w
                    ).atr.rename(f"ATR{w}")
                    factor_calculations.append((f"ATR{window}", calc))

            # 8. 移动标准差 (波动率)
            if self.config.enable_mstd and "MSTD" in available_indicators:
                if self.config.enable_all_periods:
                    mstd_windows = [5, 10, 15, 20, 25, 30]
                else:
                    mstd_windows = [20]  # 标准周期

                for window in mstd_windows:
                    calc = lambda w=window: vbt.MSTD.run(price, window=w).mstd.rename(
                        f"MSTD{w}"
                    )
                    factor_calculations.append((f"MSTD{window}", calc))

            # 9. OBV指标
            if self.config.enable_obv and "OBV" in available_indicators:
                calc = lambda: vbt.OBV.run(price, volume).obv.rename("OBV")
                factor_calculations.append(("OBV", calc))

                # OBV移动平均
                obv_ma_windows = (
                    [5, 10, 15, 20] if self.config.enable_all_periods else [20]
                )
                for window in obv_ma_windows:
                    calc = lambda w=window: vbt.MA.run(
                        factor_data["OBV"], window=w
                    ).ma.rename(f"OBV_SMA{w}")
                    factor_calculations.append((f"OBV_SMA{window}", calc))

            # 10. 其他VectorBT高级指标
            if self.config.enable_all_periods:
                # Bollinger Band相关指标
                if "BOLB" in available_indicators:
                    # BOLB直接返回结果对象，不需要rename
                    calc = lambda: vbt.BOLB.run(price, window=20)
                    factor_calculations.append(("BOLB_20", calc))

                # Fixed Lookback指标 - 暂时禁用，存在参数和rename方法问题
                # if 'FIXLB' in available_indicators:
                #     for window in [5, 10, 20]:
                #         calc = lambda w=window: vbt.FIXLB.run(price, window=w).rename(f'FIXLB{w}')
                #         factor_calculations.append((f'FIXLB{window}', calc))

                # 统计指标 - 暂时禁用，存在rename方法问题
                # for stat_func, stat_name in [(vbt.FMAX, 'FMAX'), (vbt.FMEAN, 'FMEAN'),
                #                              (vbt.FMIN, 'FMIN'), (vbt.FSTD, 'FSTD')]:
                #     if stat_name in available_indicators:
                #         for window in [5, 10, 20]:
                #             calc = lambda w=window, func=stat_func, name=stat_name: func.run(price, window=w).rename(f'{name}{w}')
                #             factor_calculations.append((f'{stat_name}{window}', calc))

                # 其他移动平均指标
                for lb_func, lb_name in [
                    (vbt.LEXLB, "LEXLB"),
                    (vbt.MEANLB, "MEANLB"),
                    (vbt.TRENDLB, "TRENDLB"),
                ]:
                    if lb_name in available_indicators:
                        for window in [5, 10, 20]:
                            if lb_name in ["LEXLB"]:
                                # LEXLB只需要close, pos_th, neg_th
                                calc = lambda w=window, func=lb_func, name=lb_name: extract_vbt_labels(
                                    func.run(price, pos_th=0.1, neg_th=-0.1)
                                )
                            elif lb_name in ["TRENDLB"]:
                                # TRENDLB需要close, pos_th, neg_th, mode
                                calc = lambda w=window, func=lb_func, name=lb_name: extract_vbt_labels(
                                    func.run(price, pos_th=0.1, neg_th=-0.1, mode=0)
                                )
                            else:
                                # MEANLB只需要close, window
                                calc = lambda w=window, func=lb_func, name=lb_name: extract_vbt_labels(
                                    func.run(price, window=w)
                                )
                            factor_calculations.append((f"{lb_name}{window}", calc))

                # OHLC统计指标 - 暂时禁用，存在rename方法问题
                # ohlc_funcs = [(vbt.OHLCSTCX, 'OHLCSTCX'), (vbt.OHLCSTX, 'OHLCSTX')]
                # for ohlc_func, ohlc_name in ohlc_funcs:
                #     if ohlc_name in available_indicators:
                #         calc = lambda func=ohlc_func, name=ohlc_name: func.run(ohlc_dict=price).rename(name)
                #         factor_calculations.append((ohlc_name, calc))

                # 随机指标和概率指标 - 暂时禁用，存在rename方法问题
                # rand_funcs = [(vbt.RAND, 'RAND'), (vbt.RANDNX, 'RANDNX'), (vbt.RANDX, 'RANDX')]
                # for rand_func, rand_name in rand_funcs:
                #     if rand_name in available_indicators:
                #         calc = lambda func=rand_func, name=rand_name: func.run(input_shape=len(price)).rename(name)
                #         factor_calculations.append((rand_name, calc))

                # 概率相关指标 - 暂时禁用，存在rename方法问题
                # prob_funcs = [(vbt.RPROB, 'RPROB'), (vbt.RPROBCX, 'RPROBCX'),
                #              (vbt.RPROBNX, 'RPROBNX'), (vbt.RPROBX, 'RPROBX')]
                # for prob_func, prob_name in prob_funcs:
                #     if prob_name in available_indicators:
                #         calc = lambda func=prob_func, name=prob_name: func.run(input_shape=len(price)).rename(name)
                #         factor_calculations.append((prob_name, calc))

                # Supertrend指标 - 暂时禁用，存在rename方法问题
                # st_funcs = [(vbt.STCX, 'STCX'), (vbt.STX, 'STX')]
                # for st_func, st_name in st_funcs:
                #     if st_name in available_indicators:
                #         calc = lambda func=st_func, name=st_name: func.run(high=high, low=low, close=price).rename(name)
                #         factor_calculations.append((st_name, calc))

            # 11. TA-Lib指标 (如果可用)
            if hasattr(vbt, "talib") and self.config.enable_all_periods:
                talib_params = {
                    "SMA": {"timeperiod": [5, 10, 20, 30, 60]},
                    "EMA": {"timeperiod": [5, 10, 20, 30, 60]},
                    "WMA": {"timeperiod": [5, 10, 20]},
                    "DEMA": {"timeperiod": [5, 10, 20]},
                    "TEMA": {"timeperiod": [5, 10, 20]},
                    "TRIMA": {"timeperiod": [5, 10, 20]},
                    "KAMA": {"timeperiod": [10, 20]},
                    "T3": {"timeperiod": [5, 10, 20]},
                    "MIDPRICE": {"timeperiod": [5, 10, 20]},
                    "SAR": {},  # 无参数
                    "ADX": {"timeperiod": [14]},
                    "ADXR": {"timeperiod": [14]},
                    "APO": {"fastperiod": [12], "slowperiod": [26], "matype": [0]},
                    "AROON": {"timeperiod": [14]},
                    "AROONOSC": {"timeperiod": [14]},
                    "CCI": {"timeperiod": [14]},
                    "DX": {"timeperiod": [14]},
                    "MFI": {"timeperiod": [14]},
                    "MOM": {"timeperiod": [10]},
                    "ROC": {"timeperiod": [10]},
                    "ROCP": {"timeperiod": [10]},
                    "ROCR": {"timeperiod": [10]},
                    "ROCR100": {"timeperiod": [10]},
                    "RSI": {"timeperiod": [14]},
                    "STOCH": {},  # 默认参数
                    "STOCHF": {},
                    "STOCHRSI": {
                        "timeperiod": [14],
                        "fastk_period": [5],
                        "fastd_period": [3],
                    },
                    "TRIX": {"timeperiod": [14]},
                    "ULTOSC": {
                        "timeperiod1": [7],
                        "timeperiod2": [14],
                        "timeperiod3": [28],
                    },
                    "WILLR": {"timeperiod": [14]},
                    "CDL2CROWS": {},  # 形态识别
                    "CDL3BLACKCROWS": {},
                    "CDL3INSIDE": {},
                    "CDL3LINESTRIKE": {},
                    "CDL3OUTSIDE": {},
                    "CDL3STARSINSOUTH": {},
                    "CDL3WHITESOLDIERS": {},
                    "CDLABANDONEDBABY": {},
                    "CDLADVANCEBLOCK": {},
                    "CDLBELTHOLD": {},
                    "CDLBREAKAWAY": {},
                    "CDLCLOSINGMARUBOZU": {},
                    "CDLCONCEALBABYSWALL": {},
                    "CDLCOUNTERATTACK": {},
                    "CDLDARKCLOUDCOVER": {},
                    "CDLDOJI": {},
                    "CDLDOJISTAR": {},
                    "CDLDRAGONFLYDOJI": {},
                    "CDLENGULFING": {},
                    "CDLEVENINGDOJISTAR": {},
                    "CDLEVENINGSTAR": {},
                    "CDLGAPSIDESIDEWHITE": {},
                    "CDLGRAVESTONEDOJI": {},
                    "CDLHAMMER": {},
                    "CDLHANGINGMAN": {},
                    "CDLHARAMI": {},
                    "CDLHARAMICROSS": {},
                    "CDLHIGHWAVE": {},
                    "CDLHIKKAKE": {},
                    "CDLHOMINGPIGEON": {},
                    "CDLIDENTICAL3CROWS": {},
                    "CDLINNECK": {},
                    "CDLINVERTEDHAMMER": {},
                    "CDLKICKING": {},
                    "CDLKICKINGBYLENGTH": {},
                    "CDLLADDERBOTTOM": {},
                    "CDLLONGLEGGEDDOJI": {},
                    "CDLLONGLINE": {},
                    "CDLMARUBOZU": {},
                    "CDLMATCHINGLOW": {},
                    "CDLMATHOLD": {},
                    "CDLMORNINGDOJISTAR": {},
                    "CDLMORNINGSTAR": {},
                    "CDLONNECK": {},
                    "CDLPIERCING": {},
                    "CDLRICKSHAWMAN": {},
                    "CDLRISEFALL3METHODS": {},
                    "CDLSEPARATINGLINES": {},
                    "CDLSHOOTINGSTAR": {},
                    "CDLSHORTLINE": {},
                    "CDLSPINNINGTOP": {},
                    "CDLSTALLEDPATTERN": {},
                    "CDLSTICKSANDWICH": {},
                    "CDLTAKURI": {},
                    "CDLTASUKIGAP": {},
                    "CDLTHRUSTING": {},
                    "CDLTRISTAR": {},
                    "CDLUNIQUE3RIVER": {},
                    "CDLUPSIDEGAP2CROWS": {},
                    "CDLXSIDEGAP3METHODS": {},
                }

                # 使用统一的数据契约系统重构TA-Lib指标计算
                for talib_name, params in talib_params.items():
                    try:
                        talib_indicator = vbt.talib(talib_name)

                        # 使用统一的数据契约系统处理所有指标
                        if not params:
                            # 默认参数指标
                            def make_unified_calc():
                                def calc():
                                    return calculate_indicator_with_validation(
                                        talib_indicator,
                                        f"TA_{talib_name}",
                                        talib_name,
                                        price,
                                        high,
                                        low,
                                        df["open"],
                                        volume,
                                        df.index,
                                    )

                                return calc

                            factor_calculations.append(
                                (f"TA_{talib_name}", make_unified_calc())
                            )
                        elif "timeperiod" in params:
                            periods = params.get("timeperiod")
                            if periods is None:
                                continue
                            if not isinstance(periods, (list, tuple, set)):
                                periods = [periods]

                            for period in periods:

                                def make_unified_period_calc(p=period):
                                    def calc():
                                        return calculate_indicator_with_validation(
                                            talib_indicator,
                                            f"TA_{talib_name}_{p}",
                                            talib_name,
                                            price,
                                            high,
                                            low,
                                            df["open"],
                                            volume,
                                            df.index,
                                            timeperiod=p,
                                        )

                                    return calc

                                factor_calculations.append(
                                    (
                                        f"TA_{talib_name}_{period}",
                                        make_unified_period_calc(),
                                    )
                                )

                    except Exception as e:
                        logger.warning(f"TA指标 {talib_name} 初始化失败: {e}")
                        continue

            # 10. 手动计算额外指标
            if self.config.enable_manual_indicators:
                logger.info("计算额外技术指标...")
                self._calculate_manual_indicators(
                    factor_data,
                    price,
                    high,
                    low,
                    volume,
                    self.config.enable_all_periods,
                )

            # 执行VectorBT指标计算
            logger.info(f"执行VectorBT指标计算，共{len(factor_calculations)}个指标...")
            successful_calcs = 0
            failed_calcs = 0

            for name, calc_func in factor_calculations:
                try:
                    result = calc_func()

                    # 处理不同类型指标的输出
                    processed = self._process_indicator_result(
                        name, result, factor_data
                    )
                    successful_calcs += processed

                except Exception as e:
                    logger.warning(f"指标 {name} 计算失败: {e}")
                    failed_calcs += 1

            # 收集所有因子
            factors_df = pd.DataFrame(factor_data, index=df.index)

            logger.info(f"原始因子数据形状: {factors_df.shape}")
            logger.info(f"因子数据空值统计: {factors_df.isnull().sum().sum()}")

            # 智能数据清理 - 保持Linus风格的不丢失数据原则
            factors_df = self._clean_factor_data_intelligently(factors_df)

            calc_time = time.time() - start_time
            logger.info(f"综合因子计算完成:")
            logger.info(f"  - 总计算指标数: {len(factor_calculations)}")
            logger.info(f"  - 成功计算: {successful_calcs} 个")
            logger.info(f"  - 失败: {failed_calcs} 个")
            logger.info(f"  - 最终因子数量: {len(factors_df.columns)} 个")
            logger.info(f"  - 数据点数: {len(factors_df)} 行")
            logger.info(f"  - 计算耗时: {calc_time:.3f}秒")

            return factors_df

        except Exception as e:
            logger.error(f"综合因子计算失败: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    def _check_available_indicators(self) -> List[str]:
        """检查VectorBT可用指标"""
        available_indicators = []
        # VectorBT 核心指标
        vbt_indicators = [
            "MA",
            "MACD",
            "RSI",
            "BBANDS",
            "STOCH",
            "ATR",
            "OBV",
            "MSTD",
            "BOLB",
            "FIXLB",
            "FMAX",
            "FMEAN",
            "FMIN",
            "FSTD",
            "LEXLB",
            "MEANLB",
            "OHLCSTCX",
            "OHLCSTX",
            "RAND",
            "RANDNX",
            "RANDX",
            "RPROB",
            "RPROBCX",
            "RPROBNX",
            "RPROBX",
            "STCX",
            "STX",
            "TRENDLB",
        ]

        for indicator in vbt_indicators:
            if hasattr(vbt, indicator):
                available_indicators.append(indicator)

        # TA-Lib 指标 (如果可用)
        try:
            if hasattr(vbt, "talib"):
                # 尝试获取一些常见的TA-Lib指标
                common_talib = [
                    "SMA",
                    "EMA",
                    "WMA",
                    "DEMA",
                    "TEMA",
                    "TRIMA",
                    "KAMA",
                    "MAMA",
                    "T3",
                    "RSI",
                    "STOCH",
                    "STOCHF",
                    "STOCHRSI",
                    "MACD",
                    "MACDEXT",
                    "MACDFIX",
                    "BBANDS",
                    "MIDPOINT",
                    "SAR",
                    "SAREXT",
                    "ADX",
                    "ADXR",
                    "APO",
                ]
                for indicator in common_talib:
                    try:
                        vbt.talib(indicator)
                        available_indicators.append(f"TA_{indicator}")
                    except:
                        pass
        except:
            pass

        return available_indicators

    def _process_indicator_result(self, name: str, result, factor_data: Dict) -> int:
        """处理不同类型指标的输出，返回成功计算的指标数量"""
        processed_count = 0

        # MACD系列指标
        if hasattr(result, "macd") and hasattr(result, "signal"):
            factor_data[f"{name}_MACD"] = result.macd
            factor_data[f"{name}_Signal"] = result.signal
            if hasattr(result, "hist"):
                factor_data[f"{name}_Hist"] = result.hist
            processed_count = 3
        # STOCH随机指标
        elif hasattr(result, "percent_k") and hasattr(result, "percent_d"):
            factor_data[f"{name}_K"] = result.percent_k
            factor_data[f"{name}_D"] = result.percent_d
            processed_count = 2
        # BOLB指标 (Binary Oscillator with Lower Bounds) - 特殊处理VectorBT对象
        elif type(result).__name__ == "BOLB" and hasattr(result, "labels"):
            # BOLB对象生成的是二元标签，不是传统的上下界
            # 直接保存labels数据（这是主要的输出）
            factor_data[name] = result.labels
            # 不保存方法属性，只保存可序列化的数据
            processed_count = 1
        # 布林带系列指标
        elif (
            hasattr(result, "upper")
            and hasattr(result, "middle")
            and hasattr(result, "lower")
        ):
            factor_data[f"{name}_Upper"] = result.upper
            factor_data[f"{name}_Middle"] = result.middle
            factor_data[f"{name}_Lower"] = result.lower
            # 布林带宽度
            bb_width = (result.upper - result.lower) / result.middle
            factor_data[f"{name}_Width"] = bb_width
            processed_count = 4
        # OBV指标
        elif hasattr(result, "obv"):
            factor_data[name] = result.obv
            processed_count = 1
        # 单输出指标
        elif hasattr(result, "atr"):
            factor_data[name] = result.atr
            processed_count = 1
        elif hasattr(result, "rsi"):
            factor_data[name] = result.rsi
            processed_count = 1
        elif hasattr(result, "ma"):
            factor_data[name] = result.ma
            processed_count = 1
        elif hasattr(result, "mstd"):
            factor_data[name] = result.mstd
            processed_count = 1
        elif hasattr(result, "real"):  # TA-Lib指标常用输出
            factor_data[name] = result.real
            processed_count = 1
        elif hasattr(result, "output"):  # 自定义指标输出
            factor_data[name] = result.output
            processed_count = 1
        # 直接是数值或数组
        elif hasattr(result, "iloc") or hasattr(result, "values"):
            factor_data[name] = result
            processed_count = 1
        # 多输出指标的通用处理
        elif hasattr(result, "_fields") and hasattr(result, "_asdict"):
            # 处理namedtuple类型的输出
            result_dict = result._asdict()
            for key, value in result_dict.items():
                factor_data[f"{name}_{key}"] = value
                processed_count += 1
        else:
            # 尝试直接使用结果
            try:
                factor_data[name] = result
                processed_count = 1
            except:
                logger.warning(f"无法处理指标 {name} 的结果类型: {type(result)}")

        return processed_count

    def _calculate_manual_indicators(
        self,
        factor_data: Dict,
        price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        all_periods: bool = False,
    ):
        """手动计算额外技术指标"""

        # 威廉指标系列
        willr_windows = [9, 14, 18, 21] if all_periods else [14]
        for window in willr_windows:
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            willr = (highest_high - price) / (highest_high - lowest_low + 1e-8) * -100
            factor_data[f"WILLR{window}"] = willr

        # 商品通道指数系列
        cci_windows = [10, 14, 20] if all_periods else [14]
        for window in cci_windows:
            tp = (high + low + price) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = np.abs(tp - sma_tp).rolling(window=window).mean()
            cci = (tp - sma_tp) / (0.015 * mad + 1e-8)
            factor_data[f"CCI{window}"] = cci

        # 价格动量指标系列
        momentum_periods = [1, 3, 5, 8, 10, 12, 15, 20] if all_periods else [5, 10, 20]
        for period in momentum_periods:
            momentum = price / price.shift(period) - 1
            factor_data[f"Momentum{period}"] = momentum

        # 价格位置指标系列
        position_windows = (
            [5, 8, 10, 12, 15, 20, 25, 30] if all_periods else [5, 10, 20]
        )
        for window in position_windows:
            position = (price - price.rolling(window=window).min()) / (
                price.rolling(window=window).max()
                - price.rolling(window=window).min()
                + 1e-8
            )
            factor_data[f"Position{window}"] = position

        # 趋势强度指标系列
        trend_windows = [5, 8, 10, 12, 15, 20, 25] if all_periods else [5, 10, 20]
        for window in trend_windows:
            trend = (price - price.rolling(window=window).mean()) / (
                price.rolling(window=window).std() + 1e-8
            )
            factor_data[f"Trend{window}"] = trend

        # 成交量指标系列
        volume_windows = [10, 15, 20, 25, 30] if all_periods else [20]
        for window in volume_windows:
            volume_sma = volume.rolling(window=window).mean()
            volume_ratio = volume / (volume_sma + 1e-8)
            factor_data[f"Volume_Ratio{window}"] = volume_ratio

            volume_momentum = volume / volume.shift(window) - 1
            factor_data[f"Volume_Momentum{window}"] = volume_momentum

        # VWAP (成交量加权平均价)
        vwap_windows = [10, 15, 20, 25, 30] if all_periods else [20]
        for window in vwap_windows:
            typical_price = (high + low + price) / 3
            vwap = (typical_price * volume).rolling(window=window).sum() / (
                volume.rolling(window=window).sum() + 1e-8
            )
            factor_data[f"VWAP{window}"] = vwap

    def _clean_factor_data_intelligently(
        self, factors_df: pd.DataFrame, timeframe: str = None
    ) -> pd.DataFrame:
        """智能清理因子数据 - Linus风格的不丢失数据原则"""

        # 关键原则：不删除任何原始数据的时间点
        factors_cleaned = factors_df.copy()

        # 识别不同类型的指标及其所需的历史数据
        indicator_periods = {
            "MA": [5, 10, 20, 30, 60],
            "MA3": 3,
            "MA5": 5,
            "MA8": 8,
            "MA10": 10,
            "MA12": 12,
            "MA15": 15,
            "MA20": 20,
            "MA25": 25,
            "MA30": 30,
            "MA40": 40,
            "MA50": 50,
            "MA60": 60,
            "MA80": 80,
            "MA100": 100,
            "MA120": 120,
            "MA150": 150,
            "MA200": 200,
            "EMA": [5, 10, 20, 30, 60],
            "EMA3": 3,
            "EMA5": 5,
            "EMA8": 8,
            "EMA12": 12,
            "EMA15": 15,
            "EMA20": 20,
            "EMA26": 26,
            "EMA30": 30,
            "EMA40": 40,
            "EMA50": 50,
            "EMA60": 60,
            "RSI": [3, 6, 9, 12, 14, 18, 21, 25, 30],
            "RSI3": 3,
            "RSI6": 6,
            "RSI9": 9,
            "RSI12": 12,
            "RSI14": 14,
            "RSI18": 18,
            "RSI21": 21,
            "RSI25": 25,
            "RSI30": 30,
            "MACD": 26,
            "BB": [10, 12, 15, 20, 25, 30],
            "STOCH": 14,
            "ATR": [7, 14, 21, 28],
            "MSTD": [5, 10, 15, 20, 25, 30],
            "WILLR": [9, 14, 18, 21],
            "CCI": [10, 14, 20],
            "Momentum": [1, 3, 5, 8, 10, 12, 15, 20],
            "Position": [5, 8, 10, 12, 15, 20, 25, 30],
            "Trend": [5, 8, 10, 12, 15, 20, 25],
            "Volume": [10, 15, 20, 25, 30],
            "VWAP": [10, 15, 20, 25, 30],
        }

        # 特殊处理15min时间框架的数据质量问题
        if timeframe == "15min":
            logger.info(f"检测到15min时间框架，应用特殊数据清理逻辑")

            # 15min框架可能有数据重采样导致的时间对齐问题
            # 增加更积极的数据填充策略
            for col in factors_cleaned.columns:
                # 统计当前的空值情况
                null_count = factors_cleaned[col].isnull().sum()
                total_count = len(factors_cleaned)

                if null_count > 0:
                    null_ratio = null_count / total_count
                    logger.info(
                        f"  列 {col}: 空值率 {null_ratio:.2%} ({null_count}/{total_count})"
                    )

                    # 对空值率过高的列进行特殊处理
                    if null_ratio > 0.5:  # 超过50%空值
                        # 找到第一个有效值
                        first_valid_idx = factors_cleaned[col].first_valid_index()
                        if first_valid_idx is not None:
                            # 从第一个有效值开始进行更积极的填充
                            factors_cleaned.loc[first_valid_idx:, col] = (
                                factors_cleaned.loc[first_valid_idx:, col].ffill()
                            )

                            # 如果还有空值，使用向后填充
                            if factors_cleaned[col].isnull().sum() > 0:
                                factors_cleaned[col] = factors_cleaned[col].bfill()

                            # 最后，如果仍然有空值（说明整个列都是空的），用0填充
                            if factors_cleaned[col].isnull().sum() > 0:
                                factors_cleaned[col] = factors_cleaned[col].fillna(0)

                            logger.info(
                                f"  列 {col} 已修复，剩余空值: {factors_cleaned[col].isnull().sum()}"
                            )

        # 标准的智能数据清理逻辑
        for col in factors_cleaned.columns:
            # 跳过已经处理过的15min列
            if timeframe == "15min" and factors_cleaned[col].isnull().sum() == 0:
                continue

            # 找出该指标需要的周期
            required_period = 60  # 默认值（最保守）
            for indicator, periods in indicator_periods.items():
                if indicator in col:
                    if isinstance(periods, list):
                        required_period = max(periods)
                    else:
                        required_period = periods
                    break

            # 只对技术指标进行前向填充，不删除任何行
            if not col.startswith("Volume_") and not col.startswith("OBV_"):
                # 找到第一个有效值的位置
                first_valid_idx = factors_cleaned[col].first_valid_index()
                if first_valid_idx is not None:
                    # 将第一个有效值之前的NaN保持原样（体现历史数据不足）
                    # 从第一个有效值开始，对后续的NaN进行前向填充
                    factors_cleaned.loc[first_valid_idx:, col] = factors_cleaned.loc[
                        first_valid_idx:, col
                    ].ffill()
            else:
                # 成交量指标可以直接前向填充
                factors_cleaned[col] = factors_cleaned[col].ffill()

        # 最终结果：保留所有原始数据行，某些指标在数据开始阶段可能为NaN
        return factors_cleaned

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子类别信息"""
        categories = {
            "移动平均线": [
                f"MA{w}"
                for w in [
                    3,
                    5,
                    8,
                    10,
                    12,
                    15,
                    20,
                    25,
                    30,
                    40,
                    50,
                    60,
                    80,
                    100,
                    120,
                    150,
                    200,
                ]
            ]
            + [f"EMA{s}" for s in [3, 5, 8, 12, 15, 20, 26, 30, 40, 50, 60]],
            "MACD指标": [
                f"MACD_{f}_{s}_{sig}"
                for f, s, sig in [
                    (6, 13, 6),
                    (8, 17, 9),
                    (12, 26, 9),
                    (15, 30, 12),
                    (6, 19, 6),
                    (12, 30, 6),
                    (8, 21, 9),
                ]
            ],
            "RSI指标": [f"RSI{w}" for w in [3, 6, 9, 12, 14, 18, 21, 25, 30]],
            "布林带": [
                f"BB_{w}_{a}"
                for w, a in [
                    (10, 1.5),
                    (12, 2.0),
                    (15, 2.0),
                    (20, 2.0),
                    (20, 2.5),
                    (25, 2.0),
                    (30, 2.0),
                ]
            ],
            "随机指标": [
                f"STOCH_{k}_{d}" for k, d in [(9, 3), (14, 3), (14, 5), (18, 6)]
            ],
            "ATR指标": [f"ATR{w}" for w in [7, 14, 21, 28]],
            "波动率指标": [f"MSTD{w}" for w in [5, 10, 15, 20, 25, 30]],
            "成交量指标": ["OBV"]
            + [f"OBV_SMA{w}" for w in [5, 10, 15, 20]]
            + [f"Volume_Ratio{w}" for w in [10, 15, 20, 25, 30]]
            + [f"Volume_Momentum{w}" for w in [10, 15, 20, 25, 30]]
            + [f"VWAP{w}" for w in [10, 15, 20, 25, 30]],
            "威廉指标": [f"WILLR{w}" for w in [9, 14, 18, 21]],
            "商品通道": [f"CCI{w}" for w in [10, 14, 20]],
            "动量指标": [f"Momentum{w}" for w in [1, 3, 5, 8, 10, 12, 15, 20]],
            "位置指标": [f"Position{w}" for w in [5, 8, 10, 12, 15, 20, 25, 30]],
            "趋势强度": [f"Trend{w}" for w in [5, 8, 10, 12, 15, 20, 25]],
        }

        # 计算每个类别的指标数量
        total_indicators = sum(len(indicators) for indicators in categories.values())
        logger.info(
            f"因子类别统计: {len(categories)} 个大类，共 {total_indicators} 个指标"
        )

        for category, indicators in categories.items():
            logger.info(f"  {category}: {len(indicators)} 个指标")

        return categories


# 测试函数
def test_enhanced_calculator():
    """测试增强版因子计算器"""
    logger.info("测试增强版因子计算器...")

    # 创建测试数据
    dates = pd.date_range("2025-01-01", periods=100, freq="5min")
    test_data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(100, 200, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    # 测试基础配置
    basic_config = IndicatorConfig(enable_all_periods=False, memory_efficient=True)

    calculator = EnhancedFactorCalculator(basic_config)

    # 计算因子
    factors = calculator.calculate_comprehensive_factors(test_data, TimeFrame.MIN_5)

    if factors is not None:
        logger.info(
            f"✅ 测试成功: 生成 {len(factors.columns)} 个因子，{len(factors)} 个数据点"
        )

        # 获取因子类别信息
        categories = calculator.get_factor_categories()
        logger.info(f"因子类别: {len(categories)} 个大类")

        return factors
    else:
        logger.error("❌ 测试失败: 因子计算失败")
        return None


if __name__ == "__main__":
    test_enhanced_calculator()
