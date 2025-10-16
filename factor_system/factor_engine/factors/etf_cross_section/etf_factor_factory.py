#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子工厂
基于现有269个技术指标自动生成参数变体，完全基于vbt和talib
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import itertools
from functools import partial

from factor_system.factor_engine import api
from .factor_registry import ETFFactorRegistry, get_factor_registry, FactorCategory, register_etf_factor
from .candidate_factor_generator import ETFCandidateFactorGenerator, FactorVariant

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


@dataclass
class VBTFactorSpec:
    """VBT因子规格"""
    factor_id: str
    indicator_class: str
    parameters: Dict[str, Any]
    output_field: str
    category: FactorCategory
    description: str


class ETFFactorFactory:
    """ETF因子工厂"""

    def __init__(self):
        """初始化因子工厂"""
        self.registry = get_factor_registry()
        self.candidate_generator = ETFCandidateFactorGenerator()
        self.vbt_indicator_map = self._create_vbt_indicator_map()
        self.talib_indicator_map = self._create_talib_indicator_map()

        logger.info("ETF因子工厂初始化完成")

    def _create_vbt_indicator_map(self) -> Dict[str, Dict[str, Any]]:
        """创建VBT指标映射"""
        return {
            # 动量指标
            "RSI": {
                "class": "RSI",
                "param_ranges": {"window": [7, 14, 21, 28]},
                "output": "rsi",
                "category": FactorCategory.MOMENTUM
            },
            "MACD": {
                "class": "MACD",
                "param_ranges": {
                    "fast_window": [12, 15, 18],
                    "slow_window": [26, 29, 32],
                    "signal_window": [9, 12]
                },
                "output": "macd",
                "category": FactorCategory.MOMENTUM
            },
            "STOCH": {
                "class": "STOCH",
                "param_ranges": {
                    "k_window": [5, 9, 14],
                    "d_window": [3, 5, 6]
                },
                "output": "stoch_k",
                "category": FactorCategory.MOMENTUM
            },
            "STOCHRSI": {
                "class": "STOCHRSI",
                "param_ranges": {
                    "rsi_window": [14, 21],
                    "stoch_window": [14],
                    "k_window": [5, 9],
                    "d_window": [3, 6]
                },
                "output": "stochrsi_k",
                "category": FactorCategory.MOMENTUM
            },
            "WILLR": {
                "class": "WILLR",
                "param_ranges": {"window": [7, 14, 21]},
                "output": "willr",
                "category": FactorCategory.MOMENTUM
            },
            "CCI": {
                "class": "CCI",
                "param_ranges": {"window": [14, 21, 28]},
                "output": "cci",
                "category": FactorCategory.MOMENTUM
            },
            "ADX": {
                "class": "ADX",
                "param_ranges": {"window": [7, 14, 21]},
                "output": "adx",
                "category": FactorCategory.MOMENTUM
            },
            "AROON": {
                "class": "AROON",
                "param_ranges": {"window": [14, 21, 28]},
                "output": "aroon_up",
                "category": FactorCategory.MOMENTUM
            },
            "MFI": {
                "class": "MFI",
                "param_ranges": {"window": [7, 14, 21]},
                "output": "mfi",
                "category": FactorCategory.MOMENTUM
            },
            "ROC": {
                "class": "ROC",
                "param_ranges": {"window": [10, 20, 60]},
                "output": "roc",
                "category": FactorCategory.MOMENTUM
            },
            "MOM": {
                "class": "MOM",
                "param_ranges": {"window": [10, 20, 60]},
                "output": "mom",
                "category": FactorCategory.MOMENTUM
            },
            "APO": {
                "class": "APO",
                "param_ranges": {
                    "fast_window": [12, 26],
                    "slow_window": [26, 50]
                },
                "output": "apo",
                "category": FactorCategory.MOMENTUM
            },
            "PPO": {
                "class": "PPO",
                "param_ranges": {
                    "fast_window": [12, 26],
                    "slow_window": [26, 50]
                },
                "output": "ppo",
                "category": FactorCategory.MOMENTUM
            },
            "ULTOSC": {
                "class": "ULTOSC",
                "param_ranges": {
                    "short_window": [7, 14],
                    "medium_window": [14, 21],
                    "long_window": [28]
                },
                "output": "ultosc",
                "category": FactorCategory.MOMENTUM
            },

            # 趋势指标
            "MA3": {
                "class": "MA",
                "param_ranges": {"window": [5, 10, 20, 50, 120]},
                "output": "ma",
                "category": FactorCategory.TREND
            },
            "MA5": {
                "class": "MA",
                "param_ranges": {"window": [5, 10, 20, 50, 120]},
                "output": "ma",
                "category": FactorCategory.TREND
            },
            "MA10": {
                "class": "MA",
                "param_ranges": {"window": [10, 20, 50, 120]},
                "output": "ma",
                "category": FactorCategory.TREND
            },
            "EMA": {
                "class": "EMA",
                "param_ranges": {"window": [5, 10, 20, 50, 120]},
                "output": "ema",
                "category": FactorCategory.TREND
            },
            "DEMA": {
                "class": "DEMA",
                "param_ranges": {"window": [5, 10, 20, 50, 120]},
                "output": "dema",
                "category": FactorCategory.TREND
            },
            "TEMA": {
                "class": "TEMA",
                "param_ranges": {"window": [5, 10, 20, 50, 120]},
                "output": "tema",
                "category": FactorCategory.TREND
            },
            "KAMA": {
                "class": "KAMA",
                "param_ranges": {"window": [10, 20, 50, 120]},
                "output": "kama",
                "category": FactorCategory.TREND
            },
            # 🔥 已移除：WMA, TRIMA, T3 - VectorBT不支持
            
            # 波动率指标
            "ATR": {
                "class": "ATR",
                "param_ranges": {"window": [7, 14, 21]},
                "output": "atr",
                "category": FactorCategory.VOLATILITY
            },
            "NATR": {
                "class": "NATR",
                "param_ranges": {"window": [7, 14, 21]},
                "output": "natr",
                "category": FactorCategory.VOLATILITY
            },
            # 🔥 已移除：MSTD, FSTD - VectorBT不支持

            # 成交量指标
            "OBV": {
                "class": "OBV",
                "param_ranges": {},
                "output": "obv",
                "category": FactorCategory.VOLUME
            },
            "OBV_SMA": {
                "class": "OBV_SMA",
                "param_ranges": {"window": [10, 20, 50]},
                "output": "obv_sma",
                "category": FactorCategory.VOLUME
            },
            "OBV_EMA": {
                "class": "OBV_EMA",
                "param_ranges": {"window": [10, 20, 50]},
                "output": "obv_ema",
                "category": FactorCategory.VOLUME
            },
            "AD": {
                "class": "AD",
                "param_ranges": {},
                "output": "ad",
                "category": FactorCategory.VOLUME
            },
            "ADOSC": {
                "class": "ADOSC",
                "param_ranges": {
                    "fast_window": [3, 7],
                    "slow_window": [10, 21]
                },
                "output": "adosc",
                "category": FactorCategory.VOLUME
            },
            "VWAP": {
                "class": "VWAP",
                "param_ranges": {"window": [10, 20, 50]},
                "output": "vwap",
                "category": FactorCategory.VOLUME
            },
            "Volume_Ratio": {
                "class": "VolumeRatio",
                "param_ranges": {
                    "short_window": [5, 10],
                    "long_window": [20, 50]
                },
                "output": "volume_ratio",
                "category": FactorCategory.VOLUME
            },
            "Volume_Momentum": {
                "class": "VolumeMomentum",
                "param_ranges": {"window": [10, 20, 50]},
                "output": "volume_momentum",
                "category": FactorCategory.VOLUME
            },

            # 均值回归指标
            "BB_": {
                "class": "BBANDS",
                "param_ranges": {
                    "window": [10, 15, 20],
                    "std": [1.5, 2.0, 2.5]
                },
                "output": "upperband",
                "category": FactorCategory.MEAN_REVERSION
            }
        }

    def _create_talib_indicator_map(self) -> Dict[str, Dict[str, Any]]:
        """创建TA-Lib指标映射"""
        return {
            # TA-Lib指标（用于VBT未覆盖的部分）
            "SAR": {
                "function": "SAR",
                "param_ranges": {
                    "acceleration": [0.02, 0.04],
                    "maximum": [0.2, 0.4]
                },
                "category": FactorCategory.TREND
            },
            "HT_SINE": {
                "function": "HT_SINE",
                "param_ranges": {},
                "category": FactorCategory.TREND
            },
            "HT_TRENDLINE": {
                "function": "HT_TRENDLINE",
                "param_ranges": {},
                "category": FactorCategory.TREND
            },
            "CDL2CROWS": {
                "function": "CDL2CROWS",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDL3BLACKCROWS": {
                "function": "CDL3BLACKCROWS",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDL3INSIDE": {
                "function": "CDL3INSIDE",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDL3OUTSIDE": {
                "function": "CDL3OUTSIDE",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDL3WHITESOLDIERS": {
                "function": "CDL3WHITESOLDIERS",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDL3STARSINSOUTH": {
                "function": "CDL3STARSINSOUTH",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            },
            "CDLABANDONEDBABY": {
                "function": "CDLABANDONEDBABY",
                "param_ranges": {},
                "category": FactorCategory.CANDLESTICK
            }
            # 可以继续添加更多TA-Lib指标
        }

    def _create_vbt_factor_function(self, spec: VBTFactorSpec) -> Callable:
        """创建VBT因子计算函数"""
        def factor_function(data: pd.DataFrame) -> pd.Series:
            try:
                # 导入vectorbt
                import vectorbt as vbt

                # 准备价格数据
                close = data['close']
                high = data['high']
                low = data['low']
                volume = data['volume'] if 'volume' in data.columns else None

                # 🔥 修复：使用.访问属性而非[]
                # 获取指标类
                indicator_class = getattr(vbt, spec.indicator_class)

                # 根据指标类型创建实例
                if spec.indicator_class == "BBANDS":
                    indicator = indicator_class.run(
                        close, **spec.parameters
                    )
                elif spec.indicator_class in ["OBV", "AD"]:
                    indicator = indicator_class.run(
                        close, volume, **spec.parameters
                    )
                elif spec.indicator_class == "ADOSC":
                    indicator = indicator_class.run(
                        close, high, low, volume, **spec.parameters
                    )
                elif spec.indicator_class == "VWAP":
                    indicator = indicator_class.run(
                        close, volume, **spec.parameters
                    )
                elif spec.indicator_class == "VolumeRatio":
                    indicator = indicator_class.run(
                        volume, **spec.parameters
                    )
                elif spec.indicator_class == "VolumeMomentum":
                    indicator = indicator_class.run(
                        volume, **spec.parameters
                    )
                else:
                    indicator = indicator_class.run(
                        close, **spec.parameters
                    )

                # 获取输出字段
                if hasattr(indicator, spec.output_field):
                    return getattr(indicator, spec.output_field)
                else:
                    # 尝试从第一个元素获取
                    if hasattr(indicator, '__getitem__') and len(indicator) > 0:
                        return indicator[0]
                    else:
                        return indicator

            except Exception as e:
                logger.error(f"VBT因子计算失败 {spec.factor_id}: {str(e)}")
                return pd.Series(0, index=data.index)

        return factor_function

    def _create_talib_factor_function(self, indicator_name: str, params: Dict[str, Any]) -> Callable:
        """创建TA-Lib因子计算函数"""
        def factor_function(data: pd.DataFrame) -> pd.Series:
            try:
                import talib

                # 准备价格数据
                close = data['close'].values
                high = data['high'].values
                low = data['low'].values
                volume = data['volume'].values if 'volume' in data.columns else None

                # 获取TA-Lib函数
                talib_func = getattr(talib, indicator_name)

                # 根据指标类型调用函数
                if indicator_name == "SAR":
                    result = talib_func(high=high, low=low, **params)
                elif indicator_name in ["OBV", "AD", "ADOSC"]:
                    if volume is not None:
                        result = talib_func(close=close, volume=volume, **params)
                    else:
                        result = talib_func(close=close, **params)
                elif indicator_name.startswith("CDL"):
                    result = talib_func(open=data['open'].values, high=high, low=low, close=close, **params)
                else:
                    result = talib_func(close=close, **params)

                return pd.Series(result, index=data.index)

            except Exception as e:
                logger.error(f"TA-Lib因子计算失败 {indicator_name}: {str(e)}")
                return pd.Series(0, index=data.index)

        return factor_function

    def generate_vbt_factor_variants(self) -> List[VBTFactorSpec]:
        """生成VBT因子变体"""
        variants = []

        for base_name, config in self.vbt_indicator_map.items():
            param_ranges = config["param_ranges"]

            if not param_ranges:
                # 无参数指标
                variant = VBTFactorSpec(
                    factor_id=f"VBT_{base_name}",
                    indicator_class=config["class"],
                    parameters={},
                    output_field=config["output"],
                    category=config["category"],
                    description=f"VBT {base_name} indicator"
                )
                variants.append(variant)
            else:
                # 生成参数组合
                param_names = list(param_ranges.keys())
                param_values = list(param_ranges.values())

                for param_combination in itertools.product(*param_values):
                    param_dict = dict(zip(param_names, param_combination))

                    # 生成因子ID
                    param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
                    factor_id = f"VBT_{base_name}_{param_str}"

                    description = f"VBT {base_name} with parameters: {param_dict}"

                    variant = VBTFactorSpec(
                        factor_id=factor_id,
                        indicator_class=config["class"],
                        parameters=param_dict,
                        output_field=config["output"],
                        category=config["category"],
                        description=description
                    )
                    variants.append(variant)

        logger.info(f"生成VBT因子变体: {len(variants)} 个")
        return variants

    def generate_talib_factor_variants(self) -> List[VBTFactorSpec]:
        """生成TA-Lib因子变体"""
        variants = []

        for indicator_name, config in self.talib_indicator_map.items():
            param_ranges = config["param_ranges"]

            if not param_ranges:
                # 无参数指标
                variant = VBTFactorSpec(
                    factor_id=f"TA_{indicator_name}",
                    indicator_class=config["function"],
                    parameters={},
                    output_field=indicator_name.lower(),
                    category=config["category"],
                    description=f"TA-Lib {indicator_name} indicator"
                )
                variants.append(variant)
            else:
                # 生成参数组合
                param_names = list(param_ranges.keys())
                param_values = list(param_ranges.values())

                for param_combination in itertools.product(*param_values):
                    param_dict = dict(zip(param_names, param_combination))

                    # 生成因子ID
                    param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
                    factor_id = f"TA_{indicator_name}_{param_str}"

                    description = f"TA-Lib {indicator_name} with parameters: {param_dict}"

                    variant = VBTFactorSpec(
                        factor_id=factor_id,
                        indicator_class=config["function"],
                        parameters=param_dict,
                        output_field=indicator_name.lower(),
                        category=config["category"],
                        description=description
                    )
                    variants.append(variant)

        logger.info(f"生成TA-Lib因子变体: {len(variants)} 个")
        return variants

    def register_all_factors(self) -> int:
        """注册所有生成的因子"""
        # 生成VBT因子变体
        vbt_variants = self.generate_vbt_factor_variants()

        # 生成TA-Lib因子变体
        talib_variants = self.generate_talib_factor_variants()

        # 合并所有变体
        all_variants = vbt_variants + talib_variants

        logger.info(f"总计划注册因子数: {len(all_variants)}")

        # 注册因子
        success_count = 0

        for variant in all_variants:
            try:
                if variant.factor_id.startswith("VBT_"):
                    # VBT因子
                    factor_func = self._create_vbt_factor_function(variant)
                else:
                    # TA-Lib因子
                    factor_func = self._create_talib_factor_function(
                        variant.indicator_class, variant.parameters
                    )

                # 注册因子
                success = self.registry.register_factor(
                    factor_id=variant.factor_id,
                    function=factor_func,
                    parameters=variant.parameters,
                    category=variant.category,
                    description=variant.description,
                    is_dynamic=True
                )

                if success:
                    success_count += 1

            except Exception as e:
                logger.error(f"注册因子失败 {variant.factor_id}: {str(e)}")
                continue

        logger.info(f"因子注册完成: {success_count}/{len(all_variants)} 个成功")
        return success_count

    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子统计信息"""
        stats = self.registry.get_statistics()
        return stats


@safe_operation
def main():
    """主函数 - 测试因子工厂"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建因子工厂
    factory = ETFFactorFactory()

    # 注册所有因子
    success_count = factory.register_all_factors()

    # 获取统计信息
    stats = factory.get_factor_statistics()

    print(f"因子注册成功: {success_count} 个")
    print(f"注册表统计: {stats}")


if __name__ == "__main__":
    main()