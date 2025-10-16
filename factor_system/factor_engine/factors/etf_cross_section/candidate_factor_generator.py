#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面候选因子生成器
基于现有269个技术指标生成参数变体，构建800-1200个候选因子库
"""

import pandas as pd
import numpy as np
import itertools
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

from factor_system.factor_engine import api
from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


@dataclass
class FactorVariant:
    """因子变体定义"""
    base_factor_id: str
    variant_id: str
    parameters: Dict[str, Any]
    category: str
    description: str


class ETFCandidateFactorGenerator:
    """ETF候选因子生成器"""

    def __init__(self):
        """初始化因子生成器"""
        self.available_factors = api.list_available_factors()
        self.factor_categories = self._categorize_factors()
        self.variant_configs = self._create_variant_configs()

    def _categorize_factors(self) -> Dict[str, List[str]]:
        """因子分类"""
        categories = {
            "momentum": [],
            "mean_reversion": [],
            "volume": [],
            "volatility": [],
            "trend": [],
            "overlap": [],
            "candlestick": []
        }

        for factor in self.available_factors:
            factor_id = factor  # factor 已经是因子ID字符串，不需要从字典中提取

            # 基于因子ID进行分类
            if any(keyword in factor_id.lower() for keyword in ["rsi", "macd", "sto", "mom", "roc", "adx", "aroon"]):
                categories["momentum"].append(factor_id)
            elif any(keyword in factor_id.lower() for keyword in ["bb_", "bollinger", "keltner"]):
                categories["mean_reversion"].append(factor_id)
            elif any(keyword in factor_id.lower() for keyword in ["vol", "obv", "vwap", "ad"]):
                categories["volume"].append(factor_id)
            elif any(keyword in factor_id.lower() for keyword in ["atr", "std", "tr"]):
                categories["volatility"].append(factor_id)
            elif any(keyword in factor_id.lower() for keyword in ["ma_", "ema", "sma", "wma", "kama"]):
                categories["trend"].append(factor_id)
            elif any(keyword in factor_id.lower() for keyword in ["cdl", "candle"]):
                categories["candlestick"].append(factor_id)
            else:
                categories["overlap"].append(factor_id)

        return categories

    def _create_variant_configs(self) -> Dict[str, Dict[str, List]]:
        """创建因子变体配置"""
        configs = {
            # 动量类因子参数变体
            "momentum": {
                "RSI": {"timeperiod": [7, 14, 21, 28]},
                "STOCH": {"k": [9, 14, 20], "d": [3, 5, 6]},
                "MACD": {"fast": [12, 15, 18], "slow": [26, 29, 32], "signal": [9, 12]},
                "MFI": {"timeperiod": [7, 14, 21]},
                "ROC": {"timeperiod": [10, 20, 60]},
                "ADX": {"timeperiod": [7, 14, 21]},
                "AROON": {"timeperiod": [14, 21, 28]},
                "WILLR": {"timeperiod": [7, 14, 21]},
                "CCI": {"timeperiod": [14, 21, 28]},
                "MOM": {"timeperiod": [10, 20, 60]},
                "APO": {"fast": [12, 26], "slow": [26, 50]},
                "PPO": {"fast": [12, 26], "slow": [26, 50]},
                "ULTOSC": {"fastperiod": [7, 14], "middleperiod": [14, 21], "slowperiod": [28]},
                "STOCHRSI": {"timeperiod": [14, 21], "fastk_period": [5, 9], "fastd_period": [3, 6]},
                "STOCHF": {"fastk_period": [5, 9], "fastd_period": [3, 6]},
                "RSI100": {"timeperiod": [14, 21, 28]},
                "RSX": {"timeperiod": [14, 21]}
            },

            # 均值回归类因子参数变体
            "mean_reversion": {
                "BB_": {"period": [10, 15, 20], "std": [1.5, 2.0, 2.5]},
                "KELTNER": {"ema_period": [20, 30], "atr_period": [10, 20], "multiplier": [1.5, 2.0]},
                "DONCHIAN": {"period": [20, 30, 50]},
                "STDEV": {"period": [20, 30, 50], "nbdev": [1.0, 1.5, 2.0]},
                "STDEV2": {"period": [20, 30, 50], "nbdev": [1.0, 1.5, 2.0]}
            },

            # 趋势类因子参数变体
            "trend": {
                "MA3": {"timeperiod": [5, 10, 20, 50, 120]},
                "MA5": {"timeperiod": [5, 10, 20, 50, 120]},
                "MA10": {"timeperiod": [10, 20, 50, 120]},
                "EMA": {"timeperiod": [5, 10, 20, 50, 120]},
                "DEMA": {"timeperiod": [5, 10, 20, 50, 120]},
                "TEMA": {"timeperiod": [5, 10, 20, 50, 120]},
                "KAMA": {"timeperiod": [10, 20, 50, 120]},
                "WMA": {"timeperiod": [5, 10, 20, 50, 120]},
                "TRIMA": {"timeperiod": [10, 20, 50, 120]},
                "T3": {"timeperiod": [5, 10, 20, 50]},
                "SAR": {"acceleration": [0.02, 0.04], "maximum": [0.2, 0.4]},
                "HT_SINE": {"sineperiod": [14, 21, 28]},
                "HT_TRENDLINE": {"trendperiod": [14, 21, 28]}
            },

            # 成交量类因子参数变体
            "volume": {
                "OBV": {},  # 无参数
                "OBV_SMA": {"timeperiod": [10, 20, 50]},
                "OBV_EMA": {"timeperiod": [10, 20, 50]},
                "AD": {},   # 无参数
                "ADOSC": {"fastperiod": [3, 7], "slowperiod": [10, 21]},
                "VWAP": {"window": [10, 20, 50]},
                "Volume_Ratio": {"short_period": [5, 10], "long_period": [20, 50]},
                "Volume_Momentum": {"period": [10, 20, 50]},
                "MFI": {"timeperiod": [7, 14, 21]},
                "NVIT": {"period": [10, 20, 50]},
                "VPT": {"period": [10, 20, 50]},
                "PVI": {"period": [10, 20, 50]},
                "NVI": {"period": [10, 20, 50]},
                "EOM": {"period": [14, 21, 28]},
                "MFIAD": {"period": [14, 21]},
                "EMV": {"period": [14, 21]},
                "CMF": {"period": [20, 30]},
                "VWAP_D": {"period": [20, 30]},
                "VWAP_M": {"period": [20, 30]},
                "VWAP_S": {"period": [20, 30]},
                "VWAP_Y": {"period": [20, 30]},
                "VWAPALL": {"period": [20, 30]},
                "ATRVR": {"period": [14, 21, 28]},
                "ATRVR1": {"period": [14, 21, 28]},
                "ATRVR2": {"period": [14, 21, 28]},
                "ATRVR3": {"period": [14, 21, 28]},
                "ATRVR4": {"period": [14, 21, 28]},
                "ATRVR5": {"period": [14, 21, 28]}
            },

            # 波动率类因子参数变体
            "volatility": {
                "ATR": {"timeperiod": [7, 14, 21]},
                "ATR1": {"timeperiod": [7, 14, 21]},
                "ATR2": {"timeperiod": [7, 14, 21]},
                "ATR3": {"timeperiod": [7, 14, 21]},
                "ATR4": {"timeperiod": [7, 14, 21]},
                "ATR5": {"timeperiod": [7, 14, 21]},
                "ATR6": {"timeperiod": [7, 14, 21]},
                "NATR": {"timeperiod": [7, 14, 21]},
                "MSTD": {"timeperiod": [20, 30, 50]},
                "FSTD": {"timeperiod": [20, 30, 50]},
                "TRANGE": {},  # 无参数
                "VHF": {"timeperiod": [14, 21, 28]},
                "STDDEV": {"timeperiod": [20, 30, 50], "nbdev": [1.0, 1.5, 2.0]},
                "VAR": {"timeperiod": [20, 30, 50]},
                "Kurtosis": {"timeperiod": [20, 30, 50]},
                "Skew": {"timeperiod": [20, 30, 50]}
            }
        }

        return configs

    def generate_variants_for_category(self, category: str) -> List[FactorVariant]:
        """为指定类别生成因子变体"""
        if category not in self.factor_categories:
            return []

        variants = []
        category_factors = self.factor_categories[category]
        config = self.variant_configs.get(category, {})

        for factor_id in category_factors:
            # 查找匹配的配置
            matching_config = None
            for config_key, params in config.items():
                if config_key in factor_id:
                    matching_config = (config_key, params)
                    break

            if matching_config:
                config_key, params = matching_config
                variants.extend(self._generate_param_variants(factor_id, config_key, params, category))
            else:
                # 无参数因子，保留原始
                variants.append(FactorVariant(
                    base_factor_id=factor_id,
                    variant_id=factor_id,
                    parameters={},
                    category=category,
                    description=f"Original {category} factor"
                ))

        return variants

    def _generate_param_variants(self, factor_id: str, config_key: str,
                                params: Dict[str, List], category: str) -> List[FactorVariant]:
        """生成参数变体"""
        variants = []

        if not params:
            # 无参数因子
            variants.append(FactorVariant(
                base_factor_id=factor_id,
                variant_id=factor_id,
                parameters={},
                category=category,
                description=f"Original {category} factor"
            ))
            return variants

        # 生成所有参数组合
        param_names = list(params.keys())
        param_values = list(params.values())

        for param_combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, param_combination))

            # 生成变体ID
            param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
            variant_id = f"{factor_id}_{param_str}"

            description = f"{factor_id} with parameters: {param_dict}"

            variants.append(FactorVariant(
                base_factor_id=factor_id,
                variant_id=variant_id,
                parameters=param_dict,
                category=category,
                description=description
            ))

        return variants

    def generate_all_variants(self) -> List[FactorVariant]:
        """生成所有因子变体"""
        all_variants = []

        for category in self.factor_categories.keys():
            logger.info(f"生成 {category} 类别因子变体...")
            variants = self.generate_variants_for_category(category)
            all_variants.extend(variants)
            logger.info(f"{category} 类别生成 {len(variants)} 个变体")

        logger.info(f"总计生成 {len(all_variants)} 个因子变体")
        return all_variants

    def save_variants_to_file(self, variants: List[FactorVariant], output_path: str):
        """保存因子变体到文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        data = []
        for variant in variants:
            data.append({
                "variant_id": variant.variant_id,
                "base_factor_id": variant.base_factor_id,
                "category": variant.category,
                "parameters": str(variant.parameters),
                "description": variant.description
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"因子变体已保存到: {output_file}")

    def create_factor_calculation_mapping(self, variants: List[FactorVariant]) -> Dict[str, FactorVariant]:
        """创建因子计算映射"""
        mapping = {}
        for variant in variants:
            mapping[variant.variant_id] = variant
        return mapping


@safe_operation
def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("开始生成ETF横截面候选因子...")

    # 初始化生成器
    generator = ETFCandidateFactorGenerator()

    # 显示基础统计
    logger.info("=== 基础统计信息 ===")
    total_base_factors = len(generator.available_factors)
    logger.info(f"基础因子数量: {total_base_factors}")

    for category, factors in generator.factor_categories.items():
        logger.info(f"{category}: {len(factors)} 个因子")

    # 生成所有变体
    variants = generator.generate_all_variants()

    # 显示变体统计
    logger.info("\n=== 变体统计信息 ===")
    for category in generator.factor_categories.keys():
        category_variants = [v for v in variants if v.category == category]
        logger.info(f"{category}: {len(category_variants)} 个变体")

    logger.info(f"\n总计候选因子: {len(variants)}")

    # 保存到文件
    output_dir = Path("factor_system/factor_engine/factors/etf_cross_section")
    generator.save_variants_to_file(variants, str(output_dir / "candidate_factors.csv"))

    # 创建映射
    mapping = generator.create_factor_calculation_mapping(variants)
    logger.info(f"因子计算映射包含 {len(mapping)} 个因子")

    # 显示示例
    logger.info("\n=== 示例因子变体 ===")
    sample_variants = variants[:10]
    for variant in sample_variants:
        logger.info(f"{variant.variant_id}: {variant.description}")

    logger.info("候选因子生成完成！")
    return variants


if __name__ == "__main__":
    main()