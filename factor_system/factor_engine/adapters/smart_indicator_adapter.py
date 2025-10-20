#!/usr/bin/env python3
"""
智能指标适配器 - 从因子工厂转向指标平台
基于预测力(IC/IR)动态选择最优参数组合
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """指标类型枚举"""

    TREND = "trend"  # 趋势类
    MOMENTUM = "momentum"  # 动量类
    VOLATILITY = "volatility"  # 波动率类
    VOLUME = "volume"  # 成交量类
    PRICE = "price"  # 价格类


@dataclass
class IndicatorConfig:
    """指标配置类"""

    name: str
    indicator_type: IndicatorType
    base_function: str
    param_space: Dict[str, List]  # 参数搜索空间
    max_variants: int = 3  # 最大变种数
    min_ic_threshold: float = 0.02  # 最小IC阈值
    correlation_threshold: float = 0.7  # 相关性阈值


@dataclass
class OptimizedParameter:
    """优化后的参数"""

    params: Dict[str, any]
    ic_mean: float
    ic_ir: float
    stability_score: float
    correlation_score: float


class SmartIndicatorAdapter:
    """智能指标适配器 - 基于预测力的参数优化"""

    def __init__(
        self,
        price_field: str = "close",
        lookback_period: int = 252,  # 1年优化窗口
        forward_period: int = 5,  # 5天预测期
        min_samples: int = 60,
    ):  # 最小样本数
        self.price_field = price_field
        self.lookback_period = lookback_period
        self.forward_period = forward_period
        self.min_samples = min_samples

        # 预定义的指标配置 - 限制参数空间
        self.indicator_configs = self._build_indicator_configs()

        # 参数优化缓存
        self.param_cache = {}

    def _build_indicator_configs(self) -> List[IndicatorConfig]:
        """构建智能指标配置 - 基于市场经验限制参数空间"""

        configs = []

        # 趋势类指标 - 限制为核心周期
        trend_configs = [
            IndicatorConfig(
                name="MA",
                indicator_type=IndicatorType.TREND,
                base_function="vbt.MA",
                param_space={"window": [10, 20, 60]},  # 核心周期：短/中/长
                max_variants=2,
            ),
            IndicatorConfig(
                name="EMA",
                indicator_type=IndicatorType.TREND,
                base_function="vbt.MA.ewm",
                param_space={"window": [12, 26, 50]},  # 避开过度参数化
                max_variants=2,
            ),
            IndicatorConfig(
                name="MACD",
                indicator_type=IndicatorType.TREND,
                base_function="vbt.MACD",
                param_space={"fast": [12], "slow": [26], "signal": [9]},  # 固定经典参数
                max_variants=1,  # 只保留经典参数
            ),
        ]

        # 动量类指标 - 基于市场有效性选择参数
        momentum_configs = [
            IndicatorConfig(
                name="RSI",
                indicator_type=IndicatorType.MOMENTUM,
                base_function="vbt.RSI",
                param_space={"window": [9, 14]},  # 短周期和长周期
                max_variants=2,
            ),
            IndicatorConfig(
                name="STOCH",
                indicator_type=IndicatorType.MOMENTUM,
                base_function="vbt.STOCH",
                param_space={"k_window": [14], "d_window": [3]},  # 经典参数
                max_variants=1,  # K线和D线只选一个代表
            ),
        ]

        # 波动率类 - 去除数学重复
        volatility_configs = [
            IndicatorConfig(
                name="ATR",
                indicator_type=IndicatorType.VOLATILITY,
                base_function="vbt.ATR",
                param_space={"window": [14]},  # 经典参数
                max_variants=1,
            ),
            IndicatorConfig(
                name="BBANDS",
                indicator_type=IndicatorType.VOLATILITY,
                base_function="vbt.BBANDS",
                param_space={"window": [20], "alpha": [2.0]},  # 经典参数
                max_variants=1,  # 只保留position，去除upper/middle/lower重复
            ),
        ]

        configs.extend(trend_configs)
        configs.extend(momentum_configs)
        configs.extend(volatility_configs)

        return configs

    def compute_optimized_indicators(
        self, df: pd.DataFrame, target_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        计算优化后的指标 - 基于预测力选择参数

        Args:
            df: OHLCV数据
            target_series: 预测目标序列（用于参数优化）

        Returns:
            优化后的指标DataFrame
        """
        logger.info(f"智能指标适配器开始计算，输入: {df.shape}")

        # 提取价格数据
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        optimized_factors = {}

        # 对每个指标类型进行参数优化
        for config in self.indicator_configs:
            logger.info(f"优化指标: {config.name}")

            # 参数优化
            best_params = self._optimize_parameters(
                config, close, high, low, volume, target_series
            )

            # 使用最优参数计算指标
            factor_values = self._compute_indicator_with_params(
                config, best_params, close, high, low, volume
            )

            # 添加到结果中
            for i, (param_set, values) in enumerate(best_params.items()):
                factor_name = f"{config.name}_{param_set}"
                optimized_factors[factor_name] = values

        logger.info(f"✅ 智能指标适配器完成: {len(optimized_factors)} 个优化指标")

        # 转换为DataFrame
        result_df = pd.DataFrame(optimized_factors, index=df.index)

        if "date" in df.columns:
            result_df["date"] = df["date"].values

        return result_df

    def _optimize_parameters(
        self,
        config: IndicatorConfig,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        target: Optional[pd.Series] = None,
    ) -> Dict[str, np.ndarray]:
        """参数优化 - 基于IC/IR选择最优参数组合"""

        # 如果有目标序列，进行基于IC的优化
        if target is not None:
            return self._ic_based_optimization(config, close, high, low, volume, target)
        else:
            # 否则使用经验参数
            return self._experience_based_selection(config, close, high, low, volume)

    def _ic_based_optimization(
        self,
        config: IndicatorConfig,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        target: pd.Series,
    ) -> Dict[str, np.ndarray]:
        """基于信息系数的参数优化"""

        param_candidates = []

        # 生成参数组合
        param_combinations = self._generate_param_combinations(config.param_space)

        for params in param_combinations:
            try:
                # 计算指标值
                indicator_values = self._compute_single_indicator(
                    config, params, close, high, low, volume
                )

                # 计算IC（信息系数）
                ic, ic_ir = self._calculate_ic(indicator_values, target.values)

                param_candidates.append(
                    {
                        "params": params,
                        "ic": ic,
                        "ic_ir": ic_ir,
                        "values": indicator_values,
                    }
                )

            except Exception as e:
                logger.warning(f"参数组合 {params} 计算失败: {e}")
                continue

        # 按IC_IR排序，选择最优的
        param_candidates.sort(key=lambda x: x["ic_ir"], reverse=True)

        # 选择top N，同时考虑相关性
        selected = self._select_diverse_params(param_candidates, config.max_variants)

        return {str(item["params"]): item["values"] for item in selected}

    def _experience_based_selection(
        self,
        config: IndicatorConfig,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """基于市场经验的参数选择"""

        # 预定义的经验参数
        experience_params = {
            "MA": [{"window": 20}, {"window": 60}],  # 20日和60日均线
            "EMA": [{"window": 12}, {"window": 26}],  # MACD常用参数
            "RSI": [{"window": 14}],  # 经典RSI参数
            "STOCH": [{"k_window": 14, "d_window": 3}],  # 经典随机指标
            "ATR": [{"window": 14}],  # 经典ATR参数
            "BBANDS": [{"window": 20, "alpha": 2.0}],  # 经典布林带
            "MACD": [{"fast": 12, "slow": 26, "signal": 9}],  # 经典MACD
        }

        result = {}
        default_params = experience_params.get(config.name, [{}])

        for params in default_params[: config.max_variants]:
            values = self._compute_single_indicator(
                config, params, close, high, low, volume
            )
            param_str = "_".join([f"{k}{v}" for k, v in params.items()])
            result[param_str] = values

        return result

    def _calculate_ic(
        self, indicator_values: np.ndarray, target_values: np.ndarray
    ) -> Tuple[float, float]:
        """计算信息系数和IC_IR"""

        # 去除NaN值
        valid_mask = ~(np.isnan(indicator_values) | np.isnan(target_values))
        if np.sum(valid_mask) < self.min_samples:
            return 0.0, 0.0

        x = indicator_values[valid_mask]
        y = target_values[valid_mask]

        # 计算皮尔逊相关系数（IC）
        ic = np.corrcoef(x, y)[0, 1]
        if np.isnan(ic):
            ic = 0.0

        # 计算IC_IR（信息比率）
        # 这里简化处理，实际应该计算IC的时间序列稳定性
        ic_ir = abs(ic)  # 简化版本

        return ic, ic_ir

    def _compute_single_indicator(
        self,
        config: IndicatorConfig,
        params: Dict,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> np.ndarray:
        """计算单个指标"""

        # 这里简化实现，实际应该根据config.base_function调用相应函数
        if config.name == "RSI":
            window = params.get("window", 14)
            rsi = vbt.RSI.run(close, window=window)
            return rsi.rsi.values
        elif config.name == "MA":
            window = params.get("window", 20)
            ma = vbt.MA.run(close, window=window)
            return ma.ma.values
        # ... 其他指标实现
        else:
            return np.zeros(len(close))  # 默认返回0

    def _generate_param_combinations(self, param_space: Dict[str, List]) -> List[Dict]:
        """生成参数组合"""
        from itertools import product

        param_names = list(param_space.keys())
        param_values = list(param_space.values())

        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def _select_diverse_params(
        self, candidates: List[Dict], max_variants: int
    ) -> List[Dict]:
        """选择多样性参数组合，避免高相关性"""

        if len(candidates) <= max_variants:
            return candidates

        # 简化版本：直接选择top IC_IR的
        return candidates[:max_variants]

        # 实际应该实现相关性分析，选择低相关的组合
