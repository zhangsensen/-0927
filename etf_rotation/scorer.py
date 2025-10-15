"""ETF横截面评分系统"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ETFScorer:
    """ETF评分器：横截面标准化+加权评分"""

    def __init__(
        self,
        weights: dict = None,
        config_file: str = None,
        factor_selection: str = "equal_weight",
        correlation_threshold: float = 0.9,
    ):
        """
        初始化评分器

        Args:
            weights: 因子权重字典，如 {"Momentum252": 0.4, "Momentum126": 0.3, ...}
            config_file: 配置文件路径，支持动态因子选择
            factor_selection: 因子选择方式 ("equal_weight", "ic_weight", "manual")
            correlation_threshold: 相关性剔除阈值，>该值则剔除
        """
        self.factor_selection = factor_selection
        self.correlation_threshold = correlation_threshold

        if weights is not None:
            self.weights = weights
            self.factor_selection = "manual"
        elif config_file:
            self.weights = self._load_config(config_file)
        else:
            raise ValueError("必须提供weights或config_file参数")

        logger.info(
            f"评分器初始化: {len(self.weights)} 个因子, 选择方式: {self.factor_selection}"
        )

    def _load_config(self, config_file: str) -> dict:
        """从配置文件加载因子权重"""
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # 获取因子列表
        factors = config.get("factors", [])
        if not factors and "factor_set_file" in config:
            # 从因子集文件加载因子列表
            factor_set_file = config["factor_set_file"]
            with open(factor_set_file) as f:
                factor_set = yaml.safe_load(f)
            factors = factor_set.get("factors", [])

        if not factors:
            raise ValueError(f"无法从配置文件 {config_file} 中找到因子列表")

        if self.factor_selection == "equal_weight":
            # 等权重
            weight = 1.0 / len(factors)
            return {factor: weight for factor in factors}
        elif self.factor_selection == "ic_weight":
            # IC加权（简化版，实际应该从历史IC计算）
            # 这里使用因子类别权重作为示例
            momentum_weight = 0.3
            trend_weight = 0.25
            oscillator_weight = 0.2
            volume_weight = 0.15
            volatility_weight = 0.1

            weights = {}
            for factor in factors:
                if any(
                    kw in factor.lower() for kw in ["momentum", "mom_", "roc", "rocp"]
                ):
                    weights[factor] = momentum_weight
                elif any(kw in factor.lower() for kw in ["ma", "ema", "sma", "trend"]):
                    weights[factor] = trend_weight
                elif any(
                    kw in factor.lower() for kw in ["rsi", "stoch", "willr", "cci"]
                ):
                    weights[factor] = oscillator_weight
                elif any(kw in factor.lower() for kw in ["obv", "volume_", "vwap"]):
                    weights[factor] = volume_weight
                elif any(kw in factor.lower() for kw in ["atr"]):
                    weights[factor] = volatility_weight
                else:
                    weights[factor] = 0.1  # 默认权重

            # 归一化权重
            total_weight = sum(weights.values())
            return {k: v / total_weight for k, v in weights.items()}
        else:
            raise ValueError(f"不支持的因子选择方式: {self.factor_selection}")

    def score(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        """
        横截面z-score标准化+相关性剔除+加权评分

        Args:
            factor_panel: 因子面板（index=ETF代码，columns=因子值）

        Returns:
            评分后的DataFrame，包含composite_score列
        """
        if factor_panel.empty:
            logger.warning("因子面板为空")
            return pd.DataFrame()

        scored = factor_panel.copy()

        # 1. 因子可用性检查和覆盖率筛选
        available_factors = []
        factor_coverage = {}

        for factor in self.weights.keys():
            if factor not in scored.columns:
                logger.warning(f"因子 {factor} 不在面板中，跳过")
                continue

            coverage = scored[factor].notna().mean()
            factor_coverage[factor] = coverage

            if coverage >= 0.8:  # 覆盖率阈值80%
                available_factors.append(factor)
            else:
                logger.warning(f"因子 {factor} 覆盖率 {coverage:.1%} < 80%，跳过")

        logger.info(
            f"因子覆盖率筛选: {len(self.weights)} -> {len(available_factors)} 个"
        )

        if not available_factors:
            logger.error("没有可用因子，无法评分")
            return pd.DataFrame()

        # 2. 相关性剔除
        selected_factors = self._correlation_filter(scored, available_factors)
        logger.info(
            f"相关性剔除: {len(available_factors)} -> {len(selected_factors)} 个因子"
        )

        # 3. 横截面标准化
        for factor in selected_factors:
            # Winsorize 1%/99%
            q1, q99 = scored[factor].quantile([0.01, 0.99])
            winsorized = np.clip(scored[factor], q1, q99)

            # Z-score标准化
            mean = winsorized.mean()
            std = winsorized.std()
            if std > 0:
                scored[f"{factor}_z"] = (winsorized - mean) / std
            else:
                scored[f"{factor}_z"] = 0.0

        # 4. 加权评分（使用调整后的权重）
        scored["composite_score"] = 0.0
        total_weight = sum(self.weights.get(f, 0) for f in selected_factors)

        for factor in selected_factors:
            weight = self.weights.get(factor, 0)
            if total_weight > 0:
                normalized_weight = weight / total_weight
            else:
                normalized_weight = 1.0 / len(selected_factors)

            z_col = f"{factor}_z"
            if z_col in scored.columns:
                scored["composite_score"] += normalized_weight * scored[z_col]

        # 5. 过滤：绝对动量>0（12个月收益为正）
        if "Momentum252" in scored.columns and scored["Momentum252"].notna().any():
            before_filter = len(scored)
            scored = scored[scored["Momentum252"] > 0]
            logger.info(
                f"绝对动量过滤：{before_filter} -> {len(scored)} 只ETF（12月收益>0）"
            )
        else:
            logger.warning("Momentum252数据不足，跳过绝对动量过滤")

        # 6. 按评分排序
        scored = scored.sort_values("composite_score", ascending=False)

        # 记录使用的因子
        logger.info(f"最终使用因子: {', '.join(selected_factors)}")
        logger.info(f"评分完成：{len(scored)} 只ETF通过筛选")

        return scored

    def _correlation_filter(self, data: pd.DataFrame, factors: list) -> list:
        """
        因子相关性剔除（分桶+贪心去重）

        策略：
        1. 分桶：按因子类别分桶（趋势/动量/摆动/波动/量价）
        2. 桶内去重：按权重排序，贪心剔除ρ>0.7的因子
        3. 跨桶去重：在保留集合上再次去重
        4. 最少独立因子门槛：≥8，不足则回退核心白名单

        Args:
            data: 因子数据
            factors: 候选因子列表

        Returns:
            剔除高相关因子后的列表
        """
        if len(factors) <= 1:
            return factors

        # 计算因子间相关性矩阵
        factor_data = data[factors].dropna()
        if len(factor_data) < 10:
            logger.warning("数据样本不足10个，跳过相关性剔除")
            return factors

        corr_matrix = factor_data.corr().abs()

        # 1. 分桶
        buckets = self._bucket_factors(factors)
        logger.debug(
            f"因子分桶: {', '.join([f'{k}({len(v)})' for k, v in buckets.items()])}"
        )

        # 2. 桶内去重
        bucket_representatives = {}
        for bucket_name, bucket_factors in buckets.items():
            if not bucket_factors:
                continue

            # 按权重排序（权重大的优先保留）
            sorted_factors = sorted(
                bucket_factors, key=lambda f: self.weights.get(f, 0), reverse=True
            )

            # 贪心去重
            selected = []
            for factor in sorted_factors:
                # 检查与已选因子的相关性
                is_redundant = False
                for selected_factor in selected:
                    if (
                        factor in corr_matrix.index
                        and selected_factor in corr_matrix.columns
                    ):
                        corr = corr_matrix.loc[factor, selected_factor]
                        if corr > 0.7:  # 桶内阈值0.7
                            is_redundant = True
                            logger.debug(
                                f"桶内剔除 {bucket_name}: {factor} (corr={corr:.2f} with {selected_factor})"
                            )
                            break

                if not is_redundant:
                    selected.append(factor)

            bucket_representatives[bucket_name] = selected
            logger.debug(
                f"桶 {bucket_name}: {len(bucket_factors)} -> {len(selected)} 个因子"
            )

        # 3. 合并桶内代表
        all_selected = []
        for bucket_factors in bucket_representatives.values():
            all_selected.extend(bucket_factors)

        # 4. 跨桶去重
        final_selected = []
        for factor in all_selected:
            is_redundant = False
            for selected_factor in final_selected:
                if (
                    factor in corr_matrix.index
                    and selected_factor in corr_matrix.columns
                ):
                    corr = corr_matrix.loc[factor, selected_factor]
                    if corr > self.correlation_threshold:  # 跨桶阈值（用户配置）
                        is_redundant = True
                        logger.debug(
                            f"跨桶剔除: {factor} (corr={corr:.2f} with {selected_factor})"
                        )
                        break

            if not is_redundant:
                final_selected.append(factor)

        # 5. 最少独立因子门槛检查
        min_independent_factors = 8
        if len(final_selected) < min_independent_factors:
            logger.warning(
                f"独立因子数 {len(final_selected)} < {min_independent_factors}，"
                f"回退至核心白名单"
            )
            # 回退：保留权重最大的8个因子
            core_factors = sorted(
                factors, key=lambda f: self.weights.get(f, 0), reverse=True
            )[:min_independent_factors]
            return core_factors

        logger.info(f"相关性剔除: {len(factors)} -> {len(final_selected)} 个独立因子")
        return final_selected

    def _bucket_factors(self, factors: list) -> dict:
        """
        因子分桶（按类别）

        Args:
            factors: 因子列表

        Returns:
            分桶字典 {bucket_name: [factors]}
        """
        buckets = {
            "momentum": [],  # 动量类
            "trend": [],  # 趋势类
            "volatility": [],  # 波动类
            "risk": [],  # 风险类
            "oscillator": [],  # 摆动类
            "volume": [],  # 量价类
            "other": [],  # 其他
        }

        for factor in factors:
            factor_lower = factor.lower()

            # 动量类
            if any(kw in factor_lower for kw in ["momentum", "mom_", "roc", "rocp"]):
                buckets["momentum"].append(factor)
            # 趋势类
            elif any(
                kw in factor_lower
                for kw in ["ma", "ema", "sma", "trend", "adx", "aroon"]
            ):
                buckets["trend"].append(factor)
            # 摆动类
            elif any(
                kw in factor_lower for kw in ["rsi", "stoch", "willr", "cci", "mfi"]
            ):
                buckets["oscillator"].append(factor)
            # 波动类
            elif any(kw in factor_lower for kw in ["volatility", "atr", "std", "bb_"]):
                buckets["volatility"].append(factor)
            # 风险类
            elif any(kw in factor_lower for kw in ["drawdown", "dd_", "risk"]):
                buckets["risk"].append(factor)
            # 量价类
            elif any(kw in factor_lower for kw in ["obv", "volume_", "vwap", "ad_"]):
                buckets["volume"].append(factor)
            else:
                buckets["other"].append(factor)

        # 移除空桶
        return {k: v for k, v in buckets.items() if v}
