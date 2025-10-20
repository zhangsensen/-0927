#!/usr/bin/env python3
"""
指标预筛选系统 - 在指标生成阶段就进行质量控制
基于统计显著性、稳定性、多样性等维度预筛选
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class IndicatorQualityMetrics:
    """指标质量评估指标"""

    name: str
    ic_mean: float  # 平均信息系数
    ic_std: float  # IC标准差
    ic_ir: float  # 信息比率
    positive_ratio: float  # 正IC比例
    t_stat: float  # t统计量
    p_value: float  # p值
    stability_score: float  # 稳定性得分
    uniqueness_score: float  # 独特性得分
    coverage: float  # 数据覆盖率
    missing_ratio: float  # 缺失值比例


class IndicatorPrescreener:
    """指标预筛选器 - 指标生成阶段的质量控制"""

    def __init__(
        self,
        min_ic_threshold: float = 0.01,  # 最小IC阈值
        min_ic_ir_threshold: float = 0.1,  # 最小IC_IR阈值
        max_missing_ratio: float = 0.3,  # 最大缺失率
        min_coverage: float = 0.7,  # 最小覆盖率
        correlation_threshold: float = 0.8,  # 相关性阈值
        min_samples: int = 60,
    ):  # 最小样本数

        self.min_ic_threshold = min_ic_threshold
        self.min_ic_ir_threshold = min_ic_ir_threshold
        self.max_missing_ratio = max_missing_ratio
        self.min_coverage = min_coverage
        self.correlation_threshold = correlation_threshold
        self.min_samples = min_samples

    def prescreen_indicators(
        self,
        indicator_df: pd.DataFrame,
        target_series: pd.Series,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, any]:
        """
        预筛选指标 - 多维度质量评估

        Args:
            indicator_df: 指标DataFrame
            target_series: 目标变量序列
            reference_date: 参考日期（用于时间序列分析）

        Returns:
            筛选结果字典
        """
        logger.info(f"开始指标预筛选，输入指标数: {indicator_df.shape[1]}")

        # 1. 基础质量筛选
        basic_qualified = self._basic_quality_screen(indicator_df, target_series)
        logger.info(f"基础质量筛选后: {len(basic_qualified)} 个指标")

        # 2. 预测力筛选
        predictive_qualified = self._predictive_power_screen(
            indicator_df[basic_qualified], target_series
        )
        logger.info(f"预测力筛选后: {len(predictive_qualified)} 个指标")

        # 3. 稳定性筛选
        if reference_date:
            stability_qualified = self._stability_screen(
                indicator_df[predictive_qualified], reference_date
            )
            logger.info(f"稳定性筛选后: {len(stability_qualified)} 个指标")
        else:
            stability_qualified = predictive_qualified

        # 4. 多样性筛选（去除高相关性）
        diversity_qualified = self._diversity_screen(indicator_df[stability_qualified])
        logger.info(f"多样性筛选后: {len(diversity_qualified)} 个指标")

        # 5. 生成质量报告
        quality_report = self._generate_quality_report(
            indicator_df,
            basic_qualified,
            predictive_qualified,
            stability_qualified,
            diversity_qualified,
            target_series,
        )

        return {
            "qualified_indicators": diversity_qualified,
            "rejected_indicators": list(
                set(indicator_df.columns) - set(diversity_qualified)
            ),
            "quality_metrics": self._calculate_quality_metrics(
                indicator_df[diversity_qualified], target_series
            ),
            "quality_report": quality_report,
            "reduction_ratio": (len(indicator_df.columns) - len(diversity_qualified))
            / len(indicator_df.columns),
        }

    def _basic_quality_screen(
        self, indicator_df: pd.DataFrame, target_series: pd.Series
    ) -> List[str]:
        """基础质量筛选"""
        qualified = []

        for column in indicator_df.columns:
            series = indicator_df[column]

            # 检查缺失率
            missing_ratio = series.isna().sum() / len(series)
            if missing_ratio > self.max_missing_ratio:
                logger.debug(
                    f"{column} 被拒绝: 缺失率 {missing_ratio:.2%} > {self.max_missing_ratio:.2%}"
                )
                continue

            # 检查覆盖率（非零值比例）
            if series.dtype in ["float64", "int64"]:
                non_zero_ratio = (series != 0).sum() / len(series)
                if non_zero_ratio < self.min_coverage:
                    logger.debug(
                        f"{column} 被拒绝: 覆盖率 {non_zero_ratio:.2%} < {self.min_coverage:.2%}"
                    )
                    continue

            # 检查方差（避免常数指标）
            if series.var() < 1e-10:
                logger.debug(f"{column} 被拒绝: 方差过小")
                continue

            # 检查样本数
            valid_samples = series.dropna()
            if len(valid_samples) < self.min_samples:
                logger.debug(
                    f"{column} 被拒绝: 样本数 {len(valid_samples)} < {self.min_samples}"
                )
                continue

            qualified.append(column)

        return qualified

    def _predictive_power_screen(
        self, indicator_df: pd.DataFrame, target_series: pd.Series
    ) -> List[str]:
        """预测力筛选 - 基于IC和统计显著性"""
        qualified = []

        for column in indicator_df.columns:
            indicator_series = indicator_df[column]

            # 对齐数据
            aligned_data = pd.DataFrame(
                {"indicator": indicator_series, "target": target_series}
            ).dropna()

            if len(aligned_data) < self.min_samples:
                continue

            # 计算IC（信息系数）
            ic = aligned_data["indicator"].corr(aligned_data["target"])

            # 计算IC的统计显著性
            if not np.isnan(ic) and abs(ic) >= self.min_ic_threshold:
                # t检验
                t_stat, p_value = stats.pearsonr(
                    aligned_data["indicator"], aligned_data["target"]
                )

                if p_value < 0.05:  # 5%显著性水平
                    ic_ir = abs(ic) / max(
                        1e-6, np.std(aligned_data["indicator"])
                    )  # 简化IC_IR

                    if ic_ir >= self.min_ic_ir_threshold:
                        qualified.append(column)
                        logger.debug(
                            f"{column} 通过预测力筛选: IC={ic:.4f}, IC_IR={ic_ir:.4f}, p={p_value:.4f}"
                        )
                    else:
                        logger.debug(
                            f"{column} IC_IR不足: {ic_ir:.4f} < {self.min_ic_ir_threshold}"
                        )
                else:
                    logger.debug(f"{column} 统计不显著: p={p_value:.4f}")
            else:
                logger.debug(f"{column} IC不足: {ic:.4f} < {self.min_ic_threshold}")

        return qualified

    def _stability_screen(
        self, indicator_df: pd.DataFrame, reference_date: pd.Timestamp
    ) -> List[str]:
        """稳定性筛选 - 时间序列稳定性分析"""
        qualified = []

        for column in indicator_df.columns:
            series = indicator_df[column]

            # 计算滚动IC（简化版本）
            rolling_window = min(60, len(series) // 4)  # 自适应窗口

            if rolling_window < 20:
                qualified.append(column)  # 数据不足，先保留
                continue

            # 计算滚动标准差（稳定性代理）
            rolling_std = series.rolling(rolling_window).std()

            # 稳定性得分（标准差变异系数）
            if rolling_std.mean() > 0:
                stability_cv = rolling_std.std() / rolling_std.mean()
                if stability_cv < 0.5:  # 稳定性阈值
                    qualified.append(column)
                    logger.debug(f"{column} 通过稳定性筛选: CV={stability_cv:.4f}")
                else:
                    logger.debug(f"{column} 稳定性不足: CV={stability_cv:.4f}")
            else:
                qualified.append(column)  # 默认通过

        return qualified

    def _diversity_screen(self, indicator_df: pd.DataFrame) -> List[str]:
        """多样性筛选 - 去除高相关性指标"""
        if indicator_df.empty or len(indicator_df.columns) <= 1:
            return list(indicator_df.columns)

        # 计算相关性矩阵
        corr_matrix = indicator_df.corr()

        # 按预测力排序（假设IC高的优先保留）
        ic_scores = {}
        for col in indicator_df.columns:
            # 这里简化处理，实际应该传入IC得分
            ic_scores[col] = (
                abs(indicator_df[col].corr(indicator_df.iloc[:, -1]))
                if len(indicator_df.columns) > 1
                else 0.1
            )

        # 排序（IC高的优先）
        sorted_indicators = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)

        selected = []
        for indicator, ic_score in sorted_indicators:
            if indicator not in corr_matrix.columns:
                continue

            # 检查与已选指标的相关性
            is_redundant = False
            for selected_indicator in selected:
                if (
                    selected_indicator in corr_matrix.columns
                    and indicator in corr_matrix.columns
                ):
                    correlation = abs(corr_matrix.loc[indicator, selected_indicator])
                    if correlation > self.correlation_threshold:
                        is_redundant = True
                        logger.debug(
                            f"{indicator} 与 {selected_indicator} 高相关({correlation:.3f})，被去除"
                        )
                        break

            if not is_redundant:
                selected.append(indicator)

        return selected

    def _calculate_quality_metrics(
        self, indicator_df: pd.DataFrame, target_series: pd.Series
    ) -> Dict[str, IndicatorQualityMetrics]:
        """计算指标质量评估"""
        metrics = {}

        for column in indicator_df.columns:
            series = indicator_df[column]

            # 对齐数据计算IC
            aligned_data = pd.DataFrame(
                {"indicator": series, "target": target_series}
            ).dropna()

            if len(aligned_data) < 10:  # 最小样本要求
                continue

            # 计算IC相关指标
            ic = aligned_data["indicator"].corr(aligned_data["target"])
            ic_values = []

            # 滚动IC（时间序列稳定性）
            window_size = min(30, len(aligned_data) // 3)
            if window_size >= 10:
                for start in range(
                    0, len(aligned_data) - window_size + 1, window_size // 2
                ):
                    end = start + window_size
                    if end <= len(aligned_data):
                        window_ic = aligned_data.iloc[start:end]["indicator"].corr(
                            aligned_data.iloc[start:end]["target"]
                        )
                        if not np.isnan(window_ic):
                            ic_values.append(window_ic)

            ic_mean = np.mean(ic_values) if ic_values else ic
            ic_std = np.std(ic_values) if ic_values else 0.1
            ic_ir = abs(ic_mean) / max(ic_std, 1e-6) if ic_std > 0 else 0
            positive_ratio = (
                np.mean([ic_val > 0 for ic_val in ic_values]) if ic_values else (ic > 0)
            )

            # 统计显著性
            if len(aligned_data) > 10:
                t_stat, p_value = stats.pearsonr(
                    aligned_data["indicator"], aligned_data["target"]
                )
            else:
                t_stat, p_value = 0.0, 1.0

            # 稳定性得分（基于IC时间序列）
            stability_score = 1.0 / (1.0 + ic_std) if ic_std > 0 else 0.0

            # 独特性得分（基于相关性分析）
            uniqueness_score = 1.0  # 简化处理

            # 覆盖率和缺失率
            coverage = 1.0 - series.isna().sum() / len(series)
            missing_ratio = series.isna().sum() / len(series)

            metrics[column] = IndicatorQualityMetrics(
                name=column,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_ir=ic_ir,
                positive_ratio=positive_ratio,
                t_stat=t_stat,
                p_value=p_value,
                stability_score=stability_score,
                uniqueness_score=uniqueness_score,
                coverage=coverage,
                missing_ratio=missing_ratio,
            )

        return metrics

    def _generate_quality_report(
        self,
        indicator_df: pd.DataFrame,
        basic_qualified: List[str],
        predictive_qualified: List[str],
        stability_qualified: List[str],
        diversity_qualified: List[str],
        target_series: pd.Series,
    ) -> Dict[str, any]:
        """生成质量报告"""

        total_indicators = len(indicator_df.columns)

        report = {
            "summary": {
                "total_indicators": total_indicators,
                "basic_qualified": len(basic_qualified),
                "predictive_qualified": len(predictive_qualified),
                "stability_qualified": len(stability_qualified),
                "diversity_qualified": len(diversity_qualified),
                "final_selection_rate": (
                    len(diversity_qualified) / total_indicators
                    if total_indicators > 0
                    else 0
                ),
            },
            "rejection_reasons": {
                "basic_rejected": list(
                    set(indicator_df.columns) - set(basic_qualified)
                ),
                "predictive_rejected": list(
                    set(basic_qualified) - set(predictive_qualified)
                ),
                "stability_rejected": list(
                    set(predictive_qualified) - set(stability_qualified)
                ),
                "diversity_rejected": list(
                    set(stability_qualified) - set(diversity_qualified)
                ),
            },
            "quality_distribution": self._analyze_quality_distribution(
                indicator_df[diversity_qualified], target_series
            ),
        }

        return report

    def _analyze_quality_distribution(
        self, qualified_df: pd.DataFrame, target_series: pd.Series
    ) -> Dict[str, any]:
        """分析质量分布"""

        if qualified_df.empty:
            return {}

        # 计算各指标类型的分布
        indicator_types = {}
        for column in qualified_df.columns:
            if "MA" in column or "EMA" in column:
                indicator_types.setdefault("趋势类", []).append(column)
            elif "RSI" in column or "STOCH" in column:
                indicator_types.setdefault("动量类", []).append(column)
            elif "ATR" in column or "BBANDS" in column:
                indicator_types.setdefault("波动率类", []).append(column)
            elif "VOLUME" in column or "VOL" in column:
                indicator_types.setdefault("成交量类", []).append(column)
            else:
                indicator_types.setdefault("其他类", []).append(column)

        # 计算平均质量指标
        avg_metrics = self._calculate_quality_metrics(qualified_df, target_series)
        avg_ic = (
            np.mean([m.ic_mean for m in avg_metrics.values()]) if avg_metrics else 0
        )
        avg_ic_ir = (
            np.mean([m.ic_ir for m in avg_metrics.values()]) if avg_metrics else 0
        )

        return {
            "indicator_type_distribution": {
                k: len(v) for k, v in indicator_types.items()
            },
            "average_ic": avg_ic,
            "average_ic_ir": avg_ic_ir,
            "total_qualified": len(qualified_df.columns),
        }
