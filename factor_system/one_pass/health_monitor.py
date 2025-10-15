"""因子健康监控系统 - 只告警不阻塞"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    因子健康监控系统

    核心原则:
    - 只告警不阻塞，计算继续进行
    - 覆盖率监控，零方差检测
    - 重复列检测，时序哨兵
    - 向量化批量检查，高效执行
    """

    def __init__(
        self, coverage_threshold: float = 0.9, correlation_threshold: float = 0.95
    ):
        """
        初始化健康监控器

        Args:
            coverage_threshold: 覆盖率阈值（默认90%）
            correlation_threshold: 相关性阈值（默认95%）
        """
        self.coverage_threshold = coverage_threshold
        self.correlation_threshold = correlation_threshold

        # 健康检查记录
        self.health_log = []
        self.warnings = {
            "low_coverage": [],
            "zero_variance": [],
            "high_correlation": [],
            "time_series_issues": [],
            "extreme_values": [],
        }

        logger.info(
            f"健康监控器初始化: 覆盖率阈值{coverage_threshold:.1%}, 相关性阈值{correlation_threshold:.1%}"
        )

    def check_factor_health(
        self,
        factor_panel: pd.DataFrame,
        factor_ids: List[str],
        strict_mode: bool = False,
    ) -> List[str]:
        """
        全面健康检查（只告警不阻塞）

        Args:
            factor_panel: 因子面板数据
            factor_ids: 因子ID列表
            strict_mode: 严格模式（更多检查）

        Returns:
            警告列表
        """
        warnings = []

        if factor_panel.empty:
            return ["因子面板为空"]

        logger.debug(f"开始健康检查: {factor_panel.shape}")

        # 检查1: 覆盖率监控
        coverage_warnings = self._check_coverage(factor_panel, factor_ids)
        warnings.extend(coverage_warnings)

        # 检查2: 零方差检测
        variance_warnings = self._check_zero_variance(factor_panel, factor_ids)
        warnings.extend(variance_warnings)

        # 检查3: 重复列检测（高相关性）
        correlation_warnings = self._check_high_correlation(factor_panel, factor_ids)
        warnings.extend(correlation_warnings)

        # 检查4: 时序哨兵（未来函数检测）
        time_series_warnings = self._check_time_series_integrity(
            factor_panel, factor_ids
        )
        warnings.extend(time_series_warnings)

        # 检查5: 极值检测
        extreme_warnings = self._check_extreme_values(factor_panel, factor_ids)
        warnings.extend(extreme_warnings)

        # 严格模式下的额外检查
        if strict_mode:
            strict_warnings = self._strict_mode_checks(factor_panel, factor_ids)
            warnings.extend(strict_warnings)

        # 记录健康检查结果
        self.health_log.append(
            {
                "timestamp": datetime.now(),
                "panel_shape": factor_panel.shape,
                "factors_checked": len(factor_ids),
                "warnings_generated": len(warnings),
                "strict_mode": strict_mode,
            }
        )

        logger.debug(f"健康检查完成: {len(warnings)}个警告")

        return warnings

    def _check_coverage(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        检查1: 覆盖率监控

        检查因子的数据覆盖率，低于阈值则告警
        """
        warnings = []

        for factor_id in factor_ids:
            if factor_id not in factor_panel.columns:
                warnings.append(f"覆盖率: 因子{factor_id}缺失")
                continue

            # 计算覆盖率（非NaN值比例）
            factor_values = factor_panel[factor_id]
            coverage = factor_values.notna().mean()

            if coverage < self.coverage_threshold:
                warning_msg = f"覆盖率: 因子{factor_id}覆盖率{coverage:.1%} < {self.coverage_threshold:.1%}"
                warnings.append(warning_msg)

                # 记录低覆盖率因子
                self.warnings["low_coverage"].append(
                    {
                        "factor_id": factor_id,
                        "coverage": coverage,
                        "threshold": self.coverage_threshold,
                        "timestamp": datetime.now(),
                    }
                )

        return warnings

    def _check_zero_variance(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        检查2: 零方差检测

        检测常数列（方差为0或接近0的因子）
        """
        warnings = []

        for factor_id in factor_ids:
            if factor_id not in factor_panel.columns:
                continue

            factor_values = factor_panel[factor_id].dropna()

            if len(factor_values) < 2:
                warnings.append(f"零方差: 因子{factor_id}有效数据不足(<2)")
                continue

            # 计算方差
            variance = factor_values.var()

            # 检查零方差或接近零方差
            zero_variance_threshold = 1e-10
            if variance < zero_variance_threshold:
                warning_msg = f"零方差: 因子{factor_id}方差接近0 ({variance:.2e})"
                warnings.append(warning_msg)

                # 记录零方差因子
                self.warnings["zero_variance"].append(
                    {
                        "factor_id": factor_id,
                        "variance": variance,
                        "threshold": zero_variance_threshold,
                        "timestamp": datetime.now(),
                    }
                )

        return warnings

    def _check_high_correlation(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        检查3: 重复列检测（高相关性）

        检测相关性过高的因子对，可能存在重复计算
        """
        warnings = []

        # 筛选存在的因子
        existing_factors = [fid for fid in factor_ids if fid in factor_panel.columns]
        if len(existing_factors) < 2:
            return warnings

        # 计算相关性矩阵
        factor_data = factor_panel[existing_factors].dropna()

        if len(factor_data) < 10:  # 数据量太少，跳过相关性检查
            return warnings

        correlation_matrix = factor_data.corr().abs()

        # 检查高相关性因子对（排除对角线）
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > self.correlation_threshold:
                    factor1 = correlation_matrix.columns[i]
                    factor2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((factor1, factor2, corr_value))

        # 生成警告
        for factor1, factor2, corr_value in high_corr_pairs:
            warning_msg = f"高相关性: {factor1} 与 {factor2} 相关性 {corr_value:.3f} > {self.correlation_threshold:.1%}"
            warnings.append(warning_msg)

            # 记录高相关性因子对
            self.warnings["high_correlation"].append(
                {
                    "factor1": factor1,
                    "factor2": factor2,
                    "correlation": corr_value,
                    "threshold": self.correlation_threshold,
                    "timestamp": datetime.now(),
                }
            )

        return warnings

    def _check_time_series_integrity(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        检查4: 时序哨兵（未来函数检测）

        检查因子计算是否违反时序约束
        """
        warnings = []

        if not isinstance(factor_panel.index, pd.MultiIndex):
            return warnings  # 非MultiIndex数据跳过时序检查

        # 获取时间索引
        time_index = factor_panel.index.get_level_values(0).unique()
        symbols = factor_panel.index.get_level_values("symbol").unique()

        # 检查每个标的的时序完整性
        for symbol in symbols:
            symbol_data = factor_panel.xs(symbol, level="symbol")

            # 检查时间索引是否严格递增
            if not symbol_data.index.is_monotonic_increasing:
                warnings.append(f"时序问题: 标的{symbol}时间索引不严格递增")

            # 检查是否有重复时间点
            duplicate_times = symbol_data.index.duplicated().sum()
            if duplicate_times > 0:
                warnings.append(
                    f"时序问题: 标的{symbol}有{duplicate_times}个重复时间点"
                )

            # 检查时间跳跃是否合理（针对不同时间框架）
            time_diffs = symbol_data.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                # 检查异常大的时间跳跃
                median_diff = time_diffs.median()
                max_diff = time_diffs.max()
                if max_diff > median_diff * 10:  # 超过中位数10倍
                    warnings.append(f"时序问题: 标的{symbol}存在异常时间跳跃")

        # 记录时序问题
        if warnings:
            self.warnings["time_series_issues"].append(
                {
                    "symbols_checked": len(symbols),
                    "issues_found": len([w for w in warnings if "时序问题" in w]),
                    "timestamp": datetime.now(),
                }
            )

        return warnings

    def _check_extreme_values(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        检查5: 极值检测

        检测因子中的异常极值，可能指示计算错误
        """
        warnings = []

        for factor_id in factor_ids:
            if factor_id not in factor_panel.columns:
                continue

            factor_values = factor_panel[factor_id].dropna()

            if len(factor_values) < 10:
                continue  # 数据量太少跳过

            # 使用IQR方法检测异常值
            Q1 = factor_values.quantile(0.25)
            Q3 = factor_values.quantile(0.75)
            IQR = Q3 - Q1

            # 异常值边界
            lower_bound = Q1 - 3 * IQR  # 3倍IQR，严格检测
            upper_bound = Q3 + 3 * IQR

            extreme_outliers = factor_values[
                (factor_values < lower_bound) | (factor_values > upper_bound)
            ]

            if len(extreme_outliers) > 0:
                outlier_ratio = len(extreme_outliers) / len(factor_values)
                if outlier_ratio > 0.01:  # 异常值比例超过1%
                    warning_msg = f"极值: 因子{factor_id}有{len(extreme_outliers)}个异常值 ({outlier_ratio:.1%})"
                    warnings.append(warning_msg)

                    # 记录极值问题
                    self.warnings["extreme_values"].append(
                        {
                            "factor_id": factor_id,
                            "outlier_count": len(extreme_outliers),
                            "outlier_ratio": outlier_ratio,
                            "min_value": factor_values.min(),
                            "max_value": factor_values.max(),
                            "timestamp": datetime.now(),
                        }
                    )

        return warnings

    def _strict_mode_checks(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> List[str]:
        """
        严格模式下的额外检查
        """
        warnings = []

        # 检查数据类型一致性
        for factor_id in factor_ids:
            if factor_id not in factor_panel.columns:
                continue

            factor_series = factor_panel[factor_id]

            # 检查是否为数值类型
            if not pd.api.types.is_numeric_dtype(factor_series):
                warnings.append(f"数据类型: 因子{factor_id}不是数值类型")
                continue

            # 检查无穷大值
            inf_count = np.isinf(factor_series).sum()
            if inf_count > 0:
                warnings.append(f"极值: 因子{factor_id}包含{inf_count}个无穷大值")

        # 检查因子分布的合理性
        for factor_id in factor_ids:
            if factor_id not in factor_panel.columns:
                continue

            factor_values = factor_panel[factor_id].dropna()

            if len(factor_values) < 20:
                continue

            # 检查分布的偏度和峰度
            from scipy import stats

            try:
                skewness = stats.skew(factor_values)
                kurtosis = stats.kurtosis(factor_values)

                # 极度偏斜的分布
                if abs(skewness) > 10:
                    warnings.append(f"分布: 因子{factor_id}偏度异常 ({skewness:.2f})")

                # 极度尖峰或平坦的分布
                if abs(kurtosis) > 20:
                    warnings.append(f"分布: 因子{factor_id}峰度异常 ({kurtosis:.2f})")

            except ImportError:
                # scipy不可用时跳过分布检查
                pass
            except Exception:
                # 分布检查失败时跳过
                pass

        return warnings

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康监控汇总"""
        total_warnings = sum(
            len(warning_list) for warning_list in self.warnings.values()
        )

        return {
            "total_checks": len(self.health_log),
            "total_warnings": total_warnings,
            "warning_categories": {
                category: len(warning_list)
                for category, warning_list in self.warnings.items()
            },
            "recent_checks": self.health_log[-5:] if self.health_log else [],
            "coverage_threshold": self.coverage_threshold,
            "correlation_threshold": self.correlation_threshold,
        }

    def generate_health_report(self) -> str:
        """生成健康监控报告"""
        summary = self.get_health_summary()

        report = f"""
因子健康监控报告
{'='*50}
总检查次数: {summary['total_checks']}
总警告数量: {summary['total_warnings']}

警告分类:
- 低覆盖率: {summary['warning_categories']['low_coverage']}
- 零方差: {summary['warning_categories']['zero_variance']}
- 高相关性: {summary['warning_categories']['high_correlation']}
- 时序问题: {summary['warning_categories']['time_series_issues']}
- 极值问题: {summary['warning_categories']['extreme_values']}

监控阈值:
- 覆盖率阈值: {summary['coverage_threshold']:.1%}
- 相关性阈值: {summary['correlation_threshold']:.1%}

{'='*50}
        """

        if summary["total_warnings"] == 0:
            report += "✅ 所有因子健康检查通过\n"
        else:
            report += "⚠️  发现健康问题，但计算继续执行\n"

        return report

    def reset_warnings(self):
        """重置警告记录"""
        self.warnings = {
            "low_coverage": [],
            "zero_variance": [],
            "high_correlation": [],
            "time_series_issues": [],
            "extreme_values": [],
        }
        self.health_log = []
        logger.info("健康监控警告记录已重置")

    def export_warnings(self, filepath: str):
        """导出警告记录到文件"""
        import json

        export_data = {
            "health_summary": self.get_health_summary(),
            "detailed_warnings": self.warnings,
            "health_log": self.health_log,
            "export_timestamp": datetime.now().isoformat(),
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"健康监控报告已导出到: {filepath}")
        except Exception as e:
            logger.error(f"导出健康监控报告失败: {e}")

    def get_factor_quality_score(
        self, factor_panel: pd.DataFrame, factor_id: str
    ) -> float:
        """
        计算单个因子的质量评分（0-100）

        Args:
            factor_panel: 因子面板
            factor_id: 因子ID

        Returns:
            质量评分
        """
        if factor_id not in factor_panel.columns:
            return 0.0

        factor_values = factor_panel[factor_id]
        score = 100.0

        # 覆盖率评分 (30%权重)
        coverage = factor_values.notna().mean()
        if coverage < self.coverage_threshold:
            score -= 30 * (1 - coverage / self.coverage_threshold)
        else:
            score -= 30 * (1 - coverage)

        # 方差评分 (25%权重)
        variance = factor_values.dropna().var()
        if variance < 1e-10:
            score -= 25

        # 极值评分 (25%权重)
        Q1 = factor_values.quantile(0.25)
        Q3 = factor_values.quantile(0.75)
        IQR = Q3 - Q1
        extreme_ratio = (
            (factor_values < Q1 - 3 * IQR) | (factor_values > Q3 + 3 * IQR)
        ).mean()
        score -= 25 * min(extreme_ratio * 10, 1.0)  # 10%极值比例扣满分

        # 分布评分 (20%权重)
        try:
            from scipy import stats

            skewness = stats.skew(factor_values.dropna())
            score -= 20 * min(abs(skewness) / 10, 1.0)  # 偏度超过10扣满分
        except:
            pass  # scipy不可用时跳过

        return max(0.0, min(100.0, score))
