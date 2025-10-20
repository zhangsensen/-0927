#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 健康监控模块
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 因子健康状态监控
- 实时质量评估
- 异常检测和报警
- 健康趋势分析
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import AlertThreshold, HealthMonitorConfig, MonitoringLevel
from .exceptions import HealthMonitorError
from .utils import (
    FileCache,
    calculate_factor_statistics,
    create_directory_if_not_exists,
)


class HealthMetrics:
    """健康指标类"""

    def __init__(self, factor_id: str):
        self.factor_id = factor_id
        self.timestamp = datetime.now()
        self.metrics: Dict[str, Any] = {}

    def add_metric(self, name: str, value: Union[float, int, str]) -> None:
        """添加指标"""
        self.metrics[name] = value

    def get_quality_score(self) -> float:
        """计算综合质量评分 (0-100)"""
        score = 100.0

        # 覆盖率评分 (30%权重)
        coverage = self.metrics.get("coverage", 1.0)
        if coverage < 0.9:
            score -= 30 * (1 - coverage / 0.9)
        else:
            score -= 30 * (1 - coverage)

        # 方差评分 (25%权重)
        variance = self.metrics.get("variance", 1.0)
        if variance < 1e-10:
            score -= 25

        # 极值评分 (25%权重)
        extreme_ratio = self.metrics.get("extreme_ratio", 0.0)
        score -= 25 * min(extreme_ratio * 10, 1.0)

        # 分布评分 (20%权重)
        skewness = self.metrics.get("skewness", 0.0)
        score -= 20 * min(abs(skewness) / 10, 1.0)

        return max(0.0, min(100.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "factor_id": self.factor_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "quality_score": self.get_quality_score(),
        }


class HealthAlert:
    """健康报警类"""

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        factor_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None,
        **kwargs,
    ):
        self.alert_type = alert_type
        self.severity = severity  # high, medium, low
        self.message = message
        self.factor_id = factor_id
        self.metric_name = metric_name
        self.threshold = threshold
        self.actual_value = actual_value
        self.timestamp = datetime.now()
        self.context = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "factor_id": self.factor_id,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class HealthTrend:
    """健康趋势类"""

    def __init__(self, factor_id: str, max_history: int = 100):
        self.factor_id = factor_id
        self.max_history = max_history
        self.observations: List[Tuple[datetime, Dict[str, Any]]] = []

    def add_observation(self, metrics: HealthMetrics) -> None:
        """添加观察记录"""
        self.observations.append((metrics.timestamp, metrics.metrics))

        # 保持最大历史记录数
        if len(self.observations) > self.max_history:
            self.observations = self.observations[-self.max_history :]

    def get_trend_analysis(self) -> Dict[str, Any]:
        """获取趋势分析"""
        if len(self.observations) < 2:
            return {"status": "insufficient_data"}

        # 提取时间序列数据
        timestamps = [obs[0] for obs in self.observations]
        quality_scores = []
        coverages = []
        variances = []

        for _, metrics in self.observations:
            # 计算质量评分
            temp_metrics = HealthMetrics(self.factor_id)
            temp_metrics.metrics = metrics
            quality_scores.append(temp_metrics.get_quality_score())
            coverages.append(metrics.get("coverage", 1.0))
            variances.append(metrics.get("variance", 1.0))

        # 趋势分析
        def analyze_trend(values: List[float]) -> Dict[str, Any]:
            if len(values) < 2:
                return {"trend": "stable"}

            # 简单线性回归分析趋势
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            if abs(slope) < 0.01:
                trend = "stable"
            elif slope > 0:
                trend = "improving"
            else:
                trend = "degrading"

            return {
                "trend": trend,
                "slope": float(slope),
                "current_value": values[-1],
                "average_value": float(np.mean(values)),
                "volatility": float(np.std(values)),
            }

        return {
            "status": "analyzed",
            "observations_count": len(self.observations),
            "time_span_days": (timestamps[-1] - timestamps[0]).days,
            "quality_trend": analyze_trend(quality_scores),
            "coverage_trend": analyze_trend(coverages),
            "variance_trend": analyze_trend(variances),
            "latest_timestamp": timestamps[-1].isoformat(),
        }


class HealthMonitor:
    """健康监控器"""

    def __init__(self, config: HealthMonitorConfig):
        self.config = config
        self.alerts: List[HealthAlert] = []
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.health_trends: Dict[str, HealthTrend] = {}
        self.cache = FileCache(
            cache_dir=Path.home() / ".future_function_guard_cache" / "health_monitor",
            config=type(
                "obj", (object,), {"ttl_hours": 24, "compression_enabled": True}
            )(),
        )

        # 报警阈值设置
        self.thresholds = self._get_alert_thresholds()

    def _get_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取报警阈值设置"""
        if self.config.alert_threshold == AlertThreshold.LIBERAL:
            return {
                "coverage": {"low": 0.7, "critical": 0.5},
                "variance": {"low": 1e-8, "critical": 1e-10},
                "extreme_ratio": {"low": 0.05, "critical": 0.1},
                "correlation": {"high": 0.98, "critical": 0.99},
                "quality_score": {"low": 60, "critical": 40},
            }
        elif self.config.alert_threshold == AlertThreshold.CONSERVATIVE:
            return {
                "coverage": {"low": 0.95, "critical": 0.9},
                "variance": {"low": 1e-6, "critical": 1e-8},
                "extreme_ratio": {"low": 0.01, "critical": 0.02},
                "correlation": {"high": 0.90, "critical": 0.95},
                "quality_score": {"low": 80, "critical": 70},
            }
        else:  # MODERATE
            return {
                "coverage": {"low": 0.85, "critical": 0.75},
                "variance": {"low": 1e-7, "critical": 1e-9},
                "extreme_ratio": {"low": 0.02, "critical": 0.05},
                "correlation": {"high": 0.95, "critical": 0.97},
                "quality_score": {"low": 70, "critical": 60},
            }

    def check_factor_health(
        self, factor_data: pd.Series, factor_id: str, strict_mode: bool = False
    ) -> HealthMetrics:
        """
        检查单个因子的健康状况

        Args:
            factor_data: 因子数据
            factor_id: 因子ID
            strict_mode: 严格模式

        Returns:
            健康指标
        """
        if not self.config.enabled:
            return HealthMetrics(factor_id)

        start_time = time.time()
        metrics = HealthMetrics(factor_id)

        try:
            if factor_data.empty:
                metrics.add_metric("status", "empty")
                self._create_alert(
                    "empty_data",
                    "high",
                    f"因子{factor_id}数据为空",
                    factor_id=factor_id,
                )
                return metrics

            # 基础统计
            total_points = len(factor_data)
            missing_points = factor_data.isna().sum()
            coverage = (total_points - missing_points) / total_points

            metrics.add_metric("total_points", total_points)
            metrics.add_metric("missing_points", missing_points)
            metrics.add_metric("coverage", coverage)

            # 覆盖率检查
            self._check_coverage(factor_id, coverage)

            if coverage > 0:
                valid_data = factor_data.dropna()

                # 统计特性
                stats = calculate_factor_statistics(valid_data)
                metrics.add_metric("mean", stats.get("mean", 0))
                metrics.add_metric("std", stats.get("std", 0))
                metrics.add_metric("min", stats.get("min", 0))
                metrics.add_metric("max", stats.get("max", 0))
                metrics.add_metric("q25", stats.get("q25", 0))
                metrics.add_metric("q50", stats.get("q50", 0))
                metrics.add_metric("q75", stats.get("q75", 0))
                metrics.add_metric("skewness", stats.get("skewness", 0))
                metrics.add_metric("kurtosis", stats.get("kurtosis", 0))

                variance = stats.get("std", 0) ** 2
                metrics.add_metric("variance", variance)

                # 方差检查
                self._check_variance(factor_id, variance)

                # 极值检查
                if len(valid_data) >= 10:
                    q1, q3 = stats.get("q25", 0), stats.get("q75", 0)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        extreme_outliers = (valid_data < lower_bound) | (
                            valid_data > upper_bound
                        )
                        extreme_ratio = extreme_outliers.mean()
                        metrics.add_metric("extreme_ratio", extreme_ratio)
                        metrics.add_metric("extreme_count", extreme_outliers.sum())

                        # 极值检查
                        self._check_extreme_ratio(factor_id, extreme_ratio)

                # 分布检查
                skewness = stats.get("skewness")
                if skewness is not None:
                    self._check_distribution(factor_id, skewness)

                # IC检查（如果有基准数据）
                if "ic" in stats:
                    metrics.add_metric("ic", stats["ic"])
                    metrics.add_metric("ic_abs", stats.get("ic_abs", 0))

            # 计算质量评分
            quality_score = metrics.get_quality_score()
            metrics.add_metric("quality_score", quality_score)

            # 质量评分检查
            self._check_quality_score(factor_id, quality_score)

            # 更新历史记录
            self._update_health_history(factor_id, metrics)

            metrics.add_metric("check_time", time.time() - start_time)
            metrics.add_metric("status", "healthy")

            # 记录健康指标
            self.health_metrics[factor_id] = metrics

            return metrics

        except Exception as e:
            metrics.add_metric("status", "error")
            metrics.add_metric("error", str(e))
            metrics.add_metric("check_time", time.time() - start_time)

            self._create_alert(
                "health_check_error",
                "medium",
                f"因子{factor_id}健康检查失败: {e}",
                factor_id=factor_id,
                error=str(e),
            )

            return metrics

    def check_batch_factors_health(
        self,
        factor_panel: pd.DataFrame,
        factor_ids: Optional[List[str]] = None,
        strict_mode: bool = False,
    ) -> Dict[str, HealthMetrics]:
        """
        批量检查因子健康状况

        Args:
            factor_panel: 因子面板数据
            factor_ids: 要检查的因子ID列表（可选）
            strict_mode: 严格模式

        Returns:
            因子健康指标字典
        """
        if factor_ids is None:
            factor_ids = list(factor_panel.columns)

        results = {}

        for factor_id in factor_ids:
            if factor_id in factor_panel.columns:
                factor_data = factor_panel[factor_id]
                results[factor_id] = self.check_factor_health(
                    factor_data, factor_id, strict_mode
                )
            else:
                # 创建缺失因子的健康指标
                metrics = HealthMetrics(factor_id)
                metrics.add_metric("status", "missing")
                metrics.add_metric("quality_score", 0.0)
                results[factor_id] = metrics

        # 检查因子间相关性
        if len(factor_ids) > 1:
            self._check_correlation_matrix(factor_panel, factor_ids)

        return results

    def _check_coverage(self, factor_id: str, coverage: float) -> None:
        """检查覆盖率"""
        threshold = self.thresholds["coverage"]
        if coverage < threshold["critical"]:
            self._create_alert(
                "low_coverage",
                "high",
                f"因子{factor_id}覆盖率严重不足: {coverage:.2%}",
                factor_id=factor_id,
                metric_name="coverage",
                threshold=threshold["critical"],
                actual_value=coverage,
            )
        elif coverage < threshold["low"]:
            self._create_alert(
                "low_coverage",
                "medium",
                f"因子{factor_id}覆盖率偏低: {coverage:.2%}",
                factor_id=factor_id,
                metric_name="coverage",
                threshold=threshold["low"],
                actual_value=coverage,
            )

    def _check_variance(self, factor_id: str, variance: float) -> None:
        """检查方差"""
        threshold = self.thresholds["variance"]
        if variance < threshold["critical"]:
            self._create_alert(
                "zero_variance",
                "high",
                f"因子{factor_id}方差接近零: {variance:.2e}",
                factor_id=factor_id,
                metric_name="variance",
                threshold=threshold["critical"],
                actual_value=variance,
            )
        elif variance < threshold["low"]:
            self._create_alert(
                "zero_variance",
                "medium",
                f"因子{factor_id}方差过低: {variance:.2e}",
                factor_id=factor_id,
                metric_name="variance",
                threshold=threshold["low"],
                actual_value=variance,
            )

    def _check_extreme_ratio(self, factor_id: str, extreme_ratio: float) -> None:
        """检查极值比例"""
        threshold = self.thresholds["extreme_ratio"]
        if extreme_ratio > threshold["critical"]:
            self._create_alert(
                "extreme_values",
                "high",
                f"因子{factor_id}极值比例过高: {extreme_ratio:.2%}",
                factor_id=factor_id,
                metric_name="extreme_ratio",
                threshold=threshold["critical"],
                actual_value=extreme_ratio,
            )
        elif extreme_ratio > threshold["low"]:
            self._create_alert(
                "extreme_values",
                "medium",
                f"因子{factor_id}极值比例偏高: {extreme_ratio:.2%}",
                factor_id=factor_id,
                metric_name="extreme_ratio",
                threshold=threshold["low"],
                actual_value=extreme_ratio,
            )

    def _check_distribution(self, factor_id: str, skewness: float) -> None:
        """检查分布偏度"""
        if abs(skewness) > 10:
            self._create_alert(
                "distribution_issue",
                "medium",
                f"因子{factor_id}分布偏度异常: {skewness:.2f}",
                factor_id=factor_id,
                metric_name="skewness",
                actual_value=skewness,
            )

    def _check_quality_score(self, factor_id: str, quality_score: float) -> None:
        """检查质量评分"""
        threshold = self.thresholds["quality_score"]
        if quality_score < threshold["critical"]:
            self._create_alert(
                "low_quality_score",
                "high",
                f"因子{factor_id}质量评分严重偏低: {quality_score:.1f}",
                factor_id=factor_id,
                metric_name="quality_score",
                threshold=threshold["critical"],
                actual_value=quality_score,
            )
        elif quality_score < threshold["low"]:
            self._create_alert(
                "low_quality_score",
                "medium",
                f"因子{factor_id}质量评分偏低: {quality_score:.1f}",
                factor_id=factor_id,
                metric_name="quality_score",
                threshold=threshold["low"],
                actual_value=quality_score,
            )

    def _check_correlation_matrix(
        self, factor_panel: pd.DataFrame, factor_ids: List[str]
    ) -> None:
        """检查因子相关性矩阵"""
        available_factors = [fid for fid in factor_ids if fid in factor_panel.columns]
        if len(available_factors) < 2:
            return

        try:
            clean_data = factor_panel[available_factors].dropna()
            if len(clean_data) < 10:
                return

            correlation_matrix = clean_data.corr().abs()
            threshold = self.thresholds["correlation"]

            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > threshold["critical"]:
                        factor1 = correlation_matrix.columns[i]
                        factor2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((factor1, factor2, corr_value))
                        self._create_alert(
                            "high_correlation",
                            "high",
                            f"因子{factor1}与{factor2}相关性过高: {corr_value:.3f}",
                            metric_name="correlation",
                            threshold=threshold["critical"],
                            actual_value=corr_value,
                        )
                    elif corr_value > threshold["high"]:
                        factor1 = correlation_matrix.columns[i]
                        factor2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((factor1, factor2, corr_value))
                        self._create_alert(
                            "high_correlation",
                            "medium",
                            f"因子{factor1}与{factor2}相关性偏高: {corr_value:.3f}",
                            metric_name="correlation",
                            threshold=threshold["high"],
                            actual_value=corr_value,
                        )

        except Exception as e:
            self._create_alert(
                "correlation_check_error", "low", f"相关性检查失败: {e}", error=str(e)
            )

    def _update_health_history(self, factor_id: str, metrics: HealthMetrics) -> None:
        """更新健康历史记录"""
        if factor_id not in self.health_trends:
            self.health_trends[factor_id] = HealthTrend(factor_id)

        self.health_trends[factor_id].add_observation(metrics)

    def _create_alert(
        self, alert_type: str, severity: str, message: str, **kwargs
    ) -> None:
        """创建报警"""
        alert = HealthAlert(alert_type, severity, message, **kwargs)
        self.alerts.append(alert)

        # 实时报警处理
        if self.config.real_time_alerts:
            self._handle_real_time_alert(alert)

        # 保持报警历史不超过一定数量
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]

    def _handle_real_time_alert(self, alert: HealthAlert) -> None:
        """处理实时报警"""
        # 这里可以扩展为发送邮件、消息、记录到日志等
        if alert.severity == "high":
            # 高危报警立即处理
            print(f"🚨 HIGH ALERT: {alert.message}")
        elif alert.severity == "medium":
            # 中危报警记录
            print(f"⚠️  MEDIUM ALERT: {alert.message}")

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康监控汇总"""
        total_factors = len(self.health_metrics)
        total_alerts = len(self.alerts)

        # 按严重程度统计报警
        alert_counts = {"high": 0, "medium": 0, "low": 0}
        for alert in self.alerts:
            if alert.severity in alert_counts:
                alert_counts[alert.severity] += 1

        # 计算平均质量评分
        quality_scores = [
            metrics.get_quality_score()
            for metrics in self.health_metrics.values()
            if metrics.metrics.get("status") == "healthy"
        ]
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0

        # 按报警类型统计
        alert_types = {}
        for alert in self.alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1

        return {
            "monitoring_level": (
                self.config.monitoring_level.value
                if hasattr(self.config.monitoring_level, "value")
                else self.config.monitoring_level
            ),
            "alert_threshold": (
                self.config.alert_threshold.value
                if hasattr(self.config.alert_threshold, "value")
                else self.config.alert_threshold
            ),
            "total_factors_monitored": total_factors,
            "total_alerts": total_alerts,
            "alert_counts": alert_counts,
            "alert_types": alert_types,
            "average_quality_score": float(avg_quality_score),
            "healthy_factors": len(
                [
                    m
                    for m in self.health_metrics.values()
                    if m.metrics.get("status") == "healthy"
                ]
            ),
            "unhealthy_factors": len(
                [
                    m
                    for m in self.health_metrics.values()
                    if m.metrics.get("status") != "healthy"
                ]
            ),
            "latest_check_time": max(
                [m.timestamp for m in self.health_metrics.values()],
                default=datetime.now(),
            ).isoformat(),
        }

    def get_factor_health_report(self, factor_id: str) -> Dict[str, Any]:
        """获取单个因子的健康报告"""
        if factor_id not in self.health_metrics:
            return {"error": f"Factor {factor_id} not found"}

        metrics = self.health_metrics[factor_id]
        trend_analysis = None

        if factor_id in self.health_trends:
            trend_analysis = self.health_trends[factor_id].get_trend_analysis()

        # 获取相关报警
        factor_alerts = [
            alert.to_dict() for alert in self.alerts if alert.factor_id == factor_id
        ]

        return {
            "factor_id": factor_id,
            "current_metrics": metrics.to_dict(),
            "trend_analysis": trend_analysis,
            "recent_alerts": factor_alerts[-10:],  # 最近10个报警
            "alert_count": len(factor_alerts),
        }

    def generate_health_report(self, output_format: str = "text") -> str:
        """生成健康监控报告"""
        summary = self.get_health_summary()

        if output_format == "json":
            return json.dumps(summary, indent=2, ensure_ascii=False, default=str)

        elif output_format == "markdown":
            return self._generate_markdown_report(summary)

        else:  # text
            return self._generate_text_report(summary)

    def _generate_text_report(self, summary: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        lines = [
            "因子健康监控报告",
            "=" * 50,
            f"监控级别: {summary['monitoring_level']}",
            f"报警阈值: {summary['alert_threshold']}",
            f"监控因子数: {summary['total_factors_monitored']}",
            f"总报警数: {summary['total_alerts']}",
            f"平均质量评分: {summary['average_quality_score']:.1f}",
            f"健康因子数: {summary['healthy_factors']}",
            f"异常因子数: {summary['unhealthy_factors']}",
            "",
            "报警分布:",
            f"- 高危: {summary['alert_counts']['high']}",
            f"- 中危: {summary['alert_counts']['medium']}",
            f"- 低危: {summary['alert_counts']['low']}",
            "",
        ]

        if summary["alert_types"]:
            lines.append("报警类型分布:")
            for alert_type, count in sorted(
                summary["alert_types"].items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"- {alert_type}: {count}")
            lines.append("")

        # 最近报警
        recent_alerts = self.alerts[-5:] if self.alerts else []
        if recent_alerts:
            lines.append("最近报警:")
            for alert in recent_alerts:
                severity_icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(
                    alert.severity, "⚠️"
                )
                lines.append(
                    f"{severity_icon} {alert.timestamp.strftime('%H:%M:%S')} - {alert.message}"
                )
        else:
            lines.append("✅ 无活跃报警")

        return "\n".join(lines)

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        lines = [
            "# 因子健康监控报告\n",
            "## 监控概览\n",
            f"- **监控级别**: {summary['monitoring_level']}",
            f"- **报警阈值**: {summary['alert_threshold']}",
            f"- **监控因子数**: {summary['total_factors_monitored']}",
            f"- **总报警数**: {summary['total_alerts']}",
            f"- **平均质量评分**: {summary['average_quality_score']:.1f}",
            f"- **健康因子数**: {summary['healthy_factors']}",
            f"- **异常因子数**: {summary['unhealthy_factors']}\n",
            "## 报警统计\n",
            "### 按严重程度分布\n",
            f"- 🔴 高危: {summary['alert_counts']['high']}",
            f"- 🟡 中危: {summary['alert_counts']['medium']}",
            f"- 🟢 低危: {summary['alert_counts']['low']}\n",
        ]

        if summary["alert_types"]:
            lines.extend(
                [
                    "### 按类型分布\n",
                ]
            )
            for alert_type, count in sorted(
                summary["alert_types"].items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"- **{alert_type}**: {count}")
            lines.append("")

        # 最近报警
        recent_alerts = self.alerts[-5:] if self.alerts else []
        if recent_alerts:
            lines.extend(
                [
                    "## 最近报警\n",
                ]
            )
            for alert in recent_alerts:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    alert.severity, "🟡"
                )
                lines.append(
                    f"- {severity_emoji} **{alert.timestamp.strftime('%H:%M:%S')}** - {alert.message}"
                )
        else:
            lines.extend(["## 报警状态\n", "✅ **无活跃报警**\n"])

        return "\n".join(lines)

    def export_health_data(self, file_path: Union[str, Path]) -> None:
        """导出健康数据到文件"""
        file_path = Path(file_path)
        create_directory_if_not_exists(file_path.parent)

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "config": {
                "monitoring_level": self.config.monitoring_level.value,
                "alert_threshold": self.config.alert_threshold.value,
                "coverage_threshold": self.config.coverage_threshold,
                "variance_threshold": self.config.variance_threshold,
            },
            "summary": self.get_health_summary(),
            "health_metrics": {
                factor_id: metrics.to_dict()
                for factor_id, metrics in self.health_metrics.items()
            },
            "health_trends": {
                factor_id: trend.get_trend_analysis()
                for factor_id, trend in self.health_trends.items()
            },
            "alerts": [alert.to_dict() for alert in self.alerts],
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            raise HealthMonitorError(f"Failed to export health data: {e}")

    def clear_alerts(self, older_than_days: Optional[int] = None) -> int:
        """清理报警记录"""
        if older_than_days is None:
            # 清空所有报警
            count = len(self.alerts)
            self.alerts.clear()
            return count
        else:
            # 清理指定天数之前的报警
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            original_count = len(self.alerts)
            self.alerts = [
                alert for alert in self.alerts if alert.timestamp > cutoff_time
            ]
            return original_count - len(self.alerts)

    def reset_monitoring(self) -> None:
        """重置监控状态"""
        self.alerts.clear()
        self.health_metrics.clear()
        self.health_trends.clear()
