#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 运行时验证模块
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 时间序列安全验证
- 数据完整性检查
- 统计特性验证
- 向量化批量处理
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import RuntimeValidationConfig, StrictMode
from .exceptions import RuntimeValidationError, TimeSeriesSafetyError
from .utils import validate_time_series_data, calculate_factor_statistics, safe_divide


class ValidationResult:
    """验证结果类"""

    def __init__(
        self,
        is_valid: bool,
        validation_type: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None
    ):
        self.is_valid = is_valid
        self.validation_type = validation_type
        self.message = message
        self.details = details or {}
        self.warnings = warnings or []
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "validation_type": self.validation_type,
            "message": self.message,
            "details": self.details,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def success(cls, validation_type: str, **kwargs) -> ValidationResult:
        """创建成功的验证结果"""
        return cls(is_valid=True, validation_type=validation_type, **kwargs)

    @classmethod
    def failure(cls, validation_type: str, message: str, **kwargs) -> ValidationResult:
        """创建失败的验证结果"""
        return cls(is_valid=False, validation_type=validation_type, message=message, **kwargs)

    @classmethod
    def warning(cls, validation_type: str, message: str, **kwargs) -> ValidationResult:
        """创建带警告的验证结果"""
        return cls(is_valid=True, validation_type=validation_type, message=message, **kwargs)


class TimeSeriesValidator:
    """时间序列安全验证器"""

    def __init__(self, config: RuntimeValidationConfig):
        self.config = config

    def validate_temporal_alignment(
        self,
        factor_data: pd.Series,
        return_data: pd.Series,
        horizon: int = 1
    ) -> ValidationResult:
        """
        验证时间序列对齐

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            horizon: 预测周期

        Returns:
            验证结果
        """
        try:
            # 基本检查
            if factor_data.empty or return_data.empty:
                return ValidationResult.failure(
                    "temporal_alignment",
                    "因子数据或收益数据为空",
                    details={
                        "factor_empty": factor_data.empty,
                        "return_empty": return_data.empty
                    }
                )

            # 检查时间索引类型
            if not isinstance(factor_data.index, pd.DatetimeIndex):
                return ValidationResult.failure(
                    "temporal_alignment",
                    "因子数据索引不是DatetimeIndex",
                    details={"index_type": str(type(factor_data.index))}
                )

            if not isinstance(return_data.index, pd.DatetimeIndex):
                return ValidationResult.failure(
                    "temporal_alignment",
                    "收益率数据索引不是DatetimeIndex",
                    details={"index_type": str(type(return_data.index))}
                )

            # 对齐数据
            common_index = factor_data.index.intersection(return_data.index)
            if len(common_index) < max(30, horizon * 2):
                return ValidationResult.failure(
                    "temporal_alignment",
                    f"对齐数据不足，需要至少{max(30, horizon * 2)}个数据点，实际{len(common_index)}个",
                    details={
                        "common_points": len(common_index),
                        "required_points": max(30, horizon * 2),
                        "horizon": horizon
                    }
                )

            aligned_factor = factor_data.loc[common_index]
            aligned_return = return_data.loc[common_index].shift(horizon)

            # 检查未来数据泄露
            if horizon < 0:
                return ValidationResult.failure(
                    "temporal_alignment",
                    f"预测周期不能为负数: {horizon}",
                    details={"horizon": horizon}
                )

            # 计算IC
            valid_mask = aligned_factor.notna() & aligned_return.notna()
            if valid_mask.sum() < 10:
                return ValidationResult.warning(
                    "temporal_alignment",
                    f"有效数据点不足，仅{valid_mask.sum()}个有效数据点",
                    details={"valid_points": valid_mask.sum()}
                )

            ic = aligned_factor[valid_mask].corr(aligned_return[valid_mask])
            if pd.isna(ic):
                return ValidationResult.warning(
                    "temporal_alignment",
                    "IC计算结果为NaN",
                    details={"valid_points": valid_mask.sum()}
                )

            return ValidationResult.success(
                "temporal_alignment",
                message=f"时间序列对齐验证通过，IC={ic:.4f}",
                details={
                    "ic": float(ic),
                    "horizon": horizon,
                    "common_points": len(common_index),
                    "valid_points": valid_mask.sum()
                }
            )

        except Exception as e:
            return ValidationResult.failure(
                "temporal_alignment",
                f"时间序列对齐验证失败: {e}",
                details={"error": str(e)}
            )

    def validate_shift_operation(self, data: pd.Series, periods: int) -> ValidationResult:
        """
        验证shift操作的安全性

        Args:
            data: 要shift的数据
            periods: shift周期

        Returns:
            验证结果
        """
        try:
            if periods < 0:
                return ValidationResult.failure(
                    "shift_operation",
                    f"不允许负数shift（未来函数）: {periods}",
                    details={
                        "periods": periods,
                        "data_length": len(data),
                        "violation_type": "negative_shift"
                    }
                )

            if periods > len(data):
                return ValidationResult.warning(
                    "shift_operation",
                    f"shift周期{periods}超过数据长度{len(data)}",
                    details={
                        "periods": periods,
                        "data_length": len(data)
                    }
                )

            return ValidationResult.success(
                "shift_operation",
                message=f"shift操作验证通过: {periods}期",
                details={"periods": periods}
            )

        except Exception as e:
            return ValidationResult.failure(
                "shift_operation",
                f"shift操作验证失败: {e}",
                details={"error": str(e)}
            )

    def validate_time_series_integrity(self, data: pd.DataFrame) -> ValidationResult:
        """
        验证时间序列数据完整性

        Args:
            data: 时间序列数据

        Returns:
            验证结果
        """
        try:
            issues = validate_time_series_data(data)

            if not issues:
                return ValidationResult.success(
                    "time_series_integrity",
                    message="时间序列数据完整性验证通过"
                )

            # 分类问题
            critical_issues = [issue for issue in issues if "空" in issue or "不是" in issue]
            warning_issues = [issue for issue in issues if issue not in critical_issues]

            if critical_issues:
                return ValidationResult.failure(
                    "time_series_integrity",
                    f"发现严重问题: {'; '.join(critical_issues)}",
                    details={
                        "critical_issues": critical_issues,
                        "warning_issues": warning_issues,
                        "total_issues": len(issues)
                    }
                )
            else:
                return ValidationResult.warning(
                    "time_series_integrity",
                    f"发现问题: {'; '.join(warning_issues)}",
                    details={
                        "warning_issues": warning_issues,
                        "total_issues": len(issues)
                    }
                )

        except Exception as e:
            return ValidationResult.failure(
                "time_series_integrity",
                f"时间序列完整性验证失败: {e}",
                details={"error": str(e)}
            )


class DataIntegrityValidator:
    """数据完整性验证器"""

    def __init__(self, config: RuntimeValidationConfig):
        self.config = config

    def validate_price_consistency(self, data: pd.DataFrame) -> ValidationResult:
        """
        验证价格数据一致性

        Args:
            data: OHLCV数据

        Returns:
            验证结果
        """
        try:
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                return ValidationResult.failure(
                    "price_consistency",
                    f"缺少必要列: {missing_columns}",
                    details={"missing_columns": missing_columns}
                )

            issues = []
            fixes_applied = []

            # 检查负价格
            price_columns = ["open", "high", "low", "close"]
            negative_prices = {}
            for col in price_columns:
                negative_mask = data[col] <= 0
                if negative_mask.any():
                    negative_prices[col] = negative_mask.sum()
                    issues.append(f"列{col}有{negative_mask.sum()}个负价格")

            # 检查负成交量
            negative_volume_mask = data["volume"] < 0
            if negative_volume_mask.any():
                issues.append(f"成交量有{negative_volume_mask.sum()}个负值")

            # 检查OHLC逻辑
            logic_errors = (
                (data["high"] < data["low"]) |
                (data["high"] < data["open"]) |
                (data["high"] < data["close"]) |
                (data["low"] > data["open"]) |
                (data["low"] > data["close"])
            )
            if logic_errors.any():
                issues.append(f"OHLC逻辑错误{logic_errors.sum()}个")

            # 检查价格跳跃异常
            for col in price_columns:
                price_changes = data[col].pct_change().abs()
                extreme_jumps = price_changes > 0.5  # 50%以上的价格跳跃
                if extreme_jumps.any():
                    issues.append(f"列{col}有{extreme_jumps.sum()}个极端价格跳跃")

            if not issues:
                return ValidationResult.success(
                    "price_consistency",
                    message="价格数据一致性验证通过"
                )

            # 根据配置决定是否失败或警告
            if self.config.strict_mode == StrictMode.ENFORCED:
                return ValidationResult.failure(
                    "price_consistency",
                    f"价格一致性问题: {'; '.join(issues)}",
                    details={
                        "issues": issues,
                        "negative_prices": negative_prices,
                        "negative_volume_count": negative_volume_mask.sum() if 'negative_volume_mask' in locals() else 0,
                        "logic_errors_count": logic_errors.sum() if 'logic_errors' in locals() else 0
                    }
                )
            else:
                return ValidationResult.warning(
                    "price_consistency",
                    f"价格一致性问题: {'; '.join(issues)}",
                    details={
                        "issues": issues,
                        "negative_prices": negative_prices,
                        "negative_volume_count": negative_volume_mask.sum() if 'negative_volume_mask' in locals() else 0,
                        "logic_errors_count": logic_errors.sum() if 'logic_errors' in locals() else 0
                    }
                )

        except Exception as e:
            return ValidationResult.failure(
                "price_consistency",
                f"价格一致性验证失败: {e}",
                details={"error": str(e)}
            )

    def validate_data_coverage(
        self,
        factor_data: pd.Series,
        min_coverage: Optional[float] = None
    ) -> ValidationResult:
        """
        验证数据覆盖率

        Args:
            factor_data: 因子数据
            min_coverage: 最小覆盖率要求

        Returns:
            验证结果
        """
        try:
            if factor_data.empty:
                return ValidationResult.failure(
                    "data_coverage",
                    "数据为空",
                    details={"data_length": 0}
                )

            min_coverage = min_coverage or self.config.coverage_threshold
            coverage = factor_data.notna().mean()

            if coverage < min_coverage:
                return ValidationResult.failure(
                    "data_coverage",
                    f"数据覆盖率{coverage:.2%}低于阈值{min_coverage:.2%}",
                    details={
                        "coverage": coverage,
                        "threshold": min_coverage,
                        "total_points": len(factor_data),
                        "missing_points": factor_data.isna().sum()
                    }
                )

            return ValidationResult.success(
                "data_coverage",
                message=f"数据覆盖率验证通过: {coverage:.2%}",
                details={
                    "coverage": coverage,
                    "threshold": min_coverage,
                    "total_points": len(factor_data),
                    "missing_points": factor_data.isna().sum()
                }
            )

        except Exception as e:
            return ValidationResult.failure(
                "data_coverage",
                f"数据覆盖率验证失败: {e}",
                details={"error": str(e)}
            )

    def validate_min_history(
        self,
        factor_data: pd.Series,
        factor_id: str,
        timeframe: str,
        custom_min_history: Optional[int] = None
    ) -> ValidationResult:
        """
        验证最小历史数据要求

        Args:
            factor_data: 因子数据
            factor_id: 因子ID
            timeframe: 时间框架
            custom_min_history: 自定义最小历史数据要求

        Returns:
            验证结果
        """
        try:
            # 获取最小历史数据要求
            min_history = (
                custom_min_history or
                self.config.min_history_map.get(timeframe, 20)
            )

            actual_length = len(factor_data.dropna())

            if actual_length < min_history:
                return ValidationResult.failure(
                    "min_history",
                    f"因子{factor_id}历史数据不足: {actual_length} < {min_history}",
                    details={
                        "factor_id": factor_id,
                        "timeframe": timeframe,
                        "actual_length": actual_length,
                        "required_length": min_history
                    }
                )

            return ValidationResult.success(
                "min_history",
                message=f"因子{factor_id}历史数据充足: {actual_length} >= {min_history}",
                details={
                    "factor_id": factor_id,
                    "timeframe": timeframe,
                    "actual_length": actual_length,
                    "required_length": min_history
                }
            )

        except Exception as e:
            return ValidationResult.failure(
                "min_history",
                f"最小历史数据验证失败: {e}",
                details={"error": str(e)}
            )


class StatisticalValidator:
    """统计特性验证器"""

    def __init__(self, config: RuntimeValidationConfig):
        self.config = config

    def validate_factor_distribution(self, factor_data: pd.Series) -> ValidationResult:
        """
        验证因子分布特性

        Args:
            factor_data: 因子数据

        Returns:
            验证结果
        """
        try:
            if factor_data.empty:
                return ValidationResult.failure(
                    "factor_distribution",
                    "因子数据为空"
                )

            valid_data = factor_data.dropna()
            if len(valid_data) < 10:
                return ValidationResult.warning(
                    "factor_distribution",
                    f"有效数据点过少: {len(valid_data)}"
                )

            # 计算统计特性
            stats = calculate_factor_statistics(valid_data)

            # 检查零方差
            variance = stats.get("std", 0) ** 2
            if variance < 1e-10:
                return ValidationResult.failure(
                    "factor_distribution",
                    f"因子方差接近零: {variance:.2e}",
                    details={
                        "variance": variance,
                        "mean": stats.get("mean", 0),
                        "constant_value": stats.get("mean", 0)
                    }
                )

            # 检查极值比例
            q1, q3 = stats.get("q25", 0), stats.get("q75", 0)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                extreme_outliers = (
                    (valid_data < lower_bound) | (valid_data > upper_bound)
                )
                extreme_ratio = extreme_outliers.mean()

                if extreme_ratio > 0.05:  # 超过5%的极值
                    return ValidationResult.warning(
                        "factor_distribution",
                        f"极值比例过高: {extreme_ratio:.2%}",
                        details={
                            "extreme_ratio": extreme_ratio,
                            "extreme_count": extreme_outliers.sum(),
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound
                        }
                    )

            # 检查分布偏度
            skewness = stats.get("skewness")
            if skewness is not None and abs(skewness) > 10:
                return ValidationResult.warning(
                    "factor_distribution",
                    f"分布偏度异常: {skewness:.2f}",
                    details={"skewness": skewness}
                )

            return ValidationResult.success(
                "factor_distribution",
                message="因子分布特性验证通过",
                details=stats
            )

        except Exception as e:
            return ValidationResult.failure(
                "factor_distribution",
                f"因子分布验证失败: {e}",
                details={"error": str(e)}
            )

    def validate_correlation_matrix(
        self,
        factor_panel: pd.DataFrame,
        max_correlation: Optional[float] = None
    ) -> ValidationResult:
        """
        验证因子相关性矩阵

        Args:
            factor_panel: 因子面板数据
            max_correlation: 最大允许相关性

        Returns:
            验证结果
        """
        try:
            if factor_panel.empty or factor_panel.shape[1] < 2:
                return ValidationResult.success(
                    "correlation_matrix",
                    message="因子数量不足，跳过相关性检查"
                )

            max_correlation = max_correlation or self.config.correlation_threshold

            # 计算相关性矩阵
            clean_data = factor_panel.dropna()
            if len(clean_data) < 10:
                return ValidationResult.warning(
                    "correlation_matrix",
                    f"有效数据不足，无法计算相关性: {len(clean_data)}"
                )

            correlation_matrix = clean_data.corr().abs()

            # 检查高相关性因子对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if corr_value > max_correlation:
                        factor1 = correlation_matrix.columns[i]
                        factor2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((factor1, factor2, corr_value))

            if high_corr_pairs:
                return ValidationResult.warning(
                    "correlation_matrix",
                    f"发现{len(high_corr_pairs)}对高相关性因子",
                    details={
                        "high_correlation_pairs": high_corr_pairs,
                        "max_correlation": max_correlation,
                        "max_observed_correlation": max(corr for _, _, corr in high_corr_pairs)
                    }
                )

            return ValidationResult.success(
                "correlation_matrix",
                message="因子相关性验证通过",
                details={
                    "factor_count": len(correlation_matrix.columns),
                    "max_correlation": max_correlation,
                    "data_points": len(clean_data)
                }
            )

        except Exception as e:
            return ValidationResult.failure(
                "correlation_matrix",
                f"相关性矩阵验证失败: {e}",
                details={"error": str(e)}
            )


class RuntimeValidator:
    """运行时验证器主类"""

    def __init__(self, config: RuntimeValidationConfig):
        self.config = config
        self.time_series_validator = TimeSeriesValidator(config)
        self.data_integrity_validator = DataIntegrityValidator(config)
        self.statistical_validator = StatisticalValidator(config)

    def validate_factor_calculation(
        self,
        factor_data: pd.Series,
        factor_id: str,
        timeframe: str,
        reference_data: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        验证因子计算的完整性和安全性

        Args:
            factor_data: 计算得到的因子数据
            factor_id: 因子ID
            timeframe: 时间框架
            reference_data: 参考数据（用于验证）

        Returns:
            综合验证结果
        """
        if not self.config.enabled:
            return ValidationResult.success(
                "factor_calculation",
                message="运行时验证已禁用"
            )

        start_time = time.time()
        results = []
        warnings = []

        try:
            # 1. 数据完整性验证
            if self.config.data_integrity:
                if reference_data is not None:
                    data_result = self.data_integrity_validator.validate_price_consistency(reference_data)
                    results.append(data_result)
                    if not data_result.is_valid:
                        warnings.extend(data_result.warnings)

                coverage_result = self.data_integrity_validator.validate_data_coverage(factor_data)
                results.append(coverage_result)
                if not coverage_result.is_valid:
                    warnings.extend(coverage_result.warnings)

                history_result = self.data_integrity_validator.validate_min_history(
                    factor_data, factor_id, timeframe
                )
                results.append(history_result)
                if not history_result.is_valid:
                    warnings.extend(history_result.warnings)

            # 2. 统计特性验证
            if self.config.statistical_checks:
                dist_result = self.statistical_validator.validate_factor_distribution(factor_data)
                results.append(dist_result)
                if not dist_result.is_valid:
                    warnings.extend(dist_result.warnings)

            # 3. 时间序列安全验证
            if self.config.time_series_safety and reference_data is not None:
                if "close" in reference_data.columns:
                    returns = reference_data["close"].pct_change()
                    temporal_result = self.time_series_validator.validate_temporal_alignment(
                        factor_data, returns, horizon=1
                    )
                    results.append(temporal_result)
                    if not temporal_result.is_valid:
                        warnings.extend(temporal_result.warnings)

            # 汇总结果
            validation_time = time.time() - start_time
            failed_results = [r for r in results if not r.is_valid]

            if failed_results:
                return ValidationResult.failure(
                    "factor_calculation",
                    f"因子计算验证失败: {len(failed_results)}个验证失败",
                    details={
                        "factor_id": factor_id,
                        "timeframe": timeframe,
                        "validation_time": validation_time,
                        "failed_validations": [r.to_dict() for r in failed_results],
                        "successful_validations": len([r for r in results if r.is_valid])
                    },
                    warnings=warnings
                )
            else:
                return ValidationResult.success(
                    "factor_calculation",
                    message=f"因子{factor_id}计算验证通过",
                    details={
                        "factor_id": factor_id,
                        "timeframe": timeframe,
                        "validation_time": validation_time,
                        "validations_performed": len(results)
                    },
                    warnings=warnings
                )

        except Exception as e:
            return ValidationResult.failure(
                "factor_calculation",
                f"因子计算验证异常: {e}",
                details={
                    "factor_id": factor_id,
                    "timeframe": timeframe,
                    "validation_time": time.time() - start_time,
                    "error": str(e)
                }
            )

    def validate_batch_factors(
        self,
        factor_panel: pd.DataFrame,
        factor_ids: List[str],
        timeframe: str,
        reference_data: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        批量验证多个因子

        Args:
            factor_panel: 因子面板数据
            factor_ids: 因子ID列表
            timeframe: 时间框架
            reference_data: 参考数据

        Returns:
            批量验证结果
        """
        if not self.config.enabled:
            return ValidationResult.success(
                "batch_factors",
                message="运行时验证已禁用"
            )

        start_time = time.time()
        results = []
        all_warnings = []

        # 逐个验证因子
        for factor_id in factor_ids:
            if factor_id in factor_panel.columns:
                factor_data = factor_panel[factor_id]
                result = self.validate_factor_calculation(
                    factor_data, factor_id, timeframe, reference_data
                )
                results.append({
                    "factor_id": factor_id,
                    "result": result.to_dict()
                })
                all_warnings.extend(result.warnings)
            else:
                results.append({
                    "factor_id": factor_id,
                    "result": ValidationResult.failure(
                        "batch_factors",
                        f"因子{factor_id}不存在",
                        details={"factor_id": factor_id}
                    ).to_dict()
                })

        # 相关性矩阵验证
        if self.config.statistical_checks and len(factor_ids) > 1:
            available_factors = [fid for fid in factor_ids if fid in factor_panel.columns]
            if available_factors:
                corr_result = self.statistical_validator.validate_correlation_matrix(
                    factor_panel[available_factors]
                )
                results.append({
                    "factor_id": "correlation_matrix",
                    "result": corr_result.to_dict()
                })
                all_warnings.extend(corr_result.warnings)

        validation_time = time.time() - start_time
        failed_count = len([r for r in results if not r["result"]["is_valid"]])

        if failed_count > 0:
            return ValidationResult.failure(
                "batch_factors",
                f"批量验证失败: {failed_count}/{len(factor_ids)}个因子验证失败",
                details={
                    "timeframe": timeframe,
                    "validation_time": validation_time,
                    "total_factors": len(factor_ids),
                    "failed_factors": failed_count,
                    "results": results
                },
                warnings=all_warnings
            )
        else:
            return ValidationResult.success(
                "batch_factors",
                message=f"批量验证通过: {len(factor_ids)}个因子",
                details={
                    "timeframe": timeframe,
                    "validation_time": validation_time,
                    "total_factors": len(factor_ids),
                    "results": results
                },
                warnings=all_warnings
            )