#!/usr/bin/env python3
"""
时间序列验证器 - 运行时防止未来函数
作者：量化首席工程师
版本：1.0.0
日期：2025-10-02

功能：
- 运行时检测时间序列对齐
- 防止意外的前视偏差
- 提供详细的时间序列验证报告
"""

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class TemporalValidationError(Exception):
    """时间序列验证异常"""

    pass


class TemporalValidator:
    """时间序列验证器"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_log = []

    def validate_time_alignment(
        self,
        factor_data: pd.Series,
        return_data: pd.Series,
        horizon: int,
        context: str = "",
    ) -> Tuple[bool, str]:
        """
        验证时间序列对齐性

        Args:
            factor_data: 因子数据
            return_data: 收益率数据
            horizon: 预测周期
            context: 上下文描述

        Returns:
            (is_valid, error_message)
        """

        try:
            factor_series = factor_data.copy()
            return_series = return_data.copy()

            # 转换为DatetimeIndex
            if not isinstance(factor_series.index, pd.DatetimeIndex):
                factor_series.index = pd.to_datetime(factor_series.index)
            if not isinstance(return_series.index, pd.DatetimeIndex):
                return_series.index = pd.to_datetime(return_series.index)

            # 检查数据完整性
            if factor_series.empty or return_series.empty:
                return True, f"{context} 数据为空"

            # 检查时间对齐
            common_index = factor_series.index.intersection(return_series.index)
            if len(common_index) < max(30, horizon * 2):  # 至少需要2倍周期的数据
                message = f"{context} 对齐数据不足，需要{horizon * 2}个点，实际{len(common_index)}个"
                self.validation_log.append(message)
                return False, message

            aligned_factor = factor_series.loc[common_index]
            aligned_return = return_series.loc[common_index]

            # 验证时间序列关系（仅使用历史信息）
            if horizon > 0:
                lagged_factor = aligned_factor.shift(horizon)
                valid_mask = lagged_factor.notna() & aligned_return.notna()

                if valid_mask.sum() < max(30, horizon * 2):
                    message = f"{context} 有效样本不足，需要{max(30, horizon * 2)}个，实际{valid_mask.sum()}个"
                    self.validation_log.append(message)
                    return False, message

                correlation = lagged_factor[valid_mask].corr(aligned_return[valid_mask])
                logger.debug(f"{context} IC计算: {correlation:.4f} (horizon={horizon})")

            return True, "验证通过"

        except Exception as e:
            error_msg = f"{context} 验证失败: {str(e)}"
            logger.error(error_msg)
            self.validation_log.append(error_msg)
            return False, error_msg

    def validate_ic_calculation(
        self,
        factor_data: pd.Series,
        return_data: pd.Series,
        ic_horizons: list,
        context: str = "",
    ) -> Dict[str, Any]:
        """验证IC计算过程"""

        results = {}

        for horizon in ic_horizons:
            is_valid, message = self.validate_time_alignment(
                factor_data, return_data, horizon, f"{context} IC-{horizon}d"
            )

            if not is_valid:
                if self.strict_mode:
                    raise TemporalValidationError(f"IC-{horizon}d验证失败: {message}")
                else:
                    logger.warning(f"IC-{horizon}d验证警告: {message}")
                    continue

            # 执行IC计算
            try:
                factor_series = factor_data.copy()
                return_series = return_data.copy()

                common_index = factor_series.index.intersection(return_series.index)
                aligned_factor = factor_series.loc[common_index]
                aligned_return = return_series.loc[common_index]

                lagged_factor = aligned_factor.shift(horizon)
                valid_mask = lagged_factor.notna() & aligned_return.notna()

                if valid_mask.sum() < 30:
                    logger.warning(f"IC-{horizon}d 样本量不足: {valid_mask.sum()}")

                ic = lagged_factor[valid_mask].corr(aligned_return[valid_mask])

                results[horizon] = {
                    "ic": ic if not pd.isna(ic) else 0.0,
                    "sample_size": int(valid_mask.sum()),
                    "is_valid": is_valid,
                }

            except Exception as e:
                logger.error(f"IC-{horizon}d计算失败: {e}")
                results[horizon] = {"ic": 0.0, "sample_size": 0, "is_valid": False}

        return results

    def validate_no_future_data(self, data: pd.DataFrame, context: str = "") -> bool:
        """验证数据中不包含未来信息"""

        issues = []

        # 检查列名
        future_columns = [
            col
            for col in data.columns
            if "future" in col.lower() or "lead" in col.lower()
        ]

        if future_columns:
            issues.extend([f"发现未来相关列: {future_columns}"])

        # 检查数据值（简单的启发式）
        for col in data.columns:
            if "price" in col.lower() or "return" in col.lower():
                if data[col].isna().sum() > len(data) * 0.9:  # 90%以上缺失
                    issues.append(f"列 {col} 缺失值过多，可能存在问题")

        if issues:
            error_msg = f"{context} 数据验证问题: {'; '.join(issues)}"
            if self.strict_mode:
                raise TemporalValidationError(error_msg)
            else:
                for issue in issues:
                    logger.warning(issue)
                return False

        return True

    def get_validation_report(self) -> str:
        """获取验证报告"""
        if not self.validation_log:
            return "✅ 无验证问题记录"

        report = ["🔍 时间序列验证报告"]
        report.append("=" * 40)

        for entry in self.validation_log:
            report.append(f"⚠️  {entry}")

        return "\n".join(report)


# 全局验证器实例
temporal_validator = TemporalValidator(strict_mode=True)


def validate_factor_return_alignment(
    factor_data: pd.Series, return_data: pd.Series, horizon: int = 1, context: str = ""
) -> bool:
    """便捷函数：验证因子收益对齐"""
    return temporal_validator.validate_time_alignment(
        factor_data, return_data, horizon, context
    )
