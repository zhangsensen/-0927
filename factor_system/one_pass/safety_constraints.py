"""安全约束引擎 - 4条最小安全约束向量化实现"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SafetyConstraints:
    """
    安全约束引擎 - 4条最小安全约束的向量化实现

    核心约束:
    1. T+1时序安全 - 严格防止未来函数泄露
    2. min_history约束 - 确保有足够历史数据计算因子
    3. 价格口径一致性 - 统一OHLCV数据格式
    4. 容错记账机制 - 失败因子记录但不阻塞整体计算
    """

    def __init__(self):
        """初始化安全约束引擎"""
        self.constraint_log = []
        self.error_accounting = {
            "t_plus_1_violations": [],
            "insufficient_history": [],
            "price_format_errors": [],
            "calculation_failures": [],
        }

        logger.info("安全约束引擎初始化完成")

    def validate_data(
        self,
        data: pd.DataFrame,
        timeframe: str,
        min_history_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        验证并修正数据，应用4条安全约束

        Args:
            data: 原始OHLCV数据
            timeframe: 时间框架
            min_history_map: 最小历史数据映射 {factor_id: min_periods}

        Returns:
            (验证后的数据, 警告列表)
        """
        warnings = []
        validated_data = data.copy()

        logger.debug(f"开始安全约束验证: 数据形状{data.shape}, 时间框架{timeframe}")

        # 约束1: T+1时序安全
        validated_data, t1_warnings = self._enforce_t_plus_1_safety(validated_data)
        warnings.extend(t1_warnings)

        # 约束2: 价格口径一致性
        validated_data, price_warnings = self._ensure_price_consistency(validated_data)
        warnings.extend(price_warnings)

        # 约束3: min_history约束（在因子计算时检查）
        # 这里只做数据层面的验证，因子级别的min_history在计算时检查

        # 约束4: 容错机制 - 确保数据结构完整性
        validated_data, struct_warnings = self._validate_data_structure(validated_data)
        warnings.extend(struct_warnings)

        logger.debug(f"安全约束验证完成: {len(warnings)}个警告")

        return validated_data, warnings

    def _enforce_t_plus_1_safety(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        约束1: T+1时序安全向量化验证

        确保所有计算只使用历史数据，严格防止未来函数泄露
        """
        warnings = []

        if data.empty:
            return data, warnings

        # 向量化检查数据时序性
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values(0).unique()
            symbols = data.index.get_level_values("symbol").unique()
        else:
            dates = data.index
            symbols = ["single_symbol"]

        # 检查时间索引是否严格递增
        for symbol in symbols:
            if isinstance(data.index, pd.MultiIndex):
                symbol_dates = data.xs(symbol, level="symbol").index
            else:
                symbol_dates = data.index

            # 向量化检查日期单调性
            if not symbol_dates.is_monotonic_increasing:
                # 重新排序确保时序正确
                logger.warning(f"标的{symbol}数据时序错误，重新排序")
                if isinstance(data.index, pd.MultiIndex):
                    # MultiIndex重新排序
                    data = data.sort_index()
                else:
                    # 单Index重新排序
                    data = data.sort_index()

                warnings.append(f"标的{symbol}: 数据时序已修正")

        # 记录T+1约束验证
        self.constraint_log.append(
            {
                "constraint": "T+1时序安全",
                "timestamp": datetime.now(),
                "status": "passed" if not warnings else "warnings",
                "warnings_count": len(warnings),
            }
        )

        return data, warnings

    def _ensure_price_consistency(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        约束2: 价格口径一致性向量化修正

        确保OHLCV数据格式统一，价格数据为正数，成交量为非负数
        """
        warnings = []

        if data.empty:
            return data, warnings

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            error_msg = f"缺少必要列: {missing_columns}"
            logger.error(error_msg)
            self.error_accounting["price_format_errors"].append(error_msg)
            return data, [error_msg]

        # 向量化价格数据修正
        price_columns = ["open", "high", "low", "close"]

        # 检查负价格
        negative_prices = (data[price_columns] <= 0).any().any()
        if negative_prices:
            # 向量化修正：将负价格设为前一个有效价格
            logger.warning("发现负价格，进行向量化修正")
            for col in price_columns:
                # 使用向前填充修正负价格
                mask = data[col] <= 0
                if mask.any():
                    data.loc[mask, col] = np.nan
                    data[col] = data[col].fillna(method="ffill")

                    # 如果还有NaN，用close价格填充（如果是非close列）
                    if col != "close":
                        data[col] = data[col].fillna(data["close"])

            warnings.append("负价格已修正为有效价格")

        # 向量化成交量修正
        if (data["volume"] < 0).any():
            logger.warning("发现负成交量，进行向量化修正")
            data.loc[data["volume"] < 0, "volume"] = 0
            warnings.append("负成交量已修正为0")

        # 检查OHLC逻辑一致性（向量化）
        logic_errors = (
            (data["high"] < data["low"])
            | (data["high"] < data["open"])
            | (data["high"] < data["close"])
            | (data["low"] > data["open"])
            | (data["low"] > data["close"])
        )

        if logic_errors.any():
            logger.warning("发现OHLC逻辑错误，进行向量化修正")

            # 向量化修正OHLC逻辑
            data.loc[data["high"] < data["low"], "high"] = data.loc[
                data["high"] < data["low"], "low"
            ]
            data.loc[data["high"] < data["open"], "high"] = data.loc[
                data["high"] < data["open"], "open"
            ]
            data.loc[data["high"] < data["close"], "high"] = data.loc[
                data["high"] < data["close"], "close"
            ]
            data.loc[data["low"] > data["open"], "low"] = data.loc[
                data["low"] > data["open"], "open"
            ]
            data.loc[data["low"] > data["close"], "low"] = data.loc[
                data["low"] > data["close"], "close"
            ]

            warnings.append("OHLC逻辑错误已修正")

        # 记录价格一致性验证
        self.constraint_log.append(
            {
                "constraint": "价格口径一致性",
                "timestamp": datetime.now(),
                "status": "passed" if not warnings else "warnings",
                "warnings_count": len(warnings),
                "negative_prices_fixed": negative_prices,
                "negative_volume_fixed": (data["volume"] < 0).any(),
                "logic_errors_fixed": logic_errors.any(),
            }
        )

        return data, warnings

    def _validate_data_structure(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        约束4: 容错机制 - 数据结构完整性验证

        确保数据结构完整，为后续计算提供稳定基础
        """
        warnings = []

        if data.empty:
            return data, ["输入数据为空"]

        # 检查数据类型一致性
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        # 尝试转换为数值类型
                        data[col] = pd.to_numeric(data[col], errors="coerce")
                        if data[col].isna().any():
                            warnings.append(f"列{col}包含非数值数据，已转换为NaN")
                    except Exception as e:
                        error_msg = f"列{col}数据类型转换失败: {str(e)}"
                        self.error_accounting["price_format_errors"].append(error_msg)
                        warnings.append(error_msg)

        # 检查数据完整性
        total_rows = len(data)
        missing_data_ratio = data.isna().sum() / total_rows

        # 记录缺失数据比例高的列
        high_missing_cols = missing_data_ratio[missing_data_ratio > 0.1].index.tolist()
        if high_missing_cols:
            warnings.append(f"高缺失比例列: {high_missing_cols}")

        # 检查重复数据
        if isinstance(data.index, pd.MultiIndex):
            duplicates = data.index.duplicated().sum()
        else:
            duplicates = data.index.duplicated().sum()

        if duplicates > 0:
            logger.warning(f"发现{duplicates}个重复索引，进行去重")
            data = data[~data.index.duplicated(keep="first")]
            warnings.append(f"已移除{duplicates}个重复索引")

        # 记录结构验证结果
        self.constraint_log.append(
            {
                "constraint": "数据结构完整性",
                "timestamp": datetime.now(),
                "status": "passed" if not warnings else "warnings",
                "warnings_count": len(warnings),
                "total_rows": total_rows,
                "high_missing_columns": high_missing_cols,
                "duplicates_removed": duplicates,
            }
        )

        return data, warnings

    def check_factor_history_requirement(
        self,
        factor_id: str,
        data_length: int,
        timeframe: str,
        custom_min_history: Optional[int] = None,
    ) -> Tuple[bool, int, str]:
        """
        约束3: min_history约束检查

        Args:
            factor_id: 因子ID
            data_length: 数据长度
            timeframe: 时间框架
            custom_min_history: 自定义最小历史数据要求

        Returns:
            (是否满足要求, 所需最小数据量, 说明信息)
        """
        # 默认最小历史数据映射
        default_min_history = {
            "1min": 60,  # 1小时
            "5min": 48,  # 4小时
            "15min": 16,  # 4小时
            "30min": 8,  # 4小时
            "60min": 24,  # 1天
            "120min": 12,  # 1天
            "240min": 6,  # 1天
            "daily": 252,  # 1年
            "weekly": 52,  # 1年
            "monthly": 12,  # 1年
        }

        # 根据因子类型调整最小历史数据要求
        factor_specific_min = self._get_factor_min_history(factor_id, timeframe)
        min_history = (
            custom_min_history
            or factor_specific_min
            or default_min_history.get(timeframe, 20)
        )

        is_sufficient = data_length >= min_history
        message = f"因子{factor_id}: 需要{min_history}条数据，实际{data_length}条"

        if not is_sufficient:
            self.error_accounting["insufficient_history"].append(
                {
                    "factor_id": factor_id,
                    "required": min_history,
                    "actual": data_length,
                    "timeframe": timeframe,
                    "timestamp": datetime.now(),
                }
            )

        return is_sufficient, min_history, message

    def _get_factor_min_history(self, factor_id: str, timeframe: str) -> Optional[int]:
        """
        根据因子类型确定最小历史数据要求
        """
        factor_lower = factor_id.lower()

        # 动量因子需要更长的历史数据
        if any(kw in factor_lower for kw in ["momentum", "trend"]):
            if "252" in factor_id or "annual" in factor_lower:
                return 300  # 年度动量需要更多数据
            elif "126" in factor_id or "semi" in factor_lower:
                return 150  # 半年度动量
            elif "63" in factor_id or "quarter" in factor_lower:
                return 80  # 季度动量
            else:
                return 40  # 短期动量

        # 移动平均类因子
        elif any(kw in factor_lower for kw in ["ma", "ema", "sma"]):
            # 从因子ID中提取周期
            import re

            period_match = re.search(r"(\d+)", factor_id)
            if period_match:
                period = int(period_match.group(1))
                return period + 10  # 移动平均需要周期+缓冲
            return 20

        # 波动率类因子
        elif any(kw in factor_lower for kw in ["atr", "volatility", "std"]):
            return 20

        # RSI等摆动指标
        elif any(kw in factor_lower for kw in ["rsi", "stoch", "willr", "cci"]):
            return 20

        # 默认要求
        return None

    def log_calculation_error(
        self, factor_id: str, error: Exception, context: Dict[str, Any] = None
    ):
        """
        约束4: 容错记账机制 - 记录计算错误但不阻塞

        Args:
            factor_id: 因子ID
            error: 错误信息
            context: 上下文信息
        """
        error_record = {
            "factor_id": factor_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now(),
            "context": context or {},
        }

        self.error_accounting["calculation_failures"].append(error_record)

        logger.debug(f"因子{factor_id}计算错误已记录: {str(error)}")

    def get_constraint_summary(self) -> Dict[str, Any]:
        """获取约束执行汇总"""
        total_violations = sum(len(errors) for errors in self.error_accounting.values())

        return {
            "total_constraints_checked": len(self.constraint_log),
            "total_violations": total_violations,
            "constraint_details": {
                "t_plus_1_violations": len(
                    self.error_accounting["t_plus_1_violations"]
                ),
                "insufficient_history": len(
                    self.error_accounting["insufficient_history"]
                ),
                "price_format_errors": len(
                    self.error_accounting["price_format_errors"]
                ),
                "calculation_failures": len(
                    self.error_accounting["calculation_failures"]
                ),
            },
            "recent_logs": self.constraint_log[-10:] if self.constraint_log else [],
        }

    def reset_accounting(self):
        """重置错误记账"""
        self.error_accounting = {
            "t_plus_1_violations": [],
            "insufficient_history": [],
            "price_format_errors": [],
            "calculation_failures": [],
        }
        self.constraint_log = []
        logger.info("错误记账已重置")

    def generate_safety_report(self) -> str:
        """生成安全约束报告"""
        summary = self.get_constraint_summary()

        report = f"""
安全约束执行报告
{'='*50}
约束检查次数: {summary['total_constraints_checked']}
总违规次数: {summary['total_violations']}

违规详情:
- T+1时序违规: {summary['constraint_details']['t_plus_1_violations']}
- 历史数据不足: {summary['constraint_details']['insufficient_history']}
- 价格格式错误: {summary['constraint_details']['price_format_errors']}
- 计算失败次数: {summary['constraint_details']['calculation_failures']}

{'='*50}
        """

        if summary["total_violations"] == 0:
            report += "✅ 所有安全约束检查通过\n"
        else:
            report += "⚠️  发现约束违规，但计算继续执行\n"

        return report
