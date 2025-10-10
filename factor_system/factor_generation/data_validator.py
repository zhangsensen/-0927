#!/usr/bin/env python3
"""
数据校对模块 - 验证重采样数据与原始数据的一致性

功能：
1. 对比重采样数据与原始数据的OHLCV一致性
2. 验证数据完整性和准确性
3. 生成校对报告
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """数据校对器"""

    def __init__(self):
        """初始化校对器"""
        self.validation_results = []
        logger.info("数据校对器初始化完成")

    def validate_resampled_data(
        self,
        original_data: pd.DataFrame,
        resampled_data: pd.DataFrame,
        timeframe: str,
        symbol: str,
    ) -> Dict:
        """
        校对重采样数据与原始数据

        Args:
            original_data: 原始数据
            resampled_data: 重采样数据
            timeframe: 时间框架
            symbol: 股票代码

        Returns:
            校对结果字典
        """
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            # 1. 基本检查
            if original_data.empty or resampled_data.empty:
                result["status"] = "FAIL"
                result["errors"].append("数据为空")
                return result

            # 2. 时间范围检查
            orig_start = original_data.index.min()
            orig_end = original_data.index.max()
            resamp_start = resampled_data.index.min()
            resamp_end = resampled_data.index.max()

            if resamp_start < orig_start or resamp_end > orig_end:
                result["warnings"].append("重采样数据时间范围超出原始数据范围")

            # 3. OHLCV列检查
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [
                col for col in required_cols if col not in resampled_data.columns
            ]
            if missing_cols:
                result["status"] = "FAIL"
                result["errors"].append(f"缺少必要列: {missing_cols}")
                return result

            # 4. 数据逻辑检查
            # 检查 high >= max(open, close) 和 low <= min(open, close)
            invalid_high = (
                resampled_data["high"] < resampled_data[["open", "close"]].max(axis=1)
            ).sum()
            invalid_low = (
                resampled_data["low"] > resampled_data[["open", "close"]].min(axis=1)
            ).sum()

            if invalid_high > 0:
                result["warnings"].append(f"{invalid_high} 行数据high值异常")
            if invalid_low > 0:
                result["warnings"].append(f"{invalid_low} 行数据low值异常")

            # 5. 成交量检查
            if (resampled_data["volume"] < 0).any():
                result["warnings"].append("存在负成交量")

            # 6. 数据完整性统计
            result["metrics"] = {
                "original_rows": len(original_data),
                "resampled_rows": len(resampled_data),
                "compression_ratio": (
                    len(original_data) / len(resampled_data)
                    if len(resampled_data) > 0
                    else 0
                ),
                "null_count": resampled_data.isnull().sum().sum(),
                "duplicate_timestamps": resampled_data.index.duplicated().sum(),
            }

            # 7. 价格连续性检查（简单检查）
            price_changes = resampled_data["close"].pct_change().abs()
            extreme_changes = (price_changes > 0.2).sum()  # 超过20%的变化
            if extreme_changes > 0:
                result["warnings"].append(f"{extreme_changes} 个极端价格变化点")

            logger.debug(f"{symbol} {timeframe} 校对完成: {result['status']}")

        except Exception as e:
            result["status"] = "ERROR"
            result["errors"].append(f"校对过程异常: {str(e)}")
            logger.error(f"{symbol} {timeframe} 校对异常: {e}")

        return result

    def compare_with_existing_data(
        self, resampled_file: Path, original_file: Path, symbol: str, timeframe: str
    ) -> Dict:
        """
        对比重采样文件与原始文件

        Args:
            resampled_file: 重采样文件路径
            original_file: 原始文件路径
            symbol: 股票代码
            timeframe: 时间框架

        Returns:
            对比结果
        """
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "PASS",
            "differences": [],
            "similarity_score": 1.0,
        }

        try:
            if not original_file.exists():
                result["status"] = "SKIP"
                result["differences"].append("原始文件不存在，跳过对比")
                return result

            # 读取数据
            resampled_data = pd.read_parquet(resampled_file)
            original_data = pd.read_parquet(original_file)

            # 确保时间索引
            if "timestamp" in resampled_data.columns:
                resampled_data.set_index("timestamp", inplace=True)
            if "timestamp" in original_data.columns:
                original_data.set_index("timestamp", inplace=True)

            # 找到共同时间范围
            common_start = max(resampled_data.index.min(), original_data.index.min())
            common_end = min(resampled_data.index.max(), original_data.index.max())

            resampled_common = resampled_data[common_start:common_end]
            original_common = original_data[common_start:common_end]

            if len(resampled_common) == 0 or len(original_common) == 0:
                result["status"] = "SKIP"
                result["differences"].append("无共同时间范围")
                return result

            # 对比OHLCV数据
            common_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [
                col
                for col in common_cols
                if col in resampled_common.columns and col in original_common.columns
            ]

            total_diff = 0
            total_points = 0

            for col in available_cols:
                # 计算相对差异
                diff = np.abs(resampled_common[col] - original_common[col]) / (
                    original_common[col] + 1e-8
                )
                avg_diff = diff.mean()
                max_diff = diff.max()

                total_diff += avg_diff
                total_points += 1

                if avg_diff > 0.01:  # 平均差异超过1%
                    result["differences"].append(f"{col}列平均差异: {avg_diff:.4f}")
                if max_diff > 0.05:  # 最大差异超过5%
                    result["differences"].append(f"{col}列最大差异: {max_diff:.4f}")

            # 计算相似度分数
            if total_points > 0:
                avg_total_diff = total_diff / total_points
                result["similarity_score"] = max(0, 1 - avg_total_diff)

            if result["similarity_score"] < 0.95:
                result["status"] = "WARN"

            logger.debug(
                f"{symbol} {timeframe} 对比完成: 相似度 {result['similarity_score']:.4f}"
            )

        except Exception as e:
            result["status"] = "ERROR"
            result["differences"].append(f"对比过程异常: {str(e)}")
            logger.error(f"{symbol} {timeframe} 对比异常: {e}")

        return result

    def generate_validation_report(
        self, results: List[Dict], output_path: Path
    ) -> None:
        """
        生成校对报告

        Args:
            results: 校对结果列表
            output_path: 报告输出路径
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("数据校对报告")
            report_lines.append("=" * 80)
            report_lines.append(f"总计校对: {len(results)} 个数据集")

            # 统计结果
            status_counts = {}
            for result in results:
                status = result.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1

            report_lines.append("\n校对结果统计:")
            for status, count in status_counts.items():
                report_lines.append(f"  {status}: {count}")

            # 详细结果
            report_lines.append("\n详细校对结果:")
            report_lines.append("-" * 80)

            for result in results:
                symbol = result.get("symbol", "UNKNOWN")
                timeframe = result.get("timeframe", "UNKNOWN")
                status = result.get("status", "UNKNOWN")

                report_lines.append(f"\n{symbol} - {timeframe}: {status}")

                # 错误信息
                errors = result.get("errors", [])
                if errors:
                    report_lines.append("  错误:")
                    for error in errors:
                        report_lines.append(f"    - {error}")

                # 警告信息
                warnings = result.get("warnings", [])
                if warnings:
                    report_lines.append("  警告:")
                    for warning in warnings:
                        report_lines.append(f"    - {warning}")

                # 差异信息
                differences = result.get("differences", [])
                if differences:
                    report_lines.append("  差异:")
                    for diff in differences:
                        report_lines.append(f"    - {diff}")

                # 相似度分数
                similarity_score = result.get("similarity_score")
                if similarity_score is not None:
                    report_lines.append(f"  相似度: {similarity_score:.4f}")

                # 指标信息
                metrics = result.get("metrics", {})
                if metrics:
                    report_lines.append("  指标:")
                    for key, value in metrics.items():
                        report_lines.append(f"    {key}: {value}")

            report_lines.append("\n" + "=" * 80)

            # 写入报告
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))

            logger.info(f"校对报告已生成: {output_path}")

        except Exception as e:
            logger.error(f"生成校对报告失败: {e}")


def create_validator() -> DataValidator:
    """创建数据校对器实例"""
    return DataValidator()
