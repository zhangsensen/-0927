"""
数据验证层 | Data Validation Layer
================================================================================
职责：
1. 满窗NaN检查 (Full Window NaN Check)
2. 覆盖率验证 (Coverage Validation: 97%)
3. 成交量异常检测 (Volume Anomaly Detection)
4. 金额估算 (Amount Estimation)

用于确保数据质量满足精确因子计算的要求
================================================================================
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器

    特点：
    - 严格的满窗检查（窗口内所有日期都需要数据）
    - 覆盖率阈值可配置
    - 异常检测和日志记录
    - 支持多种验证模式
    """

    def __init__(
        self,
        coverage_threshold: float = 0.97,
        min_window_days: int = 20,
        volume_anomaly_z_score: float = 3.0,
        verbose: bool = True,
    ):
        """
        初始化数据验证器

        Args:
            coverage_threshold: 覆盖率阈值 (default: 0.97 = 97%)
            min_window_days: 最小窗口天数，用于验证全窗数据
            volume_anomaly_z_score: 成交量异常的z-score阈值
            verbose: 是否输出详细日志
        """
        self.coverage_threshold = coverage_threshold
        self.min_window_days = min_window_days
        self.volume_anomaly_z_score = volume_anomaly_z_score
        self.verbose = verbose

        self.validation_report = {}

    def check_full_window(
        self,
        prices: Dict[str, pd.DataFrame],
        window_days: int,
        symbol: Optional[str] = None,
        consecutive_nan_threshold: Optional[int] = None,
    ) -> bool:
        """
        检查满窗条件：检查是否存在超过阈值的连续NaN

        根据精确定义规则：
        - 原始缺失 → 保留NaN（无向前填充）
        - 偶发缺失（覆盖率>97%）→ 允许（将在标准化时处理）
        - 连续缺失≥window_days → 因子计算失效，不允许

        Args:
            prices: 价格数据 {'close': price_df}
            window_days: 检查窗口大小（天数）
            symbol: 特定标的，如果为None则检查全部
            consecutive_nan_threshold: 连续NaN容许上限
                如果为None，则使用window_days（即不允许连续NaN≥window_days）

        Returns:
            bool: 通过验证返回True

        Example:
            >>> prices = {'close': pd.DataFrame(...)}
            >>> validator = DataValidator()
            >>> is_valid = validator.check_full_window(prices, window_days=20)
        """
        close_prices = prices.get("close")
        if close_prices is None:
            logger.error("未找到'close'列")
            return False

        if consecutive_nan_threshold is None:
            consecutive_nan_threshold = window_days

        symbols_to_check = [symbol] if symbol else close_prices.columns

        violations = []
        for sym in symbols_to_check:
            if sym not in close_prices.columns:
                violations.append(f"{sym}: 不存在")
                continue

            series = close_prices[sym]

            # 寻找连续NaN块
            nan_mask = series.isna()

            # 计算连续NaN的长度
            nan_groups = (nan_mask != nan_mask.shift()).cumsum()
            consecutive_counts = nan_mask.groupby(nan_groups).sum()

            # 检查是否有超过阈值的连续NaN
            max_consecutive = (
                consecutive_counts.max() if len(consecutive_counts) > 0 else 0
            )

            if max_consecutive >= consecutive_nan_threshold:
                violations.append(
                    f"{sym}: 存在{int(max_consecutive)}个连续NaN "
                    f"(超过阈值{consecutive_nan_threshold})"
                )

        if violations:
            if self.verbose:
                logger.warning(f"满窗检查失败:\n" + "\n".join(violations[:10]))
            self.validation_report["full_window_violations"] = violations
            return False
        else:
            if self.verbose:
                logger.info(f"✓ 满窗检查通过 (window_days={window_days})")
            self.validation_report["full_window_violations"] = []
            return True

    def check_coverage(
        self,
        prices: Dict[str, pd.DataFrame],
        threshold: Optional[float] = None,
        by_symbol: bool = False,
    ) -> Dict:
        """
        检查覆盖率：每个标的的非NaN数据比例

        精确因子计算需要足够的数据覆盖率（默认97%）。

        Args:
            prices: 价格数据
            threshold: 覆盖率阈值，如果为None使用默认值
            by_symbol: 是否按标的返回结果

        Returns:
            Dict: {
                'valid_symbols': 通过验证的标的列表,
                'coverage_stats': {symbol: coverage_ratio},
                'overall_coverage': 整体覆盖率,
                'passed': bool
            }

        Example:
            >>> result = validator.check_coverage(prices, threshold=0.97)
            >>> print(f"有效标的: {len(result['valid_symbols'])}")
        """
        if threshold is None:
            threshold = self.coverage_threshold

        close_prices = prices.get("close")
        if close_prices is None:
            return {"passed": False, "error": "未找到close列"}

        coverage_stats = {}
        valid_symbols = []

        for symbol in close_prices.columns:
            series = close_prices[symbol]
            total = len(series)
            non_nan = series.notna().sum()
            coverage = non_nan / total if total > 0 else 0

            coverage_stats[symbol] = coverage

            if coverage >= threshold:
                valid_symbols.append(symbol)

        # 防止除以零
        total_rows = len(close_prices)
        if total_rows > 0:
            overall_coverage = close_prices.notna().sum().mean() / total_rows
        else:
            overall_coverage = 0

        result = {
            "valid_symbols": valid_symbols,
            "coverage_stats": coverage_stats,
            "overall_coverage": overall_coverage,
            "passed": len(valid_symbols) == len(close_prices.columns),
            "threshold": threshold,
        }

        if self.verbose:
            logger.info(
                f"覆盖率检查: 整体覆盖率={overall_coverage:.2%}, "
                f"有效标的={len(valid_symbols)}/{len(close_prices.columns)}"
            )

        self.validation_report["coverage"] = result
        return result

    def detect_volume_anomalies(
        self,
        prices: Dict[str, pd.DataFrame],
        window: int = 20,
        z_score_threshold: Optional[float] = None,
    ) -> Dict:
        """
        检测成交量异常

        使用z-score检测异常高成交量（可能的交易中断或异常事件）

        Args:
            prices: 价格数据
            window: 计算z-score的窗口
            z_score_threshold: z-score阈值，如果为None使用默认值

        Returns:
            Dict: {
                'anomalies': {symbol: [(date, z_score), ...]},
                'symbols_with_anomalies': 有异常的标的列表,
                'anomaly_count': 总异常数
            }
        """
        if z_score_threshold is None:
            z_score_threshold = self.volume_anomaly_z_score

        volume_prices = prices.get("volume")
        if volume_prices is None:
            return {"anomalies": {}, "symbols_with_anomalies": [], "anomaly_count": 0}

        anomalies = {}

        for symbol in volume_prices.columns:
            volume = volume_prices[symbol]

            # 计算rolling z-score
            rolling_mean = volume.rolling(window=window).mean()
            rolling_std = volume.rolling(window=window).std()
            z_scores = (volume - rolling_mean) / rolling_std

            # 找出异常值
            anomaly_mask = z_scores.abs() > z_score_threshold
            anomaly_dates = z_scores[anomaly_mask].index.tolist()

            if anomaly_dates:
                anomalies[symbol] = [
                    (date, float(z_scores[date])) for date in anomaly_dates
                ]

        result = {
            "anomalies": anomalies,
            "symbols_with_anomalies": list(anomalies.keys()),
            "anomaly_count": sum(len(v) for v in anomalies.values()),
        }

        if self.verbose and result["anomaly_count"] > 0:
            logger.warning(
                f"检测到{result['anomaly_count']}个成交量异常 "
                f"(涉及{len(anomalies)}个标的)"
            )

        self.validation_report["volume_anomalies"] = result
        return result

    def add_amount_column(
        self, prices: Dict[str, pd.DataFrame], fill_method: str = "typical_price"
    ) -> Dict[str, pd.DataFrame]:
        """
        添加成交金额列（如果原数据不包含）

        因子计算中有些需要成交金额：
        amount = volume * price

        Args:
            prices: 价格数据 {symbol: price_df}
            fill_method: 金额计算方法
                - 'typical_price': (high + low + close) / 3 * volume
                - 'close': close * volume
                - 'vwap': (high + low) / 2 * volume

        Returns:
            Dict: 新增amount列的价格数据

        Example:
            >>> prices = validator.add_amount_column(prices)
            >>> assert 'amount' in prices
        """
        if "amount" in prices:
            if self.verbose:
                logger.info("✓ 已存在'amount'列")
            return prices

        prices_copy = prices.copy()

        high = prices_copy.get("high")
        low = prices_copy.get("low")
        close = prices_copy.get("close")
        volume = prices_copy.get("volume")

        if volume is None:
            logger.error("缺少'volume'列，无法计算amount")
            return prices_copy

        if (
            fill_method == "typical_price"
            and high is not None
            and low is not None
            and close is not None
        ):
            typical_price = (high + low + close) / 3
            amount = typical_price * volume
        elif fill_method == "vwap" and high is not None and low is not None:
            vwap = (high + low) / 2
            amount = vwap * volume
        elif close is not None:
            amount = close * volume
        else:
            logger.error("无法计算amount")
            return prices_copy

        prices_copy["amount"] = amount

        if self.verbose:
            logger.info(f"✓ 已添加'amount'列 (方法: {fill_method})")

        return prices_copy

    def validate_dates(
        self,
        prices: Dict[str, pd.DataFrame],
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> Dict:
        """
        验证日期范围和连续性

        Args:
            prices: 价格数据
            min_date: 最小日期 (可选)
            max_date: 最大日期 (可选)

        Returns:
            Dict: 日期验证结果
        """
        close_prices = prices.get("close")
        if close_prices is None:
            return {"passed": False, "error": "未找到close列"}

        date_range = {
            "min_date": close_prices.index.min(),
            "max_date": close_prices.index.max(),
            "total_days": len(close_prices),
            "business_days": (close_prices.index.notna()).sum(),
        }

        passed = True
        if min_date and date_range["min_date"] < pd.to_datetime(min_date):
            passed = False
            logger.warning(f"起始日期太早: {date_range['min_date']} < {min_date}")

        if max_date and date_range["max_date"] > pd.to_datetime(max_date):
            passed = False
            logger.warning(f"终止日期太晚: {date_range['max_date']} > {max_date}")

        result = {**date_range, "passed": passed}
        self.validation_report["dates"] = result

        if self.verbose:
            logger.info(
                f"✓ 日期范围: {date_range['min_date'].date()} 到 {date_range['max_date'].date()}"
            )

        return result

    def full_validation(
        self,
        prices: Dict[str, pd.DataFrame],
        window_days: int = 20,
        coverage_threshold: float = 0.97,
    ) -> Dict:
        """
        完整的数据验证流程

        执行所有验证检查，返回汇总结果

        Args:
            prices: 价格数据
            window_days: 满窗检查的窗口大小
            coverage_threshold: 覆盖率阈值

        Returns:
            Dict: {
                'passed': 是否通过所有检查,
                'checks': {
                    'full_window': bool,
                    'coverage': bool,
                    'dates': bool,
                    'volume_anomalies': int (异常数)
                },
                'details': {...}
            }
        """
        checks = {
            "full_window": self.check_full_window(prices, window_days),
            "coverage": self.check_coverage(prices, coverage_threshold)["passed"],
            "dates": self.validate_dates(prices)["passed"],
            "volume_anomalies": self.detect_volume_anomalies(prices)["anomaly_count"],
        }

        overall_passed = (
            checks["full_window"] and checks["coverage"] and checks["dates"]
        )

        result = {
            "passed": overall_passed,
            "checks": checks,
            "details": self.validation_report,
        }

        if self.verbose:
            status = "✅ 通过" if overall_passed else "❌ 失败"
            logger.info(f"\n========== 数据验证总结 {status} ==========")
            logger.info(f"满窗检查: {'✓' if checks['full_window'] else '✗'}")
            logger.info(f"覆盖率检查: {'✓' if checks['coverage'] else '✗'}")
            logger.info(f"日期检查: {'✓' if checks['dates'] else '✗'}")
            logger.info(f"成交量异常: {checks['volume_anomalies']}个")
            logger.info("=" * 50)

        return result

    def get_report(self) -> pd.DataFrame:
        """生成验证报告"""
        return pd.DataFrame([self.validation_report]).T


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 示例代码（需要真实数据）
    print("Data Validator 已就绪")
    print("使用方式:")
    print("  validator = DataValidator(coverage_threshold=0.97)")
    print("  result = validator.full_validation(prices, window_days=20)")
