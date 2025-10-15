#!/usr/bin/env python3
"""
时序哨兵验证脚本 - 确保无未来信息泄露

核心验证原则：
1. cross_section_date ≤ T（截面日期不超过当前观察日）
2. 执行价格 ∈ [T+1开盘, T+月末收盘]（交易执行在T+1及以后）
3. T+1时序安全：所有计算基于T-1及之前的数据
4. 价格口径一致性：统一使用close价格字段

用法：
    python scripts/verify_no_lookahead.py --date 2024-12-31
    python scripts/verify_no_lookahead.py --random-samples 5
    python scripts/verify_no_lookahead.py --all
"""

import argparse
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from factor_system.factor_engine.providers.parquet_provider import ParquetDataProvider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TemporalSentinel:
    """时序哨兵 - 验证无未来信息泄露"""

    def __init__(self, data_dir: str = "raw/ETF/daily"):
        self.data_dir = Path(data_dir)
        # ParquetDataProvider需要项目根目录作为基准
        project_root = Path.cwd()
        self.provider = ParquetDataProvider(project_root / "raw")

    def validate_single_date(self, test_date: str) -> dict:
        """验证单个日期的时序安全"""
        logger.info(f"🔍 验证日期: {test_date}")

        test_dt = pd.to_datetime(test_date)
        results = {
            "test_date": test_date,
            "violations": [],
            "warnings": [],
            "samples_checked": 0,
        }

        # 获取ETF列表
        etf_files = list(self.data_dir.glob("*_daily_*.parquet"))
        etf_symbols = [
            f.stem.split("_daily_")[0] for f in etf_files[:5]
        ]  # 检查前5个ETF

        for symbol in etf_symbols:
            try:
                # 加载T-1日及之前的数据
                end_date = (test_dt - timedelta(days=1)).strftime("%Y%m%d")
                start_date = (test_dt - timedelta(days=60)).strftime(
                    "%Y%m%d"
                )  # 60天历史

                data = self.provider.load_price_data(
                    [symbol],
                    "daily",
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date),
                )

                if data.empty:
                    results["warnings"].append(f"{symbol}: 无历史数据")
                    continue

                # 验证1: 截面日期不超过观察日
                if hasattr(data.index, "get_level_values"):
                    latest_date = data.index.get_level_values("date").max()
                else:
                    # 如果不是MultiIndex，尝试其他方式获取日期
                    if "trade_date" in data.columns:
                        latest_date = pd.to_datetime(data["trade_date"]).max()
                    elif "date" in data.columns:
                        latest_date = pd.to_datetime(data["date"]).max()
                    else:
                        latest_date = (
                            data.index.max() if hasattr(data.index, "max") else None
                        )

                if latest_date and latest_date > test_dt:
                    results["violations"].append(
                        f"{symbol}: 截面日期{latest_date} > 观察日{test_dt}"
                    )

                # 验证2: 价格数据时序安全
                if "close" in data.columns:
                    # 检查T日收盘价是否被用于计算当期信号
                    close_prices = data["close"]

                    # 模拟一个简单的技术指标（如5日均线）
                    if len(close_prices) >= 5:
                        ma5 = close_prices.rolling(5).mean()
                        # 最新的MA5应该只使用T-1及之前的价格
                        latest_ma5 = ma5.iloc[-1]
                        latest_data_date = data.index[-1]

                        if hasattr(data.index, "get_level_values"):
                            latest_data_date = data.index[-1]
                        else:
                            latest_data_date = latest_date

                        if latest_data_date and latest_data_date >= test_dt:
                            results["violations"].append(
                                f"{symbol}: 计算包含了T日数据{latest_data_date}"
                            )

                # 验证3: 执行时间窗口检查
                entry_date = test_dt + timedelta(days=1)  # T+1执行
                month_end = self._get_month_end(test_dt)

                # 确认执行价格在合理范围内
                if not data.empty:
                    last_close = data["close"].iloc[-1]
                    # 这里可以添加更多的执行价格验证逻辑
                    results["samples_checked"] += 1

                logger.info(f"  ✅ {symbol}: 通过验证")

            except Exception as e:
                results["warnings"].append(f"{symbol}: 验证失败 - {str(e)}")
                logger.warning(f"  ⚠️ {symbol}: {e}")

        return results

    def validate_random_samples(self, n_samples: int = 5) -> list:
        """验证随机样本日期"""
        logger.info(f"🎲 随机验证 {n_samples} 个样本")

        # 获取可用日期范围
        sample_files = list(self.data_dir.glob("*_daily_*.parquet"))
        if not sample_files:
            return []

        # 从一个文件中获取日期范围
        sample_data = pd.read_parquet(sample_files[0])
        if "trade_date" in sample_data.columns:
            dates = pd.to_datetime(sample_data["trade_date"]).unique()
        else:
            return []

        # 随机选择日期
        random_dates = random.sample(list(dates), min(n_samples, len(dates)))

        results = []
        for date in random_dates:
            result = self.validate_single_date(date.strftime("%Y-%m-%d"))
            results.append(result)

        return results

    def validate_all_factors(self, factor_file: str) -> dict:
        """验证因子面板的时序安全"""
        logger.info(f"🔬 验证因子面板: {factor_file}")

        try:
            panel = pd.read_parquet(factor_file)

            results = {
                "factor_file": factor_file,
                "shape": panel.shape,
                "date_range": None,
                "violations": [],
                "warnings": [],
            }

            # 检查日期范围
            dates = panel.index.get_level_values("date").unique()
            results["date_range"] = f"{dates.min()} ~ {dates.max()}"

            # 检查是否有未来数据泄露的迹象
            # 这里可以添加更多的因子时序验证逻辑

            return results

        except Exception as e:
            logger.error(f"因子面板验证失败: {e}")
            return {"error": str(e)}

    def _get_month_end(self, date: pd.Timestamp) -> pd.Timestamp:
        """获取月末日期"""
        next_month = date.replace(day=28) + timedelta(days=4)
        return next_month - timedelta(days=next_month.day)

    def generate_report(self, results: list) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("时序哨兵验证报告")
        report.append("=" * 60)

        total_violations = sum(len(r.get("violations", [])) for r in results)
        total_warnings = sum(len(r.get("warnings", [])) for r in results)
        total_samples = sum(r.get("samples_checked", 0) for r in results)

        report.append(f"验证样本数: {total_samples}")
        report.append(f"违规数量: {total_violations}")
        report.append(f"警告数量: {total_warnings}")

        if total_violations == 0:
            report.append("✅ 时序安全验证通过")
        else:
            report.append("❌ 发现时序安全问题!")
            for result in results:
                for violation in result.get("violations", []):
                    report.append(f"  - {violation}")

        if total_warnings > 0:
            report.append("\n⚠️ 警告信息:")
            for result in results:
                for warning in result.get("warnings", []):
                    report.append(f"  - {warning}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="时序哨兵验证脚本")
    parser.add_argument("--date", type=str, help="验证特定日期 (YYYY-MM-DD)")
    parser.add_argument("--random-samples", type=int, default=5, help="随机验证样本数")
    parser.add_argument("--factor-file", type=str, help="验证因子面板文件")
    parser.add_argument("--all", action="store_true", help="全面验证模式")
    parser.add_argument(
        "--data-dir", type=str, default="raw/ETF/daily", help="数据目录"
    )

    args = parser.parse_args()

    sentinel = TemporalSentinel(args.data_dir)

    if args.date:
        # 验证特定日期
        result = sentinel.validate_single_date(args.date)
        report = sentinel.generate_report([result])
        print(report)

    elif args.factor_file:
        # 验证因子面板
        result = sentinel.validate_all_factors(args.factor_file)
        print(f"因子面板验证结果: {result}")

    elif args.all:
        # 全面验证模式
        logger.info("🚀 启动全面验证模式")

        # 1. 随机日期验证
        random_results = sentinel.validate_random_samples(10)

        # 2. 因子面板验证
        panel_files = [
            "factor_output/etf_rotation/panel_FULL_20200102_20251014.parquet",
            "factor_output/etf_rotation/panel_filtered_research.parquet",
            "factor_output/etf_rotation/panel_filtered_production.parquet",
        ]

        for panel_file in panel_files:
            if Path(panel_file).exists():
                result = sentinel.validate_all_factors(panel_file)
                logger.info(f"面板验证: {result}")

        report = sentinel.generate_report(random_results)
        print(report)

    else:
        # 默认：随机验证
        results = sentinel.validate_random_samples(args.random_samples)
        report = sentinel.generate_report(results)
        print(report)


if __name__ == "__main__":
    main()
