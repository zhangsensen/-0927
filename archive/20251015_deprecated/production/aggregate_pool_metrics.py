#!/usr/bin/env python3
"""
分池指标总览汇总
合并三池回测指标，生成总表并校验 CI 阈值
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PoolMetricsAggregator:
    """分池指标聚合器"""

    def __init__(self, base_dir: str = "factor_output/etf_rotation_production"):
        self.base_dir = Path(base_dir)
        self.pools = ["A_SHARE", "QDII", "OTHER"]

        # CI 阈值（可配置化）
        self.thresholds = {
            "年化收益_min": 0.08,  # 8%
            "最大回撤_max": -0.30,  # -30%
            "夏普比率_min": 0.5,
            "月胜率_min": 0.45,
            "年化换手_max": 10.0,
        }

    def load_pool_metrics(self, pool_name: str) -> dict:
        """加载单池指标"""
        metrics_file = self.base_dir / f"panel_{pool_name}" / "backtest_metrics.json"

        if not metrics_file.exists():
            logger.warning(f"  ⚠️  {pool_name} 指标文件不存在: {metrics_file}")
            return None

        with open(metrics_file) as f:
            data = json.load(f)

        return {
            "pool": pool_name,
            "exec_mode": data.get("exec_mode", "N/A"),
            **self._parse_metrics(data.get("metrics", {})),
        }

    def _parse_metrics(self, metrics: dict) -> dict:
        """解析指标（去除百分号、转数值）"""
        parsed = {}

        for key, value in metrics.items():
            if key == "极端月":
                continue  # 跳过极端月（单独处理）

            if isinstance(value, str):
                # 去除百分号并转浮点
                if "%" in value:
                    parsed[key] = float(value.replace("%", "")) / 100
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            else:
                parsed[key] = value

        return parsed

    def aggregate(self) -> pd.DataFrame:
        """聚合所有池指标"""
        logger.info("=" * 80)
        logger.info("分池指标总览汇总")
        logger.info("=" * 80)

        all_metrics = []

        for pool in self.pools:
            logger.info(f"\n加载 {pool} 指标...")
            metrics = self.load_pool_metrics(pool)
            if metrics:
                all_metrics.append(metrics)
                logger.info(f"  ✅ 已加载")
            else:
                logger.warning(f"  ⚠️  跳过")

        if not all_metrics:
            logger.error("❌ 无可用指标")
            return None

        df = pd.DataFrame(all_metrics)

        # 计算加权平均（按池权重）
        weights = {"A_SHARE": 0.7, "QDII": 0.3, "OTHER": 0.0}  # OTHER 不参与组合

        weighted_metrics = {}
        for col in ["年化收益", "最大回撤", "夏普比率", "月胜率", "年化换手"]:
            if col in df.columns:
                weighted_metrics[col] = sum(
                    df[df["pool"] == pool][col].values[0] * weights.get(pool, 0)
                    for pool in df["pool"]
                    if pool in weights and weights[pool] > 0
                )

        # 添加组合行
        portfolio_row = {
            "pool": "PORTFOLIO",
            "exec_mode": "weighted",
            **weighted_metrics,
        }
        df = pd.concat([df, pd.DataFrame([portfolio_row])], ignore_index=True)

        logger.info("\n" + "=" * 80)
        logger.info("指标总览")
        logger.info("=" * 80)
        logger.info(df.to_string(index=False))

        return df

    def check_thresholds(self, df: pd.DataFrame) -> bool:
        """校验 CI 阈值"""
        logger.info("\n" + "=" * 80)
        logger.info("CI 阈值校验")
        logger.info("=" * 80)

        all_passed = True

        # 仅校验 PORTFOLIO 行
        portfolio = df[df["pool"] == "PORTFOLIO"]
        if portfolio.empty:
            logger.error("❌ 无组合指标，跳过校验")
            return False

        portfolio = portfolio.iloc[0]

        checks = [
            (
                "年化收益",
                portfolio.get("年化收益"),
                self.thresholds["年化收益_min"],
                ">=",
            ),
            (
                "最大回撤",
                portfolio.get("最大回撤"),
                self.thresholds["最大回撤_max"],
                ">=",
            ),
            (
                "夏普比率",
                portfolio.get("夏普比率"),
                self.thresholds["夏普比率_min"],
                ">=",
            ),
            ("月胜率", portfolio.get("月胜率"), self.thresholds["月胜率_min"], ">="),
            (
                "年化换手",
                portfolio.get("年化换手"),
                self.thresholds["年化换手_max"],
                "<=",
            ),
        ]

        for name, value, threshold, op in checks:
            if value is None:
                logger.warning(f"  ⚠️  {name}: 无数据")
                continue

            if op == ">=":
                passed = value >= threshold
            else:
                passed = value <= threshold

            status = "✅" if passed else "❌"
            logger.info(f"  {status} {name}: {value:.2%} {op} {threshold:.2%}")

            if not passed:
                all_passed = False

        return all_passed

    def save_report(self, df: pd.DataFrame):
        """保存汇总报告"""
        output_file = self.base_dir / "pool_metrics_summary.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\n✅ 汇总报告已保存: {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="分池指标总览汇总")
    parser.add_argument(
        "--base-dir",
        default="factor_output/etf_rotation_production",
        help="基础输出目录",
    )
    args = parser.parse_args()

    aggregator = PoolMetricsAggregator(base_dir=args.base_dir)

    # 聚合指标
    df = aggregator.aggregate()
    if df is None:
        logger.error("❌ 聚合失败")
        return False

    # 校验阈值
    passed = aggregator.check_thresholds(df)

    # 保存报告
    aggregator.save_report(df)

    logger.info("\n" + "=" * 80)
    if passed:
        logger.info("✅ 分池指标汇总完成，CI 阈值通过")
    else:
        logger.error("❌ 分池指标汇总完成，CI 阈值未通过")
    logger.info("=" * 80)

    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
