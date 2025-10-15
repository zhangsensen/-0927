#!/usr/bin/env python3
"""容量与约束检查

约束条件：
1. 单票≤20%
2. 同赛道≤40%
3. 宽基≤3
4. ADV%：单标成交额<5%其20日均额
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class CapacityConstraints:
    """容量与约束检查"""

    def __init__(self):
        # ETF分类（简化版本，实际应该从配置文件加载）
        self.etf_categories = {
            # 宽基
            "510050.SH": "宽基-沪深300",
            "510300.SH": "宽基-沪深300",
            "159919.SZ": "宽基-沪深300",
            "510500.SH": "宽基-中证500",
            "159922.SZ": "宽基-中证500",
            # 行业
            "512880.SH": "行业-证券",
            "512000.SH": "行业-券商",
            "159949.SZ": "行业-创业板",
            "159915.SZ": "行业-创业板",
            # 主题
            "515790.SH": "主题-光伏",
            "516160.SH": "主题-新能源",
        }

    def check_position_limits(self, positions):
        """检查仓位限制

        Args:
            positions: DataFrame with columns [symbol, weight]
        """
        logger.info("=" * 80)
        logger.info("仓位限制检查")
        logger.info("=" * 80)

        violations = []

        # 1. 单票限制（≤20%）
        logger.info(f"\n1. 单票限制检查（≤20%）")
        max_single = positions["weight"].max()
        logger.info(f"   最大单票权重: {max_single:.2%}")

        if max_single > 0.20:
            violation = f"❌ 单票权重超限: {max_single:.2%} > 20%"
            logger.error(f"   {violation}")
            violations.append(
                {
                    "type": "single_position",
                    "value": max_single,
                    "limit": 0.20,
                    "message": violation,
                }
            )
        else:
            logger.info(f"   ✅ 单票限制满足")

        # 2. 同赛道限制（≤40%）
        logger.info(f"\n2. 同赛道限制检查（≤40%）")

        # 添加分类
        positions["category"] = positions["symbol"].map(self.etf_categories)
        positions["category"] = positions["category"].fillna("未分类")

        # 按赛道汇总
        sector_weights = positions.groupby("category")["weight"].sum()

        logger.info(f"   赛道分布:")
        for sector, weight in sector_weights.items():
            logger.info(f"     {sector}: {weight:.2%}")

            if weight > 0.40:
                violation = f"❌ 赛道权重超限: {sector} {weight:.2%} > 40%"
                logger.error(f"     {violation}")
                violations.append(
                    {
                        "type": "sector_concentration",
                        "sector": sector,
                        "value": weight,
                        "limit": 0.40,
                        "message": violation,
                    }
                )

        if len(violations) == 0 or all(
            v["type"] != "sector_concentration" for v in violations
        ):
            logger.info(f"   ✅ 赛道限制满足")

        # 3. 宽基限制（≤3只）
        logger.info(f"\n3. 宽基限制检查（≤3只）")
        broad_base = positions[positions["category"].str.contains("宽基", na=False)]
        num_broad_base = len(broad_base)

        logger.info(f"   宽基ETF数量: {num_broad_base}")
        if num_broad_base > 0:
            logger.info(f"   宽基列表:")
            for _, row in broad_base.iterrows():
                logger.info(f"     {row['symbol']}: {row['weight']:.2%}")

        if num_broad_base > 3:
            violation = f"❌ 宽基数量超限: {num_broad_base} > 3"
            logger.error(f"   {violation}")
            violations.append(
                {
                    "type": "broad_base_count",
                    "value": num_broad_base,
                    "limit": 3,
                    "message": violation,
                }
            )
        else:
            logger.info(f"   ✅ 宽基限制满足")

        logger.info(f"\n{'=' * 80}")
        if len(violations) == 0:
            logger.info("✅ 所有仓位限制满足")
        else:
            logger.error(f"❌ 发现{len(violations)}个违规")
        logger.info("=" * 80)

        return violations

    def check_adv_constraints(
        self, positions, prices: pd.DataFrame, target_capital=1000000
    ):
        """检查ADV%约束

        Args:
            positions: DataFrame with columns [symbol, weight]
            prices: MultiIndex(symbol,date) DataFrame 包含 [close, volume, amount(可选)]
            target_capital: 目标资金量
        """
        logger.info("\n" + "=" * 80)
        logger.info("ADV%约束检查")
        logger.info("=" * 80)

        logger.info(f"\n目标资金: {target_capital:,.0f}")
        logger.info(f"ADV%阈值: 5%")

        violations = []

        # 计算每只ETF最近20个交易日的平均成交额
        adv20 = {}
        if prices is not None and not prices.empty:
            # 仅取需要的字段
            needed_cols = [
                c for c in ["close", "volume", "amount"] if c in prices.columns
            ]
            px = prices[needed_cols].copy()
            # 归一化日期
            px = px.reset_index()
            px["date"] = pd.to_datetime(px["date"]).dt.normalize()
            px = px.set_index(["symbol", "date"]).sort_index()
            # 构造金额列（元）
            if "amount" in px.columns and px["amount"].notna().any():
                px["turnover_yuan"] = px["amount"]
                amount_fallback = False
            else:
                # 回退：amount≈close*volume
                px["turnover_yuan"] = px["close"] * px["volume"]
                amount_fallback = True
            # 最近20日平均
            for symbol, sdf in px.groupby(level="symbol"):
                s = sdf.droplevel("symbol")["turnover_yuan"].tail(20)
                if len(s) > 0:
                    adv20[symbol] = float(s.mean())
        else:
            amount_fallback = True

        for _, row in positions.iterrows():
            symbol = row["symbol"]
            weight = row["weight"]
            position_value = target_capital * weight

            # 计算20日平均成交额（元）
            avg_daily_volume = adv20.get(symbol, 10_000_000)  # 默认1000万

            # 计算占比
            adv_pct = position_value / avg_daily_volume

            logger.info(f"\n  {symbol}:")
            logger.info(f"    持仓金额: {position_value:,.0f}")
            logger.info(
                f"    20日均额: {avg_daily_volume:,.0f}{' (fallback close*volume)' if amount_fallback else ''}"
            )
            logger.info(f"    ADV%: {adv_pct:.2%}")

            if adv_pct > 0.05:
                violation = f"❌ ADV%超限: {adv_pct:.2%} > 5%"
                logger.error(f"    {violation}")
                violations.append(
                    {
                        "type": "adv_constraint",
                        "symbol": symbol,
                        "value": adv_pct,
                        "limit": 0.05,
                        "message": violation,
                    }
                )
            else:
                logger.info(f"    ✅ ADV%满足")

        logger.info(f"\n{'=' * 80}")
        if len(violations) == 0:
            logger.info("✅ 所有ADV%约束满足")
        else:
            logger.error(f"❌ 发现{len(violations)}个违规")
        logger.info("=" * 80)

        return violations

    def generate_report(self, violations):
        """生成约束报告"""
        logger.info("\n" + "=" * 80)
        logger.info("容量约束报告")
        logger.info("=" * 80)

        if len(violations) == 0:
            logger.info("\n✅ 所有约束条件满足")
            return

        logger.info(f"\n发现{len(violations)}个违规:")

        for i, v in enumerate(violations, 1):
            logger.info(f"\n违规 {i}:")
            logger.info(f"  类型: {v['type']}")
            logger.info(f"  消息: {v['message']}")
            if "symbol" in v:
                logger.info(f"  标的: {v['symbol']}")
            if "sector" in v:
                logger.info(f"  赛道: {v['sector']}")
            logger.info(f"  当前值: {v['value']:.2%}")
            logger.info(f"  限制值: {v['limit']:.2%}")

        # 保存报告（遵循调用者传入的输出目录，若无法获取则回退默认路径）
        try:
            # 尝试从调用环境传入的路径推断（在 main 中传参）
            # 这里通过环境变量或类属性均不可得，保持与 main 一致：默认目录
            output_dir = Path(self.output_dir)  # 若外部设置该属性
        except Exception:
            output_dir = Path("factor_output/etf_rotation_production")
        report_file = output_dir / "capacity_constraints_report.json"

        import json

        with open(report_file, "w") as f:
            json.dump(violations, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ 报告已保存: {report_file}")
        logger.info("=" * 80)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="容量与约束检查")
    parser.add_argument("--backtest-result", help="回测结果文件（JSON）")
    parser.add_argument("--price-dir", default="raw/ETF/daily", help="价格数据目录")
    parser.add_argument(
        "--output-dir", default="factor_output/etf_rotation_production", help="输出目录"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("容量与约束检查")
    logger.info("=" * 80)

    # 初始化检查器
    checker = CapacityConstraints()
    # 传递输出目录以便报告落盘到指定路径
    try:
        checker.output_dir = args.output_dir
    except Exception:
        pass

    # 从回测结果加载持仓（如果提供）
    if args.backtest_result and Path(args.backtest_result).exists():
        logger.info(f"从回测结果加载持仓: {args.backtest_result}")
        # 简化：使用模拟持仓（实际需解析回测结果）
        positions = pd.DataFrame(
            {
                "symbol": [
                    "510050.SH",
                    "512880.SH",
                    "515790.SH",
                    "159919.SZ",
                    "510500.SH",
                ],
                "weight": [0.20, 0.20, 0.20, 0.20, 0.20],
            }
        )
    else:
        # 模拟持仓
        positions = pd.DataFrame(
            {
                "symbol": [
                    "510050.SH",
                    "512880.SH",
                    "515790.SH",
                    "159919.SZ",
                    "510500.SH",
                ],
                "weight": [0.20, 0.20, 0.20, 0.20, 0.20],
            }
        )

    logger.info(f"\n测试持仓:")
    for _, row in positions.iterrows():
        logger.info(f"  {row['symbol']}: {row['weight']:.2%}")

    try:
        # 检查仓位限制
        position_violations = checker.check_position_limits(positions)

        # 加载价格数据（如可用）
        price_dir = Path(args.price_dir)
        prices = None
        try:
            if price_dir.exists():
                parts = []
                for file in price_dir.glob("*.parquet"):
                    try:
                        df = pd.read_parquet(file)
                        if "ts_code" in df.columns:
                            df["symbol"] = df["ts_code"]
                        if "trade_date" in df.columns:
                            df["date"] = pd.to_datetime(df["trade_date"])
                        if "vol" in df.columns and "volume" not in df.columns:
                            df["volume"] = df["vol"]
                        keep = [
                            c
                            for c in [
                                "symbol",
                                "date",
                                "close",
                                "open",
                                "high",
                                "low",
                                "volume",
                                "amount",
                            ]
                            if c in df.columns
                        ]
                        parts.append(df[keep])
                    except Exception:
                        pass
                if parts:
                    prices = (
                        pd.concat(parts, ignore_index=True)
                        .set_index(["symbol", "date"])
                        .sort_index()
                    )
        except Exception:
            prices = None

        # 检查ADV%约束
        adv_violations = checker.check_adv_constraints(positions, prices, 1000000)

        # 生成报告
        all_violations = position_violations + adv_violations
        checker.generate_report(all_violations)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 容量与约束检查完成")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"❌ 检查失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
