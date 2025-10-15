#!/usr/bin/env python3
"""ADV容量校验 - 实际执行版本

核心功能：
1. 加载raw/ETF/daily数据（amount字段）
2. 计算ADV20（20日平均成交额）
3. 校验单标成交额/ADV20 < 5%
4. 生成月度统计报告

Linus式原则：向量化计算，快速准确
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def calculate_adv20(
    data_dir="raw/ETF/daily", output_dir="factor_output/etf_rotation_production"
):
    """计算ADV20并校验容量约束"""
    logger.info("=" * 80)
    logger.info("ADV容量校验")
    logger.info("=" * 80)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载所有ETF数据
    etf_files = list(data_path.glob("*.parquet"))
    logger.info(f"\n✅ 找到{len(etf_files)}个ETF文件")

    all_adv = []

    for etf_file in etf_files:
        try:
            # 读取数据
            df = pd.read_parquet(etf_file)

            # 提取symbol
            symbol = etf_file.stem.split("_")[0]

            # 检查amount字段
            if "amount" not in df.columns:
                logger.warning(f"  ⚠️  {symbol}: amount字段缺失，跳过")
                continue

            # 转换日期
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.sort_values("date")

            # 计算ADV20（20日滚动平均成交额）
            df["adv20"] = df["amount"].rolling(window=20, min_periods=20).mean()

            # 保存结果
            df["symbol"] = symbol
            all_adv.append(df[["symbol", "date", "amount", "adv20"]].copy())

        except Exception as e:
            logger.error(f"  ❌ {etf_file.name}: {e}")

    if len(all_adv) == 0:
        logger.error("❌ 未能计算任何ADV数据")
        return False

    # 合并所有数据
    adv_df = pd.concat(all_adv, ignore_index=True)
    adv_df = adv_df.dropna(subset=["adv20"])

    logger.info(f"\n✅ 计算完成:")
    logger.info(f"  ETF数: {adv_df['symbol'].nunique()}")
    logger.info(f"  数据行数: {len(adv_df)}")
    logger.info(f"  日期范围: {adv_df['date'].min()} ~ {adv_df['date'].max()}")

    # 保存ADV数据
    adv_file = output_path / "adv20_data.parquet"
    adv_df.to_parquet(adv_file, index=False)
    logger.info(f"\n✅ ADV数据已保存: {adv_file}")

    # 生成统计报告
    logger.info("\n" + "=" * 80)
    logger.info("ADV统计")
    logger.info("=" * 80)

    # 按symbol统计
    adv_stats = (
        adv_df.groupby("symbol")
        .agg({"adv20": ["mean", "min", "max", "std"], "amount": ["mean", "min", "max"]})
        .round(2)
    )

    adv_stats.columns = ["_".join(col).strip() for col in adv_stats.columns.values]
    adv_stats = adv_stats.sort_values("adv20_mean", ascending=False)

    logger.info(f"\nTop 10 ETF（按平均ADV20排序）:")
    logger.info(adv_stats.head(10).to_string())

    # 保存统计报告
    stats_file = output_path / "adv20_statistics.csv"
    adv_stats.to_csv(stats_file)
    logger.info(f"\n✅ 统计报告已保存: {stats_file}")

    # ADV%容量校验（假设目标资金100万，5只ETF均分）
    logger.info("\n" + "=" * 80)
    logger.info("ADV%容量校验（假设100万资金，5只ETF均分）")
    logger.info("=" * 80)

    target_capital = 1000000  # 100万
    n_positions = 5
    position_size = target_capital / n_positions  # 每只20万

    logger.info(f"\n假设条件:")
    logger.info(f"  目标资金: {target_capital:,.0f}元")
    logger.info(f"  持仓数: {n_positions}")
    logger.info(f"  单只仓位: {position_size:,.0f}元")
    logger.info(f"  ADV%阈值: 5%")

    # 计算每只ETF的ADV%
    latest_adv = adv_df.sort_values("date").groupby("symbol").tail(1)
    latest_adv["adv_pct"] = (position_size / latest_adv["adv20"]) * 100
    latest_adv = latest_adv.sort_values("adv_pct", ascending=False)

    # 检查超限
    violations = latest_adv[latest_adv["adv_pct"] > 5.0]

    logger.info(f"\nADV%检查结果:")
    logger.info(f"  总ETF数: {len(latest_adv)}")
    logger.info(f"  超限数: {len(violations)}")
    logger.info(f"  合格数: {len(latest_adv) - len(violations)}")

    if len(violations) > 0:
        logger.warning(f"\n⚠️  超限ETF（ADV% > 5%）:")
        for _, row in violations.head(10).iterrows():
            logger.warning(
                f"  {row['symbol']}: ADV20={row['adv20']:,.0f}元, ADV%={row['adv_pct']:.2f}%"
            )
    else:
        logger.info(f"\n✅ 所有ETF均满足ADV%<5%约束")

    # 保存ADV%报告
    adv_pct_file = output_path / "adv_pct_check.csv"
    latest_adv[["symbol", "date", "adv20", "adv_pct"]].to_csv(adv_pct_file, index=False)
    logger.info(f"\n✅ ADV%报告已保存: {adv_pct_file}")

    # 月度统计（按月汇总ADV）
    logger.info("\n" + "=" * 80)
    logger.info("月度ADV统计")
    logger.info("=" * 80)

    adv_df["month"] = adv_df["date"].dt.to_period("M")
    monthly_adv = (
        adv_df.groupby(["symbol", "month"])
        .agg({"adv20": "mean", "amount": "mean"})
        .reset_index()
    )

    monthly_adv["month"] = monthly_adv["month"].astype(str)

    # 保存月度统计
    monthly_file = output_path / "monthly_adv_statistics.csv"
    monthly_adv.to_csv(monthly_file, index=False)
    logger.info(f"\n✅ 月度统计已保存: {monthly_file}")

    # 最近6个月统计
    recent_months = monthly_adv["month"].unique()[-6:]
    recent_monthly = monthly_adv[monthly_adv["month"].isin(recent_months)]

    logger.info(f"\n最近6个月ADV统计:")
    pivot = recent_monthly.pivot(index="symbol", columns="month", values="adv20")
    logger.info(pivot.head(10).to_string())

    logger.info("\n" + "=" * 80)
    logger.info("✅ ADV容量校验完成")
    logger.info("=" * 80)

    # 生成总结报告
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_etfs": len(latest_adv),
        "target_capital": target_capital,
        "position_size": position_size,
        "adv_threshold_pct": 5.0,
        "violations": len(violations),
        "qualified": len(latest_adv) - len(violations),
        "qualification_rate": (len(latest_adv) - len(violations))
        / len(latest_adv)
        * 100,
    }

    logger.info(f"\n总结:")
    logger.info(f"  目标资金: {summary['target_capital']:,.0f}元")
    logger.info(f"  单只仓位: {summary['position_size']:,.0f}元")
    logger.info(f"  合格率: {summary['qualification_rate']:.1f}%")
    logger.info(f"  合格ETF数: {summary['qualified']}/{summary['total_etfs']}")

    return True


def main():
    """主函数"""
    try:
        success = calculate_adv20()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ ADV校验失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
