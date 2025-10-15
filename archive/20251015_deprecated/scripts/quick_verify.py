#!/usr/bin/env python3
"""快速核验清单 - 10分钟内完成数据字段验证

核心检查：
1. 字段存在性：open/close/volume/amount/trade_date
2. amount单位统一为"元"
3. 随机抽2只ETF验证
4. 生成核验报告

Linus式原则：快速、准确、可追溯
"""

import logging
import random
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verify_etf_data(data_dir="raw/ETF/daily", sample_size=2):
    """快速核验ETF数据"""
    logger.info("=" * 80)
    logger.info("快速核验清单 - ETF数据字段验证")
    logger.info("=" * 80)

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        return False

    # 获取所有ETF文件
    etf_files = list(data_path.glob("*.parquet"))
    if len(etf_files) == 0:
        logger.error(f"❌ 未找到任何parquet文件: {data_dir}")
        return False

    logger.info(f"\n✅ 找到{len(etf_files)}个ETF数据文件")

    # 随机抽样
    sample_files = random.sample(etf_files, min(sample_size, len(etf_files)))

    logger.info(f"\n随机抽样{len(sample_files)}只ETF进行验证:")
    for f in sample_files:
        logger.info(f"  - {f.name}")

    # 必需字段（vol是tushare的字段名，volume是标准名）
    required_fields = ["trade_date", "open", "close"]
    volume_fields = ["vol", "volume"]  # 两者之一即可
    optional_fields = ["amount", "high", "low"]

    # 验证结果
    results = {
        "total_files": len(etf_files),
        "sampled_files": len(sample_files),
        "passed": [],
        "failed": [],
        "missing_fields": {},
        "amount_status": {},
    }

    logger.info("\n" + "=" * 80)
    logger.info("字段验证")
    logger.info("=" * 80)

    for etf_file in sample_files:
        try:
            # 读取数据
            df = pd.read_parquet(etf_file)

            logger.info(f"\n{etf_file.name}:")
            logger.info(f"  行数: {len(df)}")
            logger.info(f"  列名: {list(df.columns)}")

            # 检查必需字段
            missing_required = [f for f in required_fields if f not in df.columns]

            # 检查volume字段（vol或volume之一即可）
            has_volume = any(f in df.columns for f in volume_fields)
            if not has_volume:
                missing_required.append("vol/volume")

            missing_optional = [f for f in optional_fields if f not in df.columns]

            if missing_required:
                logger.error(f"  ❌ 缺少必需字段: {missing_required}")
                results["failed"].append(etf_file.name)
                results["missing_fields"][etf_file.name] = missing_required
                continue

            if missing_optional:
                logger.warning(f"  ⚠️  缺少可选字段: {missing_optional}")

            # 检查amount字段
            if "amount" in df.columns:
                amount_sample = df["amount"].head(3)
                logger.info(f"  ✅ amount字段存在")
                logger.info(f"     示例值: {amount_sample.tolist()}")

                # 判断单位（假设>1000为元，<1000为万元）
                avg_amount = df["amount"].mean()
                if avg_amount > 1000:
                    unit = "元"
                else:
                    unit = "万元（需要转换）"

                logger.info(f"     平均值: {avg_amount:.2f}")
                logger.info(f"     推测单位: {unit}")
                results["amount_status"][etf_file.name] = unit
            else:
                logger.warning(f"  ⚠️  amount字段缺失，可用close*volume估算")
                results["amount_status"][etf_file.name] = "缺失（可估算）"

            # 检查数据质量
            volume_col = "vol" if "vol" in df.columns else "volume"
            logger.info(f"  数据范围:")
            logger.info(
                f"    日期: {df['trade_date'].min()} ~ {df['trade_date'].max()}"
            )
            logger.info(f"    价格: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
            logger.info(
                f"    成交量({volume_col}): {df[volume_col].min():.0f} ~ {df[volume_col].max():.0f}"
            )

            results["passed"].append(etf_file.name)

        except Exception as e:
            logger.error(f"  ❌ 读取失败: {e}")
            results["failed"].append(etf_file.name)

    # 生成总结报告
    logger.info("\n" + "=" * 80)
    logger.info("核验总结")
    logger.info("=" * 80)

    logger.info(f"\n总文件数: {results['total_files']}")
    logger.info(f"抽样数: {results['sampled_files']}")
    logger.info(f"通过: {len(results['passed'])}")
    logger.info(f"失败: {len(results['failed'])}")

    if results["passed"]:
        logger.info(f"\n✅ 通过验证的文件:")
        for f in results["passed"]:
            logger.info(f"  - {f}")

    if results["failed"]:
        logger.warning(f"\n❌ 未通过验证的文件:")
        for f in results["failed"]:
            logger.warning(f"  - {f}")
            if f in results["missing_fields"]:
                logger.warning(f"    缺少字段: {results['missing_fields'][f]}")

    # amount字段状态
    logger.info(f"\namount字段状态:")
    for f, status in results["amount_status"].items():
        logger.info(f"  {f}: {status}")

    # 建议
    logger.info("\n" + "=" * 80)
    logger.info("建议")
    logger.info("=" * 80)

    has_amount_issue = any(
        "万元" in s or "缺失" in s for s in results["amount_status"].values()
    )

    if has_amount_issue:
        logger.warning("\n⚠️  amount字段需要处理:")
        logger.warning("  1. 如果单位是万元，需要乘以10000转换为元")
        logger.warning("  2. 如果缺失，可用close*volume估算")
        logger.warning("  3. 在meta中注明估算方法")
    else:
        logger.info("\n✅ amount字段单位统一为元，无需转换")

    # 保存报告
    report_file = Path("factor_output/etf_rotation_production/quick_verify_report.txt")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("快速核验报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总文件数: {results['total_files']}\n")
        f.write(f"抽样数: {results['sampled_files']}\n")
        f.write(f"通过: {len(results['passed'])}\n")
        f.write(f"失败: {len(results['failed'])}\n\n")

        f.write("amount字段状态:\n")
        for file, status in results["amount_status"].items():
            f.write(f"  {file}: {status}\n")

    logger.info(f"\n✅ 核验报告已保存: {report_file}")

    logger.info("\n" + "=" * 80)
    if len(results["failed"]) == 0:
        logger.info("✅ 快速核验通过！")
    else:
        logger.warning("⚠️  部分文件未通过验证")
    logger.info("=" * 80)

    return len(results["failed"]) == 0


def main():
    """主函数"""
    try:
        success = verify_etf_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ 核验失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
