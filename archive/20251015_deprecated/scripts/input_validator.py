#!/usr/bin/env python3
"""输入验证 - 数据schema断言

核心功能：
1. 校验列名：open/close/volume/amount/trade_date
2. 校验amount单位（元）
3. 校验日期格式
4. 失败即阻断

Linus式原则：快速失败，明确报错
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class InputValidator:
    """输入验证器"""

    def __init__(self):
        self.required_fields = ["trade_date", "open", "close"]
        self.volume_fields = ["vol", "volume"]  # 两者之一即可
        self.optional_fields = ["amount", "high", "low"]

    def validate_etf_data(self, data_dir="raw/ETF/daily", sample_size=5):
        """验证ETF数据"""
        logger.info("=" * 80)
        logger.info("输入验证 - ETF数据schema")
        logger.info("=" * 80)

        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"❌ 数据目录不存在: {data_dir}")
            return False

        etf_files = list(data_path.glob("*.parquet"))
        if len(etf_files) == 0:
            logger.error(f"❌ 未找到任何parquet文件: {data_dir}")
            return False

        logger.info(f"\n✅ 找到{len(etf_files)}个ETF文件")

        # 随机抽样验证
        import random

        sample_files = random.sample(etf_files, min(sample_size, len(etf_files)))

        logger.info(f"随机抽样{len(sample_files)}个文件进行验证\n")

        all_valid = True

        for etf_file in sample_files:
            logger.info(f"验证: {etf_file.name}")

            try:
                df = pd.read_parquet(etf_file)

                # 1. 检查必需字段
                missing_required = [
                    f for f in self.required_fields if f not in df.columns
                ]
                if missing_required:
                    logger.error(f"  ❌ 缺少必需字段: {missing_required}")
                    all_valid = False
                    continue

                # 2. 检查volume字段
                has_volume = any(f in df.columns for f in self.volume_fields)
                if not has_volume:
                    logger.error(f"  ❌ 缺少volume字段（vol或volume）")
                    all_valid = False
                    continue

                volume_col = "vol" if "vol" in df.columns else "volume"

                # 3. 检查amount字段
                if "amount" in df.columns:
                    # 检查单位（假设>1000为元，<1000为万元）
                    avg_amount = df["amount"].mean()
                    if avg_amount < 1000:
                        logger.warning(
                            f"  ⚠️  amount单位可能是万元（平均值{avg_amount:.2f}），需要转换"
                        )
                    else:
                        logger.info(f"  ✅ amount单位正常（平均值{avg_amount:.2f}元）")
                else:
                    logger.warning(f"  ⚠️  amount字段缺失，可用close*{volume_col}估算")

                # 4. 检查日期格式
                if df["trade_date"].dtype == "object":
                    # 尝试解析日期
                    try:
                        pd.to_datetime(df["trade_date"], format="%Y%m%d")
                        logger.info(f"  ✅ 日期格式正常（YYYYMMDD）")
                    except:
                        logger.error(f"  ❌ 日期格式错误")
                        all_valid = False
                        continue

                # 5. 检查数据完整性
                null_counts = df[self.required_fields + [volume_col]].isnull().sum()
                if null_counts.sum() > 0:
                    logger.warning(
                        f"  ⚠️  存在缺失值: {null_counts[null_counts > 0].to_dict()}"
                    )
                else:
                    logger.info(f"  ✅ 无缺失值")

                # 6. 检查数据范围
                if df["close"].min() <= 0:
                    logger.error(f"  ❌ 价格存在非正值")
                    all_valid = False
                    continue

                if df[volume_col].min() < 0:
                    logger.error(f"  ❌ 成交量存在负值")
                    all_valid = False
                    continue

                logger.info(f"  ✅ 数据范围正常")
                logger.info(f"  ✅ 验证通过\n")

            except Exception as e:
                logger.error(f"  ❌ 读取失败: {e}\n")
                all_valid = False

        logger.info("=" * 80)
        if all_valid:
            logger.info("✅ 所有抽样文件验证通过")
        else:
            logger.error("❌ 部分文件验证失败")
        logger.info("=" * 80)

        return all_valid

    def validate_panel(self, panel_file):
        """验证面板数据"""
        logger.info("=" * 80)
        logger.info("输入验证 - 面板数据")
        logger.info("=" * 80)

        panel_path = Path(panel_file)
        if not panel_path.exists():
            logger.error(f"❌ 面板文件不存在: {panel_file}")
            return False

        logger.info(f"\n加载面板: {panel_path.name}")

        try:
            panel = pd.read_parquet(panel_path)

            # 1. 检查索引
            if not isinstance(panel.index, pd.MultiIndex):
                logger.error("❌ 索引不是MultiIndex")
                return False

            if panel.index.names != ["symbol", "date"]:
                logger.error(f"❌ 索引名称错误: {panel.index.names}")
                return False

            logger.info("✅ 索引规范: MultiIndex(symbol, date)")

            # 2. 检查重复索引
            if panel.index.duplicated().any():
                dup_count = panel.index.duplicated().sum()
                logger.error(f"❌ 存在{dup_count}个重复索引")
                return False

            logger.info("✅ 无重复索引")

            # 3. 检查因子数
            if len(panel.columns) == 0:
                logger.error("❌ 面板无因子")
                return False

            logger.info(f"✅ 因子数: {len(panel.columns)}")

            # 4. 检查覆盖率
            coverage = panel.notna().sum().sum() / panel.size
            if coverage < 0.50:
                logger.error(f"❌ 覆盖率过低: {coverage:.2%}")
                return False

            logger.info(f"✅ 覆盖率: {coverage:.2%}")

            # 5. 检查数据类型
            non_numeric = []
            for col in panel.columns:
                if not pd.api.types.is_numeric_dtype(panel[col]):
                    non_numeric.append(col)

            if non_numeric:
                logger.warning(f"⚠️  存在非数值列: {non_numeric[:5]}")
            else:
                logger.info("✅ 所有列均为数值类型")

            logger.info("\n✅ 面板验证通过")
            return True

        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="输入验证")
    parser.add_argument("--data-dir", default="raw/ETF/daily", help="ETF数据目录")
    parser.add_argument("--panel-file", help="面板文件路径（可选）")
    parser.add_argument("--sample-size", type=int, default=5, help="抽样数量")

    args = parser.parse_args()

    validator = InputValidator()

    try:
        # 验证ETF数据
        etf_valid = validator.validate_etf_data(args.data_dir, args.sample_size)

        # 验证面板（如果提供）
        panel_valid = True
        if args.panel_file:
            panel_valid = validator.validate_panel(args.panel_file)

        # 判断是否通过
        if etf_valid and panel_valid:
            logger.info("\n✅ 所有验证通过")
            sys.exit(0)
        else:
            logger.error("\n❌ 验证失败")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
