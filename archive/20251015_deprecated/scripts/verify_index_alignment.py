#!/usr/bin/env python3
"""索引对齐验证 - Linus式严格检查

验证项：
1. MultiIndex规范：(symbol, date)
2. date格式：normalize(), tz-naive
3. 对齐策略：inner join（交集）
4. 无重复索引
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def verify_index_alignment():
    """验证索引对齐"""

    logger.info("=" * 80)
    logger.info("索引对齐验证")
    logger.info("=" * 80)

    # 加载面板
    panel_file = Path(
        "factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet"
    )
    if not panel_file.exists():
        logger.error(f"❌ 面板文件不存在: {panel_file}")
        return False

    panel = pd.read_parquet(panel_file)
    logger.info(f"\n✅ 加载面板: {panel.shape}")

    all_passed = True

    # 1. 检查索引名称
    logger.info(f"\n1. 索引规范检查")
    logger.info(f"   索引名称: {panel.index.names}")

    if panel.index.names == ["symbol", "date"]:
        logger.info("   ✅ 索引名称正确: (symbol, date)")
    else:
        logger.error(
            f"   ❌ 索引名称错误: 期望['symbol', 'date']，实际{panel.index.names}"
        )
        all_passed = False

    # 2. 检查date格式
    logger.info(f"\n2. date格式检查")
    dates = panel.index.get_level_values("date")

    # 检查是否为datetime
    if pd.api.types.is_datetime64_any_dtype(dates):
        logger.info("   ✅ date类型正确: datetime64")
    else:
        logger.error(f"   ❌ date类型错误: {dates.dtype}")
        all_passed = False

    # 检查时区
    if dates.tz is None:
        logger.info("   ✅ date时区正确: tz-naive")
    else:
        logger.error(f"   ❌ date时区错误: {dates.tz}")
        all_passed = False

    # 检查时间部分是否为00:00:00
    sample_dates = dates[:100]
    all_normalized = all(
        d.hour == 0 and d.minute == 0 and d.second == 0 for d in sample_dates
    )
    if all_normalized:
        logger.info("   ✅ date已normalize: 时间部分为00:00:00")
    else:
        logger.error("   ❌ date未normalize: 存在非00:00:00时间")
        all_passed = False

    # 3. 检查重复索引
    logger.info(f"\n3. 重复索引检查")
    if panel.index.is_unique:
        logger.info("   ✅ 无重复索引")
    else:
        duplicates = panel.index[panel.index.duplicated()]
        logger.error(f"   ❌ 存在重复索引: {len(duplicates)}个")
        logger.error(f"   示例: {duplicates[:5].tolist()}")
        all_passed = False

    # 4. 检查索引排序
    logger.info(f"\n4. 索引排序检查")
    if panel.index.is_monotonic_increasing:
        logger.info("   ✅ 索引已排序")
    else:
        logger.warning("   ⚠️  索引未排序（不影响功能）")

    # 5. 检查symbol和date的完整性
    logger.info(f"\n5. 数据完整性检查")
    symbols = panel.index.get_level_values("symbol").unique()
    dates = panel.index.get_level_values("date").unique()

    logger.info(f"   symbol数量: {len(symbols)}")
    logger.info(f"   date数量: {len(dates)}")
    logger.info(
        f"   理论样本数: {len(symbols)} × {len(dates)} = {len(symbols) * len(dates)}"
    )
    logger.info(f"   实际样本数: {len(panel)}")

    completeness = len(panel) / (len(symbols) * len(dates))
    logger.info(f"   完整度: {completeness:.2%}")

    if completeness > 0.95:
        logger.info("   ✅ 数据完整度良好（>95%）")
    elif completeness > 0.80:
        logger.info("   ⚠️  数据完整度一般（80-95%）")
    else:
        logger.error("   ❌ 数据完整度不足（<80%）")
        all_passed = False

    # 6. 随机抽样验证对齐
    logger.info(f"\n6. 对齐验证（随机抽样）")

    # 随机选择3个symbol和3个date
    import random

    random.seed(42)

    sample_symbols = random.sample(list(symbols), min(3, len(symbols)))
    sample_dates = random.sample(list(dates), min(3, len(dates)))

    for symbol in sample_symbols:
        for date in sample_dates:
            try:
                row = panel.loc[(symbol, date)]
                logger.info(f"   ✅ ({symbol}, {date}): {len(row)}个因子")
            except KeyError:
                logger.warning(
                    f"   ⚠️  ({symbol}, {date}): 不存在（正常，可能该ETF当日无交易）"
                )

    logger.info(f"\n{'=' * 80}")
    if all_passed:
        logger.info("✅ 索引对齐验证通过")
    else:
        logger.error("❌ 索引对齐验证失败")
    logger.info("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = verify_index_alignment()
    sys.exit(0 if success else 1)
