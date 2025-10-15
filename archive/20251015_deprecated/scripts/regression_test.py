#!/usr/bin/env python3
"""回归测试 - 验证T+1安全性与序列连续性

验证项：
1. 首个非NaN位置 ≥ window+1（含T+1）
2. 序列连续无前向填充
3. 无异常跳变
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def regression_test():
    """回归测试"""

    logger.info("=" * 80)
    logger.info("回归测试 - T+1安全性与序列连续性")
    logger.info("=" * 80)

    # 加载生产级面板
    panel_file = Path(
        "factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet"
    )
    if not panel_file.exists():
        logger.error(f"❌ 面板文件不存在: {panel_file}")
        return False

    panel = pd.read_parquet(panel_file)
    logger.info(f"\n✅ 加载面板: {panel.shape}")

    # 测试因子配置
    test_cases = [
        {
            "factor_id": "VBT_MACD_SIGNAL_12_26_9",
            "window": 26 + 9,  # slow + signal
            "min_history": 36,
            "type": "momentum",
        },
        {"factor_id": "TA_RSI_14", "window": 14, "min_history": 15, "type": "momentum"},
        {
            "factor_id": "TA_BB_UPPER_20_2.0",
            "window": 20,
            "min_history": 21,
            "type": "volatility",
        },
    ]

    # 测试ETF
    all_symbols = panel.index.get_level_values("symbol").unique()
    test_symbols = list(all_symbols[:3])

    logger.info(f"\n测试配置:")
    logger.info(f"  因子数: {len(test_cases)}")
    logger.info(f"  ETF数: {len(test_symbols)}")
    logger.info(f"  ETF列表: {test_symbols}")

    all_passed = True

    for test_case in test_cases:
        factor_id = test_case["factor_id"]
        min_history = test_case["min_history"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"测试因子: {factor_id}")
        logger.info(f"  类型: {test_case['type']}")
        logger.info(f"  窗口: {test_case['window']}")
        logger.info(f"  min_history: {min_history}")
        logger.info(f"{'=' * 60}")

        if factor_id not in panel.columns:
            logger.error(f"  ❌ 因子不存在")
            all_passed = False
            continue

        for symbol in test_symbols:
            logger.info(f"\n  ETF: {symbol}")

            try:
                # 提取序列
                series = panel.xs(symbol, level="symbol")[factor_id]

                # 1. 检查首个非NaN位置
                first_valid_idx = series.first_valid_index()
                if first_valid_idx is None:
                    logger.error(f"    ❌ 全部为NaN")
                    all_passed = False
                    continue

                first_valid_pos = series.index.get_loc(first_valid_idx) + 1

                if first_valid_pos >= min_history:
                    logger.info(
                        f"    ✅ 首个非NaN: 第{first_valid_pos}行（≥{min_history}）"
                    )
                else:
                    logger.error(
                        f"    ❌ 首个非NaN: 第{first_valid_pos}行（<{min_history}）"
                    )
                    all_passed = False

                # 2. 检查前min_history行全为NaN
                if series.iloc[:min_history].isna().all():
                    logger.info(f"    ✅ 前{min_history}行全为NaN")
                else:
                    non_nan_count = (~series.iloc[:min_history].isna()).sum()
                    logger.error(
                        f"    ❌ 前{min_history}行存在{non_nan_count}个非NaN值"
                    )
                    all_passed = False

                # 3. 检查序列连续性（无前向填充）
                valid_series = series.dropna()
                if len(valid_series) > 1:
                    # 检查是否有连续相同值（可能是前向填充）
                    consecutive_same = (valid_series == valid_series.shift(1)).sum()
                    total_valid = len(valid_series)
                    same_ratio = consecutive_same / total_valid

                    if same_ratio < 0.5:  # 允许50%以下的连续相同
                        logger.info(
                            f"    ✅ 序列连续性正常（相同比例{same_ratio:.1%}）"
                        )
                    else:
                        logger.warning(
                            f"    ⚠️  序列可能存在前向填充（相同比例{same_ratio:.1%}）"
                        )

                # 4. 检查异常跳变
                if len(valid_series) > 1:
                    pct_change = valid_series.pct_change().abs()
                    extreme_changes = (pct_change > 10).sum()  # 超过1000%的变化

                    if extreme_changes == 0:
                        logger.info(f"    ✅ 无异常跳变")
                    else:
                        logger.warning(
                            f"    ⚠️  存在{extreme_changes}个异常跳变（>1000%）"
                        )

                # 5. 显示前30行样例
                logger.info(f"\n    前30行样例:")
                for i in range(min(30, len(series))):
                    val = series.iloc[i]
                    if pd.isna(val):
                        logger.info(f"      第{i+1}行: NaN")
                    else:
                        logger.info(f"      第{i+1}行: {val:.6f}")
                        if i >= min_history + 5:  # 显示到min_history后5行
                            break

            except Exception as e:
                logger.error(f"    ❌ 测试失败: {e}")
                all_passed = False

    logger.info(f"\n{'=' * 80}")
    if all_passed:
        logger.info("✅ 回归测试通过")
    else:
        logger.error("❌ 回归测试失败")
    logger.info("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = regression_test()
    sys.exit(0 if success else 1)
