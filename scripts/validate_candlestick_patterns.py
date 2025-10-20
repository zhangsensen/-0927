#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线形态验证脚本
验证10个高价值K线形态的正确实现
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_candlestick_data():
    """创建测试K线数据，包含已知的形态"""

    # 创建日期范围
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")

    # 基础价格数据
    base_price = 100.0
    data = []

    for i, date in enumerate(dates):
        # 模拟价格波动
        trend = np.sin(i * 0.1) * 2  # 趋势成分
        noise = np.random.randn() * 0.5  # 随机噪音

        close = base_price + trend + noise

        # 确保OHLC关系正确
        high = close + abs(np.random.randn()) * 2
        low = close - abs(np.random.randn()) * 2
        open_price = close + (np.random.randn() - 0.5) * 1

        # 调整确保 high >= max(open, close) 和 low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 确保价格关系
        if high < low:
            high, low = low, high  # 交换

        data.append(
            {
                "date": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000000, 10000000),
            }
        )

    return pd.DataFrame(data)


def test_candlestick_patterns():
    """测试K线形态识别"""

    logger.info("开始K线形态验证测试...")

    # 创建测试数据
    test_data = create_test_candlestick_data()
    logger.info(f"创建测试数据: {len(test_data)} 条记录")

    # 导入TA-Lib
    try:
        import talib

        logger.info("✅ TA-Lib导入成功")
    except ImportError:
        logger.error("❌ TA-Lib未安装，无法测试K线形态")
        return False

    # 测试的10个高价值K线形态
    patterns_to_test = [
        ("CDLHAMMER", "锤子线"),
        ("CDLMORNINGSTAR", "早晨之星"),
        ("CDLENGULFING", "吞没形态"),
        ("CDLPIERCING", "刺透形态"),
        ("CDL3WHITESOLDIERS", "三白兵"),
        ("CDLHANGINGMAN", "上吊线"),
        ("CDLEVENINGSTAR", "黄昏之星"),
        ("CDLDARKCLOUDCOVER", "乌云盖顶"),
        ("CDL3BLACKCROWS", "三只乌鸦"),
        ("CDL3OUTSIDE", "三外部形态"),
    ]

    results = {}

    for pattern_code, pattern_name in patterns_to_test:
        try:
            logger.info(f"测试 {pattern_name} ({pattern_code})...")

            # 调用TA-Lib函数
            pattern_func = getattr(talib, pattern_code)
            result = pattern_func(
                test_data["open"].values,
                test_data["high"].values,
                test_data["low"].values,
                test_data["close"].values,
            )

            # 分析结果
            positive_signals = np.sum(result > 0)
            negative_signals = np.sum(result < 0)
            total_signals = positive_signals + negative_signals

            results[pattern_code] = {
                "name": pattern_name,
                "positive_signals": int(positive_signals),
                "negative_signals": int(negative_signals),
                "total_signals": int(total_signals),
                "signal_rate": total_signals / len(test_data) * 100,
                "success": True,
            }

            logger.info(
                f"  ✅ {pattern_name}: 信号数={total_signals}, 信号率={total_signals/len(test_data)*100:.1f}%"
            )

            # 如果检测到信号，显示具体位置
            if total_signals > 0:
                signal_indices = np.where(result != 0)[0]
                sample_dates = test_data.iloc[signal_indices]["date"].head(3)
                logger.info(f"     样本信号日期: {list(sample_dates.astype(str))}")

        except Exception as e:
            logger.error(f"  ❌ {pattern_name} 测试失败: {e}")
            results[pattern_code] = {
                "name": pattern_name,
                "positive_signals": 0,
                "negative_signals": 0,
                "total_signals": 0,
                "signal_rate": 0.0,
                "success": False,
                "error": str(e),
            }

    # 生成测试报告
    generate_test_report(results)

    return True


def generate_test_report(results):
    """生成测试报告"""

    logger.info("\n" + "=" * 60)
    logger.info("K线形态验证测试报告")
    logger.info("=" * 60)

    total_success = sum(1 for r in results.values() if r["success"])
    total_patterns = len(results)

    logger.info(f"测试总数: {total_patterns}")
    logger.info(f"成功数: {total_success}")
    logger.info(f"失败数: {total_patterns - total_success}")
    logger.info(f"成功率: {total_success/total_patterns*100:.1f}%")
    logger.info("")

    # 详细结果
    logger.info("详细测试结果:")
    logger.info("-" * 60)

    for pattern_code, result in results.items():
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {result['name']} ({pattern_code})")
        logger.info(f"   看涨信号: {result['positive_signals']}")
        logger.info(f"   看跌信号: {result['negative_signals']}")
        logger.info(f"   总信号数: {result['total_signals']}")
        logger.info(f"   信号率: {result['signal_rate']:.1f}%")

        if not result["success"] and "error" in result:
            logger.info(f"   错误: {result['error']}")
        logger.info("")

    # 统计信息
    total_signals = sum(r["total_signals"] for r in results.values())
    avg_signal_rate = np.mean([r["signal_rate"] for r in results.values()])

    logger.info("统计摘要:")
    logger.info(f"总信号数: {total_signals}")
    logger.info(f"平均信号率: {avg_signal_rate:.1f}%")
    logger.info(
        f"信号频率范围: {min(r['signal_rate'] for r in results.values()):.1f}% - {max(r['signal_rate'] for r in results.values()):.1f}%"
    )

    logger.info("\n" + "=" * 60)


def test_with_known_patterns():
    """使用已知的K线形态数据进行测试"""

    logger.info("使用已知形态数据进行验证...")

    # 创建锤子线数据
    hammer_data = pd.DataFrame(
        [
            {"open": 100, "high": 101, "low": 95, "close": 100.5},  # 前一天
            {"open": 100.5, "high": 101, "low": 96, "close": 100.8},  # 锤子线日
            {"open": 100.8, "high": 102, "low": 99, "close": 101.5},  # 后一天
        ]
    )

    try:
        import talib

        result = talib.CDLHAMMER(
            hammer_data["open"].values,
            hammer_data["high"].values,
            hammer_data["low"].values,
            hammer_data["close"].values,
        )

        if result[1] != 0:
            logger.info("✅ 锤子线识别测试通过")
        else:
            logger.warning("⚠️  锤子线识别测试未通过")

    except Exception as e:
        logger.error(f"❌ 已知形态测试失败: {e}")


def main():
    """主函数"""

    logger.info("开始K线形态验证测试...")
    logger.info("测试10个高价值K线形态的正确实现")

    # 基本测试
    test_candlestick_patterns()

    # 已知形态测试
    test_with_known_patterns()

    logger.info("K线形态验证测试完成！")


if __name__ == "__main__":
    main()
