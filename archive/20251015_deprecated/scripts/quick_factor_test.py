#!/usr/bin/env python3
"""快速因子测试 - 5分钟定位问题"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """测试数据加载"""
    logger.info("=" * 60)
    logger.info("测试1: 数据加载")
    logger.info("=" * 60)

    data_dir = Path("raw/ETF/daily")
    files = list(data_dir.glob("*.parquet"))

    if not files:
        logger.error("❌ 未找到数据文件")
        return False

    logger.info(f"找到 {len(files)} 个文件")

    # 测试第一个文件
    file = files[0]
    logger.info(f"\n测试文件: {file.name}")

    df = pd.read_parquet(file)
    logger.info(f"形状: {df.shape}")
    logger.info(f"列名: {df.columns.tolist()}")

    # 检查必需字段
    required = ["open", "high", "low", "volume"]
    price_fields = ["close", "adj_close"]
    date_fields = ["date", "trade_date"]

    logger.info("\n字段检查:")
    for field in required:
        if field in df.columns:
            logger.info(f"  ✅ {field}")
        else:
            logger.error(f"  ❌ {field} 缺失")

    has_price = any(f in df.columns for f in price_fields)
    logger.info(
        f"  {'✅' if has_price else '❌'} 价格字段: {[f for f in price_fields if f in df.columns]}"
    )

    has_date = any(f in df.columns for f in date_fields)
    logger.info(
        f"  {'✅' if has_date else '❌'} 日期字段: {[f for f in date_fields if f in df.columns]}"
    )

    logger.info(f"\n前3行:\n{df.head(3)}")

    return has_price and has_date


def test_factor_registry():
    """测试因子注册表"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 因子注册表")
    logger.info("=" * 60)

    try:
        from factor_system.factor_engine.core.registry import FactorRegistry

        registry = FactorRegistry()
        all_factors = registry.list_factors()

        logger.info(f"注册因子数: {len(all_factors)}")
        logger.info(f"前10个: {all_factors[:10]}")

        return True
    except Exception as e:
        logger.error(f"❌ 注册表加载失败: {e}")
        return False


def test_single_factor():
    """测试单个因子计算"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 单因子计算")
    logger.info("=" * 60)

    try:
        from factor_system.factor_engine.core.registry import FactorRegistry

        # 加载数据
        data_dir = Path("raw/ETF/daily")
        file = list(data_dir.glob("*.parquet"))[0]
        df = pd.read_parquet(file)

        # 标准化列名
        if "trade_date" in df.columns:
            df["date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        # 统一列名
        if "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]

        # 确定价格字段并统一为close
        if "adj_close" in df.columns:
            price_field = "adj_close"
            df["close"] = df["adj_close"]
        elif "close" in df.columns:
            price_field = "close"
        else:
            logger.error("❌ 无可用价格字段")
            return False

        # 准备输入
        df = df.sort_values("date").set_index("date")
        input_data = df[["open", "high", "low", "close", "volume"]].copy()

        logger.info(f"输入数据形状: {input_data.shape}")
        logger.info(f"日期范围: {input_data.index.min()} ~ {input_data.index.max()}")

        # 测试简单因子
        registry = FactorRegistry()

        test_factors = ["TA_SMA_20", "TA_EMA_20", "TA_RSI_14"]
        results = {}

        for factor_id in test_factors:
            try:
                logger.info(f"\n测试因子: {factor_id}")
                factor_class = registry.get_factor(factor_id)
                factor = factor_class()

                result = factor.calculate(input_data)

                coverage = result.notna().mean()
                logger.info(f"  覆盖率: {coverage:.2%}")
                logger.info(f"  均值: {result.mean():.6f}")
                logger.info(f"  标准差: {result.std():.6f}")
                logger.info(f"  前5个值: {result.head().tolist()}")

                results[factor_id] = {"success": True, "coverage": coverage}

            except Exception as e:
                logger.error(f"  ❌ 失败: {e}")
                results[factor_id] = {"success": False, "error": str(e)}

        # 汇总
        success_count = sum(1 for r in results.values() if r["success"])
        logger.info(f"\n成功: {success_count}/{len(test_factors)}")

        return success_count > 0

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


def main():
    logger.info("=" * 60)
    logger.info("快速因子测试（5分钟定位）")
    logger.info("=" * 60)

    results = []

    # 测试1: 数据加载
    results.append(("数据加载", test_data_loading()))

    # 测试2: 因子注册表
    results.append(("因子注册表", test_factor_registry()))

    # 测试3: 单因子计算
    results.append(("单因子计算", test_single_factor()))

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("测试汇总")
    logger.info("=" * 60)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{name}: {status}")

    all_passed = all(r for _, r in results)

    if all_passed:
        logger.info("\n🎉 所有测试通过！可以运行全量计算")
        logger.info("\n下一步:")
        logger.info(
            "  python scripts/produce_full_etf_panel.py --start-date 20240101 --end-date 20241231"
        )
    else:
        logger.error("\n❌ 部分测试失败，请先修复")
        logger.info("\n诊断建议:")
        logger.info("  1. 检查数据文件列名是否正确")
        logger.info("  2. 确认因子注册表是否加载")
        logger.info("  3. 查看单因子计算的具体错误")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
