#!/usr/bin/env python3
"""
系统测试脚本 - 验证各模块功能
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_data_loading():
    """测试数据加载"""
    try:
        import yaml
        from data_manager import DataManager

        # 加载配置
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 初始化数据管理器
        dm = DataManager(config)

        # 测试价格数据加载
        logger.info("测试数据加载...")
        prices = dm.load_prices(use_cache=False)

        if isinstance(prices, dict):
            logger.info("✅ 数据加载成功")
            logger.info(f"  - 数据类型: {list(prices.keys())}")
            logger.info(f"  - 标的数量: {prices['close'].shape[1]}")
            logger.info(
                f"  - 日期范围: {prices['close'].index[0]} ~ {prices['close'].index[-1]}"
            )
            logger.info(f"  - 数据点数: {len(prices['close'])}")

            # 测试数据验证
            is_valid = dm.validate_data(prices["close"])
            logger.info(f"  - 数据验证: {'✅ 通过' if is_valid else '❌ 失败'}")

            # 测试收益率计算
            returns = dm.calculate_returns(prices["close"])
            logger.info(f"  - 收益率计算: {list(returns.keys())} 周期")

            # 测试标的池
            universe = dm.get_universe()
            logger.info(f"  - 有效标的池: {len(universe)}个")

            return True
        else:
            logger.error("❌ 数据格式错误")
            return False

    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_calculation():
    """测试因子计算"""
    try:
        import yaml
        from data_manager import DataManager
        from factor_calculator import FactorCalculator

        logger.info("\n测试因子计算...")

        # 加载配置
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 初始化
        dm = DataManager(config)
        fc = FactorCalculator(config)

        # 加载数据
        prices = dm.load_prices()

        # 计算因子
        panel = fc.calculate_all(prices)

        logger.info("✅ 因子计算成功")
        logger.info(f"  - 因子数量: {len(panel.columns)}")
        logger.info(f"  - 面板形状: {panel.shape}")
        logger.info(f"  - 因子列表: {list(panel.columns)}")

        return True

    except Exception as e:
        logger.error(f"❌ 因子计算失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_screening():
    """测试因子筛选"""
    try:
        import yaml
        from data_manager import DataManager
        from factor_calculator import FactorCalculator
        from factor_screener import FactorScreener

        logger.info("\n测试因子筛选...")

        # 加载配置
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 初始化
        dm = DataManager(config)
        fc = FactorCalculator(config)
        fs = FactorScreener(config)

        # 加载数据
        prices = dm.load_prices()
        returns = dm.calculate_returns(prices["close"])

        # 计算因子
        panel = fc.calculate_all(prices)

        # 筛选因子
        selected_factors, ic_stats = fs.screen(panel, returns[5])

        logger.info("✅ 因子筛选成功")
        logger.info(f"  - 通过筛选: {len(selected_factors)}/{len(panel.columns)}")
        logger.info(f"  - 选中因子: {selected_factors}")

        if not ic_stats.empty:
            logger.info(f"  - 最高IC: {ic_stats['ic_mean'].abs().max():.3f}")
            logger.info(f"  - 最高IR: {ic_stats['ic_ir'].abs().max():.2f}")

        return True

    except Exception as e:
        logger.error(f"❌ 因子筛选失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_signal_generation():
    """测试信号生成"""
    try:
        import yaml
        from data_manager import DataManager
        from factor_calculator import FactorCalculator
        from signal_generator import SignalGenerator

        logger.info("\n测试信号生成...")

        # 加载配置
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 初始化
        dm = DataManager(config)
        fc = FactorCalculator(config)
        sg = SignalGenerator(config)

        # 加载数据
        prices = dm.load_prices()

        # 计算因子
        panel = fc.calculate_all(prices)

        # 使用所有因子生成信号
        signals = sg.generate(panel, panel.columns.tolist())

        logger.info("✅ 信号生成成功")
        logger.info(f"  - 信号数量: {len(signals)}")

        for signal in signals[:5]:  # 显示前5个信号
            logger.info(
                f"  - {signal['action']}: {signal['symbol']} "
                f"({signal.get('current_weight', 0):.1%} -> {signal['target_weight']:.1%})"
            )

        return True

    except Exception as e:
        logger.error(f"❌ 信号生成失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("ETF轮动系统优化版 - 功能测试")
    logger.info("=" * 60)

    tests = [
        ("数据加载", test_data_loading),
        ("因子计算", test_factor_calculation),
        ("因子筛选", test_factor_screening),
        ("信号生成", test_signal_generation),
    ]

    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))

    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总:")
    logger.info("=" * 60)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        logger.info("\n🎉 所有测试通过!")
    else:
        logger.info("\n⚠️  部分测试失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
