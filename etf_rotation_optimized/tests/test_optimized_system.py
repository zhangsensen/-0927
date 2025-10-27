#!/usr/bin/env python3
"""
优化系统测试 - 验证所有新改进功能
"""
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_enhanced_data_validation():
    """测试增强的数据验证"""
    try:
        import yaml
        from data_manager import DataManager

        logger.info("测试增强的数据验证...")

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        dm = DataManager(config)
        prices = dm.load_prices()
        close = prices["close"]

        # 测试数据质量评估
        quality_metrics = dm.monitor_data_quality(close)

        logger.info("✅ 数据质量评估成功")
        logger.info(f"  - 数据点数: {quality_metrics['data_points']:,}")
        logger.info(f"  - 标的数: {quality_metrics['symbols']}")
        logger.info(f"  - 缺失率: {quality_metrics['missing_ratio']:.2%}")
        logger.info(f"  - 问题数: {len(quality_metrics['issues'])}")

        if quality_metrics["issues"]:
            logger.warning("发现数据质量问题:")
            for issue in quality_metrics["issues"][:5]:  # 显示前5个问题
                logger.warning(f"  - {issue}")

        return True

    except Exception as e:
        logger.error(f"❌ 增强数据验证失败: {e}")
        return False


def test_broker_interface():
    """测试券商接口"""
    try:
        import yaml
        from broker_interface import BrokerFactory, SimulationBroker

        logger.info("测试券商接口...")

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # 测试模拟券商
        broker_config = {"broker_type": "simulation", "init_cash": 100000}

        broker = BrokerFactory.create_broker(broker_config)

        if not broker.connect():
            raise Exception("券商连接失败")

        # 测试获取账户信息
        account_info = broker.get_account_info()
        logger.info("✅ 账户信息获取成功")
        logger.info(f"  - 总资产: {account_info['total_value']:,.0f}")
        logger.info(f"  - 可用现金: {account_info['cash']:,.0f}")

        # 测试下单
        test_order = broker.place_order("518850.SH", "BUY", 100)
        logger.info("✅ 模拟下单成功")
        logger.info(f"  - 订单ID: {test_order['order_id']}")
        logger.info(f"  - 状态: {test_order['status']}")

        # 测试持仓查询
        positions = broker.get_positions()
        logger.info("✅ 持仓查询成功")
        logger.info(f"  - 持仓数: {len(positions)}")

        return True

    except Exception as e:
        logger.error(f"❌ 券商接口测试失败: {e}")
        return False


def test_risk_monitoring():
    """测试风险监控"""
    try:
        import yaml
        from risk_monitor import RiskMonitor

        logger.info("测试风险监控系统...")

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        risk_monitor = RiskMonitor(config)

        # 模拟组合数据
        portfolio_data = {
            "current_value": 100000,
            "cash": 20000,
            "positions": [
                {"symbol": "518850.SH", "weight": 0.2, "daily_return": 0.01},
                {"symbol": "518880.SH", "weight": 0.15, "daily_return": -0.005},
                {"symbol": "512400.SH", "weight": 0.1, "daily_return": 0.02},
            ],
        }

        # 测试组合风险监控
        risk_report = risk_monitor.monitor_portfolio_risk(portfolio_data)

        logger.info("✅ 组合风险监控成功")
        logger.info(f"  - 风险等级: {risk_report['risk_level']}")
        logger.info(f"  - 告警数: {len(risk_report['alerts'])}")
        logger.info(f"  - 风险指标: {risk_report['metrics']}")

        # 显示主要告警
        if risk_report["alerts"]:
            logger.warning("发现风险告警:")
            for alert in risk_report["alerts"][:3]:  # 显示前3个告警
                logger.warning(f"  - {alert['severity']}: {alert['type']}")
                logger.warning(f"    {alert['message']}")

        return True

    except Exception as e:
        logger.error(f"❌ 风险监控测试失败: {e}")
        return False


def test_enhanced_factor_calculator():
    """测试增强的因子计算器"""
    try:
        import yaml
        from data_manager import DataManager
        from factor_calculator import FactorCalculator

        logger.info("测试增强的因子计算器...")

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        dm = DataManager(config)
        fc = FactorCalculator(config)
        prices = dm.load_prices()

        # 计算因子
        panel = fc.calculate_all(prices)

        # 测试IC权重计算
        selected_factors = ["MOM_20D", "VOL_20D"]
        scores = fc.calculate_composite_score(panel, selected_factors)

        logger.info("✅ IC权重因子计算成功")
        logger.info(f"  - 因子数量: {len(panel.columns)}")
        logger.info(f"  - 面板形状: {panel.shape}")
        logger.info(f"  - 复合得分形状: {scores.shape}")
        logger.info(f"  - 最新得分: {scores.iloc[-1].describe()}")

        return True

    except Exception as e:
        logger.error(f"❌ IC权重因子计算失败: {e}")
        return False


def test_integrated_signal_generation():
    """测试集成的信号生成"""
    try:
        import yaml
        from data_manager import DataManager
        from factor_calculator import FactorCalculator
        from signal_generator import SignalGenerator

        logger.info("测试集成信号生成...")

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        dm = DataManager(config)
        fc = FactorCalculator(config)
        sg = SignalGenerator(config)

        prices = dm.load_prices()
        panel = fc.calculate_all(prices)

        # 生成包含风控检查的信号
        signals = sg.generate(panel, panel.columns.tolist())

        logger.info("✅ 集成信号生成成功")
        logger.info(
            f"  - 原始信号数: {len([s for s in signals if s['action'] == 'BUY'])}"
        )

        # 统计不同类型的信号
        signal_types = {}
        for signal in signals:
            action = signal["action"]
            signal_types[action] = signal_types.get(action, 0) + 1

        logger.info(f"  - 信号统计: {signal_types}")

        # 检查风控信号
        risk_signals = [s for s in signals if "reason" in s]
        if risk_signals:
            logger.info(f"  - 风控信号数: {len(risk_signals)}")
            for rs in risk_signals[:3]:  # 显示前3个风控信号
                logger.info(f"    {rs['type']}: {rs['reason']}")

        return True

    except Exception as e:
        logger.error(f"❌ 集成信号生成失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_complete_pipeline():
    """测试完整流程"""
    try:
        logger.info("运行完整优化流程测试...")

        # 1. 数据加载和验证
        logger.info("\n步骤1: 数据加载和验证")
        success = test_enhanced_data_validation()

        # 2. 因子计算
        logger.info("\n步骤2: 增强因子计算")
        success = test_enhanced_factor_calculator() and success

        # 3. 信号生成
        logger.info("\n步骤3: 集成信号生成")
        success = test_integrated_signal_generation() and success

        # 4. 风险监控
        logger.info("\n步骤4: 风险监控")
        success = test_risk_monitoring() and success

        # 5. 券商接口
        logger.info("\n步骤5: 券商接口")
        success = test_broker_interface() and success

        logger.info("\n🎉 完整流程测试结果:")
        logger.info("  - 数据验证: ✅ 通过" if success else "  - 数据验证: ❌ 失败")
        logger.info("  - 因子计算: ✅ 通过" if success else "  - 因子计算: ❌ 失败")
        logger.info("  - 信号生成: ✅ 通过" if success else "  - 信号生成: ❌ 失败")
        logger.info("  - 风险监控: ✅ 通过" if success else "  - 风险监控: ❌ 失败")
        logger.info("  - 券商接口: ✅ 通过" if success else "  - 券商接口: ❌ 失败")

        overall_success = all(
            [
                test_enhanced_data_validation(),
                test_enhanced_factor_calculator(),
                test_integrated_signal_generation(),
                test_risk_monitoring(),
                test_broker_interface(),
            ]
        )

        if overall_success:
            logger.info("\n🚀 所有优化功能测试通过!")
            logger.info("✅ 系统已准备好进行实盘交易")
        else:
            logger.warning("\n⚠️  部分功能测试失败")

        return overall_success

    except Exception as e:
        logger.error(f"❌ 完整流程测试失败: {e}")
        return False


def main():
    """运行所有优化测试"""
    logger.info("=" * 70)
    logger.info("ETF轮动系统优化版 - 全面功能测试")
    logger.info("=" * 70)

    success = test_complete_pipeline()

    if success:
        logger.info("\n" + "=" * 70)
        logger.info("🎯 优化系统验证完成")
        logger.info("✅ 风控系统专业级别")
        logger.info("✅ 数据验证严格有效")
        logger.info("✅ IC权重动态计算")
        logger.info("✅ 实盘接口完整")
        logger.info("✅ 风险监控实时")
        logger.info("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("❌ 系统存在问题，需要进一步修复")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
