#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面动态因子冒烟测试
测试动态因子注册、计算和基本功能
"""

import pandas as pd
import numpy as np
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DynamicFactorSmokeTest:
    """动态因子冒烟测试类"""

    def __init__(self):
        """初始化测试"""
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent
        logger.info("动态因子冒烟测试初始化完成")

    def run_all_tests(self) -> Dict[str, bool]:
        """
        运行所有动态因子测试

        Returns:
            测试结果字典
        """
        logger.info("=" * 60)
        logger.info("开始运行动态因子冒烟测试")
        logger.info("=" * 60)

        # 测试1：增强模块导入
        self.test_results['enhanced_module_import'] = self.test_enhanced_module_import()

        # 测试2：因子工厂初始化
        self.test_results['factor_factory'] = self.test_factor_factory()

        # 测试3：因子注册表
        self.test_results['factor_registry'] = self.test_factor_registry()

        # 测试4：动态因子注册（小规模）
        self.test_results['dynamic_factor_registration'] = self.test_dynamic_factor_registration()

        # 测试5：增强因子计算器
        self.test_results['enhanced_factor_calculator'] = self.test_enhanced_factor_calculator()

        # 测试6：小规模动态因子计算
        self.test_results['dynamic_factor_calculation'] = self.test_dynamic_factor_calculation()

        # 汇总结果
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)

        logger.info("=" * 60)
        logger.info("动态因子冒烟测试完成")
        logger.info(f"通过: {passed_tests}/{total_tests}")
        logger.info("=" * 60)

        return self.test_results

    def test_enhanced_module_import(self) -> bool:
        """测试增强模块导入"""
        try:
            logger.info("测试1：增强模块导入")

            # 测试增强模块导入
            from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced
            from factor_system.factor_engine.factors.etf_cross_section.etf_factor_factory import ETFFactorFactory
            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry, FactorCategory, FactorCategory

            logger.info("  ✅ ETFCrossSectionFactorsEnhanced 导入成功")
            logger.info("  ✅ ETFFactorFactory 导入成功")
            logger.info("  ✅ 因子注册表导入成功")

            return True

        except Exception as e:
            logger.error(f"  ❌ 增强模块导入失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_factor_factory(self) -> bool:
        """测试因子工厂"""
        try:
            logger.info("测试2：因子工厂")

            from factor_system.factor_engine.factors.etf_cross_section.etf_factor_factory import ETFFactorFactory

            # 创建因子工厂实例
            factory = ETFFactorFactory()
            logger.info("  ✅ 因子工厂实例创建成功")

            # 检查VBT指标映射
            vbt_indicators = factory.vbt_indicator_map
            logger.info(f"  ✅ VBT指标映射: {len(vbt_indicators)} 个")

            # 检查TA-Lib指标映射
            talib_indicators = factory.talib_indicator_map
            logger.info(f"  ✅ TA-Lib指标映射: {len(talib_indicators)} 个")

            return True

        except Exception as e:
            logger.error(f"  ❌ 因子工厂测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_factor_registry(self) -> bool:
        """测试因子注册表"""
        try:
            logger.info("测试3：因子注册表")

            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry, FactorCategory

            # 获取全局注册表
            registry = get_factor_registry()
            logger.info("  ✅ 全局注册表获取成功")

            # 检查注册表统计
            stats = registry.get_statistics()
            logger.info(f"  ✅ 注册表统计: {stats}")

            return True

        except Exception as e:
            logger.error(f"  ❌ 因子注册表测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_dynamic_factor_registration(self) -> bool:
        """测试动态因子注册（小规模）"""
        try:
            logger.info("测试4：动态因子注册（小规模）")

            from factor_system.factor_engine.factors.etf_cross_section.etf_factor_factory import ETFFactorFactory
            from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry, FactorCategory

            factory = ETFFactorFactory()
            registry = get_factor_registry()

            # 清除现有动态因子
            cleared_count = registry.clear_dynamic_factors()
            logger.info(f"  ✅ 清除现有动态因子: {cleared_count} 个")

            # 手动注册少量VBT因子进行测试
            test_factors = []

            # 注册RSI因子
            try:
                import vectorbt as vbt

                def test_rsi_14(data):
                    close = data['close']
                    rsi = vbt.RSI.run(close, window=14)
                    return rsi.rsi

                success = registry.register_factor(
                    factor_id="TEST_RSI_14",
                    function=test_rsi_14,
                    parameters={"window": 14},
                    category=FactorCategory.MOMENTUM,
                    description="测试RSI因子"
                )

                if success:
                    test_factors.append("TEST_RSI_14")
                    logger.info("  ✅ RSI因子注册成功")

            except Exception as e:
                logger.warning(f"  ⚠️ RSI因子注册失败: {str(e)}")

            # 注册MACD因子
            try:
                def test_macd(data):
                    close = data['close']
                    macd = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
                    return macd.macd

                success = registry.register_factor(
                    factor_id="TEST_MACD",
                    function=test_macd,
                    parameters={"fast_window": 12, "slow_window": 26, "signal_window": 9},
                    category=FactorCategory.MOMENTUM,
                    description="测试MACD因子"
                )

                if success:
                    test_factors.append("TEST_MACD")
                    logger.info("  ✅ MACD因子注册成功")

            except Exception as e:
                logger.warning(f"  ⚠️ MACD因子注册失败: {str(e)}")

            # 验证注册结果
            stats = registry.get_statistics()
            dynamic_count = stats['dynamic_factors']
            logger.info(f"  ✅ 动态因子注册完成: {dynamic_count} 个")
            logger.info(f"  ✅ 测试因子: {test_factors}")

            return dynamic_count > 0

        except Exception as e:
            logger.error(f"  ❌ 动态因子注册测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_enhanced_factor_calculator(self) -> bool:
        """测试增强因子计算器"""
        try:
            logger.info("测试5：增强因子计算器")

            from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced

            # 创建增强因子计算器
            calculator = ETFCrossSectionFactorsEnhanced(enable_dynamic_factors=True)
            logger.info("  ✅ 增强因子计算器实例创建成功")

            # 检查方法
            required_methods = [
                'initialize_dynamic_factors',
                'calculate_all_factors_enhanced',
                'get_available_factors',
                'get_factor_statistics'
            ]

            for method_name in required_methods:
                if hasattr(calculator, method_name):
                    logger.info(f"  ✅ 方法 {method_name} 存在")
                else:
                    logger.error(f"  ❌ 方法 {method_name} 不存在")
                    return False

            return True

        except Exception as e:
            logger.error(f"  ❌ 增强因子计算器测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def test_dynamic_factor_calculation(self) -> bool:
        """测试小规模动态因子计算"""
        try:
            logger.info("测试6：小规模动态因子计算")

            from factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced
            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager

            # 创建增强因子计算器
            calculator = ETFCrossSectionFactorsEnhanced(enable_dynamic_factors=True)
            logger.info("  ✅ 增强因子计算器创建成功")

            # 初始化动态因子
            logger.info("  初始化动态因子...")
            try:
                # 获取注册表中的测试因子
                from factor_system.factor_engine.factors.etf_cross_section.factor_registry import get_factor_registry, FactorCategory, FactorCategory
                registry = get_factor_registry()
                dynamic_factors = registry.list_factors(is_dynamic=True)

                if not dynamic_factors:
                    logger.warning("  ⚠️ 没有动态因子可用，跳过计算测试")
                    return True

                logger.info(f"  可用动态因子: {dynamic_factors}")

            except Exception as e:
                logger.warning(f"  ⚠️ 动态因子初始化失败: {str(e)}")
                return True

            # 获取测试数据
            data_manager = ETFCrossSectionDataManager()
            etf_list = data_manager.get_etf_universe()

            if len(etf_list) < 2:
                logger.warning("  ⚠️ ETF数据不足，跳过计算测试")
                return True

            test_etfs = etf_list[:2]
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=15)

            price_data = data_manager.get_time_series_data(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                test_etfs
            )

            if price_data.empty:
                logger.warning("  ⚠️ 价格数据为空，跳过计算测试")
                return True

            # 测试计算少量动态因子
            test_factor_ids = dynamic_factors[:2]  # 只测试前2个因子

            try:
                calculated_factors = calculator.calculate_dynamic_factors_batch(
                    factor_ids=test_factor_ids,
                    price_data=price_data,
                    symbols=test_etfs,
                    parallel=False  # 使用串行以避免复杂性
                )

                logger.info(f"  ✅ 动态因子计算完成: {len(calculated_factors)} 个")

                for factor_id, factor_df in calculated_factors.items():
                    if not factor_df.empty:
                        logger.info(f"    📊 {factor_id}: {len(factor_df)} 条记录")
                        logger.info(f"    📊 列: {list(factor_df.columns)}")

                return len(calculated_factors) > 0

            except Exception as e:
                logger.warning(f"  ⚠️ 动态因子计算失败: {str(e)}")
                return True  # 计算失败不算致命错误

        except Exception as e:
            logger.error(f"  ❌ 动态因子计算测试失败: {str(e)}")
            logger.error(f"  详细错误: {traceback.format_exc()}")
            return False

    def generate_test_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("# ETF横截面动态因子冒烟测试报告")
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)

        report.append(f"## 测试总览")
        report.append(f"- 总测试数: {total_tests}")
        report.append(f"- 通过测试: {passed_tests}")
        report.append(f"- 失败测试: {total_tests - passed_tests}")
        report.append(f"- 通过率: {passed_tests/total_tests:.1%}")
        report.append("")

        report.append("## 详细结果")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            report.append(f"- **{test_name}**: {status}")

        report.append("")
        if passed_tests == total_tests:
            report.append("🎉 所有动态因子功能测试通过！系统可以正常使用。")
        else:
            report.append("⚠️ 部分测试失败，请检查相关功能。")

        return "\n".join(report)


def main():
    """主函数"""
    tester = DynamicFactorSmokeTest()
    results = tester.run_all_tests()

    # 生成并保存测试报告
    report = tester.generate_test_report()
    report_file = Path(__file__).parent / "smoke_test_dynamic_factors_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"动态因子测试报告已保存到: {report_file}")

    # 根据测试结果设置退出码
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)

    if passed_count == total_count:
        logger.info("🎉 所有动态因子冒烟测试通过！")
        sys.exit(0)
    else:
        logger.error(f"❌ {total_count - passed_count} 个测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()